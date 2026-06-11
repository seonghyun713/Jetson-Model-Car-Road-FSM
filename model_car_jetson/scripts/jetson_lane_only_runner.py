#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from lane_following_core import (
    BevConfig,
    LaneFollowerConfig,
    LaneFollowerState,
    estimate_lane,
    estimate_to_dict,
    make_debug_panel,
    perspective_matrices,
    warp_to_bev,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "weights" / "segmentation" / "segformer_b0_yellow_line_best_model"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "lane_only_run"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


CSV_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "fps",
    "valid",
    "confidence",
    "state",
    "center_x",
    "lookahead_y",
    "lateral_error_px",
    "heading_error_rad",
    "raw_steer",
    "steer",
    "speed",
    "lane_width_px",
    "reason",
    "row_count",
    "dashed_row_count",
]


def parse_points(text: str) -> Tuple[Tuple[float, float], ...]:
    points = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_text, y_text = chunk.split(",")
        points.append((float(x_text), float(y_text)))
    if len(points) != 4:
        raise argparse.ArgumentTypeError("Expected 4 points: 'x,y;x,y;x,y;x,y'")
    return tuple(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time lane-only runner for Jetson Orin Nano.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", default="0", help="Camera index or OpenCV/GStreamer pipeline.")
    source.add_argument("--video", type=Path, default=None, help="Video file for offline real-time smoke test.")
    parser.add_argument("--segformer-model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--model-width", type=int, default=768)
    parser.add_argument("--model-height", type=int, default=432)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--frame-fps", type=float, default=30.0)
    parser.add_argument("--src-points-ratio", type=parse_points, default=None, help="BL,BR,TR,TL ratios: 'x,y;x,y;x,y;x,y'")
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--bev-height", type=int, default=720)
    parser.add_argument("--nominal-lane-width-px", type=float, default=220.0)
    parser.add_argument("--base-speed", type=float, default=0.28)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
    else:
        camera = str(args.camera)
        is_pipeline = "!" in camera or "nvargus" in camera
        source = int(camera) if camera.isdigit() else camera
        backend = cv2.CAP_GSTREAMER if is_pipeline else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if not is_pipeline:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
            cap.set(cv2.CAP_PROP_FPS, args.frame_fps)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera/video source.")
    return cap


class SegFormerYellowLinePredictor:
    def __init__(self, model_dir: Path, device: torch.device, width: int, height: int, half: bool) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.use_half = bool(half and device.type == "cuda")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
        self.model.eval()
        if self.use_half:
            self.model.half()
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)
        if self.use_half:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        original_h, original_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        if self.use_half:
            tensor = tensor.half()
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            logits = F.interpolate(outputs.logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return pred


def write_run_config(path: Path, args: argparse.Namespace, device: torch.device, bev: BevConfig, follower: LaneFollowerConfig) -> None:
    payload = {
        "device": str(device),
        "segformer_model": str(args.segformer_model),
        "model_width": args.model_width,
        "model_height": args.model_height,
        "bev": {
            "output_width": bev.output_width,
            "output_height": bev.output_height,
            "src_points_ratio": bev.src_points_ratio,
        },
        "lane_follower": follower.__dict__,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def row_from_estimate(frame_index: int, started_at: float, fps: float, estimate_dict: Dict[str, object]) -> Dict[str, object]:
    row = {
        "frame_index": frame_index,
        "timestamp_sec": time.perf_counter() - started_at,
        "fps": fps,
    }
    row.update(estimate_dict)
    return {field: row.get(field) for field in CSV_FIELDS}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    bev = BevConfig(output_width=args.bev_width, output_height=args.bev_height)
    if args.src_points_ratio is not None:
        bev.src_points_ratio = args.src_points_ratio
    follower_config = LaneFollowerConfig(
        nominal_lane_width_px=args.nominal_lane_width_px,
        base_speed=args.base_speed,
    )
    follower_state = LaneFollowerState()
    predictor = SegFormerYellowLinePredictor(args.segformer_model, device, args.model_width, args.model_height, args.half)

    cap = open_capture(args)
    log_path = args.output_dir / "lane_commands.csv"
    config_path = args.output_dir / "run_config.json"
    video_path = args.output_dir / "lane_debug.mp4"
    write_run_config(config_path, args, device, bev, follower_config)

    writer: Optional[cv2.VideoWriter] = None
    matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}
    started_at = time.perf_counter()
    prev_time = started_at
    frame_index = 0

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            h, w = frame.shape[:2]
            key = (w, h)
            if key not in matrix_cache:
                matrix, _ = perspective_matrices(w, h, bev)
                matrix_cache[key] = matrix

            mask = predictor.predict(frame)
            bev_mask = warp_to_bev(mask, bev, matrix_cache[key], is_mask=True)
            estimate = estimate_lane(bev_mask, follower_config, follower_state)

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            estimate_dict = estimate_to_dict(estimate)
            csv_writer.writerow(row_from_estimate(frame_index, started_at, fps, estimate_dict))

            if args.display or args.save_video:
                panel = make_debug_panel(frame, mask, bev_mask, estimate, bev)
                if writer is None and args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, max(1.0, args.frame_fps), (panel.shape[1], panel.shape[0]))
                if writer is not None:
                    writer.write(panel)
                if args.display:
                    cv2.imshow("model-car lane-only runner", panel)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

            if args.print_every > 0 and frame_index % args.print_every == 0:
                print(
                    f"frame={frame_index} fps={fps:.1f} state={estimate.state} "
                    f"conf={estimate.confidence:.2f} steer={estimate.steer:+.3f} speed={estimate.speed:.3f}"
                )
            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"frames={frame_index}")
    print(f"csv={log_path}")
    if args.save_video:
        print(f"debug_video={video_path}")


if __name__ == "__main__":
    main()
