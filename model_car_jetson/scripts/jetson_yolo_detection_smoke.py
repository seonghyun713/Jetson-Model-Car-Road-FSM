#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "detection" / "yolov8s_model_car_best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "detection_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 object detection smoke test for Jetson.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", default="0", help="Camera index or OpenCV/GStreamer pipeline.")
    source.add_argument("--video", type=Path, default=None)
    source.add_argument("--image", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or 0 for Ultralytics.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--frame-fps", type=float, default=30.0)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


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


def prediction_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.device != "auto":
        kwargs["device"] = args.device
    return kwargs


def print_boxes(model: YOLO, result, frame_index: int, fps: float) -> None:
    boxes = result.boxes
    parts = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = model.names.get(cls_id, str(cls_id))
        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        parts.append(f"{name}:{conf:.2f}@{xyxy}")
    joined = " | ".join(parts) if parts else "none"
    print(f"frame={frame_index} fps={fps:.1f} detections={joined}")


def run_image(args: argparse.Namespace, model: YOLO) -> None:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    result = model.predict(image, **prediction_kwargs(args))[0]
    print_boxes(model, result, 0, 0.0)
    annotated = result.plot()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.image.stem}_detections.jpg"
    cv2.imwrite(str(out_path), annotated)
    if args.display:
        cv2.imshow("model-car YOLO detection smoke", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"image_output={out_path}")


def run_stream(args: argparse.Namespace, model: YOLO) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = open_capture(args)
    writer: Optional[cv2.VideoWriter] = None
    video_path = args.output_dir / "detection_debug.mp4"
    frame_index = 0
    prev_time = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break
        result = model.predict(frame, **prediction_kwargs(args))[0]
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        if args.print_every > 0 and frame_index % args.print_every == 0:
            print_boxes(model, result, frame_index, fps)
        if args.display or args.save_video:
            annotated = result.plot()
            if writer is None and args.save_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, max(1.0, args.frame_fps), (annotated.shape[1], annotated.shape[0]))
            if writer is not None:
                writer.write(annotated)
            if args.display:
                cv2.imshow("model-car YOLO detection smoke", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        frame_index += 1
    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"frames={frame_index}")
    if args.save_video:
        print(f"debug_video={video_path}")


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    if args.image is not None:
        run_image(args, model)
    else:
        run_stream(args, model)


if __name__ == "__main__":
    main()
