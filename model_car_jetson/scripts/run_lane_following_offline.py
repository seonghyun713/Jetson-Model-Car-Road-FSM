#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
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
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "offline_lane_following"
DEFAULT_INPUT_DIR = ROOT / "sample_images"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_points(text: str) -> Tuple[Tuple[float, float], ...]:
    points = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_text, y_text = chunk.split(",")
        points.append((float(x_text), float(y_text)))
    if len(points) != 4:
        raise argparse.ArgumentTypeError("Expected 4 semicolon-separated x,y ratio points.")
    return tuple(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline dashed-centerline following smoke test from SegFormer yellow-line masks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional precomputed mask directory. If omitted, SegFormer is used.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--lane-side", choices=["left", "right"], default="right", help="Kept for compatibility; dashed-centerline mode uses the dashed line as the target path.")
    parser.add_argument("--src-points-ratio", type=parse_points, default=None, help="BL,BR,TR,TL as 'x,y;x,y;x,y;x,y'.")
    parser.add_argument("--bev-width", type=int, default=640)
    parser.add_argument("--bev-height", type=int, default=720)
    parser.add_argument("--model-width", type=int, default=768)
    parser.add_argument("--model-height", type=int, default=432)
    parser.add_argument("--nominal-lane-width-px", type=float, default=220.0, help="Expected BEV distance from dashed centerline to either solid outer line.")
    parser.add_argument("--base-speed", type=float, default=0.28)
    parser.add_argument("--save-intermediate", action="store_true")
    return parser.parse_args()


def image_paths(root: Path) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(model_dir: Path, device: torch.device) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
    model.eval()
    return model


def predict_mask(
    model: SegformerForSemanticSegmentation,
    image_bgr: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb, mode="RGB")
    resized = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        logits = F.interpolate(outputs.logits, size=(height, width), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    if image_bgr.shape[1] != width or image_bgr.shape[0] != height:
        pred = cv2.resize(pred, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


def load_mask(mask_dir: Path, image_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    mask_path = mask_dir / f"{image_path.stem}.png"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    height, width = image_shape
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    confidences = np.array([float(row["confidence"]) for row in rows], dtype=np.float64)
    steers = np.array([float(row["steer"]) for row in rows], dtype=np.float64)
    speeds = np.array([float(row["speed"]) for row in rows], dtype=np.float64)
    valid = np.array([bool(row["valid"]) for row in rows], dtype=bool)
    state_counts: Dict[str, int] = {}
    for row in rows:
        state = str(row["state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    low_rows = sorted(rows, key=lambda row: float(row["confidence"]))[:10]
    return {
        "frames": len(rows),
        "valid_frames": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "mean_confidence": float(confidences.mean()),
        "min_confidence": float(confidences.min()),
        "mean_steer": float(steers.mean()),
        "std_steer": float(steers.std()),
        "mean_abs_steer": float(np.abs(steers).mean()),
        "mean_speed": float(speeds.mean()),
        "state_counts": state_counts,
        "low_confidence_frames": [
            {
                "image": row["image"],
                "state": row["state"],
                "confidence": row["confidence"],
                "steer": row["steer"],
                "speed": row["speed"],
            }
            for row in low_rows
        ],
    }


def build_index(output_dir: Path, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    cards = []
    for row in rows:
        rel = html.escape(str(row["debug_image"]))
        cards.append(
            f"""
            <article class="card">
              <img src="{rel}" alt="{html.escape(str(row['image']))}">
              <div class="meta">
                <strong>{html.escape(str(row['image']))}</strong>
                <span>{html.escape(str(row['state']))} · conf {float(row['confidence']):.2f} · steer {float(row['steer']):+.2f} · speed {float(row['speed']):.2f}</span>
              </div>
            </article>
            """
        )
    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashed Centerline Following Offline Test</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101217;
      color: #eef1f6;
    }}
    body {{ margin: 0; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(16, 18, 23, 0.96);
      border-bottom: 1px solid #2c3240;
      padding: 18px 24px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 24px; letter-spacing: 0; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .stat {{
      padding: 8px 11px;
      border: 1px solid #333b4c;
      border-radius: 6px;
      background: #171b23;
      font-size: 14px;
    }}
    main {{ padding: 22px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .card {{
      border: 1px solid #2c3240;
      border-radius: 8px;
      overflow: hidden;
      background: #171b23;
    }}
    .card img {{ width: 100%; display: block; }}
    .meta {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 12px;
      border-top: 1px solid #2c3240;
      font-size: 13px;
      color: #cbd2de;
    }}
    .meta strong {{ color: #f4f6fb; font-weight: 650; }}
    @media (max-width: 720px) {{
      main {{ padding: 12px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .meta {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dashed Centerline Following Offline Test</h1>
    <div class="stats">
      <div class="stat">frames {summary.get('frames', 0)}</div>
      <div class="stat">valid rate {float(summary.get('valid_rate', 0.0)):.3f}</div>
      <div class="stat">mean confidence {float(summary.get('mean_confidence', 0.0)):.3f}</div>
      <div class="stat">std steer {float(summary.get('std_steer', 0.0)):.3f}</div>
      <div class="stat">mean abs steer {float(summary.get('mean_abs_steer', 0.0)):.3f}</div>
      <div class="stat">mean speed {float(summary.get('mean_speed', 0.0)):.3f}</div>
    </div>
  </header>
  <main>
    <div class="grid">
      {"".join(cards)}
    </div>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir = args.output_dir / "masks"
    bev_mask_out_dir = args.output_dir / "bev_masks"
    if args.save_intermediate:
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        bev_mask_out_dir.mkdir(parents=True, exist_ok=True)

    bev = BevConfig(output_width=args.bev_width, output_height=args.bev_height)
    if args.src_points_ratio is not None:
        bev.src_points_ratio = args.src_points_ratio
    follower_config = LaneFollowerConfig(
        lane_side=args.lane_side,
        nominal_lane_width_px=args.nominal_lane_width_px,
        base_speed=args.base_speed,
    )
    state = LaneFollowerState()

    device = get_device(args.device)
    model = None
    if args.mask_dir is None:
        model = load_model(args.model_dir, device)

    paths = image_paths(args.input_dir)
    if args.max_images > 0:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    rows: List[Dict[str, object]] = []
    print(f"device={device}")
    print(f"images={len(paths)}")
    print(f"path_mode={follower_config.path_mode}")

    matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for index, image_path in enumerate(tqdm(paths, desc="lane-follow")):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        h, w = image_bgr.shape[:2]
        if args.mask_dir is not None:
            mask = load_mask(args.mask_dir, image_path, (h, w))
        else:
            assert model is not None
            mask = predict_mask(model, image_bgr, args.model_width, args.model_height, device)

        cache_key = (w, h)
        if cache_key not in matrix_cache:
            matrix, _ = perspective_matrices(w, h, bev)
            matrix_cache[cache_key] = matrix
        bev_mask = warp_to_bev(mask, bev, matrix_cache[cache_key], is_mask=True)
        estimate = estimate_lane(bev_mask, follower_config, state)

        debug_rel = Path("debug_frames") / f"{index:04d}_{image_path.stem}.jpg"
        debug_panel = make_debug_panel(image_bgr, mask, bev_mask, estimate, bev)
        cv2.imwrite(str(args.output_dir / debug_rel), debug_panel)
        if args.save_intermediate:
            cv2.imwrite(str(mask_out_dir / f"{image_path.stem}.png"), mask)
            cv2.imwrite(str(bev_mask_out_dir / f"{image_path.stem}.png"), bev_mask)

        row = {
            "frame_index": index,
            "image": image_path.name,
            "debug_image": debug_rel.as_posix(),
            **estimate_to_dict(estimate),
        }
        rows.append(row)

    summary = summarize(rows)
    write_csv(args.output_dir / "lane_following_frames.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "bev": {
                    "output_width": bev.output_width,
                    "output_height": bev.output_height,
                    "src_points_ratio": bev.src_points_ratio,
                },
                "lane_follower": follower_config.__dict__,
                "input_dir": str(args.input_dir),
                "mask_dir": str(args.mask_dir) if args.mask_dir else None,
                "model_dir": str(args.model_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_index(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"output={args.output_dir}")
    print(f"html={args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
