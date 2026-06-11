#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MASK_PALETTE = {
    0: (0, 0, 0),
    1: (255, 211, 42),
    2: (255, 138, 0),
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "weights" / "segmentation" / "segformer_b0_yellow_line_best_model"
DEFAULT_INPUT_DIR = ROOT / "sample_images"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "segformer_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer yellow-line segmentation inference.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def image_paths(root: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]


def build_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    overlay = rgb.copy()
    alpha = 0.55
    for value, color in MASK_PALETTE.items():
        if value == 0:
            continue
        active = mask == value
        overlay[active] = rgb[active] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def colorize_mask(mask: np.ndarray) -> Image.Image:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in MASK_PALETTE.items():
        colored[mask == value] = color
    return Image.fromarray(colored, mode="RGB")


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir).to(device)
    model.eval()

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    color_mask_dir = args.output_dir / "color_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    color_mask_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(args.input_dir)
    if args.max_images > 0:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    print(f"device={device}")
    print(f"images={len(paths)}")
    for path in tqdm(paths, desc="predict"):
        image = Image.open(path).convert("RGB")
        original_size = image.size
        resized = TF.resize(image, [args.height, args.width], interpolation=InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=tensor)
            logits = F.interpolate(outputs.logits, size=(args.height, args.width), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if original_size != (args.width, args.height):
            pred_image = Image.fromarray(pred, mode="L").resize(original_size, resample=Image.Resampling.NEAREST)
            pred = np.asarray(pred_image, dtype=np.uint8)

        mask_path = mask_dir / f"{path.stem}.png"
        Image.fromarray(pred, mode="L").save(mask_path)
        colorize_mask(pred).save(color_mask_dir / f"{path.stem}.png")
        build_overlay(image, pred).save(overlay_dir / f"{path.stem}.jpg", quality=92)

    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
