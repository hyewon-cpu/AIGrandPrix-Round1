from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import glob

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_args():
    parser = ArgumentParser(description="Create red-color segmentation masks from RGB images.")
    parser.add_argument("inputs", nargs="+", help="Image files, folders, or glob patterns.")
    parser.add_argument("--output_dir", type=str, default="red_segmentation")
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--hue_low_max", type=int, default=10, help="Upper bound for low red HSV hue range.")
    parser.add_argument("--hue_high_min", type=int, default=160, help="Lower bound for high red HSV hue range.")
    parser.add_argument("--sat_min", type=int, default=30, help="Minimum saturation for red pixels.")
    parser.add_argument("--val_min", type=int, default=30, help="Minimum value/brightness for red pixels.")
    parser.add_argument("--open_kernel", type=int, default=2, help="Morphological open kernel size. 0 disables.")
    parser.add_argument("--close_kernel", type=int, default=0, help="Morphological close kernel size. 0 disables.")
    return parser.parse_args()


def iter_image_paths(inputs):
    seen = set()
    for item in inputs:
        matches = [Path(path) for path in glob.glob(item)]
        if not matches:
            matches = [Path(item)]

        for path in matches:
            if path.is_dir():
                candidates = [
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
                ]
            else:
                candidates = [path]

            for candidate in candidates:
                candidate = candidate.resolve()
                if candidate in seen or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                seen.add(candidate)
                yield candidate


def clean_mask(mask, open_kernel, close_kernel):
    if open_kernel > 0:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_red(image_bgr, args):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_lower = np.array([0, args.sat_min, args.val_min], dtype=np.uint8)
    low_upper = np.array([args.hue_low_max, 255, 255], dtype=np.uint8)
    high_lower = np.array([args.hue_high_min, args.sat_min, args.val_min], dtype=np.uint8)
    high_upper = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, low_lower, low_upper),
        cv2.inRange(hsv, high_lower, high_upper),
    )
    return clean_mask(mask, args.open_kernel, args.close_kernel)


def make_overlay(image_bgr, mask):
    overlay = image_bgr.copy()
    red_fill = np.zeros_like(image_bgr)
    red_fill[:, :] = (0, 0, 255)
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(image_bgr, 0.4, red_fill, 0.6, 0), overlay)
    return overlay


def main():
    args = build_args()
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(args.inputs))
    if not image_paths:
        raise FileNotFoundError("No input images found.")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("[skip]", image_path, "could not be read", flush=True)
            continue

        mask = segment_red(image_bgr, args)
        mask_path = mask_dir / f"{image_path.stem}_red_mask.png"
        cv2.imwrite(str(mask_path), mask)

        if args.save_overlay:
            overlay = make_overlay(image_bgr, mask)
            overlay_path = overlay_dir / f"{image_path.stem}_red_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        pixel_count = int(np.count_nonzero(mask))
        print("[red_seg]", image_path, "red_pixels=", pixel_count, "mask=", mask_path, flush=True)


if __name__ == "__main__":
    main()
