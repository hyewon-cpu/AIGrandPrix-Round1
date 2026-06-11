from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import glob
import json
import math

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_args():
    parser = ArgumentParser(description="Detect rectangular gate candidates from black/white segmentation masks.")
    parser.add_argument("inputs", nargs="+", help="Mask image files, folders, or glob patterns.")
    parser.add_argument("--output_dir", type=str, default="mask_geometry_gate_detections")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--min_area", type=float, default=20.0)
    parser.add_argument("--max_candidates", type=int, default=5)
    parser.add_argument("--close_kernel", type=int, default=5)
    parser.add_argument("--dilate_kernel", type=int, default=3)
    parser.add_argument("--edge_samples", type=int, default=40)
    parser.add_argument("--edge_thickness", type=int, default=2)
    parser.add_argument("--min_edge_coverage", type=float, default=0.25)
    parser.add_argument("--min_rectangularity", type=float, default=0.05)
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


def order_corners(corners):
    corners = np.asarray(corners, dtype=np.float32)
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    return np.roll(ordered, -start, axis=0)


def sample_edge_coverage(mask, p0, p1, samples, thickness):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    hits = 0
    total = max(2, int(samples))
    radius = max(0, int(thickness))
    height, width = mask.shape[:2]
    for i in range(total):
        t = float(i) / float(total - 1)
        p = p0 * (1.0 - t) + p1 * t
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        if np.any(mask[y0:y1, x0:x1] > 0):
            hits += 1
    return float(hits) / float(total)


def score_rect(mask, corners, contour_area, edge_samples, edge_thickness):
    corners = order_corners(corners)
    rect_area = float(cv2.contourArea(corners.astype(np.float32)))
    if rect_area <= 1e-6:
        return None

    edge_coverages = []
    for i in range(4):
        edge_coverages.append(
            sample_edge_coverage(
                mask,
                corners[i],
                corners[(i + 1) % 4],
                samples=edge_samples,
                thickness=edge_thickness,
            )
        )

    rectangularity = float(contour_area) / rect_area
    score = float(np.mean(edge_coverages)) * 0.75 + float(min(edge_coverages)) * 0.25
    score *= min(1.0, max(0.0, rectangularity))
    return {
        "corners": corners,
        "rect_area": rect_area,
        "rectangularity": rectangularity,
        "edge_coverages": edge_coverages,
        "score": score,
    }


def detect_gates(mask, args):
    binary = np.asarray(mask, dtype=np.uint8)
    _, binary = cv2.threshold(binary, int(args.threshold), 255, cv2.THRESH_BINARY)

    if args.close_kernel > 0:
        kernel = np.ones((int(args.close_kernel), int(args.close_kernel)), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if args.dilate_kernel > 0:
        kernel = np.ones((int(args.dilate_kernel), int(args.dilate_kernel)), dtype=np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < float(args.min_area):
            continue
        if contour.shape[0] < 4:
            continue

        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
        scored = score_rect(
            binary,
            corners,
            contour_area=contour_area,
            edge_samples=args.edge_samples,
            edge_thickness=args.edge_thickness,
        )
        if scored is None:
            continue
        if min(scored["edge_coverages"]) < float(args.min_edge_coverage):
            continue
        if scored["rectangularity"] < float(args.min_rectangularity):
            continue

        center = np.mean(scored["corners"], axis=0)
        size = rect[1]
        detections.append(
            {
                "center": center,
                "corners": scored["corners"],
                "size": (float(size[0]), float(size[1])),
                "contour_area": contour_area,
                "rect_area": scored["rect_area"],
                "rectangularity": scored["rectangularity"],
                "edge_coverages": scored["edge_coverages"],
                "score": scored["score"],
            }
        )

    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return binary, detections[: max(1, int(args.max_candidates))]


def draw_detections(mask_bgr, detections):
    overlay = mask_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    palette = [(0, 255, 0), (0, 170, 255), (255, 120, 0), (255, 0, 255), (180, 180, 180)]
    for idx, detection in enumerate(detections):
        color = palette[idx % len(palette)]
        corners = np.asarray(detection["corners"], dtype=np.int32)
        cv2.polylines(overlay, [corners], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        center = detection["center"]
        center_xy = (int(round(float(center[0]))), int(round(float(center[1]))))
        cv2.drawMarker(
            overlay,
            center_xy,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=11,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"gate {idx + 1} {float(detection['score']):.2f}",
            (center_xy[0] + 6, center_xy[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def detection_to_json(detection):
    return {
        "center": np.asarray(detection["center"], dtype=np.float32).tolist(),
        "corners": np.asarray(detection["corners"], dtype=np.float32).tolist(),
        "size": list(detection["size"]),
        "contour_area": float(detection["contour_area"]),
        "rect_area": float(detection["rect_area"]),
        "rectangularity": float(detection["rectangularity"]),
        "edge_coverages": [float(value) for value in detection["edge_coverages"]],
        "score": float(detection["score"]),
    }


def main():
    args = build_args()
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(args.inputs))
    if not image_paths:
        raise FileNotFoundError("No input mask images found.")

    results = []
    for image_path in image_paths:
        mask = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("[skip]", image_path, "could not be read", flush=True)
            continue

        binary, detections = detect_gates(mask, args)
        overlay = draw_detections(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), detections)
        overlay_path = overlay_dir / f"{image_path.stem}_geometry_gates.png"
        cv2.imwrite(str(overlay_path), overlay)

        result = {
            "image": str(image_path),
            "overlay": str(overlay_path),
            "gate_count": len(detections),
            "gates": [detection_to_json(detection) for detection in detections],
        }
        results.append(result)
        print("[mask_gate_detect]", image_path, "gates=", len(detections), "overlay=", overlay_path, flush=True)

    results_path = output_dir / "mask_geometry_gate_detections.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[mask_gate_detect]", "results=", results_path, flush=True)


if __name__ == "__main__":
    main()
