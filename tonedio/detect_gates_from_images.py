from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import glob
import json
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tonedio.utils import GateDetector


DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "gate_detection_112112.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
GATE_EDGE_TYPES = (("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL"))


def build_args():
    parser = ArgumentParser(description="Run the gate detection model on image or segmentation-mask inputs.")
    parser.add_argument("inputs", nargs="+", help="Image files, folders, or glob patterns.")
    parser.add_argument("--output_dir", type=str, default="gate_detections")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_gates", type=int, default=5)
    parser.add_argument("--corner_conf_threshold", type=float, default=0.25)
    parser.add_argument("--corner_topk", type=int, default=50)
    parser.add_argument("--corner_nms_radius", type=int, default=5)
    parser.add_argument("--edge_min_score", type=float, default=0.05)
    parser.add_argument("--integral_samples", type=int, default=15)
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


def pixel_xy(point):
    if point is None:
        return None
    try:
        return int(round(float(point[0]))), int(round(float(point[1])))
    except (TypeError, ValueError, IndexError):
        return None


def draw_gate_candidate(image_bgr, candidate, color, label=None, thickness=2):
    points = candidate.get("points") or {}
    for a, b in GATE_EDGE_TYPES:
        p0 = pixel_xy(points.get(a))
        p1 = pixel_xy(points.get(b))
        if p0 is not None and p1 is not None:
            cv2.line(image_bgr, p0, p1, color, thickness, lineType=cv2.LINE_AA)

    for name, point in points.items():
        xy = pixel_xy(point)
        if xy is None:
            continue
        cv2.circle(image_bgr, xy, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            str(name),
            (xy[0] + 4, xy[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    center = pixel_xy(candidate.get("center"))
    if center is not None:
        cv2.drawMarker(
            image_bgr,
            center,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=11,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        if label:
            cv2.putText(
                image_bgr,
                label,
                (center[0] + 6, center[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )


def candidate_to_json(candidate):
    points = candidate.get("points") or {}
    return {
        "center": np.asarray(candidate.get("center", []), dtype=np.float32).tolist(),
        "size": np.asarray(candidate.get("size", []), dtype=np.float32).tolist(),
        "confidence": float(candidate.get("confidence", 0.0)),
        "gate_score": float(candidate.get("gate_score", 0.0)),
        "points": {
            key: np.asarray(value, dtype=np.float32).tolist()
            for key, value in points.items()
        },
        "scores": {key: float(value) for key, value in (candidate.get("scores") or {}).items()},
        "edge_scores": {
            key: float(value)
            for key, value in (candidate.get("edge_scores") or {}).items()
        },
    }


def main():
    args = build_args()
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    detector = GateDetector(
        checkpoint_path=args.model_path,
        device=args.device,
        corner_conf_threshold=args.corner_conf_threshold,
        corner_topk=args.corner_topk,
        corner_nms_radius=args.corner_nms_radius,
        edge_min_score=args.edge_min_score,
        integral_samples=args.integral_samples,
        load_airsim_camera_settings=False,
    )

    image_paths = list(iter_image_paths(args.inputs))
    if not image_paths:
        raise FileNotFoundError("No input images found.")

    results = []
    palette = [(0, 255, 0), (0, 170, 255), (255, 120, 0), (255, 0, 255), (180, 180, 180)]
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("[skip]", image_path, "could not be read", flush=True)
            continue

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        corner_maps, paf_maps = detector.predict_maps(rgb)
        height, width = rgb.shape[:2]
        gates = detector._extract_gate_candidates(
            corner_maps,
            paf_maps,
            image_width=width,
            image_height=height,
            max_gates=args.max_gates,
        )

        overlay = image_bgr.copy()
        for idx, gate in enumerate(gates):
            draw_gate_candidate(
                overlay,
                gate,
                palette[idx % len(palette)],
                label=f"gate {idx + 1} {float(gate.get('gate_score', 0.0)):.2f}",
                thickness=2 if idx == 0 else 1,
            )

        overlay_path = overlay_dir / f"{image_path.stem}_gates.png"
        cv2.imwrite(str(overlay_path), overlay)
        result = {
            "image": str(image_path),
            "overlay": str(overlay_path),
            "gate_count": len(gates),
            "gates": [candidate_to_json(gate) for gate in gates],
        }
        results.append(result)
        print("[gate_detect]", image_path, "gates=", len(gates), "overlay=", overlay_path, flush=True)

    results_path = output_dir / "gate_detections.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[gate_detect]", "results=", results_path, flush=True)


if __name__ == "__main__":
    main()
