from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tonedio.vision_rx import VisionRX


def build_args():
    parser = ArgumentParser(description="Save raw full-size RGB frames from the vision receiver.")
    parser.add_argument("--output_dir", type=str, default="raw_rgb")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument("--print_every", type=int, default=30)
    return parser.parse_args()


def main():
    args = build_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_data = {}
    vision_rx = VisionRX(shared_data)
    print(f"Saving raw RGB frames to: {output_dir.resolve()}", flush=True)

    last_frame_id = None
    seen_count = 0
    saved_count = 0
    save_every = max(1, int(args.save_every))

    try:
        while args.max_frames <= 0 or saved_count < args.max_frames:
            frame = shared_data.get("latest_frame")
            if frame is None:
                time.sleep(0.001)
                continue

            frame_id = int(frame["frame_id"])
            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id
            seen_count += 1

            image_bgr = frame["image_bgr"]
            if (seen_count - 1) % save_every == 0:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                out_path = output_dir / f"rgb_frame_{frame_id:06d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                saved_count += 1
                if saved_count == 1 or saved_count % max(1, int(args.print_every)) == 0:
                    height, width = image_rgb.shape[:2]
                    print(
                        "[raw_rgb]",
                        "saved=", saved_count,
                        "frame=", frame_id,
                        "size=", f"{width}x{height}",
                        "path=", out_path,
                        flush=True,
                    )
    except KeyboardInterrupt:
        pass
    finally:
        thread = vision_rx.get_thread_for_join()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        print("[raw_rgb] exited", "saved=", saved_count, flush=True)


if __name__ == "__main__":
    main()
