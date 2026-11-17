#!/usr/bin/env python3
"""Compare the vignettecorrect element with the numpy reference, on the Jetson.

Decodes the first few frames of a recording twice, once through the element
and once without it, dumps both as raw RGBA, and checks that the element's
output equals reference.apply_gain on the uncorrected frames. Each pipeline is
stopped as soon as enough frames are on disk.

    python3 tools/verify_on_device.py --video run.mp4 \
        --config configs/kowa_lm3ncm_wp__bfs_pge_63s4c.txt --plugin-dir plugin

Uses the hardware decoder and the GPU for a few seconds. Exit code 0 on a match.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vignettecorrect import apply_gain, read_config  # noqa: E402

NVMM_RGBA = "video/x-raw(memory:NVMM),format=RGBA"


def pipeline(video: str, out_dir: Path, element: str | None) -> list[str]:
    parts = ["filesrc", f"location={video}", "!", "decodebin", "!", "nvvideoconvert", "!", NVMM_RGBA]
    if element:
        parts += ["!"] + element.split()
    # sync=true plays at the recording's frame rate, so few extra frames are
    # written before the pipeline is stopped.
    parts += ["!", "nvvideoconvert", "!", "video/x-raw,format=RGBA",
              "!", "multifilesink", "sync=true", f"location={out_dir}/frame_%03d.rgba"]
    return ["gst-launch-1.0", "-q"] + parts


def capture(command: list[str], out_dir: Path, frames: int, env: dict, timeout: float) -> None:
    """Runs the pipeline until `frames` complete frames are on disk, then stops it."""
    log = out_dir / "gst.log"
    start = time.monotonic()
    with open(log, "w") as stderr:
        process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            # One extra file, because the newest one may still be being written.
            while len(list(out_dir.glob("*.rgba"))) <= frames and process.poll() is None:
                if time.monotonic() - start > timeout:
                    sys.exit(f"timed out after {timeout:.0f} s:\n{' '.join(command)}")
                time.sleep(0.02)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    files = sorted(out_dir.glob("*.rgba"))
    if len(files) <= frames:
        sys.exit(f"pipeline ended early with {len(files)} frames:\n{' '.join(command)}\n"
                 f"{log.read_text()}")
    for extra in files[frames:]:
        extra.unlink()


def load(path: Path, height: int, width: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != height * width * 4:
        sys.exit(f"{path.name}: {data.size} bytes, expected {height * width * 4} "
                 f"for {width}x{height} RGBA; check --width and --height")
    return data.reshape(height, width, 4)  # (H, W, 4) RGBA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plugin-dir", required=True, help="folder holding libgstvignettecorrect.so")
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds per pipeline")
    args = parser.parse_args()

    config = read_config(args.config)
    env = dict(os.environ)
    env["GST_PLUGIN_PATH"] = os.path.abspath(args.plugin_dir)
    element = f"vignettecorrect config-file={os.path.abspath(args.config)}"

    with tempfile.TemporaryDirectory(prefix="vignette_verify_") as tmp:
        plain_dir, fixed_dir = Path(tmp) / "plain", Path(tmp) / "corrected"
        plain_dir.mkdir()
        fixed_dir.mkdir()
        capture(pipeline(args.video, plain_dir, None), plain_dir, args.frames, env, args.timeout)
        capture(pipeline(args.video, fixed_dir, element), fixed_dir, args.frames, env, args.timeout)

        names = sorted(p.name for p in plain_dir.glob("*.rgba"))
        if not names or names != sorted(p.name for p in fixed_dir.glob("*.rgba")):
            sys.exit("the two runs produced different frame sets")

        gain = config.gain_map(args.height, args.width)  # (H, W) float32
        worst, differing, total, changed = 0, 0, 0, 0
        for name in names:
            plain = load(plain_dir / name, args.height, args.width)
            fixed = load(fixed_dir / name, args.height, args.width)
            expected = apply_gain(np.ascontiguousarray(plain[:, :, :3]), gain, knee=config.knee,
                                  highlight_mode=config.highlight_mode)
            diff = np.abs(fixed[:, :, :3].astype(np.int16) - expected.astype(np.int16))
            if not np.array_equal(fixed[:, :, 3], plain[:, :, 3]):
                sys.exit(f"{name}: the alpha channel was modified")
            worst = max(worst, int(diff.max()))
            differing += int((diff > 0).sum())
            total += diff.size
            changed += int((fixed[:, :, :3] != plain[:, :, :3]).sum())
            print(f"{name}: max diff {int(diff.max())}, mean diff {diff.mean():.5f}, "
                  f"mean brightness {plain[:, :, :3].mean():.2f} -> {fixed[:, :, :3].mean():.2f}")

    share = differing / total
    print(f"frames: {len(names)}; values differing from the reference: {share:.4%}; "
          f"worst difference: {worst} grey levels")
    print(f"values the element changed: {changed / total:.2%}")
    if changed == 0:
        print("FAIL: the element changed nothing")
        return 1
    if worst > 1 or share > 0.01:
        print("FAIL: output does not match the reference")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
