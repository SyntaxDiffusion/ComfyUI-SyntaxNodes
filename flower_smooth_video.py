"""
FloweR Video Smoother
=====================
Standalone script: takes an MP4, runs FloweR frame-by-frame to smooth it
using optical flow prediction + blending, outputs a new MP4.

Usage:
  python flower_smooth_video.py input.mp4 output.mp4
  python flower_smooth_video.py input.mp4 output.mp4 --blend 0.5 --device cuda

Requirements: torch, numpy, opencv-python, imageio, imageio-ffmpeg
"""

import argparse
import sys
import os
import numpy as np
import torch
import cv2
import imageio
from collections import deque
from pathlib import Path

# Import FloweR from the same package
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from flower_shared import (
    FloweR, flower_inference, flow_warp_frame,
    blend_flower_prediction, compute_flower_occlusion,
)


def load_flower_model(weights_path, device, h, w):
    """Load FloweR model from .pth weights."""
    flower_h = (h // 128) * 128
    flower_w = (w // 128) * 128
    if flower_h == 0 or flower_w == 0:
        raise ValueError(f"Frame dimensions ({w}x{h}) too small for FloweR (need >=128)")

    net = FloweR(input_size=(flower_h, flower_w))
    state = torch.load(weights_path, map_location="cpu")
    # Handle both raw state_dict and wrapped format
    if "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.to(device).eval()
    print(f"FloweR loaded: {weights_path}")
    print(f"  Input size: {flower_w}x{flower_h} (from {w}x{h})")
    return net, flower_h, flower_w


def smooth_video(input_path, output_path, weights_path, blend=0.5,
                 device_str="cuda", pass_count=1):
    """Run FloweR smoothing on a video.

    For each frame (after the first 4), FloweR predicts the next frame from
    the previous 4. The output is a blend of the original frame and FloweR's
    prediction, controlled by `blend`:
      - blend=0.0: original video unchanged
      - blend=0.3: subtle smoothing (recommended start)
      - blend=0.5: moderate smoothing
      - blend=0.8: heavy smoothing, FloweR-dominant
      - blend=1.0: pure FloweR prediction (experimental)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Read input video
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)
    total_frames = reader.count_frames()
    print(f"Input: {input_path}")
    print(f"  {total_frames} frames @ {fps} fps")

    # Read first frame to get dimensions
    first_frame = reader.get_data(0)
    h, w = first_frame.shape[:2]
    print(f"  {w}x{h}")

    # Load model
    net, flower_h, flower_w = load_flower_model(weights_path, device, h, w)

    for pass_num in range(pass_count):
        if pass_count > 1:
            print(f"\n--- Pass {pass_num + 1}/{pass_count} ---")

        # On subsequent passes, read from the previous output
        if pass_num > 0:
            reader.close()
            reader = imageio.get_reader(output_path)
            total_frames = reader.count_frames()

        # Setup writer
        tmp_output = output_path if pass_num == pass_count - 1 else output_path + ".tmp.mp4"
        writer = imageio.get_writer(
            tmp_output, fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p"
        )

        # Frame history buffer (FloweR needs 4 frames)
        history = deque(maxlen=4)
        smoothed_count = 0

        for i in range(total_frames):
            frame = reader.get_data(i)

            # Convert RGB to BGR for OpenCV if needed, then back
            # imageio reads as RGB, FloweR expects RGB uint8
            frame_rgb = frame[:, :, :3]  # drop alpha if present

            history.append(frame_rgb)

            if len(history) < 4:
                # Not enough history yet — pass through unchanged
                writer.append_data(frame_rgb)
                if i % 50 == 0:
                    print(f"  [{i}/{total_frames}] buffering...")
                continue

            # Run FloweR inference
            pred_flow, pred_occl, pred_next, fh, fw = flower_inference(
                net, list(history), device, h, w
            )

            # FloweR's predicted next frame
            pred_next_np = pred_next.numpy()
            pred_next_np = np.clip(pred_next_np, 0, 255).astype(np.uint8)
            if fh != h or fw != w:
                pred_next_np = cv2.resize(pred_next_np, (w, h),
                                          interpolation=cv2.INTER_LINEAR)

            # Flow-warp the current frame for per-pixel refinement
            flow_warped = flow_warp_frame(
                frame_rgb, pred_flow, fh, fw, h, w, device
            )

            # Blend: original frame ←→ FloweR prediction
            if blend > 0:
                # Blend flow-warped with FloweR prediction first
                flower_result = blend_flower_prediction(
                    flow_warped, pred_next_np, blend
                )
                # Then blend that with the original frame for control
                output_frame = cv2.addWeighted(
                    frame_rgb, 1.0 - blend,
                    flower_result, blend, 0
                )
            else:
                output_frame = frame_rgb

            writer.append_data(output_frame)
            smoothed_count += 1

            if i % 50 == 0:
                print(f"  [{i}/{total_frames}] smoothed")

        writer.close()
        print(f"  Smoothed {smoothed_count} frames")

        # Clean up temp file from previous pass
        if pass_num > 0 and pass_num < pass_count - 1:
            tmp_prev = output_path + ".tmp.mp4"
            if os.path.exists(tmp_prev):
                os.remove(tmp_prev)

    reader.close()

    # Cleanup model
    net.to("cpu")
    del net
    torch.cuda.empty_cache()

    print(f"\nDone: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Smooth a video using FloweR optical flow prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Subtle smoothing
  python flower_smooth_video.py input.mp4 smooth.mp4 --blend 0.3

  # Moderate smoothing (default)
  python flower_smooth_video.py input.mp4 smooth.mp4

  # Heavy smoothing, 2 passes
  python flower_smooth_video.py input.mp4 smooth.mp4 --blend 0.6 --passes 2

  # CPU only
  python flower_smooth_video.py input.mp4 smooth.mp4 --device cpu
        """)

    parser.add_argument("input", help="Input MP4 path")
    parser.add_argument("output", help="Output MP4 path")
    parser.add_argument("--weights", default=None,
                        help="Path to FloweR .pth weights (auto-detected if not set)")
    parser.add_argument("--blend", type=float, default=0.5,
                        help="Smoothing blend (0=none, 0.3=subtle, 0.5=moderate, 0.8=heavy)")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of smoothing passes (more = smoother)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")

    args = parser.parse_args()

    # Auto-detect weights
    weights = args.weights
    if weights is None:
        # Check common locations
        candidates = [
            script_dir / "../../models/FloweR/FloweR_0.1.2.pth",
            Path("G:/comfyui build/ComfyUI/models/FloweR/FloweR_0.1.2.pth"),
        ]
        for c in candidates:
            if c.exists():
                weights = str(c.resolve())
                break
        if weights is None:
            print("ERROR: Could not find FloweR weights. Use --weights path/to/FloweR.pth")
            sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    smooth_video(args.input, args.output, weights,
                 blend=args.blend, device_str=args.device,
                 pass_count=args.passes)


if __name__ == "__main__":
    main()
