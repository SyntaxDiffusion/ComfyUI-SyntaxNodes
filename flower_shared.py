"""
Shared FloweR optical-flow module for SyntaxNodes.

Provides the full 6-channel FloweR U-Net (flow + occlusion + next-frame
prediction) and helper functions for flow warping, prediction blending,
and occlusion extraction.  Extracted from duplicated code in
sdcn_feedback_animation.py and sdcn_feedback_animation_audio.py.

Original model: https://github.com/volotat/SD-CN-Animation (MIT)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2


# ============================================================================
# FloweR U-Net  (8 encoder + 7 decoder with skip connections + 1 output conv)
# ============================================================================

class FloweR(nn.Module):
    """Lightweight optical-flow + occlusion predictor.

    Takes 4 consecutive frames [B, 4, H, W, 3] normalised to [-1, 1] and
    returns a packed tensor [B, H, W, 6] containing:
      - channels 0-1: optical flow  (divided by 255)
      - channel  2  : occlusion     (rescaled to [-1, 1])
      - channels 3-5: composited next-frame prediction ([-1, 1])
    """

    def __init__(self, input_size=(384, 384), window_size=4):
        super(FloweR, self).__init__()

        self.input_size = input_size
        self.window_size = window_size

        # 2 channels for optical flow
        # 1 channel for occlusion mask
        # 3 channels for next frame prediction
        self.out_channels = 6

        # --- ENCODER (downscale) ---
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3 * self.window_size, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H x W x 128

        self.conv_block_2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/2 x W/2 x 128

        self.conv_block_3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/4 x W/4 x 128

        self.conv_block_4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/8 x W/8 x 128

        self.conv_block_5 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/16 x W/16 x 128

        self.conv_block_6 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/32 x W/32 x 128

        self.conv_block_7 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/64 x W/64 x 128

        self.conv_block_8 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )  # H/128 x W/128 x 128

        # --- DECODER (upscale) ---
        self.conv_block_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_10 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_11 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_12 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_13 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_14 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        self.conv_block_15 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )

        # --- OUTPUT ---
        self.conv_block_16 = nn.Conv2d(128, self.out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, input_frames):
        """Full 6-channel forward pass.

        Parameters
        ----------
        input_frames : Tensor [B, 4, H, W, 3]
            Four consecutive frames normalised to [-1, 1].

        Returns
        -------
        Tensor [B, H, W, 6]
            Packed result: flow/255, occlusion*2-1, composited next frame.
        """
        if input_frames.size(1) != self.window_size:
            raise RuntimeError(
                f"FloweR expects exactly {self.window_size} input frames, "
                f"got {input_frames.size(1)}."
            )

        h, w = self.input_size

        # [B, frames, H, W, 3] -> [B, frames, 3, H, W]
        input_frames_permuted = input_frames.permute((0, 1, 4, 2, 3))
        # -> [B, frames*3, H, W]
        in_x = input_frames_permuted.reshape(-1, self.window_size * 3, h, w)

        # --- ENCODER ---
        block_1_out = self.conv_block_1(in_x)
        block_2_out = self.conv_block_2(block_1_out)
        block_3_out = self.conv_block_3(block_2_out)
        block_4_out = self.conv_block_4(block_3_out)
        block_5_out = self.conv_block_5(block_4_out)
        block_6_out = self.conv_block_6(block_5_out)
        block_7_out = self.conv_block_7(block_6_out)
        block_8_out = self.conv_block_8(block_7_out)

        # --- DECODER with skip connections ---
        block_9_out = block_7_out + self.conv_block_9(block_8_out)
        block_10_out = block_6_out + self.conv_block_10(block_9_out)
        block_11_out = block_5_out + self.conv_block_11(block_10_out)
        block_12_out = block_4_out + self.conv_block_12(block_11_out)
        block_13_out = block_3_out + self.conv_block_13(block_12_out)
        block_14_out = block_2_out + self.conv_block_14(block_13_out)
        block_15_out = block_1_out + self.conv_block_15(block_14_out)

        block_16_out = self.conv_block_16(block_15_out)
        out = block_16_out.reshape(-1, self.out_channels, h, w)

        device = out.device

        # Extract channels
        pred_flow = out[:, :2, :, :] * 255          # pixel displacements (-255, 255)
        pred_occl = (out[:, 2:3, :, :] + 1) / 2     # [0, 1]
        pred_next = out[:, 3:6, :, :]                # [-1, 1] range raw

        # Build sampling grid for flow warping
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h), torch.arange(0, w), indexing='ij'
        )
        flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
        flow_grid = flow_grid.unsqueeze(0).to(device)
        flow_grid = flow_grid + pred_flow

        # Normalise grid to [-1, 1] for grid_sample
        flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
        flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
        flow_grid = flow_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Warp previous (4th) frame using predicted flow
        previous_frame = input_frames_permuted[:, -1, :, :, :]  # [B, 3, H, W]
        sampling_mode = "bilinear" if self.training else "nearest"
        warped_frame = torch.nn.functional.grid_sample(
            previous_frame, flow_grid, mode=sampling_mode,
            padding_mode="reflection", align_corners=False
        )

        # Composite: alpha-blend predicted next frame with warped frame
        alpha_mask = torch.clip(pred_occl * 10, 0, 1) * 0.04
        pred_next = torch.clip(pred_next, -1, 1)
        warped_frame = torch.clip(warped_frame, -1, 1)
        next_frame = pred_next * alpha_mask + warped_frame * (1 - alpha_mask)

        # Pack: [flow/255, occlusion*2-1, composited_next]
        res = torch.cat((pred_flow / 255, pred_occl * 2 - 1, next_frame), dim=1)

        # [B, 6, H, W] -> [B, H, W, 6]
        res = res.permute((0, 2, 3, 1))
        return res


# ============================================================================
# Inference orchestrator
# ============================================================================

def flower_inference(flower_net, frame_history, device, orig_h, orig_w):
    """Run FloweR on the last 4 frames and return denormalised predictions.

    Parameters
    ----------
    flower_net : FloweR
        Model instance (already on *device* and in eval mode).
    frame_history : list[np.ndarray]
        At least 4 entries of uint8 [H, W, 3].
    device : torch.device
        GPU / CPU device for inference.
    orig_h, orig_w : int
        Original frame dimensions (may differ from FloweR's internal size).

    Returns
    -------
    tuple(pred_flow, pred_occl, pred_next, flower_h, flower_w)
        pred_flow : Tensor [fH, fW, 2]  -- pixel displacements
        pred_occl : Tensor [fH, fW, 1]  -- occlusion in [0, 255]
        pred_next : Tensor [fH, fW, 3]  -- predicted next frame in [0, 255]
        flower_h, flower_w : int         -- FloweR-aligned dimensions
    """
    flower_h = (orig_h // 128) * 128
    flower_w = (orig_w // 128) * 128

    # Stack last 4 frames: [4, H, W, 3] normalised to [-1, 1]
    last4 = frame_history[-4:]
    stack = np.stack(last4, axis=0).astype(np.float32)
    stack_t = torch.from_numpy(stack / 127.5 - 1.0)  # [-1, 1]

    # Resize to FloweR-aligned dimensions if needed
    if orig_h != flower_h or orig_w != flower_w:
        stack_t = stack_t.permute(0, 3, 1, 2)  # [4, 3, H, W]
        stack_t = torch.nn.functional.interpolate(
            stack_t, size=(flower_h, flower_w), mode="bilinear", align_corners=False
        )
        stack_t = stack_t.permute(0, 2, 3, 1)  # [4, fH, fW, 3]

    with torch.no_grad():
        # FloweR expects [B, 4, H, W, 3]
        pred_data = flower_net(stack_t.unsqueeze(0).to(device))  # [1, fH, fW, 6]

    pred_data = pred_data[0]  # [fH, fW, 6]

    # Denormalise
    pred_flow = pred_data[..., 0:2] * 255.0                   # pixel displacements
    pred_occl = (pred_data[..., 2:3] + 1) * 127.5             # [0, 255]
    pred_next = (pred_data[..., 3:6] + 1) * 127.5             # [0, 255]

    return (
        pred_flow.cpu(),
        pred_occl.cpu(),
        pred_next.cpu(),
        flower_h,
        flower_w,
    )


# ============================================================================
# Flow-based frame warping
# ============================================================================

@torch.no_grad()
def flow_warp_frame(frame_np, pred_flow, flower_h, flower_w, orig_h, orig_w, device):
    """Warp a numpy frame using FloweR's predicted optical flow.

    Parameters
    ----------
    frame_np : np.ndarray [H, W, 3] uint8
        Frame to warp.
    pred_flow : Tensor [fH, fW, 2]
        Pixel displacements from flower_inference.
    flower_h, flower_w : int
        FloweR-aligned dimensions.
    orig_h, orig_w : int
        Original frame dimensions.
    device : torch.device
        GPU / CPU device for grid_sample.

    Returns
    -------
    np.ndarray [H, W, 3] uint8
        Warped frame at original resolution.
    """
    # Resize frame to FloweR dimensions if needed
    if orig_h != flower_h or orig_w != flower_w:
        frame_resized = cv2.resize(frame_np, (flower_w, flower_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame_np

    # Convert to tensor [-1, 1] on device: [1, 3, fH, fW]
    frame_t = torch.from_numpy(frame_resized.astype(np.float32) / 127.5 - 1.0)
    frame_t = frame_t.permute(2, 0, 1).unsqueeze(0).to(device)

    # Build sampling grid from flow
    fh, fw = flower_h, flower_w
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, fh), torch.arange(0, fw), indexing='ij'
    )
    flow_grid = torch.stack((grid_x, grid_y), dim=0).float().unsqueeze(0).to(device)
    flow_grid = flow_grid + pred_flow.permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalise to [-1, 1]
    flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (fw - 1) - 1
    flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (fh - 1) - 1
    flow_grid = flow_grid.permute(0, 2, 3, 1)  # [1, fH, fW, 2]

    warped_t = torch.nn.functional.grid_sample(
        frame_t, flow_grid, mode="nearest", padding_mode="reflection", align_corners=False
    )

    # Convert back to uint8 numpy
    warped_np = warped_t[0].permute(1, 2, 0).cpu().numpy()  # [fH, fW, 3]
    warped_np = ((warped_np + 1.0) * 127.5)
    warped_np = np.clip(warped_np, 0, 255).astype(np.uint8)

    # Resize back to original dimensions if needed
    if orig_h != flower_h or orig_w != flower_w:
        warped_np = cv2.resize(warped_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return warped_np


# ============================================================================
# Prediction blending
# ============================================================================

def blend_flower_prediction(affine_warped_np, flower_prediction_np, flower_blend):
    """Weighted blend between affine-warped and FloweR-predicted frames.

    Parameters
    ----------
    affine_warped_np : np.ndarray [H, W, 3] uint8
        Frame produced by affine warp.
    flower_prediction_np : np.ndarray [H, W, 3] uint8
        Frame predicted by FloweR.
    flower_blend : float
        Blend weight in [0, 1].  0 = all affine, 1 = all FloweR.

    Returns
    -------
    np.ndarray [H, W, 3] uint8
    """
    result = (
        flower_blend * flower_prediction_np.astype(np.float32)
        + (1.0 - flower_blend) * affine_warped_np.astype(np.float32)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# Occlusion mask extraction
# ============================================================================

@torch.no_grad()
def compute_flower_occlusion(pred_occl, orig_h, orig_w, flower_h, flower_w,
                             dilate_px=4, threshold=0.1, occl_multiplier=10.0):
    """Extract a binary occlusion mask from FloweR's occlusion prediction.

    Parameters
    ----------
    pred_occl : Tensor [fH, fW, 1]
        Occlusion prediction in [0, 255] (from flower_inference).
    orig_h, orig_w : int
        Target mask dimensions.
    flower_h, flower_w : int
        FloweR-aligned dimensions.
    dilate_px : int
        Dilation kernel radius (0 to disable).
    threshold : float
        Normalised threshold in [0, 1] for binarisation.
    occl_multiplier : float
        Contrast multiplier applied before thresholding.

    Returns
    -------
    np.ndarray [H, W] float32 in {0.0, 1.0}, or None if mask area < 25 px.
    """
    # Normalise to [0, 1], apply contrast multiplier, then threshold
    occl_norm = torch.clamp((pred_occl / 255.0) * occl_multiplier, 0, 1)

    # Threshold to binary
    mask = (occl_norm > threshold).float()  # [fH, fW, 1]

    # Resize to original dimensions if needed
    if orig_h != flower_h or orig_w != flower_w:
        # [fH, fW, 1] -> [1, 1, fH, fW] for interpolate
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask, size=(orig_h, orig_w), mode="nearest"
        )
        mask = mask[0, 0]  # [H, W]
    else:
        mask = mask[:, :, 0]  # [fH, fW]

    mask_np = mask.cpu().numpy().astype(np.float32)

    # Dilate to cover edge artefacts
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask_np = cv2.dilate(mask_np, k)

    # Skip trivially small masks
    if mask_np.sum() < 25:
        return None

    return mask_np


# ============================================================================
# Affine displacement measurement
# ============================================================================

def compute_affine_displacement(M, w, h):
    """Compute max pixel displacement from a 2x3 affine matrix.

    Tests the four image corners and returns the largest Euclidean distance
    between original and transformed positions.

    Parameters
    ----------
    M : np.ndarray [2, 3]
        Affine transformation matrix.
    w, h : int
        Image dimensions.

    Returns
    -------
    float
        Maximum corner displacement in pixels.
    """
    corners = np.array([
        [0,     0,     1],
        [w - 1, 0,     1],
        [0,     h - 1, 1],
        [w - 1, h - 1, 1],
    ], dtype=np.float64)

    max_disp = 0.0
    for corner in corners:
        original = corner[:2]
        transformed = M @ corner  # [2,]
        disp = np.sqrt(np.sum((transformed - original) ** 2))
        if disp > max_disp:
            max_disp = disp

    return max_disp
