"""
SD-CN Feedback Animation
========================

Merged ComfyUI node: pixel-space affine-feedback loop with occlusion-aware
two-pass inpainting.

Combines:
  - pizurny/Comfyui-FeedbackSampler   (MIT) -- affine-feedback architecture
  - pxl-pshr/ComfyUI-SD-CN-Animation  (MIT) -- two-pass occlusion inpaint
  - deforum/deforum-stable-diffusion        -- pixel-space warp + cadence

Architecture (Deforum-canonical):
  Per frame, decode the previous latent to pixels, apply an affine warp in
  pixel space, compute an analytical occlusion mask from the warp itself
  (no FloweR/RAFT needed -- we know exactly which pixels are newly revealed),
  apply color coherence and polish, encode back to latent, run a masked
  KSampler at high denoise on occluded regions plus an optional full-frame
  refinement pass at low denoise, decode. Frames where (i % cadence != 0)
  skip the SD pass entirely and just propagate the warped pixels -- this is
  the Deforum cost-amortization trick.

Why pixel-space warp: VAEs are not equivariant to translation/rotation, so
latent-space warps drift visibly. Eight years of Deforum users have settled
on pixel-space warp + VAE round-trip; we follow.
"""

from collections import deque

import torch
import numpy as np
import cv2

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management as mm

from .flower_shared import (
    FloweR, flower_inference, flow_warp_frame,
    blend_flower_prediction, compute_flower_occlusion,
    compute_affine_displacement,
)

try:
    from skimage.exposure import match_histograms
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ============================================================================
# Tensor / numpy helpers
# ============================================================================

def _tensor_to_np(img_tensor):
    """ComfyUI IMAGE tensor [1, H, W, 3] (0-1 float) -> np uint8 [H, W, 3]."""
    img = img_tensor[0].detach().cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _np_to_tensor(img_np):
    """np uint8 [H, W, 3] -> ComfyUI IMAGE tensor [1, H, W, 3] (0-1 float)."""
    img = img_np.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


# ============================================================================
# Affine warp + analytical occlusion mask
# ============================================================================

_BORDER_MODES = {
    "zeros": cv2.BORDER_CONSTANT,
    "border": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
}


def _build_affine(zoom, tx, ty, angle_deg, w, h):
    """Center-anchored: rotate + scale around image center, then translate."""
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0 + zoom)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def _warp_pixel(img_np, M, border_mode):
    h, w = img_np.shape[:2]
    return cv2.warpAffine(
        img_np, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=_BORDER_MODES.get(border_mode, cv2.BORDER_REFLECT_101),
        borderValue=(0, 0, 0),
    )


def _compute_occlusion_mask(M, w, h, dilate_px=4):
    """
    Pixels where the inverse-warp lands outside the source image are 'newly
    revealed'. Warp a ones-validity mask with the same M and threshold to
    find them. 1 = newly revealed (will be inpainted), 0 = preserved.
    """
    validity = np.ones((h, w), dtype=np.float32)
    warped = cv2.warpAffine(
        validity, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    mask = (warped < 0.99).astype(np.float32)
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k)
    return mask  # [H, W] in {0, 1}


# ============================================================================
# Color coherence
# ============================================================================

def _stat_match(src_f, ref_f):
    """Per-channel mean/std matching. Inputs/outputs are float32 [H, W, 3]."""
    out = np.zeros_like(src_f)
    for c in range(3):
        s_mean, s_std = src_f[..., c].mean(), src_f[..., c].std()
        r_mean, r_std = ref_f[..., c].mean(), ref_f[..., c].std()
        if s_std > 1e-6:
            out[..., c] = (src_f[..., c] - s_mean) * (r_std / s_std) + r_mean
        else:
            out[..., c] = src_f[..., c]
    return out


def _apply_color_match(source_np, reference_np, mode):
    """Per-frame subtle color coherence in LAB/RGB/HSV. Anchors against frame 0."""
    if mode in (None, "None"):
        return source_np
    if mode == "RGB":
        out_f = _stat_match(source_np.astype(np.float32),
                            reference_np.astype(np.float32))
        return np.clip(out_f, 0, 255).astype(np.uint8)
    if mode == "LAB":
        cvt_to, cvt_from = cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB
    elif mode == "HSV":
        cvt_to, cvt_from = cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB
    else:
        return source_np
    src = cv2.cvtColor(source_np, cvt_to).astype(np.float32)
    ref = cv2.cvtColor(reference_np, cvt_to).astype(np.float32)
    out = np.clip(_stat_match(src, ref), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cvt_from)


def _histogram_anchor(source_np, reference_np):
    """Stronger periodic re-anchor against drift. Uses skimage if available."""
    if _HAS_SKIMAGE:
        return match_histograms(
            source_np, reference_np, channel_axis=-1
        ).astype(np.uint8)
    # Fallback: per-channel CDF mapping
    out = np.zeros_like(source_np)
    for c in range(3):
        s_vals, s_cnt = np.unique(source_np[..., c].ravel(), return_counts=True)
        s_cdf = np.cumsum(s_cnt).astype(np.float64); s_cdf /= s_cdf[-1]
        r_vals, r_cnt = np.unique(reference_np[..., c].ravel(), return_counts=True)
        r_cdf = np.cumsum(r_cnt).astype(np.float64); r_cdf /= r_cdf[-1]
        interp_vals = np.interp(s_cdf, r_cdf, r_vals)
        mapped = np.interp(source_np[..., c].ravel(), s_vals, interp_vals)
        out[..., c] = mapped.reshape(source_np.shape[:2]).astype(np.uint8)
    return out


# ============================================================================
# Polish (anti-blur, anti-stagnation)
# ============================================================================

def _unsharp(img_np, amount):
    if amount <= 0:
        return img_np
    blurred = cv2.GaussianBlur(img_np, (0, 0), 1.5, 1.5)
    out = cv2.addWeighted(img_np, 1.0 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_noise(img_np, amount, rng):
    if amount <= 0:
        return img_np
    noise = rng.normal(0, amount * 255.0, img_np.shape).astype(np.float32)
    out = img_np.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================================
# Per-frame ControlNet application (mirrors comfy ControlNetApplyAdvanced)
# ============================================================================

def _apply_cn_per_frame(positive, negative, control_net, hint_tensor, strength):
    if control_net is None or strength <= 0:
        return positive, negative
    control_hint = hint_tensor.movedim(-1, 1)  # [1, H, W, 3] -> [1, 3, H, W]
    cnets = {}
    out_lists = []
    for cond in [positive, negative]:
        new_cond = []
        for t in cond:
            d = t[1].copy()
            prev = d.get("control", None)
            if prev in cnets:
                cnet = cnets[prev]
            else:
                cnet = control_net.copy().set_cond_hint(
                    control_hint, strength, (0.0, 1.0)
                )
                cnet.set_previous_controlnet(prev)
                cnets[prev] = cnet
            d["control"] = cnet
            d["control_apply_to_uncond"] = False
            new_cond.append([t[0], d])
        out_lists.append(new_cond)
    return out_lists[0], out_lists[1]


# ============================================================================
# KSampler wrapper (handles mask resize to latent dims)
# ============================================================================

def _sample(model, latent_dict, positive, negative, seed, steps, cfg,
            sampler_name, scheduler, denoise, mask_np=None):
    latent = latent_dict["samples"]
    if denoise <= 0.001:
        return {"samples": latent.clone()}

    noise = comfy.sample.prepare_noise(latent, seed)

    noise_mask = None
    if mask_np is not None:
        # Pixel-space mask -> latent-space mask via nearest-neighbor downsample.
        # Read latent dims from tensor (handles SD1.5/SDXL/Flux differences).
        lh, lw = latent.shape[-2:]
        m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
        m = torch.nn.functional.interpolate(m, size=(lh, lw), mode="nearest")
        noise_mask = m.to(latent.device)

    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent,
        denoise=denoise,
        disable_noise=False,
        start_step=None, last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        seed=seed,
    )
    return {"samples": samples}


# ============================================================================
# Node class
# ============================================================================

class SDCNFeedbackAnimation:
    """
    Pixel-space affine-feedback animation with occlusion-aware inpainting.

    Per-frame loop:
      1. decode prev latent -> pixels
      2. affine warp + analytical occlusion mask
      3. color match (LAB/RGB/HSV) against frame 0
      4. periodic histogram anchor (every N frames)
      5. unsharp + noise injection
      6. if (i % cadence == 0): VAE encode -> masked KSampler -> [refine pass]
                                -> VAE decode
         else: keep warped pixels (cadence amortization)
      7. append to output

    After the loop, optionally blend the last `loop_frames` toward frame 0
    for seamless looping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "init_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "num_frames": ("INT", {"default": 60, "min": 2, "max": 2000}),
                "processing_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fix_frame_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diffusion_cadence": ("INT", {"default": 1, "min": 1, "max": 8}),
                "zoom": ("FLOAT", {"default": 0.0, "min": -0.05, "max": 0.05, "step": 0.001}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.5}),
                "angle": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "border_mode": (["reflect", "border", "zeros"], {"default": "reflect"}),
                "occlusion_inpaint": ("BOOLEAN", {"default": True}),
                "occlusion_dilate": ("INT", {"default": 4, "min": 0, "max": 32}),
                "color_coherence": (["None", "LAB", "RGB", "HSV"], {"default": "LAB"}),
                "color_anchor_every": ("INT", {"default": 15, "min": 0, "max": 120}),
                "noise_amount": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                "sharpen_amount": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "loop_frames": ("INT", {"default": 0, "min": 0, "max": 60}),
                "warp_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "flower_model": ("FLOWER_MODEL",),
                "flower_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "How much FloweR's predicted next frame influences the init image. 0=affine warp with flow refinement, 1=pure FloweR prediction. Requires flower_model."}),
                "control_net": ("CONTROL_NET",),
                "control_hints": ("IMAGE",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "SyntaxNodes/Animation"

    def _flower_process(self, flower_net, frame_history, affine_warped_np,
                        flower_blend, device, h, w, dilate_px, M):
        """Full FloweR pipeline: flow warp + prediction blend + occlusion.

        Returns (init_image_np, occ_mask_np) or (None, None) if skipped.
        """
        # Extreme motion guard: skip FloweR if affine displacement > 50px
        displacement = compute_affine_displacement(M, w, h)
        if displacement > 50.0:
            return None, None

        # Run FloweR inference
        pred_flow, pred_occl, pred_next, flower_h, flower_w = flower_inference(
            flower_net, list(frame_history), device, h, w
        )

        # Flow-warp the affine-warped frame for per-pixel refinement
        flow_warped = flow_warp_frame(
            affine_warped_np, pred_flow, flower_h, flower_w, h, w, device
        )

        # Resize FloweR's prediction to original dimensions
        pred_next_np = pred_next.numpy()
        pred_next_np = np.clip(pred_next_np, 0, 255).astype(np.uint8)
        if flower_h != h or flower_w != w:
            pred_next_np = cv2.resize(pred_next_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blend flow-warped affine with FloweR's prediction
        if flower_blend > 0.0:
            init_image = blend_flower_prediction(flow_warped, pred_next_np, flower_blend)
        else:
            init_image = flow_warped

        # Extract occlusion mask
        occ_mask = compute_flower_occlusion(
            pred_occl, h, w, flower_h, flower_w, dilate_px
        )

        return init_image, occ_mask

    def generate(self, model, vae, positive, negative, init_image,
                 seed, steps, cfg, sampler_name, scheduler,
                 num_frames, processing_strength, fix_frame_strength,
                 diffusion_cadence, zoom, translate_x, translate_y, angle,
                 border_mode, warp_strength, occlusion_inpaint, occlusion_dilate,
                 color_coherence, color_anchor_every,
                 noise_amount, sharpen_amount, loop_frames,
                 flower_model=None, flower_blend=0.5,
                 control_net=None, control_hints=None,
                 cn_strength=1.0):

        device = mm.get_torch_device()
        rng = np.random.default_rng(seed)

        # --- FloweR setup (optional ML-based occlusion) ---
        flower_net = None
        if flower_model is not None:
            flower_h = (init_image.shape[1] // 128) * 128
            flower_w = (init_image.shape[2] // 128) * 128
            if flower_h > 0 and flower_w > 0:
                flower_net = FloweR(input_size=(flower_h, flower_w))
                flower_net.load_state_dict(flower_model["state_dict"])
                flower_net.to(device).eval()

        # Frame history buffer for FloweR (needs last 4 frames)
        frame_history = deque(maxlen=4)

        # --- Frame 0: seed ---
        first_pixel = _tensor_to_np(init_image)
        h, w = first_pixel.shape[:2]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(
                f"init_image dimensions ({w}x{h}) must be divisible by 8 for VAE."
            )

        frames = [first_pixel]
        prev_pixel = first_pixel.copy()
        frame_history.append(first_pixel)

        pbar = comfy.utils.ProgressBar(max(1, num_frames - 1))

        # --- Main loop ---
        for i in range(1, num_frames):
            # 1. Affine matrix for this frame
            M = _build_affine(zoom, translate_x, translate_y, angle, w, h)

            # 2. Warp prev pixel
            warped = _warp_pixel(prev_pixel, M, border_mode)

            # 2b. Blend warped with unwarped to reduce warp dominance
            if warp_strength < 1.0:
                warped = cv2.addWeighted(
                    warped, warp_strength,
                    prev_pixel, 1.0 - warp_strength, 0)

            # 3. Per-frame subtle color coherence (vs frame 0)
            if color_coherence != "None":
                warped = _apply_color_match(warped, first_pixel, color_coherence)

            # 4. Periodic strong histogram anchor
            if color_anchor_every > 0 and i % color_anchor_every == 0:
                warped = _histogram_anchor(warped, first_pixel)

            # 5. Polish
            warped = _unsharp(warped, sharpen_amount)
            warped = _add_noise(warped, noise_amount, rng)

            # 6. Cadence: skip SD on non-cadence frames
            if (i % diffusion_cadence) != 0:
                frames.append(warped)
                prev_pixel = warped
                frame_history.append(warped)
                pbar.update(1)
                continue

            # 7. FloweR integration (only on cadence frames)
            occ_mask = None
            flower_init = None

            if flower_net is not None and len(frame_history) >= 4:
                # Compute effective flower_blend (taper during loop closure)
                effective_blend = flower_blend
                if loop_frames > 0:
                    frames_remaining = num_frames - 1 - i
                    if frames_remaining < loop_frames:
                        loop_progress = 1.0 - (frames_remaining / loop_frames)
                        effective_blend = flower_blend * (1.0 - loop_progress)

                flower_init, occ_mask = self._flower_process(
                    flower_net, frame_history, warped,
                    effective_blend, device, h, w, occlusion_dilate, M
                )

            if occ_mask is None and occlusion_inpaint:
                # Analytical fallback from affine warp
                mask = _compute_occlusion_mask(M, w, h, occlusion_dilate)
                if mask.sum() >= 25:
                    occ_mask = mask

            # 8. Encode to latent (use FloweR-enhanced init if available)
            encode_source = flower_init if flower_init is not None else warped
            warped_t = _np_to_tensor(encode_source).to(device)
            latent = {"samples": vae.encode(warped_t)}

            # 9. Per-frame ControlNet (optional)
            if (control_net is not None
                    and control_hints is not None
                    and control_hints.shape[0] > 0):
                idx = (i - 1) % control_hints.shape[0]
                hint = control_hints[idx:idx + 1].to(device)
                pos_f, neg_f = _apply_cn_per_frame(
                    positive, negative, control_net, hint, cn_strength
                )
            else:
                pos_f, neg_f = positive, negative

            # 10. First pass -- full-frame generation (fights zoom everywhere)
            frame_seed = seed + i
            latent = _sample(
                model, latent, pos_f, neg_f,
                frame_seed, steps, cfg, sampler_name, scheduler,
                processing_strength, mask_np=None,
            )

            # 11. Occlusion refinement pass (boost newly revealed edges)
            if occ_mask is not None and fix_frame_strength > 0.001:
                refine_steps = max(8, steps // 2)
                latent = _sample(
                    model, latent, pos_f, neg_f,
                    frame_seed + 100000, refine_steps, cfg, sampler_name, scheduler,
                    fix_frame_strength, mask_np=occ_mask,
                )

            # 12. Decode to pixels
            decoded = vae.decode(latent["samples"])
            current_pixel = _tensor_to_np(decoded)

            frames.append(current_pixel)
            prev_pixel = current_pixel
            frame_history.append(current_pixel)
            pbar.update(1)

        # Cleanup FloweR
        if flower_net is not None:
            flower_net.to("cpu")
            del flower_net
            mm.soft_empty_cache()

        # 13. Loop closure -- linear blend last loop_frames toward frame 0
        if loop_frames > 0 and num_frames > loop_frames + 1:
            for k in range(loop_frames):
                idx = num_frames - loop_frames + k
                blend = (k + 1) / (loop_frames + 1)
                frames[idx] = cv2.addWeighted(
                    frames[idx], 1.0 - blend, frames[0], blend, 0
                )

        # Stack all frames as IMAGE batch
        out = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return (torch.from_numpy(out),)


# ============================================================================
# Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SDCNFeedbackAnimation": SDCNFeedbackAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNFeedbackAnimation": "SD-CN Feedback Animation",
}
