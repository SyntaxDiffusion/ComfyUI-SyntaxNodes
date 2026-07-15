"""
SyntaxNodes Prompt Travel KSampler
==================================
A KSampler that generates latent batches by interpolating between
multiple prompts using SLERP for latents and LERP for conditioning.

Enter prompts separated by | and it handles everything internally.
"""

import torch
import numpy as np
from comfy.utils import ProgressBar
import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview

from .syntax_schedule.ScheduleFuncs import pad_with_zeros


def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    original_shape = low.shape
    original_device = low.device

    low_flat = low.reshape(-1).float()
    high_flat = high.reshape(-1).float()

    low_norm = torch.norm(low_flat) + 1e-7
    high_norm = torch.norm(high_flat) + 1e-7
    low_normalized = low_flat / low_norm
    high_normalized = high_flat / high_norm

    dot = torch.clamp(torch.sum(low_normalized * high_normalized), -1.0, 1.0)

    if abs(dot) > 0.9995:
        result = lerp(val, low, high)
        return result.to(original_device)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    if sin_theta_0 < 1e-7:
        result = lerp(val, low, high)
        return result.to(original_device)

    theta_t = theta_0 * val
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    result = (s0 * low_flat + s1 * high_flat).reshape(original_shape)
    return result.to(original_device)


def lerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return (1.0 - val) * low + val * high


def interpolate_cond(cond1: list, cond2: list, alpha: float) -> list:
    """Interpolate between two ComfyUI conditioning objects."""
    if not cond1 or not cond2:
        return cond1 if cond1 else cond2

    result = []

    for i in range(max(len(cond1), len(cond2))):
        c1 = cond1[min(i, len(cond1) - 1)]
        c2 = cond2[min(i, len(cond2) - 1)]

        tensor1, meta1 = c1
        tensor2, meta2 = c2

        if tensor1.shape == tensor2.shape:
            interp_tensor = lerp(alpha, tensor1, tensor2)
        else:
            # Repeat-pad the shorter sequence (same as the schedule layer) —
            # zero tokens are out-of-distribution for variable-length
            # encoders (Qwen/Krea/Flux) and visibly degrade them.
            max_len = max(tensor1.shape[1], tensor2.shape[1])
            tensor1, _ = pad_with_zeros(tensor1, max_len)
            tensor2, _ = pad_with_zeros(tensor2, max_len)
            interp_tensor = lerp(alpha, tensor1, tensor2)

        interp_meta = meta1.copy()
        for key, value in meta2.items():
            if key not in interp_meta:
                interp_meta[key] = value
            elif key == 'pooled_output':
                p1 = meta1.get('pooled_output')
                p2 = meta2.get('pooled_output')
                # Handle None cases
                if p1 is None and p2 is None:
                    interp_meta['pooled_output'] = None
                elif p1 is None:
                    interp_meta['pooled_output'] = p2
                elif p2 is None:
                    interp_meta['pooled_output'] = p1
                elif p1.shape == p2.shape:
                    interp_meta['pooled_output'] = lerp(alpha, p1, p2)
                else:
                    interp_meta['pooled_output'] = p1 if alpha < 0.5 else p2

        result.append((interp_tensor, interp_meta))

    return result


class SyntaxPromptTravelKSampler:
    """
    Prompt Travel KSampler - All-in-one prompt travel sampling.

    Enter any number of prompts separated by | and this node handles:
    - CLIP encoding of all prompts
    - SLERP interpolation of latent noise between keyframes
    - LERP interpolation of conditioning between prompts
    - Full diffusion sampling for each frame

    Example: "sunrise | noon | sunset | night" with 30 frames per transition
    generates 90 frames of smooth transitions.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Diffusion model for denoising"
                }),
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for encoding prompts"
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE for latent dimensions"
                }),
                "prompts": ("STRING", {
                    "default": "a beautiful sunrise over mountains | a bright noon sky | a golden sunset | a starry night",
                    "multiline": True,
                    "tooltip": "Prompts separated by | - use as many as you want"
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Negative prompt applied to all frames"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Use 1.0 for Flux models"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "frames_per_transition": ("INT", {
                    "default": 30,
                    "min": 2,
                    "max": 500,
                    "tooltip": "Frames between each prompt keyframe"
                }),
                "interpolation_mode": (["slerp", "lerp"], {
                    "default": "slerp",
                    "tooltip": "slerp = spherical (smoother), lerp = linear"
                }),
            },
            "optional": {
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop back to first prompt at end"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    OUTPUT_TOOLTIPS = ("Batch of latents ready for VAE decode",)

    FUNCTION = "sample"
    CATEGORY = "SyntaxNodes/Sampling"

    DESCRIPTION = """All-in-one prompt travel KSampler.
Enter prompts separated by | (any number of prompts).
Example: "cat | dog | bird | fish" generates smooth transitions between all.
Outputs latent batch ready for VAE decode → video."""

    def encode_prompt(self, clip, text):
        """Encode text to conditioning."""
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]

    def sample(self, model, clip, vae, prompts, negative_prompt,
               width, height, seed, steps, cfg, sampler_name, scheduler,
               denoise, frames_per_transition, interpolation_mode="slerp",
               loop=False):

        # Parse prompts - handle any number
        prompt_list = [p.strip() for p in prompts.split("|") if p.strip()]

        if len(prompt_list) < 2:
            raise ValueError("Need at least 2 prompts separated by |")

        print(f"[PromptTravel] Parsed {len(prompt_list)} prompts:")
        for i, p in enumerate(prompt_list):
            preview = p[:50] + "..." if len(p) > 50 else p
            print(f"  [{i+1}] {preview}")

        # Encode all prompts
        print(f"[PromptTravel] Encoding {len(prompt_list)} prompts...")
        encoded_prompts = []
        for prompt_text in prompt_list:
            cond = self.encode_prompt(clip, prompt_text)
            encoded_prompts.append(cond)

        # Encode negative
        negative = self.encode_prompt(clip, negative_prompt) if negative_prompt else self.encode_prompt(clip, "")

        # Add loop
        if loop:
            encoded_prompts.append(encoded_prompts[0])
            print(f"[PromptTravel] Loop enabled - added return to first prompt")

        num_prompts = len(encoded_prompts)
        num_transitions = num_prompts - 1
        total_frames = frames_per_transition * num_transitions

        print(f"[PromptTravel] {num_transitions} transitions × {frames_per_transition} frames = {total_frames} total frames")

        # Create empty latent - derive channel count and downscale from the
        # model/VAE instead of assuming SD1.5's 4ch //8 (Flux/Krea/SD3 use 16ch)
        try:
            latent_format = model.get_model_object("latent_format")
        except Exception:
            latent_format = model.model.latent_format
        latent_channels = latent_format.latent_channels

        downscale = getattr(vae, "downscale_ratio", 8)
        if not isinstance(downscale, int):
            downscale = 8

        latent_height = height // downscale
        latent_width = width // downscale
        print(f"[PromptTravel] Latent format: {latent_channels}ch, {latent_width}x{latent_height} (downscale {downscale})")
        base_latent = torch.zeros((1, latent_channels, latent_height, latent_width), device="cpu")

        # Generate noise keyframes (one per prompt)
        noise_keyframes = []
        for i in range(num_prompts):
            kf_seed = seed + (i * 10000)
            generator = torch.manual_seed(kf_seed)
            noise = torch.randn(
                (1, latent_channels, latent_height, latent_width),
                generator=generator,
                device="cpu"
            )
            noise_keyframes.append(noise)

        # Select interpolation function
        interp_func = slerp if interpolation_mode == "slerp" else lerp

        # Generate all frames
        output_latents = []
        pbar = ProgressBar(total_frames)
        frame_count = 0

        for t in range(num_transitions):
            cond_start = encoded_prompts[t]
            cond_end = encoded_prompts[t + 1]
            noise_start = noise_keyframes[t]
            noise_end = noise_keyframes[t + 1]

            print(f"[PromptTravel] Transition {t + 1}/{num_transitions}")

            for f in range(frames_per_transition):
                alpha = f / (frames_per_transition - 1) if frames_per_transition > 1 else 0.0

                # Interpolate noise
                frame_noise = interp_func(alpha, noise_start, noise_end)

                # Interpolate conditioning
                frame_cond = interpolate_cond(cond_start, cond_end, alpha)

                # Sample
                callback = latent_preview.prepare_callback(model, steps)

                samples = comfy.sample.sample(
                    model,
                    frame_noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    frame_cond,
                    negative,
                    base_latent,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    callback=callback,
                    disable_pbar=True,
                    seed=seed + frame_count
                )

                samples = samples.to(comfy.model_management.intermediate_device())
                output_latents.append(samples)

                frame_count += 1
                pbar.update_absolute(frame_count)

        # Stack all frames
        batched_latents = torch.cat(output_latents, dim=0)
        print(f"[PromptTravel] Complete! Output shape: {batched_latents.shape}")

        return ({"samples": batched_latents},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SyntaxPromptTravelKSampler": SyntaxPromptTravelKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyntaxPromptTravelKSampler": "Prompt Travel KSampler",
}
