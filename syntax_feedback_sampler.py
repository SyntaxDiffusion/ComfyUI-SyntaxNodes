import torch
import torch.nn.functional as F
import comfy.samplers
import numpy as np
import gc
import copy

from .syntax_schedule import schedule_conditioning
from .syntax_feedback_progress import SyntaxFeedbackProgress

# Try to import scipy for sharpening and noise
try:
    from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Sharpening and Perlin noise will be disabled. Install with: pip install scipy")

# Memory cleanup interval (every N frames)
MEMORY_CLEANUP_INTERVAL = 5


class SyntaxFeedbackSampler:
    """
    A sampler that feeds finished latent back into itself with zoom functionality.
    Creates deforum-style zooming animations through iterative feedback loops.
    Includes LAB color matching to prevent color bleeding.

    Combines FeedbackSampler with built-in static or FizzNodes-style batch
    prompting: connect a CLIP and enter either one prompt or a prompt schedule
    without an external conditioning or BatchPromptSchedule node. Handles
    variable-length text encoders (Qwen, Flux Krea, etc.).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # === Standard KSampler Parameters ===
                "model": ("MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # === Animation Parameters ===
                "zoom_value": ("FLOAT", {"default": 0.005, "min": -0.5, "max": 0.5, "step": 0.0001, "round": 0.0001}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 1000000}),
                "feedback_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_variation": (["fixed", "increment", "random"], {"default": "fixed"}),

                # === Deforum Travel Parameters (2D) ===
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
                "translation_x": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1}),
                "translation_y": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1}),

                # === Deforum Travel Parameters (3D) ===
                "translation_z": ("FLOAT", {"default": 0.0, "min": -500.0, "max": 500.0, "step": 0.1}),
                "rotation_3d_x": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
                "rotation_3d_y": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
                "rotation_3d_z": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),

                # === Color & Quality Enhancement ===
                "color_coherence": (["None", "LAB", "RGB", "HSV"], {"default": "None"}),
                "noise_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "noise_type": (["gaussian", "perlin"], {"default": "perlin"}),
                "sharpen_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_boost": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 1.5, "step": 0.01}),

                # === Lumina/zImage Smoothing Mode ===
                "lumina_mode": ("BOOLEAN", {"default": False}),
                "temporal_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "cond_blend_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "color_coherence_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "vae": ("VAE",),

                # === Mask Input ===
                "mask": ("MASK",),  # Optional mask - single or batch for selective diffusion

                # === Conditioning Inputs ===
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_batch": ("CONDITIONING",),
                "negative_batch": ("CONDITIONING",),

                # === Built-in Prompt Scheduling ===
                # Connect a CLIP and enter either a plain prompt or a schedule
                # internally (FizzNodes BatchPromptSchedule format):
                #   "0" :"a cat", "30" :"a dog --neg blurry"
                # Takes priority over positive_batch/positive when set.
                "clip": ("CLIP",),
                "prompt_schedule": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Plain prompt text or a keyed prompt schedule. Connect CLIP to encode it internally.",
                }),
                "negative_prompt_schedule": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Plain negative prompt text or a keyed negative prompt schedule.",
                }),

                # === Motion Schedules (override static values) ===
                # Format: "frame:(value), frame:(value)" - e.g., "0:(0), 20:(360)"
                # Presets: Rotate: "0:(0), 30:(360)" | Pan: "0:(0), 30:(-200)" | Zoom: "0:(0.01), 30:(0.1)"
                "angle_schedule": ("STRING", {"default": "", "multiline": False}),
                "translation_x_schedule": ("STRING", {"default": "", "multiline": False}),
                "translation_y_schedule": ("STRING", {"default": "", "multiline": False}),
                "translation_z_schedule": ("STRING", {"default": "", "multiline": False}),
                "rotation_3d_x_schedule": ("STRING", {"default": "", "multiline": False}),
                "rotation_3d_y_schedule": ("STRING", {"default": "", "multiline": False}),
                "rotation_3d_z_schedule": ("STRING", {"default": "", "multiline": False}),
                "zoom_schedule": ("STRING", {"default": "", "multiline": False}),

                # Keep last so older serialized widget arrays do not shift.
                "frame_cadence": ("INT", {
                    "default": 1, "min": 1, "max": 1000,
                    "tooltip": "Diffuse every Nth frame. Intermediate frames receive motion transforms without diffusion; 1 preserves the original behavior.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("final_latent", "all_latents")
    FUNCTION = "sample"
    CATEGORY = "SyntaxNodes/Sampling"

    def cleanup_memory(self, frame_num, force=False):
        """
        Periodically clean up GPU memory to prevent OOM errors on long runs.

        Args:
            frame_num: Current frame number
            force: If True, always cleanup regardless of interval
        """
        if force or (frame_num > 0 and frame_num % MEMORY_CLEANUP_INTERVAL == 0):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    @staticmethod
    def should_diffuse_frame(frame_index, frame_cadence):
        """Frame 0 is always sampled; later frames follow the cadence."""
        cadence = max(1, int(frame_cadence))
        return frame_index == 0 or frame_index % cadence == 0

    @staticmethod
    def sample_with_callback(model, seed, steps, cfg, sampler_name, scheduler,
                             positive, negative, latent, denoise, callback):
        """Equivalent to nodes.common_ksampler with an outer progress callback."""
        import comfy.sample
        import comfy.utils

        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model,
            latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        batch_inds = latent.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=latent.get("noise_mask"),
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed,
        )
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return out

    @staticmethod
    def is_krea2_pipeline(model=None, clip=None):
        """Detect Krea2 without importing or depending on ComfyUI internals.

        ComfyUI exposes Krea2 through both the model patcher's wrapped model and
        the CLIP tokenizer/text encoder. Class/module names are stable enough for
        capability gating while keeping this node compatible with older ComfyUI
        versions and lightweight test doubles.
        """
        candidates = [model, clip]
        for obj in (model, clip):
            if obj is None:
                continue
            for attr in ("model", "tokenizer", "cond_stage_model"):
                candidates.append(getattr(obj, attr, None))

        for candidate in candidates:
            if candidate is None:
                continue
            cls = type(candidate)
            identity = f"{cls.__module__}.{cls.__name__}".lower()
            if "krea2" in identity:
                return True
        return False

    @staticmethod
    def krea_transition_timestep_split(denoise, structural_fraction=0.25):
        """Place the prompt-average cutoff inside the active img2img steps.

        ``denoise`` selects the final fraction of the full diffusion trajectory.
        Mixed conditioning is allowed only for the first structural fraction of
        that active range; all remaining steps use a pure prompt.
        """
        denoise = min(1.0, max(0.0, float(denoise)))
        structural_fraction = min(1.0, max(0.0, float(structural_fraction)))
        active_start = 1.0 - denoise
        return active_start + denoise * structural_fraction

    def parse_schedule(self, schedule_string, max_frames):
        """
        Parse FizzNodes-style schedule string into per-frame values.
        Format: "0:(value1), 10:(value2), 20:(value3)"
        Returns: List of interpolated values for each frame
        """
        import re
        import numexpr

        if not schedule_string or schedule_string.strip() == "":
            return None

        # Parse keyframes from string
        keyframes = {}
        pattern = r'(\d+):\s*\(([^)]+)\)'
        matches = re.findall(pattern, schedule_string)

        for frame_str, value_str in matches:
            frame = int(frame_str)
            # Support numexpr expressions
            try:
                value = float(numexpr.evaluate(value_str))
            except:
                value = float(value_str)
            keyframes[frame] = value

        if not keyframes:
            return None

        # Sort keyframes
        sorted_frames = sorted(keyframes.keys())

        # Interpolate values for all frames
        values = []
        for i in range(max_frames):
            # Find surrounding keyframes
            before_frame = None
            after_frame = None

            for kf in sorted_frames:
                if kf <= i:
                    before_frame = kf
                if kf > i and after_frame is None:
                    after_frame = kf

            # Interpolate
            if before_frame is None:
                # Before first keyframe - use first value
                values.append(keyframes[sorted_frames[0]])
            elif after_frame is None:
                # After last keyframe - use last value
                values.append(keyframes[before_frame])
            else:
                # Between keyframes - linear interpolation
                progress = (i - before_frame) / (after_frame - before_frame)
                value = keyframes[before_frame] + (keyframes[after_frame] - keyframes[before_frame]) * progress
                values.append(value)

        return values

    def parse_prompt_schedule(self, schedule_string, max_frames, clip):
        """
        Parse prompt schedule and return list of conditioning for each frame.
        Format: "0: A cat, 10: A dog, 20: A bird"
        """
        import re

        if not schedule_string or schedule_string.strip() == "":
            return None

        # Parse keyframes
        keyframes = {}
        # Match "frame: prompt text" pattern
        pattern = r'(\d+):\s*([^,]+?)(?=\s*(?:\d+:|$))'
        matches = re.findall(pattern, schedule_string.replace('\n', ' '))

        for frame_str, prompt in matches:
            frame = int(frame_str)
            keyframes[frame] = prompt.strip()

        if not keyframes:
            return None

        # Create conditioning for each keyframe
        keyframe_conds = {}
        for frame, prompt in keyframes.items():
            tokens = clip.tokenize(prompt)
            cond = clip.encode_from_tokens(tokens, return_pooled=True)
            keyframe_conds[frame] = cond

        # Interpolate between keyframes for all frames
        sorted_frames = sorted(keyframes.keys())
        frame_conds = []

        for i in range(max_frames):
            # Find surrounding keyframes
            before_frame = None
            after_frame = None

            for kf in sorted_frames:
                if kf <= i:
                    before_frame = kf
                if kf > i and after_frame is None:
                    after_frame = kf

            if before_frame is None:
                # Before first keyframe
                frame_conds.append(keyframe_conds[sorted_frames[0]])
            elif after_frame is None:
                # After last keyframe
                frame_conds.append(keyframe_conds[before_frame])
            else:
                # Between keyframes - blend conditioning
                progress = (i - before_frame) / (after_frame - before_frame)
                weight = 1.0 - progress

                # Blend the two conditionings
                cond_from = keyframe_conds[before_frame]
                cond_to = keyframe_conds[after_frame]

                # Simple weighted blend (could use FizzNodes' addWeighted for better results)
                blended = [[
                    cond_from[0][0] * weight + cond_to[0][0] * (1.0 - weight),
                    cond_from[0][1]  # Use first conditioning's metadata for now
                ]]
                frame_conds.append(blended)

        return frame_conds

    def match_color_histogram(self, source, reference, mode="LAB", strength=1.0):
        """
        Match color histogram of source image to reference image.
        This is the critical function that prevents color bleeding.

        Args:
            source: Image to adjust (numpy array HxWx3, values 0-255)
            reference: Target color distribution (numpy array HxWx3, values 0-255)
            mode: Color space for matching ("LAB", "RGB", "HSV")
            strength: How much to apply the color matching (0.0 = no change, 1.0 = full match)

        Returns:
            Color-matched image (numpy array HxWx3, values 0-255)
        """
        if mode == "None" or strength <= 0:
            return source

        # ComfyUI temporal VAEs decode a single latent batch as T,H,W,C.
        # Match each video frame independently; treating the temporal axis as
        # height corrupts both color channels and nearly the entire image.
        if source.ndim == 4:
            if reference.ndim == 3:
                references = [reference] * source.shape[0]
            elif reference.ndim == 4 and reference.shape[0] > 0:
                references = [reference[min(i, reference.shape[0] - 1)] for i in range(source.shape[0])]
            else:
                raise ValueError(f"Unsupported reference image shape: {reference.shape}")
            return np.stack([
                self.match_color_histogram(frame, references[i], mode, strength)
                for i, frame in enumerate(source)
            ], axis=0)

        if source.ndim != 3 or source.shape[-1] < 3:
            raise ValueError(f"Expected H,W,C or T,H,W,C image data, got {source.shape}")
        if reference.ndim != 3 or reference.shape[-1] < 3:
            raise ValueError(f"Expected H,W,C reference image, got {reference.shape}")

        # Ensure uint8 type
        source = source.astype(np.uint8)
        reference = reference.astype(np.uint8)

        if mode == "LAB":
            # Convert to LAB color space (most perceptually uniform)
            # LAB separates lightness from color, best for preventing color drift
            source_lab = self.rgb_to_lab(source)
            reference_lab = self.rgb_to_lab(reference)

            # Match histogram for each channel
            matched_lab = np.zeros_like(source_lab)
            for i in range(3):
                matched_lab[:, :, i] = self.match_histograms(
                    source_lab[:, :, i],
                    reference_lab[:, :, i]
                )

            # Convert back to RGB
            result = self.lab_to_rgb(matched_lab)

        elif mode == "HSV":
            # HSV mode - good for maintaining hue consistency
            source_hsv = self.rgb_to_hsv(source)
            reference_hsv = self.rgb_to_hsv(reference)

            matched_hsv = np.zeros_like(source_hsv)
            for i in range(3):
                matched_hsv[:, :, i] = self.match_histograms(
                    source_hsv[:, :, i],
                    reference_hsv[:, :, i]
                )

            result = self.hsv_to_rgb(matched_hsv)

        else:  # RGB
            # Direct RGB matching
            result = np.zeros_like(source)
            for i in range(3):
                result[:, :, i] = self.match_histograms(
                    source[:, :, i],
                    reference[:, :, i]
                )

        # Blend original with matched based on strength (partial color matching)
        if strength < 1.0:
            result = source.astype(np.float32) * (1.0 - strength) + result.astype(np.float32) * strength
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result.astype(np.uint8)
    
    def match_histograms(self, source, reference):
        """
        Match histogram of source channel to reference channel.
        Uses cumulative distribution function (CDF) matching.
        """
        # Calculate histograms
        source_values, source_counts = np.unique(source.ravel(), return_counts=True)
        reference_values, reference_counts = np.unique(reference.ravel(), return_counts=True)
        
        # Calculate CDFs
        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]
        
        reference_cdf = np.cumsum(reference_counts).astype(np.float64)
        reference_cdf /= reference_cdf[-1]
        
        # Interpolate to find mapping
        interp_values = np.interp(source_cdf, reference_cdf, reference_values)
        
        # Build lookup table
        lookup = np.zeros(256, dtype=reference.dtype)
        for i, val in enumerate(source_values):
            lookup[val] = interp_values[i]
        
        # Apply lookup table
        return lookup[source]
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        # Normalize to 0-1
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Apply gamma correction
        mask = rgb_norm > 0.04045
        rgb_linear = np.where(mask, 
                              np.power((rgb_norm + 0.055) / 1.055, 2.4),
                              rgb_norm / 12.92)
        
        # RGB to XYZ
        xyz = np.zeros_like(rgb_linear)
        xyz[:, :, 0] = rgb_linear[:, :, 0] * 0.4124564 + rgb_linear[:, :, 1] * 0.3575761 + rgb_linear[:, :, 2] * 0.1804375
        xyz[:, :, 1] = rgb_linear[:, :, 0] * 0.2126729 + rgb_linear[:, :, 1] * 0.7151522 + rgb_linear[:, :, 2] * 0.0721750
        xyz[:, :, 2] = rgb_linear[:, :, 0] * 0.0193339 + rgb_linear[:, :, 1] * 0.1191920 + rgb_linear[:, :, 2] * 0.9503041
        
        # Normalize by D65 white point
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883
        
        # XYZ to LAB
        mask = xyz > 0.008856
        f = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))
        
        lab = np.zeros_like(xyz)
        lab[:, :, 0] = (116 * f[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (f[:, :, 0] - f[:, :, 1])  # a
        lab[:, :, 2] = 200 * (f[:, :, 1] - f[:, :, 2])  # b
        
        # Scale to 0-255 for histogram matching
        lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0  # L: 0-100 -> 0-255
        lab[:, :, 1] = (lab[:, :, 1] + 128.0)  # a: -128-127 -> 0-255
        lab[:, :, 2] = (lab[:, :, 2] + 128.0)  # b: -128-127 -> 0-255
        
        return np.clip(lab, 0, 255).astype(np.uint8)
    
    def lab_to_rgb(self, lab):
        """Convert LAB back to RGB with proper bounds checking"""
        # Unscale from 0-255
        lab_float = lab.astype(np.float32)
        lab_float[:, :, 0] = lab_float[:, :, 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
        lab_float[:, :, 1] = lab_float[:, :, 1] - 128.0  # a: 0-255 -> -128-127
        lab_float[:, :, 2] = lab_float[:, :, 2] - 128.0  # b: 0-255 -> -128-127
        
        # LAB to XYZ
        fy = (lab_float[:, :, 0] + 16) / 116
        fx = lab_float[:, :, 1] / 500 + fy
        fz = fy - lab_float[:, :, 2] / 200
        
        # Ensure positive values before power operations
        fx = np.maximum(fx, 0.0)
        fy = np.maximum(fy, 0.0)
        fz = np.maximum(fz, 0.0)
        
        mask_x = fx > 0.2068966
        mask_y = fy > 0.2068966
        mask_z = fz > 0.2068966
        
        xyz = np.zeros_like(lab_float)
        xyz[:, :, 0] = np.where(mask_x, np.power(fx, 3), (fx - 16/116) / 7.787)
        xyz[:, :, 1] = np.where(mask_y, np.power(fy, 3), (fy - 16/116) / 7.787)
        xyz[:, :, 2] = np.where(mask_z, np.power(fz, 3), (fz - 16/116) / 7.787)
        
        # Clip XYZ to valid range
        xyz = np.clip(xyz, 0.0, 1.0)
        
        # Denormalize by D65 white point
        xyz[:, :, 0] *= 0.95047
        xyz[:, :, 1] *= 1.00000
        xyz[:, :, 2] *= 1.08883
        
        # XYZ to RGB
        rgb_linear = np.zeros_like(xyz)
        rgb_linear[:, :, 0] = xyz[:, :, 0] *  3.2404542 + xyz[:, :, 1] * -1.5371385 + xyz[:, :, 2] * -0.4985314
        rgb_linear[:, :, 1] = xyz[:, :, 0] * -0.9692660 + xyz[:, :, 1] *  1.8760108 + xyz[:, :, 2] *  0.0415560
        rgb_linear[:, :, 2] = xyz[:, :, 0] *  0.0556434 + xyz[:, :, 1] * -0.2040259 + xyz[:, :, 2] *  1.0572252
        
        # Clip to valid range before gamma correction (CRITICAL!)
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        
        # Apply gamma correction - now safe from negative values
        mask = rgb_linear > 0.0031308
        rgb = np.where(mask,
                      1.055 * np.power(rgb_linear, 1/2.4) - 0.055,
                      12.92 * rgb_linear)
        
        # Final clip and convert
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        # Safety check for NaN or Inf
        if np.any(np.isnan(rgb)) or np.any(np.isinf(rgb)):
            print("  WARNING: Invalid values detected in LAB->RGB conversion, using fallback")
            return np.zeros_like(rgb, dtype=np.uint8) + 128  # Return gray as fallback
        
        return rgb
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        s = np.where(maxc != 0, deltac / maxc, 0)
        
        rc = np.where(deltac != 0, (maxc - r) / deltac, 0)
        gc = np.where(deltac != 0, (maxc - g) / deltac, 0)
        bc = np.where(deltac != 0, (maxc - b) / deltac, 0)
        
        h = np.zeros_like(r)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        
        hsv = np.stack([h, s, v], axis=2)
        return (hsv * 255).astype(np.uint8)
    
    def hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        hsv_norm = hsv.astype(np.float32) / 255.0
        h, s, v = hsv_norm[:, :, 0], hsv_norm[:, :, 1], hsv_norm[:, :, 2]
        
        i = (h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        rgb = np.zeros((*h.shape, 3), dtype=np.float32)
        
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        mask = (i == 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        
        return (rgb * 255).astype(np.uint8)
    
    def latent_to_image(self, latent, vae):
        """Convert one latent batch item to H,W,C or T,H,W,C pixels."""
        decoded = vae.decode(latent)
        if decoded.ndim not in (4, 5) or decoded.shape[0] != 1:
            raise ValueError(f"Expected one decoded image/video batch, got {tuple(decoded.shape)}")

        # 2D VAEs return B,H,W,C. Temporal VAEs return B,T,H,W,C.
        img = decoded[0].detach().cpu().numpy()
        
        # Convert from 0-1 to 0-255
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        
        return img
    
    def image_to_latent(self, image, vae, expected_shape=None):
        """Convert H,W,C or T,H,W,C pixels back to a latent."""
        img = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim != 4:
            raise ValueError(f"Expected H,W,C or T,H,W,C image data, got {tuple(img_tensor.shape)}")

        # Let ComfyUI's VAE wrapper manage device placement. For temporal VAEs
        # the leading dimension is time; adding another batch dimension crops a
        # one-frame video to T=0 and triggers WanVAE's unbound `out` failure.
        latent = vae.encode(img_tensor)
        if expected_shape is not None and tuple(latent.shape) != tuple(expected_shape):
            raise ValueError(
                f"VAE round trip changed latent shape from {tuple(expected_shape)} "
                f"to {tuple(latent.shape)}"
            )
        return latent
    
    def generate_perlin_noise(self, shape, scale=10, octaves=4, rng=None):
        """
        Generate Perlin-like noise for organic texture - SIMPLIFIED for performance.
        Creates structured noise instead of random static.
        
        Args:
            shape: (H, W, C) for the noise
            scale: Lower = larger features (default 10)
            octaves: More = more detail layers (default 4)
        """
        rng = rng or np.random.default_rng()

        if not SCIPY_AVAILABLE:
            print("    [Perlin noise unavailable without scipy, using Gaussian]", flush=True)
            return rng.standard_normal(shape, dtype=np.float32) * 0.5 + 0.5

        # Handle different shape formats
        if len(shape) == 3:
            H, W, C = shape
        elif len(shape) == 4:
            # Temporal VAE image format [T, H, W, C]. Give every frame its
            # own field rather than broadcasting one pattern through time.
            return np.stack([
                self.generate_perlin_noise(shape[1:], scale, octaves, rng)
                for _ in range(shape[0])
            ], axis=0)
        else:
            raise ValueError(f"Unexpected image shape format: {shape}")

        H, W, C = shape
        noise = np.zeros(shape, dtype=np.float32)
        
        print(f"    [Generating Perlin noise {H}x{W}x{C}...]", end=" ", flush=True)
        
        for c in range(C):
            channel_noise = np.zeros((H, W), dtype=np.float32)
            
            for octave in range(octaves):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)
                
                # Generate random base at lower resolution
                grid_size = max(4, scale // freq)
                grid_h = H // grid_size + 2
                grid_w = W // grid_size + 2
                
                # Generate random values at grid points
                grid_noise = rng.standard_normal((grid_h, grid_w), dtype=np.float32) * amp
                
                # Upsample using bilinear interpolation (much faster than per-pixel)
                upsampled = scipy_zoom(grid_noise, (H / grid_h, W / grid_w), order=1)
                
                # Crop to exact size
                upsampled = upsampled[:H, :W]
                
                channel_noise += upsampled
            
            noise[:, :, c] = channel_noise
        
        # Normalize to 0-1 range
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        print("Done!", flush=True)
        return noise
    
    def apply_noise_pixel(self, image, amount, noise_type="gaussian", seed=None):
        """
        Add noise in pixel space (after color coherence).
        This is critical - adding noise BEFORE color coherence gets removed by histogram matching!
        
        Args:
            image: numpy array (H, W, C), values 0-255
            amount: noise strength (0-1)
            noise_type: "gaussian" or "perlin"
        """
        if amount <= 0:
            return image
        
        img_float = image.astype(np.float32)
        rng = np.random.default_rng(seed)
        
        if noise_type == "perlin":
            # Generate Perlin noise (organic, structured)
            noise = self.generate_perlin_noise(image.shape, scale=8, octaves=4, rng=rng)
            # Scale to -1 to 1 range
            noise = (noise - 0.5) * 2.0
            # Scale by amount (treat amount as intensity)
            noise_scaled = noise * (amount * 30.0)  # 30 is max intensity
        else:
            # Gaussian noise (random static)
            noise = rng.standard_normal(image.shape, dtype=np.float32)
            noise_scaled = noise * (amount * 15.0)  # 15 is max intensity
        
        # Add noise to image
        noisy = img_float + noise_scaled
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_noise(self, latent, amount, seed=None):
        """
        Add controlled noise to latent to prevent stagnation.
        This helps maintain detail at low denoise values.
        """
        if amount <= 0:
            return latent
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=latent.device)
            generator.manual_seed(seed)
        noise = torch.randn(
            latent.shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        ) * amount
        return latent + noise
    
    def apply_sharpening(self, image, amount):
        """
        Apply unsharp masking to recover detail lost in VAE encode/decode.
        This is critical for maintaining sharpness at low denoise values.
        
        Args:
            image: numpy array (H, W, C), values 0-255
            amount: sharpening strength (0 = no sharpening, 1 = maximum)
        """
        if amount <= 0 or not SCIPY_AVAILABLE:
            return image
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Blur spatial axes only. A scalar sigma also blurs RGB channels and,
        # for temporal VAEs, adjacent frames before applying the unsharp mask.
        sigma = [0.0] * img_float.ndim
        sigma[-3] = 1.0
        sigma[-2] = 1.0
        blurred = gaussian_filter(img_float, sigma=tuple(sigma))
        
        # Unsharp mask: original + amount * (original - blurred)
        sharpened = img_float + amount * (img_float - blurred)
        
        # Clip and convert back
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_contrast(self, image, boost):
        """
        Apply contrast adjustment to prevent washed-out colors.

        Args:
            image: numpy array (H, W, C), values 0-255
            boost: contrast multiplier (1.0 = no change, >1.0 = more contrast)
        """
        # This control is intentionally a boost. Older saved workflows may
        # contain the former 0.9 default; treat that as neutral instead of
        # washing out every feedback frame.
        if boost <= 1.0:
            return image

        # Convert to float
        img_float = image.astype(np.float32)

        # Apply contrast around midpoint (127.5)
        midpoint = 127.5
        contrasted = (img_float - midpoint) * boost + midpoint

        # Clip and convert back
        return np.clip(contrasted, 0, 255).astype(np.uint8)

    def apply_temporal_smoothing(self, current_image, previous_image, strength):
        """
        Blend current frame with previous frame for temporal consistency.
        Reduces flicker and jitter between frames.

        Args:
            current_image: numpy array (H, W, C), values 0-255
            previous_image: numpy array (H, W, C), values 0-255, or None
            strength: blend strength (0.0 = no blend, 0.5 = 50% previous frame)

        Returns:
            Temporally smoothed image
        """
        if strength <= 0 or previous_image is None:
            return current_image

        current_float = current_image.astype(np.float32)
        previous_float = previous_image.astype(np.float32)
        blended = current_float * (1.0 - strength) + previous_float * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    def blend_conditioning(self, current_cond, previous_cond, strength):
        """
        Blend current conditioning with previous frame's conditioning for smoother prompt transitions.
        This reduces abrupt changes when prompts change via FizzNodes scheduling.

        Args:
            current_cond: Current frame conditioning [[tensor, dict]]
            previous_cond: Previous frame conditioning [[tensor, dict]], or None
            strength: blend strength (0.0 = use current, 0.5 = 50% blend with previous)

        Returns:
            Blended conditioning [[tensor, dict]]
        """
        if strength <= 0 or previous_cond is None:
            return current_cond

        # Extract tensors and metadata
        tensor_curr = current_cond[0][0]
        tensor_prev = previous_cond[0][0]
        meta_curr = current_cond[0][1] if len(current_cond[0]) > 1 else {}
        meta_prev = previous_cond[0][1] if len(previous_cond[0]) > 1 else {}

        # Pad tensors to same sequence length if needed
        if tensor_curr.shape[1] != tensor_prev.shape[1]:
            max_len = max(tensor_curr.shape[1], tensor_prev.shape[1])
            if tensor_curr.shape[1] < max_len:
                tensor_curr = F.pad(tensor_curr, (0, 0, 0, max_len - tensor_curr.shape[1]))
            if tensor_prev.shape[1] < max_len:
                tensor_prev = F.pad(tensor_prev, (0, 0, 0, max_len - tensor_prev.shape[1]))

        # Interpolate conditioning tensors
        blended_tensor = tensor_curr * (1.0 - strength) + tensor_prev * strength

        # Create blended metadata
        blended_meta = meta_curr.copy()

        # Interpolate pooled_output if both exist
        pooled_curr = meta_curr.get("pooled_output")
        pooled_prev = meta_prev.get("pooled_output")
        if pooled_curr is not None and pooled_prev is not None:
            blended_meta["pooled_output"] = pooled_curr * (1.0 - strength) + pooled_prev * strength

        return [[blended_tensor, blended_meta]]

    def set_latent_noise_mask(self, latent_dict, mask):
        """
        Set the noise_mask on a latent dict, matching SetLatentNoiseMask behavior.

        The mask defines which areas to diffuse:
        - White (1.0) = areas to regenerate/diffuse
        - Black (0.0) = areas to preserve from input latent

        Args:
            latent_dict: The latent dictionary to modify
            mask: Input mask tensor - any shape, will be reshaped to (B, 1, H, W)

        Returns:
            Modified latent_dict with noise_mask set (or unchanged if mask is None)
        """
        if mask is None:
            return latent_dict

        # Reshape mask to (B, 1, H, W) format - same as SetLatentNoiseMask
        # This handles (H, W), (B, H, W), or already (B, 1, H, W)
        latent_dict["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return latent_dict

    def get_mask_for_iteration(self, mask, iteration):
        """
        Get the appropriate mask for the current iteration from a batch of masks.
        Returns mask in (H, W) format for single mask or (1, H, W) for batch element.

        Args:
            mask: Input mask tensor (B, H, W) or (H, W), or None
            iteration: Current frame index

        Returns:
            Mask tensor for this iteration (H, W), or None
        """
        if mask is None:
            return None

        # Handle 2D mask (H, W) - single mask for all iterations
        if len(mask.shape) == 2:
            return mask

        # Handle 3D mask (B, H, W)
        if mask.shape[0] == 1:
            # Single mask in batch format - use for all iterations
            return mask[0]
        elif mask.shape[0] > iteration:
            # Use matching mask for this iteration
            return mask[iteration]
        else:
            # Use last mask for remaining iterations
            return mask[-1]

    def zoom_latent(self, latent, zoom_factor):
        """
        Apply zoom transformation to latent tensor.
        Positive zoom_factor = zoom in (scale up)
        Negative zoom_factor = zoom out (scale down)
        Zero = no change
        Supports both 4D (standard) and 5D (Qwen/video) latents.

        Returns:
            tuple: (zoomed_latent, None). The second item is retained for
            internal call compatibility; zoom-out no longer uses outpainting.
        """
        if zoom_factor == 0:
            return latent, None

        # Calculate scale factor (1 + zoom means zoom in, 1 - zoom means zoom out)
        scale = 1.0 + zoom_factor

        # Handle both 4D and 5D latent formats
        is_5d = len(latent.shape) == 5
        if is_5d:
            # Temporal format: [batch, channels, depth, height, width]. Move
            # depth next to batch before flattening or channel/time data is
            # reinterpreted in the wrong order whenever depth > 1.
            batch, channels, depth, height, width = latent.shape
            latent = latent.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        else:
            # Standard format: [batch, channels, height, width]
            batch, channels, height, width = latent.shape

        # Calculate new dimensions for zoom
        if zoom_factor > 0:  # Zoom in - sample from center
            new_height = int(height / scale)
            new_width = int(width / scale)

            # Calculate crop coordinates (center crop)
            top = (height - new_height) // 2
            left = (width - new_width) // 2

            # Crop center region
            cropped = latent[:, :, top:top+new_height, left:left+new_width]

            # Scale back to original size
            zoomed = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)

        else:  # Zoom out - centered affine scale followed by full re-diffusion
            inverse_scale = 1.0 / scale
            theta = torch.tensor(
                [[inverse_scale, 0.0, 0.0], [0.0, inverse_scale, 0.0]],
                device=latent.device,
                dtype=latent.dtype,
            ).unsqueeze(0).expand(latent.shape[0], -1, -1)
            grid = F.affine_grid(theta, latent.shape, align_corners=False)

            # Reflection avoids flat/black borders. The entire transformed
            # latent is re-diffused below exactly like the zoom-in path;
            # skipping an automatic mask prevents accumulated center blur.
            zoomed = F.grid_sample(
                latent,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )

        # Reshape back to 5D if needed
        if is_5d:
            zoomed = zoomed.reshape(batch, depth, channels, height, width)
            # Transpose to original format: [batch, channels, depth, height, width]
            zoomed = zoomed.permute(0, 2, 1, 3, 4)

        return zoomed, None

    def apply_deforum_transform(self, latent, angle=0.0, translation_x=0.0, translation_y=0.0,
                                translation_z=0.0, rotation_3d_x=0.0, rotation_3d_y=0.0, rotation_3d_z=0.0):
        """
        Apply Deforum-style transformations to latent.
        Combines 2D (angle, translate) and simulated 3D transformations.

        Args:
            latent: Tensor to transform
            angle: 2D rotation in degrees (clockwise positive)
            translation_x: Horizontal shift in pixels (right positive)
            translation_y: Vertical shift in pixels (down positive)
            translation_z: Depth translation (forward positive) - simulated with zoom
            rotation_3d_x: Pitch rotation in degrees (tilt up/down)
            rotation_3d_y: Yaw rotation in degrees (pan left/right)
            rotation_3d_z: Roll rotation in degrees (roll clockwise/counter)
        """
        # Handle both 4D and 5D latent formats
        is_5d = len(latent.shape) == 5
        if is_5d:
            batch, channels, depth, height, width = latent.shape
            latent = latent.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        else:
            batch, channels, height, width = latent.shape

        # Apply 2D rotation (angle)
        if angle != 0:
            # Convert to radians
            angle_rad = torch.tensor(angle * 3.14159 / 180.0)
            # Create rotation matrix
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

            # Create affine transformation matrix
            theta = torch.tensor([[
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ]], dtype=latent.dtype, device=latent.device)

            # Expand for batch
            theta = theta.expand(latent.shape[0], 2, 3)

            # Apply rotation
            grid = F.affine_grid(theta, latent.size(), align_corners=False)
            latent = F.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Apply 2D translation
        if translation_x != 0 or translation_y != 0:
            # Normalize translation to [-1, 1] range based on latent dimensions
            tx = (translation_x / width) * 2.0
            ty = (translation_y / height) * 2.0

            # Create translation matrix
            theta = torch.tensor([[
                [1, 0, tx],
                [0, 1, ty]
            ]], dtype=latent.dtype, device=latent.device)

            theta = theta.expand(latent.shape[0], 2, 3)
            grid = F.affine_grid(theta, latent.size(), align_corners=False)
            latent = F.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Apply translation_z as zoom (simulated depth)
        if translation_z != 0:
            # Positive Z = zoom in (move forward), negative = zoom out
            zoom_factor = translation_z / 100.0  # Scale down for smoother effect
            latent, _ = self.zoom_latent(latent, zoom_factor)  # Ignore padding info here

        # 3D rotations (simplified - apply as perspective-like transforms)
        # Note: True 3D would require depth maps, so we simulate with 2D transforms

        # rotation_3d_x (pitch - tilt up/down) - simulate with vertical scaling
        if rotation_3d_x != 0:
            scale_factor = 1.0 + (rotation_3d_x / 180.0) * 0.1  # Subtle effect
            theta = torch.tensor([[
                [1, 0, 0],
                [0, scale_factor, 0]
            ]], dtype=latent.dtype, device=latent.device)
            theta = theta.expand(latent.shape[0], 2, 3)
            grid = F.affine_grid(theta, latent.size(), align_corners=False)
            latent = F.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # rotation_3d_y (yaw - pan left/right) - simulate with horizontal shift
        if rotation_3d_y != 0:
            tx = (rotation_3d_y / 180.0) * 0.2  # Convert rotation to translation
            theta = torch.tensor([[
                [1, 0, tx],
                [0, 1, 0]
            ]], dtype=latent.dtype, device=latent.device)
            theta = theta.expand(latent.shape[0], 2, 3)
            grid = F.affine_grid(theta, latent.size(), align_corners=False)
            latent = F.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # rotation_3d_z (roll) - same as 2D angle rotation
        if rotation_3d_z != 0:
            angle_rad = torch.tensor(rotation_3d_z * 3.14159 / 180.0)
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            theta = torch.tensor([[
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0]
            ]], dtype=latent.dtype, device=latent.device)
            theta = theta.expand(latent.shape[0], 2, 3)
            grid = F.affine_grid(theta, latent.size(), align_corners=False)
            latent = F.grid_sample(latent, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Reshape back to 5D if needed
        if is_5d:
            latent = latent.reshape(batch, depth, channels, height, width)
            latent = latent.permute(0, 2, 1, 3, 4)

        return latent

    def unbatch_conditioning(self, conditioning):
        """Split CONDITIONING tensors by batch while preserving all metadata."""
        if not conditioning:
            return []

        batch_sizes = [
            item[0].shape[0]
            for item in conditioning
            if isinstance(item, (list, tuple))
            and item
            and hasattr(item[0], "shape")
            and len(item[0].shape) > 0
        ]
        batch_size = max(batch_sizes, default=1)
        if batch_size <= 1:
            return [conditioning]

        print(f"[Conditioning] Unbatching {batch_size} conditioning frames")
        frames = [[] for _ in range(batch_size)]
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or not item:
                continue

            cond_tensor = item[0]
            metadata = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            tensor_is_batched = (
                hasattr(cond_tensor, "shape")
                and len(cond_tensor.shape) > 0
                and cond_tensor.shape[0] == batch_size
            )

            for i in range(batch_size):
                frame_tensor = cond_tensor[i:i + 1] if tensor_is_batched else cond_tensor
                frame_metadata = {}
                for key, value in metadata.items():
                    value_is_batched = (
                        hasattr(value, "shape")
                        and len(value.shape) > 0
                        and value.shape[0] == batch_size
                    )
                    frame_metadata[key] = value[i:i + 1] if value_is_batched else value
                frames[i].append([frame_tensor, frame_metadata])

        return frames

    def get_conditioning_for_frame(self, conditioning, frame_idx):
        """Select a frame from an upstream batched ControlNet hint, if present."""
        try:
            frame_count = 1
            for item in conditioning:
                control = item[1].get("control") if len(item) > 1 else None
                hint = getattr(control, "cond_hint_original", None)
                if hint is not None and hasattr(hint, "shape"):
                    frame_count = max(frame_count, hint.shape[0])
            if frame_count <= 1:
                return conditioning

            frame_conditioning = copy.deepcopy(conditioning)
            actual_idx = frame_idx % frame_count
            for item in frame_conditioning:
                control = item[1].get("control") if len(item) > 1 else None
                hint = getattr(control, "cond_hint_original", None)
                if hint is not None and hasattr(hint, "shape") and hint.shape[0] > 1:
                    hint_idx = actual_idx % hint.shape[0]
                    control.cond_hint_original = hint[hint_idx:hint_idx + 1]
            return frame_conditioning
        except (IndexError, KeyError, AttributeError, TypeError):
            return conditioning

    def fit_conditioning_frames(self, frames, frame_count, label):
        """Repeat or trim per-frame conditionings to the sampler frame count."""
        if not frames:
            raise ValueError(f"{label} conditioning is empty")
        if len(frames) == frame_count:
            return frames
        if len(frames) == 1:
            print(f"[Conditioning] Reusing one {label} conditioning for {frame_count} frames")
            return frames * frame_count
        if len(frames) < frame_count:
            print(f"[Conditioning] Extending {label} conditioning from {len(frames)} to {frame_count} frames")
            return frames + [frames[-1]] * (frame_count - len(frames))
        print(f"[Conditioning] Trimming {label} conditioning from {len(frames)} to {frame_count} frames")
        return frames[:frame_count]

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               latent_image, denoise, zoom_value, iterations, feedback_denoise, seed_variation,
               angle, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z,
               color_coherence, noise_amount, noise_type, sharpen_amount, contrast_boost,
               lumina_mode=False, temporal_smoothing=0.0, cond_blend_strength=0.0, color_coherence_strength=1.0,
               frame_cadence=1,
               vae=None, mask=None,
               positive=None, negative=None, positive_batch=None, negative_batch=None,
               clip=None, prompt_schedule="", negative_prompt_schedule="",
               angle_schedule="", translation_x_schedule="", translation_y_schedule="",
               translation_z_schedule="", rotation_3d_x_schedule="", rotation_3d_y_schedule="",
               rotation_3d_z_schedule="", zoom_schedule="", unique_id=None):
        """
        Main sampling function with feedback loop, zoom, and color coherence.
        Now with FizzNodes-style prompt and motion scheduling!
        Includes Lumina/zImage smoothing mode for reduced flicker and smoother transitions.
        """
        import random

        krea2_mode = self.is_krea2_pipeline(model, clip)
        frame_cadence = max(1, int(frame_cadence))

        # === LUMINA MODE: Apply optimized presets for Lumina/zImage models ===
        if lumina_mode:
            print(f"\n=== SyntaxFeedbackSampler - LUMINA MODE ENABLED ===")
            temporal_smoothing = 0.2
            cond_blend_strength = 0.25
            color_coherence_strength = 0.7
            # Clamp feedback_denoise to optimal range for Lumina
            if feedback_denoise < 0.35 or feedback_denoise > 0.45:
                original_denoise = feedback_denoise
                feedback_denoise = 0.4
                print(f"[Lumina Mode] feedback_denoise adjusted: {original_denoise:.2f} -> {feedback_denoise:.2f}")
            print(f"[Lumina Mode] Presets: temporal_smoothing=0.2, cond_blend=0.25, color_strength=0.7")
        else:
            print(f"\n=== SyntaxFeedbackSampler with Prompt & Motion Scheduling ===")

        if krea2_mode and cond_blend_strength > 0:
            print(
                f"[Krea2] conditioning blend disabled ({cond_blend_strength:.2f} -> 0.00); "
                "Krea2 prompt transitions are handled by the built-in gated scheduler"
            )
            cond_blend_strength = 0.0

        # === STEP 0: Built-in Prompt Scheduling ===
        # When a CLIP is connected and prompt_schedule is set, build batch
        # conditioning internally from either plain text or schedule syntax.
        # This replaces external CLIP Text Encode / BatchPromptSchedule nodes.
        if clip is not None and prompt_schedule.strip():
            input_batch_size = latent_image["samples"].shape[0]
            num_frames = input_batch_size if input_batch_size > 1 else iterations
            # Cache across runs: node instances persist, and re-encoding costs a
            # full text-encoder pass per unique prompt (a 4B LLM for Krea2/Qwen).
            # Seed-only re-runs must not pay that again.
            interpolation_mode = "krea_transition" if krea2_mode else "linear"
            # Timestep percentages cover the full diffusion trajectory. Feedback
            # img2img samples only its final `feedback_denoise` fraction. Krea gets
            # the average for only the first 25% of those active structural steps,
            # then snaps back to a pure prompt for the remaining 75%.
            timestep_split = (
                self.krea_transition_timestep_split(feedback_denoise)
                if krea2_mode else 0.6
            )
            cache_key = (
                prompt_schedule, negative_prompt_schedule, num_frames,
                interpolation_mode, timestep_split,
            )
            if getattr(self, "_sched_key", None) == cache_key and getattr(self, "_sched_clip", None) is clip:
                print("[Schedule] Reusing cached batch conditioning (schedule unchanged)")
                positive_batch, negative_batch = self._sched_result
            else:
                print(
                    f"[Schedule] Building batch conditioning from prompt input "
                    f"({num_frames} frames, {interpolation_mode})"
                )
                sched_pos, sched_neg = schedule_conditioning(
                    prompt_schedule, clip, num_frames,
                    interpolation_mode=interpolation_mode,
                    timestep_split=timestep_split,
                )
                positive_batch = sched_pos
                if negative_prompt_schedule.strip():
                    # A dedicated negative schedule overrides any --neg parts
                    sched_neg, _ = schedule_conditioning(
                        negative_prompt_schedule, clip, num_frames,
                        interpolation_mode=interpolation_mode,
                        timestep_split=timestep_split,
                    )
                negative_batch = sched_neg
                self._sched_key = cache_key
                self._sched_clip = clip
                self._sched_result = (positive_batch, negative_batch)
        elif prompt_schedule.strip():
            print("[Schedule] Prompt input set but no CLIP connected - connect a CLIP to encode it internally")

        # === STEP 1: Process Conditioning (Batch or Scheduled) ===
        input_batch_size = latent_image["samples"].shape[0]
        frame_count = input_batch_size if input_batch_size > 1 else iterations

        # Priority: 1. Batch/scheduled conditioning, 2. Static conditioning.
        if positive_batch is not None:
            print("[Conditioning] Using batch/scheduled conditioning")
            positive_conds = self.unbatch_conditioning(positive_batch)
            negative_source = negative_batch if negative_batch is not None else negative
            if negative_source is None:
                raise ValueError("Batch/scheduled positive conditioning requires negative_batch or negative conditioning")
            negative_conds = self.unbatch_conditioning(negative_source)
        elif positive is not None and negative is not None:
            print(f"[Conditioning] Using static conditioning for all {frame_count} frames")
            positive_conds = [positive]
            negative_conds = [negative]
        else:
            raise ValueError("Connect a CLIP and enter a prompt, or connect optional static/batch conditioning inputs")

        # Never fabricate pooled embeddings: their width and semantics are
        # model-specific. Scheduled CLIP encoding already supplies the real
        # pooled output when the encoder supports it.
        positive_conds = self.fit_conditioning_frames(positive_conds, frame_count, "positive")
        negative_conds = self.fit_conditioning_frames(negative_conds, frame_count, "negative")

        # === STEP 2: Parse Motion Schedules ===
        print(f"[Motion] Processing motion schedules...")

        # Parse all motion schedules or use static values
        angle_values = self.parse_schedule(angle_schedule, iterations) if angle_schedule else [angle] * iterations
        tx_values = self.parse_schedule(translation_x_schedule, iterations) if translation_x_schedule else [translation_x] * iterations
        ty_values = self.parse_schedule(translation_y_schedule, iterations) if translation_y_schedule else [translation_y] * iterations
        tz_values = self.parse_schedule(translation_z_schedule, iterations) if translation_z_schedule else [translation_z] * iterations
        rx_values = self.parse_schedule(rotation_3d_x_schedule, iterations) if rotation_3d_x_schedule else [rotation_3d_x] * iterations
        ry_values = self.parse_schedule(rotation_3d_y_schedule, iterations) if rotation_3d_y_schedule else [rotation_3d_y] * iterations
        rz_values = self.parse_schedule(rotation_3d_z_schedule, iterations) if rotation_3d_z_schedule else [rotation_3d_z] * iterations
        zoom_values = self.parse_schedule(zoom_schedule, iterations) if zoom_schedule else [zoom_value] * iterations

        # Print schedule info
        schedules_active = []
        if angle_schedule: schedules_active.append("angle")
        if translation_x_schedule: schedules_active.append("translation_x")
        if translation_y_schedule: schedules_active.append("translation_y")
        if translation_z_schedule: schedules_active.append("translation_z")
        if rotation_3d_x_schedule: schedules_active.append("rotation_3d_x")
        if rotation_3d_y_schedule: schedules_active.append("rotation_3d_y")
        if rotation_3d_z_schedule: schedules_active.append("rotation_3d_z")
        if zoom_schedule: schedules_active.append("zoom")

        if schedules_active:
            print(f"[Motion] Active schedules: {', '.join(schedules_active)}")
        else:
            print(f"[Motion] Using static values (no schedules)")

        # Check if VAE is available for color coherence
        if color_coherence != "None" and vae is None:
            print("WARNING: Color coherence requested but no VAE provided. Disabling color coherence.")
            color_coherence = "None"

        # Store all latents for output
        all_latents = []
        color_reference = None  # Store first frame for color matching

        progress = SyntaxFeedbackProgress(
            model, frame_count, frame_cadence, steps, unique_id=unique_id
        )

        # Get initial latent
        input_latents = latent_image["samples"].clone()
        latent_format = latent_image.copy()

        # === MASK SETUP ===
        # Mask handling follows SetLatentNoiseMask pattern - just store in latent dict
        if mask is not None:
            mask_batch_size = 1 if len(mask.shape) == 2 else mask.shape[0]
            print(f"[Mask] Input mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"[Mask] Using {mask_batch_size} mask(s) at resolution {mask.shape[-2]}x{mask.shape[-1]}")
            print(f"[Mask] Value range: min={mask.min():.3f}, max={mask.max():.3f}")
            print(f"[Mask] White (1.0) = diffuse, Black (0.0) = preserve")

        # === DETECT BATCH MODE ===
        # If input has multiple latents (batch > 1), process each once without feedback loop
        batch_size = input_latents.shape[0]
        batch_mode = batch_size > 1

        if batch_mode:
            print(f"\n=== BATCH MODE: Processing {batch_size} input latents with feedback chain ===")

            # Adjust conditioning to match batch size
            if len(positive_conds) != batch_size:
                if len(positive_conds) == 1:
                    print(f"[Conditioning] Expanding single conditioning to {batch_size} frames")
                    positive_conds = positive_conds * batch_size
                    negative_conds = negative_conds * batch_size
                elif len(positive_conds) < batch_size:
                    print(f"[Conditioning] Padding {len(positive_conds)} conditionings to {batch_size} frames")
                    while len(positive_conds) < batch_size:
                        positive_conds.append(positive_conds[-1])
                        negative_conds.append(negative_conds[-1])
                else:
                    print(f"[Conditioning] Trimming {len(positive_conds)} conditionings to {batch_size} frames")
                    positive_conds = positive_conds[:batch_size]
                    negative_conds = negative_conds[:batch_size]

            # Parse motion schedules for batch size
            angle_values = self.parse_schedule(angle_schedule, batch_size) if angle_schedule else [angle] * batch_size
            tx_values = self.parse_schedule(translation_x_schedule, batch_size) if translation_x_schedule else [translation_x] * batch_size
            ty_values = self.parse_schedule(translation_y_schedule, batch_size) if translation_y_schedule else [translation_y] * batch_size
            tz_values = self.parse_schedule(translation_z_schedule, batch_size) if translation_z_schedule else [translation_z] * batch_size
            rx_values = self.parse_schedule(rotation_3d_x_schedule, batch_size) if rotation_3d_x_schedule else [rotation_3d_x] * batch_size
            ry_values = self.parse_schedule(rotation_3d_y_schedule, batch_size) if rotation_3d_y_schedule else [rotation_3d_y] * batch_size
            rz_values = self.parse_schedule(rotation_3d_z_schedule, batch_size) if rotation_3d_z_schedule else [rotation_3d_z] * batch_size
            zoom_values = self.parse_schedule(zoom_schedule, batch_size) if zoom_schedule else [zoom_value] * batch_size

            # Process first frame
            print(f"\n[Frame 0/{batch_size}] Starting with denoise={denoise}")
            progress.begin_frame(0, will_diffuse=True)
            latent_format["samples"] = input_latents[0:1]

            # Set noise_mask for frame 0 if mask provided
            iteration_mask = self.get_mask_for_iteration(mask, 0)
            if iteration_mask is not None:
                self.set_latent_noise_mask(latent_format, iteration_mask)
            elif "noise_mask" in latent_format:
                del latent_format["noise_mask"]

            frame_positive = self.get_conditioning_for_frame(positive_conds[0], 0)
            frame_negative = self.get_conditioning_for_frame(negative_conds[0], 0)
            result = self.sample_with_callback(
                model, seed, steps, cfg, sampler_name, scheduler,
                frame_positive, frame_negative, latent_format, denoise=denoise,
                callback=progress.sampling_callback(0),
            )

            current_latent = result["samples"]
            # Store on CPU to prevent GPU memory accumulation
            all_latents.append(current_latent.clone().cpu())

            # Store first frame as color reference
            previous_image = None  # For temporal smoothing
            if color_coherence != "None" and vae is not None:
                color_reference = self.latent_to_image(current_latent, vae)
                previous_image = color_reference.copy()  # Also use as first previous image
                print(f"[Frame 0] Stored as color reference ({color_coherence} mode)")

            progress.complete_frame(0, current_latent, was_diffused=True)

            # Initialize previous conditioning for blending
            previous_positive = positive_conds[0]
            previous_negative = negative_conds[0]

            # Process remaining frames with feedback chain
            for i in range(1, batch_size):
                will_diffuse = self.should_diffuse_frame(i, frame_cadence)
                progress.begin_frame(i, will_diffuse)

                # Determine seed for this frame
                if seed_variation == "fixed":
                    frame_seed = seed
                elif seed_variation == "increment":
                    frame_seed = seed + i
                else:  # random
                    frame_seed = random.randint(0, 0xffffffffffffffff)

                # Get scheduled values for this frame
                frame_angle = angle_values[i]
                frame_tx = tx_values[i]
                frame_ty = ty_values[i]
                frame_tz = tz_values[i]
                frame_rx = rx_values[i]
                frame_ry = ry_values[i]
                frame_rz = rz_values[i]
                frame_zoom = zoom_values[i]

                print(f"\n[Frame {i}/{batch_size}] denoise={feedback_denoise} | seed={frame_seed}")

                # Apply transforms to previous output (feedback)
                transformed_latent, _ = self.zoom_latent(current_latent, frame_zoom)

                if frame_angle != 0 or frame_tx != 0 or frame_ty != 0 or frame_tz != 0 or frame_rx != 0 or frame_ry != 0 or frame_rz != 0:
                    transformed_latent = self.apply_deforum_transform(
                        transformed_latent,
                        angle=frame_angle,
                        translation_x=frame_tx,
                        translation_y=frame_ty,
                        translation_z=frame_tz,
                        rotation_3d_x=frame_rx,
                        rotation_3d_y=frame_ry,
                        rotation_3d_z=frame_rz
                    )

                # Blend transformed feedback with input latent
                input_latent = input_latents[i:i+1]
                # Use feedback_denoise to control blend: higher = more input latent, lower = more feedback
                blended_latent = transformed_latent * (1.0 - feedback_denoise) + input_latent * feedback_denoise

                # Apply color coherence with smoothing
                if color_coherence != "None" and vae is not None and color_reference is not None:
                    try:
                        current_image = self.latent_to_image(blended_latent, vae)
                        matched_image = self.match_color_histogram(current_image, color_reference, color_coherence, color_coherence_strength)

                        if contrast_boost != 1.0:
                            matched_image = self.apply_contrast(matched_image, contrast_boost)
                        if sharpen_amount > 0:
                            matched_image = self.apply_sharpening(matched_image, sharpen_amount)
                        if noise_amount > 0:
                            matched_image = self.apply_noise_pixel(
                                matched_image, noise_amount, noise_type, frame_seed
                            )

                        # Apply temporal smoothing to reduce flicker
                        if temporal_smoothing > 0 and previous_image is not None:
                            matched_image = self.apply_temporal_smoothing(matched_image, previous_image, temporal_smoothing)

                        encoded = self.image_to_latent(matched_image, vae, blended_latent.shape)
                        previous_image = matched_image.copy()
                        blended_latent = encoded
                    except Exception as e:
                        print(f"  [Warning] Color coherence failed: {e}")
                elif noise_amount > 0:
                    blended_latent = self.apply_noise(blended_latent, noise_amount, frame_seed)

                latent_format["samples"] = blended_latent

                # Get user-provided mask for this frame
                iteration_mask = self.get_mask_for_iteration(mask, i)

                # Set noise_mask for this frame
                if iteration_mask is not None:
                    self.set_latent_noise_mask(latent_format, iteration_mask)
                elif "noise_mask" in latent_format:
                    del latent_format["noise_mask"]

                # Apply conditioning blending for smoother prompt transitions
                frame_positive = positive_conds[i]
                frame_negative = negative_conds[i]
                if cond_blend_strength > 0:
                    frame_positive = self.blend_conditioning(positive_conds[i], previous_positive, cond_blend_strength)
                    frame_negative = self.blend_conditioning(negative_conds[i], previous_negative, cond_blend_strength)

                frame_positive = self.get_conditioning_for_frame(frame_positive, i)
                frame_negative = self.get_conditioning_for_frame(frame_negative, i)

                # Store current conditioning for next frame's blending
                previous_positive = positive_conds[i]
                previous_negative = negative_conds[i]

                if will_diffuse:
                    # Sample this cadence keyframe.
                    result = self.sample_with_callback(
                        model, frame_seed, steps, cfg, sampler_name, scheduler,
                        frame_positive, frame_negative, latent_format,
                        denoise=feedback_denoise,
                        callback=progress.sampling_callback(i),
                    )
                    current_latent = result["samples"]
                else:
                    # Deforum-style cadence tween: motion only, no sampler call.
                    current_latent = blended_latent
                    print(f"  [Cadence] Motion-only frame (cadence={frame_cadence})")

                # Store on CPU to prevent GPU memory accumulation
                all_latents.append(current_latent.clone().cpu())
                progress.complete_frame(i, current_latent, was_diffused=will_diffuse)

                # Periodic memory cleanup
                self.cleanup_memory(i)

            # Stack all results and return (move back to GPU for output)
            device = input_latents.device
            print(f"\n[Finalizing] Stacking {len(all_latents)} frames...")
            final_latent = torch.cat([lat.to(device) for lat in all_latents], dim=0)

            # Clear the CPU list to free memory
            all_latents.clear()

            # Final cleanup
            self.cleanup_memory(batch_size, force=True)

            # Print final memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[Memory] Final: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            print(f"=== Batch processing complete: {batch_size} frames with feedback chain ===")
            progress.finish()

            return (
                {"samples": final_latent},
                {"samples": final_latent}
            )

        # === SINGLE LATENT MODE: Use feedback iteration loop ===
        current_latent = input_latents

        # === STEP 3: First Iteration with Full Denoise ===
        print(f"\n[Frame 0] Starting first iteration with denoise={denoise}")
        progress.begin_frame(0, will_diffuse=True)
        latent_format["samples"] = current_latent

        # Set noise_mask for frame 0 if mask provided
        iteration_mask = self.get_mask_for_iteration(mask, 0)
        if iteration_mask is not None:
            self.set_latent_noise_mask(latent_format, iteration_mask)
            print(f"[Frame 0] Mask applied - shape: {latent_format['noise_mask'].shape}, "
                  f"min: {latent_format['noise_mask'].min():.3f}, max: {latent_format['noise_mask'].max():.3f}")
        elif "noise_mask" in latent_format:
            del latent_format["noise_mask"]

        # Use first frame's conditioning
        frame_positive = self.get_conditioning_for_frame(positive_conds[0], 0)
        frame_negative = self.get_conditioning_for_frame(negative_conds[0], 0)
        result = self.sample_with_callback(
            model, seed, steps, cfg, sampler_name, scheduler,
            frame_positive, frame_negative, latent_format, denoise=denoise,
            callback=progress.sampling_callback(0),
        )

        current_latent = result["samples"]
        # Store on CPU to prevent GPU memory accumulation
        all_latents.append(current_latent.clone().cpu())

        # Store first frame as color reference
        previous_image = None  # For temporal smoothing
        if color_coherence != "None" and vae is not None:
            color_reference = self.latent_to_image(current_latent, vae)
            previous_image = color_reference.copy()  # Also use as first previous image
            print(f"[Frame 0] Stored as color reference ({color_coherence} mode)")

        progress.complete_frame(0, current_latent, was_diffused=True)

        # Initialize previous conditioning for blending
        previous_positive = positive_conds[0]
        previous_negative = negative_conds[0]

        # === STEP 4: Feedback Loop with Scheduled Motion & Conditioning ===
        for i in range(1, iterations):
            will_diffuse = self.should_diffuse_frame(i, frame_cadence)
            progress.begin_frame(i, will_diffuse)

            # Determine seed for this iteration
            if seed_variation == "fixed":
                iteration_seed = seed
            elif seed_variation == "increment":
                iteration_seed = seed + i
            else:  # random
                iteration_seed = random.randint(0, 0xffffffffffffffff)

            # Get scheduled values for this frame
            frame_angle = angle_values[i]
            frame_tx = tx_values[i]
            frame_ty = ty_values[i]
            frame_tz = tz_values[i]
            frame_rx = rx_values[i]
            frame_ry = ry_values[i]
            frame_rz = rz_values[i]
            frame_zoom = zoom_values[i]

            print(f"\n[Frame {i}] zoom={frame_zoom:.4f} | angle={frame_angle:.2f} deg | tx={frame_tx:.1f} ty={frame_ty:.1f} tz={frame_tz:.1f}")
            print(f"         denoise={feedback_denoise} | seed={iteration_seed}")

            # Apply the scheduled transformations to the previous output.
            zoomed_latent, _ = self.zoom_latent(current_latent, frame_zoom)

            # Only apply other transforms if they're non-zero (skip if static)
            if frame_angle != 0 or frame_tx != 0 or frame_ty != 0 or frame_tz != 0 or frame_rx != 0 or frame_ry != 0 or frame_rz != 0:
                zoomed_latent = self.apply_deforum_transform(
                    zoomed_latent,
                    angle=frame_angle,
                    translation_x=frame_tx,
                    translation_y=frame_ty,
                    translation_z=frame_tz,
                    rotation_3d_x=frame_rx,
                    rotation_3d_y=frame_ry,
                    rotation_3d_z=frame_rz
                )

            # Get user-provided mask for this iteration
            iteration_mask = self.get_mask_for_iteration(mask, i)

            # CRITICAL: Apply color coherence + enhancements BEFORE generation
            if color_coherence != "None" and vae is not None and color_reference is not None:
                try:
                    print(f"  [1/7] Decoding latent to image...", end=" ", flush=True)
                    # Decode zoomed latent to image
                    current_image = self.latent_to_image(zoomed_latent, vae)
                    print(f"OK ({current_image.shape})", flush=True)

                    print(f"  [2/7] Matching colors ({color_coherence}, strength={color_coherence_strength:.2f})...", end=" ", flush=True)
                    # Match colors to reference frame with strength control
                    matched_image = self.match_color_histogram(current_image, color_reference, color_coherence, color_coherence_strength)
                    # Free intermediate - current_image no longer needed
                    del current_image
                    print(f"OK", flush=True)

                    # Apply contrast boost to prevent washed-out colors
                    if contrast_boost != 1.0:
                        print(f"  [3/7] Applying contrast boost ({contrast_boost})...", end=" ", flush=True)
                        matched_image = self.apply_contrast(matched_image, contrast_boost)
                        print(f"OK", flush=True)

                    # Apply sharpening to recover detail (CRITICAL for low denoise)
                    if sharpen_amount > 0:
                        print(f"  [4/7] Applying sharpening ({sharpen_amount})...", end=" ", flush=True)
                        matched_image = self.apply_sharpening(matched_image, sharpen_amount)
                        print(f"OK", flush=True)

                    # ADD NOISE IN PIXEL SPACE (after color matching, so it doesn't get removed!)
                    # This is critical for adding detail to flat/solid color areas
                    if noise_amount > 0:
                        print(f"  [5/7] Adding {noise_type} noise ({noise_amount})...", flush=True)
                        matched_image = self.apply_noise_pixel(matched_image, noise_amount, noise_type, iteration_seed)
                        print(f"  [5/7] Noise added OK", flush=True)

                    # Apply temporal smoothing to reduce flicker (blend with previous frame)
                    if temporal_smoothing > 0 and previous_image is not None:
                        print(f"  [6/7] Applying temporal smoothing ({temporal_smoothing:.2f})...", end=" ", flush=True)
                        matched_image = self.apply_temporal_smoothing(matched_image, previous_image, temporal_smoothing)
                        print(f"OK", flush=True)

                    print(f"  [7/7] Encoding image to latent...", end=" ", flush=True)
                    # Encode back to latent
                    matched_latent = self.image_to_latent(matched_image, vae, zoomed_latent.shape)
                    previous_image = matched_image.copy()
                    # Free intermediate - matched_image no longer needed
                    del matched_image
                    print(f"OK ({matched_latent.shape})", flush=True)

                    zoomed_latent = matched_latent
                    # Free reference to matched_latent (zoomed_latent now holds it)
                    del matched_latent
                    print("  [OK] All enhancements applied successfully", flush=True)
                except Exception as e:
                    import traceback
                    print(f"\n  [ERROR] Color/enhancement pipeline failed: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    print(f"  Continuing without color correction for this frame...", flush=True)
            elif noise_amount > 0:
                # If no color coherence but noise requested, add in latent space as fallback
                zoomed_latent = self.apply_noise(zoomed_latent, noise_amount, iteration_seed)

            # Prepare for next sampling
            latent_format["samples"] = zoomed_latent

            # Set noise_mask for KSampler - only diffuse in masked regions
            # Uses same format as SetLatentNoiseMask node
            if iteration_mask is not None:
                self.set_latent_noise_mask(latent_format, iteration_mask)
                print(f"  [Mask] Applied - shape: {latent_format['noise_mask'].shape}, "
                      f"min: {latent_format['noise_mask'].min():.3f}, max: {latent_format['noise_mask'].max():.3f}")
            elif "noise_mask" in latent_format:
                del latent_format["noise_mask"]

            # Apply conditioning blending for smoother prompt transitions
            frame_positive = positive_conds[i]
            frame_negative = negative_conds[i]
            if cond_blend_strength > 0:
                frame_positive = self.blend_conditioning(positive_conds[i], previous_positive, cond_blend_strength)
                frame_negative = self.blend_conditioning(negative_conds[i], previous_negative, cond_blend_strength)

            frame_positive = self.get_conditioning_for_frame(frame_positive, i)
            frame_negative = self.get_conditioning_for_frame(frame_negative, i)

            # Store current conditioning for next frame's blending
            previous_positive = positive_conds[i]
            previous_negative = negative_conds[i]

            if will_diffuse:
                # Sample this cadence keyframe with per-frame conditioning.
                result = self.sample_with_callback(
                    model, iteration_seed, steps, cfg, sampler_name, scheduler,
                    frame_positive, frame_negative, latent_format,
                    denoise=feedback_denoise,
                    callback=progress.sampling_callback(i),
                )
                current_latent = result["samples"]
            else:
                # Deforum-style cadence tween: retain the transformed frame.
                current_latent = zoomed_latent
                print(f"  [Cadence] Motion-only frame (cadence={frame_cadence})")

            # Store on CPU to prevent GPU memory accumulation
            all_latents.append(current_latent.clone().cpu())
            progress.complete_frame(i, current_latent, was_diffused=will_diffuse)

            # Periodic memory cleanup to prevent OOM on long runs
            self.cleanup_memory(i)

            # Print memory stats every 50 frames for debugging
            if i % 50 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  [Memory] Frame {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Stack all latents for batch output (move back to GPU for output)
        device = input_latents.device
        print(f"\n[Finalizing] Stacking {len(all_latents)} frames...")
        all_latents_stacked = torch.cat([lat.to(device) for lat in all_latents], dim=0)

        # Clear the CPU list to free memory
        all_latents.clear()

        # Final cleanup
        self.cleanup_memory(iterations, force=True)

        # Print final memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Memory] Final: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        print(f"=== SyntaxFeedbackSampler complete: {iterations} frames generated ===\n")
        progress.finish()

        # Return final latent and all latents as batch
        final_output = {"samples": current_latent}
        all_output = {"samples": all_latents_stacked}

        return (final_output, all_output)
