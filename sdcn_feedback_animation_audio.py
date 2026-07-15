"""
SD-CN Feedback Animation (Audio Reactive)
==========================================

Audio-reactive variant of SD-CN Feedback Animation. FFT envelope data from
Fill-Nodes (FL_Audio_Reactive_Envelope) modulates per-frame motion parameters.

Frequency-band mapping:
  - Kick + Bass (low freq)    -> zoom
  - Snare (mid freq)          -> angle/rotation
  - Hihat + Vocals (high freq) -> translate_x / translate_y

Base motion values always apply; audio modulates on top (center +/- style).
Without audio envelopes connected, behaves identically to the base node.
"""

from collections import deque

import torch
import numpy as np
import cv2

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management as mm

from .audio_envelope_handler import AudioEnvelopeHandler
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

class SDCNFeedbackAnimationAudio:
    """
    Audio-reactive pixel-space affine-feedback animation.

    Identical to SDCNFeedbackAnimation but with per-frame audio modulation:
      - Kick + Bass (low freq)     -> zoom
      - Snare (mid freq)           -> angle/rotation
      - Hihat + Vocals (high freq) -> translate_x / translate_y

    Base motion values always apply; audio modulates on top.
    Without audio envelopes connected, behaves identically to the base node.
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
                "reactivity": (["low", "medium", "high", "extreme"], {"default": "medium"}),
                "speed_ramp": ("BOOLEAN", {"default": True,
                               "tooltip": "Audio floodgate: onset energy directly gates per-frame denoise strength and motion intensity. Silence = nearly frozen, beat hits = full transformation. Same principle as StyleGAN latent speed modulation."}),
                "gate_floor": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                "tooltip": "Minimum denoise during silence (0.0 = fully frozen, 0.1 = subtle drift)"}),
                "gate_strength": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1,
                                   "tooltip": "Gate contrast — higher = more binary open/closed, lower = gradual ramp"}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Direct audio input — FFT analysis extracts beat/frequency data internally. Just connect a LoadAudio node. Alternative to wiring individual envelope strings."
                }),
                "flower_model": ("FLOWER_MODEL",),
                "flower_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                 "tooltip": "How much FloweR's predicted next frame influences the init image. 0=affine warp with flow refinement, 1=pure FloweR prediction. Requires flower_model."}),
                "control_net": ("CONTROL_NET",),
                "control_hints": ("IMAGE",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Starting frame offset for audio envelope sync"
                }),
                **AudioEnvelopeHandler.get_standard_inputs(),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "SyntaxNodes/Animation"

    @staticmethod
    def _analyze_audio_fft(audio_data, num_frames):
        """Analyze ComfyUI AUDIO input via spectral flux into per-frame
        band energies. This replaces the old raw-FFT + onset-detection
        pipeline with a single-pass approach that is both faster and more
        accurate for beat timing.

        Key improvements over the old approach:
          - Spectral flux (half-wave rectified bin diffs) captures transients
            directly — no second-derivative problem from frame-differencing
          - 35ms beat anticipation (standard in VJ tools) — visuals pre-fire
            so they appear perceptually synced
          - 512-sample analysis window + zero-pad to 1024 — tighter transients
          - Skips the noise gate and cooldown entirely

        Returns three lists of floats (one value per frame), normalized 0-1.
        Values are already onset-like — skip _detect_onsets for this path.
        """
        waveform = audio_data["waveform"]   # [batch, channels, samples]
        sr = audio_data["sample_rate"]

        # Convert to mono numpy float32
        wav = waveform[0].mean(dim=0).cpu().numpy().astype(np.float32)

        # Auto fps from audio duration
        audio_duration = len(wav) / sr
        auto_fps = num_frames / max(audio_duration, 0.001)
        samples_per_frame = sr / auto_fps

        # 512-sample analysis window, zero-padded to 1024 for frequency resolution.
        # 512 samples = ~11.6ms at 44.1kHz — tight transient resolution.
        analysis_len = 512
        n_fft = 1024
        window = np.hanning(analysis_len)

        # Frequency bin edges (based on zero-padded FFT size)
        freq_per_bin = sr / n_fft
        low_cut = max(1, int(250 / freq_per_bin))
        mid_cut = max(low_cut + 1, int(2000 / freq_per_bin))

        low_band = np.zeros(num_frames)
        mid_band = np.zeros(num_frames)
        high_band = np.zeros(num_frames)

        prev_spectrum = None

        for i in range(num_frames):
            # Center FFT window on frame midpoint
            center = int((i + 0.5) * samples_per_frame)
            start = max(0, center - analysis_len // 2)
            if start >= len(wav):
                break

            # Read audio chunk and apply window
            chunk = np.zeros(analysis_len, dtype=np.float32)
            avail = min(analysis_len, len(wav) - start)
            chunk[:avail] = wav[start:start + avail]
            chunk *= window

            # Zero-pad to n_fft for frequency resolution, then FFT
            padded = np.zeros(n_fft, dtype=np.float32)
            padded[:analysis_len] = chunk
            spectrum = np.abs(np.fft.rfft(padded))

            if prev_spectrum is not None:
                # Half-wave rectified spectral flux — only RISING bins count.
                # This IS onset detection: sustained energy produces zero flux,
                # transients produce sharp spikes. No noise gate needed.
                flux = np.maximum(0, spectrum - prev_spectrum)

                low_band[i] = np.sqrt(np.mean(flux[1:low_cut] ** 2))
                mid_band[i] = np.sqrt(np.mean(flux[low_cut:mid_cut] ** 2))
                if len(flux) > mid_cut:
                    high_band[i] = np.sqrt(np.mean(flux[mid_cut:] ** 2))

            prev_spectrum = spectrum.copy()

        # Normalize each band to 0-1
        for band in (low_band, mid_band, high_band):
            peak = band.max()
            if peak > 1e-6:
                band /= peak

        print(f"  [AUDIO FFT] {audio_duration:.1f}s @ {sr}Hz → "
              f"{num_frames} frames (auto {auto_fps:.1f}fps)")
        print(f"    Spectral flux, centered FFT")
        print(f"    low  [{low_band.min():.3f}..{low_band.max():.3f}] "
              f"mid [{mid_band.min():.3f}..{mid_band.max():.3f}] "
              f"high [{high_band.min():.3f}..{high_band.max():.3f}]")

        return low_band.tolist(), mid_band.tolist(), high_band.tolist()

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
                 border_mode, warp_strength, reactivity,
                 occlusion_inpaint, occlusion_dilate,
                 color_coherence, color_anchor_every,
                 noise_amount, sharpen_amount, loop_frames,
                 speed_ramp, gate_floor, gate_strength,
                 audio=None,
                 flower_model=None, flower_blend=0.5,
                 control_net=None, control_hints=None,
                 cn_strength=1.0, frame_index=0,
                 kick_envelope="", snare_envelope="", hihat_envelope="",
                 bass_envelope="", drums_envelope="", vocals_envelope="",
                 other_envelope="",
                 envelope_intensity=1.0, envelope_mode="add",
                 kick_weight=1.0, snare_weight=0.5, hihat_weight=0.3,
                 bass_weight=0.7, vocals_weight=0.5):

        device = mm.get_torch_device()
        rng = np.random.default_rng(seed)

        # ==================================================================
        # DIAGNOSTIC: confirm new code is running
        # ==================================================================
        print(f"\n[SDCNFeedbackAudio] === GENERATE CALLED ===")
        print(f"  speed_ramp={speed_ramp}  gate_floor={gate_floor}  gate_strength={gate_strength}")
        print(f"  processing_strength={processing_strength}  num_frames={num_frames}")
        envelope_check = {
            'kick': bool(kick_envelope and kick_envelope.strip() not in ("", "{}", "null")),
            'snare': bool(snare_envelope and snare_envelope.strip() not in ("", "{}", "null")),
            'hihat': bool(hihat_envelope and hihat_envelope.strip() not in ("", "{}", "null")),
            'bass': bool(bass_envelope and bass_envelope.strip() not in ("", "{}", "null")),
            'drums': bool(drums_envelope and drums_envelope.strip() not in ("", "{}", "null")),
            'vocals': bool(vocals_envelope and vocals_envelope.strip() not in ("", "{}", "null")),
            'other': bool(other_envelope and other_envelope.strip() not in ("", "{}", "null")),
        }
        connected = [k for k, v in envelope_check.items() if v]
        has_direct_audio = audio is not None
        print(f"  Envelopes connected: {connected if connected else 'NONE'}")
        print(f"  Direct AUDIO input: {'YES' if has_direct_audio else 'no'}")

        # ==================================================================
        # PRE-COMPUTE MOTION SCHEDULE FROM AUDIO
        # ==================================================================
        # Build the entire timeline before generation starts. Every frame
        # gets locked-in zoom/angle/translate values derived from the audio.
        # This guarantees beat alignment and lets us log the schedule.

        zoom_schedule = [zoom] * num_frames
        angle_schedule = [angle] * num_frames
        tx_schedule = [translate_x] * num_frames
        ty_schedule = [translate_y] * num_frames

        envelope_map = {
            'kick': kick_envelope, 'snare': snare_envelope,
            'hihat': hihat_envelope, 'bass': bass_envelope,
            'drums': drums_envelope, 'vocals': vocals_envelope,
            'other': other_envelope,
        }
        has_envelope_audio = any(
            env and env.strip() not in ("", "{}", "null")
            for env in envelope_map.values()
        )
        has_audio = has_direct_audio or has_envelope_audio
        print(f"  has_audio={has_audio}")

        if has_audio:
            # --- Step 1: Parse envelopes (or FFT from direct audio) ---
            stem_data = {}
            frame_scale = 1.0

            # Parse any connected envelope strings first
            for stem_name, env_str in envelope_map.items():
                if env_str and env_str.strip() not in ("", "{}", "null"):
                    env_data = AudioEnvelopeHandler.parse_envelope_json(env_str)
                    vals = env_data.get("envelope", [])
                    if vals:
                        stem_data[stem_name] = vals
                    if frame_scale == 1.0 and env_data["total_frames"] > 0:
                        frame_scale = env_data["total_frames"] / max(1, num_frames)

            # If direct audio connected and no envelope data, run FFT analysis.
            # Spectral flux values are already onset-like — skip _detect_onsets.
            skip_onset_detection = False
            if has_direct_audio and not stem_data:
                low_vals, mid_vals, high_vals = self._analyze_audio_fft(audio, num_frames)
                stem_data['kick'] = low_vals    # low band → kick slot
                stem_data['snare'] = mid_vals   # mid band → snare slot
                stem_data['hihat'] = high_vals  # high band → hihat slot
                frame_scale = 1.0
                skip_onset_detection = True  # spectral flux IS onset detection
                print(f"  Using direct AUDIO → spectral flux (skip onset detection)")

            # --- Step 2: Onset/transient detection ---
            # Instead of using raw envelope levels (which can be sustained
            # near-max the whole track), detect RISING EDGES — the actual
            # hits where the signal jumps up. This is what makes a kick
            # feel like a kick instead of sustained bass rumble.
            #
            # onset[i] = max(0, envelope[i] - envelope[i-1])
            # Then normalize onsets so the biggest hit = 1.0.

            _DECAY_RATE = 0.25  # per-frame decay after onset (balance: snappy but not choppy)

            # Reactivity presets — how much motion a peak beat produces.
            # Must be big enough to survive KSampler absorbing the warp.
            _PRESETS = {
                "low":     {"zoom": 0.12, "angle":  8.0, "translate": 20.0, "mult": 3.0},
                "medium":  {"zoom": 0.50, "angle": 30.0, "translate": 80.0, "mult": 8.0},
                "high":    {"zoom": 1.00, "angle": 60.0, "translate": 160.0, "mult": 16.0},
                "extreme": {"zoom": 2.00, "angle": 120.0, "translate": 320.0, "mult": 32.0},
            }
            preset = _PRESETS.get(reactivity, _PRESETS["medium"])
            _ZOOM_MAX = preset["zoom"]
            _ANGLE_MAX = preset["angle"]
            _TRANSLATE_MAX = preset["translate"]
            _BASE_MULT = preset["mult"]

            _MIN_BEAT_GAP = 1  # minimum frames between beats (1 = allow rapid-fire)

            def _detect_onsets(envelope_vals, frame_scale, frame_index, num_frames):
                """Detect transients with noise gate and cooldown.

                1. Detect rising edges (onset = curr - prev)
                2. Normalize so max = 1.0
                3. Noise gate: zero out onsets below adaptive threshold
                4. Cooldown: suppress beats too close together
                5. Re-normalize survivors to 0-1
                """
                onsets = np.zeros(num_frames)
                for i in range(num_frames):
                    env_i = int(i * frame_scale) + frame_index
                    env_prev = int(max(0, i - 1) * frame_scale) + frame_index

                    if env_i >= len(envelope_vals) or env_prev >= len(envelope_vals):
                        continue

                    curr = float(envelope_vals[min(env_i, len(envelope_vals) - 1)])
                    prev = float(envelope_vals[min(env_prev, len(envelope_vals) - 1)])

                    onset = max(0.0, curr - prev)
                    onsets[i] = onset

                # Normalize so largest onset = 1.0
                raw_peak = np.max(onsets)
                if raw_peak > 0.001:
                    onsets = onsets / raw_peak

                # --- Noise gate: adaptive threshold ---
                # Keep only onsets above the 25th percentile of non-zero values.
                # (Was median/50th — that killed half of all real beats.)
                nonzero = onsets[onsets > 0.01]
                if len(nonzero) > 2:
                    gate_threshold = np.percentile(nonzero, 25)
                    onsets[onsets < gate_threshold] = 0.0

                # --- Cooldown: suppress rapid-fire triggers ---
                last_beat = -_MIN_BEAT_GAP
                for i in range(num_frames):
                    if onsets[i] > 0:
                        if (i - last_beat) < _MIN_BEAT_GAP:
                            onsets[i] = 0.0  # too close, suppress
                        else:
                            last_beat = i

                # Re-normalize survivors
                peak = np.max(onsets)
                if peak > 0.001:
                    onsets = onsets / peak

                return onsets, raw_peak

            # Detect onsets for each stem (skip for FFT spectral flux path)
            stem_onsets = {}
            stem_onset_peaks = {}
            if skip_onset_detection:
                # Spectral flux values are already onset-like — just normalize
                for stem_name, vals in stem_data.items():
                    arr = np.array(vals[:num_frames], dtype=np.float64)
                    if len(arr) < num_frames:
                        arr = np.pad(arr, (0, num_frames - len(arr)))
                    peak = arr.max()
                    if peak > 0.001:
                        arr = arr / peak
                    stem_onsets[stem_name] = arr
                    stem_onset_peaks[stem_name] = peak
                print(f"  Onset detection: SKIPPED (spectral flux already onset-like)")
            else:
                for stem_name, vals in stem_data.items():
                    onsets, peak = _detect_onsets(vals, frame_scale, frame_index, num_frames)
                    stem_onsets[stem_name] = onsets
                    stem_onset_peaks[stem_name] = peak

            # --- Step 3: Build frequency bands from onsets ---
            raw_low_arr = np.zeros(num_frames)
            raw_mid_arr = np.zeros(num_frames)
            raw_high_arr = np.zeros(num_frames)

            for i in range(num_frames):
                kick_o = stem_onsets.get('kick', np.zeros(num_frames))[i]
                bass_o = stem_onsets.get('bass', np.zeros(num_frames))[i]
                snare_o = stem_onsets.get('snare', np.zeros(num_frames))[i]
                hihat_o = stem_onsets.get('hihat', np.zeros(num_frames))[i]
                vocals_o = stem_onsets.get('vocals', np.zeros(num_frames))[i]

                raw_low_arr[i] = min(1.0, (kick_o * kick_weight + bass_o * bass_weight))
                raw_mid_arr[i] = min(1.0, snare_o * snare_weight)
                raw_high_arr[i] = min(1.0, (hihat_o * hihat_weight + vocals_o * vocals_weight))

            # Apply envelope_intensity as a power curve for sensitivity:
            # >1 = more sensitive (quiet onsets register more)
            # <1 = less sensitive (only strongest hits register)
            # =1 = linear
            if envelope_intensity != 1.0:
                power = 1.0 / max(0.1, envelope_intensity)  # invert: high intensity = low power = more sensitive
                raw_low_arr = np.power(raw_low_arr, power)
                raw_mid_arr = np.power(raw_mid_arr, power)
                raw_high_arr = np.power(raw_high_arr, power)

            # --- Step 4: Apply sustain/decay ---
            low_arr = np.zeros(num_frames)
            mid_arr = np.zeros(num_frames)
            high_arr = np.zeros(num_frames)

            for i in range(num_frames):
                if i == 0:
                    low_arr[i] = raw_low_arr[i]
                    mid_arr[i] = raw_mid_arr[i]
                    high_arr[i] = raw_high_arr[i]
                else:
                    low_arr[i] = max(raw_low_arr[i], low_arr[i - 1] * _DECAY_RATE)
                    mid_arr[i] = max(raw_mid_arr[i], mid_arr[i - 1] * _DECAY_RATE)
                    high_arr[i] = max(raw_high_arr[i], high_arr[i - 1] * _DECAY_RATE)

            # --- Gate energy for floodgate mode ---
            # Two layers, like the GAN hybrid style:
            #   1. PUNCH: sharp onset transients (fast decay = snappy)
            #   2. FLOW:  slow-decay sustain (keeps movement between beats)
            # Without the flow layer, silence → beat is a hard "cut."
            # With it, energy lingers between hits = smooth transitions.

            # Layer 1: Onset punch — GAN-matched power curves
            bass_amp = np.power(low_arr, 1.5) * 3.0    # GAN: (bass ** 1.5) * 3.0
            mid_amp = np.power(mid_arr, 1.3) * 2.0     # GAN: (mids ** 1.3) * 2.0
            high_amp = np.power(high_arr, 1.2) * 2.5   # GAN: (onset ** 1.2) * 2.5
            # Clamp to [0,1] — unclamped values (up to 2.5) extend the sustain
            # tail by ~193ms, making beats feel sluggish.
            onset_punch = np.clip(
                bass_amp * 0.5 + mid_amp * 0.35 + high_amp * 0.15, 0.0, 1.0)

            # Layer 2: Sustain flow — same signal but slower decay
            # Keeps the gate partially open between beats for smooth transitions.
            # 0.70 = drops to 17% after 5 frames (was 0.85 = 44%, too sluggish)
            _SUSTAIN_DECAY = 0.70
            gate_sustain = np.zeros(num_frames)
            for i in range(num_frames):
                if i == 0:
                    gate_sustain[i] = onset_punch[i]
                else:
                    gate_sustain[i] = max(onset_punch[i], gate_sustain[i - 1] * _SUSTAIN_DECAY)

            # Blend: punch for beat sync, sustain for flow
            gate_energy_arr = onset_punch * 0.6 + gate_sustain * 0.4

            nonzero_gate = np.sum(gate_energy_arr > 0.01)
            print(f"  [GATE DIAG] gate_energy (punch+flow): "
                  f"min={gate_energy_arr.min():.4f} "
                  f"max={gate_energy_arr.max():.4f} "
                  f"nonzero={nonzero_gate}/{num_frames} "
                  f"mean={gate_energy_arr.mean():.4f}")
            if gate_energy_arr.max() < 0.01:
                print(f"  [GATE DIAG] WARNING: gate_energy_arr is ALL ZEROS — "
                      f"onset detection found no beats! Floodgate will keep everything at floor.")

            # --- Step 5: Build final motion schedule ---
            # TRAVEL MODE (speed_ramp + defaults): continuous zoom journey
            # with beat-reactive angle/translation. The zoom IS the travel —
            # always moving forward. Beats are turbulence.
            #
            # Band mapping (same as GAN script):
            #   Bass/kick  → zoom speed (the engine)
            #   Snare/mid  → angle/rotation (camera tilting)
            #   Hihat/high → translation (camera panning)
            travel_mode = speed_ramp and zoom == 0 and angle == 0

            beat_frames = []
            for i in range(num_frames):
                ge_i = min(gate_energy_arr[i], 1.0)

                # --- ZOOM: always traveling, beats accelerate ---
                if zoom != 0:
                    zoom_schedule[i] = zoom * (1.0 + low_arr[i] * _BASE_MULT)
                elif speed_ramp:
                    # Base travel (always zooming) + beat boost
                    base_travel = 0.02    # constant zoom-in (the journey)
                    beat_boost = ge_i * 0.06  # up to 6% extra on peaks
                    zoom_schedule[i] = base_travel + beat_boost
                else:
                    zoom_schedule[i] = low_arr[i] * _ZOOM_MAX

                # --- ANGLE: snare/mid drives camera tilt on beats ---
                if angle != 0:
                    angle_schedule[i] = angle * (1.0 + mid_arr[i] * _BASE_MULT)
                elif travel_mode:
                    # Slow sine oscillation for direction variety
                    tilt_dir = np.sin(i * 0.12)
                    angle_schedule[i] = mid_arr[i] * 1.8 * tilt_dir
                else:
                    angle_schedule[i] = mid_arr[i] * _ANGLE_MAX

                # --- TRANSLATE: hihat/high drives camera pan on beats ---
                if translate_x != 0:
                    tx_schedule[i] = translate_x * (1.0 + high_arr[i] * _BASE_MULT)
                elif travel_mode:
                    pan_dir_x = np.sin(i * 0.08)
                    tx_schedule[i] = high_arr[i] * 4.0 * pan_dir_x
                else:
                    tx_schedule[i] = high_arr[i] * _TRANSLATE_MAX

                if translate_y != 0:
                    ty_schedule[i] = translate_y * (1.0 + high_arr[i] * _BASE_MULT)
                elif travel_mode:
                    pan_dir_y = np.cos(i * 0.08)
                    ty_schedule[i] = high_arr[i] * 3.0 * pan_dir_y
                else:
                    ty_schedule[i] = high_arr[i] * _TRANSLATE_MAX

                # Beat = frame where any onset > 0.3
                if low_arr[i] > 0.3 or mid_arr[i] > 0.3 or high_arr[i] > 0.3:
                    beat_frames.append(i)

            # --- Log the schedule ---
            zoom_min = min(zoom_schedule)
            zoom_max = max(zoom_schedule)
            angle_min = min(angle_schedule)
            angle_max = max(angle_schedule)
            tx_max = max(abs(v) for v in tx_schedule)

            # Count frames with zero motion for contrast stats
            silent_frames = sum(1 for i in range(num_frames)
                               if low_arr[i] < 0.01 and mid_arr[i] < 0.01 and high_arr[i] < 0.01)

            print(f"\n{'='*70}")
            print(f"[SDCNFeedbackAudio] MOTION SCHEDULE (ONSET DETECTION)")
            print(f"  Reactivity: {reactivity.upper()} "
                  f"(zoom_max={_ZOOM_MAX}, angle_max={_ANGLE_MAX}, "
                  f"translate_max={_TRANSLATE_MAX}, mult={_BASE_MULT})")
            print(f"{'='*70}")
            print(f"  Stems: {list(stem_data.keys())}")
            print(f"  Onset peaks: {{{', '.join(f'{k}={v:.4f}' for k, v in stem_onset_peaks.items())}}}")
            print(f"  Frame scale: {frame_scale:.2f}x "
                  f"({num_frames} video -> {int(num_frames * frame_scale)} envelope frames)")
            print(f"  envelope_intensity: {envelope_intensity} "
                  f"(sensitivity curve power={1.0/max(0.1, envelope_intensity):.2f})")
            print(f"  Beat frames: {len(beat_frames)} / {num_frames} "
                  f"({len(beat_frames)/num_frames*100:.0f}%)")
            print(f"  Silent frames: {silent_frames} / {num_frames} "
                  f"({silent_frames/num_frames*100:.0f}%)")
            print(f"  Zoom range:  [{zoom_min:.5f} ... {zoom_max:.5f}]")
            print(f"  Angle range: [{angle_min:.2f} ... {angle_max:.2f}] deg")
            print(f"  Max translate: {tx_max:.1f} px")
            print(f"")
            # Print first 30 frames of schedule
            show_n = min(30, num_frames)
            print(f"  Frame-by-frame (first {show_n}):")
            for i in range(show_n):
                marker = " <<BEAT" if i in beat_frames else ""
                print(f"    [{i:3d}] zoom={zoom_schedule[i]:+.5f} "
                      f"angle={angle_schedule[i]:+.2f} "
                      f"tx={tx_schedule[i]:+.1f} ty={ty_schedule[i]:+.1f}"
                      f"{marker}")
            if num_frames > show_n:
                print(f"    ... ({num_frames - show_n} more frames)")

            # --- Floodgate preview ---
            if speed_ramp:
                print(f"")
                print(f"  AUDIO FLOODGATE: ON")
                print(f"    gate_floor={gate_floor:.3f}  gate_strength={gate_strength:.1f}")
                print(f"    Denoise range: [{gate_floor:.3f} .. {processing_strength:.3f}]")
                print(f"    Gate energy: [{gate_energy_arr.min():.3f} .. {gate_energy_arr.max():.3f}]")
                frozen = np.sum(gate_energy_arr < 0.05)
                full_open = np.sum(np.clip(gate_energy_arr * gate_strength, 0, 1) > 0.9)
                print(f"    Frozen frames (energy<0.05): {frozen}/{num_frames} ({frozen/num_frames*100:.0f}%)")
                print(f"    Full-blast frames (gate>0.9): {full_open}/{num_frames} ({full_open/num_frames*100:.0f}%)")
                print(f"    Per-frame gate (first {show_n}):")
                for i in range(show_n):
                    ge = np.clip(gate_energy_arr[i] * gate_strength, 0, 1)
                    fd = gate_floor + ge * (processing_strength - gate_floor)
                    mg = gate_floor + ge * (1.0 - gate_floor)
                    marker = " <<BEAT" if i in beat_frames else ""
                    print(f"      [{i:3d}] energy={gate_energy_arr[i]:.3f} "
                          f"denoise={fd:.3f} motion={mg:.3f}{marker}")
                if num_frames > show_n:
                    print(f"      ... ({num_frames - show_n} more)")

            print(f"{'='*70}\n")

        # (Dead code from pre-floodgate era removed — absorption compensation,
        #  per-frame sustain state, fallback multipliers were all vestigial.)

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

            # --- Audio floodgate: gate denoise + motion by onset energy ---
            # Like StyleGAN speed modulation: audio controls HOW MUCH change
            # happens per frame, not what the change looks like.
            #   Silence → gate ≈ floor → nearly frozen
            #   Beat hit → gate ≈ 1.0 → full transformation
            if speed_ramp and has_audio:
                ge = np.clip(gate_energy_arr[i] * gate_strength, 0.0, 1.0)
                # In travel mode the continuous zoom NEEDS denoise to fill
                # new content. Floor must be high enough to regenerate, and
                # beats should hit full processing_strength for punch.
                travel_floor = max(gate_floor, 0.15) if (zoom == 0) else gate_floor
                frame_denoise = travel_floor + ge * (processing_strength - travel_floor)
                motion_gate = gate_floor + ge * (1.0 - gate_floor)
            else:
                frame_denoise = processing_strength
                motion_gate = 1.0

            # Read pre-computed motion from schedule.
            # In travel mode, motion is already audio-baked — no double-gating.
            # Otherwise, scale by motion_gate.
            if speed_ramp and has_audio:
                # Travel: zoom/angle/translate already encode audio energy
                frame_zoom = zoom_schedule[i]
                frame_angle = angle_schedule[i]
                frame_tx = tx_schedule[i]
                frame_ty = ty_schedule[i]
            else:
                frame_zoom = zoom_schedule[i] * motion_gate
                frame_angle = angle_schedule[i] * motion_gate
                frame_tx = tx_schedule[i] * motion_gate
                frame_ty = ty_schedule[i] * motion_gate

            # Loop diagnostic (first 10 frames)
            if i <= 10 and speed_ramp and has_audio:
                print(f"  [LOOP {i:3d}] ge={ge:.3f} denoise={frame_denoise:.3f} "
                      f"zoom={frame_zoom:.4f} angle={frame_angle:+.2f} "
                      f"tx={frame_tx:+.1f}")

            # 1. Affine matrix for this frame
            M = _build_affine(frame_zoom, frame_tx, frame_ty, frame_angle, w, h)

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

            # 5. Polish — noise always active (feeds variation into KSampler)
            warped = _unsharp(warped, sharpen_amount)
            warped = _add_noise(warped, noise_amount, rng)

            # 6. Cadence: skip SD on non-cadence frames
            #    BUT force diffusion on strong beats even when cadence skips.
            beat_force = (has_audio and speed_ramp
                          and gate_energy_arr[i] > 0.55)
            if not beat_force and (i % diffusion_cadence) != 0:
                frames.append(warped)
                prev_pixel = warped
                frame_history.append(warped)
                pbar.update(1)
                continue

            # 7. FloweR integration (only on cadence/beat frames)
            occ_mask = None
            flower_init = None

            if flower_net is not None and len(frame_history) >= 4:
                # Compute effective flower_blend
                effective_blend = flower_blend

                # In travel mode, FloweR is a subtle texture enhancer — NOT
                # a motion driver. Its optical flow predictions fight the
                # audio-driven zoom, causing drift. Cap it low.
                if speed_ramp and has_audio:
                    effective_blend = min(flower_blend * 0.20, 0.12)

                # Taper during loop closure
                if loop_frames > 0:
                    frames_remaining = num_frames - 1 - i
                    if frames_remaining < loop_frames:
                        loop_progress = 1.0 - (frames_remaining / loop_frames)
                        effective_blend = effective_blend * (1.0 - loop_progress)

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

            # 9. Per-frame ControlNet — reactive strength
            #    Pulses with energy, loosens on kicks (image "breaks open")
            if (control_net is not None
                    and control_hints is not None
                    and control_hints.shape[0] > 0):
                idx = (i - 1) % control_hints.shape[0]
                hint = control_hints[idx:idx + 1].to(device)
                cn_strength_i = cn_strength
                if has_audio and speed_ramp:
                    cn_strength_i = cn_strength * (0.75 + 0.35 * ge)
                    cn_strength_i *= (1.0 - 0.25 * low_arr[i])  # loosen on kick
                pos_f, neg_f = _apply_cn_per_frame(
                    positive, negative, control_net, hint, cn_strength_i
                )
            else:
                pos_f, neg_f = positive, negative

            # 10. Beat-synced seed — same seed during quiet, new on major hits.
            #     Big visual decisions change with the music, not every frame.
            if has_audio and speed_ramp:
                if gate_energy_arr[i] > 0.75:
                    _beat_seed_counter = getattr(self, '_bsc', 0) + 1
                    self._bsc = _beat_seed_counter
                else:
                    _beat_seed_counter = getattr(self, '_bsc', 0)
                frame_seed = seed + _beat_seed_counter
            else:
                frame_seed = seed + i

            # 11. First pass -- denoise gated by audio energy
            latent = _sample(
                model, latent, pos_f, neg_f,
                frame_seed, steps, cfg, sampler_name, scheduler,
                frame_denoise, mask_np=None,
            )

            # 12. Occlusion refinement pass — reactive strength
            #     Kick/hihat boost refinement (more detail on transients)
            reactive_fix = fix_frame_strength
            if has_audio and speed_ramp:
                reactive_fix += 0.20 * low_arr[i] + 0.10 * high_arr[i]
                reactive_fix = np.clip(reactive_fix, 0.0, 0.75)
            if occ_mask is not None and reactive_fix > 0.001:
                refine_steps = max(8, steps // 2)
                latent = _sample(
                    model, latent, pos_f, neg_f,
                    frame_seed + 100000, refine_steps, cfg, sampler_name, scheduler,
                    reactive_fix, mask_np=occ_mask,
                )

            # 13. Decode to pixels
            decoded = vae.decode(latent["samples"])
            current_pixel = _tensor_to_np(decoded)

            # 14. Post-decode FX — tasteful finishing driven by audio
            if has_audio and speed_ramp:
                # Contrast pulse on kick (image "thumps")
                kick_e = low_arr[i]
                if kick_e > 0.1:
                    alpha = 1.0 + kick_e * 0.12
                    current_pixel = cv2.convertScaleAbs(
                        current_pixel, alpha=alpha, beta=int(kick_e * 8))

                # Bloom on highs (bright areas glow on hihat/cymbal)
                high_e = high_arr[i]
                if high_e > 0.15:
                    bright = cv2.GaussianBlur(current_pixel, (0, 0), 8, 8)
                    bloom_strength = high_e * 0.25
                    current_pixel = cv2.addWeighted(
                        current_pixel, 1.0,
                        bright, bloom_strength, 0)
                    current_pixel = np.clip(current_pixel, 0, 255).astype(np.uint8)

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

        # 14. No post-process — the in-loop floodgate handles everything.
        #     Like the GAN script: each frame is a clean generation, never
        #     blended or held. The denoise/motion gating per frame creates
        #     smooth speed variation: silence = slow drift, beat = dramatic pop.
        #     Post-process blending caused smearing; hold/snap caused choppiness.
        #     Clean per-frame generation avoids both.

        # Stack all frames as IMAGE batch
        out = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return (torch.from_numpy(out),)


# ============================================================================
# Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SDCNFeedbackAnimationAudio": SDCNFeedbackAnimationAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCNFeedbackAnimationAudio": "SD-CN Feedback Animation (Audio Reactive)",
}
