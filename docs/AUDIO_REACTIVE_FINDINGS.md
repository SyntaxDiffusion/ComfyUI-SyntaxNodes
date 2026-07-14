# Audio-Reactive Feedback Node — Complete Findings & Architecture

All research, experiments, and design decisions from the audio-reactive development sessions (May 2026).

---

## What Was Built

An audio-reactive variant of SD-CN Feedback Animation that accepts raw AUDIO input, runs FFT analysis, and modulates the generation loop per-frame. The goal: a VJ-style audio-reactive travel through latent space.

## Reference Implementation

`G:\gan\windows_gan_interpolate.py` — StyleGAN2 audio-reactive interpolation script. The "hybrid" style (lines 533-628) is the gold standard for how audio should drive visual generation:

- **Speed modulation**: `speed_mult = 0.3 + beat_intensity * sensitivity * 4.0` (range 0.3x to 15x)
- **Power curve amplification**: `bass_amp = (bass ** 1.5) * 3.0`, `onset_amp = (onset ** 1.2) * 2.5`
- **Weighted mix**: `beat_intensity = bass * 0.5 + onset * 0.35 + energy * 0.15`
- **Continuous movement**: Always moving through latent space, never frozen. Audio controls SPEED, not on/off.
- **Texture layers**: Mids/highs add subtle directional noise on top of the base trajectory.

---

## Audio Engine: What Works

### Spectral Flux (replaces raw FFT + onset detection)

The final FFT approach that worked best:

```python
analysis_len = 512          # ~11.6ms at 44.1kHz — tight transient resolution
n_fft = 1024                # zero-padded for frequency resolution
window = np.hanning(analysis_len)

# Center FFT window on frame midpoint
center = int((i + 0.5) * samples_per_frame)

# Half-wave rectified spectral flux — only RISING bins count
flux = np.maximum(0, spectrum - prev_spectrum)
low_band[i] = np.sqrt(np.mean(flux[1:low_cut] ** 2))
mid_band[i] = np.sqrt(np.mean(flux[low_cut:mid_cut] ** 2))
high_band[i] = np.sqrt(np.mean(flux[mid_cut:] ** 2))
```

**Why spectral flux over raw FFT + onset detection:**
- Raw FFT gives energy per frame. Onset detection (frame differencing) on top of that computes the SECOND DERIVATIVE — halves peak energy of 2-frame beats.
- Spectral flux does onset detection INSIDE the FFT: sustained energy = zero flux, transients = sharp spikes.
- No noise gate needed, no cooldown needed, no second-derivative problem.
- Skip `_detect_onsets()` entirely for the direct FFT path.

### Band Mapping

```
low  (20-250 Hz)   → kick / bass   → zoom (the travel engine)
mid  (250-2000 Hz) → snare / mids  → angle / rotation (camera tilting)
high (2000+ Hz)    → hihat / cyms  → translation (camera panning)
```

### Auto FPS

```python
audio_duration = len(wav) / sr
auto_fps = num_frames / max(audio_duration, 0.001)
samples_per_frame = sr / auto_fps
```

Entire audio maps exactly across all frames. No drift.

---

## Audio Engine: Timing Issues Found

| Issue | Impact | Fix |
|-------|--------|-----|
| FFT window 2048 samples = 46ms blur | Beats smeared across frames | Use 512 analysis + zero-pad to 1024 |
| Onset detection = second derivative of FFT | Halves 2-frame beat peaks | Replace with spectral flux |
| Median noise gate killed 50% of beats | Missing beats | Use 25th percentile (or skip for FFT path) |
| Cooldown `_MIN_BEAT_GAP=2` blocked fast hihats | Missing rapid hits | Reduce to 1 or remove |
| Unclamped power curves (up to 2.575) | Extended sustain tail by ~193ms | Clamp onset_punch to [0, 1] |
| Sustain decay 0.85 = 44% after 5 frames | Too sluggish, reduced beat contrast | Use 0.70 (17% after 5 frames) |
| `int()` truncation in envelope path | Up to 1-frame jitter | Use interpolation for envelope path |
| 35ms anticipation offset | Too aggressive, beats appeared early | Remove — centered FFT is sufficient |

---

## Floodgate Architecture: What We Learned

### The Concept
Audio energy gates per-frame denoise strength and motion intensity. Silence = low denoise (image barely changes), beats = high denoise (dramatic transformation). Same principle as StyleGAN latent speed modulation.

### What Worked
- **GAN-style power curve amplification** for gate energy
- **Two-layer gate**: onset punch (fast decay 0.25) + sustain flow (slow decay 0.70)
- **Denoise gating**: `frame_denoise = floor + ge * (processing_strength - floor)`
- **Motion gating**: separate from denoise, controls zoom/angle/translate

### What Didn't Work
- **Post-process frame blending** (addWeighted) → ghostly smearing between held and new frames
- **Binary hold/snap** → choppy slideshow effect
- **Post-process time remapping** → source frames too similar, skipping through them had no visual impact
- **70% denoise ceiling** → not enough visual change on beats, especially with continuous zoom
- **Double-gating motion** → zoom_schedule already audio-encoded, multiplying by motion_gate crushed it
- **Gating noise by motion_gate** → starved the KSampler of variation during silence
- **FloweR at full blend** → optical flow predictions fought the audio-driven zoom, causing drift

### CRITICAL OVERSIGHT: Latent Diffusion Was Effectively Disabled

**This was the root bug we missed for nearly a week.** The floodgate's `gate_floor` of 0.05 meant `frame_denoise = 0.05` during silence. While `_sample()` has an early return at `denoise <= 0.001`, the real issue was that 5% denoise produces virtually no visual change — the KSampler runs but does nothing meaningful. Combined with the 70% denoise ceiling (max 0.47 on beats), the node was barely running latent diffusion at all compared to the base node's constant `processing_strength=0.65`.

**Symptom:** The audio node ran MUCH FASTER than the base node. We interpreted this as the audio engine working. In reality, the KSampler was doing almost nothing — the speed was because there was no real diffusion happening.

**The base node runs KSampler at full processing_strength every cadence frame. That is correct.** Any audio-reactive version MUST preserve meaningful latent diffusion on every cadence frame. The denoise floor must be high enough to produce visible content generation (at least 0.15-0.25), and beats should reach full processing_strength.

**Lesson:** If the audio-reactive node is significantly faster than the base node at the same settings, something is wrong — the KSampler is being starved. Speed ≠ working. Latent diffusion is the whole point of the feedback loop.

### Key Insight
The feedback loop is NOT like a GAN interpolation. In the GAN, high speed = smooth slerp between waypoints (clean images). In the feedback node, high denoise = destructive noise injection (scene resets). Motion carries the visual punch, denoise should morph not replace.

---

## Travel Mode Design

The vision: continuous zoom journey through the image, beats accelerate the travel.

```python
# Always traveling — base zoom + beat boost
base_travel = 0.02    # constant zoom-in (the journey)
beat_boost = ge_i * 0.06  # up to 6% extra on peaks
zoom_schedule[i] = base_travel + beat_boost

# Snare/mid → angle (camera tilting on beats)
tilt_dir = np.sin(i * 0.12)  # slow oscillation for direction variety
angle_schedule[i] = mid_arr[i] * 1.8 * tilt_dir

# Hihat/high → translation (camera panning on beats)
tx_schedule[i] = high_arr[i] * 4.0 * np.sin(i * 0.08)
ty_schedule[i] = high_arr[i] * 3.0 * np.cos(i * 0.08)
```

**Critical**: When zoom is always on, denoise floor must be ≥0.15 so KSampler regenerates content to fill the zoomed areas. Otherwise the image degrades into blur.

---

## Specific Code Improvements (Validated)

### 1. Force diffusion on strong beats through cadence
```python
beat_force = has_audio and gate_energy_arr[i] > 0.55
if not beat_force and (i % diffusion_cadence) != 0:
    # skip SD
    continue
```

### 2. Reactive ControlNet strength
```python
cn_strength_i = cn_strength * (0.75 + 0.35 * ge)
cn_strength_i *= (1.0 - 0.25 * low_arr[i])  # loosen on kick
```
Image structure breathes — tighter during quiet, breaks open on drums.

### 3. Reactive refine strength
```python
reactive_fix = fix_frame_strength + 0.20 * low_arr[i] + 0.10 * high_arr[i]
reactive_fix = np.clip(reactive_fix, 0.0, 0.75)
```

### 4. Beat-synced seed
```python
if gate_energy_arr[i] > 0.75:
    current_seed += 1
frame_seed = seed + current_seed
```
Same seed during quiet (consistency), new seed on major hits (change with the music).

### 5. Post-decode FX
```python
# Contrast pulse on kick
alpha = 1.0 + kick_e * 0.12
current_pixel = cv2.convertScaleAbs(current_pixel, alpha=alpha, beta=int(kick_e * 8))

# Bloom on highs
bright = cv2.GaussianBlur(current_pixel, (0, 0), 8, 8)
current_pixel = cv2.addWeighted(current_pixel, 1.0, bright, high_e * 0.25, 0)
```

---

## Long-Term Architecture: Reaction Graph

### Musical Feature Layer
Instead of raw FFT bands, expose structured features:
```python
features = {
    "kick_hit": fast transient,
    "kick_body": short low-end decay,
    "snare_hit": mid transient,
    "hat_texture": high-frequency density,
    "rms_energy": overall loudness,
    "spectral_centroid": brightness,
    "section_energy": long-term intensity (verse vs chorus),
    "beat_phase": 0.0-1.0 within beat,
    "bar_phase": 0.0-1.0 within phrase,
}
```

### Per-Feature Envelopes
```python
kick_punch = envelope(raw_kick, attack=0.0, release=0.18)
bass_flow = envelope(raw_low_rms, attack=0.08, release=0.65)
hat_sparkle = envelope(raw_high_flux, attack=0.0, release=0.08)
section_drive = envelope(rms_energy, attack=0.5, release=2.0)
```

### Reaction Channels
```python
reaction = {
    "camera_zoom": kick_body,
    "camera_shake": kick_hit,
    "rotation_impulse": snare_hit,
    "micro_jitter": hat_texture,
    "denoise": section_drive + kick_hit,
    "detail_injection": high_flux,
    "cn_strength": beat_phase_gated,
    "mask_expansion": kick_hit,
    "color_push": spectral_centroid,
}
```

### Artistic Presets
Each preset remaps the same audio features differently:
- **tunnel_travel** — deep zoom, bass-driven
- **breathing_portrait** — organic bloom, vocals-driven
- **glitch_pulse** — hard cuts, onset-driven
- **hard_techno_camera** — aggressive zoom + shake
- **dream_swell** — slow morph, section-energy driven
- **liquid_feedback** — flow-warp dominant, continuous

---

## FloweR Integration Notes

- FloweR at full blend overpowers audio-driven motion (drift)
- In audio mode, cap at ~12% blend (texture enhancer only)
- FloweR's optical flow predictions are based on frame history — sudden audio-driven motion confuses it
- Works best as occlusion filler + detail smoother, NOT as motion predictor
- Gate flower_blend by audio energy: silence = less FloweR, beats = more

---

## ComfyUI Integration Notes

- AUDIO type: `{"waveform": tensor[batch, channels, samples], "sample_rate": int}`
- Required inputs with defaults show as widgets; missing from old workflows = validation error
- Use optional for new params to avoid breaking existing workflows
- Widget shift rule: never insert required inputs mid-list
- `__pycache__` must be cleared after code changes, or restart ComfyUI
