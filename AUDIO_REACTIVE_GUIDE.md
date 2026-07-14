# Audio-Reactive Effect Guide

## Overview

All SyntaxNodes now use an **ultra-responsive audio-reactive system** that provides instant, dramatic visual reactions to music. The system uses raw audio envelope values with proportional frame mapping for perfect synchronization.

## How It Works

### Core Concepts

1. **Proportional Frame Mapping**
   - Video frames are mapped across the full audio envelope timeline
   - Example: 76 video frames automatically map across 6,250 audio frames
   - Frame scale calculated as: `envelope_total_frames / video_batch_size`
   - Each video frame gets its own unique audio envelope frame

2. **Raw Audio Values (No Adaptive Processing)**
   - Uses `adaptive=False` for maximum dynamic range
   - Envelope values are 0.0 (silence) to 1.0 (peak)
   - No smoothing, no normalization, no artificial thresholds
   - Instant response to every beat, kick, and snare

3. **Direct Parameter Modulation**
   - Audio values directly multiply or add to effect parameters
   - No intermediate processing layers
   - **Multiplication:** `param × (1.0 + stem_value × multiplier × intensity)`
   - **Addition:** `param + (stem_value × amount × intensity)`

### Audio Stems Available

When using Fill-Nodes FL_Audio_Reactive_Envelope, you can connect:

- **kick_envelope**: Kick drum hits (low frequency transients)
- **snare_envelope**: Snare hits (mid frequency transients)
- **hihat_envelope**: Hi-hat cymbals (high frequency)
- **bass_envelope**: Bass lines (sustained low frequency)
- **drums_envelope**: Overall drum mix
- **vocals_envelope**: Vocal stems
- **other_envelope**: Other musical elements

## Node-Specific Mappings

### VoxelNode

| Parameter | Audio Stem | Effect | Formula |
|-----------|------------|--------|---------|
| `block_size` | Kick | EXPLODES on kicks | `base × (1 + kick×4×intensity)` |
| `block_depth` | Kick | 3D pop on kicks | `base + (kick×24×intensity)` |
| `shading` | Snare | Flash on snares | `base + (snare×0.5×intensity)` |

**Example Settings:**
- Base: `block_size=4`, `block_depth=4`, `intensity=2.0`
- On kick (1.0): `block_size=4→36`, `block_depth=4→52` (clamped to 32)

### RGB Streak Node

| Parameter | Audio Stem | Effect | Formula |
|-----------|------------|--------|---------|
| `streak_length` | Kick + Bass | Extends on low freq | `base × (1 + low_freq×2×intensity)` |
| `red_intensity` | Snare | Flash on snares | `base × (1 + snare×3×intensity)` |
| `green_intensity` | Overall Energy | Pulse on beats | `base × (1 + energy×2×intensity)` |
| `blue_intensity` | Hi-hat | Shimmer on hi-hats | `base × (1 + hihat×2×intensity)` |

### Pointillism Node

| Parameter | Audio Stem | Effect | Formula |
|-----------|------------|--------|---------|
| `dot_density` | Kick + Bass | More dots on bass | `base × (1 + low_freq×3×intensity)` |
| `dot_radius` | Hi-hat | Sparkles on hi-hats | `base × (1 + hihat×2×intensity)` |

### Low Poly Node

| Parameter | Audio Stem | Effect | Formula |
|-----------|------------|--------|---------|
| `num_points` | Kick + Bass | More detail on bass | `base × (1 + low_freq×4×intensity)` |
| `edge_points` | Snare | Edge emphasis | `base × (1 + snare×3×intensity)` |

### Paper Craft Node

| Parameter | Audio Stem | Effect | Formula |
|-----------|------------|--------|---------|
| `fold_depth` | Kick | Deeper folds | `base × (1 + kick×4×intensity)` |
| `shadow_strength` | Snare | Flash | `base + (snare×0.5×intensity)` |

## Recommended Settings

### For Subtle Effects
```
envelope_intensity: 0.5 - 1.0
Base parameters: mid-range values
Result: Gentle pulsing with music
```

### For Moderate Effects (Recommended)
```
envelope_intensity: 2.0 - 3.0
Base parameters: low-mid range values
Result: Clear visual response to beats
```

### For Dramatic Effects
```
envelope_intensity: 5.0 - 10.0
Base parameters: minimal (near zero)
Result: Explosive visual changes on beats
```

**Example (VoxelNode):**
- Subtle: `intensity=1.0`, `block_size=16`, `block_depth=4`
- Dramatic: `intensity=10.0`, `block_size=4`, `block_depth=0`

## Frame Index Parameter

The `frame_index` parameter is a **starting offset** for envelope mapping:

- `frame_index=0`: Start from beginning of audio envelope (default)
- `frame_index=100`: Start from frame 100 of the audio envelope
- Use for synchronizing multiple effect nodes
- Use for looping animations that need different starting points

**Frame mapping formula:**
```
envelope_frame = int(video_frame × frame_scale) + frame_index
```

## Troubleshooting

### "Effects are too subtle"
1. ✅ **Increase `envelope_intensity`** (try 3.0, 5.0, or even 10.0)
2. ✅ **Lower base parameter values** (allows more range for modulation)
3. ✅ **Check console output** - Look for debug messages showing stem values
4. ❌ Don't adjust `kick_weight`, `snare_weight` (deprecated, no longer used)

### "Effects are too extreme"
1. ✅ **Decrease `envelope_intensity`** (try 0.5 or 1.0)
2. ✅ **Raise base parameter values** (provides more stable base effect)
3. ✅ **Use fewer audio envelopes** (disconnect some stems)

### "Not hitting on the beat"
1. ✅ **Check envelope connection** - Ensure FL_Audio_Reactive_Envelope is connected
2. ✅ **Set `frame_index=0`** for proper sync
3. ✅ **Check console debug output:**
   ```
   [NodeName] Mapping 76 video frames to 6250 envelope frames (scale=82.24x)
   [NodeName] Video frame 4 → Envelope frame 328:
     STEMS: kick=1.000, snare=0.000, hihat=0.000
     RESULT: block_size 4→44, depth 4→32
   ```
4. ✅ **Verify envelope has data** - Look for non-zero stem values in console

### "All stem values are 0.000"
1. ✅ **Check FL_Audio_Drum_Detector settings** - Ensure it detected beats
2. ✅ **Audio might start with silence** - First 80+ frames may be quiet
3. ✅ **Check total_frames in envelope** - Should match audio length
4. ✅ **Try different frame ranges** - Use `frame_index` to skip silent intro

### "Console spam / too much debug output"
- Debug output only shows:
  - First 3 frames of each batch
  - Frames with audio activity (stem values > 0.1)
- This is by design to reduce spam while showing important events

## Technical Details

### Envelope Format (from Fill-Nodes)
```json
{
  "envelope": [0.0, 0.0, 0.4, 1.0, 0.6, ...],
  "total_frames": 6250
}
```

### Performance Optimization
- Envelope JSON parsing is cached (parsed once per unique envelope)
- Proportional frame mapping calculated once per batch
- Stem extraction per-frame (necessary for temporal accuracy)

### Why Adaptive Processing is Disabled
- `adaptive=True` normalizes each stem independently
- This crushes dynamic range: weak kicks (0.4) become 1.0, strong kicks (1.0) become 1.0
- `adaptive=False` preserves variation: 0.4 stays 0.4, 1.0 stays 1.0
- Result: More expressive, proportional visual response

## Best Practices

1. **Start with default settings** - Most nodes have sensible defaults
2. **Adjust intensity first** - This has the biggest impact
3. **Use console debug** - Watch for stem values and modulation results
4. **Experiment with base values** - Lower base = more dramatic range
5. **Connect multiple stems** - Combine kick+bass, snare+hihat for richer effects
6. **Match video length to audio** - Frame mapping handles mismatches automatically

## Common Workflows

### Music Video Effect
1. Load video frames
2. Connect FL_Audio_Drum_Detector → FL_Audio_Reactive_Envelope
3. Connect kick and snare envelopes to effect node
4. Set `envelope_intensity=3.0`
5. Set `frame_index=0`
6. Render and review console output

### Layered Effects
1. Use multiple effect nodes with same audio envelopes
2. Use different `frame_index` values for variety
3. Blend results using masks or compositing
4. Different nodes respond to different stems (kick vs snare vs hihat)

### Looping Animation
1. Ensure video length matches audio envelope
2. Use `frame_index=0` for perfect loop sync
3. Proportional mapping handles frame count automatically
