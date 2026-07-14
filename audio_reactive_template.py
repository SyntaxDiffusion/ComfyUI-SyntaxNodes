"""
Audio-Reactive Node Template
-----------------------------
Copy this template to create any audio-reactive effect node.
The AudioEnvelopeHandler automatically adapts to each stem's frequency characteristics.

Key features:
- Adaptive normalization per stem (handles different volume levels automatically)
- Adaptive smoothing with attack/decay detection
- Dynamic range compression for consistent response
- Universal API - just call get_all_stems() and you're done!
"""

import numpy as np
import torch
from .audio_envelope_handler import AudioEnvelopeHandler


class AudioReactiveTemplateNode:
    """
    Template for creating audio-reactive nodes.

    Integration steps:
    1. Add audio inputs using AudioEnvelopeHandler.get_standard_inputs()
    2. Call AudioEnvelopeHandler.get_all_stems() to get processed values
    3. Map stem values to your effect parameters
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get standard audio inputs - this adds all stem/drum inputs automatically
        audio_inputs = AudioEnvelopeHandler.get_standard_inputs()

        return {
            "required": {
                # Your effect parameters here
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Current frame number for audio sync"
                }),
            },
            # Audio inputs are optional - effect works without them
            "optional": audio_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "image/effects"

    def apply_effect(self, image, intensity, frame_index=0,
                    # Audio envelope parameters (all optional)
                    kick_envelope="", snare_envelope="", hihat_envelope="",
                    bass_envelope="", drums_envelope="", vocals_envelope="", other_envelope="",
                    envelope_intensity=1.0, envelope_mode="multiply",
                    kick_weight=1.0, snare_weight=0.5, hihat_weight=0.3,
                    bass_weight=0.7, vocals_weight=0.5):

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Get all stem values (automatically adaptive!)
        # ═══════════════════════════════════════════════════════════════
        stems = AudioEnvelopeHandler.get_all_stems(
            frame_index,
            kick_envelope, snare_envelope, hihat_envelope,
            bass_envelope, drums_envelope, vocals_envelope, other_envelope,
            adaptive=True  # Each stem auto-normalized and smoothed based on its characteristics
        )

        # stems is now a dict: {'kick': 0.0-1.0, 'snare': 0.0-1.0, ...}
        # Each value is already:
        # - Normalized to its own dynamic range
        # - Smoothed with appropriate attack/decay
        # - Compressed if needed for consistent response

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Map stems to your effect parameters
        # ═══════════════════════════════════════════════════════════════

        # Example mappings (customize these for your effect):

        # Low frequency energy (kick + bass) → parameter A
        low_freq = (stems['kick'] * kick_weight + stems['bass'] * bass_weight) * envelope_intensity

        # Mid frequency energy (snare) → parameter B
        mid_freq = stems['snare'] * snare_weight * envelope_intensity

        # High frequency energy (hihat + vocals) → parameter C
        high_freq = (stems['hihat'] * hihat_weight + stems['vocals'] * vocals_weight) * envelope_intensity

        # Overall drum energy → parameter D
        drums_energy = (stems['kick'] + stems['snare'] + stems['hihat']) / 3.0 * envelope_intensity

        # Full mix (all stems) → parameter E
        full_mix = sum(stems.values()) / len(stems) * envelope_intensity

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Apply modulation to your parameters
        # ═══════════════════════════════════════════════════════════════

        # Option A: Use built-in modulation helper
        modulated_intensity = AudioEnvelopeHandler.apply_envelope_to_parameter(
            intensity,
            low_freq,  # or mid_freq, high_freq, etc.
            envelope_intensity,
            envelope_mode  # "multiply", "add", or "replace"
        )

        # Option B: Manual modulation (full control)
        if envelope_mode == "multiply":
            # Scale intensity by audio (0.0 = 0%, 1.0 = 200%)
            modulated_intensity = intensity * (1.0 + low_freq)
        elif envelope_mode == "add":
            # Add audio value to intensity
            modulated_intensity = intensity + low_freq
        elif envelope_mode == "replace":
            # Audio directly sets intensity
            modulated_intensity = low_freq
        else:
            modulated_intensity = intensity

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Apply your effect (example: brightness modulation)
        # ═══════════════════════════════════════════════════════════════

        img = image[0].cpu().numpy()

        # Example effect: modulate brightness
        result = img * modulated_intensity
        result = np.clip(result, 0, 1)

        return (torch.from_numpy(result).float().unsqueeze(0),)


# ═══════════════════════════════════════════════════════════════════════
# ADVANCED USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════

def example_single_stem_usage(frame_index):
    """
    Example: Get just one stem value
    """
    kick_value = AudioEnvelopeHandler.get_stem_value(
        'kick',
        kick_envelope_json,
        frame_index,
        adaptive=True
    )
    return kick_value


def example_frequency_band_mapping(stems):
    """
    Example: Map frequency bands to RGB channels
    """
    return {
        'red': stems['bass'] + stems['kick'],      # Low frequencies
        'green': stems['snare'] + stems['drums'],   # Mid frequencies
        'blue': stems['hihat'] + stems['vocals'],   # High frequencies
    }


def example_rhythmic_effects(stems):
    """
    Example: Detect hits for rhythmic effects
    """
    # Stems are already smoothed, so you can detect peaks easily
    kick_hit = stems['kick'] > 0.7  # Kick drum hit
    snare_hit = stems['snare'] > 0.6  # Snare hit

    return {
        'flash_on_kick': kick_hit,
        'shake_on_snare': snare_hit,
    }


def example_custom_mix(stems, weights):
    """
    Example: Create custom stem mixes
    """
    custom_mix = sum(stems[stem] * weight for stem, weight in weights.items())
    return custom_mix


# ═══════════════════════════════════════════════════════════════════════
# USAGE TIPS
# ═══════════════════════════════════════════════════════════════════════

"""
FREQUENCY MAPPING GUIDE:
- Kick/Bass (low):     Long streaks, slow movement, size/scale, warmth
- Snare (mid):         Color shifts, flash effects, rotation
- Hihat/Vocals (high): Sparkle, noise, fine detail, rapid changes
- Overall energy:      General intensity, brightness, opacity

MODULATION MODES:
- "multiply": Scales parameter (good for intensity, size, speed)
- "add": Adds to parameter (good for offsets, positions)
- "replace": Audio becomes the parameter (good for direct control)

WEIGHTS:
- Set higher weights (1.0-2.0) for primary rhythmic elements
- Set lower weights (0.3-0.5) for accent elements
- Adjust envelope_intensity to control overall audio influence

ADAPTIVE PROCESSING:
- Each stem auto-analyzes: peak, RMS, attack time, decay time, sparsity
- Normalization adapts to each stem's dynamic range
- Smoothing adapts to each stem's transient characteristics
- Works with any audio content - no manual tuning required!
"""


# Register the node
NODE_CLASS_MAPPINGS = {
    "AudioReactiveTemplateNode": AudioReactiveTemplateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioReactiveTemplateNode": "Audio-Reactive Template"
}
