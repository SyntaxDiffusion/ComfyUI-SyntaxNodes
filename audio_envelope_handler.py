"""
Audio Envelope Handler for SyntaxNodes
Seamlessly integrates with Fill-Nodes audio envelope system
Features adaptive, frequency-aware processing per stem/drum element
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


def _mean_run_length(mask: np.ndarray) -> float:
    """Average length of consecutive True runs in a boolean array."""
    runs = []
    count = 0
    for m in mask:
        if m:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return float(np.mean(runs)) if runs else 0.0


class EnvelopeProcessor:
    """
    Adaptive processor for individual envelope streams.
    Handles normalization, smoothing, and attack/decay characteristics dynamically.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history = deque(maxlen=window_size * 10)  # Keep larger history for statistics
        self.smoothed_values = {}
        self.last_frame_index = None
        self.peak_value = 0.0
        self.rms_value = 0.0
        self.attack_time = 0
        self.decay_time = 0

    def analyze_characteristics(self, envelope: List[float]) -> Dict[str, float]:
        """
        Analyze envelope to determine its dynamic characteristics.
        Returns attack/decay times, peak, RMS, etc.
        """
        if not envelope or len(envelope) < 2:
            return {
                'peak': 0.0,
                'rms': 0.0,
                'attack_frames': 1,
                'decay_frames': 5,
                'sparsity': 1.0,
                'dynamic_range': 0.0
            }

        arr = np.array(envelope)

        # Calculate statistics
        peak = np.max(arr)
        rms = np.sqrt(np.mean(arr ** 2))

        # Detect attack/decay from how long transients actually take: the
        # average number of consecutive rising/falling frames. (Slope
        # magnitudes are amplitudes, not durations — using them as frame
        # counts produced arbitrary filter speeds.)
        diff = np.diff(arr)

        rise_len = _mean_run_length(diff > 0.01)
        attack_frames = max(1, min(10, int(round(rise_len)))) if rise_len > 0 else 1

        # Decay runs are doubled: the release should feel slower than the
        # measured fall so hits ring out instead of cutting off.
        fall_len = _mean_run_length(diff < -0.01)
        decay_frames = max(2, min(30, int(round(fall_len * 2)))) if fall_len > 0 else 5

        # Sparsity: how often it's active vs silent
        threshold = rms * 0.1
        active_frames = np.sum(arr > threshold)
        sparsity = 1.0 - (active_frames / len(arr)) if len(arr) > 0 else 1.0

        # Dynamic range
        if peak > 0:
            dynamic_range = 20 * np.log10(peak / max(rms, 1e-6))
        else:
            dynamic_range = 0.0

        return {
            'peak': float(peak),
            'rms': float(rms),
            'attack_frames': attack_frames,
            'decay_frames': decay_frames,
            'sparsity': sparsity,
            'dynamic_range': dynamic_range
        }

    def adaptive_normalize(self, value: float, characteristics: Dict[str, float],
                          adaptive: bool = True) -> float:
        """
        Normalize value based on analyzed characteristics.
        Adapts to the dynamic range and intensity of the signal.
        """
        if not adaptive or characteristics['peak'] <= 0:
            return value

        # Peak normalization: rescale so the stem's own loudest hit maps to
        # 1.0. Linear scaling preserves relative dynamics exactly (no
        # amplify-then-clip). The floor caps the boost at 4x so near-silent
        # stems stay near-silent instead of being pinned to full scale.
        norm_factor = max(characteristics['peak'], 0.25)

        normalized = value / norm_factor

        # Apply gentle compression for high dynamic range signals
        if characteristics['dynamic_range'] > 20:  # High dynamic range
            normalized = np.tanh(normalized * 1.2) / np.tanh(1.2)

        return float(np.clip(normalized, 0.0, 1.0))

    def apply_smoothing(self, value: float, frame_index: int,
                       attack_frames: int = 1, decay_frames: int = 5) -> float:
        """
        Apply attack/decay smoothing to value.
        Fast attack, slower decay - typical for audio-reactive effects.

        Smooths against the PREVIOUS frame's smoothed value, so it works on
        the sequential one-call-per-frame pattern effect nodes actually use.
        """
        # Frame counter going backwards means a new run started on this
        # shared processor — drop state from the previous run.
        if self.last_frame_index is not None and frame_index < self.last_frame_index:
            self.smoothed_values.clear()
        self.last_frame_index = frame_index

        prev_value = self.smoothed_values.get(frame_index - 1)
        if prev_value is None:
            # No previous frame to smooth against (start of sequence).
            self.smoothed_values[frame_index] = value
            return value

        # Attack (rising): fast response
        if value > prev_value:
            alpha = 1.0 / max(1, attack_frames)
        # Decay (falling): slower response
        else:
            alpha = 1.0 / max(1, decay_frames)

        smoothed = prev_value + alpha * (value - prev_value)
        self.smoothed_values[frame_index] = smoothed

        return smoothed

    def process_value(self, value: float, frame_index: int,
                     characteristics: Dict[str, float],
                     normalize: bool = True,
                     smooth: bool = True) -> float:
        """
        Full processing pipeline: normalize and smooth.
        """
        processed = value

        if normalize:
            processed = self.adaptive_normalize(processed, characteristics)

        if smooth:
            processed = self.apply_smoothing(
                processed,
                frame_index,
                characteristics['attack_frames'],
                characteristics['decay_frames']
            )

        return processed


class AudioEnvelopeHandler:
    """
    Handles audio envelope data from Fill-Nodes for modulating VFX parameters.
    Supports kick, snare, hihat, and custom frequency band envelopes.
    Features adaptive processing per stem with automatic characteristic detection.
    """

    # Stem processors cache - shared across all instances for efficiency
    _stem_processors: Dict[str, EnvelopeProcessor] = {}
    _stem_characteristics: Dict[str, Dict[str, float]] = {}

    # JSON parse cache - avoid re-parsing the same JSON string repeatedly
    _parse_cache: Dict[str, Dict[str, Any]] = {}
    _parse_cache_max_size = 100  # Limit cache size

    def __init__(self):
        self.envelope_cache = {}

    @classmethod
    def get_or_create_processor(cls, stem_name: str) -> EnvelopeProcessor:
        """Get or create a processor for a specific stem."""
        if stem_name not in cls._stem_processors:
            cls._stem_processors[stem_name] = EnvelopeProcessor()
        return cls._stem_processors[stem_name]

    @classmethod
    def analyze_stem(cls, stem_name: str, envelope_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze a stem's envelope to determine its characteristics.
        Results are cached for performance.
        """
        cache_key = f"{stem_name}_{len(envelope_data.get('envelope', []))}"

        if cache_key not in cls._stem_characteristics:
            processor = cls.get_or_create_processor(stem_name)
            characteristics = processor.analyze_characteristics(envelope_data.get('envelope', []))
            cls._stem_characteristics[cache_key] = characteristics

        return cls._stem_characteristics[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clear all cached processors, characteristics, and parsed envelopes."""
        cls._stem_processors.clear()
        cls._stem_characteristics.clear()
        cls._parse_cache.clear()

    @classmethod
    def parse_envelope_json(cls, envelope_json: str) -> Dict[str, Any]:
        """
        Parse envelope JSON string from Fill-Nodes (with caching)

        Args:
            envelope_json: JSON string with format {"envelope": [values...], "total_frames": N}

        Returns:
            Dict with 'envelope' list and 'total_frames' int
        """
        if not envelope_json or envelope_json.strip() in ["", "{}", "null"]:
            return {"envelope": [], "total_frames": 0}

        # Check cache first (use hash of string as key for efficiency)
        cache_key = str(hash(envelope_json))

        if cache_key in cls._parse_cache:
            return cls._parse_cache[cache_key]

        # Not in cache - parse it
        try:
            data = json.loads(envelope_json)
            envelope_list = data.get("envelope", [])
            total_frames = data.get("total_frames", len(envelope_list))

            result = {
                "envelope": envelope_list,
                "total_frames": total_frames
            }

            # Add to cache (with size limit)
            if len(cls._parse_cache) >= cls._parse_cache_max_size:
                # Remove oldest entry
                cls._parse_cache.pop(next(iter(cls._parse_cache)))

            cls._parse_cache[cache_key] = result

            # DEBUG: Log only on first parse (when adding to cache)
            print(f"[AudioEnvelopeHandler] Parsed new envelope: {len(envelope_list)} values, "
                  f"range=[{min(envelope_list) if envelope_list else 0:.4f}, "
                  f"{max(envelope_list) if envelope_list else 0:.4f}], "
                  f"avg={sum(envelope_list)/len(envelope_list) if envelope_list else 0:.4f}")

            return result

        except (json.JSONDecodeError, TypeError) as e:
            print(f"[AudioEnvelopeHandler] ERROR: Failed to parse envelope JSON: {e}")
            return {"envelope": [], "total_frames": 0}

    @staticmethod
    def get_envelope_value(envelope_data: Dict[str, Any], frame_index: int,
                          default: float = 0.0, stem_name: Optional[str] = None,
                          adaptive: bool = True) -> float:
        """
        Get envelope value at specific frame index with bounds checking and adaptive processing

        Args:
            envelope_data: Parsed envelope dictionary
            frame_index: Frame number to query
            default: Default value if frame is out of range
            stem_name: Name of stem for adaptive processing (e.g., 'kick', 'bass')
            adaptive: Apply adaptive normalization and smoothing

        Returns:
            Envelope value (0.0 to 1.0), adaptively processed if enabled
        """
        envelope = envelope_data.get("envelope", [])

        if not envelope or frame_index < 0 or frame_index >= len(envelope):
            return default

        value = max(0.0, min(1.0, float(envelope[frame_index])))

        # Apply adaptive processing if stem name provided
        if adaptive and stem_name:
            characteristics = AudioEnvelopeHandler.analyze_stem(stem_name, envelope_data)
            processor = AudioEnvelopeHandler.get_or_create_processor(stem_name)
            value = processor.process_value(value, frame_index, characteristics)

        return value

    @staticmethod
    def interpolate_envelope(envelope_data: Dict[str, Any],
                            time_position: float,
                            fps: int = 30) -> float:
        """
        Get interpolated envelope value at fractional time position

        Args:
            envelope_data: Parsed envelope dictionary
            time_position: Time in seconds
            fps: Frames per second for conversion

        Returns:
            Interpolated envelope value
        """
        envelope = envelope_data.get("envelope", [])
        if not envelope:
            return 0.0

        frame_pos = time_position * fps
        frame_idx = int(frame_pos)
        fraction = frame_pos - frame_idx

        # Get current and next frame values
        val1 = AudioEnvelopeHandler.get_envelope_value(envelope_data, frame_idx, 0.0)
        val2 = AudioEnvelopeHandler.get_envelope_value(envelope_data, frame_idx + 1, val1)

        # Linear interpolation
        return val1 + (val2 - val1) * fraction

    @staticmethod
    def combine_envelopes(envelopes: List[Dict[str, Any]],
                         weights: Optional[List[float]] = None,
                         mode: str = "add") -> Dict[str, Any]:
        """
        Combine multiple envelopes using various blend modes

        Args:
            envelopes: List of envelope data dicts
            weights: Optional weights for each envelope (default: equal)
            mode: Combine mode - "add", "multiply", "max", "average"

        Returns:
            Combined envelope data
        """
        if not envelopes:
            return {"envelope": [], "total_frames": 0}

        # Filter out empty envelopes
        valid_envelopes = [e for e in envelopes if e.get("envelope")]
        if not valid_envelopes:
            return {"envelope": [], "total_frames": 0}

        # Find max length
        max_frames = max(e.get("total_frames", 0) for e in valid_envelopes)

        # Setup weights
        if weights is None:
            weights = [1.0] * len(valid_envelopes)

        # Combine envelopes
        combined = []
        for frame_idx in range(max_frames):
            values = [
                AudioEnvelopeHandler.get_envelope_value(env, frame_idx, 0.0) * weight
                for env, weight in zip(valid_envelopes, weights)
            ]

            if mode == "add":
                result = sum(values)
            elif mode == "multiply":
                result = np.prod(values) if values else 0.0
            elif mode == "max":
                result = max(values) if values else 0.0
            elif mode == "average":
                result = np.mean(values) if values else 0.0
            else:
                result = sum(values)

            combined.append(max(0.0, min(1.0, result)))

        return {
            "envelope": combined,
            "total_frames": len(combined)
        }

    @staticmethod
    def apply_envelope_to_parameter(base_value: float,
                                    envelope_value: float,
                                    intensity: float = 1.0,
                                    mode: str = "multiply") -> float:
        """
        Apply envelope modulation to a parameter value

        Args:
            base_value: Base parameter value
            envelope_value: Envelope value (0.0 to 1.0)
            intensity: Modulation intensity/depth
            mode: "multiply", "add", "replace"

        Returns:
            Modulated parameter value
        """
        envelope_value = max(0.0, min(1.0, envelope_value))

        if mode == "multiply":
            # Envelope scales the base value
            modulation = 1.0 + (envelope_value - 0.5) * 2.0 * intensity
            return base_value * modulation

        elif mode == "add":
            # Envelope adds to the base value
            return base_value + (envelope_value * intensity)

        elif mode == "replace":
            # Envelope replaces the base value
            return base_value * (1.0 - intensity) + envelope_value * intensity

        else:
            return base_value

    @staticmethod
    def create_multi_band_modulator(kick_env: Optional[str] = None,
                                   snare_env: Optional[str] = None,
                                   hihat_env: Optional[str] = None,
                                   bass_env: Optional[str] = None,
                                   drums_env: Optional[str] = None,
                                   other_env: Optional[str] = None,
                                   vocals_env: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Create a multi-band modulator from multiple frequency-separated envelopes

        Args:
            kick_env: Kick drum envelope JSON
            snare_env: Snare envelope JSON
            hihat_env: Hi-hat envelope JSON
            bass_env: Bass stem envelope JSON
            drums_env: Drums stem envelope JSON
            other_env: Other stem envelope JSON
            vocals_env: Vocals stem envelope JSON

        Returns:
            Dict mapping frequency band names to parsed envelope data
        """
        bands = {}

        envelope_map = {
            "kick": kick_env,
            "snare": snare_env,
            "hihat": hihat_env,
            "bass": bass_env,
            "drums": drums_env,
            "other": other_env,
            "vocals": vocals_env
        }

        for band_name, env_json in envelope_map.items():
            if env_json:
                bands[band_name] = AudioEnvelopeHandler.parse_envelope_json(env_json)

        return bands

    @staticmethod
    def get_standard_inputs() -> Dict[str, Any]:
        """
        Get standard optional input definitions for ComfyUI nodes

        Returns:
            Dict of input type definitions compatible with ComfyUI INPUT_TYPES
        """
        return {
            # Drum-specific envelopes (from FL_Audio_Drum_Detector -> FL_Audio_Reactive_Envelope)
            "kick_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Kick drum envelope JSON from FL_Audio_Reactive_Envelope"
            }),
            "snare_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Snare envelope JSON from FL_Audio_Reactive_Envelope"
            }),
            "hihat_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Hi-hat envelope JSON from FL_Audio_Reactive_Envelope"
            }),

            # Stem-specific envelopes (from FL_Audio_Separation)
            "bass_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Bass stem envelope JSON (optional custom envelope)"
            }),
            "drums_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Drums stem envelope JSON (optional custom envelope)"
            }),
            "vocals_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Vocals stem envelope JSON (optional custom envelope)"
            }),
            "other_envelope": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Other stem envelope JSON (optional custom envelope)"
            }),

            # Modulation controls
            "envelope_intensity": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "tooltip": "Overall envelope modulation intensity"
            }),

            "envelope_mode": (["multiply", "add", "replace"], {
                "default": "add",
                "tooltip": "How envelope modulates effect parameters"
            }),

            # Frequency band mixing
            "kick_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Weight for kick envelope influence"
            }),
            "snare_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Weight for snare envelope influence"
            }),
            "hihat_weight": ("FLOAT", {
                "default": 0.3,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Weight for hi-hat envelope influence"
            }),
            "bass_weight": ("FLOAT", {
                "default": 0.7,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Weight for bass envelope influence"
            }),
            "vocals_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Weight for vocals envelope influence"
            }),
        }

    @staticmethod
    def get_stem_value(stem_name: str, envelope_json: str, frame_index: int,
                      adaptive: bool = True) -> float:
        """
        Universal method to get a processed stem value.
        Automatically applies adaptive normalization and smoothing.

        Args:
            stem_name: Name of stem ('kick', 'snare', 'hihat', 'bass', 'drums', 'vocals', 'other')
            envelope_json: Envelope JSON string
            frame_index: Current frame number
            adaptive: Apply adaptive processing (recommended: True)

        Returns:
            Processed envelope value (0.0 to 1.0)
        """
        if not envelope_json or envelope_json.strip() in ["", "{}", "null"]:
            return 0.0

        env_data = AudioEnvelopeHandler.parse_envelope_json(envelope_json)
        return AudioEnvelopeHandler.get_envelope_value(
            env_data, frame_index, 0.0, stem_name=stem_name, adaptive=adaptive
        )

    @staticmethod
    def get_all_stems(frame_index: int,
                     kick_env: str = "", snare_env: str = "", hihat_env: str = "",
                     bass_env: str = "", drums_env: str = "", vocals_env: str = "",
                     other_env: str = "", adaptive: bool = True) -> Dict[str, float]:
        """
        Get all stem values at once as a dictionary.
        Each stem is independently processed with adaptive algorithms.

        Args:
            frame_index: Current frame number
            *_env: Envelope JSON strings
            adaptive: Apply adaptive processing per stem

        Returns:
            Dict mapping stem names to processed values
        """
        stems = {
            'kick': AudioEnvelopeHandler.get_stem_value('kick', kick_env, frame_index, adaptive),
            'snare': AudioEnvelopeHandler.get_stem_value('snare', snare_env, frame_index, adaptive),
            'hihat': AudioEnvelopeHandler.get_stem_value('hihat', hihat_env, frame_index, adaptive),
            'bass': AudioEnvelopeHandler.get_stem_value('bass', bass_env, frame_index, adaptive),
            'drums': AudioEnvelopeHandler.get_stem_value('drums', drums_env, frame_index, adaptive),
            'vocals': AudioEnvelopeHandler.get_stem_value('vocals', vocals_env, frame_index, adaptive),
            'other': AudioEnvelopeHandler.get_stem_value('other', other_env, frame_index, adaptive),
        }
        return stems

    @staticmethod
    def get_modulation_value(frame_index: int,
                            kick_env: str = "",
                            snare_env: str = "",
                            hihat_env: str = "",
                            bass_env: str = "",
                            drums_env: str = "",
                            vocals_env: str = "",
                            other_env: str = "",
                            kick_weight: float = 1.0,
                            snare_weight: float = 0.5,
                            hihat_weight: float = 0.3,
                            bass_weight: float = 0.7,
                            vocals_weight: float = 0.5,
                            other_weight: float = 0.3,
                            drums_weight: float = 0.8,
                            adaptive: bool = True) -> float:
        """
        Get combined modulation value from all envelope sources.
        Now uses adaptive processing per stem for better results.

        Args:
            frame_index: Current frame number
            *_env: Envelope JSON strings
            *_weight: Weight for each envelope
            adaptive: Apply adaptive processing per stem (recommended: True)

        Returns:
            Combined modulation value (0.0 to 1.0+)
        """
        # Get all stems with adaptive processing
        stems = AudioEnvelopeHandler.get_all_stems(
            frame_index, kick_env, snare_env, hihat_env,
            bass_env, drums_env, vocals_env, other_env, adaptive
        )

        # Apply weights
        total_modulation = (
            stems['kick'] * kick_weight +
            stems['snare'] * snare_weight +
            stems['hihat'] * hihat_weight +
            stems['bass'] * bass_weight +
            stems['drums'] * drums_weight +
            stems['vocals'] * vocals_weight +
            stems['other'] * other_weight
        )

        return max(0.0, total_modulation)


# Convenience function for quick integration
def get_audio_reactive_inputs():
    """Quick access to standard audio-reactive input definitions"""
    return AudioEnvelopeHandler.get_standard_inputs()
