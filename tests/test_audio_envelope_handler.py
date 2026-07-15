"""Tests for EnvelopeProcessor math in audio_envelope_handler.py."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "audio_envelope_handler", ROOT / "audio_envelope_handler.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EnvelopeProcessor = _mod.EnvelopeProcessor


def test_smoothing_fires_on_sequential_per_frame_calls():
    # Effect nodes call once per frame with increasing frame_index. A drop
    # from 1.0 to 0.0 with decay_frames=5 must NOT return the raw 0.0.
    proc = EnvelopeProcessor()
    first = proc.apply_smoothing(1.0, 0, attack_frames=1, decay_frames=5)
    second = proc.apply_smoothing(0.0, 1, attack_frames=1, decay_frames=5)

    assert first == 1.0
    assert second == 1.0 + (1.0 / 5) * (0.0 - 1.0)  # 0.8, not 0.0


def test_smoothing_resets_when_frame_counter_rewinds():
    proc = EnvelopeProcessor()
    proc.apply_smoothing(1.0, 0)
    proc.apply_smoothing(0.5, 1)

    # New run starts at frame 0 — previous run's state must not leak in.
    restart = proc.apply_smoothing(0.2, 0)
    assert restart == 0.2


def test_adaptive_normalize_preserves_relative_dynamics():
    proc = EnvelopeProcessor()
    characteristics = {'peak': 0.5, 'rms': 0.2, 'dynamic_range': 10.0}

    full = proc.adaptive_normalize(0.5, characteristics)
    half = proc.adaptive_normalize(0.25, characteristics)

    assert full == 1.0
    # Linear peak scaling: half the input stays half the output.
    assert abs(half - 0.5) < 1e-9


def test_adaptive_normalize_does_not_pin_quiet_signals_to_max():
    proc = EnvelopeProcessor()
    characteristics = {'peak': 0.01, 'rms': 0.005, 'dynamic_range': 10.0}

    out = proc.adaptive_normalize(0.01, characteristics)
    assert out < 0.1  # boost capped at 4x, not amplified to full scale


def test_attack_decay_frames_measure_transient_durations():
    # Sawtooth: rises over 4 frames, falls over 6, repeated. Attack/decay
    # should reflect those durations (decay is doubled by design).
    rise = np.linspace(0.0, 1.0, 5)          # 4 rising diffs
    fall = np.linspace(1.0, 0.0, 7)[1:]      # 6 falling diffs
    envelope = np.concatenate([rise, fall] * 4).tolist()

    proc = EnvelopeProcessor()
    characteristics = proc.analyze_characteristics(envelope)

    assert characteristics['attack_frames'] == 4
    assert characteristics['decay_frames'] == 12
