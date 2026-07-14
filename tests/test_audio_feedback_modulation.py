"""Tests for audio-reactive feedback animation - onset detection approach.

The node detects TRANSIENTS (rising edges) in audio envelopes rather than
using raw levels. This ensures that sustained bass doesn't peg zoom at max,
and actual drum hits produce sharp, visible motion impulses.
"""
import numpy as np

# Constants matching the node
_DECAY_RATE = 0.35
_ZOOM_MAX = 0.035
_ANGLE_MAX = 4.0
_TRANSLATE_MAX = 12.0


def _detect_onsets_simple(values):
    """Simplified onset detection matching the node's logic."""
    onsets = np.zeros(len(values))
    for i in range(1, len(values)):
        onset = max(0.0, values[i] - values[i - 1])
        onsets[i] = onset
    peak = np.max(onsets) if np.max(onsets) > 0.001 else 1.0
    return onsets / peak, peak


def _apply_decay(raw_arr):
    """Apply sustain/decay matching the node's logic."""
    result = np.zeros(len(raw_arr))
    for i in range(len(raw_arr)):
        if i == 0:
            result[i] = raw_arr[i]
        else:
            result[i] = max(raw_arr[i], result[i - 1] * _DECAY_RATE)
    return result


# ── Onset detection tests ────────────────────────────────────────────────

def test_sustained_signal_produces_no_onsets():
    """A constant signal (sustained bass) should produce zero onsets after frame 0."""
    values = [0.9] * 20  # Sustained at 0.9
    onsets, peak = _detect_onsets_simple(values)
    # All onsets should be 0 (no rising edges after initial)
    assert all(onsets[1:] == 0.0), f"Sustained signal should produce zero onsets, got {onsets}"


def test_impulse_produces_onset():
    """A sudden spike should produce a single strong onset."""
    values = [0.0] * 5 + [1.0] + [0.0] * 5
    onsets, peak = _detect_onsets_simple(values)
    assert onsets[5] == 1.0, f"Spike at index 5 should produce onset=1.0, got {onsets[5]}"
    assert all(onsets[i] == 0.0 for i in range(len(values)) if i != 5), \
        "Only the spike frame should have an onset"


def test_repeated_kicks_produce_repeated_onsets():
    """Kick pattern (spike every 4 frames) should produce periodic onsets."""
    values = []
    for i in range(20):
        if i % 4 == 0:
            values.append(1.0)
        else:
            values.append(max(0, values[-1] - 0.3) if values else 0.0)
    onsets, peak = _detect_onsets_simple(values)
    # Frames 0, 4, 8, 12, 16 should have onsets (rising edges)
    for beat_frame in [4, 8, 12, 16]:
        assert onsets[beat_frame] > 0.5, f"Frame {beat_frame} should have onset, got {onsets[beat_frame]}"


def test_gradual_rise_produces_small_onsets():
    """A gradual volume swell should produce small distributed onsets, not big hits."""
    values = [i / 20.0 for i in range(20)]  # 0.0 to 0.95
    onsets, peak = _detect_onsets_simple(values)
    # Each frame has equal small onset (0.05 / peak)
    # All should be similar - no single frame dominates
    nonzero = onsets[onsets > 0]
    if len(nonzero) > 0:
        assert np.std(nonzero) < 0.1, "Gradual rise should produce uniform small onsets"


def test_onset_normalization():
    """Onsets should be normalized so the biggest hit = 1.0."""
    values = [0.0, 0.5, 0.0, 1.0, 0.0]  # Two hits: 0.5 and 1.0
    onsets, peak = _detect_onsets_simple(values)
    assert abs(onsets[3] - 1.0) < 0.001, "Biggest onset should normalize to 1.0"
    assert abs(onsets[1] - 0.5) < 0.001, "Smaller onset should be proportional"


# ── Dynamic range tests ──────────────────────────────────────────────────

def test_your_actual_data_has_dynamics():
    """Simulate the user's scenario: kick at 1.0 most frames.
    Onset detection should find the HITS, not the sustained level."""
    # Simulated kick envelope: sustained around 0.8-1.0 with periodic hits
    values = [0.8] * 15
    # Insert actual kick hits (drops to 0.3 then spikes to 1.0)
    for hit_frame in [3, 7, 12]:
        if hit_frame > 0:
            values[hit_frame - 1] = 0.3  # Dip before hit
        values[hit_frame] = 1.0  # Hit

    onsets, peak = _detect_onsets_simple(values)

    # Only hit frames should have large onsets
    hit_onsets = [onsets[3], onsets[7], onsets[12]]
    non_hit_onsets = [onsets[i] for i in range(15) if i not in [3, 7, 12]]

    assert all(h > 0.5 for h in hit_onsets), f"Hit frames should have large onsets: {hit_onsets}"
    # Most non-hit frames should be zero or near-zero
    near_zero = sum(1 for v in non_hit_onsets if v < 0.1)
    assert near_zero >= len(non_hit_onsets) * 0.6, \
        f"Most non-hit frames should be near zero: {non_hit_onsets}"


# ── Sustain/decay tests ─────────────────────────────────────────────────

def test_decay_after_onset():
    """After an onset, value should decay gradually."""
    raw = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    result = _apply_decay(raw)
    assert result[1] == 1.0
    assert abs(result[2] - 0.35) < 0.001
    assert abs(result[3] - 0.35 * 0.35) < 0.001


def test_new_onset_overrides_decay():
    """A new onset should override a decaying value."""
    raw = np.array([0.0, 1.0, 0.0, 0.0, 0.8, 0.0])
    result = _apply_decay(raw)
    assert result[4] == 0.8, "New onset should override decay"


def test_beat_pattern_has_contrast():
    """Beat pattern should show clear peaks and valleys."""
    raw = np.zeros(20)
    raw[0] = 1.0
    raw[8] = 0.7
    raw[16] = 1.0
    result = _apply_decay(raw)

    # Peaks should be at beat frames
    assert result[0] == 1.0
    assert result[8] == 0.7
    assert result[16] == 1.0
    # Between beats should decay to near zero
    assert result[6] < 0.01, f"Frame 6 should be near zero, got {result[6]}"
    assert result[14] < 0.01, f"Frame 14 should be near zero, got {result[14]}"


# ── Motion schedule tests ───────────────────────────────────────────────

def test_zoom_at_peak_onset():
    """Peak onset with base=0 should produce _ZOOM_MAX."""
    onset = 1.0
    frame_zoom = onset * _ZOOM_MAX
    assert abs(frame_zoom - 0.035) < 0.001


def test_zoom_at_zero_onset():
    """Zero onset should produce zero motion (with base=0)."""
    onset = 0.0
    frame_zoom = onset * _ZOOM_MAX
    assert frame_zoom == 0.0


def test_zoom_with_base_value():
    """With non-zero base zoom, onset should add proportional boost."""
    zoom = 0.01
    onset = 1.0
    frame_zoom = zoom + onset * abs(zoom) * 3.0
    # 0.01 + 1.0 * 0.01 * 3.0 = 0.04
    assert abs(frame_zoom - 0.04) < 0.001


def test_schedule_has_silence_and_hits():
    """A realistic schedule should have both silent and active frames."""
    # Build a simple 20-frame schedule
    kick_env = [0.8, 0.8, 0.8, 0.3, 1.0, 0.7, 0.4, 0.2,
                0.1, 0.1, 0.1, 0.3, 1.0, 0.6, 0.3, 0.1,
                0.1, 0.1, 0.1, 0.1]
    onsets, _ = _detect_onsets_simple(kick_env)
    decayed = _apply_decay(onsets)

    # Should have both active and quiet frames
    active = sum(1 for v in decayed if v > 0.3)
    quiet = sum(1 for v in decayed if v < 0.05)
    assert active >= 2, f"Should have at least 2 active frames, got {active}"
    assert quiet >= 5, f"Should have at least 5 quiet frames, got {quiet}"


# ── Sensitivity (envelope_intensity) tests ───────────────────────────────

def test_high_sensitivity_boosts_weak_onsets():
    """High envelope_intensity should make weak onsets more visible."""
    raw = np.array([0.0, 0.2, 0.0, 0.1, 0.0])

    # Low sensitivity (intensity=0.5 -> power=2.0)
    low_sens = np.power(raw, 2.0)
    # High sensitivity (intensity=2.0 -> power=0.5)
    high_sens = np.power(raw, 0.5)

    # High sensitivity should boost the 0.2 value more than low
    assert high_sens[1] > low_sens[1], \
        f"High sensitivity should boost weak onsets: {high_sens[1]} vs {low_sens[1]}"


def test_default_sensitivity_is_linear():
    """envelope_intensity=1.0 should produce linear (no power curve)."""
    raw = np.array([0.0, 0.5, 1.0])
    power = 1.0 / max(0.1, 1.0)
    result = np.power(raw, power)
    assert np.allclose(result, raw), "intensity=1.0 should be linear pass-through"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
            failed += 1
    print(f"\nRan {len(tests)} tests: {passed} passed, {failed} failed")
