"""Dependency-free tests for feedback render progress reporting."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "syntax_feedback_progress", ROOT / "syntax_feedback_progress.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SyntaxFeedbackProgress = MODULE.SyntaxFeedbackProgress


class RecordingProgressBar:
    def __init__(self):
        self.updates = []

    def update_absolute(self, value, preview=None):
        self.updates.append((value, preview))


class RecordingPreviewer:
    def __init__(self):
        self.latents = []

    def decode_latent_to_preview_image(self, image_format, latent):
        self.latents.append(latent)
        return (image_format, f"image:{latent}", 512)


def test_sampling_and_completed_frames_share_sequence_progress():
    times = iter([0.0, 1.0, 3.0, 4.0, 4.25, 5.0])
    messages = []
    native_step_updates = []
    native_frame_updates = []
    progress_bar = RecordingProgressBar()
    progress = SyntaxFeedbackProgress(
        None,
        frame_count=4,
        frame_cadence=2,
        steps=2,
        unique_id="node-7",
        clock=lambda: next(times),
        progress_bar=progress_bar,
        native_step_callback=lambda step, x0, x, total: native_step_updates.append(
            (step, x0, total)
        ),
        native_frame_callback=lambda step, x0, x, total: native_frame_updates.append(
            (step, x0, total)
        ),
        text_sender=lambda text, node_id: messages.append((text, node_id)),
    )

    progress.begin_frame(0, will_diffuse=True)
    callback = progress.sampling_callback(0)
    callback(0, "step-1", None, 2)
    callback(1, "step-2", None, 2)
    progress.complete_frame(0, "frame-0", was_diffused=True)
    progress.begin_frame(1, will_diffuse=False)
    progress.complete_frame(1, "frame-1", was_diffused=False)
    progress.finish()

    assert [value for value, _ in progress_bar.updates] == [500, 1000, 1000, 2000]
    assert native_step_updates == [(0, "step-1", 2), (1, "step-2", 2)]
    assert native_frame_updates == [(0, "frame-0", 1), (0, "frame-1", 1)]
    assert any("ETA" in text for text, _ in messages)
    assert messages[-1][0].startswith("4/4 frames complete")
    assert all(node_id == "node-7" for _, node_id in messages)


def test_live_previews_are_attached_to_step_and_frame_progress():
    progress_bar = RecordingProgressBar()
    previewer = RecordingPreviewer()
    native_updates = []
    progress = SyntaxFeedbackProgress(
        None,
        frame_count=1,
        frame_cadence=1,
        steps=2,
        clock=lambda: 0.0,
        progress_bar=progress_bar,
        previewer=previewer,
        native_step_callback=lambda *args: native_updates.append(args),
        native_frame_callback=lambda *args: native_updates.append(args),
        text_sender=None,
    )

    progress.begin_frame(0, will_diffuse=True)
    callback = progress.sampling_callback(0)
    callback(0, "step-1", None, 2)
    callback(1, "step-2", None, 2)
    progress.complete_frame(0, "frame-0", was_diffused=True)

    assert previewer.latents == ["step-1", "step-2", "frame-0"]
    assert [preview[1] for _, preview in progress_bar.updates] == [
        "image:step-1",
        "image:step-2",
        "image:frame-0",
    ]
    assert native_updates == []


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
