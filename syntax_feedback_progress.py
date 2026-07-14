"""Display-only progress reporting for the prompt-scheduled feedback sampler."""

from __future__ import annotations

import math
import logging
import time


_AUTO = object()


def format_duration(seconds):
    """Format a rolling render duration without noisy sub-second precision."""
    seconds = max(0, int(math.ceil(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


class SyntaxFeedbackProgress:
    """Combine sampler previews, frame progress, and ETA in one UI stream.

    The callback only observes sampler tensors. It does not modify noise,
    latents, conditioning, or sampler settings.
    """

    UNITS_PER_FRAME = 1000

    def __init__(
        self,
        model,
        frame_count,
        frame_cadence,
        steps,
        unique_id=None,
        *,
        clock=time.perf_counter,
        progress_bar=None,
        previewer=_AUTO,
        native_step_callback=_AUTO,
        native_frame_callback=_AUTO,
        text_sender=_AUTO,
    ):
        self.frame_count = max(1, int(frame_count))
        self.frame_cadence = max(1, int(frame_cadence))
        self.unique_id = unique_id
        self.clock = clock
        self.started_at = clock()
        self.frame_started_at = None
        self.frame_times = {True: [], False: []}

        total_units = self.frame_count * self.UNITS_PER_FRAME
        if progress_bar is None:
            import comfy.utils

            # Match native KSampler: let the execution context resolve the node.
            progress_bar = comfy.utils.ProgressBar(total_units)
        self.progress_bar = progress_bar

        latent_preview_module = None
        needs_forced_previewer = previewer is _AUTO and model is not None
        if (
            needs_forced_previewer
            or native_step_callback is _AUTO
            or native_frame_callback is _AUTO
        ):
            import latent_preview

            latent_preview_module = latent_preview

        if previewer is _AUTO:
            previewer = None
            # ComfyUI starts with --preview-method none unless it is overridden.
            # This node promises live previews, so build the inexpensive model-
            # supplied Latent2RGB previewer directly instead of inheriting that
            # global switch. This only reads x0 and cannot affect sampling.
            latent_format = getattr(getattr(model, "model", None), "latent_format", None)
            factors = getattr(latent_format, "latent_rgb_factors", None)
            if factors is not None:
                previewer = latent_preview_module.Latent2RGBPreviewer(
                    factors,
                    getattr(latent_format, "latent_rgb_factors_bias", None),
                    getattr(latent_format, "latent_rgb_factors_reshape", None),
                )
        self.previewer = previewer
        self._preview_error_reported = False

        if native_step_callback is _AUTO or native_frame_callback is _AUTO:
            if native_step_callback is _AUTO:
                native_step_callback = latent_preview_module.prepare_callback(
                    model, max(1, int(steps))
                )
            if native_frame_callback is _AUTO:
                native_frame_callback = latent_preview_module.prepare_callback(model, 1)
        self.native_step_callback = native_step_callback
        self.native_frame_callback = native_frame_callback

        if text_sender is _AUTO:
            text_sender = None
            if unique_id is not None:
                try:
                    from server import PromptServer

                    if PromptServer.instance is not None and hasattr(
                        PromptServer.instance, "send_progress_text"
                    ):
                        text_sender = PromptServer.instance.send_progress_text
                except (AttributeError, ImportError):
                    text_sender = None
        self.text_sender = text_sender
        self._send_text(f"0/{self.frame_count} frames | starting")

    def _send_text(self, text):
        if self.text_sender is None:
            return
        try:
            self.text_sender(text, self.unique_id)
        except (AttributeError, RuntimeError, TypeError):
            # Progress text is optional on older ComfyUI frontends.
            pass

    def _preview(self, latent):
        if self.previewer is None:
            return None
        try:
            return self.previewer.decode_latent_to_preview_image("JPEG", latent)
        except Exception as exc:
            # A preview failure must never abort or alter a render. Fall back to
            # ComfyUI's native callback, while making the reason visible once.
            if not self._preview_error_reported:
                logging.warning("Syntax live latent preview failed: %s", exc)
                self._preview_error_reported = True
            self.previewer = None
            return None

    def _eta_seconds(self, completed_frame):
        observed = self.frame_times[True] + self.frame_times[False]
        if not observed:
            return None

        overall_average = sum(observed) / len(observed)
        diffuse_average = (
            sum(self.frame_times[True]) / len(self.frame_times[True])
            if self.frame_times[True]
            else overall_average
        )
        motion_average = (
            sum(self.frame_times[False]) / len(self.frame_times[False])
            if self.frame_times[False]
            else overall_average
        )

        eta = 0.0
        for frame_index in range(completed_frame + 1, self.frame_count):
            if frame_index == 0 or frame_index % self.frame_cadence == 0:
                eta += diffuse_average
            else:
                eta += motion_average
        return eta

    def _eta_text(self, completed_frame):
        eta = self._eta_seconds(completed_frame)
        return "estimating ETA" if eta is None else f"ETA {format_duration(eta)}"

    def begin_frame(self, frame_index, will_diffuse):
        self.frame_started_at = self.clock()
        mode = "preparing diffusion" if will_diffuse else "motion-only"
        self._send_text(
            f"Frame {frame_index + 1}/{self.frame_count} | {mode} | "
            f"{self._eta_text(frame_index - 1)}"
        )

    def sampling_callback(self, frame_index):
        """Build a native ComfyUI callback for one frame's diffusion pass."""
        self._send_text(
            f"Frame {frame_index + 1}/{self.frame_count} | diffusing | "
            f"{self._eta_text(frame_index - 1)}"
        )

        def callback(step, x0, _x, total_steps):
            preview = self._preview(x0)
            # Fall back to ComfyUI's native callback for latent formats that do
            # not provide Latent2RGB factors.
            if preview is None and self.native_step_callback is not None:
                self.native_step_callback(step, x0, _x, total_steps)

            total_steps = max(1, int(total_steps))
            frame_fraction = min(1.0, max(0.0, (step + 1) / total_steps))
            value = (
                frame_index * self.UNITS_PER_FRAME
                + round(frame_fraction * self.UNITS_PER_FRAME)
            )
            # Carry the image on sequence-wide progress so the frontend binds
            # it to this node while retaining overall frame progress.
            self.progress_bar.update_absolute(value, preview=preview)

        return callback

    def complete_frame(self, frame_index, latent, was_diffused):
        now = self.clock()
        if self.frame_started_at is not None:
            self.frame_times[bool(was_diffused)].append(now - self.frame_started_at)
        value = (frame_index + 1) * self.UNITS_PER_FRAME
        preview = self._preview(latent)
        if preview is None and self.native_frame_callback is not None:
            self.native_frame_callback(0, latent, latent, 1)
        self.progress_bar.update_absolute(value, preview=preview)
        self._send_text(
            f"Frame {frame_index + 1}/{self.frame_count} complete | "
            f"{self._eta_text(frame_index)}"
        )

    def finish(self):
        elapsed = self.clock() - self.started_at
        self._send_text(
            f"{self.frame_count}/{self.frame_count} frames complete | "
            f"{format_duration(elapsed)}"
        )
