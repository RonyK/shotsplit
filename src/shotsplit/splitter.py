from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .autoshot_model import TransNetV2Supernet


DEFAULT_THRESHOLD = 0.296
DEFAULT_WEIGHTS_RESOURCE = ("weights", "ckpt_0_200_0.pth")
DEFAULT_FRAME_SIZE = (48, 27)

ShotSegment = dict[str, int]
ShotBoundary = dict[str, int | float]
ShotAnalysis = dict[str, Any]


class ShotSplitter:
    """AutoShot-based shot segment range extractor."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.device = torch.device(device or "cpu")
        self.frame_size = frame_size
        self._model: TransNetV2Supernet | None = None

    def __enter__(self) -> "ShotSplitter":
        self._ensure_model()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._model = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def split(
        self,
        video_path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        include_transition_frames: bool = True,
    ) -> list[ShotSegment]:
        """Return shot segments as half-open frame ranges."""
        threshold = _validate_threshold(threshold)
        include_transition_frames = _validate_bool("include_transition_frames", include_transition_frames)
        frames = self._read_frames(video_path)
        scores = self._predict_boundary_scores(frames)
        return _scores_to_segments(scores, threshold, include_transition_frames=include_transition_frames)

    def analyze(
        self,
        video_path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        include_transition_frames: bool = True,
        include_scores: bool = False,
    ) -> ShotAnalysis:
        """Return shot segments, boundary runs, and optional frame scores."""
        threshold = _validate_threshold(threshold)
        include_transition_frames = _validate_bool("include_transition_frames", include_transition_frames)
        include_scores = _validate_bool("include_scores", include_scores)
        frames = self._read_frames(video_path)
        scores = self._predict_boundary_scores(frames)
        return _scores_to_analysis(
            scores,
            threshold,
            include_transition_frames=include_transition_frames,
            include_scores=include_scores,
        )

    def _ensure_model(self) -> TransNetV2Supernet:
        if self._model is not None:
            return self._model

        model = TransNetV2Supernet().to(self.device)
        with self._weights_path() as weights_path:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        model_state = model.state_dict()
        compatible_state = {key: value for key, value in state_dict.items() if key in model_state}
        missing_keys, _ = model.load_state_dict(compatible_state, strict=False)
        if missing_keys:
            missing = ", ".join(missing_keys[:5])
            raise RuntimeError(f"AutoShot weights are missing model parameters: {missing}")

        model.eval()
        self._model = model
        return model

    @contextmanager
    def _weights_path(self) -> Iterator[Path]:
        if self.weights_path is not None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"AutoShot weights not found: {self.weights_path}")
            yield self.weights_path
            return

        resource = files("shotsplit").joinpath(*DEFAULT_WEIGHTS_RESOURCE)
        if not resource.is_file():
            raise FileNotFoundError("Packaged AutoShot weights not found.")

        with as_file(resource) as path:
            yield Path(path)

    def _read_frames(self, video_path: str | Path) -> np.ndarray:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        width, height = self.frame_size
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Video file contains no decodable frames: {path}")

        return np.stack(frames, axis=0).astype(np.uint8, copy=False)

    def _predict_boundary_scores(self, frames: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        predictions: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in _iter_autoshot_batches(frames):
                tensor = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...])
                tensor = tensor.to(device=self.device, dtype=torch.float32)

                logits = model(tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]

                scores = torch.sigmoid(logits[0]).squeeze(-1)
                predictions.append(scores[25:75].detach().cpu().numpy())

        return np.concatenate(predictions, axis=0)[: len(frames)]


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("net"), dict):
        return checkpoint["net"]
    if isinstance(checkpoint, dict) and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise RuntimeError("Unsupported AutoShot checkpoint format.")


def _validate_threshold(threshold: float) -> float:
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a float between 0 and 1.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be a finite float between 0 and 1.")
    return threshold


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def _iter_autoshot_batches(frames: np.ndarray):
    remainder = 50 - len(frames) % 50
    if remainder == 50:
        remainder = 0

    padded = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (remainder + 25), axis=0)
    for start in range(0, len(padded) - 50, 50):
        yield padded[start : start + 100]


def _scores_to_analysis(
    scores: np.ndarray,
    threshold: float,
    *,
    include_transition_frames: bool = True,
    include_scores: bool = False,
) -> ShotAnalysis:
    score_array = np.asarray(scores)
    boundaries = _scores_to_boundaries(score_array, threshold)
    analysis: ShotAnalysis = {
        "frame_count": int(len(score_array)),
        "threshold": float(threshold),
        "include_transition_frames": include_transition_frames,
        "segments": _segments_from_boundaries(
            int(len(score_array)),
            boundaries,
            include_transition_frames=include_transition_frames,
        ),
        "boundaries": boundaries,
    }
    if include_scores:
        analysis["frame_scores"] = [float(score) for score in score_array]
    return analysis


def _scores_to_segments(
    scores: np.ndarray,
    threshold: float,
    *,
    include_transition_frames: bool = True,
) -> list[ShotSegment]:
    boundaries = _scores_to_boundaries(scores, threshold)
    return _segments_from_boundaries(
        int(len(scores)),
        boundaries,
        include_transition_frames=include_transition_frames,
    )


def _scores_to_boundaries(scores: np.ndarray, threshold: float) -> list[ShotBoundary]:
    frame_count = int(len(scores))
    if frame_count == 0:
        return []

    score_array = np.asarray(scores)
    positive = score_array >= threshold
    boundaries: list[ShotBoundary] = []

    index = 0
    while index < frame_count:
        if not positive[index]:
            index += 1
            continue

        run_start = index
        while index + 1 < frame_count and positive[index + 1]:
            index += 1
        run_end = index

        run_scores = score_array[run_start : run_end + 1]
        peak_frame = run_start + int(np.argmax(run_scores))
        boundaries.append(
            {
                "split_frame": int((run_start + run_end) // 2 + 1),
                "run_start_frame": int(run_start),
                "run_end_frame": int(run_end),
                "peak_frame": int(peak_frame),
                "peak_score": float(score_array[peak_frame]),
            }
        )

        index += 1

    return boundaries


def _segments_from_boundaries(
    frame_count: int,
    boundaries: list[ShotBoundary],
    *,
    include_transition_frames: bool,
) -> list[ShotSegment]:
    if frame_count == 0:
        return []

    if not include_transition_frames:
        return _segments_excluding_transition_frames(frame_count, boundaries)

    split_points = [0]
    split_points.extend(
        int(boundary["split_frame"])
        for boundary in boundaries
        if 0 < int(boundary["split_frame"]) < frame_count
    )
    split_points.append(frame_count)

    segments: list[ShotSegment] = []
    for start, end in zip(split_points, split_points[1:]):
        if start < end:
            segments.append({"start_frame": int(start), "end_frame": int(end)})
    return segments


def _segments_excluding_transition_frames(
    frame_count: int,
    boundaries: list[ShotBoundary],
) -> list[ShotSegment]:
    segments: list[ShotSegment] = []
    cursor = 0

    for boundary in boundaries:
        run_start = int(boundary["run_start_frame"])
        run_end = int(boundary["run_end_frame"])
        if cursor < run_start:
            segments.append({"start_frame": int(cursor), "end_frame": int(run_start)})
        cursor = max(cursor, run_end + 1)

    if cursor < frame_count:
        segments.append({"start_frame": int(cursor), "end_frame": int(frame_count)})

    return segments
