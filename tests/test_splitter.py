from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

import cv2
import numpy as np

from shotsplit import ShotSplitter, __version__
from shotsplit.splitter import _iter_autoshot_batches, _scores_to_analysis, _scores_to_segments, _validate_threshold


ROOT = Path(__file__).resolve().parents[1]


class ShotSplitterUnitTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]

        self.assertEqual(__version__, project["version"])

    def test_scores_to_segments_collapses_boundary_runs(self) -> None:
        scores = np.array([0.0, 0.8, 0.9, 0.0, 0.1, 0.7, 0.0], dtype=np.float32)

        segments = _scores_to_segments(scores, threshold=0.5)

        self.assertEqual(
            segments,
            [
                {"start_frame": 0, "end_frame": 2},
                {"start_frame": 2, "end_frame": 6},
                {"start_frame": 6, "end_frame": 7},
            ],
        )

    def test_scores_to_segments_returns_single_segment_without_boundaries(self) -> None:
        scores = np.array([0.0, 0.1, 0.2], dtype=np.float32)

        self.assertEqual(_scores_to_segments(scores, threshold=0.5), [{"start_frame": 0, "end_frame": 3}])

    def test_scores_to_segments_can_exclude_transition_frames(self) -> None:
        scores = np.array([0.0, 0.8, 0.9, 0.0, 0.1, 0.7, 0.0], dtype=np.float32)

        segments = _scores_to_segments(scores, threshold=0.5, include_transition_frames=False)

        self.assertEqual(
            segments,
            [
                {"start_frame": 0, "end_frame": 1},
                {"start_frame": 3, "end_frame": 5},
                {"start_frame": 6, "end_frame": 7},
            ],
        )

    def test_scores_to_analysis_returns_boundary_runs(self) -> None:
        scores = np.array([0.0, 0.8, 0.9, 0.0, 0.1, 0.7, 0.0], dtype=np.float64)

        analysis = _scores_to_analysis(scores, threshold=0.5, include_transition_frames=True)

        self.assertEqual(analysis["frame_count"], 7)
        self.assertEqual(analysis["threshold"], 0.5)
        self.assertTrue(analysis["include_transition_frames"])
        self.assertEqual(
            analysis["segments"],
            [
                {"start_frame": 0, "end_frame": 2},
                {"start_frame": 2, "end_frame": 6},
                {"start_frame": 6, "end_frame": 7},
            ],
        )
        self.assertEqual(
            analysis["boundaries"],
            [
                {
                    "split_frame": 2,
                    "run_start_frame": 1,
                    "run_end_frame": 2,
                    "peak_frame": 2,
                    "peak_score": 0.9,
                },
                {
                    "split_frame": 6,
                    "run_start_frame": 5,
                    "run_end_frame": 5,
                    "peak_frame": 5,
                    "peak_score": 0.7,
                },
            ],
        )
        self.assertNotIn("frame_scores", analysis)

    def test_scores_to_analysis_can_include_frame_scores(self) -> None:
        scores = np.array([0.0, 0.8, 0.9], dtype=np.float64)

        analysis = _scores_to_analysis(scores, threshold=0.5, include_scores=True)

        self.assertEqual(analysis["frame_scores"], [0.0, 0.8, 0.9])
        self.assertEqual(len(analysis["frame_scores"]), analysis["frame_count"])

    def test_excluding_edge_transition_runs_does_not_return_empty_segments(self) -> None:
        scores = np.array([0.9, 0.1, 0.1, 0.8], dtype=np.float32)

        segments = _scores_to_segments(scores, threshold=0.5, include_transition_frames=False)

        self.assertEqual(segments, [{"start_frame": 1, "end_frame": 3}])

    def test_threshold_validation(self) -> None:
        self.assertEqual(_validate_threshold(0), 0.0)
        self.assertEqual(_validate_threshold(1), 1.0)

        with self.assertRaises(ValueError):
            _validate_threshold(-0.1)
        with self.assertRaises(ValueError):
            _validate_threshold(1.1)
        with self.assertRaises(TypeError):
            _validate_threshold(True)

    def test_autoshot_batching_matches_expected_window_shape(self) -> None:
        frames = np.zeros((51, 27, 48, 3), dtype=np.uint8)

        batches = list(_iter_autoshot_batches(frames))

        self.assertEqual(len(batches), 2)
        self.assertTrue(all(batch.shape == (100, 27, 48, 3) for batch in batches))

    def test_missing_video_fails_before_model_load(self) -> None:
        splitter = ShotSplitter(weights_path="does-not-matter.pth", device="cpu")

        with self.assertRaises(FileNotFoundError):
            splitter.split(ROOT / "missing.mp4")

    def test_empty_video_file_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            writer.release()
            splitter = ShotSplitter(weights_path="does-not-matter.pth", device="cpu")

            with self.assertRaises(ValueError):
                splitter.split(path)


class ShotSplitterSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        (ROOT / "data" / "example.mp4").exists() and (ROOT / "weights" / "ckpt_0_200_0.pth").exists(),
        "example video or AutoShot weights are not available",
    )
    def test_example_video_split_returns_contiguous_segments(self) -> None:
        with ShotSplitter(weights_path=ROOT / "weights" / "ckpt_0_200_0.pth") as splitter:
            segments = splitter.split(ROOT / "data" / "example.mp4", threshold=0.2)

        self.assertIsInstance(segments, list)
        self.assertGreaterEqual(len(segments), 1)
        self.assertEqual(segments[0]["start_frame"], 0)
        self.assertEqual(segments[-1]["end_frame"], 300)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous["end_frame"], current["start_frame"])
        for segment in segments:
            self.assertIsInstance(segment["start_frame"], int)
            self.assertIsInstance(segment["end_frame"], int)
            self.assertLess(segment["start_frame"], segment["end_frame"])


if __name__ == "__main__":
    unittest.main()
