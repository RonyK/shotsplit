# shotsplit

Shot segment range extraction for video files.

The PyPI distribution name and Python import package are both `shotsplit`.
`shotsplit` currently uses AutoShot as its default shot-boundary detection
model.

## Install

```bash
python -m pip install shotsplit
```

Or with `uv`:

```bash
uv pip install shotsplit
```

This installs the package and its runtime dependencies, including PyTorch.
`ShotSplitter()` runs on CPU by default.

## Install With A Specific PyTorch Build

PyPI package metadata cannot encode alternate PyTorch wheel indexes for CPU and
CUDA builds. To force a specific PyTorch build, install PyTorch first, then
install `shotsplit`.

CPU-only PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install shotsplit
```

CUDA 12.8 PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install shotsplit
```

The same sequence works with `python -m pip install ...` if you do not use `uv`.

## Usage

```python
from shotsplit import ShotSplitter, __version__

print(__version__)

with ShotSplitter() as splitter:
    clips = splitter.split("example.mp4", threshold=0.2)

print(clips)
```

`ShotSplitter()` uses CPU by default. To use CUDA, install a CUDA-enabled
PyTorch build and pass `device="cuda"` explicitly:

```python
with ShotSplitter(device="cuda") as splitter:
    clips = splitter.split("example.mp4", threshold=0.2)
```

The result is a list of half-open frame ranges:

```python
[{"start_frame": 0, "end_frame": 100}, {"start_frame": 100, "end_frame": 150}]
```

`start_frame` and `end_frame` are frame indexes. `end_frame` is exclusive.
By default, transition frames are included in the adjacent segments so the
returned segments cover every decoded frame.

To exclude transition frames from the returned segments, pass
`include_transition_frames=False`:

```python
with ShotSplitter() as splitter:
    clips = splitter.split(
        "example.mp4",
        threshold=0.2,
        include_transition_frames=False,
    )
```

For boundary metadata and optional per-frame scores, use `analyze()`:

```python
with ShotSplitter() as splitter:
    result = splitter.analyze(
        "example.mp4",
        threshold=0.2,
        include_transition_frames=False,
        include_scores=True,
    )

print(result["segments"])
print(result["boundaries"])
```

`boundaries` contains thresholded AutoShot score runs:

```python
[
    {
        "split_frame": 100,
        "run_start_frame": 98,
        "run_end_frame": 102,
        "peak_frame": 101,
        "peak_score": 0.91,
    }
]
```

`run_start_frame` and `run_end_frame` are inclusive. When
`include_transition_frames=False`, these transition run frames are omitted from
`segments`. If `include_scores=True`, `frame_scores` contains one score per
decoded frame.

The default AutoShot weights are included in the package. To use a different
checkpoint, pass `weights_path`.
