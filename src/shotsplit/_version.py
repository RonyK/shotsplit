from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("autoshot-splitter")
except PackageNotFoundError:
    __version__ = "0.1.0"

