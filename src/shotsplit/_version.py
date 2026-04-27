from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("shotsplit")
except PackageNotFoundError:
    __version__ = "0.1.0"
