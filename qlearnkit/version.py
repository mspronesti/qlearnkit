"""Qlearnkit version information"""

try:
    from importlib import metadata
except ImportError:
    # for Python < 3.8
    import importlib_metadata as metadata


__version__ = "unknown"
try:
    # set __version__ from package metadata at runtime
    __version__ = metadata.version("qlearnkit")
except metadata.PackageNotFoundError:
    # package not installed
    pass
