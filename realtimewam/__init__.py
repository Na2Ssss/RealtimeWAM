"""RealtimeWAM: parallel OpenWAM inference with optional adaptive computation."""

from .runtime import OpenWAM, Runtime as accelerate

__all__ = ["OpenWAM", "accelerate"]
