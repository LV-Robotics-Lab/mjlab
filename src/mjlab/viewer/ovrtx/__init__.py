"""ovrtx (NVIDIA RTX) viewer backend.

Importing this package pulls in the optional ``ovrtx`` / ``usd-core`` stack, so
it is imported lazily (only when ``--viewer ovrtx`` is selected).
"""

from mjlab.viewer.ovrtx.viewer import OvrtxViewer

__all__ = ["OvrtxViewer"]
