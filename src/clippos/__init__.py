"""Clippos — local video clipping engine.

The TF_USE_LEGACY_KERAS guard is set here, before *any* downstream import,
because WhisperX → pytorch-lightning eagerly imports TensorFlow on its
first checkpoint load. Once TF is loaded, the env var has no effect; if
left unset, RetinaFace's Keras-2 functional API is incompatible with TF
2.16+'s Keras-3 default and `RetinaFace.build_model()` raises:
    ValueError: A KerasTensor cannot be used as input to a TensorFlow
    function.
The `tf-keras` package (in our engine extras) provides the Keras-2 shim
this opt-in flag selects. Setting it at package-import time guarantees
correctness regardless of which entry point (CLI, hermes_clip, tests,
or `python -c`) loads us first.
"""
import os as _os

_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
del _os

__all__ = ["__version__"]
__version__ = "0.2.0"
