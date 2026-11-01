"""Postprocessing: ids to task output.

* :func:`laboratory.postprocessing.trim_at_eos` — drop control tokens.

Add a module per output shape when it is needed: ``boxes.py`` for detection,
``text.py`` for detokenisation. Postprocessing must not depend on the model.
"""

from .sequences import trim_at_eos

__all__ = ["trim_at_eos"]
