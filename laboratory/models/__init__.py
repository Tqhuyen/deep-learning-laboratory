"""Assembled architectures.

Each module here is a complete model built from :mod:`laboratory.blocks`:

* :mod:`laboratory.models.transformer` — encoder/decoder blocks, GPT, and the
  encoder–decoder sequence-to-sequence model.
* :mod:`laboratory.models.recurrent` — RNN, LSTM and GRU stacks.

Add one file per model family when it exists: ``convnet.py`` for ResNet-style
networks, ``jepa.py`` for a joint-embedding predictive architecture, and so on.
Keep the constructor arguments descriptive (width, depth, heads) so experiment
presets read like the paper's table.
"""

from .recurrent import GRU, LSTM, RNN, GRUCell, LSTMCell, RNNCell
from .transformer import (
    DecoderLayer,
    EncoderLayer,
    GPT,
    Seq2SeqTransformer,
    TransformerDecoder,
    TransformerEncoder,
    build_positional,
    build_rope,
)

__all__ = [
    "DecoderLayer",
    "EncoderLayer",
    "GPT",
    "GRU",
    "GRUCell",
    "LSTM",
    "LSTMCell",
    "RNN",
    "RNNCell",
    "Seq2SeqTransformer",
    "TransformerDecoder",
    "TransformerEncoder",
    "build_positional",
    "build_rope",
]
