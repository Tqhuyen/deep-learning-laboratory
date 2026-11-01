"""Deep Learning Laboratory.

Readable, from-scratch implementations of the pieces that almost every deep
learning paper is assembled from:

* :mod:`laboratory.tokenization` — character, word and byte-pair tokenizers.
* :mod:`laboratory.data` — language-model windows and sequence-to-sequence
  batching with teacher-forcing targets.
* :mod:`laboratory.positional_encoding` — sinusoidal, learned, rotary (RoPE)
  and ALiBi position information.
* :mod:`laboratory.attention` — scaled dot-product attention, multi-head,
  grouped-query and cross-attention, with explicit masks.
* :mod:`laboratory.layers` — LayerNorm, RMSNorm, feed-forward blocks, SwiGLU.
* :mod:`laboratory.transformer` — encoder/decoder layers, GPT, sequence-to-
  sequence transformer with greedy decoding.
* :mod:`laboratory.recurrent` — RNN, LSTM and GRU cells written from the
  equations.
* :mod:`laboratory.training` — a task-agnostic training loop and learning-rate
  schedules.
* :mod:`laboratory.experiment` — conventions for experiments: tiers, seeds,
  output layout.

Design rules: no hidden magic, shapes documented, masks explicit, and every
non-obvious equation in a docstring. Experiments live under ``experiments/``
and always use the same tier system so they run on a laptop.
"""

from . import (
    attention,
    data,
    experiment,
    layers,
    positional_encoding,
    recurrent,
    tokenization,
    training,
    transformer,
    utilities,
)
from .attention import MultiHeadAttention, repeat_kv, scaled_dot_product_attention
from .data import (
    LanguageModelDataset,
    Seq2SeqCollator,
    Seq2SeqDataset,
    pad_sequences,
    random_lm_batch,
    train_val_split,
)
from .layers import FeedForward, LayerNorm, RMSNorm, SwiGLU
from .positional_encoding import (
    ALiBi,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    apply_rotary,
)
from .recurrent import GRU, LSTM, RNN, GRUCell, LSTMCell, RNNCell
from .tokenization import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    load_tokenizer,
    save_tokenizer,
)
from .training import (
    Trainer,
    accuracy,
    cosine_warmup,
    load_checkpoint,
    noam,
    save_checkpoint,
)
from .transformer import (
    DecoderLayer,
    EncoderLayer,
    GPT,
    Seq2SeqTransformer,
    TransformerDecoder,
    TransformerEncoder,
)
from .utilities import causal_mask, count_parameters, get_device, lengths_mask, set_seed

__version__ = "0.1.0"

__all__ = [
    "ALiBi",
    "BPETokenizer",
    "CharTokenizer",
    "DecoderLayer",
    "EncoderLayer",
    "FeedForward",
    "GPT",
    "GRU",
    "GRUCell",
    "LSTM",
    "LSTMCell",
    "LanguageModelDataset",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "RNN",
    "RNNCell",
    "RMSNorm",
    "RotaryEmbedding",
    "Seq2SeqCollator",
    "Seq2SeqDataset",
    "Seq2SeqTransformer",
    "SinusoidalPositionalEncoding",
    "SwiGLU",
    "Trainer",
    "TransformerDecoder",
    "TransformerEncoder",
    "WordTokenizer",
    "accuracy",
    "apply_rotary",
    "causal_mask",
    "cosine_warmup",
    "count_parameters",
    "get_device",
    "lengths_mask",
    "load_checkpoint",
    "load_tokenizer",
    "noam",
    "pad_sequences",
    "random_lm_batch",
    "repeat_kv",
    "save_checkpoint",
    "save_tokenizer",
    "scaled_dot_product_attention",
    "set_seed",
    "train_val_split",
]
