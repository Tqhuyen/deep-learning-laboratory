"""Deep Learning Laboratory.

A small, readable collection of the pieces every deep learning paper is built
from, grouped by the role they play in a model:

* :mod:`laboratory.blocks` — reusable components: attention, positional
  encodings, normalisation, feed-forward blocks.
* :mod:`laboratory.models` — assembled architectures: transformer (encoder,
  decoder, GPT, sequence-to-sequence) and recurrent networks.
* :mod:`laboratory.objectives` — losses.
* :mod:`laboratory.data` — tokenization and batching.
* :mod:`laboratory.training` — training loop, schedules, metrics, checkpoints.
* :mod:`laboratory.experiment` — conventions for experiments: tiers, seeds,
  output layout.

Design rules: no hidden magic, shapes documented, masks explicit, and every
non-obvious equation in a docstring. Experiments live under ``experiments/``
and always use the same tier system so they run on a laptop.
"""

from . import blocks, data, experiment, models, objectives, training, utilities
from .blocks import (
    ALiBi,
    FeedForward,
    LayerNorm,
    LearnedPositionalEmbedding,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
    SwiGLU,
    apply_rotary,
    make_ffn,
    make_norm,
    repeat_kv,
    scaled_dot_product_attention,
)
from .data import (
    BPETokenizer,
    CharTokenizer,
    LanguageModelDataset,
    Seq2SeqCollator,
    Seq2SeqDataset,
    WordTokenizer,
    load_tokenizer,
    pad_sequences,
    random_lm_batch,
    save_tokenizer,
    train_val_split,
)
from .models import (
    DecoderLayer,
    EncoderLayer,
    GPT,
    GRU,
    GRUCell,
    LSTM,
    LSTMCell,
    RNN,
    RNNCell,
    Seq2SeqTransformer,
    TransformerDecoder,
    TransformerEncoder,
)
from .objectives import token_cross_entropy
from .training import (
    RunningMean,
    Trainer,
    accuracy,
    cosine_warmup,
    load_checkpoint,
    noam,
    save_checkpoint,
)
from .utilities import (
    causal_mask,
    count_parameters,
    get_device,
    lengths_mask,
    set_seed,
)

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
    "RunningMean",
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
    "make_ffn",
    "make_norm",
    "noam",
    "pad_sequences",
    "random_lm_batch",
    "repeat_kv",
    "save_checkpoint",
    "save_tokenizer",
    "scaled_dot_product_attention",
    "set_seed",
    "token_cross_entropy",
    "train_val_split",
]
