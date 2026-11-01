"""Deep Learning Laboratory.

The library is organised by **pipeline stage**, with one directory for each
stage every experiment passes through, plus shared components:

``datasets`` -> ``preprocessing`` -> ``models`` -> ``losses`` ->
``optimization`` -> ``training`` -> ``evaluation``, and at inference time
``inference`` -> ``postprocessing``.

* :mod:`laboratory.blocks` — shared model components: attention, positional
  encodings, normalisation, feed-forward blocks.
* :mod:`laboratory.datasets` — indexable examples and splits.
* :mod:`laboratory.preprocessing` — tokenizers, padding, collation.
* :mod:`laboratory.models` — assembled architectures.
* :mod:`laboratory.losses` — objectives.
* :mod:`laboratory.optimization` — optimizers and schedules.
* :mod:`laboratory.training` — loop and checkpoints.
* :mod:`laboratory.evaluation` — metrics and exact match.
* :mod:`laboratory.inference` — sampling and greedy decoding.
* :mod:`laboratory.postprocessing` — ids to task output.
* :mod:`laboratory.experiment` — tiers, seeds, output layout.
* :mod:`laboratory.utilities` — seeds, devices, masks, running averages.

Architecture-specific differences (RNN vs LSTM, transformer vs CNN) are files
inside ``models/``; a different loss, optimizer or preprocessing step is a
different module in the corresponding stage directory, selected by the
experiment. Nothing is duplicated per architecture.
"""

from . import (
    blocks,
    datasets,
    evaluation,
    experiment,
    inference,
    losses,
    models,
    optimization,
    postprocessing,
    preprocessing,
    training,
    utilities,
)
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
from .datasets import (
    LanguageModelDataset,
    Seq2SeqDataset,
    SentimentDataset,
    read_sentiment_tsv,
    stratified_split,
    train_val_split,
)
from .evaluation import (
    accuracy,
    bootstrap_confidence_interval,
    decay_slope,
    filter_normalized_direction,
    exact_match,
    holm_adjust,
    loss_surface,
    mcnemar_exact,
    orthogonalize,
    paired_bootstrap_difference,
    paired_bootstrap_p_value,
    perplexity,
    shifted_state_dict,
    timestep_gradient_norms,
)
from .inference import generate, greedy_decode, next_token
from .losses import classification_cross_entropy, token_cross_entropy
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
    RecurrentLanguageModel,
    RecurrentSequenceClassifier,
    Seq2SeqTransformer,
    TransformerDecoder,
    TransformerEncoder,
    language_model_parameters,
    match_classifier_hidden_size,
    match_language_model_hidden_size,
    recurrent_stack_parameters,
    sequence_classifier_parameters,
)
from .optimization import cosine_warmup, make_optimizer, noam
from .postprocessing import trim_at_eos
from .preprocessing import (
    BPETokenizer,
    CharTokenizer,
    Seq2SeqCollator,
    classification_collate,
    WordTokenizer,
    load_tokenizer,
    pad_sequences,
    random_lm_batch,
    save_tokenizer,
)
from .training import Trainer, load_checkpoint, save_checkpoint
from .utilities import (
    RunningMean,
    causal_mask,
    count_parameters,
    get_device,
    lengths_mask,
    set_seed,
)

__version__ = "0.1.0"

__all__ = [
    "timestep_gradient_norms",
    "stratified_split",
    "sequence_classifier_parameters",
    "SentimentDataset",
    "RecurrentSequenceClassifier",
    "RecurrentLanguageModel",
    "recurrent_stack_parameters",
    "read_sentiment_tsv",
    "paired_bootstrap_difference",
    "paired_bootstrap_p_value",
    "mcnemar_exact",
    "orthogonalize",
    "match_language_model_hidden_size",
    "match_classifier_hidden_size",
    "language_model_parameters",
    "holm_adjust",
    "loss_surface",
    "decay_slope",
    "filter_normalized_direction",
    "classification_collate",
    "classification_cross_entropy",
    "bootstrap_confidence_interval",
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
    "exact_match",
    "generate",
    "get_device",
    "greedy_decode",
    "lengths_mask",
    "load_checkpoint",
    "load_tokenizer",
    "make_ffn",
    "make_norm",
    "make_optimizer",
    "next_token",
    "noam",
    "pad_sequences",
    "perplexity",
    "shifted_state_dict",
    "random_lm_batch",
    "repeat_kv",
    "save_checkpoint",
    "save_tokenizer",
    "scaled_dot_product_attention",
    "set_seed",
    "token_cross_entropy",
    "train_val_split",
    "trim_at_eos",
]
