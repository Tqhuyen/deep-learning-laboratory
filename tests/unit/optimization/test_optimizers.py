from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from laboratory.optimization.muon import Muon, muon_parameter_groups, newton_schulz
from laboratory.optimization.optimizers import make_optimizer


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 8)
        self.hidden = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 10)


@pytest.mark.parametrize(
    "name, expected_type",
    [("muon", Muon), ("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
)
def test_make_optimizer_chooses_the_named_optimizer(name, expected_type):
    optimizer = make_optimizer(nn.Linear(3, 2), name=name, learning_rate=0.01, weight_decay=0.1)
    assert isinstance(optimizer, expected_type)
    assert all(group["lr"] == 0.01 and group["weight_decay"] == 0.1 for group in optimizer.param_groups)


def test_make_optimizer_defaults_to_muon():
    assert isinstance(make_optimizer(nn.Linear(3, 2)), Muon)


def test_make_optimizer_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer(nn.Linear(3, 2), name="lbfgs")


def test_muon_gets_only_hidden_matrices():
    model = TinyLanguageModel()
    groups = {group["use_muon"]: group["params"] for group in muon_parameter_groups(model)}
    assert [id(p) for p in groups[True]] == [id(model.hidden.weight)]
    assert {id(p) for p in groups[False]} == {id(p) for p in model.parameters()} - {id(model.hidden.weight)}


def test_output_layer_tied_to_the_embedding_stays_with_adamw():
    model = TinyLanguageModel()
    model.output = nn.Linear(8, 10, bias=False)
    model.output.weight = model.embedding.weight
    groups = {group["use_muon"]: group["params"] for group in muon_parameter_groups(model)}
    assert id(model.embedding.weight) not in {id(p) for p in groups[True]}
    assert [id(p) for p in groups[False]].count(id(model.embedding.weight)) == 1


@pytest.mark.parametrize("shape", [(16, 32), (32, 16), (4, 8, 3, 3)])
def test_muon_update_is_orthogonal(shape):
    torch.manual_seed(0)
    matrix = torch.randn(shape).reshape(shape[0], -1)
    singular_values = torch.linalg.svdvals(newton_schulz(matrix))
    assert singular_values.min() > 0.6 and singular_values.max() < 1.25
