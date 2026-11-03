from __future__ import annotations

import pytest

from laboratory.agents.ensembles import DebateAgent, SelfConsistencyAgent, majority_vote
from laboratory.agents.loops import DirectAgent
from laboratory.agents.scripted import ScriptedModel


@pytest.mark.parametrize(
    ("answers", "winner"),
    [
        (["391", "400", "391.0"], "391"),
        (["391 crates", "400", "391"], "391 crates"),
        (["Brask", "the town of Brask", "brask."], "Brask"),
        (["a", "b"], "a"),
        ([None, "7", None], "7"),
        ([None, None], None),
    ],
)
def test_majority_vote_groups_equivalent_answers(answers, winner):
    assert majority_vote(answers)[0] == winner


def test_self_consistency_takes_the_majority_of_independent_samples():
    model = ScriptedModel(["Final Answer: 5", "Final Answer: 7", "Final Answer: 5"])
    episode = SelfConsistencyAgent(DirectAgent(model), samples=3).run("q")
    assert (episode.answer, episode.calls) == ("5", 3)
    assert episode.events[0]["votes"] == {"5": 2, "7": 1}


def test_debaters_read_each_other_and_the_last_round_decides():
    model = ScriptedModel(
        [
            "Because of x.\nFinal Answer: 5",
            "Because of y.\nFinal Answer: 7",
            "Agent 2 convinced me.\nFinal Answer: 7",
            "I keep my answer.\nFinal Answer: 7",
        ]
    )
    episode = DebateAgent(DirectAgent(model), debaters=2, rounds=2).run("q")
    assert episode.answer == "7"
    round_two_prompt = model.requests[2]["messages"][1].content
    assert "Agent 2: 7 (Because of y.)" in round_two_prompt
    assert [event.get("answers") for event in episode.events[:2]] == [["5", "7"], ["7", "7"]]


def test_debate_needs_two_debaters():
    with pytest.raises(ValueError):
        DebateAgent(DirectAgent(ScriptedModel([])), debaters=1)
