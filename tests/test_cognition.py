"""Unit tests for the predictive thought loop (cognition.py) using fakes.

The tick logic is tested in isolation: organs are replaced by callables/fakes,
no real graph. We check: construction, prediction error, role-stop,
single-step thought, multi-step composition to a fixed point, protection against looping.
"""
from cognition import CognitiveCycle, SettleResult, ThoughtTrace


# ---------- organ fakes ----------
class FakePFC:
    def __init__(self):
        self.goal = None
        self.steps = 0

    def set_goal(self, tokens, metadata=None):
        self.goal = tuple(tokens)

    def step(self):
        self.steps += 1

    def add_retrieval_result(self, episode_words, query_words):
        return True

    def get_multi_hop_cues(self, query_words):
        return set(query_words)


class FakeEpisode:
    def __init__(self, roles):
        self._roles = set(roles)

    def has_role(self, role):
        return role in self._roles

    @property
    def input_neurons(self):
        return frozenset()


class RoleEpisode:
    """Episode with role location=<loc> and the given words."""
    def __init__(self, words, location):
        self._words = frozenset(words)
        self._loc = location

    @property
    def input_neurons(self):
        return self._words

    def has_role(self, role):
        return role == "location"

    def get_role(self, role):
        return frozenset({self._loc}) if role == "location" else frozenset()


class HopPFC:
    """PFC fake: carries the new entity from the last episode into the next cue."""
    def __init__(self):
        self.goal = None
        self.carried = set()

    def set_goal(self, tokens, metadata=None):
        self.goal = tuple(tokens)
        self.carried = set()

    def step(self):
        pass

    def add_retrieval_result(self, episode_words, query_words):
        new = {w for w in episode_words if w not in query_words}
        self.carried |= new
        return bool(new)

    def get_multi_hop_cues(self, query_words):
        return set(query_words) | self.carried


def _cycle(**over):
    deps = dict(
        pfc=FakePFC(),
        settle_fn=lambda cue: SettleResult(settled=set(), primary_episode=None),
        predict_fn=lambda cue: set(),
        expected_roles_fn=lambda q: ["location"],
        readout_fn=lambda res, qw, qc: "ok",
        parse_fn=lambda q: ({"a"}, {"a"}, None),
    )
    deps.update(over)
    return CognitiveCycle(**deps)


# ---------- Task 1 ----------
def test_construct_and_defaults():
    c = _cycle()
    assert c.max_ticks == 6
    assert isinstance(c.trace, ThoughtTrace)


# ---------- Task 2 ----------
def test_prediction_error_is_set_difference():
    surprise, miss = CognitiveCycle._prediction_error({"b", "c"}, {"c", "d"})
    assert surprise == {"d"}   # active, but not predicted
    assert miss == {"b"}       # predicted, but silent


# ---------- Task 3 ----------
def test_role_goal_met():
    c = _cycle()
    assert c._role_goal_met(FakeEpisode({"location"}), ["location"]) is True
    assert c._role_goal_met(FakeEpisode({"agent"}), ["location"]) is False
    assert c._role_goal_met(None, ["location"]) is False


# ---------- Task 4 ----------
def test_single_tick_confident_answer():
    ep = FakeEpisode({"location"})
    c = _cycle(
        predict_fn=lambda cue: {"paris"},
        settle_fn=lambda cue: SettleResult({"paris"}, primary_episode=ep),
        readout_fn=lambda res, qw, qc: "paris",
        parse_fn=lambda q: ({"france", "capital"}, {"france", "capital"}, None),
    )
    assert c.think("what is the capital of france") == "paris"
    assert c.trace.ticks == 1
    assert c.trace.stopped_by == "settled"


# ---------- Task 5 ----------
def test_transitive_composition_to_fixpoint():
    kb = {
        frozenset({"a"}): RoleEpisode({"a", "in", "b"}, "b"),
        frozenset({"b"}): RoleEpisode({"b", "in", "c"}, "c"),
    }

    def settle(cue):
        for subj in ("a", "b"):
            if subj in cue:
                ep = kb.get(frozenset({subj}))
                if ep and ep.get_role("location") <= cue:   # already resolved — skip
                    continue
                if ep:
                    return SettleResult(set(ep.input_neurons), primary_episode=ep)
        last = kb[frozenset({"b"})]
        return SettleResult(set(last.input_neurons), primary_episode=last)

    c = CognitiveCycle(
        pfc=HopPFC(),
        settle_fn=settle,
        predict_fn=lambda cue: set(),
        expected_roles_fn=lambda q: ["location"],
        readout_fn=lambda res, qw, qc: sorted(res.primary_episode.get_role("location"))[0],
        parse_fn=lambda q: ({"a", "where"}, {"a", "where"}, None),
        max_ticks=6,
    )
    assert c.think("where is a") == "c"          # inferred the unstated
    assert c.trace.stopped_by == "fixpoint"
    assert c.trace.ticks >= 2


# ---------- Task 6 ----------
def test_cyclic_graph_stops_by_budget():
    ping = {"a": {"a", "in", "b"}, "b": {"b", "in", "a"}}
    state = {"n": 0}

    def settle(cue):
        state["n"] += 1
        words = ping["a"] if state["n"] % 2 else ping["b"]
        ep = RoleEpisode(words, "b")
        return SettleResult(set(words) | {f"tick{state['n']}"}, primary_episode=ep)

    c = CognitiveCycle(
        pfc=HopPFC(), settle_fn=settle, predict_fn=lambda cue: set(),
        expected_roles_fn=lambda q: ["nonexistent"],
        readout_fn=lambda res, qw, qc: "x",
        parse_fn=lambda q: ({"a"}, {"a"}, None), max_ticks=4,
    )
    c.think("where is a")
    assert c.trace.ticks == 4
    assert c.trace.stopped_by == "budget"


def test_collapse_returns_unknown():
    c = _cycle(settle_fn=lambda cue: SettleResult(set(), primary_episode=None))
    assert c.think("where is a") == "I don't know."
    assert c.trace.stopped_by == "collapse"
