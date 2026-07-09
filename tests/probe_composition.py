"""Composition probe set: facts are stored separately, the question requires JOINING them.

This is a proof of reasoning, independent of bAbI: success = correct answer through the circuit
over a chain the model has never seen as a single phrase. A measurement, not a strict gate —
at the start the pass-rate may be low (see risks in the spec).

Run:
    uv run python tests/probe_composition.py
"""
from typing import Callable, List, Tuple

# NEW (nonexistent) tokens: they have NO competing priors from the curriculum,
# so the only way to answer is to JOIN two fresh facts. This is a clean
# measurement of composition on the pretrained model.
# (individual facts, question, expected answer substring)
CHAINS: List[Tuple[List[str], str, str]] = [
    (["zorp is in blen", "blen is in quix"], "where is zorp", "quix"),
    (["florp is in dax", "dax is in wug"], "where is florp", "wug"),
    (["glim is in trell", "trell is in vok"], "where is glim", "vok"),
    (["snod is in prin", "prin is in yex"], "where is snod", "yex"),
    (["brek is in lom", "lom is in fintu"], "where is brek", "fintu"),
    (["kwen is in jorb", "jorb is in muxel"], "where is kwen", "muxel"),
    (["plor is in henk", "henk is in razu"], "where is plor", "razu"),
    (["twix is in gorm", "gorm is in belq"], "where is twix", "belq"),
    (["vunk is in shael", "shael is in dropo"], "where is vunk", "dropo"),
    (["mirp is in colt", "colt is in wexa"], "where is mirp", "wexa"),
]

# How many times to repeat training on each fact: so that the connections reach USED (threshold 5),
# otherwise a one-shot fact yields only NEW connections, invisible to retrieval/prediction.
TRAIN_REPS = 8


def run_probe(train_fn: Callable[[str], None],
              ask_fns: dict) -> dict:
    """Trains the facts of each chain ONCE, then asks with each mode.

    ask_fns: {mode_name: ask_fn}. Returns {mode_name: (passed, total)}.
    Short-circuit answer: the short expected substring is contained in the answer.
    """
    scores = {name: 0 for name in ask_fns}
    trunc = lambda s: (s[:80] + "…") if len(s) > 80 else s
    for facts, question, expected in CHAINS:
        for fact in facts:
            for _ in range(TRAIN_REPS):
                train_fn(fact)
        print(f"\nQ: {question!r} (expected {expected!r})")
        for name, ask_fn in ask_fns.items():
            answer = (ask_fn(question) or "").lower()
            ok = expected.lower() in answer
            scores[name] += int(ok)
            print(f"  [{name:8}] {'PASS' if ok else 'FAIL'}: {trunc(answer)!r}")
    total = len(CHAINS)
    print("\n" + "=" * 50)
    for name, passed in scores.items():
        print(f"Composition probe [{name}]: {passed}/{total} ({100 * passed // max(1, total)}%)")
    return {name: (passed, total) for name, passed in scores.items()}


if __name__ == "__main__":
    import train

    train.load_model_numpy("models/brain_model")

    def train_fn(sentence: str) -> None:
        # FACT mode: an ordinary world fact (not self). Low-level encoder,
        # creates an episode + connections. Robust to the pre-existing enc()→None bug
        # (train.py:1583) when re-training the same sentence (episode dedup).
        try:
            train.train_sentence_with_context(sentence)
        except AttributeError:
            pass  # connections strengthen before episode encoding; dedup-None does not interfere

    run_probe(train_fn, {
        "emergent": lambda q: train.ask(q, mode="emergent"),
        "legacy": lambda q: train.ask(q, mode="legacy"),
    })
