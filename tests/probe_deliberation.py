"""Probe-demo of layer 3: the basal ganglia learn ON THEIR OWN when to think (real RPE).

Demonstrates end-to-end gate operation: empty -> reflex; reflex failure in context
(DA-dip) -> switch to deliberation and hold; an easy context is left untouched.
The gate logic is covered by unit tests (tests/test_deliberation.py); this is a live scenario.

Run:  uv run python tests/probe_deliberation.py
"""
import train


def show(gate, ctx, label):
    print(f"  [{label}] context={ctx!r} -> select={gate.select(ctx)} "
          f"(value reflex={gate.value(ctx, 'reflex')}, deliberate={gate.value(ctx, 'deliberate')})")


def main() -> None:
    g = train.DELIBERATION_GATE
    hard = train._deliberation_context("where is the key")   # multi-step locative
    easy = train._deliberation_context("what is a dog")      # direct fact

    print("=== Deliberation gate — learning to decide reflex vs deliberate ===")
    show(g, hard, "start")                       # reflex (default)

    # reflex fails on the hard context -> DA-dip, learning
    g.select(hard)
    rpe = g.learn(realized_success=False)
    print(f"  reflex failed on {hard!r}: RPE(dopamine) = {rpe}")

    show(g, hard, "after failure")               # deliberation
    g.select(hard); g.learn(realized_success=True)
    show(g, hard, "deliberation held")           # holds

    show(g, easy, "easy untouched")              # reflex — no need to think

    hard_deliberates = g.select(hard) == "deliberate"
    easy_reflexes = g.select(easy) == "reflex"
    print(f"\nLearned routing: hard->deliberate={hard_deliberates}, easy->reflex={easy_reflexes}")
    assert hard_deliberates and easy_reflexes, "gate did not learn the expected routing"
    print("OK — the brain learned when to deliberate, from a true reward-prediction error.")


if __name__ == "__main__":
    main()
