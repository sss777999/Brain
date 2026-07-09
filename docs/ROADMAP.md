# Brain Model — Development Roadmap

## Scientific Foundation

### Hiersche, Saygin & Osher (2026) — "Connectivity and function are coupled across cognitive domains"
*Network Neuroscience, DOI: 10.1162/NETN.a.504*

**Key findings integrated into this roadmap:**
1. **Connectivity Fingerprints**: Each brain region has a unique wiring pattern that determines its specialized function. Function is a *consequence* of connectivity topology.
2. **Connectivity predicts activation**: Wiring patterns reliably predict whether a region will be active or inactive during different cognitive tasks.
3. **Hierarchical coupling**: Higher-level regions (executive function, memory — PFC, hippocampus) show *tighter* connectivity-function coupling than sensory areas.
4. **Developmental timeline**: Higher-level coupling takes years to develop through continuous use. Sensory/social skills develop faster.
5. **Implication**: Connectivity is a *fundamental organizational principle* governing brain function.

### How this changes our architecture

Currently the model has a "flat" topology — all neurons are equal, all connections follow the same plasticity rules. The article proves this is biologically incorrect. **Connection topology must determine function, and different regions must have different plasticity dynamics.**

---

## Step 7: Thinking loop — predictive cognitive cycle (northstar "all like a brain")

Northstar: make the model reason "all like a brain" in three layers. **Layer 1 (predictive cognitive cycle) is DONE.** Layers 2 and 3 are TODO.

### Biology
**Predictive processing (Rao & Ballard, 1999; Friston active inference):** cognition is a loop where a top-down prediction meets bottom-up evidence and the *prediction error* drives the next step. Thinking is not a scripted number of steps — it runs until the error settles.

**Attractor dynamics (Hopfield, 1982):** "settling" the current state onto a stored memory is a fixpoint of recurrent CA3 excitation. A thought ends when the state stops changing.

**PFC recurrent working memory:** the prefrontal cortex holds and updates a state across the loop (recurrent excitation + decay), so each tick can depend on the last.

### Implementation (layer 1 — DONE)
New module `cognition.py` (class `CognitiveCycle`), wired to the real organs via `cognition_adapters.py`, reachable through `ask(question, mode="emergent")` in `train.py`. The old scripted path stays as `ask(mode="legacy")` and is **still the default**.

One "thought tick" is the missing recurrent loop `state_{t+1} = settle(memory, state_t)`:
```
one thought tick:
  P_t = top-down prediction   # a SET of neurons reachable from the cue
                              # via strong existing USED / MYELINATED edges
  S_t = CA3.settle(cue)       # bottom-up attractor settling
  surprise = S_t \ P_t        # prediction error = structural SET DIFFERENCE
  miss     = P_t \ S_t
  cue = surprise              # the surprising neurons become the next cue
  PFC.step()                  # runs its recurrent excitation / decay
  stop when: fixpoint reached / role-goal met / collapse / tick budget (~6)
```
The number of thought steps is *emergent*, not scripted.

**Forbidden-compliance (kept deliberately un-connectionist):** no weights, no gradients, no backprop, no softmax, no distance metric, no global search, and no LLM anywhere in the reasoning path. Prediction is a *set*, error is a *set difference*, and "minimization" is attractor settling.

### Evidence
- 13/13 unit tests green, including transitive composition (`a->b`, `b->c` ⇒ `c`) settling to a fixpoint on a synthetic graph, with no verb-specific code.
- On the **live trained model**: the facts "zorp is in blen" and "blen is in quix" were learned *separately* (never in one sentence). `ask("where is zorp", mode="emergent")` composes `zorp -> blen -> quix` and answers **"blen is in quix"** (the deep location), whereas legacy returns only the single hop **"is in blen"**. This is the first genuine multi-step reasoning in the project.
- Legacy regression intact: CURRICULUM 98.0%, STRICT 100.0%, QA AVG 99.0% — unchanged.

### Honest limits
- Emergent mode is **not the default yet** (legacy is).
- A composition probe (`tests/probe_composition.py`) scores 1/10 on the live model. 9 of those 10 failures are a **pre-existing storage bug**, not a failure of the cycle: `train.py:1583` `HIPPOCAMPUS.encode` returns `None` deterministically on some "X is in Y" sentences, so those facts never store. Verified: when facts store, the cycle composes; when `encode` returns `None`, both modes say "I don't know".
- Prediction quality of `P_t` and population read-out still need tuning.

### Layers 2 and 3 (TODO)
- **Layer 2 — inference during sleep:** sharp-wave-ripple (SWR) replay composes a *new* edge `A-C` from `A-B` + `B-C` over multiple offline cycles, so the composition becomes a stored one-hop fact rather than being recomputed each query.
- **Layer 3 — basal ganglia RPE gating:** a basal-ganglia loop with real reward-prediction-error learning (cf. Schultz, Step 5) so the brain itself decides whether to answer reflexively or to deliberate.

Spec: `docs/superpowers/specs/2026-07-09-predictive-cognitive-cycle-design.md` · plan: `docs/superpowers/plans/2026-07-09-predictive-cognitive-cycle.md` · notes: `implementation-notes.md`.

---

## Step 3: Community Outreach

### Platforms
- **Hacker News** (Show HN): Engineering focus. Problem: LLMs don't think. Solution: separate storage (Hippocampus), logic (PFC), verbalization (Broca's area).
- **Reddit** (r/MachineLearning, r/neuroscience, r/LocalLLaMA): Ask biologists directly: "I implemented CA3 Attractor Dynamics and PFC Working Memory in code. Where am I wrong biologically?"
- **LessWrong**: Philosophy + AGI Alignment. Observations: thinking faster than speech, knowledge ≠ episodes, source memory = trust levels.

### Key selling points
- No backpropagation for episodic memory
- Source Memory prevents hallucinations
- Hiersche et al. validates our connectivity-determines-function principle
- Open source, seeking expert feedback

---

## Step 4: LTD / Forgetting (Synaptic Homeostasis)

### Biology
**Tononi & Cirelli (2006)** — Synaptic Homeostasis Hypothesis (SHY):
- During wakefulness, synapses strengthen (LTP dominates)
- During slow-wave sleep (NREM), global downscaling occurs
- Weak/noise synapses fall below threshold → pruned
- Strong/meaningful synapses survive but at lower absolute level
- This clears capacity and energy for next day's learning

**Hiersche integration**: LTD is the mechanism that *sharpens* connectivity fingerprints. Without forgetting, all regions would blur into uniform high connectivity. Pruning creates the sparse, specific patterns that define each region's function.

### Implementation Plan

#### 4.1 Episode Aging (`episode.py`)
```
Add to Episode:
  - strength: float = 1.0          # Decays over sleep cycles
  - last_accessed_time: int = 0     # When last used in pattern completion
  - access_count: int = 0           # How many times retrieved
```

#### 4.2 Global Downscaling in NREM Sleep (`hippocampus.py`)
```
During sleep_consolidation():
  1. NREM phase: iterate ALL episodes
  2. Apply global decay: episode.strength *= DECAY_FACTOR (e.g., 0.95)
  3. Access bonus: recently accessed episodes resist decay
  4. Source protection: LEARNING/EXPERIENCE episodes decay slower than MEDIA/NARRATIVE
  5. Prune threshold: if strength < MIN_STRENGTH → remove episode
```

#### 4.3 Connection Pruning (`connection.py`)
```
Extend mark_unused_cycle():
  - Currently: only non-MYELINATED connections get pruned after THRESHOLD_TO_PRUNE cycles
  - Add: USED connections that lose all associated episodes → accelerated pruning
  - Add: connections with context_diversity=1 decay faster (episodic, not semantic)
```

#### 4.4 Expected Impact
- Memory footprint reduction (currently ~76k episodes, growing unbounded)
- Faster retrieval (fewer candidates to score)
- Better signal-to-noise in CA3 pattern completion
- Biologically correct: knowledge consolidates, noise disappears

---

## Step 5: NeuromodulatorSystem

### Biology
**Gerstner et al. (2018)** — Three-Factor Learning:
- STDP creates eligibility trace
- Neuromodulator converts trace to actual plasticity
- Without neuromodulator signal → eligibility decays, no learning

**Schultz (1998)** — Dopamine = Reward Prediction Error:
- Expected reward → no DA
- Unexpected reward → DA burst → strengthened learning
- Expected reward absent → DA dip → weakened connections

**Sara (2009)** — Norepinephrine = Novelty/Alertness:
- Novel stimuli → NE release → narrowed attention focus
- Familiar stimuli → low NE → broad, relaxed processing

**Hasselmo (2006)** — Acetylcholine = Encoding vs Retrieval:
- High ACh → encoding mode (new memories)
- Low ACh → retrieval mode (recall existing)

**Hiersche integration**: Different brain regions have different neuromodulator receptor densities. PFC has dense DA receptors (reward-guided learning). Hippocampus has dense ACh receptors (encoding/retrieval switching). This is WHY higher-order regions develop differently — they respond to different chemical signals.

### Implementation Plan

#### 5.1 Global Modulator State (`neuromodulation.py` — new file)
```python
class NeuromodulatorState:
    dopamine: float = 0.5       # Reward/novelty signal
    norepinephrine: float = 0.3  # Alertness/focus
    acetylcholine: float = 0.5   # Encoding vs retrieval
    serotonin: float = 0.5       # Mood/inhibition baseline

    def update_on_query(self, question, confidence):
        """Update modulators based on query processing."""
        # No answer found → NE rises (novelty/stress)
        # Confident answer → DA rises (reward)
        # Question asked → ACh shifts to retrieval mode
```

#### 5.2 Modulator Effects on Components

| Modulator | Target | Effect |
|-----------|--------|--------|
| Dopamine | `connection.py` | Lowers myelination threshold (PKA pathway) |
| Dopamine | `ca3.py` | Strengthens winning attractor (reward consolidation) |
| Norepinephrine | `ca3.py` | Tightens WTA — fewer but stronger candidates |
| Norepinephrine | `activation.py` | Narrows lateral inhibition threshold |
| Acetylcholine | `hippocampus.py` | High=encode new episodes, Low=retrieve existing |
| Serotonin | `pfc.py` | Impulse control — veto power on low-confidence answers |

#### 5.3 Integration with ask() cycle
```
ask(question):
  1. modulators.update_on_query(question)
  2. ACh → retrieval mode
  3. CA3 scoring uses NE to set WTA threshold
  4. If no answer → NE spikes → retry with tighter focus
  5. If answer found → DA signal → consolidate this retrieval path
  6. modulators.decay_to_baseline()
```

---

## Step 6: Sparse Distributed Representations (SDR)

### Biology
**Numenta HTM (Hawkins 2004)** — Hierarchical Temporal Memory:
- Concepts = sparse binary vectors (e.g., 2048 bits, ~40 active = 2% sparsity)
- Similarity = bitwise overlap
- Union = bitwise OR
- Capacity = combinatorial (C(2048,40) ≈ 10^84 unique patterns)

**McClelland et al. (1995)** — Complementary Learning Systems:
- Hippocampus: sparse, pattern-separated (distinct episodes)
- Cortex: overlapping, distributed (generalized knowledge)

**Hiersche integration**: The "connectivity fingerprint" IS an SDR. Each region's identity is defined by its distributed pattern of connections to other regions. Two regions with similar fingerprints have similar functions. This is exactly SDR overlap = semantic similarity.

### Implementation Plan (Architectural Revolution)

#### 6.1 Word → SDR Encoding (`sdr.py` — new file)
```
Replace string tokens with binary vectors:
  word "dog" → SDR(2048 bits, ~40 active)
  word "cat" → SDR(2048 bits, ~40 active, ~15 shared with "dog")

Encoding methods:
  1. Random initial assignment (like random hash)
  2. Semantic overlap via shared features learned during training
  3. Context-dependent: "bank" (river) vs "bank" (money) → different SDRs
```

#### 6.2 Episode Storage as SDR Unions
```
Episode "dog is an animal" → union of word SDRs
  = SDR_dog | SDR_is | SDR_animal
  
Matching: episode_sdr & query_sdr → overlap count
  High overlap = relevant episode
  This replaces set intersection of input_words
```

#### 6.3 Connection Rewiring
```
Currently: Neuron("dog") → Neuron("animal")
SDR model: SDR_dog bits activate SDR_animal bits via learned weights

The connection IS the overlap pattern.
Myelination = stable bit-to-bit mappings
```

#### 6.4 Migration Strategy
- Phase A: Keep current word-based system, add SDR as parallel representation
- Phase B: Use SDR for similarity scoring (replaces set overlap in _score_episodes)
- Phase C: Full SDR-based retrieval (replaces string matching entirely)
- Phase D: Remove string-based system

#### 6.5 Expected Impact
- Natural synonym handling ("dog" ≈ "canine" via bit overlap)
- Graceful degradation (partial matches work naturally)
- Scales to millions of concepts without vocabulary explosion
- True generalization without hardcoded rules

---

## Priority Order

| Priority | Step | Effort | Impact | Dependencies |
|----------|------|--------|--------|-------------|
| 1 | Step 4: LTD/Forgetting | Medium | High — enables scaling | None |
| 2 | Step 5: Neuromodulators | Medium | High — dynamic behavior | Step 4 (decay mechanics) |
| 3 | Step 3: Community Outreach | Low | High — expert feedback | Steps 4-5 implemented |
| 4 | Step 6: SDR | Very High | Revolutionary — true generalization | Steps 4-5 stable |

**Rationale**: LTD first because unbounded memory growth is the immediate scaling bottleneck. Neuromodulators second because they provide the dynamic control framework that all other components need. Outreach third because we want expert feedback AFTER implementing the core biology. SDR last because it's an architectural revolution that requires a stable foundation.
