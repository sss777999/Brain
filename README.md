# Brain 🧠

A biologically plausible memory model that learns from text using **discrete synaptic states** and **local plasticity rules**. No numeric weights, no gradients, no backpropagation—only mechanisms found in the biological brain.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

**📊 [Full Test Results & Baseline Comparison](docs/RESULTS.md)** — Brain vs TF-IDF/BM25: **+43-49% advantage**

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/sss777999/brain.git
cd brain

# Install dependencies (using uv)
uv sync

# Run tests ((fast, without LLM and without GPT evaluation; baselines: TF-IDF, BM25 for QA, MemNet/NTM for bAbI)
uv run python test_brain.py --no-llm --no-gpt

# Run full tests with Broca's area (LLM verbalization) but without gpt evaluation
uv run python test_brain.py --no-gpt
```

### Requirements

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager
- **[Ollama](https://ollama.ai/)** (optional) — for Broca's area verbalization (`gemma3:4b`)

### Train your own model

```bash
# Train on built-in curriculum (curriculum → preschool → grade1 → FineWeb-Edu)
uv run python train.py

# Model saved to: models/brain_model_*.{npz,pkl}
```

### Use in code

```python
from train import ask, load_model

load_model()
answer = ask("What is the capital of France?")
print(answer)  # "paris"
```

---

## What makes this different?

| Traditional Neural Networks | Brain Model |
|----------------------------|-------------|
| Continuous weights (float32) | Discrete states: NEW → USED → MYELINATED → PRUNE |
| Gradient descent + backprop | Local Hebbian learning + STDP |
| Vector embeddings | Neurons as discrete units |
| Attention matrices | Spreading activation + lateral inhibition |
| Fixed context window | Episodic memory + pattern completion |

**Key biological mechanisms implemented:**
- **STDP** — Spike-Timing Dependent Plasticity for directional connections
- **Four-factor learning** — STDP + eligibility traces + neuromodulators (DA/ACh/NE/5-HT)
- **Hippocampal circuit** — DG (pattern separation) → CA3 (pattern completion) → CA1 (output)
- **PFC working memory** — NMDA-like sustained activity, distractor resistance
- **Basal ganglia** — Action selection via Go/NoGo pathways
- **Sleep consolidation** — Sharp-wave ripples, forward/reverse replay

---

## How it answers questions

```
INPUT: "What is the capital of France?"
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. BROCA (broca.py): Parse question → subject="france", connector="is_a" │
│ 2. PFC (pfc.py): Set goal, load context, classify question type          │
│ 3. BASAL GANGLIA (basal_ganglia.py): Select action (retrieve vs multi_hop)│
│ 4. ACTIVATION (activation.py): Spread through SEMANTIC connections        │
│    - MYELINATED paths conduct first, lateral inhibition, hub penalty     │
│ 5. HIPPOCAMPUS (hippocampus.py + ca3.py): Pattern completion             │
│    - CA3 attractor dynamics: spread → WTA → stability check              │
│    - Source filter: preferred + selective inclusion (Phase 21)           │
│    - Score: query overlap, connections, temporal bonus (P19), roles     │
│    - Connector: string ×5/×0.2 (biased), frozenset ×2 (soft)           │
│    - Unconnected context filter, dedup top-K (Phase 20)                 │
│    - Best episode: ("capital", "france", "paris")                        │
│ 6. CA1 (ca1.py): Output layer, projects to PFC                           │
│ 7. MOTOR OUTPUT (motor_output.py): Filter question words → ["paris"]     │
│ 8. LLM (optional): Grammatical verbalization → "Paris"                   │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT: "Paris"
```

**Spikes ARE used!** `_simulate_spike_pair()` in train.py creates spike_history, applies STDP with eligibility traces, and neuromodulators (DA/ACh/NE/5-HT) modulate plasticity. Connection imports EligibilityTrace, CalciumState, MetaplasticState from spiking.py.

---

## Key principles

### 1. Memory is not a "value", but a PATTERN

- A pattern is a set of neurons and connections that were frequently activated together
- We do NOT store a "value" as a number
- We store: "this pathway/ensemble → pattern X"

### 2. Learning is not weight training, but pathway selection

- Frequently used pathways are stabilized (kept)
- Rarely used pathways disappear (pruning)
- Stable patterns emerge from repeated pathways

### 3. Activation is lightning

> "Activation passes like LIGHTNING in the sky — quickly, along the paths of least resistance"

- The signal goes FORWARD ONLY; there is no backward path
- Myelinated connections conduct faster (deep "grooves")
- Like a ball rolling along a carved landscape

### 4. Myelination is PRECISE KNOWLEDGE

- Not just "confidence after repetition"
- These are fundamental facts about reality
- Examples: "a cat meows" (a healthy cat does not bark), "you will fall if you jump from a height"

### 5. Knowledge expands with experience

- A lack of categories is normal when there is little knowledge
- An adult knows physics not because they read it 1000 times, but because they have enough experience
- The model accumulates knowledge gradually

### 6. Directed connections (STDP)

**STDP (Spike-Timing-Dependent Plasticity)** is a biological mechanism:
- If neuron B fires AFTER neuron A, the A→B connection is strengthened (forward)
- If neuron A fires AFTER neuron B, the B→A connection is strengthened (backward)

```python
class Connection:
    forward_usage: int   # A→B (A was before B)
    backward_usage: int  # B→A (B was before A)
```

Example:
- "cat meows" → strengthens `cat→meows` (forward)
- "meows cat" → strengthens `meows→cat` (forward for that order)

### 7. Connection limit (~7000 per neuron)

- In the real brain a neuron has ~7000 connections, not billions
- Connections are created as needed (Hebbian rule), not upfront
- When the limit is reached, old unused connections are removed

### 8. Word forms emerge naturally

- "cat" and "cats" are different neurons
- But if they often co-occur in text, the connection will strengthen
- No artificial lemmatization is needed—the data will form connections

### 9. Tests describe REQUIRED BEHAVIOR

- Tests are NOT adjusted to match the code
- If a test fails, the problem is in the code, not in the test
- The real check is: train the network and verify what it recalls

---

## Current state

### ✅ What's done

| Component | Status | Description |
|-----------|--------|-------------|
| **CORE** | | |
| Neuron | ✅ | Binary state (active/inactive), no numeric weights |
| Connection | ✅ | Discrete states: NEW → USED → MYELINATED → PRUNE |
| ConnectionType | ✅ | SEMANTIC (ventral) / SYNTACTIC (dorsal) — Dual Stream |
| Activation | ✅ | Propagation like "lightning" along connections |
| **SEMANTIC MEMORY** | | |
| Hebbian rule | ✅ | Connections are created during co-activation |
| STDP | ✅ | Connection directionality (forward_usage / backward_usage) |
| Myelination | ✅ | Consolidation of frequently used pathways |
| Chunking | ✅ | Merging frequent sequences |
| Inhibition | ✅ | Inhibitory neurons suppress weak branches |
| **EMERGENT HIERARCHY** | | |
| find_categories() | ✅ | Categories from graph topology (nodes with many incoming edges) |
| get_related_concepts() | ✅ | Related concepts by connection strength |
| NO IS_A/HAS_PROPERTY | ✅ | Hierarchy emerges implicitly, not explicitly |
| **BIOLOGICAL ATTENTION** | | |
| generate_with_attention() | ✅ | Generation with accumulating context |
| Decay | ✅ | Decay of old activations |
| Hub penalty | ✅ | log(1+n) — Weber–Fechner law |
| Lateral inhibition | ✅ | Top-N strong activations suppress weaker ones |
| Winner-take-all | ✅ | Only winners remain active |
| Seed anchoring | ✅ | The topic (seed) always remains in memory |
| Working memory | ✅ | Limited capacity (~7 items) |
| **HOMEOSTATIC PLASTICITY** | | |
| Sparse coding | ✅ | ~2% active neurons in DG (Rolls et al., 2007) |
| Diluted connectivity | ✅ | Hebbian window of 4 words (not fully connected) |
| Heterosynaptic LTD | ✅ | Weak synapses weaken while strong ones strengthen |
| Synaptic Scaling | ✅ | Homeostatic plasticity, stable activity level |
| Competitive Learning | ✅ | Winner-Take-All in DG, experienced neurons win |
| Predictive Coding | ✅ | MYELINATED connections do not strengthen (already predictable) |
| **SPIKING NEURAL NETWORK** | | |
| Hodgkin-Huxley Model | ✅ | Biologically accurate membrane potential dynamics |
| Real STDP | ✅ | Spike-timing dependent plasticity based on spike_history |
| Ion Channels | ✅ | Na+, K+, Leak channels with gating variables m, h, n |
| Refractory Period | ✅ | Absolute (2ms) and relative (5ms) refractory period |
| Short-Term Plasticity | ✅ | Facilitation and Depression (Tsodyks-Markram model) |
| Dendritic Computation | ✅ | Proximal/Distal compartments with different integration |
| Metaplasticity | ✅ | Plasticity of plasticity (BCM rule) |
| Calcium Dynamics | ✅ | Ca2+-dependent plasticity |
| Three-Factor Learning | ✅ | Eligibility traces + neuromodulation |
| **NEUROMODULATION** | | |
| BrainOscillator | ✅ | Theta (6Hz) and Gamma (40Hz) oscillations |
| Dopamine System | ✅ | Novelty → DA release → boosted STDP |
| Acetylcholine | ✅ | Attention gate, modulates learning |
| Norepinephrine | ✅ | Arousal/surprise, increases excitability |
| Serotonin | ✅ | Behavioral inhibition, patience |
| **SOURCE MEMORY** | | |
| SourceType enum | ✅ | LEARNING / EXPERIENCE / CONVERSATION / MEDIA |
| QuestionType enum | ✅ | SEMANTIC_FACT / EXPERIENCE / LOCATION / TEMPORAL |
| Episode.trust | ✅ | Trust level based on source type |
| PFC routing | ✅ | classify_question() + get_preferred_sources() |
| CA3 filtering | ✅ | Selective inclusion: preferred always + MEDIA only if ALL query words match |
| Unconnected context filter | ✅ | Lateral inhibition: hard skip for structurally unconnected episodes |
| Source preference bonus | ✅ | Preferred-source episodes get additive scoring advantage |
| **CA3 ATTRACTOR DYNAMICS** | | |
| CA3 class | ✅ | Separate recurrent module for pattern completion |
| Iterative dynamics | ✅ | Spread activation + WTA + stability check |
| Full scoring | ✅ | 2-hop paths, context diversity, top-down modulation |
| **PFC PERSISTENT ACTIVITY** | | |
| NMDA slow decay | ✅ | tau ~100ms for sustained firing (Wang 2001) |
| Recurrent excitation | ✅ | Related slots reinforce each other |
| Distractor resistance | ✅ | GABAergic inhibitory gating (Miller & Cohen 2001) |
| Attractor dynamics | ✅ | Bistable states for stable activity |

### Current model (brain_model)

```
Training pipeline: curriculum → preschool → grade1 → bAbI → FineWeb-Edu (1000 articles, 40K sentences)
Neurons: 48,318
Connections: 1,471,243
MYELINATED: 23,792 (1.6%)
USED: 76,375 (5.2%)
NEW: 1,371,076
Episodes: 76,688
  - NEW: 35,086
  - REPLAYED: 2,185
  - CONSOLIDATED: 38,065
  - DECAYING: 1,352
```

**Test results** (07.02.2026):
```
CURRICULUM: 50/50 (100.0%) — hard tests
STRICT: 3/3 (100%) — tests for "I do not know"
PRESCHOOL: 48/48 (100.0%) — preschool tests
GRADE1: 64/64 (100.0%) — world-knowledge tests
FineWeb-Edu: 9/9 (100.0%) — direct facts from educational texts
PARAPHRASE: 50/50 (100.0%) — paraphrase robustness tests
bAbI Task 1: 250/250 (100%) — working memory tests
TOTAL: 474/474 (100.0%)
```

**Comparison with IR baselines** (same training data):
```
Test          Brain    TF-IDF   BM25     Brain advantage
CURRICULUM    100.0%   64.0%    70.0%    +30-36%
STRICT        100.0%   33.3%    33.3%    +66.7%
PRESCHOOL     100.0%   81.2%    87.5%    +12-19%
GRADE1        100.0%   68.8%    71.9%    +28-31%
FINEWEB       100.0%   11.1%    33.3%    +67-89%
PARAPHRASE    100.0%   48.0%    48.0%    +52.0%
bAbI Task 1*  100%     N/A      N/A      N/A
─────────────────────────────────────────────────
AVERAGE       100.0%   51.1%    57.3%    +43-49%
```
*bAbI requires working memory — TF-IDF/BM25 cannot track entity states.

📊 **[Full results with analysis](docs/RESULTS.md)**

**New mechanisms (February 2026):**
- **Broca's Area Phase 3 Reanalysis (PHASE 17)** — paraphrase normalization (Friederici 2011)
  - Transforms non-canonical question forms to canonical WH-questions
  - Inverted questions: "The sky is what color?" → "What color is the sky?"
  - Imperative forms: "Name a farm animal" → "What is a farm animal?"
  - Classifier stripping: "What kind of food is an apple?" → "What is an apple?" (Croft 2001)
  - Passive constructions: "Cooking is done with what?" → "What do we cook with?"
  - Possessive decomposition: "What is hot's opposite?" → "What is opposite of hot?"
  - Temporal embedding: "What time of day do people wake up?" → "When do people wake up?"
  - Result: PARAPHRASE 100.0% (was 50.0%)
- **Temporal Concept Inference (PHASE 19)** — on-the-fly temporal recognition (Eichenbaum 2014)
  - PFC sends "temporal" goal for 'when' questions → primes temporal concept representations
  - Hippocampus checks if episode contains NEW temporal info (not already in query)
  - Combined with soft attentional facilitation (frozenset of before/after connectors)
  - Biology: anterior temporal lobe distinguishes temporal from spatial context
  - Result: all temporal questions now pass ("brush teeth"→day, "leaves fall"→autumn, "wash hands"→eating)
- **Episode Deduplication in Top-K (PHASE 20)** — consolidated memory merging (Born & Wilhelm 2012)
  - Multiple consolidated copies of same episode strengthen ONE attractor, not fill all top-K slots
  - Enables diverse secondary contributions from competing attractors via CA1 blending
  - Prevents echolalia when primary episode contains only query words
  - Result: sedimentary rock and paraphrase questions now pass
- **Source Memory Selective Inclusion (PHASE 21)** — biologically plausible retrieval hierarchy (Johnson et al. 1993)
  - Preferred sources (LEARNING, EXPERIENCE) always in candidate pool
  - Non-preferred sources (MEDIA) included ONLY when ALL content query words present in episode
  - Prevents MEDIA noise from overwhelming trusted sources while preserving domain-specific knowledge
  - Combined with unconnected context filter (lateral inhibition, Desimone & Duncan 1995)
  - "What disappears from leaves?" → "green chlorophyll" (MEDIA selectively included)
  - "Who is the president of Mars?" → "I do not know" (anti-hallucination preserved)
  - Result: **224/224 (100.0%)** — all 6 test suites at 100%
- **Hippocampal Time Cells for "When" Questions (PHASE 18)** — temporal retrieval (Eichenbaum 2014)
  - "When" as interrogative activates hippocampal time cells, biasing retrieval toward temporal info
  - Searches both 'before' and 'after' connections for temporal answers
  - Consolidation threshold: only consolidated connections (usage ≥ 1) are reliable (Born & Wilhelm 2012)
  - "When should you wash your hands?" → "before eating"
  - Falls through to general retrieval when no temporal connections found

**New mechanisms (January 2026):**
- **Basal Ganglia Action Selection (PHASE 4)** — Go/NoGo/STN for strategy selection
  - D1 (Go) / D2 (NoGo) pathways in Striatum
  - GPi/GPe tonic inhibition, STN hyperdirect pathway
  - Neuromodulators (DA/ACh/NE/5-HT) modulate selection
  - Selection of "retrieve" vs "multi_hop" in `ask()`
- **TRUE SWR Replay (PHASE 6)** — Sharp Wave-Ripples with temporal compression
  - `_swr_event()` — generation of spike times with 15x compression
  - Forward replay: memory consolidation (Buzsáki 2015)
  - Reverse replay (~30%): planning (Diba & Buzsáki 2007)
  - NREM/REM phases with different replay mechanisms
  - Synaptic homeostasis: downscaling after sleep (Tononi & Cirelli 2006)
- **NMDA Receptor Mechanism** — dynamic threshold for context attention (Malenka & Bear 2004)
  - When strongly activated (≥4 neurons) threshold decreases from 3 to 1
  - Weak synapses participate in Hebbian learning with high depolarization
  - Biology: Mg²⁺ block of NMDA receptor is removed at ~-40mV
- **Cross-Episode Linking** — semantic connections through shared context (McClelland et al. 1995)
  - During REM sleep, episodes with shared elements are replayed
  - Connections are formed between unique elements (dog↔cat through "animal")
  - Biology: Complementary Learning Systems — the hippocampus "teaches" the cortex
- **Source Memory (Johnson et al., 1993)** — the brain remembers WHERE knowledge came from
  - SourceType: LEARNING / EXPERIENCE / CONVERSATION / MEDIA
  - PFC classifies the question and routes to the appropriate sources
- **CA3 Attractor Dynamics** — biologically correct pattern completion
  - Iterative dynamics: spread activation + WTA + stability check
- **PFC Persistent Activity (PHASE 9.4)** — sustained activity in working memory
  - NMDA-like slow decay (tau ~100ms) for sustained firing (Wang 2001)
  - Recurrent excitation between related slots (attractor dynamics)
  - Distractor resistance via GABAergic inhibitory gating (Miller & Cohen 2001)
  - Goal-relevant inputs pass the barrier (top-down facilitation)
- **CA1 Output Layer (PHASE 9.2)** — the full hippocampal trisynaptic pathway
  - EC → DG → CA3 → CA1 → EC/PFC (Amaral & Witter 1989)
  - Schaffer collaterals (70%) + temporoammonic pathway (30%)
  - Projection to EC Layer V for consolidation and to PFC for working memory
- **Developmental Phases (PHASE 9.3)** — critical developmental periods
  - 4 stages: INFANT → CHILD → ADOLESCENT → ADULT (Hensch 2005)
  - Critical periods for language/semantic/syntactic
  - Experience-expectant plasticity with learning bonuses
  - Synaptic pruning peaking in ADOLESCENT (Huttenlocher 1979)
- **Broca's Area / Syntactic Processing (PHASE 11)** — syntactic processing
  - SyntacticProcessor extracts subject/predicate from questions (Friederici 2011)
  - Subject bonus in CA3 scoring to prioritize relevant episodes
  - Binary choice: "Is winter cold or hot?" → "cold"
- **Cause-Effect Relations (PHASE 12)** — cause-effect relations
  - Parsing questions of the form "What happens when X?"
  - CA3 filtering: the episode must contain the cause (subject)
  - Example: "What happens when ice gets warm?" → "melts"
- **Temporal Sequence Fix (PHASE 13)** — temporal retrieval fix
  - Excluding question words from answer candidates
  - "What month comes after January?" → "february" (not "month")
- **Antonym Relations (PHASE 14)** — biologically plausible antonym storage
  - Antonymy is encoded as connections with `connector='opposite'`
  - The same mechanism as temporal sequences (`connector='after'/'before'`)
  - Pattern "X is the opposite of Y" → bidirectional connections X↔Y
  - Works for ALL words including function words ("in"/"out")
  - "What is the opposite of in?" → "out" (Murphy 2003)
- **Iterative Retrieval (PHASE 15)** — PFC-Hippocampus reasoning loop
  - `IterativeRetriever` class in `pfc.py` for multi-step reasoning
  - PFC maintains goal state, iteratively queries hippocampus
  - Each retrieval adds context to working memory (accumulation)
  - Confidence = goal overlap + consolidation bonus
  - Max 4 iterations (like humans — Eichenbaum 2017)
  - **Integrated into the main `ask()`**: when direct retrieval does not find an answer
  - Also used in `ask_multi_hop()` for explicit multi-step reasoning
  - Biology: Preston & Eichenbaum 2013, Miller & Cohen 2001
- **Semantic Roles (PHASE 16)** — event structure for goal-conditioned retrieval
  - Episodes store semantic roles: agent, patient, theme, cause, location, time, etc.
  - Based on Fillmore's Case Grammar (1968) and event semantics (Zacks & Tversky 2001)
  - 18 role types biologically grounded in temporal-parietal processing
  - `get_expected_roles()` — PFC determines expected roles based on question type
  - Goal-conditioned retrieval: "What is X?" → category/property roles, "Where is X?" → location role
  - Roles stored in Episode and serialized with model
- **Baseline Comparison** — scientific evaluation against standard IR methods
  - TF-IDF and BM25 baselines on the same curriculum data
  - Brain significantly outperforms: +40% vs TF-IDF, +50% vs BM25
  - Tests integrated: `--compare-baselines` flag in test_brain.py
- Hodgkin-Huxley spiking neurons with realistic membrane potential dynamics
- Real STDP based on spike timing
- **BrainOscillator** — theta/gamma oscillations
- **NeuromodulatorSystem** — dopamine, acetylcholine, norepinephrine, serotonin

**Examples of working questions (Brain raw → Broca's area):**
| Question | Brain raw | Broca's area (LLM) |
|--------|-----------|-----------|
| What is a dog? | animal | An animal. |
| What color is the sky? | blue | blue |
| What is the capital of France? | paris | Paris |
| What does a cat say? | says meow meow | says meow meow |
| What comes after five? | six | Six comes after five. |
| What is the meaning of life? | love | Love is the meaning of life. |
| Who is the president of Mars? | I do not know | I do not know. |

**Why Broca's area (LLM)?**

Brain outputs **semantics**—a set of related words without grammar. This is the "thought" in its pure form.
The LLM (Qwen2.5:3b via Ollama) **verbalizes** the thought into speech—similar to how Broca's area in the brain is responsible for speech production.

**Important:**
- The LLM does **NOT change facts**—it only adds grammar (articles, word order, punctuation)
- We **see both outputs** (Brain raw + Broca's area) for transparency and debugging
- Correctness is evaluated on **Brain raw**, not on Broca's area—the LLM can make grammatical mistakes, but facts always come from Brain

**What the model knows:**
- Colors of objects (sky→blue, grass→green, apple→red)
- Animal sounds (dog→bark, cow→moo)
- Body parts (see→eyes, hear→ears)
- Opposites (hot→cold, big→small)
- Categories (dog+cat→animal, apple+banana→fruit)
- Emotions (laugh→happy, cry→sad)
- Places (learn→school, play→park)

### ✅ Why 100% is NOT test-specific tuning

Each Phase 19–21 mechanism solves a **class of problems**, not a specific test case. None contains hardcoded words, question-specific thresholds, or answer lookups.

| Mechanism | Biological Basis | Generality |
|-----------|-----------------|------------|
| **Phase 19**: Temporal Concept Inference | Hippocampal time cells (Eichenbaum 2014). PFC top-down modulation (Miller & Cohen 2001). | ANY "when" question. 89-word temporal set (time-of-day, seasons, months, days, life stages). No question-specific logic. |
| **Phase 20**: Episode Deduplication | Consolidation merges traces into unified representations (Born & Wilhelm 2012). | ALL consolidated episodes. Generic `input_words` dedup — any episode with N copies → 1. |
| **Phase 21**: Source Memory Selective Inclusion | Source memory = retrieval advantage, not gate (Johnson et al. 1993). Lateral inhibition (Desimone & Duncan 1995). | ALL questions with preferred sources. Generic `issubset()` check for non-preferred. Anti-hallucination preserved. |

**Free-form verification** (questions NOT in any test suite):
```
Q: Who is the king of Jupiter?      → "I do not know"              ✅ anti-hallucination
Q: What is the capital of Germany?   → "berlin..."                  ✅ LEARNING retrieval
Q: What is a cat?                    → "animal and a pet that..."   ✅ standard retrieval
Q: When do children sleep?           → temporal retrieval attempt    ✅ temporal inference
```

**Key criteria:**
1. **No hardcoded words** — temporal concepts are a general lexicon (89+ words), not test answers
2. **No question-specific logic** — all conditions are generic (`issubset()`, `input_words` dedup, role bonus)
3. **Anti-hallucination preserved** — novel nonsense questions correctly return "I do not know"
4. **Works on unseen data** — free-form questions answered from learned knowledge

### ⚠️ Current limitations

1. **Word order in the answer** — Hippocampal Time Cells are implemented: episodes preserve word order (`input_words: Tuple`). When connections have equal priority, the episode order is used. LLM post-processing adds grammar.

2. **Scaling** — tested on 1000 FineWeb-Edu articles (40K sentences). Needs validation on larger datasets.

3. **Language Interpretation (Rule-Based Parsing)** ⚠️
   
   The model uses **rule-based parsing** to interpret language, NOT learned linguistic knowledge:

   **⚠️ CRITICAL DISTINCTION: Grammar Coverage vs Fitting to Tests**

   | ❌ Fitting to tests (FORBIDDEN) | ✅ Expanding grammar coverage (ALLOWED) |
   |----------------------------------|-------------------------------------------|
   | Code works only for a specific test | Code handles a pattern that EXISTS in the curriculum |
   | Not in the data, added only to pass | The curriculum contains "hot and cold are opposites" → the parser must understand it |
   | Hardcoded answer | Adding a grammar rule for a pattern present in the data |
   
   **Example:** The curriculum contains BOTH patterns:
   - "hot is the opposite of cold"
   - "hot and cold are opposites"
   
   The parser MUST support both. This is NOT fitting — it's **grammar coverage for existing data**.
   This corresponds to the theory of **Universal Grammar**: humans have innate syntactic structures.
   
   | Component | What it does | Why this is acceptable |
   |-----------|--------------|------------------------|
   | `broca.py` | Question patterns ("What is X?", "X and Y are opposites") | Models Universal Grammar |
   | `pfc.py` | Question classification by keywords | Category routing is biologically plausible |
   | `lexicon.py` | Lists of function words | Closed-class words are finite |
   | `motor_output.py` | Rules for inserting copulas | Models learned syntactic frames |
   | `train.py` | Pattern extraction (temporal, opposite, cause-effect) | Recognizes patterns from the curriculum |
   
   **Why this is done this way:**
   - The model is trained on ~1,000 basic sentences (plus 40K from FineWeb-Edu), not billions like an LLM
   - A child learns language from ~10M words by age 6—we do not have that volume of data
   - Rule-based parsing approximates what would be learned from a large body of language data
   
   **What IS learned (not rule-based):**
   - ✅ Semantic memory — associations via Hebbian learning
   - ✅ Episodic memory — storage and retrieval of events
   - ✅ Connection strength — MYELINATED through usage (STDP)
   - ✅ Pattern completion — CA3 attractor dynamics
   - ✅ Antonyms/temporal relations — learned from sentences, not hardcoded
   
   **Analogy:** Like a person who knows facts but uses a dictionary to translate—the KNOWLEDGE is real, only the INTERFACE is simplified.

### 🗄️ Data storage (NumPy)

```
Format: NumPy arrays + a pickle dictionary
Load time: 0.6 seconds (2.3M connections)
Files:
  - graph_edges.npz — connections (src, dst, state, forward, backward, conn_type)
  - graph_vocab.pkl — vocabulary + connectors
```

**Connection format (STDP + Dual Stream):**
- `forward` — how many times `to` came AFTER `from`
- `backward` — how many times `from` came AFTER `to`
- `conn_type` — SEMANTIC (1) or SYNTACTIC (2)
- `connector` — a function word between content words (optional)

---

## How it works

### 1. Neuron (Hodgkin-Huxley Spiking Model)

```python
class Neuron:
    # Identification
    id: str                    # Unique identifier (word)
    neuron_type: NeuronType    # EXCITATORY / INHIBITORY
    
    # Hodgkin-Huxley dynamics
    V: float                   # Membrane potential (mV)
    m, h, n: float             # Ion-channel gating variables
    phase: NeuronPhase         # RESTING / DEPOLARIZING / REPOLARIZING / REFRACTORY
    spike_history: List[float] # Spike history for STDP
    
    # Connections
    connections_out: Set       # Outgoing connections
    connections_in: Set        # Incoming connections
```

**Biological model (Hodgkin & Huxley, 1952):**
- Membrane potential V changes under ionic currents (Na+, K+, Leak)
- Gating variables m, h, n control ion channel opening
- A spike is generated when V reaches the threshold (-55mV)
- After a spike there is a refractory period (absolute 2ms, relative 5ms)

**Biological constants:**
```python
V_REST = -70.0      # Resting potential (mV)
V_THRESHOLD = -55.0 # Spike threshold (mV)
V_PEAK = 40.0       # Action potential peak (mV)
E_NA = 50.0         # Na+ reversal potential (mV)
E_K = -77.0         # K+ reversal potential (mV)
```

### 2. Connection (with real STDP)

```python
class Connection:
    from_neuron: Neuron
    to_neuron: Neuron
    state: ConnectionState    # NEW / USED / MYELINATED / PRUNE
    forward_usage: int        # Pre before Post → LTP
    backward_usage: int       # Post before Pre → LTD
    
    # Real STDP based on spike timing
    def apply_stdp(self, current_time: float) -> None:
        """Applies STDP based on neurons' spike_history."""
        
    def propagate_spike(self, spike_time: float) -> None:
        """Propagates a spike to the postsynaptic neuron."""
```

**Biological STDP (Bi & Poo, 1998):**
- Pre before Post (dt > 0) → **LTP** (Long-Term Potentiation) — strengthening
- Post before Pre (dt < 0) → **LTD** (Long-Term Depression) — weakening
- The effect decays exponentially: `exp(-|dt| / tau)`, tau = 20ms

**Connection states:**
- `NEW` — new, unstable (0-4 uses)
- `USED` — strengthened (5-49 uses)
- `MYELINATED` — myelinated, precise knowledge (50+ uses)
- `PRUNE` — to be removed (unused for a long time)

**Thresholds (from config.py):**
```python
THRESHOLD_NEW_TO_USED = 5
THRESHOLD_USED_TO_MYELINATED = 50
THRESHOLD_TO_PRUNE = 100  # cycles without usage
```

### 3. Neuromodulation System (NEW!)

```python
class NeuromodulatorSystem:
    dopamine: float      # Reward/novelty signal
    acetylcholine: float # Attention gate
    norepinephrine: float # Arousal/surprise
    serotonin: float     # Behavioral inhibition
    
    def release(modulator, amount)  # Neuromodulator release
    def get_learning_rate_modifier() # Learning-rate modifier
    def get_excitability_modifier()  # Excitability modifier
```

**Biology (Schultz 1998, Gerstner 2018):**
- **Dopamine** — novelty/reward signal, boosts STDP for new connections
- **Acetylcholine** — attention gate, opens "gates" for learning
- **Norepinephrine** — arousal/surprise, increases neuronal excitability
- **Serotonin** — behavioral inhibition, patience

**Dopamine during learning:**
```
New connection → is_novel=True → _release_dopamine(0.3) → DA↑ (0.1→0.4)
→ da_modifier = 1.0 + (DA - 0.1) * 2 = 1.6
→ eligibility.value *= da_modifier → enhanced LTP
```

### 4. Brain Oscillator

```python
class BrainOscillator:
    theta_freq: float = 6.0   # Hz (episodic memory)
    gamma_freq: float = 40.0  # Hz (local computation)
    
    def update(dt_ms) → (theta, gamma)
    def get_excitability() → float  # Modulation from theta phase
```

**Biology (Buzsaki 2006):**
- **Theta (4-8 Hz)** — hippocampus, episodic memory, navigation
- **Gamma (30-100 Hz)** — binding, attention, local computation
- **Theta-Gamma Coupling** — sequence encoding

### 5. Activation

Activation spreads like "lightning":

```
Step 1: cat (start)
        ↓ ⚡ (MYELINATED)
Step 2: meows
```

**Neuron activation conditions:**
1. Receives a signal via a MYELINATED connection — activates immediately
2. Receives signals from 2+ active neighbors via USED connections — co-activation

### 4. Hebbian rule

"Neurons that fire together, wire together"

```python
# When learning from the sentence "cat meows":
conn = Connection.get_or_create(cat, meows)
conn.mark_used()  # usage_count += 1
```

Connections are created at first co-activation and strengthened with repetition.

### 5. Pattern

A pattern is a set of connected neurons that activate together.

```
        meows
           ⚡
           │
fluffy   ══⚡══ CAT  ══⚡══ pet
           │
           ⚡
        animal
```

---

## Files

```
Brain/
├── neuron.py              # Hodgkin-Huxley spiking neuron
├── connection.py          # Connection with real STDP (spike timing)
├── activation.py          # Activation propagation + spike simulation
├── spiking.py             # Full spiking module (STP, Dendritic, Metaplasticity)
├── hippocampus.py         # Episodic memory (DG, CA3, SWR)
├── cortex.py              # Semantic memory (Pattern storage)
├── config.py              # Single config for all model parameters
├── llm_postprocess.py     # LLM post-processing (Broca's area)
├── train.py               # Training with STDP and Q&A
├── curriculum.py          # Curriculum data (facts for a 5-year-old)
├── pattern.py             # Pattern class, patterns
├── episode.py             # Episodic memory (Episode class)
├── pyproject.toml         # Dependencies (uv/pip)
└── tests/
    └── test_brain.py      # Tests (curriculum, grade1, fineweb)
```

---

## What matches the specification (plan.md)

### ✅ FORBIDDEN and complied with:

| Restriction | Status |
|------------|--------|
| Numeric connection weights | ✅ No |
| Metrics/distances (cosine, dot) | ✅ No |
| Optimization (gradients, backprop) | ✅ No |
| Global search (Dijkstra, BFS) | ✅ No |
| Probabilistic models (softmax) | ✅ No |
| Deep Learning layers | ✅ No |
| Embedding as a meaning vector | ✅ No |

### ✅ ALLOWED and implemented:

| Mechanism | Status |
|----------|--------|
| Local connection history (usage_count) | ✅ |
| Discrete states | ✅ |
| Activation as lightning | ✅ |
| Pattern as a set of neurons | ✅ |
| Hebbian rule | ✅ |
| Connection limit (~7000) | ✅ |

---

## Roadmap

### ✅ PHASE 1: Semantic memory [DONE]
- Connections between concepts via Hebbian learning
- Myelination of frequent pathways (STDP)
- Spreading activation
- Chunking as an emergent property
- Dual Stream: SEMANTIC + SYNTACTIC

### ✅ PHASE 2: Emergent hierarchy [DONE]
- NO explicit IS_A/HAS_PROPERTY — this is not biologically plausible
- Categories emerge from the structure of connections
- find_categories() — category discovery from graph topology
- get_related_concepts() — related concepts by connection strength

### ✅ PHASE 2.5: Biological attention [DONE]
- generate_with_attention() — generation with context
- ACCUMULATIVE CONTEXT: each word adds its neighbors
- DECAY: old activations decay
- HUB PENALTY: log(1+n) — Weber–Fechner law
- LATERAL INHIBITION: top-N strong activations suppress weaker ones
- WINNER-TAKE-ALL: only winners remain
- SEED ANCHORING: the topic always stays in memory
- Working memory: ~7 items

### ✅ PHASE 2.6: Curriculum training [DONE]
- Training on basic facts (like a 5-year-old child)
- Tests: 10/10 (100%)
- Biological mechanisms fully implemented

### ✅ PHASE 3: Episodic memory [DONE]
- Hippocampus as a temporary buffer for new events
- DG (Dentate Gyrus) — pattern separation (sparse coding, ~2%, Rolls et al. 2007)
- CA3 — pattern completion (reconstruction from a partial cue)
- Episodes store input_neurons for retrieval
- Consolidation via replay

### ✅ PHASE 3.5: Attention during Retrieval [DONE]
- **Question context** is preserved during pattern_complete
- **query_overlap** — prioritizes episodes containing the original question words
- **avg_strength** — average strength of query→answer connections (myelinated pathways)
- **Activation history** — the full history is used, not only the final state
- The same mechanisms work both during training and inference

### ✅ PHASE 3.6: Interrogative Words + LLM Postprocess [DONE]
- **INTERROGATIVE_WORDS** — a separate class (what, where, who, when, why, how)
  - Create neurons and participate in activation
  - Do NOT form connections among themselves (like function words)
  - BIOLOGY: activate an "expectation template" in the prefrontal cortex
- **NO connector normalization** — "is", "was", "are" are stored as-is (biologically plausible)
- **Temperature** — probabilistic episode selection (like softmax in GPT)
- **config.py** — a single config for all model parameters
- **LLM Postprocess** (llm_postprocess.py) — Broca's area
  - Brain outputs semantics: "dog is animal"
  - The LLM formats into speech: "A dog is an animal."
  - The LLM does **NOT change facts**, only grammar

### ✅ PHASE 3.7: Anti-Hallucination (Context Word Connectivity) [DONE]
- **Problem:** The model answered "Who is the president of Mars?" → "president of country is leader"
- **Solution:** A biologically grounded check of context-word connectivity to the episode
  - Context words = query words that are NOT in the episode
  - If a context word is not connected to any word in the episode → the episode is irrelevant
  - Example: "mars" is not connected to {president, country, leader} → skip → "I do not know"
- **BIOLOGY:** The hippocampus rejects memories that are not activated by the input signal
- **Result:** 100% on hard tests (53/53), including "I do not know" for nonsensical questions

### ✅ PHASE 3.8: Top-Down Modulation + VERB_FORMS [DONE]
- **Top-Down Modulation** (Zanto et al. 2011)
  - `connector_filter` in Activation — prioritizes connections with a matching connector
  - For the question "What IS X?" connections with `connector="is"` are activated
  - `query_connector` in pattern_complete — +50 bonus for matching connector
  - **BIOLOGY:** PFC modulates retrieval by task type
- **Context Diversity** (Spens & Burgess 2024)
  - Counter of distinct episodes in which a connection occurred
  - Connections from diverse contexts are more semantic
- **Multi-hop Context**
  - CA3 looks 2 steps ahead to understand context
  - Recurrent connections in CA3 for pattern completion
- **VERB_FORMS** — morphological verb forms
  - `fall/falls/fell/falling`, `give/gives/gave/giving`, etc.
  - Query expansion to search episodes with different forms
  - **BIOLOGY:** The brain links different forms of the same word
- **Result:** Grade1 64/64 (100%)

### ✅ PHASE 4: Basal Ganglia Action Selection [DONE]
- Go/NoGo/STN for cognitive strategy selection
- D1 (Go) / D2 (NoGo) pathways in Striatum
- GPi/GPe tonic inhibition, STN hyperdirect pathway
- Integrated into `ask()`: selection of "retrieve" vs "multi_hop"
- Biology: Cortex → Striatum → GPi/GPe → Thalamus → Cortex

### ✅ PHASE 6: TRUE REPLAY / SWR [DONE]
- Sharp Wave-Ripples with temporal compression (15x)
- Forward replay: memory consolidation (Buzsáki 2015)
- Reverse replay (~30%): planning (Diba & Buzsáki 2007)
- SleepPhase enum: WAKE / NREM / REM
- NREM: SWR replay + slow oscillations
- REM: random reactivation for integration
- Synaptic homeostasis: downscaling after sleep

### ✅ PHASE 7: Internal Actions [DONE]
- BG selects cognitive actions: RETRIEVE / MULTI_HOP / INFER / WAIT
- Working Memory / Semantic Memory / Episodic Memory routing
- Integrated with PFC for routing

### 🟡 PHASE 8: Learn VERB_FORMS [NEXT]
- Remove hardcoded `VERB_FORMS` dict
- Morphology via learning (like children)
- Links goes↔went via shared context

### ⚪ PHASE 9: Additional Improvements
- DG Pattern Separation without hash()
- Sparse coding: 5:1 compression, WTA
- Scaling to 50K+ articles

### 🟢 Improvements

- **Pruning at the connection limit** — automatic removal of old connections
- **Multimodality** — visual input, modality binding

---

## How to run

### Installation
```bash
git clone https://github.com/sss777999/Brain.git
cd Brain
uv sync  # or pip install -e .
```

### Training on FineWeb-Edu
```bash
# Test (100 articles, ~10 seconds)
PYTHONPATH=. python train.py

# Full training (10K articles, ~30-40 minutes)
# Change in train.py: max_articles=10000, max_sentences=500000
```

### Brain model tests
```bash
python3 test_brain.py              # ALL tests (curriculum + grade1)
python3 test_brain.py --curriculum # Curriculum-only tests
python3 test_brain.py --grade1     # Grade 1 tests only
python3 test_brain.py --train      # Train a single model (curriculum → grade1)
python3 test_brain.py --strict     # Hard tests with correctness checks
python3 test_brain.py --raw        # Without LLM post-processing
```

### Model training
```bash
python3 test_brain.py --train      # Full pipeline: curriculum → grade1 → brain_model
python3 train.py full              # Alternative training method
```

### Loading the model in Python
```python
from graph_storage import GraphStorage
storage = GraphStorage.load('graph')
print(storage.get_neighbors('science', min_state=1)[:10])
```

---

## Project files

| File | Purpose |
|------|---------|
| `neuron.py` | Neuron class (binary state) |
| `connection.py` | Connection class with STDP (forward/backward) |
| `activation.py` | Activation logic (lightning over connections) |
| `graph_storage.py` | NumPy graph storage (fast loading) |
| `train.py` | Training on FineWeb-Edu |

### Saved models

| File | Description |
|------|-------------|
| `graph_edges.npz` | Connections (src, dst, state, forward, backward) |
| `graph_vocab.pkl` | Word vocabulary |

---

## Principles

1. **Biological plausibility** — everything as in the brain, without artificial computations
2. **Locality** — a neuron knows only its neighbors; there is no global observer
3. **Discreteness** — states, not numbers
4. **Natural selection** — frequently used connections strengthen, rare ones die off
5. **Patterns** — memory = connection structure, not values

---

## Metaphor

> Memory is like a landscape with grooves.
> Activation is like a ball rolling along those grooves.
> Myelinated connections are deep grooves.
> The ball rolls where the strengthened paths lead.

---

## Memory types (LLM architecture)

### Semantic memory (✅ implemented)
- World knowledge: "a cat meows", "the sun is a star"
- Not tied to time/place
- Stored in the cortex (cortex)

### Episodic memory (✅ implemented)
- Hippocampus as a temporary buffer for new events (`hippocampus.py`)
- **DG (Dentate Gyrus)** — pattern separation (sparse coding, ~2%)
- **CA3** — pattern completion (reconstruction from a partial cue)
- **SWR (Sharp Wave-Ripples)** — replay and consolidation during sleep
- Episodes store `input_words` (word order) for correct generation
- Consolidation via `sleep()` — strengthening connections and myelination
- 64,013 episodes in the current model (26,160 CONSOLIDATED)

```
Episodic memory (hippocampus)
    ↓ consolidation (replay)
Semantic memory (cortex)
```

### Cause-effect relations (🔴 needed)
- "pressed the button" → "the light turned on"
- For reasoning, not for memory
- Requires understanding of time and agency

### Grammar/syntax (🔴 needed)
- Word order in a sentence
- For text generation
- The next step after memory

---

## STRICTLY FORBIDDEN (from the plan.md specification)

| Forbidden | Why |
|----------|-----|
| Numeric connection weights (0.37, 0.85) | A connection exists or not; at most it has qualitative states |
| Metrics and distances (cosine, dot product) | You cannot choose a pattern by "minimum distance" |
| Optimization (gradients, backprop, loss) | There is no "error as a number"; only stabilization or decay |
| Global search (Dijkstra, BFS, A*) | Activation propagates locally; a neuron knows only its neighbors |
| Probabilistic models (softmax, Bayes) | Randomness only as a source of chaos, not as a knowledge model |
| Deep Learning layers (Linear, ReLU, Transformer) | Structures can be used for storage, but not as carriers of meaning |
| Symbolic rules (if A and B then C) | Logic can exist in control code, but memory is not stored as rules |
| Embedding as a meaning vector | Embedding is only a packaging of structure, not a geometric object |

---

## Data volume estimation

### Thresholds (from config.py)
- NEW → USED: 5 repetitions
- USED → MYELINATED: 50 repetitions

### How much data is needed
```
For one word pair (cat→meows):
  - Need 50 sentences where both words occur together
  
For 100 basic facts:
  - Need ~50,000 sentences

This matches real learning:
  - A child hears "cat meows" hundreds of times
  - Before it becomes stable knowledge
```

### Current status
- Synthetic dataset: works (repetitions are manually specified)
- Real dataset: 580 sentences (too few, connections do not reach thresholds)
- Needed: real texts (Wikipedia, books, FineWeb)


---

## How meaning is formed

### Meaning = the pattern of connections around a word

```
                    meows
                      ⚡
                      │
    fluffy   ══⚡══ CAT  ══⚡══ pet
                      │
                      ⚡
                   animal
                      │
                      →
              mammal
                      │
                      →
                    lion (also a feline!)
```

### Different phrasings strengthen the SAME connections

```
"The cat is fluffy" → connection cat↔fluffy +1
"A fluffy cat" → connection cat↔fluffy +1 (the same connection!)
"The cat is soft and fluffy" → connection cat↔fluffy +1
```

After 50 repetitions: `cat ══⚡══> fluffy` (MYELINATED)

### What is NOT overwritten

- Connections only strengthen or weaken
- New information adds new connections
- Repeated information strengthens existing ones
- Unused connections → PRUNE (forgetting)

---

## Pattern visualization

### Example: query "einstein"

```
📍 STEP 1: einstein⚡ → theory, relativity
📍 STEP 2: theory⚡ → darwin, evolution
📍 STEP 3: darwin + einstein → scientist
📍 STEP 4: scientist⚡ → physicist, biologist
📍 STEP 5: physicist + scientist → newton

🧠 FINAL PATTERN:
⚡ Precise knowledge: darwin, relativity, theory, scientist, physicist, evolution
→  Associations: biologist, newton, evolution
```

### Legend
- `══⚡══>` — myelinated connection (precise knowledge)
- `──────>` — USED connection (association)
- `+` — co-activation from multiple sources

---

## Decision history

### Why connections in both directions?
- Initially we thought: only forward in word order
- But: "meows" should activate "cat" (backward association)
- Decision: connections in both directions as independent synapses

### Why not lower thresholds?
- Temptation: lower thresholds for a small dataset
- But: that is artificial, not like the brain
- Decision: increase the data, do not lower thresholds

### Why not lemmatization?
- Temptation: artificially treat "cat" = "cats" = "cat (acc.)"
- But: in the brain these are different forms linked through experience
- Decision: learn it naturally from data

### STDP (temporal dynamics) — IMPLEMENTED!
- **Spike-Timing-Dependent Plasticity** — biological mechanism
- Connections now store `forward_usage` and `backward_usage`
- Word order matters: "cat meows" ≠ "meows cat"
- This enables generating text in the correct order

### Hebbian rule and window size
> **"Neurons that fire together wire together"** — Donald Hebb, 1949

Hebbian rule: connections form between neurons that are **co-active in time**.

#### Biologically grounded window size

The original Hebbian rule does not define a "window size". We derive it from biology:

| Parameter | Value | Source |
|----------|-------|--------|
| Reading speed | ~250ms per word | Psycholinguistics |
| Working memory | ~2000ms (~7±2 items) | Miller's law, 1956 |
| Hebbian time window | ~100ms | Neuroscience (STDP) |

**Calculation (diluted connectivity, Rolls et al. 2007):**
```
Diluted connectivity: 4 words (HEBBIAN_WINDOW_SIZE)
```

**Window = 4 words** — sparser connectivity increases memory capacity and reduces interference.

#### How it works

When reading text, words activate sequentially:
- Words within window (4 words) form connections
- Words beyond that are too far → NO connection

```
"The cat sat on the mat"

Hebbian rule (window = 4, diluted connectivity):
  cat ↔ sat ✓ (within window — connection)
  cat ↔ on ✓ (within window — connection)
  cat ↔ mat ✗ (beyond window — NO connection)
```

**This is biologically plausible:** Connections form between words that a person holds in consciousness at the same time.

---

## Stop-word handling

### Principle
- Stop words (prepositions, conjunctions, pronouns) participate in connections with other words
- Stop words do NOT create connections among themselves
- During recall, stop words are filtered out from results

### Example
```
Sentence: "a book lies on the table"

Connections created:
  ✓ on → table (stop + content)
  ✓ table → lies (content + content)
  ✓ lies → book (content + content)
  ✗ on → and (stop + stop — skip)

When recalling "table":
  → lies, book (stop words filtered out)
```

### Why this way?
- In the brain, a child hears "on the table", "under the table" and learns that "on/under" change meaning
- But by themselves "on", "under" without context are meaningless
- Connections to them are needed, but they should not dominate results

---

## Lateral inhibition (Lateral Inhibition)

### Biological mechanism
In the brain, strongly activated neurons suppress weakly activated neighboring neurons via inhibitory interneurons.

**Implementation:**
1. Top-N strongest activations are "winners" (not inhibited)
2. The rest receive inhibition proportional to the gap from the leader
3. The weaker relative to max, the stronger the inhibition

```python
# Code in graph_storage.py process_sentence()
max_score = scored[0][1]
top_n_inhibitors = 5  # Winners

for i, (word, score) in enumerate(scored):
    if i < top_n_inhibitors:
        # Top-N are not inhibited
        inhibited_scores.append((word, score))
    else:
        # Inhibition proportional to the difference from max
        relative_strength = score / max_score
        inhibition_factor = 0.5 + 0.5 * relative_strength
        final_score = score * inhibition_factor
```

### Hub Penalty (Weber–Fechner law)
Neurons with many connections have a higher activation threshold:

```python
hub_penalty = 1.0 / math.log1p(num_neighbors + 1)
```

This is biologically plausible: hubs (common words) are less specific and require more input signal to activate.

---

## Numbers and dates handling

### Principle
- Digits are preserved: "1945", "2024", "15th" — these are important data
- Pure numbers ("10", "100") are filtered during recall (too much noise)
- Numbers with letters ("15th", "2024th") remain

### Why not remove digits?
- Dates are knowledge: "1945 — the end of the war"
- Numbers in context matter: "100 rubles", "15 kilometers"
- Decision: keep them in the graph; filter pure numbers during recall

---

## Storage architecture (NumPy)

### Why NumPy?
| Solution | Speed | Complexity | Scalability |
|---------|-------|------------|-------------|
| Pickle (old) | ❌ Slow | ✅ Simple | ❌ Poor |
| **NumPy arrays** | ✅ Fast | ✅ Simple | ✅ Good |
| SQLite | ⚠️ Medium | ⚠️ Medium | ✅ Good |
| Neo4j | ✅ Fast | ❌ Complex | ✅ Excellent |

### Data structure
```python
{
    "word_to_id": {"russia": 0, "moscow": 1, ...},  # dict
    "id_to_word": ["russia", "moscow", ...],        # list
    "edges_src": np.array([0, 0, 1, ...]),          # int32
    "edges_dst": np.array([1, 2, 3, ...]),          # int32
    "edges_state": np.array([2, 1, 2, ...]),        # int8 (0=NEW, 1=USED, 2=MYELINATED)
    "edges_usage": np.array([50, 10, 55, ...]),     # int32
    "neighbors": {0: [(1, 2, 50), ...], ...}        # index for fast lookup
}
```

### Does not violate project principles
- NumPy is a storage format, not a computation model
- Like a book vs an e-book: the content is the same, the medium is different
- The activation logic remains biologically plausible

---

## Scientific analysis: what we are doing in the context of AI history

### How LLMs were created

**2017: Transformer (Vaswani et al.)**
- Idea: attention instead of recurrence
- They did not know whether it would scale
- They just tried it

**2018-2020: GPT-1 → GPT-2 → GPT-3**
- They observed: more parameters = better quality
- Scaling laws (Kaplan et al., 2020): a predictable relationship
- The revolution: "just scale it up and it will work"

**Key point:** They did not understand WHY it worked. They simply scaled up and observed correlation.

---

### What we are doing — an honest analysis

**Similarities with early LLMs:**
1. We also do not know for sure whether it will scale
2. We also rely on a hypothesis (biological plausibility = the right path)
3. We need to increase data and observe correlation

**Differences:**
1. LLMs are an empirical approach ("scale and see")
2. We are a theoretical approach ("do it like the brain")

---

### Honest assessment of our approach

**Strengths:**

1. **Biological foundation** — the brain works, so the principle is valid
2. **Interpretability** — we see patterns and understand why
3. **Efficiency** — O(k×s) instead of O(n²×d)
4. **Incremental learning** — no need to retrain the whole model

**Weaknesses / unknowns:**

1. **Text generation** — the brain generates speech via other mechanisms (Broca's area); we do not model this yet
2. **Scaling** — not validated at millions of neurons
3. **Quality** — not compared against LLMs on real tasks

---

### What science says

**Neuroscience:**
- Memory in the brain really works via strengthening connections (Hebb, 1949)
- Myelination is real and speeds signal conduction
- Hippocampus → cortex is a real consolidation mechanism

**Scale:**
- The brain has ~86 billion neurons
- We model ~1000 neurons
- A difference of 86 million times

**Question:** Will emergent properties appear with scaling?

LLMs showed: yes—new capabilities emerge with scaling (in-context learning, reasoning).

---

### Where we are now

**Analogy with LLMs:**
- GPT-1 was weak but showed the direction
- GPT-2 showed that scaling works
- GPT-3 was a breakthrough

**We are currently at the "GPT-1" stage** — proof of concept is done; we need to scale.

---

### Next step — a large-scale experiment

1. Take a real dataset (Russian Wikipedia, ~2 million articles)
2. Train the model (millions of sentences)
3. Measure:
   - How many connections become MYELINATED?
   - What patterns form?
   - Does recall work on complex queries?

**If this shows strong results, we have grounds for a revolution.**
**If not, we will learn what needs to change.**

> This is what LLM creators did: they scaled and observed.

---

## Current status (January 2026)

### Trained model statistics
```
Pipeline: curriculum → preschool → grade1 → FineWeb-Edu (1000 articles, 40K sentences)
Neurons: 48,301
Connections: 1,453,469
MYELINATED: 19,252 (1.3%)
USED: 77,745 (5.3%)
NEW: 1,356,472
Episodes: 68,947 (30,748 CONSOLIDATED, 2,139 REPLAYED)
```

### Test results (23.01.2026)
```
CURRICULUM: 49/50 (98.0%)
STRICT: 3/3 (100%)
PRESCHOOL: 46/48 (95.8%)
GRADE1: 64/64 (100%)
FineWeb-Edu: 7/9 (77.8%)
bAbI: 250/250 (100%)
TOTAL: 419/424 (98.8%)
```

### Usage

**Tests (without retraining):**
```bash
python3 test_brain.py --no-gpt --no-llm --skip-babi  # Fast tests
python3 test_brain.py                                 # All tests with LLM
```

**Model training:**
```bash
python3 test_brain.py --train  # Full training pipeline
```

---

## Future plan: Morpheme segmentation

### When to introduce
- After scaling (10,000+ articles)
- When recall shows morphology issues

### What to do
```python
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

# "cat" → ["cat", ""]
# "cats" → ["cat", "s"]
# A shared root can create connections
```

### Biological plausibility
- The brain has areas for morphemes (fusiform gyrus, ~130ms)
- Morphemes are minimal units of meaning
- Enables understanding new words: "reboot" = [re][boot]

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{belyi2026brain,
  title={Brain: Structural Memory, Thought Formation, and Language in a Biologically Grounded System},
  author={Belyi, Vitalii},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026},
  url={https://github.com/sss777999/Brain}
}
```

Full paper: [docs/arxiv.pdf](docs/arxiv.pdf)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **v1.0** | Jan 24, 2026 | Initial release. 98.8% accuracy (419/424 tests). Hippocampus, PFC, Basal Ganglia, Broca's area, STDP, sleep consolidation. |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

This is an open research project. Contributions, suggestions, and collaborations are welcome!

- **Issues**: Report bugs or suggest features
- **Pull requests**: Code improvements
- **Discussions**: Ideas about biological plausibility, new mechanisms

Contact: [GitHub Issues](https://github.com/sss777999/Brain/issues)
