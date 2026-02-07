# Brain Model Test Results

**Date:** February 07, 2026 (auto-generated)
**Model:** brain_model
**Training:** curriculum → preschool → grade1 → bAbI → FineWeb-Edu

---

## Model Statistics

| Metric | Value |
|--------|-------|
| Neurons | 48,367 |
| Connections | 1,471,255 |
| MYELINATED | 23,792 (1.6%) |
| USED | 76,387 (5.2%) |
| NEW | 1,371,076 |
| Episodes | 76,688 |
| — NEW | 35,086 |
| — REPLAYED | 2,185 |
| — CONSOLIDATED | 38,065 |
| — DECAYING | 1,352 |

---

## Test Results Summary

| Test Suite | Passed | Total | Accuracy | Time | Description |
|------------|--------|-------|----------|------|-------------|
| **CURRICULUM** | 50 | 50 | **100.0%** | 52.1s | Core knowledge tests |
| **STRICT** | 3 | 3 | **100.0%** | 4.8s | "I do not know" tests |
| **PRESCHOOL** | 48 | 48 | **100.0%** | 50.3s | Ages 3-6 knowledge |
| **GRADE1** | 64 | 64 | **100.0%** | 78.3s | Grade 1 world knowledge |
| **FINEWEB** | 9 | 9 | **100.0%** | 9.9s | Educational text facts |
| **PARAPHRASE** | 50 | 50 | **100.0%** | 53.1s | Surface form robustness |
| bAbI-1 | 25 | 25 | 100.0% | 2.8s | bAbI Task 1 |
| bAbI-2 | 25 | 25 | 100.0% | 6.3s | bAbI Task 2 |
| bAbI-3 | 25 | 25 | 100.0% | 27.6s | bAbI Task 3 |
| bAbI-4 | 5 | 5 | 100.0% | 1.0s | bAbI Task 4 |
| bAbI-5 | 25 | 25 | 100.0% | 7.0s | bAbI Task 5 |
| bAbI-6 | 25 | 25 | 100.0% | 3.6s | bAbI Task 6 |
| bAbI-7 | 25 | 25 | 100.0% | 3.9s | bAbI Task 7 |
| bAbI-8 | 25 | 25 | 100.0% | 3.5s | bAbI Task 8 |
| bAbI-9 | 25 | 25 | 100.0% | 3.3s | bAbI Task 9 |
| bAbI-10 | 25 | 25 | 100.0% | 3.4s | bAbI Task 10 |
| bAbI-11 | 25 | 25 | 100.0% | 3.2s | bAbI Task 11 |
| bAbI-12 | 25 | 25 | 100.0% | 3.2s | bAbI Task 12 |
| bAbI-13 | 25 | 25 | 100.0% | 3.4s | bAbI Task 13 |
| bAbI-14 | 25 | 25 | 100.0% | 4.0s | bAbI Task 14 |
| bAbI-15 | 20 | 20 | 100.0% | 3.4s | bAbI Task 15 |
| bAbI-16 | 5 | 5 | 100.0% | 0.9s | bAbI Task 16 |
| bAbI-17 | 40 | 40 | 100.0% | 2.3s | bAbI Task 17 |
| bAbI-18 | 29 | 29 | 100.0% | 4.4s | bAbI Task 18 |
| bAbI-19 | 5 | 5 | 100.0% | 0.6s | bAbI Task 19 |
| bAbI-20 | 52 | 52 | 100.0% | 6.1s | bAbI Task 20 |
| **bAbI TOTAL** | **481** | **481** | **100.0%** | 93.9s | All 20 bAbI tasks |
| **TOTAL** | **705** | **705** | **100.0%** | | All tests combined |

---

## Baseline Comparison

QA baselines (TF-IDF, BM25) trained on **identical data**. Working memory baselines (MemNet, NTM) tested on bAbI Task 1 only.

| Test | Brain | TF-IDF | BM25 | MemNet | NTM |
|------|-------|--------|------|--------|-----|
| CURRICULUM | **100.0%** | 64.0% | 70.0% | N/A | N/A |
| STRICT | **100.0%** | 33.3% | 33.3% | N/A | N/A |
| PRESCHOOL | **100.0%** | 81.2% | 87.5% | N/A | N/A |
| GRADE1 | **100.0%** | 68.8% | 71.9% | N/A | N/A |
| FINEWEB | **100.0%** | 11.1% | 33.3% | N/A | N/A |
| PARAPHRASE | **100.0%** | 48.0% | 48.0% | N/A | N/A |
| bAbI-1 | **100.0%** | N/A | N/A | 100.0% | 84.0% |
| bAbI-2 | **100.0%** | N/A | N/A | 28.0% | 0.0% |
| bAbI-3 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-4 | **100.0%** | N/A | N/A | 60.0% | 80.0% |
| bAbI-5 | **100.0%** | N/A | N/A | 64.0% | 68.0% |
| bAbI-6 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-7 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-8 | **100.0%** | N/A | N/A | 56.0% | 56.0% |
| bAbI-9 | **100.0%** | N/A | N/A | 32.0% | 24.0% |
| bAbI-10 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-11 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-12 | **100.0%** | N/A | N/A | 100.0% | 72.0% |
| bAbI-13 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-14 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-15 | **100.0%** | N/A | N/A | 45.0% | 5.0% |
| bAbI-16 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-17 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-18 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-19 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-20 | **100.0%** | N/A | N/A | 1.9% | 0.0% |
| **bAbI TOTAL** | **100.0%** | N/A | N/A | 24.3% | 19.4% |
| **QA AVG** | **100.0%** | **51.1%** | **57.3%** | N/A | N/A |

*bAbI requires working memory — TF-IDF/BM25 cannot track entity states. MemNet/NTM tested on all 20 tasks.*

### Key Findings

1. **Brain significantly outperforms simple IR methods** (+48-88%)
2. **Working memory (bAbI 1-20)** — Brain achieves 100% (481/481), TF-IDF/BM25 cannot handle context
3. **Paraphrase robustness** — 100% accuracy on surface form variation
4. **"I don't know" capability** — Brain correctly abstains on unknown queries


---

## Failed Tests Analysis

**No failures!** All tests pass at 100%.

---

## Answer Quality Audit

Manual audit of all 224 QA answers (bAbI excluded — uses WMStateTracker, answers are single words).

### Quality Distribution

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Clean | ~120 | 54% | Concise, meaningful, directly answers the question |
| Noisy but correct | ~70 | 31% | Correct keyword present + irrelevant secondary words |
| Shallow match | ~5 | 2% | Keyword present by accident, answer doesn't address the question |
| Unacceptable noise | ~5 | 2% | Correct keyword but offensive/absurd words mixed in |

### Critical Issues Found

#### 1. Fable/story contamination in factual answers

Secondary episodes from Aesop's fables and children's stories leak into factual answers via CA1 population coding.

| Question | Brain Answer | Problem |
|----------|-------------|---------|
| What is not good for teeth? | `sweets sport to watch the big play on the wet sand polly and the cat friends up tree admiringly...` | "sweets" correct, rest is **Fox and Crow fable** |
| What is a wolf? | `wild in the animal that lives in the forest killed a great many sheep boy ran toward the village...` | "wild animal" correct, rest is **Boy Who Cried Wolf** |
| What does a lion say? | `says roar mouse quickly recover found struggling in the net parted and soon free` | "roar" correct, rest is **Lion and Mouse fable** |
| What does a cat say? | `says meow meow polly poor poll wants mew heard like boy not a good friends catch` | "meow" correct, rest is children's story fragment |

#### 2. Inappropriate noise words

| Question | Brain Answer | Problem |
|----------|-------------|---------|
| What do we need for strong bones? | `milk feces body good kept food...` | **"feces"** from unrelated episode |
| What is the Earth? | `planet earth round like a ball goes around the sun chloroplasts...` | **"chloroplasts"** from botany episode |
| What gives us energy? | `eating bread and rice give chloroplasts sun from the light...` | **"chloroplasts"** again |

#### 3. Shallow keyword matches

| Question | Brain Answer | Expected | Problem |
|----------|-------------|----------|---------|
| What is the moon? | `stars appear in the sky at night when they would come out to see shines` | night | Answer describes **night sky**, not the moon itself |

### Root Cause

The CA1 population coding (`motor_output.py`) correctly blends top-K attractors from CA3 — this is biologically accurate. The noise comes from **training data mixing**: fables and stories share vocabulary with factual episodes (wolf, lion, crow, cheese are all connected to animal/food concepts). The `_is_relevant_to_query` filter checks MYELINATED connections but story words pass because they ARE genuinely connected to query concepts.

---

## Roadmap

### Completed

- [x] Core QA pipeline — 224/224 tests passing (100%)
- [x] bAbI Tasks 1-20 — 481/481 working memory tests (100%)
- [x] Paraphrase robustness — 50/50 surface form variants (100%)
- [x] Baseline comparison — TF-IDF, BM25, MemNet, NTM on all tasks
- [x] Population coding (CA1 readout) — blending from top-K attractors
- [x] Lateral inhibition threshold (0.6) — biologically grounded noise filter
- [x] Source memory selective inclusion (ca3.py)
- [x] Episode deduplication in top-K (hippocampus.py)
- [x] Temporal concept inference for 'when' questions
- [x] Removed artificial limits (MAX_SECONDARY_WORDS, MAX_SECONDARY_EPISODES)

### Open Issues

- [ ] **Answer noise from fables/stories** — secondary episodes from narrative texts contaminate factual answers. ~31% of answers contain irrelevant words from stories (Lion and Mouse, Fox and Crow, Boy Who Cried Wolf). Need source-aware filtering in `generate_from_population`.
- [ ] **Shallow keyword matching in tests** — `check_answer_simple` matches ANY keyword occurrence. "What is the moon?" passes on 'night' but answer doesn't describe the moon. Need semantic answer validation.
- [ ] **Inappropriate noise words** — words like 'feces', 'chloroplasts' appear in unrelated answers. Source memory filtering works at episode level but not at secondary word level in motor_output.
- [ ] **LLM postprocessing broken** — LLM just echoes Brain answer without reformulating. Needs investigation.
- [ ] **Answer verbosity** — many answers are 15-30 words when 2-5 would suffice. Broca's area sentence construction needs improvement.

### Planned Improvements

- [ ] **Source-aware secondary filtering** — in `generate_from_population`, check source type of secondary episodes. Narrative/fable episodes should contribute less than factual/learning episodes.
- [ ] **Semantic answer validation** — replace simple keyword matching with semantic check: does the answer actually ADDRESS the question?
- [ ] **Training data separation** — separate factual knowledge from narrative stories in training pipeline to reduce cross-contamination.
- [ ] **Broca's area sentence generation** — improve `_insert_connectors` to produce grammatical sentences, not word lists.
- [ ] **Consolidation-based noise reduction** — frequently consolidated factual episodes should dominate over narrative episodes in secondary contributions.

---

## How to Reproduce

```bash
# Train model
python train.py

# Run all tests with baseline comparison
python test_brain.py --no-gpt --no-llm

# Run specific test suite
python test_brain.py --curriculum --no-gpt --no-llm
```

---

*Auto-generated tables by `test_brain.py`. Quality audit and roadmap maintained manually.*
