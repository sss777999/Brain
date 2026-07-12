# Brain Model Test Results

**Date:** July 12, 2026 (auto-generated)
**Model:** brain_model
**Training:** curriculum → preschool → grade1 → bAbI → FineWeb-Edu

---

## Model Statistics

| Metric | Value |
|--------|-------|
| Neurons | 48,318 |
| Connections | 1,519,981 |
| MYELINATED | 23,858 (1.6%) |
| USED | 138,875 (9.1%) |
| NEW | 1,357,248 |
| Episodes | 59,026 |
| — NEW | 35,060 |
| — REPLAYED | 2,101 |
| — CONSOLIDATED | 20,542 |
| — DECAYING | 1,323 |

---

## Test Results Summary

| Test Suite | Passed | Total | Accuracy | Time | Description |
|------------|--------|-------|----------|------|-------------|
| **CURRICULUM** | 49 | 50 | **98.0%** | 37.4s | Core knowledge tests |
| **STRICT** | 3 | 3 | **100.0%** | 2.1s | "I do not know" tests |
| **PRESCHOOL** | 48 | 48 | **100.0%** | 37.7s | Ages 3-6 knowledge |
| **GRADE1** | 63 | 64 | **98.4%** | 62.3s | Grade 1 world knowledge |
| **FINEWEB** | 6 | 9 | **66.7%** | 4.4s | Educational text facts |
| **PARAPHRASE** | 47 | 50 | **94.0%** | 38.3s | Surface form robustness |
| bAbI-1 | 25 | 25 | 100.0% | 9.9s | bAbI Task 1 |
| bAbI-2 | 21 | 25 | 84.0% | 22.5s | bAbI Task 2 |
| bAbI-3 | 21 | 25 | 84.0% | 46.0s | bAbI Task 3 |
| bAbI-4 | 5 | 5 | 100.0% | 4.3s | bAbI Task 4 |
| bAbI-5 | 25 | 25 | 100.0% | 7.2s | bAbI Task 5 |
| bAbI-6 | 25 | 25 | 100.0% | 3.7s | bAbI Task 6 |
| bAbI-7 | 25 | 25 | 100.0% | 3.3s | bAbI Task 7 |
| bAbI-8 | 25 | 25 | 100.0% | 3.1s | bAbI Task 8 |
| bAbI-9 | 25 | 25 | 100.0% | 2.9s | bAbI Task 9 |
| bAbI-10 | 25 | 25 | 100.0% | 2.9s | bAbI Task 10 |
| bAbI-11 | 25 | 25 | 100.0% | 10.3s | bAbI Task 11 |
| bAbI-12 | 24 | 25 | 96.0% | 10.3s | bAbI Task 12 |
| bAbI-13 | 24 | 25 | 96.0% | 10.8s | bAbI Task 13 |
| bAbI-14 | 5 | 25 | 20.0% | 19.5s | bAbI Task 14 |
| bAbI-15 | 20 | 20 | 100.0% | 2.9s | bAbI Task 15 |
| bAbI-16 | 5 | 5 | 100.0% | 0.8s | bAbI Task 16 |
| bAbI-17 | 0 | 40 | 0.0% | 25.7s | bAbI Task 17 |
| bAbI-18 | 0 | 25 | 0.0% | 15.6s | bAbI Task 18 |
| bAbI-19 | 0 | 5 | 0.0% | 2.9s | bAbI Task 19 |
| bAbI-20 | 56 | 56 | 100.0% | 6.2s | bAbI Task 20 |
| **bAbI TOTAL** | **381** | **481** | **79.2%** | 211.0s | All 20 bAbI tasks |
| **TOTAL** | **597** | **705** | **84.7%** | | All tests combined |

---

## Baseline Comparison

QA baselines (TF-IDF, BM25) trained on **identical data**. Working memory baselines (MemNet, NTM) tested on all bAbI tasks.
QA SUITE AVG is a macro-average across QA suites, not weighted by question count.

| Test | Brain | TF-IDF | BM25 | MemNet | NTM |
|------|-------|--------|------|--------|-----|
| CURRICULUM | **98.0%** | 64.0% | 70.0% | N/A | N/A |
| STRICT | **100.0%** | 33.3% | 33.3% | N/A | N/A |
| PRESCHOOL | **100.0%** | 79.2% | 87.5% | N/A | N/A |
| GRADE1 | **98.4%** | 67.2% | 68.8% | N/A | N/A |
| FINEWEB | **66.7%** | 11.1% | 11.1% | N/A | N/A |
| PARAPHRASE | **94.0%** | 48.0% | 48.0% | N/A | N/A |
| bAbI-1 | **100.0%** | N/A | N/A | 100.0% | 100.0% |
| bAbI-2 | **84.0%** | N/A | N/A | 32.0% | 0.0% |
| bAbI-3 | **84.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-4 | **100.0%** | N/A | N/A | 20.0% | 80.0% |
| bAbI-5 | **100.0%** | N/A | N/A | 68.0% | 72.0% |
| bAbI-6 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-7 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-8 | **100.0%** | N/A | N/A | 48.0% | 44.0% |
| bAbI-9 | **100.0%** | N/A | N/A | 24.0% | 24.0% |
| bAbI-10 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-11 | **100.0%** | N/A | N/A | 0.0% | 4.0% |
| bAbI-12 | **96.0%** | N/A | N/A | 100.0% | 80.0% |
| bAbI-13 | **96.0%** | N/A | N/A | 0.0% | 4.0% |
| bAbI-14 | **20.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-15 | **100.0%** | N/A | N/A | 10.0% | 0.0% |
| bAbI-16 | **100.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-17 | **0.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-18 | **0.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-19 | **0.0%** | N/A | N/A | 0.0% | 0.0% |
| bAbI-20 | **100.0%** | N/A | N/A | 5.4% | 0.0% |
| **bAbI TOTAL** | **79.2%** | N/A | N/A | 20.4% | 20.4% |
| **QA SUITE AVG** | **92.9%** | **50.5%** | **53.1%** | N/A | N/A |

*bAbI requires working memory — TF-IDF/BM25 cannot track entity states. MemNet/NTM tested on all 20 tasks.*

### Key Findings

1. **Brain significantly outperforms simple IR methods** (+42-66%)
2. **Working memory (bAbI 1-20)** — Brain achieves 79% (381/481), TF-IDF/BM25 cannot handle context
3. **Paraphrase robustness** — 94% accuracy on surface form variation
4. **"I don't know" capability** — Brain correctly abstains on unknown queries


---

## Failed Tests Analysis

### CURRICULUM (1 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| What is the moon? | and stars appear in the sky at night | ['satellite', 'round', 'night'] |

### GRADE1 (1 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| What does a plant need? | roots bring would water | {'all_of': ['water', 'sunlight', 'soil']} |

### FINEWEB (3 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| What was Booth? | 26 years old and one of the nations most famous actors | ['actor', 'famous'] |
| What is sedimentary rock made of? | shells and bones of sea animals many layers | ['bones', 'shells', 'organic', 'sandstone', 'limestone', 'shale'] |
| What is the origin of species? | overshadow charles darwin the remarkably likeable countryman of shropshire and kent the loyal friend of leading intellectuals around the world the loving playful father of accomplished children and especially the devoted husband of emma wedgwood | ['darwin', 'selection'] |

### PARAPHRASE (3 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| What category does a dog belong to? | has a black spot on his back | ['animal', 'pet', 'mammal'] |
| By what organ do we smell? | food nose blood good | ['nose'] |
| What time of day do people wake up? | drink their water every day | ['morning'] |

### bAbI-2 (4 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Where is the milk? | daniel went back to the garden | garden |
| Where is the milk? | daniel went back to the garden | garden |
| Where is the apple? | mary went back to the bedroom | bedroom |
| Where is the apple? | mary went back to the bedroom | bedroom |

### bAbI-3 (4 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Where was the football before the garden? | mary went back to the kitchen | kitchen |
| Where was the football before the garden? | mary went back to the kitchen | kitchen |
| Where was the football before the garden? | sandra went back to the hallway | hallway |
| Where was the milk before the kitchen? | sandra went back to the office | office |

### bAbI-12 (1 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Where is Mary? | and sandra went back to the bedroom | bedroom |

### bAbI-13 (1 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Where is Daniel? | and john went back to the hallway | hallway |

### bAbI-14 (10 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Where was Mary before the cinema? | journeyed to the school | kitchen |
| Where was Bill before the office? | this morning | bedroom |
| Where was Bill before the office? | journeyed to the kitchen | bedroom |
| Where was Mary before the school? | this evening | cinema |
| Where was Mary before the school? | this evening | cinema |
| Where was Bill before the school? | this morning | bedroom |
| Where was Mary before the cinema? | moved to the kitchen | office |
| Where was Julie before the office? | this morning travelled to the | school |
| Where was Julie before the office? | this morning travelled to the | school |
| Where was Mary before the cinema? | this morning journeyed to the | park |

### bAbI-17 (10 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Is the pink rectangle to the right of the red square? | triangle | no |
| Is the pink rectangle to the left of the red square? | triangle | yes |
| Is the pink rectangle to the left of the red square? | triangle | yes |
| Is the pink rectangle to the left of the red square? | triangle | yes |
| Is the pink rectangle to the right of the red square? | triangle | no |
| Is the red square to the right of the pink rectangle? | square | yes |
| Is the pink rectangle to the left of the red square? | triangle | yes |
| Is the pink rectangle to the left of the red square? | triangle | yes |
| Is the blue square below the pink rectangle? | square | no |
| Is the pink rectangle to the left of the blue square? | square | no |

### bAbI-18 (10 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| Does the box fit in the chocolate? | fits inside the chest is of | no |
| Is the chocolate bigger than the box? | container bigger than | no |
| Does the box fit in the chocolate? | fits inside the chest is of | no |
| Does the box fit in the chocolate? | fits inside the chest is of | no |
| Is the chocolate bigger than the box? | container bigger than | no |
| Is the chocolate bigger than the chest? | container bigger than | no |
| Is the chocolate bigger than the chest? | container bigger than | no |
| Does the chest fit in the chocolate? | the box of chocolates fits inside the is | no |
| Is the chest bigger than the chocolate? | container | yes |
| Does the suitcase fit in the chocolate? | is bigger than the container | no |

### bAbI-19 (5 failures)
| Question | Brain Answer | Expected |
|----------|--------------|----------|
| How do you go from the bathroom to the hallway? | is north of the bedroom | s,s |
| How do you go from the bedroom to the hallway? | is east of the kitchen and the bathroom | e,e |
| How do you go from the bathroom to the hallway? | is west of the office | n,e |
| How do you go from the kitchen to the bathroom? | the bedroom is east of the | n,w |
| How do you go from the bedroom to the kitchen? | the office is north of the bathroom | s,s |

---

## How to Reproduce

```bash
# Train model
python train.py

# Run all tests with baseline comparison
python test_brain.py --no-gpt --no-llm --babi-limit 5

# Run specific test suite
python test_brain.py --curriculum --no-gpt --no-llm

# Run only the mechanism/unit suite (fast, no model needed)
python test_brain.py --unit
```

---

*This file is auto-generated by `test_brain.py`. Do not edit manually.*
