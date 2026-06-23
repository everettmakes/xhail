# XHAIL — eXtended Hybrid Abductive Inductive Learning

[![CI](https://github.com/everettmakes/xhail/actions/workflows/ci.yml/badge.svg)](https://github.com/everettmakes/xhail/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/everettmakes/xhail/branch/main/graph/badge.svg)](https://codecov.io/gh/everettmakes/xhail)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python implementation of the XHAIL Inductive Logic Programming framework, built on [clingo](https://potassco.org/clingo/) 5.6+.

XHAIL learns logic-program rules from background knowledge and examples. The learned rules are normal ASP rules — they can be read, checked, and extended directly.

```prolog
% Input
bird(a). bird(b). bird(c). penguin(d).
bird(X) :- penguin(X).
#modeh flies(+bird).
#modeb not penguin(+bird).
#example flies(a). #example not flies(d).

% Output (< 5 ms)
flies(V1) :- not penguin(V1).
```

---

## Contents

- [How it works](#how-it-works)
- [Benchmark results](#benchmark-results)
- [Installation](#installation)
- [Usage](#usage)
- [Input format](#input-format)
- [Repository layout](#repository-layout)
- [Engineering notes](#engineering-notes)
- [Comparison to other ILP systems](#comparison-to-other-ilp-systems)
- [Citation](#citation)
- [Local development](#local-development)

---

## How it works

XHAIL splits hypothesis search into three ASP solve calls:

```
Input (.lp file)
│
├─ Background knowledge — domain facts, rules, integrity constraints
├─ Mode declarations (#modeh / #modeb) — define the hypothesis language
└─ Examples (#example) — positive and negative observations
│
▼
╔══════════════════════════════════════════════════════════════╗
║  Phase 1 · Abduction                                         ║
║                                                              ║
║  Find the minimal set of ground atoms Δ (consistent with BG) ║
║  such that BG ∪ Δ satisfies every example.                   ║
║                                                              ║
║  Δ = { happens(infect(bob),1), happens(infect(carol),2) }   ║
╚══════════════════════════════════════════════════════════════╝
│
▼
╔══════════════════════════════════════════════════════════════╗
║  Phase 2 · Deduction — BFS kernel construction               ║
║                                                              ║
║  Build the kernel set K: one maximally-specific clause per   ║
║  leaf of a BFS over (abduced atoms × mode schemas).          ║
║  Parent-pointer reconstruction gives O(depth) chain recall.  ║
║  Full ancestor tracking prevents BFS cycles in O(1) per step.║
║                                                              ║
║  K = { happens(infect(bob),T)  :- holdsAt(ill(alice),T),    ║
║         happens(infect(carol),T) :- holdsAt(ill(bob),T), … } ║
╚══════════════════════════════════════════════════════════════╝
│
▼
╔══════════════════════════════════════════════════════════════╗
║  Phase 3 · Induction                                         ║
║                                                              ║
║  Search for the smallest subset H ⊆ K (by literal count)    ║
║  such that BG ∪ H covers every positive example and          ║
║  violates no negative example. Solved with a clingo          ║
║  optimisation program (minimize{use(I,J)}).                  ║
║                                                              ║
║  H = { happens(infect(bob),V1)   :- holdsAt(ill(alice),V1). ║
║         happens(infect(carol),V1) :- holdsAt(ill(bob),V1).  }║
╚══════════════════════════════════════════════════════════════╝
│
▼
Learned hypothesis — printed to stdout or returned via Python API
```

Each intermediate result (Δ, K, H) is inspectable. The `--debug` flag writes the intermediate ASP programs to disk.

---

## Benchmark results

Ten canonical ILP tasks across classical logic, Event Calculus, negation-as-failure, and multi-body reasoning. Timings on Python 3.11, clingo 5.7, 4-core laptop (wall time with `--jobs 4`).

| Benchmark | Domain | Rules | CPU time | Notes |
|---|---|:---:|---:|---|
| `animals` | Classification | 1 | 6 ms | `mammal(V1) :- produces_milk(V1).` |
| `blocks` | Event Calculus | 2 | 8 ms | pick_up / put_down rules |
| `epidemic` | Event Calculus | 2 | 9 ms | Chained infection cascade, NAF temporal negatives |
| `event_calculus` | Event Calculus | 1 | 8 ms | `happens(work(alice),V1) :- holdsAt(awake(alice),V1).` |
| `grandfather` | Recursive relations | 1 | 41 ms | 2-literal chain; predicate-indexed BFS |
| `penguins` | NAF / exceptions | 1 | 2 ms | `flies(V1) :- not penguin(V1).` |
| `propositional` | Propositional | 1 | 4 ms | `output :- .` (zero-arity rule) |
| `sugar` | Event Calculus | 2 | 7 ms | Priority-ordered resource consumption, NAF |
| `traffic` | Rules | 1 | 4 ms | `stop(V1) :- red(V1).` |
| `trains` | Structural | 1 | 28 ms | 3-body rule; induction selects subset of kernel |
| **Total** | | **13** | **117 ms CPU / 58 ms wall** | 10 / 10 solved |

```bash
python experiments/run_benchmarks.py           # parallel (default: all cores)
python experiments/run_benchmarks.py --jobs 1  # sequential
```

---

## Installation

Requires **Python ≥ 3.10**. `clingo` is installed automatically.

```bash
pip install xhail
```

Verify:

```bash
xhail --version
```

---

## Usage

### Command line

```bash
xhail run myfile.lp
xhail run myfile.lp --depth 15       # increase deduction depth (default 10)
xhail run myfile.lp --verbose        # show phase-by-phase output
xhail run myfile.lp --debug --debug-output ./debug_out/
```

### Python API

```python
from xhail import learn

result = learn("experiments/benchmarks/trains.lp", depth=10)

print(result.success)          # True
print(result.n_rules)          # 1
for rule in result.hypothesis:
    print(rule)
# eastbound(V1) :- has_car(V1,V2), triangle_load(V2), rectangle(V2).
```

`learn()` is thread-safe — each call creates a fresh `Model` with no shared mutable state.

---

## Input format

XHAIL `.lp` files are standard ASP with three extra directives:

```prolog
% Background knowledge — any valid ASP
bird(a). bird(b). bird(c).
penguin(d).
bird(X) :- penguin(X).

% Mode declarations
#modeh flies(+bird).           % head predicate to learn
#modeb penguin(+bird).         % body predicates available
#modeb not penguin(+bird).     % NAF body literal

% Examples
#example flies(a).             % positive — must be entailed by H
#example not flies(d).         % negative — must NOT be entailed by H
```

### Placemarker reference

| Marker | Role |
|--------|------|
| `+type` | Input variable — grounded by the head or a prior body literal |
| `-type` | Output variable — introduced by this literal |
| `#type` | Ground constant — the constant appears literally in the learned rule |

---

## Repository layout

```
xhail/
├── xhail/                      Core Python package
│   ├── __init__.py             Public API — learn(), LearningResult
│   ├── cli.py                  CLI entry point
│   ├── core.py                 Pipeline orchestrator
│   ├── language/
│   │   ├── terms.py            Atom, Clause, Literal, Normal, PlaceMarker, Fact
│   │   └── structures.py       Mode declarations
│   ├── parser/
│   │   └── parser.py           PLY-based .lp parser
│   └── reasoning/
│       ├── abduction.py        Phase 1 — ASP abduction, builds Δ
│       ├── deduction.py        Phase 2 — BFS kernel construction
│       ├── induction.py        Phase 3 — ASP minimisation, builds H
│       ├── model.py            Shared state (clingo bridge, subsumption cache)
│       └── utils.py            ASP serialisation helpers
│
├── experiments/
│   ├── benchmarks/             10 .lp benchmark files
│   ├── run_benchmarks.py       Benchmark runner (timing, memory, hypothesis)
│   └── results/                CSV / JSON metrics (git-ignored)
│
├── tests/                      183 tests, 94% line coverage
├── scripts/                    Utility scripts
└── .github/workflows/ci.yml    GitHub Actions — lint, type-check, test
```

---

## Engineering notes

**BFS kernel construction.** The original tracked only the immediate BFS predecessor, causing O(depth × level_size) chain reconstruction and allowing A→B→A cycles. The rewrite stores a `frozenset` of all ancestor keys per node (O(1) cycle detection) and a direct `parent_node` pointer for O(depth) chain recall.

**Leaf-node kernel collection.** The kernel is collected from BFS *leaf nodes* rather than the deepest BFS level. This matters when different head atoms produce chains of different depths — e.g. in the epidemic benchmark, the bob-rule terminates at depth 1 while the carol-rule extends to depth 2. Collecting only from `levels[top]` silently dropped the bob-rule.

**Type membership caching.** `isSubsumed(atom, mode)` checks that each constant belongs to the right type. The original called `getMatches` over the full model per check (quadratic). The rewrite builds a `dict[type_name, frozenset[str]]` once per abduced model, reducing subsumption to a set lookup.

**Predicate-indexed BFS.** The inner deduction loop previously cross-joined every (schema, fact) pair — O(|schemas| × |all_facts|) per level. A `{predicate → [Atom]}` index built once from the abduced model reduces this to O(|schemas| × |bucket_size|). On `grandfather` this cuts per-run time from ~270 ms to ~41 ms; on `trains` from 81 ms to 28 ms.

**Induction kernel cap.** After generalisation, abstract clauses typically collapse to one body length (~5 literals). A cap of 10 shortest abstract clauses keeps the induction ASP program small without affecting solve rate:

```
cap=5  → 90 ms,  10/10 solved
cap=10 → 97 ms,  10/10 solved  ← default
cap=50 → 375 ms, 10/10 solved  ← former default, 4× slower
```

**Parallel clingo.** Both abduction and induction pass `--parallel-mode=N` (N = min(4, cpu_count)) to clingo.

**Parallel benchmark runner.** `run_benchmarks.py` uses `ProcessPoolExecutor`. All 10 benchmarks complete in ~58 ms wall time vs ~117 ms sequential.

---

## Running the tests

```bash
pytest                                          # full suite
pytest -m "not integration"                    # unit tests only (< 1 s)
pytest --cov=xhail --cov-report=term-missing   # with coverage
pytest tests/test_benchmarks.py::TestTrainsBenchmark -v
```

183 tests pass on Python 3.10, 3.11, and 3.12 (94% line coverage).

---

## Comparison to other ILP systems

| System | Hypothesis language | NAF | Solver | Intermediate artefacts | Recursive programs |
|--------|---------------------|:---:|--------|:----------------------:|:------------------:|
| [Aleph](https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html) | Horn clauses | ✗ | Prolog | ✗ | Limited |
| [Metagol](https://github.com/metagol/metagol) | Metarule instances | ✗ | Prolog | ✗ | ✓ |
| [ILASP](https://doc.ilasp.com/) | Full ASP | ✓ | clingo | ✗ | Limited |
| [FastLAS](https://spike-imperial.github.io/FastLAS/) | Normal + choice rules | ✓ | clingo | ✗ | Limited |
| **XHAIL** (this work) | Normal rules with NAF | ✓ | clingo | ✓ (Δ, K, H) | ✗ |

**vs Aleph / Metagol.** Neither supports negation-as-failure, so defeasible rules like `flies(V1) :- not penguin(V1)` are not expressible.

**vs ILASP.** Both use clingo and support NAF. ILASP treats hypothesis search as a single optimisation; XHAIL exposes Δ and K as intermediate results, which makes it easier to trace why a particular hypothesis was or wasn't found.

**vs FastLAS.** FastLAS uses a faster partial evaluation strategy and scales better to large search spaces. XHAIL is a better fit for smaller problems where interpretability of the search process matters.

---

## Citation

This implementation is based on:

```bibtex
@article{ray2009xhail,
  author  = {Ray, Oliver},
  title   = {Nonmonotonic abductive inductive learning},
  journal = {Journal of Applied Logic},
  volume  = {7},
  number  = {3},
  pages   = {329--340},
  year    = {2009}
}
```

---

## Local development

Clone the repo and install in editable mode with dev dependencies:

```bash
git clone https://github.com/everettmakes/xhail.git
cd xhail
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
pytest --cov=xhail --cov-report=term-missing   # with coverage
```

Run the benchmarks:

```bash
python experiments/run_benchmarks.py           # parallel (default: all cores)
python experiments/run_benchmarks.py --jobs 1  # sequential
```

---

## License

MIT — see [LICENSE](LICENSE).
