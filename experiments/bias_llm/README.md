# Learning to write XHAIL language bias

A minimal pipeline for fine-tuning a small LLM to generate XHAIL mode declarations
(`#modeh` / `#modeb`) from background knowledge and examples. The model is scored by
what XHAIL does with its bias, not by string match.

```
generate_tasks.py ──► data/{train,val,test}.jsonl      synthetic tasks + gold modes (verified by XHAIL)
        │
train_unsloth.py  ──► checkpoints/lora                 QLoRA SFT, one GPU
        │
predict.py        ──► preds/*.jsonl                    modes for test split or real benchmarks
        │
evaluate.py       ──► results/*.csv, *.summary.json    splice modes → run XHAIL → score
```

## Tasks

Four families, generated with random names, types and distractor predicates:

| family | hidden target | mirrors |
|---|---|---|
| `unary` | `h(X) :- [not] p(X), [not] q(X).` | animals, penguins |
| `part` | `h(X) :- has(X,Y), p(Y), [q(Y)].` | trains |
| `chain` | `h(X,Y) :- r1(X,Z), r2(Z,Y).` | grandfather |
| `ec` | `happens(e(#c),T) :- holdsAt(f(#d),T).` | event_calculus, epidemic |

Labels come from solving BG + target with clingo. About 30% of labelled atoms are held out.
A task is kept only if XHAIL, **given the gold modes**, learns rules with 100% held-out
accuracy, so every gold bias is known to be sufficient.

The prompt is the program with comments and modes removed. The benchmark comments
state the expected hypothesis, so leaving them in would leak the answer.

## Metrics (evaluate.py)

| metric | meaning |
|---|---|
| `valid` | output has a `#modeh` and XHAIL parses it |
| `solved` | XHAIL returns a hypothesis within the timeout |
| `acc` | held-out accuracy of the learned rules. **This is the headline metric.** |
| `hyp_match` | same hypothesis as with gold modes (noisy: XHAIL breaks cost ties arbitrarily, e.g. trains) |
| `runtime_s`, `n_modes` | search cost / bias size |
| `mode_f1` | overlap with gold modes (diagnostic only) |

## Baselines

- `gold`: the reference bias.
- `exhaustive`: a typed declare-everything heuristic (every BG predicate, with `not` and `+/-` variants).
- Base model with few-shot prompting (`predict.py --fewshot 3`, no fine-tuning).

## Baseline results (seed 0, 160 synthetic test tasks + 10 benchmarks)

| bias | split | solved | held-out acc | hyp match | median runtime | modes |
|---|---|---:|---:|---:|---:|---:|
| gold | synthetic test | 1.00 | 0.997 | 0.99 | 0.09 s | 5.6 |
| exhaustive | synthetic test | 0.85 | 0.84 | 0.79 | 0.30 s | 8.3 |
| gold | benchmarks | 1.00 | n/a | 0.90 | 0.01 s | 2.9 |
| exhaustive | benchmarks | 0.80 | n/a | 0.50 | 0.02 s | 5.9 |

The exhaustive baseline mostly fails on `chain` (0.64 solved) and `part` (0.77); the next
section explains why. The fine-tuned model has to beat about 0.84 held-out accuracy to be worth anything.

## Finding: XHAIL is sensitive to mode *order*

Two biases with the **same mode set** in a different order can make XHAIL succeed or fail.
The cause is that induction keeps only the first `XHAIL_MAX_KERNEL=10` kernel clauses
(`xhail/reasoning/induction.py`), and kernel order follows `#modeb` order. When the target
chain's schemas come later, the right kernel clauses get truncated away. Setting
`XHAIL_MAX_KERNEL=100000` recovers these tasks, but runs about 30x slower (0.3 s → 8–9 s).

Consequences:
- A bias is an ordered list, not a set. Order is part of what the model must get right.
- Gold data only includes orderings XHAIL can solve, so in `chain` tasks the target relation
  tends to be listed first. The model can learn this. It is arguably useful, but it is an artifact of the cap.
- This is a search-cost trade-off worth reporting: a tight, well-ordered bias works under the
  default cap, while a loose bias needs the cap lifted and pays in runtime.

## Run it

CPU, locally:

```bash
python experiments/bias_llm/generate_tasks.py --n 2000 --seed 0          # ~10 min on 2 cores
python experiments/bias_llm/evaluate.py --tasks experiments/bias_llm/data/test.jsonl --method gold
python experiments/bias_llm/evaluate.py --tasks experiments/bias_llm/data/test.jsonl --method exhaustive
python experiments/bias_llm/evaluate.py --tasks benchmarks --method exhaustive
```

GPU: open `colab_finetune.ipynb` in Colab (it mounts this repo from Google Drive), or run:

```bash
pip install unsloth
pip install -e .
cd experiments/bias_llm
python train_unsloth.py --model unsloth/Qwen2.5-1.5B-Instruct --epochs 2 --out checkpoints/qwen1.5b
python predict.py --model checkpoints/qwen1.5b --tasks data/test.jsonl --out preds/ft_test.jsonl
python predict.py --model checkpoints/qwen1.5b --tasks benchmarks      --out preds/ft_bench.jsonl
python predict.py --model unsloth/Qwen2.5-1.5B-Instruct --fewshot 3 --tasks data/test.jsonl --out preds/base_test.jsonl
python predict.py --model unsloth/Qwen2.5-1.5B-Instruct --fewshot 3 --tasks benchmarks      --out preds/base_bench.jsonl
```

Then evaluate each predictions file (CPU is fine):

```bash
python evaluate.py --tasks data/test.jsonl --method file --pred preds/ft_test.jsonl  --name ft_test
python evaluate.py --tasks benchmarks      --method file --pred preds/ft_bench.jsonl --name ft_bench
```

## Known limits of this simple version

- The synthetic families are narrow. A model that scores well on `test` may still fail on
  the real benchmarks, which is why the 10 benchmarks are kept as an out-of-distribution set.
- Gold bias is "sufficient and typed", not "minimal". The model learns a style of bias, not the unique right answer.
- The filter keeps only tasks XHAIL can solve with gold modes. This skews the data toward
  what XHAIL can learn (e.g. some chain tasks fail even with gold modes).
- SFT imitates gold. It does not optimise for XHAIL success directly. The natural next step is GRPO
  with `evaluate.score` as the reward (valid → solved → held-out acc, minus a runtime penalty).
