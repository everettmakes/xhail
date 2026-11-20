# Bias generator v2: results

Qwen2.5-1.5B, QLoRA (Unsloth, Kaggle T4), trained on two jobs: **write** a bias from the program
(1,680 rows) and **repair** a broken bias given XHAIL's feedback (2,500 rows, 500 per failure level).
Prompts list the program's vocabulary (predicates/arity, example predicates, unary predicates as types).
Full tables: [results/v2_report.md](results/v2_report.md). Scores: [results/v2_scores.json](results/v2_scores.json).

## How biases are scored

Each bias is placed on a ladder; a task's score is the highest level passed. Levels are ordinal, so we
report the share of tasks reaching each level, never an average level.

1. **format**: every line is a valid mode declaration, and there is a `#modeh`
2. **semantics**: predicates, arities and types exist in the program; the head is the example predicate or an abducible
3. **expressive**: the intended rule can be written in this bias (types, `+`/`-`/`#` markers, variable links)
4. **correct**: XHAIL learns rules that agree with the intended rule on the labelled atoms (synthetic) or the whole domain (gold)
5. **tight**: no more modes than the gold bias

Three test tiers: **seen** (160 training programs), **unseen synthetic** (160 test programs), and
**gold** (16 hand-written problems: the 10 benchmarks and 6 new ones, frozen before training).
`k=4` means up to 4 attempts, stopping at the first bias XHAIL can learn from. Retries either
**resample** the same prompt or use **feedback** (a repair prompt with XHAIL's diagnosis). The same
budget is used for both, with 3 seeds on unseen synthetic and gold.

## Headline (as generated, % reaching *correct* / *tight*)

| method | seen | unseen synthetic | gold (n=16) |
|---|---|---|---|
| gold bias (ceiling) | 100 / 100 | 100 / 100 | 100 / 100 |
| exhaustive (declare everything) | 84 / 25 | 84 / 28 | 75 / 25 |
| untuned 1.5B, 3-shot | – | 29 / 26 | 19 / 12 |
| v1 fine-tune (gold modes only, no vocabulary) | – | 91 / 89 | 40 / 0 (10 benchmarks only) |
| **v2, single attempt** | 88 / 88 | 90 / 90 | 50 / 25 |
| **v2, resample k=4** | 96 / 96 | 95 (94–96) / 95 | 60 (56–62) / 29 (25–31) |
| **v2, feedback k=4** | 97 / 97 | 95 (95–96) / 95 | 56 / 31 |

**Repair** (one greedy fix of a corrupted bias, % reaching *correct*): seen 98, unseen synthetic 98, gold 69.

## What the numbers say

1. **The model learned the training skill.** Repair is 98% on both seen and unseen synthetic
   programs, and writing reaches 95% with retries. The sanity floor holds.
2. **Retries help; XHAIL's feedback does not beat blind resampling.** Four attempts add about 5 points
   on synthetic tasks and about 6–10 on gold, but feedback and resampling are level within seed noise
   (95 vs 95; 56 vs 60). With this model the gain comes from XHAIL *verifying* candidates, not from
   the content of its diagnosis.
3. **The gap is distribution shift.** On hand-written problems: 50% correct in one attempt (v1: 40%
   on the benchmarks), 56–60% with retries, and repair falls from 98% to 69%. The exhaustive baseline
   is still more often correct on gold (75%), but it is almost never tight (25%).
4. **Gold failures are specific and recurring** (single attempt):
   - **Event-calculus heads with constants** (`event_calculus`, `sugar`, `sprinkler`): the model can't
     produce heads like `happens(work(#person),+time)`.
   - **Ground constants in the body** (`ripe`: `colour(+fruit,#hue)`) and **needed negation** (`adoptable`).
   - **Invented or misshapen literals** (`blocks`: `holdsAt(on(#block,-block)`; `penguins`: the head
     predicate used in the body; `propositional`: `output/1` for `output/0`).
   - **Correct but loose** (`animals`, `grandfather`, `trains`, `uncle`): it works, but with 5–8 modes where 3–5 would do.
5. **Without fine-tuning, the loop hurts.** The 3-shot base model's valid-format rate falls from 96%
   to 48% once it is shown repair prompts.

## Caveats

- **Gold has 16 problems**, so each one is about 6 points. The per-problem grid in `results/v2_report.md` is the safer read.
- **"Correct" means correct on the data.** Some problems can't separate the intended rule from a
  simpler one. Two examples: in `animals`, `produces_milk` alone fits; in `adoptable`, `not aggressive`
  alone fits, which was a design slip in a problem written for this study.
- **Verify-then-prune is not in the headline.** It keeps only the modes XHAIL's own rule uses. On
  ambiguous data that drops modes the intended rule needs, so it lowers even the gold biases (unseen
  synthetic 100 → 86%). Pruned tables are in the full report.
- **The 6 new gold problems were written after seeing v1's failures** (but before v2 was trained, and
  without changing the training data). They test known weak spots on purpose.

## Next steps, by expected gain

1. **Broaden the training families** toward the gold failure types: event-calculus heads with `#`
   constants, `#` constants in body literals, required negation, propositional tasks. Write them from
   general templates, not from gold programs.
2. **Use XHAIL as a verifier at sampling time** (best-of-k), since that is where retries gain.
   Drop the feedback text unless a larger model makes use of it.
3. **Add discriminating examples** to under-constrained problems (`adoptable`, `animals`), or report
   them separately.
