# Bias generator v3: results

v3 tests two ideas against v2 in a 2×2 design:
- **New training templates (v3 model).** These are families aimed at v2's gold failures: event-calculus heads with `#` constants (`ec_named`), `#` constants in body literals (`const_attr`), and propositional tasks (`prop`). They are written from general templates, not from the gold programs.
- **Retrieved examples (shots2).** Two training programs of similar shape, with their biases, are added to the prompt.

Same setup as v2: Qwen2.5-1.5B, QLoRA (Unsloth, Kaggle T4), 2 epochs, with write and repair jobs. Up to 4 resampled attempts, 3 seeds.
Full tables: [results/v3_report.md](results/v3_report.md). Scores: [results/v3_scores.json](results/v3_scores.json).
Ladder and tiers are as in [RESULTS_v2.md](RESULTS_v2.md). There is one new tier: **unseen new families** (80 test programs from the new templates).

## Headline (% reaching *correct*, k=4, mean of 3 seeds, min–max in brackets)

| method | unseen, old families (160) | unseen, new families (80) | gold (16) |
|---|---|---|---|
| exhaustive (declare everything) | 84 | – | 75 |
| v2, no examples | 95 (94–96) | 38 (38–40) | 60 (56–62) |
| v2, 2 retrieved examples | 79 (76–81) | 41 (31–48) | 56 (50–62) |
| **v3, no examples** | **94 (93–95)** | **93 (92–94)** | **69 (62–75)** |
| v3, 2 retrieved examples | 79 (71–84) | 65 (59–68) | 58 (56–62) |

Single attempt, v3 with no examples: 89 / 88 / 62. **Repair (v3):** gold 73% (v2: 69%), unseen new families 99%.
Seen programs (sanity floor, v3): old families 98%, new families 88%.

## What the numbers say

1. **The new templates worked, and cost nothing.** On the new families, v3 reaches 93% against v2's 38%. On the old families it holds at 94% (v2: 95%). On gold it reaches 69% against v2's 60%: from 50% to 62% in a single attempt.
2. **Gold moved for the reasons we targeted.** `propositional`, `ripe` (`#` constant in the body) and `adoptable` (needed negation) now pass. Gold correct with 4 attempts (69%) is close to the exhaustive baseline (75%).
3. **Retrieved examples hurt both models.** Adding two similar examples lowers v3 by 15 points on old families, 28 on new and 11 on gold. The likely reason is that the models were fine-tuned on prompts without examples, so examples move them off the training distribution. The model seems to copy the examples' surface form (event names, extra modes) instead of reading the program. Examples help only where the model had never seen the pattern (v2 on `prop` and `const_attr`). That is what fine-tuning on that pattern already does better.
4. **The remaining gold failures are specific.**
   - **Event calculus (`event_calculus`, `sugar`, `sprinkler`).** The head now has the right *shape* (`happens(e(#person),+time)`), but the model picks the wrong *event*. It chooses an event already in the `happens` facts (`add`, `ignite`) instead of the one the examples need (`use`, `spray`). In `sugar` the needed event *terminates* a fluent. The synthetic `ec_named` family only teaches events that *initiate* the example fluent, so this is a training-template gap, not a modelling limit.
   - **`penguins`.** The model writes `not flies(+bird)`, negating the head predicate, instead of `not penguin(+bird)`.
   - **`grandfather` and `enrol`.** Type and link errors in chains (`prereq(+course,-student)`).
5. **Verification can pass a wrong bias.** On seen new-family programs, 100% of biases let XHAIL learn *something*, but only 88% learn the intended rule. Retrying stops at the first bias XHAIL accepts, so k=4 adds nothing there. A wrong rule that fits the training data is the limit of using XHAIL as a verifier.

## Caveats

- **Gold has 16 problems.** One problem is about 6 points, and the seed range on gold is 62–75.
- **The new families were designed after studying v2's gold failures.** They were written as general templates, not from the gold programs, but this is still targeted. The gold gain is evidence that the targeting transferred, not an independent test.
- Feedback retries were not rerun for v3, because in v2 they matched resampling.

## Next steps, by expected gain

1. **Extend `ec_named`:** events that *terminate* the example fluent, and events with more than one effect. This targets the 3 EC gold failures directly (about 19 points of gold).
2. **Drop retrieved examples at inference,** or train with them if they're wanted.
3. **Add a template where negation applies to a background predicate,** not the head (`penguins`).
