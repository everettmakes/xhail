# Bias generator v2: ladder results

% of tasks reaching each level (cumulative). Seeded runs: mean over 3 seeds (min–max). `k=4` = up to 4 attempts, stopping at the first bias XHAIL can learn from.

## Seen programs (160 training tasks): as generated

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 160 | 100 | 100 | 100 | 100 | 100 |
| exhaustive | 160 | 100 | 100 | 100 | 84 | 25 |
| sft single | 160 | 100 | 100 | 88 | 88 | 88 |
| sft resample k=4 | 160 | 100 | 100 | 96 | 96 | 96 |
| sft feedback k=4 | 160 | 100 | 100 | 97 | 97 | 97 |

## Seen programs (160 training tasks): after verify-then-prune

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 160 | 100 | 100 | 92 | 92 | 92 |
| exhaustive | 160 | 100 | 100 | 93 | 78 | 76 |
| sft single | 160 | 100 | 100 | 81 | 81 | 81 |
| sft resample k=4 | 160 | 100 | 100 | 89 | 89 | 89 |
| sft feedback k=4 | 160 | 99 | 99 | 89 | 89 | 89 |

## Unseen synthetic (160 test tasks): as generated

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 160 | 100 | 100 | 100 | 100 | 100 |
| exhaustive | 160 | 100 | 100 | 100 | 84 | 28 |
| base3shot single | 160 | 96 | 79 | 39 | 29 | 26 |
| base3shot feedback k=4 | 160 | 48 | 39 | 31 | 30 | 27 |
| sft single | 160 | 100 | 100 | 90 | 90 | 90 |
| sft resample k=4 | 160 | 100 | 100 | 95 (94–96) | 95 (94–96) | 95 (94–96) |
| sft feedback k=4 | 160 | 100 | 100 | 95 (95–96) | 95 (95–96) | 95 (95–96) |

## Unseen synthetic (160 test tasks): after verify-then-prune

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 160 | 100 | 100 | 86 | 86 | 86 |
| exhaustive | 160 | 100 | 100 | 86 | 71 | 69 |
| base3shot single | 160 | 96 | 79 | 36 | 26 | 26 |
| base3shot feedback k=4 | 160 | 48 | 39 | 29 | 27 | 27 |
| sft single | 160 | 100 | 100 | 78 | 78 | 78 |
| sft resample k=4 | 160 | 100 | 100 | 82 (81–83) | 82 (81–83) | 82 (81–83) |
| sft feedback k=4 | 160 | 100 | 100 | 83 (82–83) | 83 (82–83) | 83 (82–83) |

## Gold set (16 hand-written problems): as generated

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 16 | 100 | 100 | 100 | 100 | 100 |
| exhaustive | 16 | 100 | 100 | 94 | 75 | 25 |
| base3shot single | 16 | 88 | 75 | 19 | 19 | 12 |
| base3shot feedback k=4 | 16 | 31 | 25 | 19 | 19 | 12 |
| sft single | 16 | 100 | 81 | 50 | 50 | 25 |
| sft resample k=4 | 16 | 100 | 90 (88–94) | 60 (56–62) | 60 (56–62) | 29 (25–31) |
| sft feedback k=4 | 16 | 100 | 85 (81–88) | 56 | 56 | 31 |

## Gold set (16 hand-written problems): after verify-then-prune

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 16 | 100 | 100 | 81 | 81 | 81 |
| exhaustive | 16 | 100 | 100 | 75 | 56 | 50 |
| base3shot single | 16 | 88 | 75 | 12 | 12 | 12 |
| base3shot feedback k=4 | 16 | 31 | 25 | 12 | 12 | 12 |
| sft single | 16 | 100 | 81 | 38 | 38 | 38 |
| sft resample k=4 | 16 | 100 | 90 (88–94) | 44 | 44 | 44 |
| sft feedback k=4 | 16 | 100 | 85 (81–88) | 38 | 38 | 38 |

## Gold set, per problem (highest level reached, after pruning, seed 0)

Levels: 0 = format fails, 1 = semantics fails, 2 = expressive fails, 3 = correct fails, 4 = tight fails, 5 = all pass.

| problem | gold bias | exhaustive | base3shot single | base3shot feedback k=4 | sft single | sft resample k=4 | sft feedback k=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bench-animals | 2 | 4 | 2 | 2 | 2 | 2 | 2 |
| bench-blocks | 5 | 3 | 0 | 0 | 1 | 5 | 1 |
| bench-epidemic | 5 | 5 | 2 | 0 | 5 | 5 | 5 |
| bench-event_calculus | 5 | 5 | 1 | 0 | 2 | 2 | 2 |
| bench-grandfather | 5 | 5 | 2 | 0 | 5 | 5 | 5 |
| bench-penguins | 5 | 5 | 2 | 0 | 1 | 1 | 2 |
| bench-propositional | 5 | 5 | 0 | 0 | 1 | 1 | 1 |
| bench-sugar | 5 | 5 | 1 | 0 | 2 | 2 | 2 |
| bench-traffic | 5 | 2 | 5 | 5 | 5 | 5 | 5 |
| bench-trains | 2 | 3 | 2 | 0 | 2 | 2 | 2 |
| gold-adoptable | 2 | 2 | 2 | 1 | 2 | 2 | 2 |
| gold-enrol | 5 | 5 | 2 | 0 | 5 | 5 | 5 |
| gold-ripe | 5 | 2 | 2 | 0 | 2 | 2 | 2 |
| gold-sprinkler | 5 | 3 | 2 | 0 | 2 | 2 | 2 |
| gold-toxic | 5 | 2 | 2 | 2 | 5 | 5 | 5 |
| gold-uncle | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

## Repair: one greedy fix of a corrupted bias

| tier | n | reach correct | correct, by the level the broken bias failed at |
|---|---:|---:|---|
| seen | 300 | 98 | correct 100% (n=13), expressive 94% (n=50), format 100% (n=100), semantics 96% (n=78), tight 100% (n=59) |
| test | 300 | 98 | correct 100% (n=14), expressive 93% (n=42), format 100% (n=103), semantics 98% (n=87), tight 100% (n=54) |
| gold | 176 | 69 | correct 67% (n=3), expressive 63% (n=46), format 69% (n=64), semantics 70% (n=44), tight 79% (n=19) |
