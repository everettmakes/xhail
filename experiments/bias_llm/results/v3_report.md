# Bias generator v3: ladder results

% of tasks reaching each level (cumulative). Seeded runs: mean over 3 seeds (min–max). `k=4` = up to 4 resampled attempts, stopping at the first bias XHAIL can learn from. v2 = trained on the old families; v3 = old + new families. shots2 = two retrieved examples of similar shape.

## Unseen synthetic, old families (160)

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 160 | 100 | 100 | 100 | 99 | 99 |
| v2 shots0 single | 160 | 100 | 100 | 90 | 89 | 89 |
| v2 shots0 k=4 | 160 | 100 | 100 | 95 (94–96) | 94 (93–95) | 94 (93–95) |
| v2 shots2 single | 160 | 99 (98–100) | 99 (98–100) | 76 (71–81) | 76 (71–81) | 69 (64–79) |
| v2 shots2 k=4 | 160 | 100 (99–100) | 99 (99–99) | 80 (77–81) | 79 (76–81) | 72 (69–79) |
| v3 shots0 single | 160 | 100 | 100 | 89 | 89 | 88 |
| v3 shots0 k=4 | 160 | 100 | 100 | 94 (93–95) | 94 (92–94) | 93 (92–94) |
| v3 shots2 single | 160 | 100 | 99 (98–99) | 74 (62–83) | 73 (62–82) | 66 (58–78) |
| v3 shots2 k=4 | 160 | 100 | 99 (99–99) | 79 (71–84) | 78 (71–84) | 71 (66–79) |

## Unseen synthetic, new families (80)

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 80 | 100 | 100 | 100 | 100 | 100 |
| v2 shots0 single | 80 | 98 | 86 | 32 | 31 | 11 |
| v2 shots0 k=4 | 80 | 98 (96–99) | 87 (86–88) | 38 (38–40) | 37 (35–39) | 15 (15–16) |
| v2 shots2 single | 80 | 100 | 100 | 36 (31–40) | 36 (31–39) | 29 (25–35) |
| v2 shots2 k=4 | 80 | 100 (99–100) | 100 (99–100) | 41 (31–48) | 41 (31–48) | 32 (25–38) |
| v3 shots0 single | 80 | 100 | 100 | 88 | 88 | 86 |
| v3 shots0 k=4 | 80 | 100 | 100 | 93 (92–94) | 93 (92–94) | 92 (91–92) |
| v3 shots2 single | 80 | 100 | 100 | 62 (57–66) | 62 (57–66) | 56 (52–61) |
| v3 shots2 k=4 | 80 | 100 | 100 | 65 (59–68) | 65 (59–68) | 59 (52–68) |

## Gold set (16 hand-written problems)

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| gold bias | 16 | 100 | 100 | 100 | 100 | 100 |
| v2 shots0 single | 16 | 100 | 81 | 50 | 50 | 25 |
| v2 shots0 k=4 | 16 | 100 | 90 (88–94) | 60 (56–62) | 60 (56–62) | 29 (25–31) |
| v2 shots2 single | 16 | 100 | 96 (94–100) | 56 (50–62) | 50 (44–56) | 25 (19–31) |
| v2 shots2 k=4 | 16 | 100 | 100 | 60 (56–62) | 56 (50–62) | 25 (19–31) |
| v3 shots0 single | 16 | 100 | 94 | 69 | 62 | 25 |
| v3 shots0 k=4 | 16 | 100 | 94 | 73 (69–75) | 69 (62–75) | 25 |
| v3 shots2 single | 16 | 100 | 98 (94–100) | 62 (56–69) | 58 (56–62) | 27 (25–31) |
| v3 shots2 k=4 | 16 | 100 | 96 (94–100) | 60 (56–62) | 58 (56–62) | 27 (25–31) |

## Seen, old families (80 training tasks)

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| v3 shots0 single | 80 | 100 | 100 | 90 | 90 | 90 |
| v3 shots0 k=4 | 80 | 100 | 100 | 98 | 98 | 98 |

## Seen, new families (80 training tasks)

| method | n | fmt | sem | expr | corr | tight |
|---|---:|---:|---:|---:|---:|---:|
| v3 shots0 single | 80 | 100 | 100 | 88 | 88 | 86 |
| v3 shots0 k=4 | 80 | 100 | 100 | 88 | 88 | 86 |

## New families: % correct by family (k=4, seed 0)

| family | v2 shots0 k=4 | v2 shots2 k=4 | v3 shots0 k=4 | v3 shots2 k=4 |
|---|---:|---:|---:|---:|
| const_attr | 0 | 41 | 100 | 100 |
| ec_named | 67 | 19 | 89 | 22 |
| prop | 0 | 100 | 100 | 100 |
| unary | 81 | 62 | 88 | 69 |

## Gold set, per problem (highest level reached, seed 0)

0 = format fails, 1 = semantics, 2 = expressive, 3 = correct, 4 = tight fails, 5 = all pass. Correct means 4 or 5.

| problem | gold bias | v2 shots0 single | v2 shots0 k=4 | v2 shots2 single | v2 shots2 k=4 | v3 shots0 single | v3 shots0 k=4 | v3 shots2 single | v3 shots2 k=4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bench-animals | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| bench-blocks | 5 | 1 | 4 | 5 | 5 | 4 | 4 | 3 | 1 |
| bench-epidemic | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| bench-event_calculus | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| bench-grandfather | 5 | 4 | 4 | 4 | 4 | 3 | 3 | 4 | 4 |
| bench-penguins | 5 | 1 | 1 | 2 | 2 | 1 | 1 | 2 | 2 |
| bench-propositional | 5 | 1 | 1 | 5 | 5 | 5 | 5 | 5 | 5 |
| bench-sugar | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| bench-traffic | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 5 | 5 |
| bench-trains | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| gold-adoptable | 5 | 2 | 2 | 2 | 2 | 5 | 5 | 2 | 2 |
| gold-enrol | 5 | 5 | 5 | 3 | 3 | 2 | 4 | 3 | 3 |
| gold-ripe | 5 | 2 | 2 | 2 | 2 | 4 | 4 | 5 | 5 |
| gold-sprinkler | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| gold-toxic | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| gold-uncle | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

## Repair (v3): one greedy fix of a corrupted bias

| tier | n | % correct | % correct by the level the broken bias failed at |
|---|---:|---:|---|
| gold | 176 | 73 | correct 0 (n=3), expressive 65 (n=46), format 77 (n=64), semantics 84 (n=44), tight 68 (n=19) |
| new | 474 | 99 | correct 100 (n=3), expressive 96 (n=78), format 100 (n=160), semantics 99 (n=104), tight 98 (n=129) |
