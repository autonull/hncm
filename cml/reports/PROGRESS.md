# CML search progress

```
CML SEARCH · P1 · 486 arms · budget 20.0/20.0 h · simulated
frontier  concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null·s2  metric 2.244  margin +0.049  cost 1.0 h
gates     P1a 2.6 | P1b 2.293 | P1c 2.29 | P1d 0.2 | P1e 0.1  →  binding margin +0.049
budget    [████████████] 20.0/20.0 h
leaderboard
   1  concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null  margin +0.046  n=3  cost 3.0 h
   2  concept·neural·lexicon·span·joint-em·verify·tied+residual·rotating·enwik8_10m·40m·null  margin +0.017  n=1  cost 1.0 h
   3  concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·20m·null  margin +0.016  n=1  cost 1.0 h
   4  concept·neural·lexicon·span·joint-em·verify·tied+residual·fixed·enwik8_10m·40m·null  margin +0.007  n=1  cost 1.0 h
   5  concept·neural·lexicon·span·joint-em·verify·composed·mixed·enwik8_10m·40m·null  margin +0.003  n=1  cost 1.0 h
   6  concept·neural·lexicon·span·joint-em·verify·tied·mixed·enwik8_10m·40m·null  margin +0.003  n=1  cost 1.0 h
   7  concept·neural·lexicon·span·joint-em·mlp·tied+residual·mixed·enwik8_10m·40m·null  margin -0.004  n=1  cost 1.0 h
   8  concept·neural·lexicon·span·fixed·verify·tied+residual·mixed·enwik8_10m·40m·null  margin -0.009  n=1  cost 1.0 h
   9  concept·neural·lexicon·span·joint-em·logistic·tied+residual·mixed·enwik8_10m·40m·null  margin -0.009  n=1  cost 1.0 h
  10  concept·neural·lexicon·span·joint-em·verify·tied·fixed·enwik8_10m·40m·null  margin -0.023  n=1  cost 1.0 h
tree      tested 29 · pruned 21 · promoted 8 · waiting 457
axes      arm        concept(29)
axes      price      neural(29)
axes      charge     lexicon(29)
axes      emission   span(29)
axes      parse      fixed(13)  joint-em(16)
axes      gate       logistic(12)  mlp(3)  verify(14)
axes      embedding  composed(12)  tied(3)  tied+residual(14)
axes      schedule   fixed(16)  mixed(11)  rotating(2)
axes      distill    none(29)
axes      data       enwik8_10m(29)
axes      scale      10m(11)  20m(2)  40m(16)
axes      adapt      null(29)
axes      seed       0(25)  1(2)  2(2)
next      concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null·s2  reason max UCB
sparkline r1:-0.12 r3:+0.05
```

## decision log

| rung | arm | metric | margin | cost | pass |
|---:|---|---:|---:|---:|---|
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·fixed·enwik8_10m·40m·null | 2.418 | -0.125 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·verify·composed·fixed·enwik8_10m·10m·null | 2.443 | -0.150 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·fixed·enwik8_10m·20m·null | 2.447 | -0.154 | 0.2 | no |
| 1 | concept·neural·lexicon·span·joint-em·logistic·composed·fixed·enwik8_10m·10m·null | 2.454 | -0.161 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·tied+residual·fixed·enwik8_10m·10m·null | 2.459 | -0.166 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·mixed·enwik8_10m·10m·null | 2.473 | -0.180 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·fixed·enwik8_10m·10m·null | 2.495 | -0.202 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·fixed·enwik8_10m·10m·null | 2.497 | -0.204 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·mlp·composed·fixed·enwik8_10m·10m·null | 2.497 | -0.204 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·tied·fixed·enwik8_10m·10m·null | 2.499 | -0.206 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·fixed·enwik8_10m·10m·null | 2.499 | -0.206 | 0.2 | no |
| 1 | concept·neural·lexicon·span·fixed·logistic·composed·rotating·enwik8_10m·10m·null | 2.501 | -0.208 | 0.2 | no |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null | 2.244 | +0.049 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null | 2.248 | +0.045 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null | 2.249 | +0.044 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·rotating·enwik8_10m·40m·null | 2.276 | +0.017 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·20m·null | 2.277 | +0.016 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·fixed·enwik8_10m·40m·null | 2.286 | +0.007 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·composed·mixed·enwik8_10m·40m·null | 2.290 | +0.003 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied·mixed·enwik8_10m·40m·null | 2.290 | +0.003 | 1.0 | yes |
| 3 | concept·neural·lexicon·span·joint-em·mlp·tied+residual·mixed·enwik8_10m·40m·null | 2.297 | -0.004 | 1.0 | no |
| 3 | concept·neural·lexicon·span·fixed·verify·tied+residual·mixed·enwik8_10m·40m·null | 2.302 | -0.009 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·logistic·tied+residual·mixed·enwik8_10m·40m·null | 2.302 | -0.009 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied·fixed·enwik8_10m·40m·null | 2.316 | -0.023 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·verify·composed·fixed·enwik8_10m·40m·null | 2.322 | -0.029 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·10m·null | 2.324 | -0.031 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·mlp·tied+residual·fixed·enwik8_10m·40m·null | 2.330 | -0.037 | 1.0 | no |
| 3 | concept·neural·lexicon·span·joint-em·logistic·tied+residual·fixed·enwik8_10m·40m·null | 2.331 | -0.038 | 1.0 | no |
| 3 | concept·neural·lexicon·span·fixed·verify·tied+residual·fixed·enwik8_10m·40m·null | 2.337 | -0.044 | 1.0 | no |

## what the search learned

- the frontier is `concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null` (seed 2) at margin +0.049.
- it spent 20.0 h of 20.0 h on 29 of 486 arms.
- arm: `concept` (29/29).
- price: `neural` (29/29).
- charge: `lexicon` (29/29).
- emission: `span` (29/29).
- parse: `joint-em` (16/29).
- gate: `verify` (14/29).
- embedding: `tied+residual` (14/29).
- schedule: `mixed` (11/29).
- distill: `none` (29/29).
- data: `enwik8_10m` (29/29).
- scale: `40m` (16/29).
- adapt: `null` (29/29).
- seed: `2` (2/29).
- next: `concept·neural·lexicon·span·joint-em·verify·tied+residual·mixed·enwik8_10m·40m·null` (max UCB).
