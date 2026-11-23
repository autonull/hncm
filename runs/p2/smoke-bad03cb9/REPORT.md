# H-NCM v6 - report

format: 6  cfg: `bad03cb99a2f46a609aa8f551ece965a6e763da8bfa4a2059b9060b501119490`  device: cuda  torch: 2.15.0
corpora: {'shakespeare': '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structure | shakespeare | 0 | DP | 0.0 | 6.713 | 5.104 | 1.607 | 0.002 | 0.244 | 1.000 | 0.389 | 2 | 0.0077 |
| structure | shakespeare | 0 | DP | 2.0 | 6.713 | 5.104 | 1.607 | 0.002 | 0.244 | 1.000 | 0.389 | 2 | 0.0059 |
| compression | shakespeare | None | byte-AR | None | 5.584 | 5.584 | 0.000 | 0.000 |  |  |  | 63 | 0.0002 |
| compression | shakespeare | None | byte-GPT | None | 5.009 | 5.009 | 0.000 | 0.000 |  |  |  | 63 | 0.0002 |
| structure | shakespeare | 0 | fixed | 0.0 | 7.149 | 5.117 | 2.032 | 0.000 | 0.268 | 1.000 | 0.792 | 2 | 0.0023 |
| structure | shakespeare | 0 | fixed | 2.0 | 7.149 | 5.117 | 2.032 | 0.000 | 0.268 | 1.000 | 0.792 | 2 | 0.0009 |
| compression | shakespeare | None | gzip | None | 3.218 | 3.218 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |
| compression | shakespeare | None | unigram | None | 4.995 | 4.995 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G1 | P1a | DP=6.713 unigram=4.995 | DP<uni-0.1 | **fail** |
| G2 | P1b | DP=6.713 GPT=5.009 | DP<=GPT+0.05 | **fail** |
| G2b | P1c | DP=6.713 gzip=3.218 | DP<=gzip+0.1 | **fail** |
| G3 | calls shakespeare | best |dTotal|=1.129 | <=0.05 | **fail** |
| G9 | statistics shakespeare | seeds=1 | 3 seeds | **warn** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

4 gate(s) fail; report every claim, including negatives. Do not cherry-pick.
A negative result with clean controls is publishable.
