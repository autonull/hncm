# Log delta: exact CI log history

- 5 logs, 504,465 B raw
- chain **18,445 B = 27.3x**; gzip 32,827 B (15.4x); the chain is **1.8x better**
- all versions decoded and every SHA-256 checked; 0.7 s CPU

| log | versions | raw B |
|---|---:|---:|
| `run0.log` | 1 | 100,895 |
| `run1.log` | 1 | 100,890 |
| `run2.log` | 1 | 100,895 |
| `run3.log` | 1 | 100,890 |
| `run4.log` | 1 | 100,895 |

Chain written to `/tmp/logs-demo.cmlh`.

