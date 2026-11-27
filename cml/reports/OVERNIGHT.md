# Overnight queue

Updated 2026-11-27T00:12:04. One task per 5-minute timer.
Done 7/13; a reboot is retried from the pointer checkpoint.

| task | status | attempts | payload/shared bpb |
|---|---|---:|---|
| `scale-10m-n1` | done | 1 | codec-scale-code-756752-p10m-n1 1.541/1.571 |
| `scale-10m+20m-c48m` | done | 2 | codec-scale-code-756752-p10m-c48m 1.569/1.599; codec-scale-code-756752-p20m-c48m 1.488/1.518 |
| `scale-10m-n2` | done | 1 | codec-scale-code-756752-p10m-n2 1.557/1.587 |
| `scale-10m-n3` | done | 1 | codec-scale-code-756752-p10m-n3 1.546/1.576 |
| `scale-5m-n2` | done | 1 | codec-scale-code-756752-p5m-n2 1.599/1.629 |
| `scale-5m-n3` | done | 1 | codec-scale-code-756752-p5m-n3 1.597/1.627 |
| `scale-10m-n4` | done | 1 | codec-scale-code-756752-p10m-n4 1.533/1.563 |
| `scale-10m-n5` | interrupted | 3 | - |
| `scale-5m-n4` | running | 1 | - |
| `scale-10m+20m-n1-c48m` | queued | 0 | - |
| `scale-10m+20m-n2-c48m` | queued | 0 | - |
| `scale-20m+40m-c64m` | queued | 0 | - |
| `portfolio-round4` | queued | 0 | - |

Logs: `reports/overnight/<task>.log`.
