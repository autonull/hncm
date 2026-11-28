# Repo memory: exact history index

- repo `/home/me/.cache/hncm/edit/repos/tiangolo__fastapi.git`; 80 files, 9,839,584 B raw
- chain **787,482 B = 12.5x**; gzip 3,413,837 B (2.9x); the chain is **4.3x better than gzip**
- every file decoded and every SHA-256 checked; 46.8 s on CPU

## Largest files

| path | versions | raw B | chain B | ratio |
|---|---:|---:|---:|---:|
| `README.md` | 195 | 4,077,812 | 90,008 | 45.3x |
| `.github/workflows/test.yml` | 112 | 654,264 | 28,861 | 22.7x |
| `docs/de/docs/deployment/docker.md` | 12 | 429,784 | 49,634 | 8.7x |
| `docs/de/docs/alternatives.md` | 11 | 300,262 | 33,644 | 8.9x |
| `.github/workflows/build-docs.yml` | 86 | 280,780 | 18,397 | 15.3x |
| `docs/de/docs/async.md` | 9 | 249,225 | 32,194 | 7.7x |
| `docs/de/docs/advanced/behind-a-proxy.md` | 13 | 213,694 | 20,252 | 10.6x |
| `docs/de/docs/advanced/settings.md` | 14 | 212,039 | 24,026 | 8.8x |
| `docs/de/docs/advanced/security/oauth2-scopes.md` | 11 | 203,498 | 28,300 | 7.2x |
| `docs/de/docs/advanced/generate-clients.md` | 17 | 197,547 | 18,699 | 10.6x |
| `docs/de/docs/deployment/concepts.md` | 8 | 172,418 | 24,744 | 7.0x |
| `docs/de/docs/advanced/custom-response.md` | 11 | 151,145 | 19,072 | 7.9x |
| `.github/workflows/deploy-docs.yml` | 51 | 134,655 | 11,823 | 11.4x |
| `docs/de/docs/advanced/events.md` | 14 | 133,687 | 13,129 | 10.2x |
| `.github/workflows/translate.yml` | 31 | 130,416 | 9,724 | 13.4x |
| `docs/de/docs/advanced/additional-responses.md` | 13 | 130,222 | 13,527 | 9.6x |
| `docs/de/docs/advanced/advanced-dependencies.md` | 15 | 123,447 | 12,378 | 10.0x |
| `docs/de/docs/deployment/https.md` | 8 | 121,032 | 17,388 | 7.0x |
| `docs/de/docs/_llm-test.md` | 9 | 114,835 | 15,312 | 7.5x |
| `docs/de/docs/advanced/path-operation-advanced-configuration.md` | 13 | 112,545 | 12,503 | 9.0x |

## Use

```bash
uv run cml repo --repo /path/to/repo.git --out repo-index --limit 50
uv run cml repo --index repo-index --search 'def routing'
```

The index is exact: each `.cmlh` replays to the same bytes and the
manifest carries the HEAD blob SHA. Search decodes the last version
and reports path, line, and the exact matching text.
