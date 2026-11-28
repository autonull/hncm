# Repo memory: exact history index

- repo `/home/me/.cache/hncm/edit/repos/tiangolo__fastapi.git`; 80 files, 9,839,584 B raw; one global chain
- chain **390,024 B = 25.2x**; gzip 3,413,837 B (2.9x); the chain is **8.8x better than gzip**
- every file decoded and every SHA-256 checked; 50.8 s on CPU

## Largest files

| path | versions | raw B |
|---|---:|---:|
| `README.md` | 195 | 4,077,812 |
| `.github/workflows/test.yml` | 112 | 654,264 |
| `docs/de/docs/deployment/docker.md` | 12 | 429,784 |
| `docs/de/docs/alternatives.md` | 11 | 300,262 |
| `.github/workflows/build-docs.yml` | 86 | 280,780 |
| `docs/de/docs/async.md` | 9 | 249,225 |
| `docs/de/docs/advanced/behind-a-proxy.md` | 13 | 213,694 |
| `docs/de/docs/advanced/settings.md` | 14 | 212,039 |
| `docs/de/docs/advanced/security/oauth2-scopes.md` | 11 | 203,498 |
| `docs/de/docs/advanced/generate-clients.md` | 17 | 197,547 |
| `docs/de/docs/deployment/concepts.md` | 8 | 172,418 |
| `docs/de/docs/advanced/custom-response.md` | 11 | 151,145 |
| `.github/workflows/deploy-docs.yml` | 51 | 134,655 |
| `docs/de/docs/advanced/events.md` | 14 | 133,687 |
| `.github/workflows/translate.yml` | 31 | 130,416 |
| `docs/de/docs/advanced/additional-responses.md` | 13 | 130,222 |
| `docs/de/docs/advanced/advanced-dependencies.md` | 15 | 123,447 |
| `docs/de/docs/deployment/https.md` | 8 | 121,032 |
| `docs/de/docs/_llm-test.md` | 9 | 114,835 |
| `docs/de/docs/advanced/path-operation-advanced-configuration.md` | 13 | 112,545 |

## Use

```bash
uv run cml repo --repo /path/to/repo.git --out repo-index --limit 50
uv run cml repo --index repo-index --search 'def routing'
```

The index is exact: each `.cmlh` replays to the same bytes and the
manifest carries the HEAD blob SHA. Search decodes the last version
and reports path, line, and the exact matching text.
