# Repo memory: exact history index

- repo `/home/me/.cache/hncm/edit/repos/tiangolo__fastapi.git`; 80 files, 9,839,584 B raw
- chain **476,080 B = 20.7x**; gzip 3,413,837 B (2.9x); the chain is **7.2x better than gzip**
- every file decoded and every SHA-256 checked; 34.3 s on CPU

## Largest files

| path | versions | raw B | chain B | ratio |
|---|---:|---:|---:|---:|
| `README.md` | 195 | 4,077,812 | 74,803 | 54.5x |
| `.github/workflows/test.yml` | 112 | 654,264 | 23,594 | 27.7x |
| `docs/de/docs/deployment/docker.md` | 12 | 429,784 | 23,093 | 18.6x |
| `docs/de/docs/alternatives.md` | 11 | 300,262 | 16,116 | 18.6x |
| `.github/workflows/build-docs.yml` | 86 | 280,780 | 14,118 | 19.9x |
| `docs/de/docs/async.md` | 9 | 249,225 | 14,880 | 16.7x |
| `docs/de/docs/advanced/behind-a-proxy.md` | 13 | 213,694 | 11,583 | 18.4x |
| `docs/de/docs/advanced/settings.md` | 14 | 212,039 | 11,750 | 18.0x |
| `docs/de/docs/advanced/security/oauth2-scopes.md` | 11 | 203,498 | 11,038 | 18.4x |
| `docs/de/docs/advanced/generate-clients.md` | 17 | 197,547 | 10,659 | 18.5x |
| `docs/de/docs/deployment/concepts.md` | 8 | 172,418 | 11,478 | 15.0x |
| `docs/de/docs/advanced/custom-response.md` | 11 | 151,145 | 9,976 | 15.2x |
| `.github/workflows/deploy-docs.yml` | 51 | 134,655 | 8,823 | 15.3x |
| `docs/de/docs/advanced/events.md` | 14 | 133,687 | 7,197 | 18.6x |
| `.github/workflows/translate.yml` | 31 | 130,416 | 7,174 | 18.2x |
| `docs/de/docs/advanced/additional-responses.md` | 13 | 130,222 | 6,322 | 20.6x |
| `docs/de/docs/advanced/advanced-dependencies.md` | 15 | 123,447 | 9,255 | 13.3x |
| `docs/de/docs/deployment/https.md` | 8 | 121,032 | 8,869 | 13.6x |
| `docs/de/docs/_llm-test.md` | 9 | 114,835 | 7,442 | 15.4x |
| `docs/de/docs/advanced/path-operation-advanced-configuration.md` | 13 | 112,545 | 6,681 | 16.8x |

## Use

```bash
uv run cml repo --repo /path/to/repo.git --out repo-index --limit 50
uv run cml repo --index repo-index --search 'def routing'
```

The index is exact: each `.cmlh` replays to the same bytes and the
manifest carries the HEAD blob SHA. Search decodes the last version
and reports path, line, and the exact matching text.
