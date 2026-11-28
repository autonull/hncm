# Template Pack

- root `/home/me/hncm/cml`: 50 files, 798,560 B
- templates 2,990; fills 7,318; pack 844,272 B
- min fill 8 B, min count 3, max n 6, budget 2,000,000 B

| count | fixed text | files |
|---:|---|---|
| 15 | `    lines += [
        "",
        <S>,
        "",
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/copylm.py, /home/me/hncm/cml/cortex.py |
| 24 | `    print(
        f<S>
        f<S>` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editlm.py, /home/me/hncm/cml/memory.py |
| 14 | `    lines = [
        <S>,
        "",
        f` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/cortex.py, /home/me/hncm/cml/memory.py |
| 44 | `        <S>,
        <S>,
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py, /home/me/hncm/cml/cortex.py |
| 20 | `<S>,
        "",
        <S>,
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/cortex.py, /home/me/hncm/cml/editlm.py |
| 29 | `            f<S>
            f<S>
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/concepts.py, /home/me/hncm/cml/cortex.py |
| 14 | `[run]
name = <S>
seed = 0
gates = ` | /home/me/hncm/cml/PLAN.md, /home/me/hncm/cml/experiments/p0-adapter.toml, /home/me/hncm/cml/experiments/p0-code.toml |
| 3 | `<E>
                    checks += 1
                    if length > best:
                        be` | /home/me/hncm/cml/copylm.py |
| 4 | `                        "tau": tau,
                        "hit": hit / max<E>,
                   ` | /home/me/hncm/cml/cml.py |
| 4 | `            for tau in <E>:
                exact = hit = 0
                bits = 0.0
             ` | /home/me/hncm/cml/cml.py |
| 5 | `                    "arm": f<S>,
                    "task": "t1",
                    "exact": row<` | /home/me/hncm/cml/cml.py |
| 3 | `                for p in <I>(cand):
                    if p < lo or seen >= depth:
                ` | /home/me/hncm/cml/bytecodec.py, /home/me/hncm/cml/codec.py |
| 23 | `        <D>(
            f<S>
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/concepts.py, /home/me/hncm/cml/cortex.py |
| 17 | `) -> tuple<E>:
    ""<S>""
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/copylm.py, /home/me/hncm/cml/cortex.py |
| 10 | `        print(
            f<S>
            f<S>` | /home/me/hncm/cml/cml.py |
| 4 | `,
                        "bits": bits / max<E>,
                    }
                )
` | /home/me/hncm/cml/cml.py |
| 27 | `        f<S>
        f<S>,
` | /home/me/hncm/cml/cml.py, /home/me/hncm/cml/cortex.py, /home/me/hncm/cml/editlm.py |
| 19 | `

def <I>() -> None:
` | /home/me/hncm/cml/tests/test_copylm.py, /home/me/hncm/cml/tests/test_editlm.py, /home/me/hncm/cml/tests/test_invariants.py |
| 16 | `"""

from <I> import <I>` | /home/me/hncm/cml/bytecodec.py, /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py |
| 18 | `            <D>,
            <D>,
` | /home/me/hncm/cml/cml.py |

## Search

- **template** b'\n\ndef (\n' x22 /home/me/hncm/cml/bytecodec.py, /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py
- **template** b'\n\n\ndef ' x22 /home/me/hncm/cml/bytecodec.py, /home/me/hncm/cml/cml.py, /home/me/hncm/cml/concepts.py
- **template** b'\n\ndef () -> None:\n' x19 /home/me/hncm/cml/tests/test_copylm.py, /home/me/hncm/cml/tests/test_editlm.py, /home/me/hncm/cml/tests/test_invariants.py
- **template** b'\n\ndef  -> None:\n' x15 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editdata.py, /home/me/hncm/cml/fillcopy.py
- **template** b'\n\n@()\ndef ' x10 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/cortex.py, /home/me/hncm/cml/editlm.py
- **template** b'\n\ndef ' x10 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py, /home/me/hncm/cml/copylm.py
- **template** b'    return out\n\n\ndef ' x10 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/copylm.py, /home/me/hncm/cml/editdata.py
- **template** b'\n\ndef (\n    ' x9 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editlm.py, /home/me/hncm/cml/pointer.py
- **template** b'\n\ndef  -> bytes:\n' x9 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editdata.py, /home/me/hncm/cml/editlm.py
- **template** b'    }\n\n\ndef ' x9 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editlm.py, /home/me/hncm/cml/mcp_server.py
- **template** b'\n\ndef  -> int:\n' x7 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py, /home/me/hncm/cml/copylm.py
- **template** b'    )\n\n\ndef ' x7 /home/me/hncm/cml/bytecodec.py, /home/me/hncm/cml/cml.py, /home/me/hncm/cml/codec.py
- **template** b'\n@\ndef ' x6 /home/me/hncm/cml/tests/test_invariants.py
- **template** b'\n\n\ndef ' x6 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editdata.py, /home/me/hncm/cml/memory.py
- **template** b'\n\ndef  -> dict:\n' x5 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/copylm.py, /home/me/hncm/cml/mcp_server.py
- **template** b'\n\ndef  -> list' x5 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/cortex.py, /home/me/hncm/cml/memory.py
- **template** b'\n\ndef  -> str:\n' x5 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editdata.py, /home/me/hncm/cml/memory.py
- **template** b'    return \n\n\ndef ' x5 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/editlm.py, /home/me/hncm/cml/memory.py
- **template** b'\n\n\ndef ' x5 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/overnight.py, /home/me/hncm/cml/tests/test_editlm.py
- **template** b'\n\ndef ' x4 /home/me/hncm/cml/cml.py, /home/me/hncm/cml/tests/test_literallm.py
