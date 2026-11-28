# Template-fill probe

- code corpus `7ae9de7b4a15`, train 2,000,000 B, held-out 100,000 B
- templates: token runs 2-6, min count 3, at least 6 fixed bytes, fewer holes than tokens
- fill lexicon: exact typed atoms from train

| item | value |
|---|---:|
| templates | 18,925 |
| fill values | 15,873 |
| table bytes | 893,746 |
| fixed bytes | 0.224 |
| hole bytes | 0.278 |
| literal fixed | 0.200 |
| literal atom | 0.298 |
| atom in lexicon | 0.486 |
| hole: seen slot | 0.121 |
| hole: new combination | 0.297 |
| hole: novel value | 0.582 |
| slot pairs | 15,374 |
| distinct fills/slot | 4.4 |
| decisions/B | 0.218 |
| bytes/decision | 4.58 |
| round-trip | OK |

## Read

- fixed syntax is 22.4% of held-out bytes and needs no model call; holes are another 27.8%.
- a sequential event model would take 0.218 decisions/B (4.58 B/decision), before any learned predictor.
- 48.6% of atom bytes are in the 2 MB train lexicon; the rest need context copy or the byte fallback.
- of hole bytes, 12.1% repeat the same slot, 29.7% are new combinations with a known fill, and 58.2% are novel values. New combinations are the capability target: a flat memorizer must learn each pair; a template model can share the slot.
- this is parser headroom, not a trained LM. The decisive test is held-out fill composition, not the raw decision count.

## Architecture: TF-LM

Bytes -> deterministic tokenizer -> greedy template parse -> events.
An event is `T(key, fills)` or `L(token)`. The model is a
decoder-only transformer over event embeddings:

- template embedding = a small causal encoder of the fixed tokens
  plus a hole-type vector;
- fill embedding = a content encoder of the fill bytes, tied across
  templates, so a new combination lands in a shared space;
- heads: next template, per-slot fill under the hole type, and a
  byte fallback for novel values;
- exact output by table lookup; a digest verifies every event;
- library = templates + fills (the table above), installable and
  diffable like a repo pack.

One forward per event, not per byte. Fixed bytes are memcpy.
Novel fills or missed templates go to the byte fallback.

## Decisive experiment (T1)

Event model ~30M: d512, 12 layers, 8 heads, 512 events, batch 8,
AdamW lr 3e-4, 20k steps on 8 MB code. Baselines: the byte trunk
and a BPE token model at equal FLOPs. Split: hold out 20% of
(template, slot, fill) pairs while each held-out fill still appears
in other slots; the model must share content, not memorize the pair.

Gate: 100% exact round-trip, <=0.03 bpb regression against the byte
trunk, >=+10 points exact fill accuracy on held-out combinations,
and >=2x bytes/decision at equal quality.

T2: install a library from repo A on repo B with an empty-library
control; gate >=0.15 bpb improvement from installation alone.

Took 3.9 s CPU.
