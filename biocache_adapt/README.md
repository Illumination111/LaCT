# BioCache adapter for LaCT

This package is a dependency-free **proxy adapter**, not a complete VLM
integration. It exposes the LaCT memory boundary as:

- `persistent_state`: explicit `fast_weights` (`w0`, `w1`, `w2`), `momentum`,
  and `local_window`
- `static_extra_state`: the wrapped layer or other caller-owned static state
- `temporary_state`: per-call scratch data

`LaCTBioCacheAdapter.from_layer(layer)` exports the layer's initial fast-weight
parameters and `window_size`. The upstream minimal layer does not return its
sequence-updated fast weights or momentum, so callers must supply an
`update_fn` (or an explicit `LaCTPersistentState`) if they instrument those
values. The adapter does not claim to capture state that upstream code
discards.

The common callable API is `reset()`, `update()`, `query()`,
`persistent_state()`, `static_extra_state()`, and `temporary_state()`. It also
provides recursive byte accounting, operation timing, configurable byte/time
budgets, a manifest schema, and JSONL schemas for delayed recall and VNBench
features.

No-model validation:

```bash
python -m biocache_adapt --dry-run --pretty
python -m biocache_adapt --help
```
