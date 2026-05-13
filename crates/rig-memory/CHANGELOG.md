# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.1.1](https://github.com/0xPlaygrounds/rig/compare/rig-memory-v0.1.0...rig-memory-v0.1.1) - 2026-05-13

### Added

- *(memory)* add Compactor trait, CompactingMemory adapter, and TemplateCompactor ([#1748](https://github.com/0xPlaygrounds/rig/pull/1748)) (by @ForeverAngry)
- *(memory)* Rig-managed conversation memory + rig-memory companion crate ([#1702](https://github.com/0xPlaygrounds/rig/pull/1702)) (by @ForeverAngry)

### Other

- Memory adapter cancellation safety and trait-object forwarding ([#1756](https://github.com/0xPlaygrounds/rig/pull/1756)) (by @gold-silver-copper) - #1756
- Add demotion hooks for bounded conversation memory ([#1737](https://github.com/0xPlaygrounds/rig/pull/1737)) (by @ForeverAngry) - #1737

### Contributors

* @gold-silver-copper
* @ForeverAngry

### Added

- `Compactor` trait (re-exported from `rig_core::memory`) +
  `CompactingMemory<M, P, C>` adapter and a `TemplateCompactor`
  reference implementation. Where `DemotingPolicyMemory` only
  *observes* messages a policy truncates out of active history, a
  `Compactor` *substitutes* them: it derives a single `Message`-shaped
  `Artifact` from the evicted prefix (and, optionally, the previous
  summary), and `CompactingMemory` splices that artifact at the front
  of the loaded history. The resulting prompt shape is
  `[summary_message, ...kept_window]`, with the summary rolling
  forward on every load that produces newly-evicted messages — the
  canonical recursive-summary pattern for long-running agents.
  Concurrent loads on the same `conversation_id` are serialised at
  the compaction seam via an in-flight gate: only one load at a time
  invokes the compactor; others observe the gate and immediately
  return the previously-stored summary spliced in front of `kept`,
  without re-running the compactor. Watermarks and the carry-over
  summary are in-process only — `Compactor` implementations with
  durable side effects (LLM calls, vector-store writes) must
  deduplicate. `clear` drops the carry-over so a freshly-populated
  backend re-compacts from scratch. `TemplateCompactor` is a
  zero-dependency, no-LLM rollup useful as a default and for tests;
  it produces a `TextSummary` that converts into a `Message::System`
  with header + previous-summary + per-line `role: text` body. The
  rollup represents out-of-band context about the prior conversation
  rather than a turn from any participant, so the system role is the
  semantically correct framing across providers. `TemplateCompactor`
  exposes `with_max_bytes` so long-running conversations can cap the
  rolled-up text; when exceeded, the oldest portion of the body is
  dropped at a UTF-8 boundary and replaced with a `"[…truncated…]"`
  marker, preserving the most recent context. Note that the spliced
  summary sits **outside** the wrapped `MemoryPolicy`'s budget, so
  pairing `CompactingMemory` with a token-budgeted policy requires a
  bounded compactor (or accepting that the loaded prompt may exceed
  the policy budget by the artifact size).

- `DemotionHook` trait + `DemotingPolicyMemory<M, P, H>` adapter and a
  `NoopDemotionHook` no-op default. The trait itself lives in
  `rig_core::memory` (re-exported here) so any memory backend can
  implement it without taking a `rig-memory` dependency; the composing
  adapter lives in this crate. `DemotingPolicyMemory` calls the hook
  with messages that the policy truncated out of active history,
  turning eviction into demotion. It tracks per-conversation demotion
  watermarks so repeated `load` calls do not replay the same demoted
  messages into append-only long-tail stores. Concurrent loads on the
  same `conversation_id` are serialised at the demotion seam via an
  in-flight gate: only one load at a time delivers to the hook;
  others observe the gate and return the truncated history without
  re-firing. Watermarks are in-process only — `DemotionHook`
  implementations must be idempotent on `(conversation_id, messages)`
  to survive process restarts. Bridges `SlidingWindowMemory` /
  `TokenWindowMemory` to long-tail stores such as `MemvidPersistHook`
  without coupling either crate to the other.
- `DemotingPolicyMemory::forget(conversation_id)` and
  `tracked_conversations()` for explicit watermark-map cleanup and
  leak diagnostics. Both are infallible: a poisoned internal lock
  is treated as "nothing to forget / zero tracked" rather than a
  caller-visible error.
- `MemoryPolicy::apply_with_demoted` companion method that reports
  `(kept, demoted)`. `apply` remains the required method; the default
  `apply_with_demoted` returns `(apply(...)?, Vec::new())` so existing
  policies keep compiling unchanged. `SlidingWindowMemory` and
  `TokenWindowMemory` override it to populate the demoted prefix that
  `DemotingPolicyMemory` hands to the hook.
- `HeuristicTokenCounter` — provider-agnostic, zero-dependency
  `TokenCounter` implementation that approximates token cost from
  UTF-8 byte lengths (`str::len`, O(1)). Ships `default` / `openai` /
  `anthropic` / `gemini` presets so
  `TokenWindowMemory::new(budget, HeuristicTokenCounter::default())`
  works out of the box without a tokenizer dependency. Also handles the
  `Message::System` variant and tool-call argument payloads. The
  configurable ratio is named `bytes_per_token` to match the
  implementation; for ASCII text bytes and characters coincide, and
  for non-ASCII text the counter slightly over-estimates, which is the
  safe direction for a hard budget.
- `PolicyMemory<M, P>` adapter — wrap any `ConversationMemory` with a
  `MemoryPolicy` and propagate policy failures to the caller as
  `MemoryError::Policy`. Hard-fail counterpart to
  `InMemoryConversationMemory::with_filter` + `IntoFilter::into_filter`,
  which degrade to identity on policy error.

## [0.1.0] - Initial release

### Added

- `MemoryPolicy` trait for shaping loaded conversation history.
- `IntoFilter` blanket extension for converting any policy into a
  `MessageFilter` closure consumable by
  `rig::memory::InMemoryConversationMemory::with_filter`.
- `NoopMemoryPolicy` — identity policy.
- `SlidingWindowMemory::last_messages(n)` — keep the most recent `n` messages,
  dropping any leading orphan tool result.
- `TokenCounter` trait (with a blanket impl for any `Fn(&Message) -> usize`)
  and `TokenWindowMemory` — keep the most recent messages that fit within a
  token budget.
- Re-exports of `ConversationMemory`, `InMemoryConversationMemory`, and
  `MemoryError` from `rig-core` so callers depend on a single crate.
