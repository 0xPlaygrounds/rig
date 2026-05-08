# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
