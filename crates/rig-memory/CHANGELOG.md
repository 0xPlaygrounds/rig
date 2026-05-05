# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
