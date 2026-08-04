# rig-memory

Conversation memory policies for the [Rig](https://github.com/0xPlaygrounds/rig)
agent framework.

`rig-core` ships the concrete in-process `InMemoryConversationMemory` store.
This crate adds reusable, *data-shaped* policies for the history a host loads
and appends — no behavior traits: policies, token counters, and compactors are
exhaustive enums.

- `MemoryPolicy::Noop` — identity.
- `MemoryPolicy::SlidingWindow` — keep the most recent `N` messages, demoting
  any leading orphan tool result.
- `MemoryPolicy::TokenWindow` — keep the most recent messages that fit within a
  token budget, counted by a `TokenCounter` (`Heuristic` or `Fixed`).
- `Compactor::Template` — a zero-dependency textual rollup (`TextSummary`) for
  rolling summaries.
- `PolicyMemory` — the store plus a policy and an optional compactor.

## Usage

Apply a policy directly to any history you loaded yourself:

```rust
use rig_core::completion::Message;
use rig_memory::MemoryPolicy;

let policy = MemoryPolicy::sliding_window(20);
let outcome = policy.apply(vec![Message::user("hi")]);
assert!(outcome.demoted.is_empty());
let _history = outcome.kept;
```

Or let `PolicyMemory` shape the store and report what fell out. Memory is
host-owned, so the recipe is load-before / append-after:

```rust
# fn run() -> Result<(), Box<dyn std::error::Error>> {
use rig_core::completion::Message;
use rig_memory::{Compactor, InMemoryConversationMemory, MemoryPolicy, PolicyMemory};

let memory = PolicyMemory::new(
    InMemoryConversationMemory::new(),
    MemoryPolicy::sliding_window(20),
)
.with_compactor(Compactor::template());

// Before the run: the policy-shaped history (rolling summary first, if any).
let history = memory.load("user-42")?;
# let _ = history;

// After the run: append its committed transcript and act on the outcome.
let outcome = memory.append("user-42", vec![Message::user("hi")])?;
for demoted in &outcome.demoted {
    // Archive into a long-tail store (vector RAG, episodic recall, ...).
    # let _ = demoted;
}
if let Some(request) = &outcome.compaction {
    memory.compact(request);
}
# Ok(()) }
```

Demotion and compaction are values, not callbacks: the host sees each demoted
message exactly once, when it appends, so there is no delivery watermark and no
idempotency contract to honour.
