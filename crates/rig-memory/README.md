# rig-memory

Conversation memory policies for the [Rig](https://github.com/0xPlaygrounds/rig)
agent framework.

`rig-core` ships the `ConversationMemory` trait and an in-process
`InMemoryConversationMemory` backend. This crate provides reusable named
policies for shaping loaded history before it is sent to the model:

- [`NoopMemoryPolicy`] — identity policy, useful as a default.
- [`SlidingWindowMemory`] — keep the most recent `N` messages, dropping any
  leading orphan tool result.
- [`TokenWindowMemory`] — keep the most recent messages that fit within a token
  budget supplied by a [`TokenCounter`].

## Usage

```rust,no_run
use rig_memory::{InMemoryConversationMemory, SlidingWindowMemory, IntoFilter};

let memory = InMemoryConversationMemory::new()
    .with_filter(SlidingWindowMemory::last_messages(20).into_filter());
```

For backends other than `InMemoryConversationMemory`, apply a policy directly:

```rust,ignore
use rig_memory::{MemoryPolicy, SlidingWindowMemory};

let policy = SlidingWindowMemory::last_messages(20);
let trimmed = policy.apply(loaded_messages)?;
```

To wrap any backend with a policy and propagate policy errors to the caller
(rather than silently degrading to identity on failure), use `PolicyMemory`:

```rust,no_run
use rig_memory::{InMemoryConversationMemory, PolicyMemory, SlidingWindowMemory};

let memory = PolicyMemory::new(
    InMemoryConversationMemory::new(),
    SlidingWindowMemory::last_messages(20),
);
```
