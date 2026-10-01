# rig-memory

Conversation memory policies for the [Rig](https://github.com/0xPlaygrounds/rig)
agent framework.

`rig-core` ships the `ConversationMemory` trait and an in-process
`InMemoryConversationMemory` backend. This crate adds a file backend and
reusable named policies for shaping loaded history before it is sent to the
model:

- [`NoopMemoryPolicy`] — identity policy, useful as a default.
- [`SlidingWindowMemory`] — keep at most the most recent `N` messages.
- [`TokenWindowMemory`] — keep the most recent messages that fit within a token
  budget supplied by a [`TokenCounter`].

Both window policies remove the leading prefix through any tool-result
messages whose assistant calls were truncated. Results are detected anywhere
in a user message, including after text and across intervening system messages.
Cleanup stops at the first retained assistant; later paired exchanges remain
intact. Whole removed messages are included in `apply_with_demoted`'s demoted
prefix in original order, including accompanying text, so demotion hooks lose
no content.

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

## File backend

The `file` feature adds `FileConversationMemory` on native targets. It stores
each conversation as a JSON Lines file in one directory, one message per line:

```rust,ignore
use rig_memory::{ConversationMemory, FileConversationMemory};

let memory = FileConversationMemory::new("conversations");
memory.append(&"thread-1".into(), messages).await?;
let history = memory.load(&"thread-1".into()).await?;
let newest_first = memory.list().await?;
```

- File names are derived from conversation ids, so any id is safe: no path
  traversal, no collisions on case-insensitive file systems, and long ids are
  hashed with the id kept in a sibling `.id` file.
- `append` syncs its lines to disk before returning. A last line left
  incomplete by a crash is skipped on load and truncated by the next append;
  a malformed line elsewhere is an error.
- `replace` swaps a conversation's whole history atomically, for example after
  compaction.
- Operations on one conversation through one value, or its clones, run one at
  a time. Nothing locks across separately constructed values or processes, so
  give each conversation one writer at a time.
