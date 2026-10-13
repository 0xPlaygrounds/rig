# rig-memory

Conversations over time for the [Rig](https://github.com/0xPlaygrounds/rig)
agent framework, over rig-core's one `Conversation` type:

- `window`: what of a conversation a request sends (`LastMessages`,
  `TokenBudget`), never a tool result without its call.
- compaction: a `CompactionPolicy` that says when (near the model's window)
  and where to cut and plans the summary request, a `SummaryState` carried
  from one summary to the next, `apply_summary`, and `ClearToolOutputs`,
  which frees context without a model call. rig-harness's compaction plugin
  is built on it.
- saving and resuming: the session format `rig-session/2` (one JSON-lines
  log per agent, images kept as blobs), a `Store` trait with `MemoryStore`
  and, with feature `fs`, `DirStore`, and its `Files` half for named files
  on their own (`DirFiles` over a plain directory, where rig-harness keeps
  plugins' data per project and per user); `load`/`save` for one conversation,
  `Session` for a whole session, `Writer` for a running host, `fork`,
  `history` and `as_of`. Reading is strict: a session that cannot be read
  whole is refused, and nothing is written. `Session::read_lenient` reads
  each log that reads and names the others, for readers such as an export.

With default features the crate is pure data and builds for wasm32.

```rust,ignore
use rig_core::message::Message;
use rig_memory::window::{LastMessages, view};
use rig_memory::{MemoryStore, load, save};

let store = MemoryStore::default();
let mut conversation = load(&store, "user-123")?;
conversation.push(Message::user("My name is Alice."));
save(&store, "user-123", &mut conversation)?;
let request = view(conversation.messages(), &LastMessages(20));
```
