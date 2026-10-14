# rig-memory

Conversations over time for the [Rig](https://github.com/0xPlaygrounds/rig)
agent framework, over rig-core's one `Conversation` type:

- `window`: what of a conversation a request sends (`LastMessages`,
  `TokenBudget`), never a tool result without its call.
- compaction: a `CompactionPolicy` that says when (near the model's window)
  and where to cut and plans the summary request, a `SummaryState` carried
  from one summary to the next, `apply_summary`, which keeps what each
  compaction replaced in a `rig.compaction` aside (`Compacted`), so
  `undo_summary` puts the newest one's entries back with their ids, and
  `ClearToolOutputs`, which frees context without a model call.
  rig-harness's compaction plugin is built on it.
- saving and resuming: the session format `rig-session/3` (one JSON-lines
  log per agent, images kept as blobs), a `Store` trait with `MemoryStore`
  and, with feature `fs`, `DirStore`, and its `Files` half for named files
  on their own (`DirFiles` over a plain directory, where rig-harness keeps
  plugins' data per project and per user); `load`/`save` for one conversation,
  `Session` for a whole session, `Writer` for a running host, `history` and
  `as_of`. Reading is strict: a session that cannot be read whole, or one a
  newer rig saved, is refused, and nothing is written. Sessions from
  `rig-session/2` on read through a chain of steps (`upgrade`) and are
  rewritten in the current format the first time anything writes to them;
  saved values carry their own version (`Versioned`, `Upgrades`).
  `Session::read_lenient` reads each log that reads and names the others,
  for readers such as an export.
- rewriting and copying: records are appended while the session runs.
  Nothing is rewritten except through rig-memory: `rewrite` (or
  `Writer::rewrite` for a running session) maps every saved record in place,
  keeping each record's place and stamp, and every file of the store,
  through a `Rewrite` (`Strings` for a redaction or a path remap,
  `Messages`, or hooks of its own); each log is replaced atomically, gets a
  fresh lineage and records who rewrote it. A running writer's saves and a
  rewrite never interleave; a value read before it is refused by `save`
  (`SessionError::Changed`); the crate docs list the whole promise. `fork`
  copies a session as it was before a message through the same `Rewrite`
  (`NoRewrite` for none), with what the copied messages' asides say now.
- what a session is: `SessionMeta` (what a listing shows: title, labels,
  when it changed), `ForkedFrom` (the session it was forked from), and a
  saved agent's model and title under stable keys (`SavedAgent::model`,
  `title`, `Session::main_agent`), all readable without the host that wrote
  them.

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
