# rig-memory

Saved conversations for [Rig](https://github.com/0xPlaygrounds/rig): recent messages,
compaction, and sessions saved as JSON-lines logs (`rig-session/4`) that can be resumed,
rewritten, and an agent's log forked up to a turn boundary. With default features it is pure data and builds for wasm32;
feature `fs` adds `DirStore` and `DirFiles`. `load_async` and `save_async` keep a conversation in
any `AsyncStore`, such as a database or IndexedDB, and every local store is one; rig-agent's
`AgentBuilder::memory` keeps an agent's conversation there. See the crate docs for the API.
