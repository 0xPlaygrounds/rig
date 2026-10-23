# rig-memory

Saved conversations for [Rig](https://github.com/0xPlaygrounds/rig): recent messages,
compaction, and sessions saved as JSON-lines logs (`rig-session/4`) that can be resumed,
rewritten, and an agent's log forked up to a turn boundary. With default features it is pure data and builds for wasm32;
feature `fs` adds `DirStore` and `DirFiles`. `load_async` and `save_async` keep a conversation in
any `AsyncStore`, such as a database or IndexedDB, and every local store is one; rig-agent's
`AgentBuilder::memory` keeps an agent's conversation there. A store appends to a log only if the
log is still as long as the writer read it, checking and appending in one step, so of two saves
that read one log the second is refused. Database stores ship in `rig-sqlite` (`SqliteSessionStore`)
and `rig-postgres` (`PostgresSessionStore`) behind their `memory` feature. Feature `test-utils`
adds `conformance::check_store`, the contract suite every store's tests run against it. See the
crate docs for the API.
