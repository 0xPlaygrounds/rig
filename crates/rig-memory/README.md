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
adds `conformance::check_store`, the contract suite every store's tests run against it, and
`migration::testing`, a synthetic earlier format with its migration.

A session in another format is refused untouched. `migration::SHIPPED` is the chain of steps, each
taking the whole session from one format to the next, that brings an earlier one to the current
format through the session's own `Store`, keeping a copy of it as it was under `rig/kept/`. It has
no step yet, as no earlier format was released. See the crate docs for the API.

`Session::save` refuses a snapshot with `SessionError::Changed` if any log has
changed since it was read or last saved, including saved values and rewrites.
Read the session again before reapplying edits after a conflict. Construct a new
session with `Session::default()` and then set its public fields.
