# rig-memory

Saved conversations for [Rig](https://github.com/0xPlaygrounds/rig): recent messages,
compaction, and sessions saved as JSON-lines logs (`rig-session/4`) that can be resumed,
rewritten, and an agent's log forked up to a turn boundary. With default features it is pure data and builds for wasm32;
feature `fs` adds `DirStore` and `DirFiles`. See the crate docs for the API.
