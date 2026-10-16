# rig-memory

Saved conversations for [Rig](https://github.com/0xPlaygrounds/rig): recent messages,
compaction, and sessions saved as JSON-lines logs (`rig-session/4`) that can be resumed,
rewritten, forked and split. With default features it is pure data and builds for wasm32;
feature `fs` adds `DirStore` and `DirFiles`. See the crate docs for the API.
