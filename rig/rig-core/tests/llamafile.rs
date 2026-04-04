//! Llamafile integration tests.
//!
//! Run the full provider target with:
//! `cargo test -p rig-core --test llamafile`
//!
//! Run all ignored provider-backed tests with:
//! `cargo test -p rig-core --test llamafile -- --ignored`
//!
//! Run a single ignored smoke test with:
//! `cargo test -p rig-core --test llamafile llamafile::agent_with_llamafile::completion_smoke -- --ignored`

#[path = "common/support.rs"]
mod support;

#[path = "llamafile/mod.rs"]
mod llamafile;
