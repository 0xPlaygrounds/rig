//! The integration tests of rig-memory, one module per suite, linked into one
//! binary.

mod migration_policy;
// Public, as its items were at a crate root: they are a compile-time check
// that no test calls, so a private module would leave them unused.
pub mod wasm_store;
