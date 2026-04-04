//! Core integration tests that are not provider-specific.
//!
//! Run the target with:
//! `cargo test -p rig-core --test core`

#[cfg(feature = "derive")]
#[path = "core/embed_macro.rs"]
mod embed_macro;
#[path = "core/prompt_response_messages.rs"]
mod prompt_response_messages;
