//! The plugins other corpus plugins build on: confirmations, plan mode,
//! protected paths, permission modes and few-shot examples, and the editor
//! lookup and read-only shell rule they share.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod confirm_destructive;
pub mod editor;
pub mod few_shot_examples;
pub mod permission_modes;
pub mod plan_mode;
pub mod protected_paths;
pub mod shell;
