//! The plugins other corpus plugins build on: confirmations, plan mode, protected paths, permission modes and few-shot examples.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod confirm_destructive;
pub mod few_shot_examples;
pub mod permission_modes;
pub mod plan_mode;
pub mod protected_paths;
