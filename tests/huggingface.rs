#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_test_support::support;

#[path = "providers/huggingface/mod.rs"]
mod huggingface;
