#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/reasoning.rs"]
mod reasoning;
#[path = "common/support.rs"]
mod support;

#[path = "providers/moonshot/mod.rs"]
mod moonshot;
