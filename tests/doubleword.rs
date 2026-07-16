#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/support.rs"]
mod support;

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;

#[path = "providers/doubleword/mod.rs"]
mod doubleword;
