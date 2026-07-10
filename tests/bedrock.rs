#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[path = "common/cassette_safety.rs"]
mod cassette_safety;
#[path = "common/cassettes.rs"]
mod cassettes;
#[path = "common/support.rs"]
mod support;

#[cfg(feature = "bedrock")]
#[path = "providers/bedrock/mod.rs"]
mod bedrock;
