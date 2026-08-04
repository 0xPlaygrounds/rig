// The mistralrs cassette suite's async test bodies nest deep enough (provider
// future + agent driver future + cassette guard) to exceed rustc's default
// query depth when the whole workspace is checked with unified features.
#![recursion_limit = "256"]
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

#[path = "providers/mistralrs/mod.rs"]
mod mistralrs;
