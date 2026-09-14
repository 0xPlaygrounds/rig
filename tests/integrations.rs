#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[cfg(feature = "bedrock")]
#[path = "integrations/bedrock/mod.rs"]
mod bedrock;
