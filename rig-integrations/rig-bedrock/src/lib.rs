#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]

pub mod client;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod streaming;
pub mod types;
