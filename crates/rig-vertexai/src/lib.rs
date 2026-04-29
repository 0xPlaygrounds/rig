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
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
