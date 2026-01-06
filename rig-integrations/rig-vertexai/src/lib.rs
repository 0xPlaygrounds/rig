pub mod auth;
pub mod client;
pub mod completion;
pub mod streaming;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
pub use streaming::StreamingCompletionModel;
