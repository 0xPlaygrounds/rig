//! Zai API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::zai;
//!
//! let client = zai::Client::new("YOUR_API_KEY");
//!
//! let glm = client.completion_model(zai::GLM_4_7);
//! ```

pub mod client;
pub mod completion;
pub mod decoders;
pub mod streaming;

pub use client::{Client, ClientBuilder};
