//! xAi API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::xai;
//!
//! let client = xai::Client::new("YOUR_API_KEY");
//!
//! let groq_embedding_model = client.embedding_model(xai::v1);
//! ```

pub mod client;
pub mod completion;
pub mod embedding;

pub use client::Client;
pub use completion::GROK_BETA;
pub use embedding::EMBEDDING_V1;
