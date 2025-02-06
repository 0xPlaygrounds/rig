//! Together AI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::together_ai;
//!
//! let client = together_ai::Client::new("YOUR_API_KEY");
//!
//! let together_embedding_model = client.embedding_model(together_ai::EMBEDDING_V1);
//! ```

pub mod client;
pub mod completion;
pub mod embedding;

pub use client::Client;
pub use completion::TOGETHER_MODEL; // Assuming this is the name for a completion model, adjust as needed
pub use embedding::EMBEDDING_V1;
