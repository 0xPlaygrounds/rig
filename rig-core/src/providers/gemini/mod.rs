//! Google API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::google;
//!
//! let client = google::Client::new("YOUR_API_KEY");
//! 
//! let gemini_embedding_model = client.embedding_model(google::EMBEDDING_001);
//! ```

pub mod client;
pub mod completion;
pub mod embedding;

pub use client::Client;
