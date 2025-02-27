//! Cohere API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::cohere;
//!
//! let client = cohere::Client::new("YOUR_API_KEY");
//!
//! let command_r = client.completion_model(cohere::COMMAND_R);
//! ```


pub mod client;
pub mod completion;

pub use client::{Client};
pub use completion::{
    
};
