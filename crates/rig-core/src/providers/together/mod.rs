//! Together AI's model identifiers.
//!
//! Together AI is an OpenAI chat-completions dialect, so it has no client and
//! no models of its own:
//! [`openai::wire::TOGETHER`](crate::providers::openai::wire::TOGETHER)
//! carries the base URL, the `TOGETHER_API_KEY` variable, and the `/v1`
//! paths. What lives here is the model identifiers.
//!
//! # Example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::openai::wire::{OpenAI, TOGETHER};
//! use rig_core::providers::together;
//! use rig_reqwest::DefaultTransport;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let together = OpenAI::from_env_with(&TOGETHER)?.bound()?;
//! let embedding = together.embedding(together::BGE_BASE_EN_V1_5, None);
//! let chat = together.completion(together::MIXTRAL_8X7B_INSTRUCT_V0_1);
//! # let _ = (embedding, chat);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;

pub use completion::*;
pub use embedding::*;
