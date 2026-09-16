//! Together AI's model identifiers.
//!
//! Together AI is an OpenAI chat-completions dialect, so it has no client and
//! no models of its own:
//! [`openai::wire::TOGETHER`](crate::providers::openai::wire::TOGETHER)
//! carries the base URL, the `TOGETHER_API_KEY` variable, and the `/v1`
//! paths. What lives here is the model identifiers.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, TOGETHER};
//! use rig_core::providers::together;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let together = OpenAI::from_env_with(&TOGETHER)?;
//! let embedding = together.embeddings(together::BGE_BASE_EN_V1_5, None);
//! let chat = together.chat(together::MIXTRAL_8X7B_INSTRUCT_V0_1);
//! # let _ = (embedding, chat);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;

pub use completion::*;
pub use embedding::*;
