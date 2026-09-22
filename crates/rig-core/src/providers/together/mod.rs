//! Together AI's model identifiers.
//!
//! Configure requests with [`crate::providers::openai::wire::TOGETHER`],
//! using `TOGETHER_API_KEY`.
//!
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
