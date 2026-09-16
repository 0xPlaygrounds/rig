//! Mistral's model identifiers and its own view of a reply.
//!
//! Mistral is an OpenAI chat-completions dialect, so it has no client and no
//! models of its own:
//! [`openai::wire::MISTRAL`](crate::providers::openai::wire::MISTRAL) carries
//! the base URL, the `MISTRAL_API_KEY` variable, the `/v1`-prefixed chat,
//! embeddings, models and transcription paths, the `mistral-correlation-id`
//! request id, and Mistral's body rewrite.
//!
//! What lives here is data: the model identifiers ([`completion`],
//! [`embedding`], [`transcription`]) and the typed reads of Mistral's own
//! reply documents ([`CompletionResponse`], [`MistralTranscriptionResponse`]).
//!
//! # Example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::mistral;
//! use rig_core::providers::openai::wire::{MISTRAL, OpenAI};
//! use rig_reqwest::DefaultTransport;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mistral = OpenAI::from_env_with(&MISTRAL)?.bound()?;
//! let small = mistral.completion(mistral::MISTRAL_SMALL);
//! let embed = mistral.embedding(mistral::embedding::MISTRAL_EMBED, None);
//! # let _ = (small, embed);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod transcription;

pub use completion::*;
pub use embedding::*;
pub use transcription::*;
