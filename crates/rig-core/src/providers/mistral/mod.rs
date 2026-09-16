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
//! [`embedding`], [`transcription`]) and [`CompletionResponse`] — the typed
//! read of Mistral's own chat reply document. Transcripts have no typed read
//! of their own: Mistral's segments and audio-second accounting stay on the
//! normalized response's `raw` value.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::mistral;
//! use rig_core::providers::openai::wire::{MISTRAL, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mistral = OpenAI::from_env_with(&MISTRAL)?;
//! let small = mistral.chat(mistral::MISTRAL_SMALL);
//! let embed = mistral.embeddings(mistral::embedding::MISTRAL_EMBED, None);
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
