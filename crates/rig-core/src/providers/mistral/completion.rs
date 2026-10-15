//! Mistral chat model identifiers.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//! let model = mistral::from_env()?.chat(mistral::MISTRAL_SMALL);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";

/// Voxtral Small (latest version): a chat model that takes audio input. It
/// is not a transcription model; transcribe with
/// [`VOXTRAL_MINI`](super::VOXTRAL_MINI).
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";
