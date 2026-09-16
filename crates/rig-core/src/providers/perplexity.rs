//! Perplexity's model identifiers.
//!
//! Perplexity is an OpenAI chat-completions dialect, so it has no client and
//! no completion model of its own: everything but these constants is the
//! shared wire.
//! [`openai::wire::PERPLEXITY`](crate::providers::openai::wire::PERPLEXITY)
//! holds the base URL, the `PERPLEXITY_API_KEY` variable, and the quirks that
//! make it Perplexity — no tool calling, no `response_format`, no
//! `stream_options`, a plain-text message history with strict user/assistant
//! alternation, and no endpoint that checks a credential without spending
//! tokens.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, PERPLEXITY};
//! use rig_core::providers::perplexity;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = OpenAI::from_env_with(&PERPLEXITY)?.chat(perplexity::SONAR);
//! # let _ = sonar;
//! # Ok(())
//! # }
//! ```

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";
