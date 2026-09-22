//! Perplexity's model identifiers.
//!
//! Configure requests with [`crate::providers::openai::wire::PERPLEXITY`],
//! using `PERPLEXITY_API_KEY`.
//!
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
