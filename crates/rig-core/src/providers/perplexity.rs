//! Perplexity's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`PERPLEXITY`](crate::providers::openai::wire::PERPLEXITY) dialect,
//! using `PERPLEXITY_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::perplexity;
//!
//! # fn run() -> Result<(), rig_core::RigError> {
//! let sonar = perplexity::from_env()?.chat(perplexity::SONAR);
//! # let _ = sonar;
//! # Ok(())
//! # }
//! ```

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::PERPLEXITY, "Perplexity");
