//! Perplexity's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`PERPLEXITY`](crate::providers::openai::wire::PERPLEXITY) dialect,
//! using `PERPLEXITY_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::perplexity;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = perplexity::from_env()?.chat(perplexity::SONAR);
//! # let _ = sonar;
//! # Ok(())
//! # }
//! ```

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";

pub mod extension;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "perplexity";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::PERPLEXITY, "Perplexity");
