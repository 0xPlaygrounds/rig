//! Xiaomi MiMo's endpoints and model identifiers.
//!
//! Configure chat requests with [`crate::providers::openai::wire::XIAOMIMIMO`].
//!
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, XIAOMIMIMO};
//! use rig_core::providers::xiaomimimo;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env_with(&XIAOMIMIMO)?.chat(xiaomimimo::MIMO_V2_5_PRO);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// OpenAI-compatible base URL.
pub const API_BASE_URL: &str = "https://api.xiaomimimo.com/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.xiaomimimo.com/anthropic/v1";

/// `mimo-v2-flash`
pub const MIMO_V2_FLASH: &str = "mimo-v2-flash";
/// `mimo-v2-omni`
pub const MIMO_V2_OMNI: &str = "mimo-v2-omni";
/// `mimo-v2-pro`
pub const MIMO_V2_PRO: &str = "mimo-v2-pro";
/// `mimo-v2.5`
pub const MIMO_V2_5: &str = "mimo-v2.5";
/// `mimo-v2.5-pro`
pub const MIMO_V2_5_PRO: &str = "mimo-v2.5-pro";
