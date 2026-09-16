//! Xiaomi MiMo's endpoints and model identifiers.
//!
//! Xiaomi serves MiMo over two wires, so this module is data for both and
//! nothing else:
//!
//! - the OpenAI chat-completions wire, as
//!   [`openai::wire::XIAOMIMIMO`](crate::providers::openai::wire::XIAOMIMIMO),
//!   which also serves the model listing (`GET /models`);
//! - the Anthropic Messages wire, as
//!   [`anthropic::wire::XIAOMIMIMO`](crate::providers::anthropic::wire::XIAOMIMIMO).
//!
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//!
//! # OpenAI-compatible example
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
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::anthropic::wire::{Anthropic, XIAOMIMIMO};
//! use rig_core::providers::xiaomimimo;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Anthropic::from_env_with(&XIAOMIMIMO)?.messages(xiaomimimo::MIMO_V2_5_PRO);
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
