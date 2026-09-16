//! Z.AI's endpoints and model identifiers.
//!
//! Z.AI serves the same models over two wires, so this module is data for
//! both and nothing else:
//!
//! - the OpenAI chat-completions wire, as
//!   [`openai::wire::ZAI`](crate::providers::openai::wire::ZAI) (general
//!   platform) and
//!   [`openai::wire::ZAI_CODING`](crate::providers::openai::wire::ZAI_CODING)
//!   (coding platform) — one dialect at two base URLs;
//! - the Anthropic Messages wire, as
//!   [`anthropic::wire::ZAI`](crate::providers::anthropic::wire::ZAI), for
//!   tools that speak Claude Code's format.
//!
//! Both read `ZAI_API_KEY`.
//!
//! # OpenAI-compatible example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::openai::wire::{OpenAI, ZAI, ZAI_CODING};
//! use rig_core::providers::zai;
//! use rig_reqwest::DefaultTransport;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let glm_4_6 = OpenAI::from_env_with(&ZAI)?
//!     .bound()?
//!     .completion(zai::GLM_4_6);
//!
//! // The coding platform is the same dialect at `CODING_API_BASE_URL`.
//! let coding = OpenAI::from_env_with(&ZAI_CODING)?
//!     .bound()?
//!     .completion(zai::GLM_4_6);
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::anthropic::wire::{Anthropic, ZAI};
//! use rig_core::providers::zai;
//! use rig_reqwest::DefaultTransport;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let glm_4_6 = Anthropic::from_env_with(ZAI)?
//!     .bound()?
//!     .completion(zai::GLM_4_6);
//! # Ok(())
//! # }
//! ```

/// General-purpose OpenAI-compatible base URL.
pub const GENERAL_API_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
/// Coding-focused OpenAI-compatible base URL.
pub const CODING_API_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.z.ai/api/anthropic";

/// `glm-4.6`
pub const GLM_4_6: &str = "glm-4.6";
/// `glm-4.6-air`
pub const GLM_4_6_AIR: &str = "glm-4.6-air";
/// `glm-4.6-x`
pub const GLM_4_6_X: &str = "glm-4.6-x";
/// `glm-4.5`
pub const GLM_4_5: &str = "glm-4.5";
/// `glm-4.5-air`
pub const GLM_4_5_AIR: &str = "glm-4.5-air";
/// `glm-4.5v`
pub const GLM_4_5V: &str = "glm-4.5v";
/// `glm-4.5-airx`
pub const GLM_4_5_AIRX: &str = "glm-4.5-airx";
