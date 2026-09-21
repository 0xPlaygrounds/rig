//! Z.AI's endpoints and model identifiers.
//!
//! Configure chat requests with [`crate::providers::openai::wire::ZAI`] or
//! [`crate::providers::openai::wire::ZAI_CODING`], using `ZAI_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, ZAI};
//! use rig_core::providers::zai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let glm_4_6 = OpenAI::from_env_with(&ZAI)?.chat(zai::GLM_4_6);
//! # let _ = glm_4_6;
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
