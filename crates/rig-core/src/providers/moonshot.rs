//! Moonshot AI (Kimi) endpoints and model identifiers.
//!
//! Configure chat requests with [`crate::providers::openai::wire::MOONSHOT`]
//! or its China variant. Environment configuration reads `MOONSHOT_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::moonshot;
//! use rig_core::providers::openai::wire::{MOONSHOT, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let kimi = OpenAI::from_env_with(&MOONSHOT)?.chat(moonshot::KIMI_K3);
//! # let _ = kimi;
//! # Ok(())
//! # }
//! ```

/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.moonshot.ai/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.moonshot.cn/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.moonshot.ai/anthropic";
/// China Anthropic-compatible base URL.
pub const CHINA_ANTHROPIC_API_BASE_URL: &str = "https://api.moonshot.cn/anthropic";

/// Identifier for the Kimi K3 model.
pub const KIMI_K3: &str = "kimi-k3";

/// Identifier for the Kimi K2.7 Code model.
pub const KIMI_K2_7_CODE: &str = "kimi-k2.7-code";

/// Identifier for the high-speed Kimi K2.7 Code model.
pub const KIMI_K2_7_CODE_HIGHSPEED: &str = "kimi-k2.7-code-highspeed";

/// Identifier for the Kimi K2.6 model.
pub const KIMI_K2_6: &str = "kimi-k2.6";
