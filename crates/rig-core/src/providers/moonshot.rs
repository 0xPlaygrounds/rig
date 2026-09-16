//! Moonshot AI (Kimi) endpoints and model identifiers.
//!
//! Moonshot serves the same models over two wires from two regions, so this
//! module is data for all four combinations and nothing else:
//!
//! - the OpenAI chat-completions wire, as
//!   [`openai::wire::MOONSHOT`](crate::providers::openai::wire::MOONSHOT)
//!   (global) and
//!   [`openai::wire::MOONSHOT_CHINA`](crate::providers::openai::wire::MOONSHOT_CHINA)
//!   — one dialect at two base URLs, also serving the model listing
//!   (`GET /models`). Its quirks are Moonshot's: `json_schema` response
//!   formats are refused, a forced specific tool is an error, and
//!   `tool_choice: "required"` is coerced to `auto` with a steering message;
//! - the Anthropic Messages wire, as
//!   [`anthropic::wire::MOONSHOT`](crate::providers::anthropic::wire::MOONSHOT),
//!   whose China endpoint is [`CHINA_ANTHROPIC_API_BASE_URL`].
//!
//! Every combination reads `MOONSHOT_API_KEY`.
//!
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::providers::moonshot;
//! use rig_core::providers::openai::wire::{MOONSHOT, MOONSHOT_CHINA, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let kimi = OpenAI::from_env_with(&MOONSHOT)?.chat(moonshot::KIMI_K3);
//!
//! // The China entrypoint is the same dialect at `CHINA_API_BASE_URL`.
//! let china = OpenAI::from_env_with(&MOONSHOT_CHINA)?.chat(moonshot::KIMI_K3);
//! # let _ = (kimi, china);
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::anthropic::wire::{Anthropic, MOONSHOT};
//! use rig_core::providers::moonshot;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let kimi = Anthropic::from_env_with(&MOONSHOT)?
//!     .with_base_url(moonshot::CHINA_ANTHROPIC_API_BASE_URL)
//!     .messages(moonshot::KIMI_K3);
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

// Model IDs follow <https://platform.kimi.ai/docs/models>, which also lists the
// discontinued `moonshot-v1-*`, `kimi-k2*` and `kimi-k2.5` IDs.

/// Kimi K3 — flagship multimodal model with a 1M-token context window.
pub const KIMI_K3: &str = "kimi-k3";

/// Kimi K2.7 Code — coding-focused model with a 256K context window.
pub const KIMI_K2_7_CODE: &str = "kimi-k2.7-code";

/// Kimi K2.7 Code (high-speed) — faster-output variant of `kimi-k2.7-code`.
pub const KIMI_K2_7_CODE_HIGHSPEED: &str = "kimi-k2.7-code-highspeed";

/// Kimi K2.6 — general-purpose multimodal model with thinking and non-thinking modes, 256K context.
pub const KIMI_K2_6: &str = "kimi-k2.6";
