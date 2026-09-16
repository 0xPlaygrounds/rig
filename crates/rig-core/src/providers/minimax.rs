//! MiniMax's endpoints and model identifiers.
//!
//! MiniMax serves the same models over two wires from two regions, so this
//! module is data for all four combinations and nothing else:
//!
//! - the OpenAI chat-completions wire, as
//!   [`openai::wire::MINIMAX`](crate::providers::openai::wire::MINIMAX)
//!   (global) and
//!   [`openai::wire::MINIMAX_CHINA`](crate::providers::openai::wire::MINIMAX_CHINA)
//!   — one dialect at two base URLs, also serving the model listing
//!   (`GET /models`);
//! - the Anthropic Messages wire, as
//!   [`anthropic::wire::MINIMAX`](crate::providers::anthropic::wire::MINIMAX),
//!   whose China endpoint is [`CHINA_ANTHROPIC_API_BASE_URL`].
//!
//! Every combination reads `MINIMAX_API_KEY`.
//!
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::providers::minimax;
//! use rig_core::providers::openai::wire::{MINIMAX, MINIMAX_CHINA, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env_with(&MINIMAX)?.chat(minimax::MINIMAX_M2_7);
//!
//! // The China entrypoint is the same dialect at `CHINA_API_BASE_URL`.
//! let china = OpenAI::from_env_with(&MINIMAX_CHINA)?.chat(minimax::MINIMAX_M2_7);
//! # let _ = (model, china);
//! # Ok(())
//! # }
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::providers::anthropic::wire::{Anthropic, MINIMAX};
//! use rig_core::providers::minimax;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Anthropic::from_env_with(&MINIMAX)?
//!     .with_base_url(minimax::CHINA_ANTHROPIC_API_BASE_URL)
//!     .messages(minimax::MINIMAX_M2);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.minimax.io/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.minimaxi.com/v1";
/// Global Anthropic-compatible base URL.
pub const GLOBAL_ANTHROPIC_API_BASE_URL: &str = "https://api.minimax.io/anthropic";
/// China Anthropic-compatible base URL.
pub const CHINA_ANTHROPIC_API_BASE_URL: &str = "https://api.minimaxi.com/anthropic";

/// `MiniMax-M2.7`
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
/// `MiniMax-M2.7-highspeed`
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
/// `MiniMax-M2.5`
pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
/// `MiniMax-M2.5-highspeed`
pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";
/// `MiniMax-M2.1`
pub const MINIMAX_M2_1: &str = "MiniMax-M2.1";
/// `MiniMax-M2.1-highspeed`
pub const MINIMAX_M2_1_HIGHSPEED: &str = "MiniMax-M2.1-highspeed";
/// `MiniMax-M2`
pub const MINIMAX_M2: &str = "MiniMax-M2";
