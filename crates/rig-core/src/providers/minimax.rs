//! MiniMax's endpoints and model identifiers.
//!
//! Global and China configurations support chat-completions and Messages wires,
//! using `MINIMAX_API_KEY`. [`from_env`] and [`new`] build a client on the
//! chat-completions [`MINIMAX`](crate::providers::openai::wire::MINIMAX)
//! dialect; [`anthropic_from_env`] and [`anthropic_new`] build a Messages-format
//! client.
//!
//! ```no_run
//! use rig_core::providers::minimax;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = minimax::from_env()?.chat(minimax::MINIMAX_M2_7);
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

crate::providers::internal::client::openai_vendor!(
    crate::providers::openai::wire::MINIMAX,
    "MiniMax"
);
crate::providers::internal::client::anthropic_vendor!(
    crate::providers::anthropic::wire::MINIMAX,
    "MiniMax",
    anthropic_from_env,
    anthropic_new
);
