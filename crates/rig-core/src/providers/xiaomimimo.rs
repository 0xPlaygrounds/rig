//! Xiaomi MiMo's endpoints and model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the chat-completions
//! [`XIAOMIMIMO`](crate::providers::openai::wire::XIAOMIMIMO) dialect, the
//! module's primary one; [`anthropic_from_env`] and [`anthropic_new`] build a
//! Messages-format client.
//!
//! ```no_run
//! use rig_core::providers::xiaomimimo;
//!
//! # fn run() -> Result<(), rig_core::RigError> {
//! let model = xiaomimimo::from_env()?.chat(xiaomimimo::MIMO_V2_5_PRO);
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

crate::client::macros::openai_vendor!(crate::providers::openai::wire::XIAOMIMIMO, "Xiaomi MiMo");
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::XIAOMIMIMO,
    "Xiaomi MiMo",
    anthropic_from_env,
    anthropic_new
);
