//! DeepSeek model identifiers. [`from_env`] and [`new`] build a client on the [`DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK) dialect.
//!
//! A chat model with [strict tools](crate::providers::openai::wire::Chat::with_strict_tools)
//! uses DeepSeek's strict function calling: requests go to its `/beta`
//! endpoint, with tool schemas closed and without the length and item-count
//! bounds strict mode refuses.
//!
//! ```no_run
//! use rig_core::providers::deepseek;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut deepseek_chat = deepseek::from_env()?.chat(deepseek::DEEPSEEK_V4_FLASH);
//! deepseek_chat.wire = deepseek_chat.wire.with_strict_tools();
//! # let _ = deepseek_chat;
//! # Ok(())
//! # }
//! ```

/// `deepseek-flash` completion model. DeepSeek points this unversioned name at
/// its latest Flash model, which is DeepSeek-V4.1-Flash as of September 2026.
pub const DEEPSEEK_FLASH: &str = "deepseek-flash";
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

pub mod extension;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "deepseek";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DEEPSEEK, "DeepSeek");

#[cfg(test)]
mod tests;
