//! Moonshot AI (Kimi) endpoints and model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the
//! [`MOONSHOT`](crate::providers::openai::wire::MOONSHOT) dialect, the module's
//! primary one; [`anthropic_from_env`] and [`anthropic_new`] build a
//! Messages-format client. Environment configuration reads `MOONSHOT_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::moonshot;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let kimi = moonshot::from_env()?.chat(moonshot::KIMI_K3);
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

/// Whether Kimi `model` reads images: Kimi K2 before K2.5 and the
/// `moonshot-v1` text models do not; K2.5 and later, and `*-vision-*`, do.
/// Every wire Moonshot serves applies this rule to an id the catalog does
/// not list.
pub(crate) fn reads_images(model: &str) -> bool {
    !(model == "kimi-k2"
        || model.starts_with("kimi-k2-")
        || (model.starts_with("moonshot-v1") && !model.contains("vision")))
}

pub mod extension;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "moonshot";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::MOONSHOT, "Moonshot");
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::MOONSHOT,
    "Moonshot",
    anthropic_from_env,
    anthropic_new
);

#[cfg(test)]
mod tests;
