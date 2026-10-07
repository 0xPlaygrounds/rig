//! Z.AI's endpoints and model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the chat-completions
//! [`ZAI`](crate::providers::openai::wire::ZAI) dialect, the module's primary
//! one; [`anthropic_from_env`] and [`anthropic_new`] build a Messages-format
//! client. The coding endpoint is
//! [`ZAI_CODING`](crate::providers::openai::wire::ZAI_CODING). Both read
//! `ZAI_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::zai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let glm_4_6 = zai::from_env()?.chat(zai::GLM_4_6);
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

/// Whether GLM `model` reads images: the vision models put `v` right after
/// the version (`glm-4.5v`, `glm-4.6v-flash`, `glm-5v-turbo`). Every wire
/// Z.AI serves applies this rule to an id the catalog does not list.
pub(crate) fn reads_images(model: &str) -> bool {
    model.strip_prefix("glm-").is_some_and(|rest| {
        rest.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.')
            .starts_with('v')
    })
}

pub mod extension;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "zai";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::ZAI, "Z.AI");
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::ZAI,
    "Z.AI",
    anthropic_from_env,
    anthropic_new
);
