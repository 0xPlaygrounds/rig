mod anthropic;
mod coding;
mod general;

use rig::prelude::*;
use rig::providers::zai;

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

/// Z.AI's general-purpose OpenAI-compatible surface for `model`.
pub(crate) fn general_config(model: &str) -> ProviderConfig {
    ProviderConfig::Zai(
        zai::functions::Config::new(model)
            .with_api_key(api_key())
            .with_base_url(zai::GENERAL_API_BASE_URL),
    )
}

/// Z.AI's coding OpenAI-compatible surface for `model`.
pub(crate) fn coding_config(model: &str) -> ProviderConfig {
    ProviderConfig::Zai(
        zai::functions::Config::new(model)
            .with_api_key(api_key())
            .with_base_url(zai::CODING_API_BASE_URL),
    )
}

/// Z.AI's Anthropic-compatible surface for `model` (reached through
/// `anthropic::functions` with a Z.AI base URL and credential).
pub(crate) fn anthropic_config(model: &str) -> ProviderConfig {
    ProviderConfig::Anthropic(
        zai::functions::anthropic_config_from_env(model)
            .expect("Z.AI Anthropic-compatible config should build"),
    )
}
