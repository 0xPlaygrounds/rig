mod support;

mod anthropic;
mod coding;
mod general;

mod cassette {
    mod agent;
    mod error_envelope;
    mod models;
    mod reasoning;
    mod streaming;
    mod structured_output;
    mod tools;
}

use rig::providers::zai;

/// The cheapest model the general endpoint serves: Z.AI's pricing table lists
/// `glm-4.5-flash` as free for input, cached input and output alike. It has no
/// constant in `rig::providers::zai` (the module exports the 4.5/4.6 paid
/// tiers), so cells that do not need a specific capability name it literally.
pub(crate) const CHEAP_GENERAL_MODEL: &str = "glm-4.5-flash";

/// The coding endpoint serves a smaller model set than the general one, and
/// `glm-4.5-air` is the cheapest member documented for it.
pub(crate) const CODING_MODEL: &str = zai::GLM_4_5_AIR;

/// The Anthropic-compatible endpoint is documented as the Coding Plan URL, so
/// it is named separately from [`CODING_MODEL`] even though both resolve to
/// `glm-4.5-air` today: they are two different endpoints' model sets, and one
/// can move without the other.
pub(crate) const ANTHROPIC_MODEL: &str = zai::GLM_4_5_AIR;

/// GLM's thinking output is what the reasoning cells are about, so they name a
/// model documented to produce it rather than the free flash tier.
pub(crate) const THINKING_MODEL: &str = zai::GLM_4_5_AIR;

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> zai::Client {
    zai::Client::builder()
        .api_key(api_key())
        .general()
        .build()
        .expect("Z.AI general client should build")
}

pub(crate) fn coding_client() -> zai::Client {
    zai::Client::builder()
        .api_key(api_key())
        .coding()
        .build()
        .expect("Z.AI coding client should build")
}

pub(crate) fn anthropic_client() -> zai::AnthropicClient {
    zai::AnthropicClient::builder()
        .api_key(api_key())
        .general()
        .build()
        .expect("Z.AI Anthropic-compatible client should build")
}
