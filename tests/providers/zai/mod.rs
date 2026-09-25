mod anthropic;
mod coding;
mod general;

use rig::providers::anthropic::wire::{self as anthropic_wire, Anthropic};
use rig::providers::openai::wire::{self as openai_wire, OpenAI};
use rig_test_support::endpoint::Endpoint;

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> Endpoint<OpenAI> {
    Endpoint::new(
        OpenAI::with_key(&openai_wire::ZAI, api_key()),
        rig::rig_reqwest::shared(),
    )
}

pub(crate) fn coding_client() -> Endpoint<OpenAI> {
    Endpoint::new(
        OpenAI::with_key(&openai_wire::ZAI_CODING, api_key()),
        rig::rig_reqwest::shared(),
    )
}

pub(crate) fn anthropic_client() -> Endpoint<Anthropic> {
    Endpoint::new(
        Anthropic::with_dialect(api_key(), &anthropic_wire::ZAI),
        rig::rig_reqwest::shared(),
    )
}
