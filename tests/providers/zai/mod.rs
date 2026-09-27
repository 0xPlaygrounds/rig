mod anthropic;
mod coding;
mod general;

use rig::providers::anthropic::Anthropic;
use rig::providers::openai::wire::{self as openai_wire};
use rig::providers::openai::{OpenAI, OpenAIConfig};
use rig::providers::zai;

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> OpenAI {
    zai::new(api_key())
}

pub(crate) fn coding_client() -> OpenAI {
    OpenAIConfig::with_key(&openai_wire::ZAI_CODING, api_key()).client()
}

pub(crate) fn anthropic_client() -> Anthropic {
    zai::anthropic_new(api_key())
}
