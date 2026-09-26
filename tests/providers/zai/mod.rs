mod anthropic;
mod coding;
mod general;

use rig::providers::anthropic::wire::{self as anthropic_wire, Anthropic};
use rig::providers::openai::wire::{self as openai_wire, OpenAI};

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> OpenAI {
    OpenAI::with_key(&openai_wire::ZAI, api_key())
}

pub(crate) fn coding_client() -> OpenAI {
    OpenAI::with_key(&openai_wire::ZAI_CODING, api_key())
}

pub(crate) fn anthropic_client() -> Anthropic {
    Anthropic::with_dialect(api_key(), &anthropic_wire::ZAI)
}
