mod anthropic;
mod coding;
mod general;

use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::anthropic::wire::{self as anthropic_wire, Anthropic};
use rig::providers::openai::wire::{self as openai_wire, OpenAI};

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> Bound<OpenAI> {
    OpenAI::with_key(&openai_wire::ZAI, api_key())
        .bound()
        .expect("Z.AI general client should build")
}

pub(crate) fn coding_client() -> Bound<OpenAI> {
    OpenAI::with_key(&openai_wire::ZAI_CODING, api_key())
        .bound()
        .expect("Z.AI coding client should build")
}

pub(crate) fn anthropic_client() -> Bound<Anthropic> {
    Anthropic::with_dialect(api_key(), &anthropic_wire::ZAI)
        .bound()
        .expect("Z.AI Anthropic-compatible client should build")
}
