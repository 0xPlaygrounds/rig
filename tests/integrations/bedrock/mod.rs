use rig_test_support::support;

use rig::bedrock::{client::Client, completion, image as bedrock_image};

pub(crate) const BEDROCK_COMPLETION_MODEL: &str = completion::AMAZON_NOVA_LITE;
pub(crate) const BEDROCK_IMAGE_MODEL: &str = bedrock_image::AMAZON_NOVA_CANVAS;

pub(crate) fn anthropic_adaptive_model() -> String {
    std::env::var("BEDROCK_ANTHROPIC_ADAPTIVE_MODEL")
        .unwrap_or_else(|_| "us.anthropic.claude-sonnet-4-6".to_string())
}

pub(crate) fn anthropic_signature_only_model() -> String {
    std::env::var("BEDROCK_ANTHROPIC_SIGNATURE_ONLY_MODEL")
        .unwrap_or_else(|_| "global.anthropic.claude-opus-4-7".to_string())
}

pub(crate) fn client() -> Client {
    Client::from_env().expect("client should build")
}

mod adaptive_thinking;
mod documents;
mod extractor;
mod image_generation;
mod image_prompt;
mod streaming;
