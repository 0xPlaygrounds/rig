#[path = "../../common/support.rs"]
mod support;

use rig::bedrock::functions::Config;
use rig::bedrock::{completion, embedding, image as bedrock_image};

pub(crate) const BEDROCK_COMPLETION_MODEL: &str = completion::AMAZON_NOVA_LITE;
pub(crate) const BEDROCK_EMBEDDING_MODEL: &str = embedding::AMAZON_TITAN_EMBED_TEXT_V2_0;
pub(crate) const BEDROCK_IMAGE_MODEL: &str = bedrock_image::AMAZON_NOVA_CANVAS;

pub(crate) fn anthropic_adaptive_model() -> String {
    std::env::var("BEDROCK_ANTHROPIC_ADAPTIVE_MODEL")
        .unwrap_or_else(|_| "us.anthropic.claude-sonnet-4-6".to_string())
}

pub(crate) fn anthropic_signature_only_model() -> String {
    std::env::var("BEDROCK_ANTHROPIC_SIGNATURE_ONLY_MODEL")
        .unwrap_or_else(|_| "global.anthropic.claude-opus-4-7".to_string())
}

/// The AWS Bedrock SDK client for the default credential chain — the
/// `Client::from_env` replacement for the free-function provider surface
/// (`rig::bedrock::functions::{complete, open_stream, embed}` all take it).
pub(crate) async fn aws_client() -> aws_sdk_bedrockruntime::Client {
    rig::bedrock::functions::client_from_config(&Config::new(BEDROCK_COMPLETION_MODEL)).await
}

/// A `ProviderConfig` for `model` using the SDK's default credential chain
/// and region resolution — the `Client::from_env` equivalent for the
/// non-generic agent path.
pub(crate) fn bedrock_config(model: &str) -> rig::provider::ProviderConfig {
    rig::provider::ProviderConfig::Bedrock(rig::bedrock::functions::Config::new(model))
}

/// An `AgentBuilder` for `model` over the default-credential Bedrock config.
pub(crate) fn agent(model: &str) -> rig::agent::AgentBuilder {
    rig::agent::AgentBuilder::new(bedrock_config(model))
}

mod adaptive_thinking;
mod agent;
mod documents;
mod embeddings;
mod extractor;
mod image_generation;
mod image_prompt;
mod streaming;
