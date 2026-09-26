use rig_test_support::support;

use rig::Model;
use rig::agent::AgentBuilder;
use rig::bedrock::{
    client::BedrockRuntime,
    completion::{self, Converse},
    image::{self as bedrock_image, Images},
};
use rig::extractor::ExtractorBuilder;

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

/// The live Bedrock runtime, building each model over it.
pub(crate) struct Bedrock(BedrockRuntime);

impl Bedrock {
    pub(crate) fn completion(&self, model: impl Into<String>) -> Model<Converse, BedrockRuntime> {
        Model::new(Converse::new(model), self.0.clone())
    }

    pub(crate) fn image_generation(&self, model: &str) -> Model<Images, BedrockRuntime> {
        Model::new(Images::new(model), self.0.clone())
    }

    pub(crate) fn agent(&self, model: &str) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    pub(crate) fn extractor<T>(&self, model: &str) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

pub(crate) fn client() -> Bedrock {
    Bedrock(BedrockRuntime::from_env())
}

mod adaptive_thinking;
mod documents;
mod extractor;
mod image_generation;
mod image_prompt;
mod streaming;
