use super::client::OpenRouterExt;
use crate::providers::openai::embedding::{GenericEmbeddingModel, OpenAIEmbeddingsCompatible};

impl OpenAIEmbeddingsCompatible for OpenRouterExt {
    const PROVIDER_NAME: &'static str = "openrouter";
    const REQUIRES_USAGE: bool = false;
}

pub type EmbeddingModel<H> = GenericEmbeddingModel<OpenRouterExt, H>;

#[cfg(test)]
mod tests;
