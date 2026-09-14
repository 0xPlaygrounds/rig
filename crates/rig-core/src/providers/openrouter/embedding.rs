use super::client::OpenRouter;
use crate::providers::openai::embedding::{GenericEmbeddingModel, OpenAIEmbeddingsCompatible};

impl OpenAIEmbeddingsCompatible for OpenRouter {
    const PROVIDER_NAME: &'static str = "openrouter";
    const REQUIRES_USAGE: bool = false;
}

pub type EmbeddingModel<H = crate::http_client::BoxedHttpClient> =
    GenericEmbeddingModel<OpenRouter, H>;

#[cfg(test)]
mod tests;
