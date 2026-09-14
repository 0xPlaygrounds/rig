use crate::client::{
    self, BearerAuth, HasCompletion, HasEmbeddings, ModelTransport, Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};

// ================================================================
// Doubleword Client
// ================================================================
// Base URL carries the `/v1`, so request paths are bare (`/chat/completions`).
const DOUBLEWORD_API_BASE_URL: &str = "https://api.doubleword.ai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct Doubleword;
type DoublewordApiKey = BearerAuth;

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Doubleword, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Doubleword, H>;

impl Provider for Doubleword {
    const NAME: &'static str = "doubleword";
    const BASE_URL: &'static str = DOUBLEWORD_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = DoublewordApiKey;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &DoublewordApiKey) -> http_client::Result<Self> {
        Ok(Doubleword)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("DOUBLEWORD_API_KEY", Some("DOUBLEWORD_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Doubleword {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Doubleword {
    type Model<H>
        = super::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::EmbeddingModel::make(client, model, ndims)
    }
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for Doubleword {
    const PROVIDER_NAME: &'static str = "doubleword";

    type StreamingUsage = crate::providers::openai::Usage;
    type Response = crate::providers::openai::CompletionResponse;
}

#[cfg(test)]
mod tests;
