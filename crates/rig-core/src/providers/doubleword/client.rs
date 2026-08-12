use crate::client::{self, BearerAuth, DebugExt, Provider};

// ================================================================
// Doubleword Client
// ================================================================
// Base URL carries the `/v1`, so request paths are bare (`/chat/completions`).
const DOUBLEWORD_API_BASE_URL: &str = "https://api.doubleword.ai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct DoublewordExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct DoublewordExtBuilder;

type DoublewordApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<DoublewordExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<DoublewordExtBuilder, DoublewordApiKey, H>;

impl Provider for DoublewordExt {
    type Builder = DoublewordExtBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl DebugExt for DoublewordExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for DoublewordExt {
    const PROVIDER_NAME: &'static str = "doubleword";

    type StreamingUsage = crate::providers::openai::Usage;
    type Response = crate::providers::openai::CompletionResponse;
}

client::impl_capabilities!(
    DoublewordExt,
    completion = super::completion::CompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
);

client::impl_default_provider_builder!(
    DoublewordExtBuilder => DoublewordExt,
    api_key = DoublewordApiKey,
    base_url = DOUBLEWORD_API_BASE_URL,
);

client::impl_provider_client!(
    Client,
    input = String,
    api_key_env = "DOUBLEWORD_API_KEY",
    base_url_env_first = "DOUBLEWORD_BASE_URL",
);

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::doubleword::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::doubleword::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
