use crate::client::{self, BearerAuth, DebugExt, Provider};

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

#[derive(Debug, Default, Clone, Copy)]
pub struct TogetherExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct TogetherExtBuilder;

type TogetherApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<TogetherExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<TogetherExtBuilder, TogetherApiKey, H>;

impl Provider for TogetherExt {
    type Builder = TogetherExtBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl DebugExt for TogetherExt {}

impl crate::providers::openai::completion::OpenAICompatibleProvider for TogetherExt {
    const PROVIDER_NAME: &'static str = "together";

    type StreamingUsage = crate::providers::openai::Usage;

    // Together's structured-output support is model-dependent; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = crate::providers::openai::CompletionResponse;

    // The client base URL is the bare host; embeddings build their own v1 path.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }
}

client::impl_capabilities!(
    TogetherExt,
    completion = super::CompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
);

client::impl_default_provider_builder!(
    TogetherExtBuilder => TogetherExt,
    api_key = TogetherApiKey,
    base_url = TOGETHER_AI_BASE_URL,
);

client::impl_provider_client!(Client, input = String, api_key_env = "TOGETHER_API_KEY");

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::together::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::together::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
