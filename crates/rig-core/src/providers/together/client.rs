use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

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

impl<H> Capabilities<H> for TogetherExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl ProviderBuilder for TogetherExtBuilder {
    type Extension<H>
        = TogetherExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = TogetherApiKey;

    const BASE_URL: &'static str = TOGETHER_AI_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(TogetherExt)
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Together AI client from the `TOGETHER_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("TOGETHER_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

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
