use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

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

impl<H> Capabilities<H> for DoublewordExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl ProviderBuilder for DoublewordExtBuilder {
    type Extension<H>
        = DoublewordExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = DoublewordApiKey;

    const BASE_URL: &'static str = DOUBLEWORD_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(DoublewordExt)
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Doubleword client from the `DOUBLEWORD_API_KEY` environment
    /// variable. The base URL can optionally be overridden with
    /// `DOUBLEWORD_BASE_URL` (defaults to `https://api.doubleword.ai/v1`).
    fn from_env() -> Result<Self, Self::Error> {
        let base_url = crate::client::optional_env_var("DOUBLEWORD_BASE_URL")?;
        let api_key = crate::client::required_env_var("DOUBLEWORD_API_KEY")?;

        let mut builder = Client::builder().api_key(&api_key);

        if let Some(base) = base_url {
            builder = builder.base_url(&base);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

pub mod doubleword_api_types {
    use serde::Deserialize;

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            self.error.message.clone()
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: ApiError,
    }

    #[derive(Debug, Deserialize)]
    pub struct ApiError {
        pub message: String,
        #[serde(default)]
        pub r#type: Option<String>,
        #[serde(default)]
        pub code: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse),
    }
}

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
