use crate::client::{self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider};

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
