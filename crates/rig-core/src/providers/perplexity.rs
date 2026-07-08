//! Perplexity API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::perplexity};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = perplexity::Client::new("YOUR_API_KEY")?;
//!
//! let sonar = client.completion_model(perplexity::SONAR);
//! # Ok(())
//! # }
//! ```
use crate::client::BearerAuth;
use crate::completion::CompletionError;
use crate::providers::openai;
use crate::{
    client::{
        self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    },
    http_client::{self, HttpClientExt},
};

// ================================================================
// Main Perplexity Client
// ================================================================
const PERPLEXITY_API_BASE_URL: &str = "https://api.perplexity.ai";

#[derive(Debug, Default, Clone, Copy)]
pub struct PerplexityExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct PerplexityBuilder;

type PerplexityApiKey = BearerAuth;

impl Provider for PerplexityExt {
    type Builder = PerplexityBuilder;

    // There is currently no way to verify a perplexity api key without consuming tokens
    const VERIFY_PATH: &'static str = "";
}

impl openai::completion::OpenAICompatibleProvider for PerplexityExt {
    const PROVIDER_NAME: &'static str = "perplexity";

    // Perplexity's structured-output support predates rig's `output_schema`
    // mapping; keep the pre-migration behavior of dropping it with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Perplexity has no tool-calling support; drop tools rather than
        // sending parameters its API rejects.
        if !request.tools.is_empty() {
            tracing::warn!("Tool use is not supported by Perplexity; tools will be ignored");
            request.tools.clear();
        }
        if request.tool_choice.take().is_some() {
            tracing::warn!("Tool choice is not supported by Perplexity and will be ignored");
        }

        Ok(())
    }
}

impl<H> Capabilities<H> for PerplexityExt {
    type Completion = Capable<CompletionModel<H>>;
    type Transcription = Nothing;
    type Embeddings = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for PerplexityExt {}

impl ProviderBuilder for PerplexityBuilder {
    type Extension<H>
        = PerplexityExt
    where
        H: HttpClientExt;
    type ApiKey = PerplexityApiKey;

    const BASE_URL: &'static str = PERPLEXITY_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(PerplexityExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<PerplexityExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<PerplexityBuilder, PerplexityApiKey, H>;

/// Perplexity completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<PerplexityExt, H>;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Perplexity client from the `PERPLEXITY_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("PERPLEXITY_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

// ================================================================
// Perplexity Completion API
// ================================================================

pub const SONAR_PRO: &str = "sonar_pro";
pub const SONAR: &str = "sonar";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };
    use crate::test_utils::MockCompletionModel;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::perplexity::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::perplexity::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn perplexity_prepare_request_drops_tools() {
        let request = crate::completion::CompletionRequestBuilder::new(
            MockCompletionModel::default(),
            "What's new today?",
        )
        .tool(crate::completion::ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
        })
        .tool_choice(crate::message::ToolChoice::Required)
        .build();

        let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: SONAR.to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: PerplexityExt::SUPPORTS_RESPONSE_FORMAT,
        })
        .expect("request should convert");
        PerplexityExt
            .prepare_request(&mut request)
            .expect("prepare_request should succeed");

        let body = serde_json::to_value(request).expect("request should serialize");
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }
}
