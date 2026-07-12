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

    type StreamingUsage = openai::Usage;

    // Perplexity has no tool-calling support; `tools`/`tool_choice` are
    // dropped with a warning during request conversion.
    const SUPPORTS_TOOLS: bool = false;

    // Perplexity's structured-output support predates rig's `output_schema`
    // mapping; keep the pre-migration behavior of dropping it with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    // The pre-migration streaming request sent `stream: true` with no
    // `stream_options`.
    const STREAM_INCLUDE_USAGE: bool = false;

    type Response = openai::CompletionResponse;

    fn finalize_request_body(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
        // Perplexity historically only accepted plain `{role, content: String}`
        // messages, and its API accepts only system/user/assistant roles
        // with strict user/assistant alternation. Strip tool-exchange
        // remnants from shared histories and flatten text-only content-part
        // arrays; arrays with non-text parts (e.g. images on sonar models)
        // are left for the API's multimodal handling.
        if let Some(messages) = body
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            openai::completion::sanitize_plain_text_history(
                messages,
                Some(("\n", true)),
                false,
                true,
            );
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

/// Raw completion payload, shared with the OpenAI Chat Completions path.
pub type CompletionResponse = openai::CompletionResponse;

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
    fn perplexity_finalize_flattens_text_only_content_arrays() {
        let mut body = serde_json::json!({
            "model": SONAR,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "First."},
                    {"type": "text", "text": "Second."}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Look:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/i.png"}}
                ]}
            ]
        });

        PerplexityExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        assert_eq!(body["messages"][0]["content"], "Be brief.");
        assert_eq!(body["messages"][1]["content"], "First.\nSecond.");
        // Mixed content stays an array for the API's multimodal handling.
        assert!(body["messages"][2]["content"].is_array());
    }

    #[test]
    fn perplexity_drops_tool_choice_instead_of_erroring() {
        // Multi-name Specific errors on tool-supporting providers; with
        // SUPPORTS_TOOLS = false it must be dropped before that validation.
        let mut request = crate::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello!".into()),
            documents: vec![],
            max_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: Some(crate::message::ToolChoice::Specific {
                function_names: vec!["a".to_string(), "b".to_string()],
            }),
            additional_params: None,
            output_schema: None,
        };
        request.tools = vec![crate::completion::ToolDefinition {
            output_schema: None,
            metadata: Default::default(),
            name: "lookup".to_string(),
            description: String::new(),
            parameters: serde_json::json!({}),
        }];

        let converted = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: SONAR.to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: PerplexityExt::SUPPORTS_RESPONSE_FORMAT,
            supports_tools: PerplexityExt::SUPPORTS_TOOLS,
        })
        .expect("unsupported tools should be dropped, not an error");

        let json = serde_json::to_value(converted).expect("request should serialize");
        assert!(
            json.get("tools")
                .is_none_or(|tools| tools.as_array().is_none_or(|tools| tools.is_empty()))
        );
        assert!(json.get("tool_choice").is_none());
    }

    #[test]
    fn perplexity_finalize_strips_tool_history_and_preserves_alternation() {
        let mut body = serde_json::json!({
            "model": SONAR,
            "messages": [
                {"role": "user", "content": "Look it up."},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                {"role": "assistant", "content": "It is crimson.", "reasoning_content": "hmm"},
                {"role": "user", "content": "Thanks!"}
            ]
        });

        PerplexityExt
            .finalize_request_body(&mut body)
            .expect("finalize should succeed");

        let messages = body["messages"].as_array().expect("messages array");
        let roles = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(roles, ["user", "assistant", "user"]);
        assert_eq!(messages[1]["content"], "It is crimson.");
        assert!(messages[1].get("reasoning_content").is_none());
        assert!(messages[1].get("tool_calls").is_none());
    }

    #[test]
    fn perplexity_prepare_request_drops_tools() {
        let request = crate::completion::CompletionRequestBuilder::new(
            MockCompletionModel::default(),
            "What's new today?",
        )
        .tool(crate::completion::ToolDefinition {
            output_schema: None,
            metadata: Default::default(),
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
            supports_tools: false,
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
