//! Groq API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::groq};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = groq::Client::new("YOUR_API_KEY")?;
//!
//! let llama = client.completion_model(groq::LLAMA_3_1_8B_INSTANT);
//! # Ok(())
//! # }
//! ```
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::openai::{self, TranscriptionResponse};
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::CompletionError;
use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm};
use crate::transcription::{self, TranscriptionError};

// ================================================================
// Main Groq Client
// ================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct GroqExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct GroqBuilder;

type GroqApiKey = BearerAuth;

impl Provider for GroqExt {
    type Builder = GroqBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl openai::completion::OpenAICompatibleProvider for GroqExt {
    const PROVIDER_NAME: &'static str = "groq";

    type StreamingUsage = openai::Usage;

    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    type Response = openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Groq's provider-native tools (`browser_search`, `code_interpreter`,
        // ...) arrive via `additional_params.tools`. Left in place they would
        // clobber the function-tool array on serialization, so fold them into
        // `compound_custom.enabled_tools` (deduplicated by tool type).
        let Some(map) = request
            .additional_params
            .as_mut()
            .and_then(Value::as_object_mut)
        else {
            return Ok(());
        };
        let Some(raw_tools) = map.remove("tools") else {
            return Ok(());
        };
        let native_tools = serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid Groq `additional_params.tools` payload: {err}").into(),
            )
        })?;
        apply_native_tools_to_additional_params(map, native_tools);

        Ok(())
    }
}

impl<H> Capabilities<H> for GroqExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Capable<TranscriptionModel<H>>;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for GroqExt {}

impl ProviderBuilder for GroqBuilder {
    type Extension<H>
        = GroqExt
    where
        H: HttpClientExt;
    type ApiKey = GroqApiKey;

    const BASE_URL: &'static str = GROQ_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(GroqExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<GroqExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<GroqBuilder, GroqApiKey, H>;

/// Groq completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<GroqExt, H>;

/// Final streaming response, shared with the OpenAI Chat Completions path.
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Groq client from the `GROQ_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("GROQ_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

fn apply_native_tools_to_additional_params(
    extra: &mut Map<String, Value>,
    native_tools: Vec<Value>,
) {
    if native_tools.is_empty() {
        return;
    }

    let mut compound_custom = match extra.remove("compound_custom") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };

    let mut enabled_tools = match compound_custom.remove("enabled_tools") {
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };

    for native_tool in native_tools {
        let already_enabled = enabled_tools
            .iter()
            .any(|existing| native_tools_match(existing, &native_tool));
        if !already_enabled {
            enabled_tools.push(native_tool);
        }
    }

    compound_custom.insert("enabled_tools".to_string(), Value::Array(enabled_tools));
    extra.insert(
        "compound_custom".to_string(),
        Value::Object(compound_custom),
    );
}

fn native_tools_match(lhs: &Value, rhs: &Value) -> bool {
    if let (Some(lhs_type), Some(rhs_type)) = (native_tool_kind(lhs), native_tool_kind(rhs)) {
        return lhs_type == rhs_type;
    }

    lhs == rhs
}

fn native_tool_kind(value: &Value) -> Option<&str> {
    match value {
        Value::String(kind) => Some(kind),
        Value::Object(map) => map.get("type").and_then(Value::as_str),
        _ => None,
    }
}

// ================================================================
// Groq Completion API
// ================================================================

/// The `deepseek-r1-distill-llama-70b` model. Used for chat completion.
pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
/// The `gemma2-9b-it` model. Used for chat completion.
pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
/// The `llama-3.1-8b-instant` model. Used for chat completion.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.2-11b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_11B_VISION_PREVIEW: &str = "llama-3.2-11b-vision-preview";
/// The `llama-3.2-1b-preview` model. Used for chat completion.
pub const LLAMA_3_2_1B_PREVIEW: &str = "llama-3.2-1b-preview";
/// The `llama-3.2-3b-preview` model. Used for chat completion.
pub const LLAMA_3_2_3B_PREVIEW: &str = "llama-3.2-3b-preview";
/// The `llama-3.2-90b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_90B_VISION_PREVIEW: &str = "llama-3.2-90b-vision-preview";
/// The `llama-3.2-70b-specdec` model. Used for chat completion.
pub const LLAMA_3_2_70B_SPECDEC: &str = "llama-3.2-70b-specdec";
/// The `llama-3.2-70b-versatile` model. Used for chat completion.
pub const LLAMA_3_2_70B_VERSATILE: &str = "llama-3.2-70b-versatile";
/// The `llama-guard-3-8b` model. Used for chat completion.
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
/// The `llama3-70b-8192` model. Used for chat completion.
pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
/// The `llama3-8b-8192` model. Used for chat completion.
pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
/// The `mixtral-8x7b-32768` model. Used for chat completion.
pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    Parsed,
    Raw,
    Hidden,
}

/// Additional parameters to send to the Groq API. Serialize this into the
/// request's `additional_params` to set Groq's reasoning options.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroqAdditionalParameters {
    /// The reasoning format. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// Whether or not to include reasoning. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    /// Any other properties not included by default on this struct (that you want to send)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, serde_json::Value>>,
}

// ================================================================
// Groq Transcription API
// ================================================================

pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3_EN: &str = "distil-whisper-large-v3-en";

#[derive(Clone)]
pub struct TranscriptionModel<T> {
    client: Client<T>,
    /// Name of the model (e.g.: whisper-large-v3)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}
impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = TranscriptionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = MultipartForm::new()
            .text("model", self.model.clone())
            .part(Part::bytes("file", data).filename(request.filename.clone()));

        if let Some(language) = request.language {
            body = body.text("language", language);
        }

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;

            for (key, value) in params {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let req = self
            .client
            .post("/audio/transcriptions")?
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send_multipart::<Bytes>(req).await?;

        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();

        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => {
                    tracing::warn!(message = %api_error_response.message, "provider returned an error response");
                    Err(TranscriptionError::from_http_response(
                        status,
                        String::from_utf8_lossy(&response_body).into_owned(),
                    ))
                }
            }
        } else {
            Err(TranscriptionError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };
    use crate::{completion::CompletionRequestBuilder, test_utils::MockCompletionModel};

    #[test]
    fn groq_request_maps_output_schema_max_tokens_and_specific_tool_choice() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Return JSON")
            .max_tokens(64)
            .tool(crate::completion::ToolDefinition {
                output_schema: None,
                metadata: Default::default(),
                name: "choose_beta".to_string(),
                description: "Choose beta".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            })
            .tool_choice(crate::message::ToolChoice::Specific {
                function_names: vec!["choose_beta".to_string()],
            })
            .output_schema(schemars::schema_for!(serde_json::Value))
            .build();

        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("Groq request should convert");
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["max_tokens"], 64);
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type":"function","function":{"name":"choose_beta"}})
        );
        // The shared path defers `response_format` while tools are present and
        // no tool result exists yet (see `should_apply_response_format`).
        assert_eq!(json["response_format"], serde_json::Value::Null);

        let no_tools_request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "Return JSON")
                .output_schema(schemars::schema_for!(serde_json::Value))
                .build();
        let no_tools_request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request: no_tools_request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request should convert");
        let json = serde_json::to_value(no_tools_request).expect("request should serialize");
        assert_eq!(json["response_format"]["type"], "json_schema");
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn groq_prepare_request_merges_native_tools_into_compound_custom() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "search")
            .tool(crate::completion::ToolDefinition {
                output_schema: None,
                metadata: Default::default(),
                name: "local_tool".to_string(),
                description: "A local function tool".to_string(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
            })
            .additional_params(serde_json::json!({
                "tools": [{"type": "browser_search"}, {"type": "browser_search"}],
            }))
            .build();

        let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request should convert");

        super::GroqExt
            .prepare_request(&mut request)
            .expect("prepare_request should succeed");

        let json = serde_json::to_value(request).expect("request should serialize");
        assert_eq!(
            json["compound_custom"]["enabled_tools"],
            serde_json::json!([{"type": "browser_search"}])
        );
        // The rig-level function tool array must survive the native-tool merge.
        assert_eq!(json["tools"][0]["function"]["name"], "local_tool");
    }

    #[test]
    fn groq_reasoning_params_flatten_into_request_body() {
        let additional_params = serde_json::to_value(super::GroqAdditionalParameters {
            reasoning_format: Some(super::ReasoningFormat::Parsed),
            include_reasoning: Some(true),
            extra: None,
        })
        .expect("params should serialize");
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "Think about it")
                .additional_params(additional_params)
                .build();

        let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
            model: "llama-3.3-70b-versatile".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            supports_tools: true,
        })
        .expect("request should convert");
        let json = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(json["reasoning_format"], "parsed");
        assert_eq!(json["include_reasoning"], true);
    }

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::groq::Client::new("dummy-key").expect("Client::new() failed");
        let builder: crate::providers::groq::ClientBuilder =
            crate::providers::groq::Client::builder().api_key("dummy-key");
        let _client_from_builder = builder.build().expect("Client::builder() failed");
    }

    #[tokio::test]
    async fn completion_preserves_raw_provider_error_json_on_api_error_envelope() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"model overloaded","type":"server_error","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("llama-3.3-70b-versatile");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
                assert_eq!(error.provider_response_body(), Some(body));
                let json = error
                    .provider_response_json()
                    .expect("raw body should be valid JSON")
                    .expect("parsed JSON should be present");
                assert_eq!(json["code"], "503");
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_http_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"service unavailable","code":"503"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model("llama-3.3-70b-versatile");
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("completion should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_http_non_success_preserves_status_and_body() {
        use crate::client::transcription::TranscriptionClient;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::{TranscriptionError, TranscriptionModel as _};

        let body = r#"{"error":{"message":"bad audio","code":"400"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model("whisper-large-v3");

        let error = match model
            .transcription_request()
            .data(vec![0u8; 16])
            .send()
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("transcription should fail with non-success status"),
        };

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
