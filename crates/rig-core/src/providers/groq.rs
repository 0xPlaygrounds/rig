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
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::openai;
use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;

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

    /// Groq reports its transport request id on the same `x-request-id`
    /// header OpenAI uses (verified live; see the recorded
    /// `response_identity_edge` fixture, where the header arrives scrubbed).
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

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

client::impl_capabilities!(
    GroqExt,
    completion = CompletionModel<H>,
    transcription = TranscriptionModel<H>,
    model_listing = GroqModelLister<H>,
);

/// A Groq listing entry.
///
/// Groq reports its context window and output ceiling on every entry, and
/// [`Model`](crate::model::Model) has fields for both — `max_output_tokens`
/// exists precisely because rig used to drop a provider-reported output
/// ceiling on the floor (rig#2322). The shared `ListModelEntry` decodes
/// neither (Groq spells them `context_window` / `max_completion_tokens`, and
/// the spellings differ across providers), so Groq keeps its own DTO rather
/// than losing them.
#[derive(Debug, serde::Deserialize)]
struct GroqModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

impl From<GroqModelEntry> for crate::model::Model {
    fn from(value: GroqModelEntry) -> Self {
        let mut model = crate::model::Model::from_id(value.id);
        model.name = value.name;
        model.created_at = value.created;
        model.owned_by = value.owned_by;
        model.context_length = value.context_window;
        model.max_output_tokens = value.max_completion_tokens;
        model
    }
}

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the Groq
    /// API (`GET /models`), the same path [`GroqExt::VERIFY_PATH`] already
    /// uses.
    GroqModelLister,
    Client<H>,
    GroqModelEntry,
    "Groq",
    "/models"
);

impl DebugExt for GroqExt {}

client::impl_default_provider_builder!(
    GroqBuilder => GroqExt,
    api_key = GroqApiKey,
    base_url = GROQ_API_BASE_URL,
);

pub type Client<H = reqwest::Client> = client::Client<GroqExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<GroqBuilder, GroqApiKey, H>;

/// Groq completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<GroqExt, H>;

/// Groq's provider-native terminal streaming record: the value carried by the
/// final item of the stream returned by `CompletionModel::raw_stream`. Shared
/// with the OpenAI Chat Completions path, usage payload included.
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse;

client::impl_provider_client!(Client, input = String, api_key_env = "GROQ_API_KEY");

#[cfg(test)]
use crate::providers::openai::client::ApiResponse;

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

/// Groq transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };
    use crate::{completion::CompletionRequestBuilder, test_utils::MockCompletionModel};

    /// An OpenAI-style nested error body (`{"error": {"message": ...}}`) on a
    /// 2xx status must classify as the error envelope — not fail both untagged
    /// arms and surface as a serde error that loses the provider body.
    #[test]
    fn nested_error_object_parses_as_the_error_envelope() {
        #[derive(serde::Deserialize)]
        struct Success {
            #[allow(dead_code)]
            choices: Vec<serde_json::Value>,
        }

        let nested = r#"{"error":{"message":"model not found","type":"invalid_request_error"}}"#;
        match serde_json::from_str::<super::ApiResponse<Success>>(nested)
            .expect("nested error envelope should deserialize")
        {
            super::ApiResponse::Err(err) => assert!(err.message.contains("model not found")),
            super::ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
        }

        let plain = r#"{"error":"over capacity"}"#;
        match serde_json::from_str::<super::ApiResponse<Success>>(plain)
            .expect("string error envelope should deserialize")
        {
            super::ApiResponse::Err(err) => assert_eq!(err.message, "over capacity"),
            super::ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
        }
    }

    #[test]
    fn groq_request_maps_output_schema_max_tokens_and_specific_tool_choice() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Return JSON")
            .max_tokens(64)
            .tool(crate::completion::ToolDefinition {
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

    #[tokio::test]
    async fn transcription_routes_model_in_multipart_body() {
        use crate::client::transcription::TranscriptionClient;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::TranscriptionModel as _;

        let http_client = RecordingHttpClient::new(r#"{"text":"transcribed"}"#);
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.transcription_model(super::WHISPER_LARGE_V3);

        let response = model
            .transcription_request()
            .data(vec![1, 2, 3])
            .filename(Some("audio.mp3".to_owned()))
            .send()
            .await
            .expect("transcription should succeed");

        assert_eq!(response.text, "transcribed");
        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert_eq!(
            request.uri,
            "https://api.groq.com/openai/v1/audio/transcriptions"
        );
        let body = String::from_utf8_lossy(&request.body);
        assert!(
            body.contains("name=\"model\"\r\n\r\nwhisper-large-v3\r\n"),
            "{body}"
        );
        assert!(
            body.contains("name=\"file\"; filename=\"audio.mp3\""),
            "{body}"
        );
    }

    #[test]
    fn groq_prepare_request_merges_native_tools_into_compound_custom() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "search")
            .tool(crate::completion::ToolDefinition {
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

        // rig#2314: a provider with a request-id contract preserves its
        // non-success responses as ProviderResponse, so the transport id has
        // a home on the error; this mock sent no header, so the id is None.
        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(error.provider_request_id(), None);
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
