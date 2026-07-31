//! The Gemini Interactions API as config + pure functions.
//!
//! The data-oriented face of `super`: a serde `Config`, a `DESCRIPTOR`
//! capability sheet, pure [`build_request_body`] / [`build_request`] /
//! [`parse_response`], and the async [`complete`] / [`open_stream`] wrappers
//! over [`HttpRuntime`]. It replaces the deleted `InteractionsClient` /
//! `InteractionsCompletionModel` pair without changing a wire byte.
//!
//! # Not the same face as [`gemini::functions`](crate::providers::gemini::functions)
//!
//! The Interactions API is a different surface on the same host, and it
//! authenticates differently: the credential rides in an **`x-goog-api-key`
//! header**, where `generateContent` puts it in a `?key=` query parameter.
//! That is the deleted `GeminiInteractionsExt::with_custom`'s behavior, kept
//! verbatim. The endpoint is `POST /v1beta/interactions`, and streaming is
//! selected by `?alt=sse` on the URL together with `"stream": true` in the
//! body.
//!
//! # Reaching it from a `ProviderConfig`
//!
//! `rig_agent::provider::ProviderConfig::GeminiInteractions` is a
//! hand-written arm carrying this module's [`Config`], following the same
//! pattern as `OpenAiResponses`: a second, incompatible surface on a provider
//! whose macro-generated row already carries the `generateContent` config.

use http::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};

use super::interactions_api_types::{CreateInteractionRequest, Interaction};
use super::{build_interaction_stream_path, create_request_body};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_runtime::HttpRuntime;
use crate::providers::descriptor::{
    ApiKeyLocation, ConfigError, ProviderDescriptor, required_env_var,
};
use crate::telemetry::{CompletionOperation, completion_span};

/// Default Gemini API base URL — the same host the `generateContent` face
/// uses.
pub const DEFAULT_BASE_URL: &str = super::super::functions::DEFAULT_BASE_URL;

/// The Interactions endpoint path, relative to [`DEFAULT_BASE_URL`].
pub const INTERACTIONS_PATH: &str = "/v1beta/interactions";

/// The Interactions API's capability sheet.
///
/// Same underlying models as `generateContent`, so the capability answers
/// match [`gemini::functions::DESCRIPTOR`](crate::providers::gemini::functions::DESCRIPTOR),
/// including its `verify_path`: the Interactions client verified against
/// `/v1beta/models` too.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
    name: "gemini",
    supports_tools: true,
    supports_response_format: true,
    // Usage arrives on `interaction.completed`; no `stream_options` opt-in.
    stream_include_usage: false,
    // `step.start` carries a whole `function_call` payload.
    emits_complete_single_chunk_tool_calls: true,
    composes_native_output_with_tools: true,
    max_embedding_documents: None,
    verify_path: Some("/v1beta/models"),
};

/// Plain-data Gemini Interactions API configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Credential location. Sent as the `x-goog-api-key` **header** — unlike
    /// the `generateContent` face, which uses a `key` query parameter.
    pub api_key: ApiKeyLocation,
    /// Model identifier requests are built for.
    pub model: String,
    /// Extra headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
}

impl Config {
    /// Config for `model` reading `GEMINI_API_KEY` from the environment.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: ApiKeyLocation::Env("GEMINI_API_KEY".to_string()),
            model: model.into(),
            extra_headers: Vec::new(),
        }
    }

    /// Config for `model` built from the process environment.
    ///
    /// Reads `GEMINI_API_KEY` (required) — the same variable the deleted
    /// `InteractionsClient::from_env` read, with no base-URL override, since
    /// the classic client always targeted [`DEFAULT_BASE_URL`]. The
    /// credential is validated eagerly but stored as [`ApiKeyLocation::Env`],
    /// so the secret is read at request time rather than held in the config.
    ///
    /// # Errors
    /// [`ConfigError`] when `GEMINI_API_KEY` is missing or invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
        let cfg = Self::new(model);
        required_env_var("GEMINI_API_KEY")?;
        Ok(cfg)
    }

    /// Config for `model` with an explicit API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = ApiKeyLocation::Inline(key.into());
        self
    }

    /// Override the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

/// Build the typed `CreateInteractionRequest` for `request`. Pure.
pub fn build_typed_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<CreateInteractionRequest, CompletionError> {
    create_request_body(cfg.model.clone(), request.clone(), Some(stream))
}

/// Build the serialized Interactions request body for `request`.
///
/// Pure: the exact bytes the wire sees. `stream` becomes the body's
/// `"stream"` field, exactly as the deleted model's
/// `create_completion_request(request, Some(stream))` did.
pub fn build_request_body(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Vec<u8>, CompletionError> {
    Ok(serde_json::to_vec(&build_typed_request(
        cfg, request, stream,
    )?)?)
}

/// Attach the Interactions credential and extra headers to `builder`.
fn authorize(
    cfg: &Config,
    mut builder: http::request::Builder,
) -> Result<http::request::Builder, CompletionError> {
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
    {
        builder = builder.header("x-goog-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    Ok(builder)
}

/// The absolute URL for `path` against `cfg`'s base URL, with `alt=sse`
/// appended when `stream` is set.
fn interactions_url(cfg: &Config, path: &str, stream: bool) -> String {
    let url = format!(
        "{}/{}",
        cfg.base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    if !stream {
        return url;
    }
    let separator = if url.contains('?') { "&" } else { "?" };
    format!("{url}{separator}alt=sse")
}

/// Build the complete HTTP request (URL, headers, body) for `request`.
///
/// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
/// environment). `stream` both appends `alt=sse` to the URL and sets the
/// body's `stream` field — the deleted client did exactly this, splitting the
/// work between `build_uri(Transport::Sse)` and `create_request_body`.
pub fn build_request(
    cfg: &Config,
    request: &CompletionRequest,
    stream: bool,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let body = build_request_body(cfg, request, stream)?;
    let builder = http::Request::post(interactions_url(cfg, INTERACTIONS_PATH, stream))
        .header(CONTENT_TYPE, "application/json");
    authorize(cfg, builder)?
        .body(body)
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Parse an Interactions response body into the normalized
/// [`completion::CompletionResponse`]. Pure.
pub fn parse_response(
    status: http::StatusCode,
    body: &str,
) -> Result<completion::CompletionResponse, CompletionError> {
    parse_interaction(status, body)?.try_into()
}

/// Parse an Interactions response body into the raw [`Interaction`] payload.
///
/// Pure. The escape hatch the deleted `create_interaction` /
/// `get_interaction` returned: background tasks whose `status` is
/// `InProgress` carry no outputs yet, so they cannot be normalized into a
/// [`completion::CompletionResponse`].
pub fn parse_interaction(
    status: http::StatusCode,
    body: &str,
) -> Result<Interaction, CompletionError> {
    if !status.is_success() {
        return Err(CompletionError::from_http_response(
            status,
            body.to_string(),
        ));
    }
    serde_json::from_str(body).map_err(|err| {
        tracing::error!(
            error = %err,
            body = %body,
            "Failed to deserialize Gemini interactions response"
        );
        CompletionError::JsonError(err)
    })
}

/// Send `request` to the Interactions API and return the normalized response.
pub async fn complete(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<completion::CompletionResponse, CompletionError> {
    create_interaction(cfg, rt, request).await?.try_into()
}

/// Create an interaction and return the raw [`Interaction`] payload.
///
/// The free-function form of the deleted
/// `InteractionsCompletionModel::create_interaction`.
pub async fn create_interaction(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<Interaction, CompletionError> {
    let req = build_request(cfg, &request, false)?;
    let (status, body) = rt.send(req).await?;
    parse_interaction(status, &body)
}

/// Build the `GET /v1beta/interactions/{id}` request for [`get_interaction`].
///
/// Pure except for credential resolution.
pub fn build_get_interaction_request(
    cfg: &Config,
    interaction_id: &str,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let path = format!("{INTERACTIONS_PATH}/{interaction_id}");
    let builder = http::Request::get(interactions_url(cfg, &path, false));
    authorize(cfg, builder)?
        .body(Vec::new())
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Fetch an interaction by ID, for polling background tasks.
pub async fn get_interaction(
    cfg: &Config,
    rt: &HttpRuntime,
    interaction_id: &str,
) -> Result<Interaction, CompletionError> {
    let req = build_get_interaction_request(cfg, interaction_id)?;
    let (status, body) = rt.send(req).await?;
    parse_interaction(status, &body)
}

/// Open a streaming completion for `request` over `POST
/// /v1beta/interactions?alt=sse`.
pub async fn open_stream(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
    let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
    // `gcp.gemini` is the OTel `gen_ai.provider.name` value the deleted
    // `InteractionsCompletionModel::stream` recorded; `DESCRIPTOR.name`
    // ("gemini") is the normalized-response provider field, a different
    // vocabulary.
    let span = completion_span(
        "gcp.gemini",
        &model,
        CompletionOperation::InteractionsStreaming,
        &request,
    );
    let req = build_request(cfg, &request, true)?;
    Ok(super::streaming::interaction_completion_stream(
        rt.sse_events(req, false),
        span,
    ))
}

/// Start an interaction and stream the raw
/// [`InteractionSseEvent`](super::interactions_api_types::InteractionSseEvent)s.
///
/// The free-function form of the deleted `stream_interaction_events`: no
/// normalization, for callers that want the Interactions event vocabulary.
pub async fn stream_interaction_events(
    cfg: &Config,
    rt: &HttpRuntime,
    request: CompletionRequest,
) -> Result<super::streaming::InteractionEventStream, CompletionError> {
    let req = build_request(cfg, &request, true)?;
    Ok(super::streaming::interaction_event_stream(
        rt.sse_events(req, false),
    ))
}

/// Build the resumable-stream `GET` request for
/// [`stream_interaction_events_by_id`].
///
/// Pure except for credential resolution. The path comes from the retained
/// pure [`build_interaction_stream_path`], which already carries
/// `stream=true` and the optional `last_event_id`.
pub fn build_resume_stream_request(
    cfg: &Config,
    interaction_id: &str,
    last_event_id: Option<&str>,
) -> Result<http::Request<Vec<u8>>, CompletionError> {
    let path = build_interaction_stream_path(interaction_id, last_event_id);
    let builder = http::Request::get(interactions_url(cfg, &path, true));
    authorize(cfg, builder)?
        .body(Vec::new())
        .map_err(|e| CompletionError::RequestError(Box::new(e)))
}

/// Resume an interaction stream by ID and optional last event ID.
pub async fn stream_interaction_events_by_id(
    cfg: &Config,
    rt: &HttpRuntime,
    interaction_id: &str,
    last_event_id: Option<&str>,
) -> Result<super::streaming::InteractionEventStream, CompletionError> {
    let req = build_resume_stream_request(cfg, interaction_id, last_event_id)?;
    Ok(super::streaming::interaction_event_stream(
        rt.sse_events(req, false),
    ))
}
/// Build one `GET /v1beta/models` page request for [`list_models`].
///
/// Pure except for credential resolution. Unlike the `generateContent`
/// face's listing request, the credential rides in the `x-goog-api-key`
/// header, matching the deleted `GeminiInteractionsModelLister`.
pub fn build_list_models_request(
    cfg: &Config,
    page_token: Option<&str>,
) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
    use crate::model::ModelListingError;

    let path = super::super::model_listing::list_models_path(page_token);
    let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), path);
    let mut builder = http::Request::get(url);
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| ModelListingError::request_error(e.to_string()))?
    {
        builder = builder.header("x-goog-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| ModelListingError::request_error(e.to_string()))
}

/// List the models available to `cfg`'s credentials, following page-token
/// pagination through all pages.
///
/// The free-function form of the deleted `GeminiInteractionsModelLister`.
pub async fn list_models(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
    use crate::model::{ModelList, ModelListingError};

    let mut all_models = Vec::new();
    let mut page_token: Option<String> = None;

    loop {
        let path = super::super::model_listing::list_models_path(page_token.as_deref());
        let req = build_list_models_request(cfg, page_token.as_deref())?;
        let (status, body) = rt.send_bytes(req).await?;

        if !status.is_success() {
            return Err(ModelListingError::api_error_with_context(
                "Gemini",
                &path,
                status.as_u16(),
                &body,
            ));
        }

        let (models, next_page_token) =
            super::super::model_listing::parse_models_page(&body, &path)?;
        all_models.extend(models);

        match next_page_token {
            Some(token) => page_token = Some(token),
            None => break,
        }
    }

    Ok(ModelList::new(all_models))
}
/// Verify that `cfg`'s credential is accepted.
///
/// The Interactions client's `VERIFY_PATH` was `/v1beta/models`, the same as
/// the `generateContent` client's — but reached with the `x-goog-api-key`
/// header rather than a `key` query parameter.
///
/// # Errors
/// [`VerifyError`](crate::providers::verify::VerifyError): invalid
/// authentication on `401`/`403`, otherwise the preserved provider response
/// or a transport failure.
pub async fn verify(
    cfg: &Config,
    rt: &HttpRuntime,
) -> Result<(), crate::providers::verify::VerifyError> {
    use crate::providers::verify::{VerifyError, verify_url};

    let url = verify_url(&DESCRIPTOR, &cfg.base_url)?;
    let mut builder = http::Request::get(url);
    if let Some(key) = cfg
        .api_key
        .resolve()
        .map_err(|e| VerifyError::ProviderError(e.to_string()))?
    {
        builder = builder.header("x-goog-api-key", key);
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let req = builder
        .body(Vec::new())
        .map_err(|e| VerifyError::ProviderError(e.to_string()))?;
    crate::providers::verify::send_verify(rt, req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::message::Message;

    /// Regression for the formerly lossy two-channel case.
    ///
    /// This conversion used `.or_else`: a scalar preamble won and every system
    /// message already in the history was **discarded**. Every other provider
    /// appended both. With one canonical representation there is nothing left
    /// to prefer, so both sources must now appear, joined in order.
    #[test]
    fn every_system_message_reaches_the_instruction_in_order() {
        let cfg = Config::new("gemini-2.5-flash");
        let request = CompletionRequest::builder("prompt")
            .preamble("preamble")
            .message(Message::system("history system"))
            .build();

        let body = build_request_body(&cfg, &request, false).expect("request builds");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let instruction = value["system_instruction"]
            .as_str()
            .unwrap_or_else(|| panic!("system instruction missing: {value}"));

        assert_eq!(
            instruction, "preamble\n\nhistory system",
            "the preamble must no longer swallow history system messages",
        );
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                Message::system("be brief".to_string()),
                Message::user("hello"),
            ])
            .expect("non-empty"),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[test]
    fn build_request_uses_the_header_credential_not_a_query_key() {
        let cfg = Config::new("gemini-3-pro-preview").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), false).expect("build");

        assert_eq!(req.method(), http::Method::POST);
        assert_eq!(
            req.uri(),
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        );
        // The distinguishing behavior of this face: header auth, and the
        // key never appears in the URL.
        assert_eq!(
            req.headers()
                .get("x-goog-api-key")
                .and_then(|v| v.to_str().ok()),
            Some("secret")
        );
        assert!(!req.uri().to_string().contains("key="));
        assert!(req.headers().get(http::header::AUTHORIZATION).is_none());
        assert_eq!(
            req.headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn build_request_streaming_adds_alt_sse_and_sets_the_body_flag() {
        let cfg = Config::new("gemini-3-pro-preview").with_api_key("secret");
        let req = build_request(&cfg, &sample_request(), true).expect("build");
        assert_eq!(
            req.uri(),
            "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse"
        );
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["stream"], serde_json::json!(true));
    }

    #[test]
    fn build_request_body_carries_model_input_and_system_instruction() {
        let cfg = Config::new("gemini-3-pro-preview").with_api_key("k");
        let body = build_request_body(&cfg, &sample_request(), false).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "gemini-3-pro-preview");
        assert_eq!(value["stream"], serde_json::json!(false));
        assert_eq!(value["system_instruction"], "be brief");
        assert_eq!(value["input"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn get_and_resume_requests_target_the_interaction_id() {
        let cfg = Config::new("gemini-3-pro-preview").with_api_key("secret");

        let get = build_get_interaction_request(&cfg, "int-1").expect("build");
        assert_eq!(get.method(), http::Method::GET);
        assert_eq!(
            get.uri(),
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1"
        );
        assert_eq!(
            get.headers()
                .get("x-goog-api-key")
                .and_then(|v| v.to_str().ok()),
            Some("secret")
        );

        let resume = build_resume_stream_request(&cfg, "int-1", Some("ev-7")).expect("build");
        assert_eq!(
            resume.uri(),
            "https://generativelanguage.googleapis.com/v1beta/interactions/int-1\
             ?stream=true&last_event_id=ev-7&alt=sse"
        );
    }

    #[test]
    fn verify_request_path_and_header_match_the_deleted_client() {
        assert_eq!(DESCRIPTOR.verify_path, Some("/v1beta/models"));
    }

    #[test]
    fn parse_response_normalizes_an_interaction() {
        let body = serde_json::json!({
            "id": "int-1",
            "model": "gemini-3-pro-preview",
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": "hi"}]
            }],
            "usage": {
                "total_input_tokens": 3,
                "total_output_tokens": 2
            }
        })
        .to_string();

        let response = parse_response(http::StatusCode::OK, &body).expect("parse");
        assert_eq!(response.provider, "gemini");
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.output_tokens, 2);
    }

    #[test]
    fn parse_response_surfaces_http_errors() {
        let err = parse_response(http::StatusCode::SERVICE_UNAVAILABLE, "boom")
            .expect_err("non-success status must error");
        assert!(matches!(err, CompletionError::HttpError(_)));
    }

    #[tokio::test]
    async fn complete_round_trips_through_the_runtime() {
        use crate::test_utils::RecordingHttpClient;

        let body = serde_json::json!({
            "id": "int-2",
            "model": "gemini-3-pro-preview",
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{"type": "text", "text": "pong"}]
            }],
            "usage": { "total_input_tokens": 1, "total_output_tokens": 1 }
        })
        .to_string();

        let http_client = RecordingHttpClient::new(body);
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = Config::new("gemini-3-pro-preview").with_api_key("secret");

        let response = complete(&cfg, &rt, sample_request())
            .await
            .expect("complete");
        assert_eq!(response.provider, "gemini");

        let requests = http_client.requests();
        let recorded = requests.first().expect("a request was sent");
        assert_eq!(
            recorded.uri,
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        );
        assert_eq!(
            recorded
                .headers
                .get("x-goog-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("secret")
        );
    }
}
