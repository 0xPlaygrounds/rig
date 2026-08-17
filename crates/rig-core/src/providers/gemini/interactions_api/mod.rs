//! Google Gemini Interactions API integration.
//! From <https://ai.google.dev/api/interactions-api>

use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::message::{self, MimeType, Reasoning};
use crate::providers::internal::completion_send::send_completion;
use crate::providers::internal::envelope::DirectPayload;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::{Map, Value};
use tracing_futures::Instrument;
use url::form_urlencoded;

use super::client::InteractionsClient;

/// Streaming helpers for the Interactions API.
pub mod streaming;
pub use interactions_api_types::*;

// =================================================================
// Rig Implementation Types
// =================================================================

/// Stable descriptor name for the Gemini Interactions API.
///
/// The Interactions API is a second surface over the same provider, so it
/// reports the same descriptor as GenerateContent — matching the telemetry
/// spans, which have always shared it.
pub(crate) const PROVIDER_NAME: &str = "gcp.gemini";

/// Completion model wrapper for the Gemini Interactions API.
#[derive(Clone, Debug)]
pub struct InteractionsCompletionModel<T = reqwest::Client> {
    pub(crate) client: InteractionsClient<T>,
    pub model: String,
}

impl<T> InteractionsCompletionModel<T> {
    /// Create a new Interactions completion model for the given client and model name.
    pub fn new(client: InteractionsClient<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Create a new Interactions completion model using a string model name.
    pub fn with_model(client: InteractionsClient<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    /// Use the GenerateContent API instead of Interactions.
    pub fn generate_content_api(self) -> super::completion::CompletionModel<T> {
        super::completion::CompletionModel::with_model(
            self.client.generate_content_api(),
            &self.model,
        )
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
        stream_override: Option<bool>,
    ) -> Result<CreateInteractionRequest, CompletionError> {
        create_request_body(self.model.clone(), completion_request, stream_override)
    }
}

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    /// Create an interaction and return the raw response payload.
    pub async fn create_interaction(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Interaction, CompletionError> {
        let request = self.create_completion_request(completion_request, Some(false))?;
        self.client.create_interaction(request).await
    }

    /// Fetch an interaction by ID for polling background tasks.
    pub async fn get_interaction(
        &self,
        interaction_id: impl AsRef<str>,
    ) -> Result<Interaction, CompletionError> {
        self.client.get_interaction(interaction_id).await
    }

    /// Start an interaction and stream raw SSE events.
    pub async fn stream_interaction_events(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::InteractionEventStream, CompletionError> {
        let request = self.create_completion_request(completion_request, Some(true))?;
        self.client.stream_interaction_events(request).await
    }

    /// Resume an interaction stream by ID and optional last event ID.
    pub async fn stream_interaction_events_by_id(
        &self,
        interaction_id: impl AsRef<str>,
        last_event_id: Option<&str>,
    ) -> Result<streaming::InteractionEventStream, CompletionError> {
        self.client
            .stream_interaction_events_by_id(interaction_id, last_event_id)
            .await
    }
}

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    /// Execute a completion and return the Interactions API's own payload.
    ///
    /// This is the escape hatch for interaction fields rig does not normalize —
    /// step history, lifecycle status, hosted-tool exchanges. It shares the
    /// request builder, transport, telemetry, and error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one network
    /// request either way.
    pub async fn raw_completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Interaction, CompletionError> {
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &self.model,
            CompletionOperation::Interactions,
        )
        .system_instructions(
            completion_request.preamble.as_deref(),
            completion_request.record_telemetry_content,
        )
        .build();

        let request = self.create_completion_request(completion_request, Some(false))?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Gemini interactions completion request",
            &request,
        );

        let body = serde_json::to_vec(&request)?;
        let request = self
            .client
            .post("/v1beta/interactions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_completion::<_, DirectPayload<Interaction>, _>(
            &self.client,
            request,
            "Gemini interactions completion",
            // Gemini reports no transport request-id response header (verified
            // against the live API); the normalized id is None by design.
            None,
            |response| {
                let span = tracing::Span::current();
                span.record_response_metadata(response);
                let usage = crate::completion::Usage::from(response);
                span.record_token_usage(&usage);
            },
        )
        .instrument(span)
        .await
        .map(|(payload, _)| payload)
    }
}

impl<T> completion::CompletionModel for InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `try_into` consumes the raw value.
        let raw = self.raw_completion(completion_request).await?;
        let captured = serde_json::to_value(&raw)?;
        let response: completion::CompletionResponse = raw.try_into()?;
        Ok(response.with_raw(captured))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        InteractionsCompletionModel::stream(self, request).await
    }
}

impl<T> crate::client::ConstructCompletionModel<InteractionsClient<T>>
    for InteractionsCompletionModel<T>
where
    InteractionsClient<T>: Clone,
{
    fn construct(client: &InteractionsClient<T>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

impl<T> InteractionsClient<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    /// Create a new interaction and return the raw response payload.
    pub async fn create_interaction(
        &self,
        request: CreateInteractionRequest,
    ) -> Result<Interaction, CompletionError> {
        if request.stream == Some(true) {
            return Err(CompletionError::RequestError(Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "stream=true requires stream_interaction_events",
                ),
            )));
        }

        let body = serde_json::to_vec(&request)?;
        let request = self
            .post("/v1beta/interactions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_interaction_request(self, request).await
    }

    /// Fetch an interaction by ID (useful for polling background tasks).
    pub async fn get_interaction(
        &self,
        interaction_id: impl AsRef<str>,
    ) -> Result<Interaction, CompletionError> {
        let path = format!("/v1beta/interactions/{}", interaction_id.as_ref());
        let request = self
            .get(path)?
            .body(Vec::new())
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_interaction_request(self, request).await
    }

    /// Start an interaction and stream raw SSE events.
    pub async fn stream_interaction_events(
        &self,
        mut request: CreateInteractionRequest,
    ) -> Result<streaming::InteractionEventStream, CompletionError> {
        request.stream = Some(true);
        let body = serde_json::to_vec(&request)?;
        let request = self
            .post_sse("/v1beta/interactions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        Ok(streaming::stream_interaction_events(self.clone(), request))
    }

    /// Resume an interaction stream by ID and optional last event ID.
    pub async fn stream_interaction_events_by_id(
        &self,
        interaction_id: impl AsRef<str>,
        last_event_id: Option<&str>,
    ) -> Result<streaming::InteractionEventStream, CompletionError> {
        let path = build_interaction_stream_path(interaction_id.as_ref(), last_event_id);
        let request = self
            .get_sse(path)?
            .body(Vec::new())
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        Ok(streaming::stream_interaction_events(self.clone(), request))
    }
}

pub(crate) fn create_request_body(
    model: String,
    completion_request: CompletionRequest,
    stream_override: Option<bool>,
) -> Result<CreateInteractionRequest, CompletionError> {
    let chat_history = completion_request.chat_history_with_documents();

    let mut history = Vec::new();
    history.extend(chat_history);
    // functionResponse.name keys the replay: cross-provider ingested
    // results arrive with an empty name and their call carries it.
    crate::providers::internal::resolve_empty_tool_result_names(&mut history);
    let (history_system, history) = split_system_messages_from_history(history);

    let steps = history
        .into_iter()
        .map(Step::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| CompletionError::RequestError(Box::new(err)))?;

    let input = InteractionInput::Steps(steps);

    let raw_params = completion_request
        .additional_params
        .unwrap_or_else(|| Value::Object(Map::new()));

    let mut params: AdditionalParameters = serde_json::from_value(raw_params)?;

    let mut generation_config = params.generation_config.take().unwrap_or_default();
    if let Some(temp) = completion_request.temperature {
        generation_config.temperature = Some(temp);
    }
    if let Some(max_tokens) = completion_request.max_tokens {
        generation_config.max_output_tokens = Some(max_tokens);
    }
    if let Some(tool_choice) = completion_request.tool_choice {
        generation_config.tool_choice = Some(tool_choice.try_into()?);
    }
    let generation_config = if generation_config.is_empty() {
        None
    } else {
        Some(generation_config)
    };

    let system_instruction = completion_request
        .preamble
        .or_else(|| {
            if history_system.is_empty() {
                None
            } else {
                Some(history_system.join("\n\n"))
            }
        })
        .or(params.system_instruction.take());

    let mut tools = Vec::new();
    if !completion_request.tools.is_empty() {
        tools.extend(
            completion_request
                .tools
                .into_iter()
                .map(Tool::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if let Some(mut extra_tools) = params.tools.take() {
        tools.append(&mut extra_tools);
    }
    let tools = if tools.is_empty() { None } else { Some(tools) };

    let stream = stream_override.or(params.stream.take());

    let (agent, agent_config) = if params.agent.is_some() {
        (params.agent.take(), params.agent_config.take())
    } else {
        (None, None)
    };

    let response_format = params.response_format.take();
    let response_mime_type = params.response_mime_type.take();

    if response_format.is_some() && response_mime_type.is_none() {
        return Err(CompletionError::RequestError(Box::new(
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "response_mime_type is required when response_format is set",
            ),
        )));
    }

    Ok(CreateInteractionRequest {
        model: if agent.is_some() { None } else { Some(model) },
        agent,
        input,
        system_instruction,
        tools,
        response_format,
        response_mime_type,
        stream,
        store: params.store.take(),
        background: params.background.take(),
        generation_config,
        agent_config,
        response_modalities: params.response_modalities.take(),
        previous_interaction_id: params.previous_interaction_id.take(),
        additional_params: params.additional_params.take(),
    })
}

use super::completion::split_system_messages_from_history;

async fn send_interaction_request<T>(
    client: &InteractionsClient<T>,
    request: crate::http_client::Request<Vec<u8>>,
) -> Result<Interaction, CompletionError>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    let response = client.send::<_, Vec<u8>>(request).await?;

    if response.status().is_success() {
        let response_body = response
            .into_body()
            .await
            .map_err(CompletionError::HttpError)?;

        let response_text = String::from_utf8_lossy(&response_body).to_string();

        let response: Interaction = serde_json::from_slice(&response_body).map_err(|err| {
            tracing::error!(
                error = %err,
                body = %response_text,
                "Failed to deserialize Gemini interactions response"
            );
            CompletionError::JsonError(err)
        })?;

        Ok(response)
    } else {
        let status = response.status();
        let body = response
            .into_body()
            .await
            .map_err(CompletionError::HttpError)?;

        Err(CompletionError::from_http_response(
            status,
            String::from_utf8_lossy(&body),
        ))
    }
}

fn build_interaction_stream_path(interaction_id: &str, last_event_id: Option<&str>) -> String {
    let mut serializer = form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("stream", "true");
    if let Some(last_event_id) = last_event_id {
        serializer.append_pair("last_event_id", last_event_id);
    }
    format!(
        "/v1beta/interactions/{}?{}",
        interaction_id,
        serializer.finish()
    )
}

/// Normalize a Gemini Interactions API payload.
impl TryFrom<Interaction> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: Interaction) -> Result<Self, Self::Error> {
        let output_contents = response.output_contents();
        if output_contents.is_empty() {
            let message = match response.status.as_ref() {
                Some(InteractionStatus::InProgress) => {
                    "Interaction contained no outputs yet (status: InProgress). Use get_interaction for background tasks.".to_string()
                }
                Some(status) => format!("Interaction contained no outputs (status: {status:?})."),
                None => "Interaction contained no outputs".to_string(),
            };
            return Err(CompletionError::ResponseError(message));
        }

        let content = output_contents
            .into_iter()
            .filter_map(|output| match assistant_content_from_output(output) {
                Ok(Some(content)) => Some(Ok(content)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = crate::message::require_non_empty_response(content)?;

        let usage = response
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();

        let finish_reason = response.status.as_ref().map(map_interaction_status);

        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME)
                .with_optional_response_id(Some(response.id.as_str()).filter(|id| !id.is_empty()))
                .with_optional_model(response.model.as_deref())
                .with_optional_finish_reason(finish_reason),
        )
    }
}

fn assistant_content_from_output(
    output: Content,
) -> Result<Option<completion::AssistantContent>, CompletionError> {
    match output {
        Content::Text(TextContent { text, .. }) => {
            Ok(Some(completion::AssistantContent::text(text)))
        }
        Content::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
            ..
        }) => {
            let Some(name) = name else {
                return Ok(None);
            };
            // An id-less call mints its correlation handle — never
            // name-as-id, which collides two same-tool calls in one turn.
            Ok(Some(completion::AssistantContent::tool_call(
                id.unwrap_or_default(),
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
            )))
        }
        Content::Thought(ThoughtContent {
            summary, signature, ..
        }) => {
            let mut reasoning_content = summary
                .unwrap_or_default()
                .into_iter()
                .filter_map(|content| match content {
                    ThoughtSummaryContent::Text(text) => Some(message::ReasoningContent::Text {
                        text: text.text,
                        signature: None,
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();

            if reasoning_content.is_empty() {
                return Ok(None);
            }

            if let Some(signature) = signature
                && let Some(message::ReasoningContent::Text {
                    signature: first_signature,
                    ..
                }) = reasoning_content
                    .iter_mut()
                    .find(|content| matches!(content, message::ReasoningContent::Text { .. }))
            {
                *first_signature = Some(signature);
            }

            Ok(Some(completion::AssistantContent::Reasoning(Reasoning {
                id: None,
                content: reasoning_content,
            })))
        }
        Content::Image(ImageContent {
            data,
            uri,
            mime_type,
            ..
        }) => {
            let Some(mime_type) = mime_type else {
                return Err(CompletionError::ResponseError(
                    "Image output missing mime_type".to_owned(),
                ));
            };

            let media_type =
                message::ImageMediaType::from_mime_type(&mime_type).ok_or_else(|| {
                    CompletionError::ResponseError(format!(
                        "Unsupported image output mime type {mime_type}"
                    ))
                })?;

            let image = if let Some(data) = data {
                message::AssistantContent::image_base64(
                    data,
                    Some(media_type),
                    Some(message::ImageDetail::default()),
                )
            } else if let Some(uri) = uri {
                completion::AssistantContent::Image(message::Image {
                    data: message::DocumentSourceKind::Url(uri),
                    media_type: Some(media_type),
                    detail: Some(message::ImageDetail::default()),
                    additional_params: None,
                })
            } else {
                return Err(CompletionError::ResponseError(
                    "Image output missing data or uri".to_owned(),
                ));
            };

            Ok(Some(image))
        }
        _ => Ok(None),
    }
}

/// Shared preamble for Gemini Interactions media parts: require the media
/// type, render its MIME string, and split the source into data/uri.
fn media_parts<M: MimeType>(
    data: message::DocumentSourceKind,
    media_type: Option<M>,
    kind: &str,
) -> Result<(Option<String>, Option<String>, String), message::MessageError> {
    let media_type = media_type.ok_or_else(|| {
        message::MessageError::ConversionError(format!(
            "Media type for {kind} is required for Gemini"
        ))
    })?;
    let mime_type = media_type.to_mime_type().to_string();
    let (data, uri) = split_data_uri(data)?;
    Ok((data, uri, mime_type))
}

fn split_data_uri(
    src: message::DocumentSourceKind,
) -> Result<(Option<String>, Option<String>), message::MessageError> {
    match src {
        message::DocumentSourceKind::Url(uri) => Ok((None, Some(uri))),
        message::DocumentSourceKind::Base64(data) => Ok((Some(data), None)),
        message::DocumentSourceKind::String(data) => {
            Ok((Some(BASE64_STANDARD.encode(data.as_bytes())), None))
        }
        message::DocumentSourceKind::Raw(data) => Ok((Some(BASE64_STANDARD.encode(data)), None)),
        message::DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
            "Provider file IDs are not supported for Gemini Interactions inputs".to_string(),
        )),
        message::DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
            "Unknown content source".to_string(),
        )),
    }
}

/// Raw request/response types and convenience helpers for the Gemini Interactions API.
pub mod interactions_api_types {
    use super::{media_parts, split_data_uri};
    use crate::completion::{CompletionError, Usage};
    use crate::message::{self, MimeType};
    use crate::telemetry::ProviderResponseExt;
    use base64::{Engine, prelude::BASE64_STANDARD};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    // =================================================================
    // Request / Response Types
    // =================================================================

    /// Optional parameters for creating an interaction.
    #[derive(Debug, Deserialize, Serialize, Default, Clone)]
    #[serde(rename_all = "snake_case")]
    pub struct AdditionalParameters {
        pub agent: Option<String>,
        pub agent_config: Option<AgentConfig>,
        pub background: Option<bool>,
        pub generation_config: Option<GenerationConfig>,
        pub previous_interaction_id: Option<String>,
        pub response_modalities: Option<Vec<ResponseModality>>,
        pub response_format: Option<Value>,
        pub response_mime_type: Option<String>,
        pub store: Option<bool>,
        pub stream: Option<bool>,
        pub system_instruction: Option<String>,
        pub tools: Option<Vec<Tool>>,
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<Value>,
    }

    /// Request body for the create interaction endpoint.
    #[derive(Debug, Deserialize, Serialize, Clone)]
    #[serde(rename_all = "snake_case")]
    pub struct CreateInteractionRequest {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub agent: Option<String>,
        pub input: InteractionInput,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_instruction: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<Tool>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub store: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub background: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub generation_config: Option<GenerationConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub agent_config: Option<AgentConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_modalities: Option<Vec<ResponseModality>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub previous_interaction_id: Option<String>,
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<Value>,
    }

    /// Interaction response payload.
    #[derive(Clone, Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub struct Interaction {
        #[serde(default)]
        pub id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub agent: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub status: Option<InteractionStatus>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub object: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub created: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub updated: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub role: Option<String>,
        #[serde(default)]
        pub steps: Vec<Step>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<InteractionUsage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_instruction: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<Tool>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub background: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_modalities: Option<Vec<ResponseModality>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub previous_interaction_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input: Option<InteractionInput>,
    }

    impl From<&Interaction> for Usage {
        fn from(value: &Interaction) -> Usage {
            value.usage.as_ref().map(Usage::from).unwrap_or_default()
        }
    }

    impl From<Interaction> for Usage {
        fn from(value: Interaction) -> Usage {
            (&value).into()
        }
    }

    impl ProviderResponseExt for Interaction {
        type Usage = InteractionUsage;

        fn get_response_id(&self) -> Option<String> {
            if self.id.is_empty() {
                None
            } else {
                Some(self.id.clone())
            }
        }

        fn get_response_model_name(&self) -> Option<String> {
            self.model.clone()
        }

        fn get_text_response(&self) -> Option<String> {
            let text = self
                .output_contents()
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            if text.is_empty() { None } else { Some(text) }
        }

        fn get_usage(&self) -> Option<Self::Usage> {
            self.usage.clone()
        }
    }

    /// Groups tool calls and results of one built-in tool family for a single
    /// interaction.
    #[derive(Clone, Debug)]
    pub struct Exchange<C, R> {
        /// Call identifier used to match calls to results.
        pub call_id: Option<String>,
        /// One or more tool calls.
        pub calls: Vec<C>,
        /// One or more tool results.
        pub results: Vec<R>,
    }

    impl<C, R> Default for Exchange<C, R> {
        fn default() -> Self {
            Self {
                call_id: None,
                calls: Vec::new(),
                results: Vec::new(),
            }
        }
    }

    /// A tool call content type that carries an optional call identifier.
    trait ExchangeCall {
        fn id(&self) -> Option<&str>;
    }

    /// A tool result content type that carries an optional call identifier.
    trait ExchangeResult {
        fn call_id(&self) -> Option<&str>;
    }

    macro_rules! impl_exchange_ids {
        ($call:ty, $result:ty) => {
            impl ExchangeCall for $call {
                fn id(&self) -> Option<&str> {
                    self.id.as_deref()
                }
            }
            impl ExchangeResult for $result {
                fn call_id(&self) -> Option<&str> {
                    self.call_id.as_deref()
                }
            }
        };
    }

    impl_exchange_ids!(GoogleSearchCallContent, GoogleSearchResultContent);
    impl_exchange_ids!(UrlContextCallContent, UrlContextResultContent);
    impl_exchange_ids!(CodeExecutionCallContent, CodeExecutionResultContent);

    /// Pairs tool calls with their results by call_id.
    ///
    /// When a call_id is missing, results are grouped with the most recent
    /// call (identified or not) as a best-effort fallback.
    fn pair_exchanges<C, R>(
        contents: &[Content],
        as_call: impl Fn(&Content) -> Option<&C>,
        as_result: impl Fn(&Content) -> Option<&R>,
    ) -> Vec<Exchange<C, R>>
    where
        C: Clone + ExchangeCall,
        R: Clone + ExchangeResult,
    {
        let mut exchanges: Vec<Exchange<C, R>> = Vec::new();
        let mut last_call_index: Option<usize> = None;
        let position_of = |exchanges: &[Exchange<C, R>], call_id: &str| {
            exchanges
                .iter()
                .position(|exchange| exchange.call_id.as_deref() == Some(call_id))
        };

        for content in contents {
            if let Some(call) = as_call(content) {
                let index = match call.id() {
                    Some(call_id) => match position_of(&exchanges, call_id) {
                        Some(index) => {
                            if let Some(exchange) = exchanges.get_mut(index) {
                                exchange.calls.push(call.clone());
                            }
                            index
                        }
                        None => {
                            exchanges.push(Exchange {
                                call_id: Some(call_id.to_string()),
                                calls: vec![call.clone()],
                                results: Vec::new(),
                            });
                            exchanges.len() - 1
                        }
                    },
                    None => {
                        exchanges.push(Exchange {
                            call_id: None,
                            calls: vec![call.clone()],
                            results: Vec::new(),
                        });
                        exchanges.len() - 1
                    }
                };
                last_call_index = Some(index);
            } else if let Some(result) = as_result(content) {
                if let Some(call_id) = result.call_id() {
                    if let Some(index) = position_of(&exchanges, call_id) {
                        if let Some(exchange) = exchanges.get_mut(index) {
                            exchange.results.push(result.clone());
                        }
                    } else {
                        exchanges.push(Exchange {
                            call_id: Some(call_id.to_string()),
                            calls: Vec::new(),
                            results: vec![result.clone()],
                        });
                    }
                } else if let Some(index) = last_call_index {
                    if let Some(exchange) = exchanges.get_mut(index) {
                        exchange.results.push(result.clone());
                    }
                } else {
                    exchanges.push(Exchange {
                        call_id: None,
                        calls: Vec::new(),
                        results: vec![result.clone()],
                    });
                    last_call_index = Some(exchanges.len() - 1);
                }
            }
        }

        exchanges
    }

    /// Groups Google Search tool calls and results for a single interaction.
    pub type GoogleSearchExchange = Exchange<GoogleSearchCallContent, GoogleSearchResultContent>;

    impl GoogleSearchExchange {
        /// Collects all queries from the stored Google Search tool calls.
        pub fn queries(&self) -> Vec<String> {
            self.calls
                .iter()
                .filter_map(|call| call.arguments.as_ref()?.queries.as_ref())
                .flatten()
                .cloned()
                .collect()
        }

        /// Collects all Google Search result entries from tool results.
        pub fn result_items(&self) -> Vec<GoogleSearchResult> {
            self.results
                .iter()
                .filter_map(|result| result.result.as_ref())
                .flatten()
                .cloned()
                .collect()
        }
    }

    /// Groups URL context tool calls and results for a single interaction.
    pub type UrlContextExchange = Exchange<UrlContextCallContent, UrlContextResultContent>;

    impl UrlContextExchange {
        /// Collects all URLs from the stored URL context tool calls.
        pub fn urls(&self) -> Vec<String> {
            self.calls
                .iter()
                .filter_map(|call| call.arguments.as_ref()?.urls.as_ref())
                .flatten()
                .cloned()
                .collect()
        }

        /// Collects all URL context result entries from tool results.
        pub fn result_items(&self) -> Vec<UrlContextResult> {
            self.results
                .iter()
                .filter_map(|result| result.result.as_ref())
                .flatten()
                .cloned()
                .collect()
        }
    }

    /// Groups code execution tool calls and results for a single interaction.
    pub type CodeExecutionExchange = Exchange<CodeExecutionCallContent, CodeExecutionResultContent>;

    impl CodeExecutionExchange {
        /// Collects all code snippets from the stored code execution tool calls.
        pub fn code_snippets(&self) -> Vec<String> {
            self.calls
                .iter()
                .filter_map(|call| call.arguments.as_ref()?.code.clone())
                .collect()
        }

        /// Collects all code execution outputs from tool results.
        pub fn outputs(&self) -> Vec<String> {
            self.results
                .iter()
                .filter_map(|result| result.result.clone())
                .collect()
        }
    }

    /// Generates the `Interaction` accessor family for one built-in tool:
    /// the call_id-grouped exchanges plus flattened views over their calls,
    /// results, and per-exchange collector methods.
    macro_rules! interaction_exchange_accessors {
        (
            $tool:literal, $exchange:ty, $call_variant:ident, $result_variant:ident,
            $exchanges_fn:ident, $call_contents_fn:ident -> $call_ty:ty,
            $result_contents_fn:ident -> $result_ty:ty,
            $($flat_doc:literal $flat_fn:ident => $method:ident -> $flat_ty:ty),* $(,)?
        ) => {
            #[doc = concat!("Groups ", $tool, " tool calls and results by call_id.")]
            ///
            /// When a call_id is missing, results are grouped with the most recent
            /// call (identified or not) as a best-effort fallback.
            pub fn $exchanges_fn(&self) -> Vec<$exchange> {
                pair_exchanges(
                    &self.output_contents(),
                    |content| match content {
                        Content::$call_variant(call) => Some(call),
                        _ => None,
                    },
                    |content| match content {
                        Content::$result_variant(result) => Some(result),
                        _ => None,
                    },
                )
            }

            #[doc = concat!("Collects ", $tool, " tool call contents from the interaction outputs.")]
            pub fn $call_contents_fn(&self) -> Vec<$call_ty> {
                self.$exchanges_fn()
                    .into_iter()
                    .flat_map(|exchange| exchange.calls)
                    .collect()
            }

            #[doc = concat!("Collects ", $tool, " result contents from the interaction outputs.")]
            pub fn $result_contents_fn(&self) -> Vec<$result_ty> {
                self.$exchanges_fn()
                    .into_iter()
                    .flat_map(|exchange| exchange.results)
                    .collect()
            }

            $(
                #[doc = $flat_doc]
                pub fn $flat_fn(&self) -> Vec<$flat_ty> {
                    self.$exchanges_fn()
                        .into_iter()
                        .flat_map(|exchange| exchange.$method())
                        .collect()
                }
            )*
        };
    }

    impl Interaction {
        pub(crate) fn output_contents(&self) -> Vec<Content> {
            self.steps.iter().flat_map(Step::output_contents).collect()
        }

        interaction_exchange_accessors!(
            "Google Search", GoogleSearchExchange, GoogleSearchCall, GoogleSearchResult,
            google_search_exchanges,
            google_search_call_contents -> GoogleSearchCallContent,
            google_search_result_contents -> GoogleSearchResultContent,
            "Collects all Google Search queries from tool calls in the outputs."
                google_search_queries => queries -> String,
            "Collects all Google Search result entries from tool results in the outputs."
                google_search_results => result_items -> GoogleSearchResult,
        );

        interaction_exchange_accessors!(
            "URL context", UrlContextExchange, UrlContextCall, UrlContextResult,
            url_context_exchanges,
            url_context_call_contents -> UrlContextCallContent,
            url_context_result_contents -> UrlContextResultContent,
            "Collects all URLs from URL context tool calls in the outputs."
                url_context_urls => urls -> String,
            "Collects all URL context result entries from tool results in the outputs."
                url_context_results => result_items -> UrlContextResult,
        );

        interaction_exchange_accessors!(
            "code execution", CodeExecutionExchange, CodeExecutionCall, CodeExecutionResult,
            code_execution_exchanges,
            code_execution_call_contents -> CodeExecutionCallContent,
            code_execution_result_contents -> CodeExecutionResultContent,
            "Collects all code snippets from code execution calls in the outputs."
                code_execution_snippets => code_snippets -> String,
            "Collects all code execution outputs from tool results in the outputs."
                code_execution_outputs => outputs -> String,
        );

        /// Returns concatenated text outputs with inline citations appended.
        pub fn text_with_inline_citations(&self) -> Option<String> {
            let text = self
                .output_contents()
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.with_inline_citations()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            if text.is_empty() { None } else { Some(text) }
        }

        /// Returns true when the interaction is in a terminal state.
        pub fn is_terminal(&self) -> bool {
            self.status
                .as_ref()
                .is_some_and(InteractionStatus::is_terminal)
        }

        /// Returns true when the interaction completed successfully.
        pub fn is_completed(&self) -> bool {
            matches!(self.status, Some(InteractionStatus::Completed))
        }
    }

    /// Lifecycle status of an interaction.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum InteractionStatus {
        InProgress,
        RequiresAction,
        Incomplete,
        BudgetExceeded,
        Completed,
        Failed,
        Cancelled,
        /// A status this crate does not know yet. Google adds wire values
        /// without notice; carrying the spelling verbatim keeps the whole
        /// payload deserializable instead of failing on the new value.
        #[serde(untagged)]
        Unknown(String),
    }

    impl InteractionStatus {
        /// Returns true when polling can stop: the status will not advance
        /// on its own.
        ///
        /// The known *in-flight* statuses are the allowlist, so a status this
        /// crate does not know yet reads as terminal: a poll loop that treated
        /// an unknown status as in-flight would wait on it forever, whereas
        /// surfacing it lets the caller act on the provider's own spelling.
        ///
        /// [`InteractionStatus::RequiresAction`] is terminal *for the poll*
        /// even though the interaction itself is resumable: it only advances
        /// when the caller submits tool results, so waiting on it can never
        /// succeed. Callers must branch on it as a distinct, resumable
        /// outcome rather than a completion.
        pub fn is_terminal(&self) -> bool {
            !matches!(self, InteractionStatus::InProgress)
        }

        /// The exact spelling the Interactions API uses for this status on the
        /// wire.
        ///
        /// Spelled out rather than derived from `Debug` (which would yield
        /// `BudgetExceeded`, not `budget_exceeded`) so the string that reaches
        /// [`crate::completion::FinishReason::Other`] is the provider's own.
        pub fn as_wire_str(&self) -> &str {
            match self {
                Self::InProgress => "in_progress",
                Self::RequiresAction => "requires_action",
                Self::Incomplete => "incomplete",
                Self::BudgetExceeded => "budget_exceeded",
                Self::Completed => "completed",
                Self::Failed => "failed",
                Self::Cancelled => "cancelled",
                Self::Unknown(status) => status,
            }
        }
    }

    /// Map an interaction's lifecycle status onto rig's normalized finish
    /// reasons.
    ///
    /// The Interactions API has no `finishReason` field — the interaction's
    /// terminal state is the closest equivalent. Only the three statuses with a
    /// normalized counterpart are folded in; the rest (including the
    /// non-terminal `in_progress`) are carried verbatim rather than guessed at.
    pub(crate) fn map_interaction_status(
        status: &InteractionStatus,
    ) -> crate::completion::FinishReason {
        match status {
            InteractionStatus::Completed => crate::completion::FinishReason::Stop,
            InteractionStatus::RequiresAction => crate::completion::FinishReason::ToolCalls,
            InteractionStatus::BudgetExceeded => crate::completion::FinishReason::Length,
            other => crate::completion::FinishReason::Other(other.as_wire_str().to_owned()),
        }
    }

    /// Token usage metadata for an interaction.
    #[derive(Clone, Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub struct InteractionUsage {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_tokens: Option<u64>,
    }

    impl From<&InteractionUsage> for Usage {
        fn from(value: &InteractionUsage) -> Usage {
            let mut usage = Usage::new();
            usage.input_tokens = value.total_input_tokens.unwrap_or_default();
            usage.output_tokens = value.total_output_tokens.unwrap_or_default();
            usage.total_tokens = value
                .total_tokens
                .unwrap_or(usage.input_tokens + usage.output_tokens);
            usage
        }
    }

    impl From<InteractionUsage> for Usage {
        fn from(value: InteractionUsage) -> Usage {
            (&value).into()
        }
    }

    /// Input payload accepted by the Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum InteractionInput {
        Text(String),
        Content(Content),
        Steps(Vec<Step>),
        Contents(Vec<Content>),
    }

    /// Single interaction step.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Step {
        // `content` is defaulted: a streaming `step.start` announces the step
        // with the content omitted (it follows in `step.delta` events), e.g.
        // `{"type":"model_output"}` on the recorded wire.
        UserInput {
            #[serde(default)]
            content: Vec<Content>,
        },
        ModelOutput {
            #[serde(default)]
            content: Vec<Content>,
        },
        Thought(ThoughtContent),
        FunctionCall(FunctionCallContent),
        FunctionResult(FunctionResultContent),
        CodeExecutionCall(CodeExecutionCallContent),
        CodeExecutionResult(CodeExecutionResultContent),
        UrlContextCall(UrlContextCallContent),
        UrlContextResult(UrlContextResultContent),
        GoogleSearchCall(GoogleSearchCallContent),
        GoogleSearchResult(GoogleSearchResultContent),
        McpServerToolCall(McpServerToolCallContent),
        McpServerToolResult(McpServerToolResultContent),
        FileSearchResult(FileSearchResultContent),
    }

    impl Step {
        fn output_contents(&self) -> Vec<Content> {
            match self {
                Step::UserInput { .. } => Vec::new(),
                Step::ModelOutput { content } => content.clone(),
                Step::Thought(content) => vec![Content::Thought(content.clone())],
                Step::FunctionCall(content) => vec![Content::FunctionCall(content.clone())],
                Step::FunctionResult(content) => vec![Content::FunctionResult(content.clone())],
                Step::CodeExecutionCall(content) => {
                    vec![Content::CodeExecutionCall(content.clone())]
                }
                Step::CodeExecutionResult(content) => {
                    vec![Content::CodeExecutionResult(content.clone())]
                }
                Step::UrlContextCall(content) => vec![Content::UrlContextCall(content.clone())],
                Step::UrlContextResult(content) => {
                    vec![Content::UrlContextResult(content.clone())]
                }
                Step::GoogleSearchCall(content) => {
                    vec![Content::GoogleSearchCall(content.clone())]
                }
                Step::GoogleSearchResult(content) => {
                    vec![Content::GoogleSearchResult(content.clone())]
                }
                Step::McpServerToolCall(content) => {
                    vec![Content::McpServerToolCall(content.clone())]
                }
                Step::McpServerToolResult(content) => {
                    vec![Content::McpServerToolResult(content.clone())]
                }
                Step::FileSearchResult(content) => {
                    vec![Content::FileSearchResult(content.clone())]
                }
            }
        }
    }

    impl TryFrom<crate::completion::Message> for Step {
        type Error = message::MessageError;

        fn try_from(message: crate::completion::Message) -> Result<Self, Self::Error> {
            match message {
                crate::completion::Message::System { content } => Ok(Self::UserInput {
                    content: vec![Content::Text(TextContent {
                        text: content,
                        annotations: None,
                    })],
                }),
                crate::completion::Message::User { content } => {
                    let content = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self::UserInput { content })
                }
                crate::completion::Message::Assistant { content, .. } => {
                    let content = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self::ModelOutput { content })
                }
            }
        }
    }

    // =================================================================
    // Content
    // =================================================================

    /// Text annotation metadata for citations.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Annotation {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub start_index: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub end_index: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub source: Option<String>,
    }

    /// Normalized citation extracted from an annotation.
    #[derive(Clone, Debug)]
    pub struct Citation {
        pub start_index: usize,
        pub end_index: usize,
        pub source: String,
    }

    /// Text content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct TextContent {
        pub text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Vec<Annotation>>,
    }

    impl TextContent {
        /// Collects citations extracted from annotations.
        pub fn citations(&self) -> Vec<Citation> {
            let mut citations = Vec::new();
            let Some(annotations) = self.annotations.as_ref() else {
                return citations;
            };

            for annotation in annotations {
                let (Some(start), Some(end), Some(source)) = (
                    annotation.start_index,
                    annotation.end_index,
                    annotation.source.as_ref(),
                ) else {
                    continue;
                };

                if start < 0 || end < 0 {
                    continue;
                }
                let start = start as usize;
                let end = end as usize;
                if end <= start || end > self.text.len() {
                    continue;
                }
                if !self.text.is_char_boundary(start) || !self.text.is_char_boundary(end) {
                    continue;
                }

                citations.push(Citation {
                    start_index: start,
                    end_index: end,
                    source: source.clone(),
                });
            }

            citations.sort_by(|a, b| {
                a.start_index
                    .cmp(&b.start_index)
                    .then_with(|| a.end_index.cmp(&b.end_index))
            });

            citations
        }

        /// Returns the text with inline citations appended after annotated spans.
        pub fn with_inline_citations(&self) -> String {
            let citations = self.citations();
            if citations.is_empty() {
                return self.text.clone();
            }

            let mut source_order = Vec::new();
            for citation in &citations {
                if !source_order.contains(&citation.source) {
                    source_order.push(citation.source.clone());
                }
            }

            let mut inserts = citations
                .iter()
                .map(|citation| {
                    let index = source_order
                        .iter()
                        .position(|source| source == &citation.source)
                        .map(|idx| idx + 1)
                        .unwrap_or(0);
                    (
                        citation.start_index,
                        citation.end_index,
                        index,
                        &citation.source,
                    )
                })
                .collect::<Vec<_>>();

            inserts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

            let mut text = self.text.clone();
            for (_, end, index, source) in inserts {
                if index == 0 {
                    continue;
                }
                let citation = format!("[{}]({})", index, source);
                text.insert_str(end, &citation);
            }

            text
        }
    }

    /// Image content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ImageContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resolution: Option<MediaResolution>,
    }

    /// Audio content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct AudioContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }

    /// Document content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct DocumentContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }

    /// Video content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct VideoContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resolution: Option<MediaResolution>,
    }

    /// Thought summary content.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ThoughtContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub summary: Option<Vec<ThoughtSummaryContent>>,
    }

    /// Thought summary item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum ThoughtSummaryContent {
        Text(TextContent),
        Image(ImageContent),
    }

    /// Function call content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FunctionCallContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Function result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FunctionResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Arguments for a code execution call.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct CodeExecutionCallArguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub language: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub code: Option<String>,
    }

    /// Code execution call content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct CodeExecutionCallContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<CodeExecutionCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Code execution result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct CodeExecutionResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Arguments for a URL context call.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextCallArguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub urls: Option<Vec<String>>,
    }

    /// URL context call content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextCallContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<UrlContextCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// URL context result entry.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub status: Option<String>,
    }

    /// URL context result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<UrlContextResult>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Arguments for a Google Search call.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchCallArguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub queries: Option<Vec<String>>,
    }

    /// Google Search call content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchCallContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<GoogleSearchCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Google Search result entry.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub rendered_content: Option<String>,
    }

    /// Google Search result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<GoogleSearchResult>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// MCP server tool call content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct McpServerToolCallContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// MCP server tool result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct McpServerToolResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// File search result entry.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FileSearchResult {
        pub title: String,
        pub text: String,
        pub file_search_store: String,
    }

    /// File search result content item.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FileSearchResultContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<FileSearchResult>>,
    }

    /// Content item produced or consumed by the Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Content {
        Text(TextContent),
        Image(ImageContent),
        Audio(AudioContent),
        Document(DocumentContent),
        Video(VideoContent),
        Thought(ThoughtContent),
        FunctionCall(FunctionCallContent),
        FunctionResult(FunctionResultContent),
        CodeExecutionCall(CodeExecutionCallContent),
        CodeExecutionResult(CodeExecutionResultContent),
        UrlContextCall(UrlContextCallContent),
        UrlContextResult(UrlContextResultContent),
        GoogleSearchCall(GoogleSearchCallContent),
        GoogleSearchResult(GoogleSearchResultContent),
        McpServerToolCall(McpServerToolCallContent),
        McpServerToolResult(McpServerToolResultContent),
        FileSearchResult(FileSearchResultContent),
    }

    fn rich_function_result_block(
        content: message::ToolResultContent,
    ) -> Result<Value, message::MessageError> {
        let content = match content {
            message::ToolResultContent::Text(text) => Content::Text(TextContent {
                text: text.text,
                annotations: None,
            }),
            message::ToolResultContent::Json { value } => Content::Text(TextContent {
                text: value.to_string(),
                annotations: None,
            }),
            message::ToolResultContent::Image(message::Image {
                data, media_type, ..
            }) => {
                let media_type = media_type.ok_or_else(|| {
                    message::MessageError::ConversionError(
                        "Image media type is required for Gemini Interactions tool results"
                            .to_string(),
                    )
                })?;
                let (data, uri) = split_data_uri(data)?;

                Content::Image(ImageContent {
                    data,
                    uri,
                    mime_type: Some(media_type.to_mime_type().to_string()),
                    resolution: None,
                })
            }
        };

        serde_json::to_value(content).map_err(|err| {
            message::MessageError::ConversionError(format!(
                "Failed to serialize Gemini Interactions tool result content: {err}"
            ))
        })
    }

    impl TryFrom<message::UserContent> for Content {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text, .. }) => {
                    Ok(Self::Text(TextContent {
                        text,
                        annotations: None,
                    }))
                }
                message::UserContent::ToolResult(tool_result) => {
                    // The wire requires a call id: the provider-issued one
                    // when it exists, else rig's minted handle — always
                    // present, so the old "results require call_id" error
                    // is unrepresentable.
                    let call_id = tool_result.wire_call_id().to_owned();
                    let name = tool_result.name;

                    let mut contents = tool_result.content.into_iter().collect::<Vec<_>>();
                    let result = if contents.len() == 1 {
                        let content = contents.pop().ok_or_else(|| {
                            message::MessageError::ConversionError(
                                "Tool result content must not be empty".to_string(),
                            )
                        })?;

                        match content {
                            message::ToolResultContent::Text(text) => Value::String(text.text),
                            message::ToolResultContent::Json { value } => match value {
                                value @ (Value::String(_) | Value::Object(_)) => value,
                                value => Value::Array(vec![rich_function_result_block(
                                    message::ToolResultContent::Json { value },
                                )?]),
                            },
                            rich_content => {
                                Value::Array(vec![rich_function_result_block(rich_content)?])
                            }
                        }
                    } else {
                        Value::Array(
                            contents
                                .into_iter()
                                .map(rich_function_result_block)
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                    };

                    Ok(Self::FunctionResult(FunctionResultContent {
                        // The executed tool's name travels as required data.
                        name: Some(name),
                        is_error: None,
                        result: Some(result),
                        call_id: Some(call_id),
                    }))
                }
                message::UserContent::Image(message::Image {
                    data, media_type, ..
                }) => {
                    let (data, uri, mime_type) = media_parts(data, media_type, "image")?;
                    Ok(Self::Image(ImageContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                        resolution: None,
                    }))
                }
                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => {
                    let (data, uri, mime_type) = media_parts(data, media_type, "audio")?;
                    Ok(Self::Audio(AudioContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                    }))
                }
                message::UserContent::Video(message::Video {
                    data, media_type, ..
                }) => {
                    let (data, uri, mime_type) = media_parts(data, media_type, "video")?;
                    Ok(Self::Video(VideoContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                        resolution: None,
                    }))
                }
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => {
                    let media_type = media_type.ok_or_else(|| {
                        message::MessageError::ConversionError(
                            "Media type for document is required for Gemini".to_string(),
                        )
                    })?;
                    if matches!(media_type, message::DocumentMediaType::TXT) {
                        let text = match data {
                            message::DocumentSourceKind::String(text) => text,
                            message::DocumentSourceKind::Base64(data) => {
                                let decoded = BASE64_STANDARD.decode(data).map_err(|error| {
                                    message::MessageError::ConversionError(format!(
                                        "Failed to decode text document base64 data: {error}"
                                    ))
                                })?;
                                String::from_utf8(decoded).map_err(|error| {
                                    message::MessageError::ConversionError(format!(
                                        "Text document data must be UTF-8: {error}"
                                    ))
                                })?
                            }
                            message::DocumentSourceKind::Raw(data) => String::from_utf8(data)
                                .map_err(|error| {
                                    message::MessageError::ConversionError(format!(
                                        "Text document data must be UTF-8: {error}"
                                    ))
                                })?,
                            message::DocumentSourceKind::Url(_) => {
                                return Err(message::MessageError::ConversionError(
                                    "Text document URLs are not supported for Gemini Interactions inputs"
                                        .to_string(),
                                ));
                            }
                            message::DocumentSourceKind::FileId(_) => {
                                return Err(message::MessageError::ConversionError(
                                    "Provider file IDs are not supported for Gemini Interactions inputs"
                                        .to_string(),
                                ));
                            }
                            message::DocumentSourceKind::Unknown => {
                                return Err(message::MessageError::ConversionError(
                                    "Unknown content source".to_string(),
                                ));
                            }
                        };
                        return Ok(Self::Text(TextContent {
                            text,
                            annotations: None,
                        }));
                    }
                    let (data, uri, mime_type) = media_parts(data, Some(media_type), "document")?;
                    Ok(Self::Document(DocumentContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                    }))
                }
            }
        }
    }

    impl TryFrom<message::AssistantContent> for Content {
        type Error = message::MessageError;

        fn try_from(content: message::AssistantContent) -> Result<Self, Self::Error> {
            match content {
                message::AssistantContent::Text(message::Text { text, .. }) => {
                    Ok(Self::Text(TextContent {
                        text,
                        annotations: None,
                    }))
                }
                message::AssistantContent::ToolCall(tool_call) => {
                    let call_id = tool_call.wire_call_id().to_owned();
                    Ok(Self::FunctionCall(FunctionCallContent {
                        name: Some(tool_call.function.name),
                        arguments: Some(tool_call.function.arguments),
                        id: Some(call_id),
                    }))
                }
                message::AssistantContent::Reasoning(message::Reasoning { content, .. }) => {
                    let mut signature = None;
                    let summary = content
                        .into_iter()
                        .map(|reasoning_content| {
                            let text = match reasoning_content {
                                message::ReasoningContent::Text {
                                    text,
                                    signature: content_signature,
                                } => {
                                    if signature.is_none() {
                                        signature = content_signature;
                                    }
                                    text
                                }
                                message::ReasoningContent::Summary(text)
                                | message::ReasoningContent::Encrypted(text) => text,
                                message::ReasoningContent::Redacted { data } => data,
                            };

                            ThoughtSummaryContent::Text(TextContent {
                                text,
                                annotations: None,
                            })
                        })
                        .collect();

                    Ok(Self::Thought(ThoughtContent {
                        signature,
                        summary: Some(summary),
                    }))
                }
                message::AssistantContent::Image(message::Image {
                    data, media_type, ..
                }) => {
                    let media_type = media_type.ok_or_else(|| {
                        message::MessageError::ConversionError(
                            "Media type for image is required for Gemini".to_string(),
                        )
                    })?;
                    let mime_type = media_type.to_mime_type().to_string();
                    let (data, uri) = split_data_uri(data)?;
                    Ok(Self::Image(ImageContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                        resolution: None,
                    }))
                }
            }
        }
    }

    // =================================================================
    // Tools / Config
    // =================================================================

    /// Response modalities supported by the model.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ResponseModality {
        Text,
        Image,
        Audio,
    }

    /// Thinking depth hint for generation.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ThinkingLevel {
        Minimal,
        Low,
        Medium,
        High,
    }

    /// Thinking summary behavior.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ThinkingSummaries {
        Auto,
        None,
    }

    /// Speech synthesis configuration.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub struct SpeechConfig {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub voice: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub language: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub speaker: Option<String>,
    }

    /// Generation configuration for the Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub struct GenerationConfig {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub seed: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<ToolChoice>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_level: Option<ThinkingLevel>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_summaries: Option<ThinkingSummaries>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub speech_config: Option<Vec<SpeechConfig>>,
    }

    impl GenerationConfig {
        /// Returns true when no generation fields are set.
        pub fn is_empty(&self) -> bool {
            self.temperature.is_none()
                && self.top_p.is_none()
                && self.seed.is_none()
                && self.stop_sequences.is_none()
                && self.tool_choice.is_none()
                && self.thinking_level.is_none()
                && self.thinking_summaries.is_none()
                && self.max_output_tokens.is_none()
                && self.speech_config.is_none()
        }
    }

    /// Tool selection strategy.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum ToolChoice {
        Type(ToolChoiceType),
        Config(ToolChoiceConfig),
    }

    /// Tool selection mode.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ToolChoiceType {
        Auto,
        Any,
        None,
        Validated,
    }

    /// Tool selection configuration.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ToolChoiceConfig {
        pub allowed_tools: AllowedTools,
    }

    /// Allowed tools for tool selection.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct AllowedTools {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mode: Option<ToolChoiceType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<String>>,
    }

    /// Tool definition for Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Tool {
        Function(FunctionTool),
        GoogleSearch,
        CodeExecution,
        UrlContext,
        ComputerUse(ComputerUseTool),
        McpServer(McpServerTool),
        FileSearch(FileSearchTool),
    }

    /// Function tool definition.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FunctionTool {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<Value>,
    }

    /// Computer use tool configuration.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ComputerUseTool {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub environment: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub excluded_predefined_functions: Option<Vec<String>>,
    }

    /// MCP server tool configuration.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct McpServerTool {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub headers: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub allowed_tools: Option<AllowedTools>,
    }

    /// File search tool configuration.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FileSearchTool {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_search_store_names: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub metadata_filter: Option<String>,
    }

    impl TryFrom<crate::completion::ToolDefinition> for Tool {
        type Error = CompletionError;

        fn try_from(tool: crate::completion::ToolDefinition) -> Result<Self, Self::Error> {
            Ok(Tool::Function(FunctionTool {
                name: Some(tool.name),
                description: Some(tool.description),
                parameters: Some(tool.parameters),
            }))
        }
    }

    impl TryFrom<message::ToolChoice> for ToolChoice {
        type Error = CompletionError;

        fn try_from(tool_choice: message::ToolChoice) -> Result<Self, Self::Error> {
            match tool_choice {
                message::ToolChoice::Auto => Ok(ToolChoice::Type(ToolChoiceType::Auto)),
                message::ToolChoice::None => Ok(ToolChoice::Type(ToolChoiceType::None)),
                message::ToolChoice::Required => Ok(ToolChoice::Type(ToolChoiceType::Any)),
                message::ToolChoice::Specific { function_names } => {
                    Ok(ToolChoice::Config(ToolChoiceConfig {
                        allowed_tools: AllowedTools {
                            mode: Some(ToolChoiceType::Validated),
                            tools: Some(function_names),
                        },
                    }))
                }
            }
        }
    }

    /// Agent configuration for Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "kebab-case")]
    pub enum AgentConfig {
        Dynamic,
        DeepResearch {
            #[serde(skip_serializing_if = "Option::is_none")]
            thinking_summaries: Option<ThinkingSummaries>,
        },
    }

    /// Media resolution hint for multimodal content.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum MediaResolution {
        Low,
        Medium,
        High,
        UltraHigh,
    }

    // =================================================================
    // Streaming Events
    // =================================================================

    /// Server-sent event payloads for streaming interactions.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "event_type")]
    pub enum InteractionSseEvent {
        #[serde(rename = "interaction.created")]
        InteractionCreated {
            interaction: Interaction,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "interaction.completed")]
        InteractionCompleted {
            interaction: Interaction,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "interaction.status_update")]
        InteractionStatusUpdate {
            interaction_id: String,
            status: InteractionStatus,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "step.start")]
        StepStart {
            index: u32,
            step: Step,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "step.delta")]
        StepDelta {
            index: u32,
            delta: ContentDelta,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "step.stop")]
        StepStop {
            index: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "error")]
        Error {
            error: ErrorEvent,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
    }

    /// Error payload for streaming events.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ErrorEvent {
        pub code: String,
        pub message: String,
    }

    /// Content delta item in streaming events.
    ///
    /// Most deltas repeat a whole [`Content`] payload rather than a fragment of
    /// one, so they reuse the `*Content` types directly; the wire tags come
    /// from this enum's own `type` tagging. Only the variants whose payloads
    /// genuinely differ from their `Content` counterpart — a partial text run,
    /// a raw arguments fragment, and the identity-less thought deltas — carry
    /// their own struct.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ContentDelta {
        Text(TextDelta),
        Image(ImageContent),
        Audio(AudioContent),
        Document(DocumentContent),
        Video(VideoContent),
        ThoughtSummary(ThoughtSummaryDelta),
        ThoughtSignature(ThoughtSignatureDelta),
        FunctionCall(FunctionCallContent),
        ArgumentsDelta(ArgumentsDelta),
        FunctionResult(FunctionResultContent),
        CodeExecutionCall(CodeExecutionCallContent),
        CodeExecutionResult(CodeExecutionResultContent),
        UrlContextCall(UrlContextCallContent),
        UrlContextResult(UrlContextResultContent),
        GoogleSearchCall(GoogleSearchCallContent),
        GoogleSearchResult(GoogleSearchResultContent),
        McpServerToolCall(McpServerToolCallContent),
        McpServerToolResult(McpServerToolResultContent),
        FileSearchResult(FileSearchResultContent),
    }

    /// Streaming function-call arguments fragment: the wire fragments a
    /// `function_call` step's arguments as raw JSON text across
    /// `arguments_delta` events at the step's index (recorded live in
    /// `streaming_grammar/interactions_same_tool_twice`; the `step.start`
    /// announces the call with `"arguments": {}` and the real payload
    /// arrives here).
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ArgumentsDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<String>,
    }

    /// Streaming text delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct TextDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Vec<Annotation>>,
    }

    /// Streaming thought summary delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ThoughtSummaryDelta {
        pub content: ThoughtSummaryContent,
    }

    /// Streaming thought signature delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ThoughtSignatureDelta {
        pub signature: String,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionRequest, Message};
    use crate::message::{self, ToolChoice as MessageToolChoice};
    use serde_json::json;

    #[test]
    fn test_create_request_body_simple() {
        let prompt = Message::User {
            content: vec![message::UserContent::text("Hello")],
        };

        let request = CompletionRequest {
            record_telemetry_content: false,
            model: None,
            preamble: Some("Be precise.".to_string()),
            chat_history: vec![prompt],
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(128),
            tool_choice: Some(MessageToolChoice::Required),
            additional_params: None,
            output_schema: None,
        };

        let result = create_request_body("gemini-2.5-flash".to_string(), request, Some(false))
            .expect("request should build");

        assert_eq!(result.model.as_deref(), Some("gemini-2.5-flash"));
        assert!(result.agent.is_none());
        assert_eq!(result.stream, Some(false));
        assert_eq!(result.system_instruction.as_deref(), Some("Be precise."));

        let config = result.generation_config.expect("generation config missing");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(128));
        assert!(matches!(
            config.tool_choice,
            Some(ToolChoice::Type(ToolChoiceType::Any))
        ));

        let InteractionInput::Steps(steps) = result.input else {
            panic!("expected steps input");
        };
        assert_eq!(steps.len(), 1);
        let Step::UserInput { content: contents } = &steps[0] else {
            panic!("expected user input step");
        };
        assert_eq!(contents.len(), 1);
        match &contents[0] {
            Content::Text(TextContent { text, .. }) => assert_eq!(text, "Hello"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    /// `functionResponse.name` is the executed function's name: read from
    /// the required `ToolResult::name` — never an identifier.
    #[test]
    fn tool_result_serializes_the_executed_name_not_an_identifier() {
        use message::{AssistantContent, ToolCall, ToolFunction, ToolResultContent};

        let call = |item_id: Option<&str>, call_id: &str, name: &str| {
            let function = ToolFunction {
                name: name.to_owned(),
                arguments: json!({}),
            };
            let tool_call = match item_id {
                Some(item_id) => ToolCall::from_dual_wire(item_id, call_id, function),
                None => ToolCall::from_wire(call_id, function),
            };
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::ToolCall(tool_call)],
            }
        };
        let result = |item_id: Option<&str>, call_id: &str, name: &str| Message::User {
            content: vec![match item_id {
                Some(item_id) => message::UserContent::tool_result_with_call_id(
                    item_id,
                    call_id,
                    name,
                    vec![ToolResultContent::text("out")],
                ),
                None => message::UserContent::tool_result_from_wire(
                    call_id,
                    name,
                    vec![ToolResultContent::text("out")],
                ),
            }],
        };

        let request = CompletionRequest {
            record_telemetry_content: false,
            model: None,
            preamble: None,
            chat_history: vec![
                // A driver-built result carries the executed name (a repair
                // hook renamed the call: `sum` ran, not `add`).
                call(None, "call_1", "sum"),
                result(None, "call_1", "sum"),
                // An OpenAI-shaped correlator travels as the call id while
                // the required `name` field carries the executed name —
                // `call_abc` must never reach the wire as a name.
                call(None, "call_abc", "get_weather"),
                result(None, "call_abc", "get_weather"),
                // A dual-identifier result (OpenAI Responses: item id `fc_…`
                // + `call_id` `call_…`) keeps the correlator on the wire and
                // the executed name in `name` — `fc_1` must never reach the
                // wire as a name.
                call(Some("fc_1"), "call_9", "get_time"),
                result(Some("fc_1"), "call_9", "get_time"),
            ],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let body = create_request_body("gemini-2.5-flash".to_string(), request, None)
            .expect("request should build");
        let input = serde_json::to_value(&body.input).expect("input should serialize");
        let mut names = Vec::new();
        let mut call_ids = Vec::new();
        fn collect(value: &serde_json::Value, names: &mut Vec<String>, call_ids: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(map) => {
                    if map.get("type").and_then(|t| t.as_str()) == Some("function_result") {
                        if let Some(name) = map.get("name").and_then(|n| n.as_str()) {
                            names.push(name.to_owned());
                        }
                        if let Some(call_id) = map.get("call_id").and_then(|c| c.as_str()) {
                            call_ids.push(call_id.to_owned());
                        }
                    }
                    for nested in map.values() {
                        collect(nested, names, call_ids);
                    }
                }
                serde_json::Value::Array(items) => {
                    for nested in items {
                        collect(nested, names, call_ids);
                    }
                }
                _ => {}
            }
        }
        collect(&input, &mut names, &mut call_ids);

        assert_eq!(
            names,
            vec![
                "sum".to_owned(),
                "get_weather".to_owned(),
                "get_time".to_owned()
            ]
        );
        assert_eq!(
            call_ids,
            vec![
                "call_1".to_owned(),
                "call_abc".to_owned(),
                "call_9".to_owned()
            ]
        );
    }

    #[test]
    fn test_tool_result_without_provider_id_sends_minted_call_id() {
        // A call id is always available now: the wire gets the
        // provider-issued id when one exists, else rig's minted handle —
        // the old "Tool results require call_id" error is unrepresentable.
        let call = message::ToolCallId::mint();
        let content = message::UserContent::ToolResult(message::ToolResult {
            call: call.clone(),
            provider: None,
            name: "get_weather".to_string(),
            content: vec![message::ToolResultContent::text("ok")],
        });

        let converted = Content::try_from(content).expect("tool result should convert");
        let Content::FunctionResult(result) = converted else {
            panic!("expected function result");
        };
        assert_eq!(result.call_id.as_deref(), Some(call.as_str()));
        assert_eq!(result.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn test_tool_result_preserves_text_and_json_types() {
        let content = message::UserContent::ToolResult(message::ToolResult {
            call: message::ToolCallId::new_or_mint("call-123"),
            provider: message::ProviderCallId::new("call-123"),
            name: "get_weather".to_string(),
            content: vec![
                message::ToolResultContent::text(r#"{"status":"literal"}"#),
                message::ToolResultContent::json(json!({ "status": "structured" })),
            ],
        });

        let converted = Content::try_from(content).expect("tool result should convert");
        let Content::FunctionResult(result) = converted else {
            panic!("expected function result");
        };
        let expected_result = json!([
            {
                "type": "text",
                "text": "{\"status\":\"literal\"}"
            },
            {
                "type": "text",
                "text": "{\"status\":\"structured\"}"
            }
        ]);
        assert_eq!(result.result, Some(expected_result.clone()));
        assert_eq!(
            serde_json::to_value(Content::FunctionResult(result))
                .expect("function result should serialize"),
            json!({
                "type": "function_result",
                "name": "get_weather",
                "result": expected_result,
                "call_id": "call-123"
            })
        );
    }

    #[test]
    fn test_tool_result_text_and_json_singletons_remain_scalar() {
        let cases = [
            (
                message::ToolResultContent::text(r#"{"status":"literal"}"#),
                json!("{\"status\":\"literal\"}"),
            ),
            (
                message::ToolResultContent::json(json!({ "status": "structured" })),
                json!({ "status": "structured" }),
            ),
            (
                message::ToolResultContent::json(json!("structured string")),
                json!("structured string"),
            ),
        ];

        for (tool_content, expected) in cases {
            let content = message::UserContent::ToolResult(message::ToolResult {
                call: message::ToolCallId::new_or_mint("call-123"),
                provider: message::ProviderCallId::new("call-123"),
                name: "get_weather".to_string(),
                content: vec![tool_content],
            });

            let Content::FunctionResult(result) =
                Content::try_from(content).expect("tool result should convert")
            else {
                panic!("expected function result");
            };
            assert_eq!(result.result, Some(expected));
        }
    }

    #[test]
    fn test_tool_result_rich_singletons_use_tagged_content() {
        let cases = [
            (
                message::ToolResultContent::json(json!(["sunny", 72])),
                json!([{
                    "type": "text",
                    "text": "[\"sunny\",72]"
                }]),
            ),
            (
                message::ToolResultContent::image_base64(
                    "image-data",
                    Some(message::ImageMediaType::PNG),
                    None,
                ),
                json!([{
                    "type": "image",
                    "data": "image-data",
                    "mime_type": "image/png"
                }]),
            ),
        ];

        for (tool_content, expected) in cases {
            let content = message::UserContent::ToolResult(message::ToolResult {
                call: message::ToolCallId::new_or_mint("call-123"),
                provider: message::ProviderCallId::new("call-123"),
                name: "get_weather".to_string(),
                content: vec![tool_content],
            });

            let Content::FunctionResult(result) =
                Content::try_from(content).expect("tool result should convert")
            else {
                panic!("expected function result");
            };
            assert_eq!(result.result, Some(expected));
        }
    }

    #[test]
    fn test_tool_result_images_and_text_serialize_as_ordered_tagged_content() {
        let tool_result = message::UserContent::ToolResult(message::ToolResult {
            call: message::ToolCallId::new_or_mint("call-image"),
            provider: message::ProviderCallId::new("call-image"),
            name: "render".to_string(),
            content: vec![
                message::ToolResultContent::image_base64(
                    "first-image",
                    Some(message::ImageMediaType::PNG),
                    None,
                ),
                message::ToolResultContent::text("between-images"),
                message::ToolResultContent::Image(message::Image {
                    data: message::DocumentSourceKind::Url(
                        "https://example.com/second.jpg".to_string(),
                    ),
                    media_type: Some(message::ImageMediaType::JPEG),
                    detail: None,
                    additional_params: None,
                }),
            ],
        });
        let request = CompletionRequest {
            record_telemetry_content: false,
            model: None,
            preamble: None,
            chat_history: vec![Message::User {
                content: vec![tool_result],
            }],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let request = create_request_body("gemini-2.5-flash".to_string(), request, None)
            .expect("request should build");
        let serialized = serde_json::to_value(request).expect("request should serialize");

        assert_eq!(
            serialized.pointer("/input/0/content/0"),
            Some(&json!({
                "type": "function_result",
                "name": "render",
                "result": [
                    {
                        "type": "image",
                        "data": "first-image",
                        "mime_type": "image/png"
                    },
                    {
                        "type": "text",
                        "text": "between-images"
                    },
                    {
                        "type": "image",
                        "uri": "https://example.com/second.jpg",
                        "mime_type": "image/jpeg"
                    }
                ],
                "call_id": "call-image"
            }))
        );
    }

    #[test]
    fn test_response_function_call_mapping() {
        let interaction = Interaction {
            id: "interaction-1".to_string(),
            steps: vec![Step::FunctionCall(FunctionCallContent {
                name: Some("get_weather".to_string()),
                arguments: Some(json!({"location": "Paris"})),
                id: Some("call-123".to_string()),
            })],
            usage: Some(InteractionUsage {
                total_input_tokens: Some(5),
                total_output_tokens: Some(7),
                total_tokens: Some(12),
            }),
            ..Default::default()
        };

        let response: completion::CompletionResponse =
            interaction.try_into().expect("conversion should succeed");

        let choice = response.choice.first();
        match choice {
            Some(completion::AssistantContent::ToolCall(tool_call)) => {
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.id, "call-123");
                assert_eq!(
                    tool_call.provider.as_ref().expect("wire id").call_id,
                    "call-123"
                );
            }
            other => panic!("unexpected content: {other:?}"),
        }

        assert_eq!(response.usage.input_tokens, 5);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.usage.total_tokens, 12);
    }

    #[test]
    fn test_google_search_tool_serialization() {
        let tool = Tool::GoogleSearch;
        let value = serde_json::to_value(tool).expect("tool should serialize");
        assert_eq!(value, json!({ "type": "google_search" }));
    }

    #[test]
    fn test_url_context_tool_serialization() {
        let tool = Tool::UrlContext;
        let value = serde_json::to_value(tool).expect("tool should serialize");
        assert_eq!(value, json!({ "type": "url_context" }));
    }

    #[test]
    fn test_code_execution_tool_serialization() {
        let tool = Tool::CodeExecution;
        let value = serde_json::to_value(tool).expect("tool should serialize");
        assert_eq!(value, json!({ "type": "code_execution" }));
    }

    #[test]
    fn test_google_search_helpers() {
        let interaction = Interaction {
            steps: vec![
                Step::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-one".to_string(), "query-two".to_string()]),
                    }),
                    id: Some("call-1".to_string()),
                }),
                Step::GoogleSearchResult(GoogleSearchResultContent {
                    result: Some(vec![GoogleSearchResult {
                        url: Some("https://example.com".to_string()),
                        title: Some("Example One".to_string()),
                        rendered_content: None,
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: Some("call-1".to_string()),
                }),
                Step::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-three".to_string()]),
                    }),
                    id: Some("call-2".to_string()),
                }),
                Step::GoogleSearchResult(GoogleSearchResultContent {
                    result: Some(vec![GoogleSearchResult {
                        url: Some("https://example.org".to_string()),
                        title: Some("Example Two".to_string()),
                        rendered_content: None,
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: Some("call-2".to_string()),
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.google_search_exchanges();
        assert_eq!(exchanges.len(), 2);
        assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(
            exchanges[0].queries(),
            vec!["query-one".to_string(), "query-two".to_string()]
        );
        let exchange_results = exchanges[0].result_items();
        assert_eq!(exchange_results.len(), 1);
        assert_eq!(exchange_results[0].title.as_deref(), Some("Example One"));

        assert_eq!(exchanges[1].call_id.as_deref(), Some("call-2"));
        assert_eq!(exchanges[1].queries(), vec!["query-three".to_string()]);
        let exchange_results = exchanges[1].result_items();
        assert_eq!(exchange_results.len(), 1);
        assert_eq!(exchange_results[0].title.as_deref(), Some("Example Two"));

        let queries = interaction.google_search_queries();
        assert_eq!(queries, vec!["query-one", "query-two", "query-three"]);

        let results = interaction.google_search_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title.as_deref(), Some("Example One"));
        assert_eq!(results[1].title.as_deref(), Some("Example Two"));

        let call_contents = interaction.google_search_call_contents();
        assert_eq!(call_contents.len(), 2);
        assert_eq!(call_contents[0].id.as_deref(), Some("call-1"));
        assert_eq!(call_contents[1].id.as_deref(), Some("call-2"));

        let result_contents = interaction.google_search_result_contents();
        assert_eq!(result_contents.len(), 2);
        assert_eq!(result_contents[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(result_contents[1].call_id.as_deref(), Some("call-2"));
    }

    #[test]
    fn test_google_search_helpers_without_call_id() {
        let interaction = Interaction {
            steps: vec![
                Step::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-one".to_string()]),
                    }),
                    id: None,
                }),
                Step::GoogleSearchResult(GoogleSearchResultContent {
                    result: Some(vec![GoogleSearchResult {
                        url: Some("https://example.com".to_string()),
                        title: Some("Example One".to_string()),
                        rendered_content: None,
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
                Step::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-two".to_string()]),
                    }),
                    id: Some("call-2".to_string()),
                }),
                Step::GoogleSearchResult(GoogleSearchResultContent {
                    result: Some(vec![GoogleSearchResult {
                        url: Some("https://example.org".to_string()),
                        title: Some("Example Two".to_string()),
                        rendered_content: None,
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.google_search_exchanges();
        assert_eq!(exchanges.len(), 2);

        let no_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.is_none())
            .expect("expected no-id exchange");
        assert_eq!(no_id.calls.len(), 1);
        assert_eq!(no_id.results.len(), 1);

        let with_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
            .expect("expected call-2 exchange");
        assert_eq!(with_id.calls.len(), 1);
        assert_eq!(with_id.results.len(), 1);
    }

    #[test]
    fn test_url_context_helpers() {
        let interaction = Interaction {
            steps: vec![
                Step::UrlContextCall(UrlContextCallContent {
                    arguments: Some(UrlContextCallArguments {
                        urls: Some(vec![
                            "https://example.com".to_string(),
                            "https://example.org".to_string(),
                        ]),
                    }),
                    id: Some("call-1".to_string()),
                }),
                Step::UrlContextResult(UrlContextResultContent {
                    result: Some(vec![UrlContextResult {
                        url: Some("https://example.com".to_string()),
                        status: Some("success".to_string()),
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: Some("call-1".to_string()),
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.url_context_exchanges();
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(
            exchanges[0].urls(),
            vec!["https://example.com", "https://example.org"]
        );
        let results = exchanges[0].result_items();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status.as_deref(), Some("success"));

        let urls = interaction.url_context_urls();
        assert_eq!(urls, vec!["https://example.com", "https://example.org"]);

        let results = interaction.url_context_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url.as_deref(), Some("https://example.com"));

        let call_contents = interaction.url_context_call_contents();
        assert_eq!(call_contents.len(), 1);
        assert_eq!(call_contents[0].id.as_deref(), Some("call-1"));

        let result_contents = interaction.url_context_result_contents();
        assert_eq!(result_contents.len(), 1);
        assert_eq!(result_contents[0].call_id.as_deref(), Some("call-1"));
    }

    #[test]
    fn test_url_context_helpers_without_call_id() {
        let interaction = Interaction {
            steps: vec![
                Step::UrlContextCall(UrlContextCallContent {
                    arguments: Some(UrlContextCallArguments {
                        urls: Some(vec!["https://example.com".to_string()]),
                    }),
                    id: None,
                }),
                Step::UrlContextResult(UrlContextResultContent {
                    result: Some(vec![UrlContextResult {
                        url: Some("https://example.com".to_string()),
                        status: Some("success".to_string()),
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
                Step::UrlContextCall(UrlContextCallContent {
                    arguments: Some(UrlContextCallArguments {
                        urls: Some(vec!["https://example.org".to_string()]),
                    }),
                    id: Some("call-2".to_string()),
                }),
                Step::UrlContextResult(UrlContextResultContent {
                    result: Some(vec![UrlContextResult {
                        url: Some("https://example.org".to_string()),
                        status: Some("success".to_string()),
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.url_context_exchanges();
        assert_eq!(exchanges.len(), 2);

        let no_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.is_none())
            .expect("expected no-id exchange");
        assert_eq!(no_id.calls.len(), 1);
        assert_eq!(no_id.results.len(), 1);

        let with_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
            .expect("expected call-2 exchange");
        assert_eq!(with_id.calls.len(), 1);
        assert_eq!(with_id.results.len(), 1);
    }

    #[test]
    fn test_code_execution_helpers() {
        let interaction = Interaction {
            steps: vec![
                Step::CodeExecutionCall(CodeExecutionCallContent {
                    arguments: Some(CodeExecutionCallArguments {
                        language: Some("python".to_string()),
                        code: Some("print(2 + 2)".to_string()),
                    }),
                    id: Some("call-1".to_string()),
                }),
                Step::CodeExecutionResult(CodeExecutionResultContent {
                    result: Some("4\n".to_string()),
                    signature: None,
                    is_error: None,
                    call_id: Some("call-1".to_string()),
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.code_execution_exchanges();
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(exchanges[0].code_snippets(), vec!["print(2 + 2)"]);
        assert_eq!(exchanges[0].outputs(), vec!["4\n"]);

        let calls = interaction.code_execution_call_contents();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call-1"));

        let results = interaction.code_execution_result_contents();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].call_id.as_deref(), Some("call-1"));

        let snippets = interaction.code_execution_snippets();
        assert_eq!(snippets, vec!["print(2 + 2)"]);

        let outputs = interaction.code_execution_outputs();
        assert_eq!(outputs, vec!["4\n"]);
    }

    #[test]
    fn test_code_execution_helpers_without_call_id() {
        let interaction = Interaction {
            steps: vec![
                Step::CodeExecutionCall(CodeExecutionCallContent {
                    arguments: Some(CodeExecutionCallArguments {
                        language: Some("python".to_string()),
                        code: Some("print(1 + 1)".to_string()),
                    }),
                    id: None,
                }),
                Step::CodeExecutionResult(CodeExecutionResultContent {
                    result: Some("2\n".to_string()),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
                Step::CodeExecutionCall(CodeExecutionCallContent {
                    arguments: Some(CodeExecutionCallArguments {
                        language: Some("python".to_string()),
                        code: Some("print(2 + 2)".to_string()),
                    }),
                    id: Some("call-2".to_string()),
                }),
                Step::CodeExecutionResult(CodeExecutionResultContent {
                    result: Some("4\n".to_string()),
                    signature: None,
                    is_error: None,
                    call_id: None,
                }),
            ],
            ..Default::default()
        };

        let exchanges = interaction.code_execution_exchanges();
        assert_eq!(exchanges.len(), 2);

        let no_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.is_none())
            .expect("expected no-id exchange");
        assert_eq!(no_id.calls.len(), 1);
        assert_eq!(no_id.results.len(), 1);

        let with_id = exchanges
            .iter()
            .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
            .expect("expected call-2 exchange");
        assert_eq!(with_id.calls.len(), 1);
        assert_eq!(with_id.results.len(), 1);
    }

    #[test]
    fn test_interaction_status_helpers() {
        let mut interaction = Interaction {
            status: Some(InteractionStatus::InProgress),
            ..Default::default()
        };
        assert!(!interaction.is_terminal());
        assert!(!interaction.is_completed());

        // RequiresAction is terminal for the poll (it never advances without
        // the caller submitting tool results) but is not a completion.
        interaction.status = Some(InteractionStatus::RequiresAction);
        assert!(interaction.is_terminal());
        assert!(!interaction.is_completed());

        interaction.status = Some(InteractionStatus::Completed);
        assert!(interaction.is_terminal());
        assert!(interaction.is_completed());

        interaction.status = Some(InteractionStatus::Failed);
        assert!(interaction.is_terminal());
        assert!(!interaction.is_completed());

        interaction.status = Some(InteractionStatus::BudgetExceeded);
        assert!(interaction.is_terminal());
        assert!(!interaction.is_completed());
    }

    #[test]
    fn test_interaction_status_maps_every_wire_variant() {
        use crate::completion::FinishReason as Normalized;

        for (status, expected) in [
            (InteractionStatus::Completed, Normalized::Stop),
            (InteractionStatus::RequiresAction, Normalized::ToolCalls),
            (InteractionStatus::BudgetExceeded, Normalized::Length),
            // Statuses rig does not model survive in the provider's own
            // spelling rather than being guessed at.
            (
                InteractionStatus::InProgress,
                Normalized::Other("in_progress".to_string()),
            ),
            (
                InteractionStatus::Incomplete,
                Normalized::Other("incomplete".to_string()),
            ),
            (
                InteractionStatus::Failed,
                Normalized::Other("failed".to_string()),
            ),
            (
                InteractionStatus::Cancelled,
                Normalized::Other("cancelled".to_string()),
            ),
        ] {
            assert_eq!(
                map_interaction_status(&status),
                expected,
                "status {status:?}"
            );
        }
    }

    #[test]
    fn test_interaction_status_wire_spelling_matches_serde() {
        // `as_wire_str` is hand-written; keep it honest against the serde
        // representation the same enum deserializes from.
        for status in [
            InteractionStatus::InProgress,
            InteractionStatus::RequiresAction,
            InteractionStatus::Incomplete,
            InteractionStatus::BudgetExceeded,
            InteractionStatus::Completed,
            InteractionStatus::Failed,
            InteractionStatus::Cancelled,
        ] {
            let serialized = serde_json::to_value(&status).expect("status should serialize");
            assert_eq!(serialized, json!(status.as_wire_str()));
        }
    }

    #[test]
    fn test_unknown_interaction_status_round_trips_verbatim() {
        // A status this crate does not know must land in `Unknown` with the
        // provider's spelling intact — and serialize back to the same string —
        // rather than failing the whole payload.
        let status: InteractionStatus = serde_json::from_value(json!("status_future"))
            .expect("unknown status should deserialize");
        assert!(matches!(&status, InteractionStatus::Unknown(s) if s == "status_future"));
        assert_eq!(status.as_wire_str(), "status_future");
        assert_eq!(
            serde_json::to_value(&status).expect("status should serialize"),
            json!("status_future")
        );
        assert_eq!(
            map_interaction_status(&status),
            crate::completion::FinishReason::Other("status_future".to_string())
        );
    }

    #[test]
    fn test_interaction_with_unknown_status_stays_parseable() {
        // A status Google ships tomorrow must not fail the interaction
        // payload; the unknown status is conservatively *terminal* — only the
        // known in-flight statuses keep a poll loop waiting, so a future
        // status surfaces to the caller instead of hanging it.
        let interaction: Interaction = serde_json::from_value(json!({
            "id": "int-future",
            "status": "status_future",
            "usage": {"total_tokens": 5}
        }))
        .expect("unknown status should not fail the payload");

        assert_eq!(interaction.id, "int-future");
        assert!(matches!(
            interaction.status,
            Some(InteractionStatus::Unknown(ref s)) if s == "status_future"
        ));
        assert!(interaction.is_terminal());
        assert!(!interaction.is_completed());
        assert_eq!(
            interaction.usage.as_ref().and_then(|u| u.total_tokens),
            Some(5)
        );
    }

    #[test]
    fn test_completion_response_carries_normalized_metadata() {
        let interaction = Interaction {
            id: "interaction-meta".to_string(),
            model: Some("gemini-2.5-pro".to_string()),
            status: Some(InteractionStatus::BudgetExceeded),
            steps: vec![Step::ModelOutput {
                content: vec![Content::Text(TextContent {
                    text: "partial answer".to_string(),
                    annotations: None,
                })],
            }],
            ..Default::default()
        };

        let response: completion::CompletionResponse =
            interaction.try_into().expect("conversion should succeed");

        assert_eq!(response.provider, PROVIDER_NAME);
        assert_eq!(response.model.as_deref(), Some("gemini-2.5-pro"));
        assert_eq!(response.response_id.as_deref(), Some("interaction-meta"));
        assert_eq!(response.message_id, None);
        assert_eq!(
            response.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
    }

    #[test]
    fn test_completion_response_upgrades_completed_to_tool_calls() {
        // A `completed` interaction whose outputs are function calls is a tool
        // turn; the normalized response must say so.
        let interaction = Interaction {
            id: "interaction-tool".to_string(),
            status: Some(InteractionStatus::Completed),
            steps: vec![Step::FunctionCall(FunctionCallContent {
                name: Some("get_weather".to_string()),
                arguments: Some(json!({"location": "Paris"})),
                id: Some("call-123".to_string()),
            })],
            ..Default::default()
        };

        let response: completion::CompletionResponse =
            interaction.try_into().expect("conversion should succeed");

        assert_eq!(
            response.finish_reason(),
            Some(crate::completion::FinishReason::ToolCalls)
        );
        assert_eq!(response.model, None);
    }

    #[test]
    fn test_budget_exceeded_status_deserializes() {
        let status: InteractionStatus = serde_json::from_value(json!("budget_exceeded"))
            .expect("budget_exceeded should deserialize");

        assert!(matches!(status, InteractionStatus::BudgetExceeded));
        assert!(status.is_terminal());
    }

    #[test]
    fn test_budget_exceeded_status_update_deserializes() {
        let event: InteractionSseEvent = serde_json::from_value(json!({
            "event_type": "interaction.status_update",
            "interaction_id": "interaction-123",
            "status": "budget_exceeded",
            "event_id": "event-456"
        }))
        .expect("budget_exceeded status update should deserialize");

        match event {
            InteractionSseEvent::InteractionStatusUpdate {
                interaction_id,
                status,
                event_id,
            } => {
                assert_eq!(interaction_id, "interaction-123");
                assert!(matches!(status, InteractionStatus::BudgetExceeded));
                assert!(status.is_terminal());
                assert_eq!(event_id.as_deref(), Some("event-456"));
            }
            other => panic!("expected status update event, got {other:?}"),
        }
    }

    #[test]
    fn test_build_interaction_stream_path() {
        let path = build_interaction_stream_path("interaction-123", None);
        assert_eq!(path, "/v1beta/interactions/interaction-123?stream=true");

        let path = build_interaction_stream_path("interaction-123", Some("event-456"));
        assert_eq!(
            path,
            "/v1beta/interactions/interaction-123?stream=true&last_event_id=event-456"
        );
    }

    #[test]
    fn test_inline_citations_from_annotations() {
        let text_content = TextContent {
            text: "Hello world".to_string(),
            annotations: Some(vec![
                Annotation {
                    start_index: Some(6),
                    end_index: Some(11),
                    source: Some("https://example.com".to_string()),
                },
                Annotation {
                    start_index: Some(0),
                    end_index: Some(5),
                    source: Some("https://hello.example".to_string()),
                },
            ]),
        };

        let cited = text_content.with_inline_citations();
        assert_eq!(
            cited,
            "Hello[1](https://hello.example) world[2](https://example.com)"
        );

        let interaction = Interaction {
            steps: vec![Step::ModelOutput {
                content: vec![Content::Text(text_content)],
            }],
            ..Default::default()
        };

        let cited_text = interaction.text_with_inline_citations();
        assert_eq!(
            cited_text.as_deref(),
            Some("Hello[1](https://hello.example) world[2](https://example.com)")
        );
    }
}
