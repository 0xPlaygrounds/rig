//! Google Gemini Interactions API integration.
//! From <https://ai.google.dev/api/interactions-api>

use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::message::{self, MimeType, Reasoning};
use crate::telemetry::SpanCombinator;
use serde_json::{Map, Value};
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;
use url::form_urlencoded;

use super::client::InteractionsClient;

/// Streaming helpers for the Interactions API.
pub mod streaming;
pub use interactions_api_types::*;

// =================================================================
// Rig Implementation Types
// =================================================================

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

impl<T> completion::CompletionModel for InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    type Response = Interaction;
    type StreamingResponse = streaming::StreamingCompletionResponse;
    type Client = InteractionsClient<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Interaction>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "interactions",
                gen_ai.operation.name = "interactions",
                gen_ai.provider.name = "gcp.gemini",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = self.create_completion_request(completion_request, Some(false))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Gemini interactions completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let request = self
            .client
            .post("/v1beta/interactions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send::<_, Vec<u8>>(request).await?;

            if response.status().is_success() {
                let response_body = response
                    .into_body()
                    .await
                    .map_err(CompletionError::HttpError)?;

                let response_text = String::from_utf8_lossy(&response_body).to_string();

                let response: Interaction =
                    serde_json::from_slice(&response_body).map_err(|err| {
                        tracing::error!(
                            error = %err,
                            body = %response_text,
                            "Failed to deserialize Gemini interactions response"
                        );
                        CompletionError::JsonError(err)
                    })?;

                let span = tracing::Span::current();
                span.record_response_metadata(&response);
                span.record_token_usage(&response);

                if enabled!(Level::TRACE) {
                    tracing::trace!(
                        target: "rig::completions",
                        "Gemini interactions completion response: {}",
                        serde_json::to_string_pretty(&response)?
                    );
                }

                response.try_into()
            } else {
                let text = String::from_utf8_lossy(
                    &response
                        .into_body()
                        .await
                        .map_err(CompletionError::HttpError)?,
                )
                .into();

                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        InteractionsCompletionModel::stream(self, request).await
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
    let mut history = Vec::new();
    if let Some(docs) = completion_request.normalized_documents() {
        history.push(docs);
    }
    history.extend(completion_request.chat_history);

    let turns = history
        .into_iter()
        .map(Turn::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| CompletionError::RequestError(Box::new(err)))?;

    let input = InteractionInput::Turns(turns);

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
        let text = String::from_utf8_lossy(
            &response
                .into_body()
                .await
                .map_err(CompletionError::HttpError)?,
        )
        .into();

        Err(CompletionError::ProviderError(text))
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

impl TryFrom<Interaction> for completion::CompletionResponse<Interaction> {
    type Error = CompletionError;

    fn try_from(response: Interaction) -> Result<Self, Self::Error> {
        if response.outputs.is_empty() {
            let status = response.status.as_ref().map(|status| format!("{status:?}"));
            let message = match status {
                Some(status) => format!(
                    "Interaction contained no outputs (status: {status}). Use get_interaction for background tasks."
                ),
                None => "Interaction contained no outputs".to_string(),
            };
            return Err(CompletionError::ResponseError(message));
        }

        let content = response
            .outputs
            .iter()
            .cloned()
            .filter_map(|output| match assistant_content_from_output(output) {
                Ok(Some(content)) => Some(Ok(content)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .and_then(|usage| usage.token_usage())
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
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
            let call_id = id.unwrap_or_else(|| name.clone());
            Ok(Some(completion::AssistantContent::tool_call_with_call_id(
                name.clone(),
                call_id,
                name,
                arguments.unwrap_or(Value::Object(Map::new())),
            )))
        }
        Content::Thought(ThoughtContent {
            summary, signature, ..
        }) => {
            let summary = summary
                .unwrap_or_default()
                .into_iter()
                .filter_map(|content| match content {
                    ThoughtSummaryContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect::<Vec<_>>();

            if summary.is_empty() {
                return Ok(None);
            }

            Ok(Some(completion::AssistantContent::Reasoning(
                Reasoning::multi(summary).with_signature(signature),
            )))
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

fn split_data_uri(
    src: message::DocumentSourceKind,
) -> Result<(Option<String>, Option<String>), message::MessageError> {
    match src {
        message::DocumentSourceKind::Url(uri) => Ok((None, Some(uri))),
        message::DocumentSourceKind::Base64(data) | message::DocumentSourceKind::String(data) => {
            Ok((Some(data), None))
        }
        message::DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
            "Raw content is not supported, encode as base64 first".to_string(),
        )),
        message::DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
            "Unknown content source".to_string(),
        )),
    }
}

/// Raw request/response types and convenience helpers for the Gemini Interactions API.
pub mod interactions_api_types {
    use super::split_data_uri;
    use crate::completion::{CompletionError, GetTokenUsage, Usage};
    use crate::message::{self, MimeType};
    use crate::telemetry::ProviderResponseExt;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

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
        pub outputs: Vec<Content>,
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

    impl GetTokenUsage for Interaction {
        fn token_usage(&self) -> Option<Usage> {
            self.usage.as_ref().and_then(|usage| usage.token_usage())
        }
    }

    impl ProviderResponseExt for Interaction {
        type OutputMessage = Content;
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

        fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
            self.outputs.clone()
        }

        fn get_text_response(&self) -> Option<String> {
            let text = self
                .outputs
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

    /// Groups Google Search tool calls and results for a single interaction.
    #[derive(Clone, Debug, Default)]
    pub struct GoogleSearchExchange {
        /// Call identifier used to match calls to results.
        pub call_id: Option<String>,
        /// One or more Google Search tool calls.
        pub calls: Vec<GoogleSearchCallContent>,
        /// One or more Google Search tool results.
        pub results: Vec<GoogleSearchResultContent>,
    }

    impl GoogleSearchExchange {
        /// Collects all queries from the stored Google Search tool calls.
        pub fn queries(&self) -> Vec<String> {
            let mut queries = Vec::new();
            for call in &self.calls {
                if let Some(args) = &call.arguments {
                    if let Some(call_queries) = &args.queries {
                        queries.extend(call_queries.clone());
                    }
                }
            }
            queries
        }

        /// Collects all Google Search result entries from tool results.
        pub fn result_items(&self) -> Vec<GoogleSearchResult> {
            let mut items = Vec::new();
            for result in &self.results {
                if let Some(entries) = &result.result {
                    items.extend(entries.clone());
                }
            }
            items
        }
    }

    /// Groups URL context tool calls and results for a single interaction.
    #[derive(Clone, Debug, Default)]
    pub struct UrlContextExchange {
        /// Call identifier used to match calls to results.
        pub call_id: Option<String>,
        /// One or more URL context tool calls.
        pub calls: Vec<UrlContextCallContent>,
        /// One or more URL context tool results.
        pub results: Vec<UrlContextResultContent>,
    }

    impl UrlContextExchange {
        /// Collects all URLs from the stored URL context tool calls.
        pub fn urls(&self) -> Vec<String> {
            let mut urls = Vec::new();
            for call in &self.calls {
                if let Some(args) = &call.arguments {
                    if let Some(call_urls) = &args.urls {
                        urls.extend(call_urls.clone());
                    }
                }
            }
            urls
        }

        /// Collects all URL context result entries from tool results.
        pub fn result_items(&self) -> Vec<UrlContextResult> {
            let mut items = Vec::new();
            for result in &self.results {
                if let Some(entries) = &result.result {
                    items.extend(entries.clone());
                }
            }
            items
        }
    }

    impl Interaction {
        /// Groups Google Search tool calls and results by call_id.
        pub fn google_search_exchanges(&self) -> Vec<GoogleSearchExchange> {
            let mut exchanges: Vec<GoogleSearchExchange> = Vec::new();

            for content in &self.outputs {
                match content {
                    Content::GoogleSearchCall(call) => {
                        if let Some(call_id) = call.id.as_ref() {
                            if let Some(index) = exchanges
                                .iter()
                                .position(|exchange| exchange.call_id.as_deref() == Some(call_id))
                            {
                                exchanges[index].calls.push(call.clone());
                            } else {
                                exchanges.push(GoogleSearchExchange {
                                    call_id: Some(call_id.clone()),
                                    calls: vec![call.clone()],
                                    results: Vec::new(),
                                });
                            }
                        } else {
                            exchanges.push(GoogleSearchExchange {
                                call_id: None,
                                calls: vec![call.clone()],
                                results: Vec::new(),
                            });
                        }
                    }
                    Content::GoogleSearchResult(result) => {
                        if let Some(call_id) = result.call_id.as_ref() {
                            if let Some(index) = exchanges
                                .iter()
                                .position(|exchange| exchange.call_id.as_deref() == Some(call_id))
                            {
                                exchanges[index].results.push(result.clone());
                            } else {
                                exchanges.push(GoogleSearchExchange {
                                    call_id: Some(call_id.clone()),
                                    calls: Vec::new(),
                                    results: vec![result.clone()],
                                });
                            }
                        } else {
                            exchanges.push(GoogleSearchExchange {
                                call_id: None,
                                calls: Vec::new(),
                                results: vec![result.clone()],
                            });
                        }
                    }
                    _ => {}
                }
            }

            exchanges
        }

        /// Collects Google Search tool call contents from the interaction outputs.
        pub fn google_search_call_contents(&self) -> Vec<GoogleSearchCallContent> {
            self.google_search_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.calls)
                .collect()
        }

        /// Collects Google Search result contents from the interaction outputs.
        pub fn google_search_result_contents(&self) -> Vec<GoogleSearchResultContent> {
            self.google_search_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.results)
                .collect()
        }

        /// Collects all Google Search queries from tool calls in the outputs.
        pub fn google_search_queries(&self) -> Vec<String> {
            self.google_search_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.queries())
                .collect()
        }

        /// Collects all Google Search result entries from tool results in the outputs.
        pub fn google_search_results(&self) -> Vec<GoogleSearchResult> {
            self.google_search_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.result_items())
                .collect()
        }

        /// Groups URL context tool calls and results by call_id.
        pub fn url_context_exchanges(&self) -> Vec<UrlContextExchange> {
            let mut exchanges: Vec<UrlContextExchange> = Vec::new();

            for content in &self.outputs {
                match content {
                    Content::UrlContextCall(call) => {
                        if let Some(call_id) = call.id.as_ref() {
                            if let Some(index) = exchanges
                                .iter()
                                .position(|exchange| exchange.call_id.as_deref() == Some(call_id))
                            {
                                exchanges[index].calls.push(call.clone());
                            } else {
                                exchanges.push(UrlContextExchange {
                                    call_id: Some(call_id.clone()),
                                    calls: vec![call.clone()],
                                    results: Vec::new(),
                                });
                            }
                        } else {
                            exchanges.push(UrlContextExchange {
                                call_id: None,
                                calls: vec![call.clone()],
                                results: Vec::new(),
                            });
                        }
                    }
                    Content::UrlContextResult(result) => {
                        if let Some(call_id) = result.call_id.as_ref() {
                            if let Some(index) = exchanges
                                .iter()
                                .position(|exchange| exchange.call_id.as_deref() == Some(call_id))
                            {
                                exchanges[index].results.push(result.clone());
                            } else {
                                exchanges.push(UrlContextExchange {
                                    call_id: Some(call_id.clone()),
                                    calls: Vec::new(),
                                    results: vec![result.clone()],
                                });
                            }
                        } else {
                            exchanges.push(UrlContextExchange {
                                call_id: None,
                                calls: Vec::new(),
                                results: vec![result.clone()],
                            });
                        }
                    }
                    _ => {}
                }
            }

            exchanges
        }

        /// Collects URL context tool call contents from the interaction outputs.
        pub fn url_context_call_contents(&self) -> Vec<UrlContextCallContent> {
            self.url_context_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.calls)
                .collect()
        }

        /// Collects URL context result contents from the interaction outputs.
        pub fn url_context_result_contents(&self) -> Vec<UrlContextResultContent> {
            self.url_context_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.results)
                .collect()
        }

        /// Collects all URLs from URL context tool calls in the outputs.
        pub fn url_context_urls(&self) -> Vec<String> {
            self.url_context_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.urls())
                .collect()
        }

        /// Collects all URL context result entries from tool results in the outputs.
        pub fn url_context_results(&self) -> Vec<UrlContextResult> {
            self.url_context_exchanges()
                .into_iter()
                .flat_map(|exchange| exchange.result_items())
                .collect()
        }

        /// Returns concatenated text outputs with inline citations appended.
        pub fn text_with_inline_citations(&self) -> Option<String> {
            let text = self
                .outputs
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
                .map_or(false, InteractionStatus::is_terminal)
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
        Completed,
        Failed,
        Cancelled,
    }

    impl InteractionStatus {
        /// Returns true if the status is terminal.
        pub fn is_terminal(&self) -> bool {
            matches!(
                self,
                InteractionStatus::Completed
                    | InteractionStatus::Failed
                    | InteractionStatus::Cancelled
            )
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

    impl GetTokenUsage for InteractionUsage {
        fn token_usage(&self) -> Option<Usage> {
            let mut usage = Usage::new();
            usage.input_tokens = self.total_input_tokens.unwrap_or_default();
            usage.output_tokens = self.total_output_tokens.unwrap_or_default();
            usage.total_tokens = self
                .total_tokens
                .unwrap_or(usage.input_tokens + usage.output_tokens);
            Some(usage)
        }
    }

    /// Input payload accepted by the Interactions API.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum InteractionInput {
        Text(String),
        Content(Content),
        Turns(Vec<Turn>),
        Contents(Vec<Content>),
    }

    /// Role for a conversation turn.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Model,
    }

    /// Single conversational turn with role and content.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Turn {
        pub role: Role,
        pub content: TurnContent,
    }

    /// Content for a single turn.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum TurnContent {
        Text(String),
        Contents(Vec<Content>),
    }

    impl TryFrom<crate::completion::Message> for Turn {
        type Error = message::MessageError;

        fn try_from(message: crate::completion::Message) -> Result<Self, Self::Error> {
            match message {
                crate::completion::Message::User { content } => {
                    let contents = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self {
                        role: Role::User,
                        content: TurnContent::Contents(contents),
                    })
                }
                crate::completion::Message::Assistant { content, .. } => {
                    let contents = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self {
                        role: Role::Model,
                        content: TurnContent::Contents(contents),
                    })
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
        pub language: String,
        pub code: String,
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

    impl TryFrom<message::UserContent> for Content {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text }) => Ok(Self::Text(TextContent {
                    text,
                    annotations: None,
                })),
                message::UserContent::ToolResult(message::ToolResult {
                    id,
                    call_id,
                    content,
                }) => {
                    let Some(call_id) = call_id else {
                        return Err(message::MessageError::ConversionError(
                            "Tool results require call_id for Gemini Interactions API".to_string(),
                        ));
                    };

                    let content = content.first();

                    let message::ToolResultContent::Text(text) = content else {
                        return Err(message::MessageError::ConversionError(
                            "Tool result content must be text".to_string(),
                        ));
                    };

                    let result: Value = serde_json::from_str(&text.text).unwrap_or_else(|error| {
                        tracing::trace!(?error, "Tool result is not valid JSON; sending as string");
                        json!(text.text)
                    });

                    Ok(Self::FunctionResult(FunctionResultContent {
                        name: Some(id),
                        is_error: None,
                        result: Some(result),
                        call_id: Some(call_id),
                    }))
                }
                message::UserContent::Image(message::Image {
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
                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => {
                    let media_type = media_type.ok_or_else(|| {
                        message::MessageError::ConversionError(
                            "Media type for audio is required for Gemini".to_string(),
                        )
                    })?;
                    let mime_type = media_type.to_mime_type().to_string();
                    let (data, uri) = split_data_uri(data)?;
                    Ok(Self::Audio(AudioContent {
                        data,
                        uri,
                        mime_type: Some(mime_type),
                    }))
                }
                message::UserContent::Video(message::Video {
                    data, media_type, ..
                }) => {
                    let media_type = media_type.ok_or_else(|| {
                        message::MessageError::ConversionError(
                            "Media type for video is required for Gemini".to_string(),
                        )
                    })?;
                    let mime_type = media_type.to_mime_type().to_string();
                    let (data, uri) = split_data_uri(data)?;
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
                    let mime_type = media_type.to_mime_type().to_string();
                    let (data, uri) = split_data_uri(data)?;
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
                message::AssistantContent::Text(message::Text { text }) => {
                    Ok(Self::Text(TextContent {
                        text,
                        annotations: None,
                    }))
                }
                message::AssistantContent::ToolCall(tool_call) => {
                    let call_id = tool_call.call_id.unwrap_or_else(|| tool_call.id.clone());
                    Ok(Self::FunctionCall(FunctionCallContent {
                        name: Some(tool_call.function.name),
                        arguments: Some(tool_call.function.arguments),
                        id: Some(call_id),
                    }))
                }
                message::AssistantContent::Reasoning(message::Reasoning {
                    reasoning,
                    signature,
                    ..
                }) => Ok(Self::Thought(ThoughtContent {
                    signature,
                    summary: Some(
                        reasoning
                            .into_iter()
                            .map(|text| {
                                ThoughtSummaryContent::Text(TextContent {
                                    text,
                                    annotations: None,
                                })
                            })
                            .collect(),
                    ),
                })),
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
        #[serde(rename = "interaction.start")]
        InteractionStart {
            interaction: Interaction,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "interaction.complete")]
        InteractionComplete {
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
        #[serde(rename = "content.start")]
        ContentStart {
            index: i32,
            content: Content,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "content.delta")]
        ContentDelta {
            index: i32,
            delta: ContentDelta,
            #[serde(skip_serializing_if = "Option::is_none")]
            event_id: Option<String>,
        },
        #[serde(rename = "content.stop")]
        ContentStop {
            index: i32,
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
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ContentDelta {
        Text(TextDelta),
        Image(ImageDelta),
        Audio(AudioDelta),
        Document(DocumentDelta),
        Video(VideoDelta),
        ThoughtSummary(ThoughtSummaryDelta),
        ThoughtSignature(ThoughtSignatureDelta),
        FunctionCall(FunctionCallDelta),
        FunctionResult(FunctionResultDelta),
        CodeExecutionCall(CodeExecutionCallDelta),
        CodeExecutionResult(CodeExecutionResultDelta),
        UrlContextCall(UrlContextCallDelta),
        UrlContextResult(UrlContextResultDelta),
        GoogleSearchCall(GoogleSearchCallDelta),
        GoogleSearchResult(GoogleSearchResultDelta),
        McpServerToolCall(McpServerToolCallDelta),
        McpServerToolResult(McpServerToolResultDelta),
        FileSearchResult(FileSearchResultDelta),
    }

    /// Streaming text delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct TextDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Vec<Annotation>>,
    }

    /// Streaming image delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ImageDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resolution: Option<MediaResolution>,
    }

    /// Streaming audio delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct AudioDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }

    /// Streaming document delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct DocumentDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }

    /// Streaming video delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct VideoDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub resolution: Option<MediaResolution>,
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

    /// Streaming function call delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FunctionCallDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Streaming function result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FunctionResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
    }

    /// Streaming code execution call delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct CodeExecutionCallDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<CodeExecutionCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Streaming code execution result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct CodeExecutionResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Streaming URL context call delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextCallDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<UrlContextCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Streaming URL context result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct UrlContextResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<UrlContextResult>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Streaming Google Search call delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchCallDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<GoogleSearchCallArguments>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Streaming Google Search result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct GoogleSearchResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<GoogleSearchResult>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Streaming MCP server tool call delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct McpServerToolCallDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    /// Streaming MCP server tool result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct McpServerToolResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub call_id: Option<String>,
    }

    /// Streaming file search result delta.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FileSearchResultDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Vec<FileSearchResult>>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OneOrMany;
    use crate::completion::{CompletionRequest, Message};
    use crate::message::{self, ToolChoice as MessageToolChoice};
    use serde_json::json;

    #[test]
    fn test_create_request_body_simple() {
        let prompt = Message::User {
            content: OneOrMany::one(message::UserContent::text("Hello")),
        };

        let request = CompletionRequest {
            preamble: Some("Be precise.".to_string()),
            chat_history: OneOrMany::one(prompt),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(128),
            tool_choice: Some(MessageToolChoice::Required),
            additional_params: None,
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

        let InteractionInput::Turns(turns) = result.input else {
            panic!("expected turns input");
        };
        assert_eq!(turns.len(), 1);
        let turn = &turns[0];
        assert!(matches!(turn.role, Role::User));
        let TurnContent::Contents(contents) = &turn.content else {
            panic!("expected content array");
        };
        assert_eq!(contents.len(), 1);
        match &contents[0] {
            Content::Text(TextContent { text, .. }) => assert_eq!(text, "Hello"),
            other => panic!("unexpected content: {other:?}"),
        }
    }

    #[test]
    fn test_tool_result_requires_call_id() {
        let content = message::UserContent::ToolResult(message::ToolResult {
            id: "get_weather".to_string(),
            call_id: None,
            content: OneOrMany::one(message::ToolResultContent::text("ok")),
        });

        let err = Content::try_from(content).expect_err("should require call_id");
        assert!(format!("{err}").contains("call_id"));
    }

    #[test]
    fn test_response_function_call_mapping() {
        let interaction = Interaction {
            id: "interaction-1".to_string(),
            outputs: vec![Content::FunctionCall(FunctionCallContent {
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

        let response: completion::CompletionResponse<Interaction> =
            interaction.try_into().expect("conversion should succeed");

        let choice = response.choice.first();
        match choice {
            completion::AssistantContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.call_id.as_deref(), Some("call-123"));
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
    fn test_google_search_helpers() {
        let interaction = Interaction {
            outputs: vec![
                Content::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-one".to_string(), "query-two".to_string()]),
                    }),
                    id: Some("call-1".to_string()),
                }),
                Content::GoogleSearchResult(GoogleSearchResultContent {
                    result: Some(vec![GoogleSearchResult {
                        url: Some("https://example.com".to_string()),
                        title: Some("Example One".to_string()),
                        rendered_content: None,
                    }]),
                    signature: None,
                    is_error: None,
                    call_id: Some("call-1".to_string()),
                }),
                Content::GoogleSearchCall(GoogleSearchCallContent {
                    arguments: Some(GoogleSearchCallArguments {
                        queries: Some(vec!["query-three".to_string()]),
                    }),
                    id: Some("call-2".to_string()),
                }),
                Content::GoogleSearchResult(GoogleSearchResultContent {
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
    fn test_url_context_helpers() {
        let interaction = Interaction {
            outputs: vec![
                Content::UrlContextCall(UrlContextCallContent {
                    arguments: Some(UrlContextCallArguments {
                        urls: Some(vec![
                            "https://example.com".to_string(),
                            "https://example.org".to_string(),
                        ]),
                    }),
                    id: Some("call-1".to_string()),
                }),
                Content::UrlContextResult(UrlContextResultContent {
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
    fn test_interaction_status_helpers() {
        let mut interaction = Interaction {
            status: Some(InteractionStatus::InProgress),
            ..Default::default()
        };
        assert!(!interaction.is_terminal());
        assert!(!interaction.is_completed());

        interaction.status = Some(InteractionStatus::Completed);
        assert!(interaction.is_terminal());
        assert!(interaction.is_completed());

        interaction.status = Some(InteractionStatus::Failed);
        assert!(interaction.is_terminal());
        assert!(!interaction.is_completed());
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
            outputs: vec![Content::Text(text_content)],
            ..Default::default()
        };

        let cited_text = interaction.text_with_inline_citations();
        assert_eq!(
            cited_text.as_deref(),
            Some("Hello[1](https://hello.example) world[2](https://example.com)")
        );
    }
}
