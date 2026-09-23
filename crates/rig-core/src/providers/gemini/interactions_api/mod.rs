//! Wires and payload types for the [Gemini Interactions API](https://ai.google.dev/api/interactions-api).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, completion::GEMINI_2_5_FLASH};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Gemini::from_env()?.interactions(GEMINI_2_5_FLASH);
//! # Ok(())
//! # }
//! ```

use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::message::{self, MimeType};
use crate::telemetry::GenAiOperation;
use crate::wire::Mode;
use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::{Map, Value};
use url::form_urlencoded;

/// Streaming helpers for the Interactions API.
pub mod streaming;
pub use interactions_api_types::*;

/// Gemini provider name used in normalized records and telemetry.
pub(crate) const PROVIDER_NAME: &str = "gcp.gemini";

/// Create interactions with `POST /v1beta/interactions`.
/// Streaming mode sets `alt=sse` and `stream: true`; unary mode reads a whole resource.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interactions {
    /// The key and the API root.
    pub provider: crate::providers::gemini::Gemini,
    /// The model to address.
    pub model: String,
}

impl Interactions {
    /// The wire for `model`.
    pub fn new(provider: crate::providers::gemini::Gemini, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl crate::wire::Wire for Interactions {
    type Op = crate::operation::Completion;
    type Decoder = streaming::InteractionsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn telemetry(&self, streaming: bool) -> GenAiOperation {
        if streaming {
            GenAiOperation::InteractionsStreaming
        } else {
            GenAiOperation::Interactions
        }
    }

    fn encode(
        &self,
        request: CompletionRequest,
        mode: crate::wire::Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        // `stream` is part of the request body on this wire, so the mode is
        // in the bytes as well as in the path.
        let streaming = matches!(mode, crate::wire::Mode::Streaming);
        let body = create_request_body(self.model.clone(), request, Some(streaming))?;
        crate::providers::internal::trace_json(
            if streaming {
                crate::providers::internal::LogTarget::Streaming
            } else {
                crate::providers::internal::LogTarget::Completions
            },
            "Gemini interactions completion request",
            &body,
        );
        let (path, framing) = if streaming {
            (
                "/v1beta/interactions?alt=sse",
                crate::http_client::framing::Framing::Sse,
            )
        } else {
            (
                "/v1beta/interactions",
                crate::http_client::framing::Framing::Whole,
            )
        };
        let request = http::Request::post(self.provider.interactions_uri(path))
            .header("Content-Type", "application/json")
            .header(
                crate::providers::gemini::Gemini::INTERACTIONS_KEY_HEADER,
                self.provider.api_key.expose(),
            )
            .body(crate::wire::Body::Bytes(serde_json::to_vec(&body)?))?;
        // Gemini supplies no transport request-id response header.
        Ok(crate::wire::Encoded::new(request, framing))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        streaming::InteractionsDecoder::default()
    }
}

/// Read an existing interaction resource or resume its event stream.
/// Unary mode retrieves the resource once. Streaming mode resumes after
/// `last_event_id`, or from the beginning if no event id is supplied.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionResume {
    /// The key and the API root.
    pub provider: crate::providers::gemini::Gemini,
    /// The interaction to read.
    pub interaction_id: String,
    /// The last event the consumer saw, so a resumed stream does not
    /// redeliver it. `None` resumes from the beginning, as the API defaults.
    pub last_event_id: Option<String>,
}

impl InteractionResume {
    /// The wire for the interaction `interaction_id`.
    pub fn new(
        provider: crate::providers::gemini::Gemini,
        interaction_id: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            interaction_id: interaction_id.into(),
            last_event_id: None,
        }
    }

    /// Resume a streamed read after the event `last_event_id`.
    pub fn after_event(mut self, last_event_id: impl Into<String>) -> Self {
        self.last_event_id = Some(last_event_id.into());
        self
    }
}

impl crate::wire::Wire for InteractionResume {
    type Op = crate::operation::Completion;
    type Decoder = streaming::InteractionsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    /// The interaction names its own model; this wire addresses none.
    fn model(&self) -> Option<&str> {
        None
    }

    fn telemetry(&self, streaming: bool) -> GenAiOperation {
        if streaming {
            GenAiOperation::InteractionsStreaming
        } else {
            GenAiOperation::Interactions
        }
    }

    /// Reads an existing interaction, so the request carries no body and the
    /// [`CompletionRequest`] contributes nothing: what to read is the wire's
    /// own data.
    fn encode(
        &self,
        _request: CompletionRequest,
        mode: crate::wire::Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        let (path, framing) = match mode {
            Mode::Unary => (
                format!("/v1beta/interactions/{}", self.interaction_id),
                crate::http_client::framing::Framing::Whole,
            ),
            Mode::Streaming => (
                format!(
                    "{}&alt=sse",
                    build_interaction_stream_path(
                        &self.interaction_id,
                        self.last_event_id.as_deref(),
                    )
                ),
                crate::http_client::framing::Framing::Sse,
            ),
        };
        let request = http::Request::get(self.provider.interactions_uri(&path))
            .header(
                crate::providers::gemini::Gemini::INTERACTIONS_KEY_HEADER,
                self.provider.api_key.expose(),
            )
            .body(crate::wire::Body::empty())?;
        Ok(crate::wire::Encoded::new(request, framing))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        streaming::InteractionsDecoder::default()
    }
}

pub(crate) fn create_request_body(
    model: String,
    completion_request: CompletionRequest,
    stream_override: Option<bool>,
) -> Result<CreateInteractionRequest, EncodeError> {
    let chat_history = completion_request.chat_history_with_documents();

    let mut history = Vec::new();
    history.extend(chat_history);
    // functionResponse.name keys the replay: cross-provider ingested
    // results arrive with an empty name and their call carries it.
    crate::providers::internal::resolve_empty_tool_result_names(&mut history);
    let (history_system, history) = split_system_messages_from_history(history);

    let tool_ids = crate::providers::internal::tool_call_ids::ToolCallIds::new(&history)
        .map_err(EncodeError::request)?;
    let mut steps = Vec::new();
    for (position, message) in history.into_iter().enumerate() {
        let mut converted = Step::from_message(message).map_err(EncodeError::request)?;
        tool_ids
            .apply(
                position,
                converted.iter_mut().filter_map(|step| match step {
                    Step::FunctionCall(call) => call.id.as_mut(),
                    Step::FunctionResult(result) => result.call_id.as_mut(),
                    _ => None,
                }),
            )
            .map_err(EncodeError::request)?;
        steps.extend(converted);
    }

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

    let system_instruction = (!history_system.is_empty())
        .then(|| history_system.join("\n\n"))
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
        return Err(EncodeError::request(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "response_mime_type is required when response_format is set",
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

/// Request and response types for the Gemini Interactions API.
///
/// ```
/// use rig_core::providers::gemini::interactions_api::InteractionStatus;
///
/// assert!(InteractionStatus::RequiresAction.is_terminal());
/// ```
pub mod interactions_api_types {
    use super::{media_parts, split_data_uri};
    use crate::completion::Usage;
    use crate::error::EncodeError;
    use crate::message::{self, MimeType};
    use base64::{Engine, prelude::BASE64_STANDARD};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

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
        /// An unrecognized status, preserved verbatim without rejecting the response.
        #[serde(untagged)]
        Unknown(String),
    }

    impl InteractionStatus {
        /// Return false only for [`Self::InProgress`].
        /// Stop polling on unknown statuses and handle them explicitly.
        /// [`Self::RequiresAction`] needs caller-supplied tool results, not further polling.
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

    /// Normalize completed, requires-action, and budget-exceeded statuses.
    /// Preserve every other status verbatim as `Other`.
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
    #[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "snake_case")]
    pub struct InteractionUsage {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub total_tokens: Option<u64>,
        /// Input tokens served from Gemini's cache.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub total_cached_tokens: Option<u64>,
        /// Thinking tokens, reported separately from `total_output_tokens`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub total_thought_tokens: Option<u64>,
        /// Prompt tokens attributed to built-in tool use.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub total_tool_use_tokens: Option<u64>,
    }

    impl From<&InteractionUsage> for Usage {
        fn from(value: &InteractionUsage) -> Usage {
            // Thinking and tool-use tokens are separate from input and output.
            // Derive a fallback only when both base counts exist; reported totals take precedence.
            let derived_total =
                value
                    .total_input_tokens
                    .zip(value.total_output_tokens)
                    .map(|(input, output)| {
                        input
                            + output
                            + value.total_thought_tokens.unwrap_or(0)
                            + value.total_tool_use_tokens.unwrap_or(0)
                    });
            Usage {
                input_tokens: value.total_input_tokens,
                output_tokens: value.total_output_tokens,
                cached_input_tokens: value.total_cached_tokens,
                reasoning_tokens: value.total_thought_tokens,
                tool_use_prompt_tokens: value.total_tool_use_tokens,
                total_tokens: value.total_tokens.or(derived_total),
                cache_creation_input_tokens: None,
            }
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
        // Streaming announcements may omit content until later step.delta events.
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

    impl Step {
        /// Convert a history message into ordered interaction steps.
        /// Calls, results, and thoughts become separate steps; adjacent text and
        /// media remain grouped. Return conversion errors for unsupported content.
        pub(crate) fn from_message(
            message: crate::completion::Message,
        ) -> Result<Vec<Self>, message::MessageError> {
            match message {
                crate::completion::Message::System { content } => Ok(vec![Self::UserInput {
                    content: vec![Content::Text(TextContent {
                        text: content,
                        annotations: None,
                    })],
                }]),
                crate::completion::Message::User { content } => {
                    let contents = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self::split(contents, |content| Self::UserInput { content }))
                }
                crate::completion::Message::Assistant { content, .. } => {
                    let contents = content
                        .into_iter()
                        .map(Content::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Self::split(contents, |content| Self::ModelOutput {
                        content,
                    }))
                }
            }
        }

        /// Lift the contents that are steps of their own out of `contents`,
        /// grouping each run of the others under `group`.
        fn split(contents: Vec<Content>, group: impl Fn(Vec<Content>) -> Self) -> Vec<Self> {
            let mut steps = Vec::new();
            let mut run: Vec<Content> = Vec::new();
            for content in contents {
                let own = match content {
                    Content::Thought(thought) => Some(Self::Thought(thought)),
                    Content::FunctionCall(call) => Some(Self::FunctionCall(call)),
                    Content::FunctionResult(result) => Some(Self::FunctionResult(result)),
                    Content::CodeExecutionCall(call) => Some(Self::CodeExecutionCall(call)),
                    Content::CodeExecutionResult(result) => Some(Self::CodeExecutionResult(result)),
                    Content::UrlContextCall(call) => Some(Self::UrlContextCall(call)),
                    Content::UrlContextResult(result) => Some(Self::UrlContextResult(result)),
                    Content::GoogleSearchCall(call) => Some(Self::GoogleSearchCall(call)),
                    Content::GoogleSearchResult(result) => Some(Self::GoogleSearchResult(result)),
                    Content::McpServerToolCall(call) => Some(Self::McpServerToolCall(call)),
                    Content::McpServerToolResult(result) => Some(Self::McpServerToolResult(result)),
                    Content::FileSearchResult(result) => Some(Self::FileSearchResult(result)),
                    grouped @ (Content::Text(_)
                    | Content::Image(_)
                    | Content::Audio(_)
                    | Content::Document(_)
                    | Content::Video(_)) => {
                        run.push(grouped);
                        None
                    }
                };
                if let Some(step) = own {
                    if !run.is_empty() {
                        steps.push(group(std::mem::take(&mut run)));
                    }
                    steps.push(step);
                }
            }
            if !run.is_empty() {
                steps.push(group(run));
            }
            steps
        }
    }

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
                        .map_or(0, |idx| idx + 1);
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
                let citation = format!("[{index}]({source})");
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

    /// Thought summary item with the `type` tag required for replay.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
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
                    // The wire requires a call id even when the original provider issued none.
                    let call_id = tool_result.wire_call_id().into_owned();
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
                            // A scalar or array JSON result is wrapped as the
                            // generate wire wraps it (`{"result": value}`): sent
                            // as a text block it is a multimodal response, which
                            // the models refuse.
                            message::ToolResultContent::Json { value } => match value {
                                value @ (Value::String(_) | Value::Object(_)) => value,
                                value @ (Value::Null
                                | Value::Bool(_)
                                | Value::Number(_)
                                | Value::Array(_)) => serde_json::json!({ "result": value }),
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
                    let call_id = tool_call.wire_call_id().into_owned();
                    Ok(Self::FunctionCall(FunctionCallContent {
                        name: Some(tool_call.function.name),
                        arguments: Some(tool_call.function.arguments),
                        id: Some(call_id),
                    }))
                }
                message::AssistantContent::Reasoning(message::Reasoning { content, .. }) => {
                    // Preserve signature-only thoughts without empty summary items,
                    // which the API rejects.
                    let signature = content.iter().find_map(|part| match part {
                        message::ReasoningContent::Text { signature, .. } => signature.clone(),
                        message::ReasoningContent::Summary(_)
                        | message::ReasoningContent::Encrypted(_)
                        | message::ReasoningContent::Redacted { .. } => None,
                    });
                    let summary: Vec<ThoughtSummaryContent> = content
                        .into_iter()
                        .map(|part| match part {
                            message::ReasoningContent::Text { text, .. }
                            | message::ReasoningContent::Summary(text)
                            | message::ReasoningContent::Encrypted(text) => text,
                            message::ReasoningContent::Redacted { data } => data,
                        })
                        .filter(|text| !text.is_empty())
                        .map(|text| {
                            ThoughtSummaryContent::Text(TextContent {
                                text,
                                annotations: None,
                            })
                        })
                        .collect();

                    Ok(Self::Thought(ThoughtContent {
                        signature,
                        summary: (!summary.is_empty()).then_some(summary),
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
        type Error = EncodeError;

        fn try_from(tool: crate::completion::ToolDefinition) -> Result<Self, Self::Error> {
            Ok(Tool::Function(FunctionTool {
                name: Some(tool.name),
                description: Some(tool.description),
                parameters: Some(tool.parameters),
            }))
        }
    }

    impl TryFrom<message::ToolChoice> for ToolChoice {
        type Error = EncodeError;

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

    /// A tagged content delta containing a whole item or a text, argument, or thought fragment.
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

    /// Raw JSON argument fragment for the function call at the enclosing step index.
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
mod tests;

#[cfg(test)]
mod interaction_usage_tests;
