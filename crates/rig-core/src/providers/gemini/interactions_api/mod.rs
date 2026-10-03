//! Wires for the [Gemini Interactions API](https://ai.google.dev/api/interactions-api).
//! A request is built as JSON steps, and a reply, a whole interaction or a
//! stream of step events, is read by [`InteractionsDecoder`](streaming::InteractionsDecoder).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, completion::GEMINI_2_5_FLASH};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Gemini::from_env()?.interactions(GEMINI_2_5_FLASH);
//! # Ok(())
//! # }
//! ```

use serde_json::{Map, Value, json};
use url::form_urlencoded;

use crate::completion::{CompletionRequest, Media, Replay, ReplayTarget};
use crate::error::EncodeError;
use crate::message::{
    AssistantContent, DocumentMediaType, DocumentSourceKind as Source, Message, MimeType,
    ToolChoice as Choice, ToolResultContent, UserContent,
};
use crate::providers::internal::wire_ids::WireIds;
use crate::telemetry::GenAiOperation;
use crate::wire::{Descriptor, Mode};

/// Streaming helpers for the Interactions API.
pub mod streaming;

/// Gemini provider name used in normalized records and telemetry.
pub(crate) const PROVIDER_NAME: &str = "gcp.gemini";

/// The wire format both Interactions wires speak.
const API: crate::message::Api = crate::message::Api::from_static("gemini.interactions");

/// Create interactions with `POST /v1beta/interactions`.
/// Streaming mode sets `alt=sse` and `stream: true`; unary mode reads a whole resource.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interactions {
    /// The key and the API root.
    pub provider: crate::providers::gemini::GeminiConfig,
    /// The model to address.
    pub model: String,
}

impl Interactions {
    /// The wire for `model`.
    pub fn new(provider: crate::providers::gemini::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

fn telemetry(mode: Mode) -> GenAiOperation {
    match mode {
        Mode::Unary => GenAiOperation::Interactions,
        Mode::Streaming => GenAiOperation::InteractionsStreaming,
    }
}

impl crate::wire::Wire for Interactions {
    type Op = crate::operation::Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = streaming::InteractionsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .telemetry(telemetry)
            .replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        use crate::providers::internal::LogTarget;
        // `stream` is part of the request body on this wire, so the mode is
        // in the bytes as well as in the path.
        let streaming = matches!(mode, Mode::Streaming);
        let body = create_request_body(self, request, Some(streaming))?;
        let (path, framing, target) = match streaming {
            true => (
                "/v1beta/interactions?alt=sse",
                crate::wire::Framing::Sse,
                LogTarget::Streaming,
            ),
            false => (
                "/v1beta/interactions",
                crate::wire::Framing::Whole,
                LogTarget::Completions,
            ),
        };
        crate::providers::internal::trace_json(
            target,
            "Gemini interactions completion request",
            &body,
        );
        let request = http::Request::post(self.provider.interactions_uri(path))
            .header("Content-Type", "application/json")
            .header(
                crate::providers::gemini::GeminiConfig::INTERACTIONS_KEY_HEADER,
                self.provider.api_key.expose(),
            )
            .body(crate::wire::Body::Bytes(serde_json::to_vec(&body)?))?;
        // Gemini supplies no transport request-id response header.
        Ok(crate::wire::Encoded::new(request, framing))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        streaming::InteractionsDecoder::default()
    }
}

impl ReplayTarget for Interactions {
    fn api(&self) -> crate::message::Api {
        API
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// What the model reads, by the classifier every Gemini wire shares.
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        super::completion::accepts(model)
    }

    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        encodes(media)
    }

    /// A request naming `previous_interaction_id` continues an interaction
    /// the API stored, which holds the calls its first results answer.
    fn continues_stored(&self, request: &CompletionRequest) -> bool {
        request
            .additional_params
            .as_ref()
            .and_then(|params| params.get("previous_interaction_id"))
            .is_some_and(|id| !id.is_null())
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        normalize_tool_call_id(id)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    /// A step's `id` survives an edit of its block.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        item.get("id")
            .map(|id| Map::from_iter([("id".to_owned(), id.clone())]))
            .unwrap_or_default()
    }
}

/// Whether this API takes `media`: data or a URL with a media type, an
/// image of a type Gemini reads, and a document other than a PDF only as a
/// string, which is sent as text. A file id is never taken.
fn encodes(media: Media<'_>) -> bool {
    let carried = |source: &Source| {
        matches!(
            source,
            Source::Url(_) | Source::Base64(_) | Source::String(_)
        )
    };
    match media {
        Media::Image(image, place) => {
            super::completion::reads_image(image.media_type.as_ref(), place) && carried(&image.data)
        }
        Media::Audio(audio) => audio.media_type.is_some() && carried(&audio.data),
        Media::Video(video) => video.media_type.is_some() && carried(&video.data),
        Media::Document(document) => match (&document.media_type, &document.data) {
            (None, _) => false,
            (Some(DocumentMediaType::PDF), data) => carried(data),
            (Some(_), data) => matches!(data, Source::String(_)),
        },
    }
}

/// A foreign call id as this wire accepts it: `[a-zA-Z0-9_-]`, at most 64
/// characters.
fn normalize_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => c,
            _ => '_',
        })
        .take(64)
        .collect()
}

/// Read an existing interaction resource or resume its event stream.
/// Unary mode retrieves the resource once. Streaming mode resumes after
/// `last_event_id`, or from the beginning if no event id is supplied.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionResume {
    /// The key and the API root.
    pub provider: crate::providers::gemini::GeminiConfig,
    /// The interaction to read.
    pub interaction_id: String,
    /// The last event the consumer saw, so a resumed stream does not
    /// redeliver it. `None` resumes from the beginning, as the API defaults.
    pub last_event_id: Option<String>,
}

impl InteractionResume {
    /// The wire for the interaction `interaction_id`.
    pub fn new(
        provider: crate::providers::gemini::GeminiConfig,
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
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = streaming::InteractionsDecoder;

    /// The interaction names its own model; this wire addresses no model id.
    /// The decoder reports the model the interaction names, which the turn's
    /// origin takes.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .telemetry(telemetry)
            .replay(self)
    }

    /// Reads an existing interaction, so the request carries no body and the
    /// [`CompletionRequest`] contributes nothing: what to read is the wire's
    /// own data.
    fn encode(
        &self,
        _request: CompletionRequest,
        mode: Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        let id = &self.interaction_id;
        let (path, framing) = match mode {
            Mode::Unary => (
                format!("/v1beta/interactions/{id}"),
                crate::wire::Framing::Whole,
            ),
            Mode::Streaming => {
                let mut query = form_urlencoded::Serializer::new(String::new());
                query.append_pair("stream", "true");
                if let Some(last_event_id) = &self.last_event_id {
                    query.append_pair("last_event_id", last_event_id);
                }
                let path = format!("/v1beta/interactions/{id}?{}&alt=sse", query.finish());
                (path, crate::wire::Framing::Sse)
            }
        };
        let request = http::Request::get(self.provider.interactions_uri(&path))
            .header(
                crate::providers::gemini::GeminiConfig::INTERACTIONS_KEY_HEADER,
                self.provider.api_key.expose(),
            )
            .body(crate::wire::Body::empty())?;
        Ok(crate::wire::Encoded::new(request, framing))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        streaming::InteractionsDecoder::default()
    }
}

impl ReplayTarget for InteractionResume {
    fn api(&self) -> crate::message::Api {
        API
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        ""
    }

    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        super::completion::accepts(model)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        normalize_tool_call_id(id)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }
}

/// The create-interaction body `request` sends on `wire`. `additional_params`
/// is merged into the body as the API names its fields: its
/// `generation_config` is the base the typed fields override, its `tools`
/// add to the request's, and an `agent` replaces the model. System messages
/// become the system instruction. `stream`, when set, overrides the one the
/// parameters name.
///
/// # Errors
///
/// When `additional_params` is not an object, or sets `response_format`
/// without `response_mime_type`.
pub(crate) fn create_request_body(
    wire: &Interactions,
    request: CompletionRequest,
    stream: Option<bool>,
) -> Result<Value, EncodeError> {
    let model = request.model.clone().unwrap_or_else(|| wire.model.clone());
    let mut body = match request.additional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(EncodeError::request(
                "Gemini Interactions `additional_params` should be an object",
            ));
        }
    };
    let set = |key: &str| body.get(key).is_some_and(|value| !value.is_null());
    let agent = set("agent");
    if set("response_format") && !set("response_mime_type") {
        return Err(EncodeError::request(
            "response_mime_type is required when response_format is set",
        ));
    }
    let mut config = match body.shift_remove("generation_config") {
        Some(Value::Object(config)) => config,
        _ => Map::new(),
    };
    let choice = request.tool_choice.map(|choice| match choice {
        Choice::Auto => json!("auto"),
        Choice::None => json!("none"),
        Choice::Required => json!("any"),
        Choice::Specific { function_names } => {
            json!({ "allowed_tools": { "mode": "validated", "tools": function_names } })
        }
    });
    let temperature = ("temperature", request.temperature.map(Value::from));
    let max_tokens = ("max_output_tokens", request.max_tokens.map(Value::from));
    for (key, value) in [temperature, max_tokens, ("tool_choice", choice)] {
        if let Some(value) = value {
            config.insert(key.to_owned(), value);
        }
    }
    config.retain(|_, value| !value.is_null());
    if !config.is_empty() {
        body.insert("generation_config".to_owned(), Value::Object(config));
    }
    let mut tools: Vec<Value> = request
        .tools
        .into_iter()
        .map(|tool| json!({ "type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters }))
        .collect();
    if let Some(Value::Array(extra)) = body.shift_remove("tools") {
        tools.extend(extra);
    }
    if !tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(tools));
    }
    let (mut system, mut history) = (Vec::new(), Vec::new());
    for message in request.chat_history {
        match message {
            Message::System { content } => system.push(content),
            message => history.push(message),
        }
    }
    if !system.is_empty() {
        body.insert("system_instruction".to_owned(), json!(system.join("\n\n")));
    }
    if !agent {
        body.shift_remove("agent_config");
        body.insert("model".to_owned(), json!(model));
    }
    if let Some(stream) = stream {
        body.insert("stream".to_owned(), json!(stream));
    }
    body.insert(
        "input".to_owned(),
        Value::Array(steps(history, wire, &model)?),
    );
    body.retain(|_, value| !value.is_null());
    Ok(Value::Object(body))
}

/// The steps `history` sends: a user message's content grouped into
/// `user_input` steps around its function results, each a step of its own,
/// and one step per assistant block.
fn steps(
    history: Vec<Message>,
    target: &dyn ReplayTarget,
    model: &str,
) -> Result<Vec<Value>, EncodeError> {
    let ids = WireIds::for_target(&history, target, model);
    let mut steps = Vec::new();
    for message in history {
        match message {
            Message::System { content } => {
                steps.push(json!({ "type": "user_input", "content": [{ "type": "text", "text": content }] }));
            }
            Message::User { content } => {
                let mut run = Vec::new();
                for part in content {
                    let UserContent::ToolResult(result) = part else {
                        run.push(user_content(part)?);
                        continue;
                    };
                    if !run.is_empty() {
                        steps.push(
                            json!({ "type": "user_input", "content": std::mem::take(&mut run) }),
                        );
                    }
                    let mut contents = result.content;
                    let value = match (contents.len(), contents.pop()) {
                        (1, Some(ToolResultContent::Text(text))) => Value::String(text.text),
                        // A scalar or array JSON result is wrapped as the
                        // generate wire wraps it: sent as a text block it is a
                        // multimodal response, which the models refuse.
                        (
                            1,
                            Some(ToolResultContent::Json {
                                value: value @ (Value::String(_) | Value::Object(_)),
                            }),
                        ) => value,
                        (1, Some(ToolResultContent::Json { value })) => json!({ "result": value }),
                        (_, last) => {
                            contents.extend(last);
                            Value::Array(
                                contents
                                    .into_iter()
                                    .map(result_content)
                                    .collect::<Result<_, _>>()?,
                            )
                        }
                    };
                    let mut step = json!({
                        "type": "function_result",
                        "name": result.name,
                        "call_id": ids.of(&result.call),
                        "result": value,
                    });
                    if let (true, Some(step)) = (result.is_error, step.as_object_mut()) {
                        step.insert("is_error".to_owned(), Value::Bool(true));
                    }
                    steps.push(step);
                }
                if !run.is_empty() {
                    steps.push(json!({ "type": "user_input", "content": run }));
                }
            }
            Message::Assistant(turn) => {
                for block in &turn.content {
                    steps.extend(assistant_step(block, target, &ids)?);
                }
            }
        }
    }
    Ok(steps)
}

/// A block of a tool result as a content item.
fn result_content(content: ToolResultContent) -> Result<Value, EncodeError> {
    Ok(match content {
        ToolResultContent::Text(text) => json!({ "type": "text", "text": text.text }),
        ToolResultContent::Json { value } => json!({ "type": "text", "text": value.to_string() }),
        ToolResultContent::Image(image) => media("image", image.media_type, image.data)?,
    })
}

/// A user part as a content item. A text document goes as text, so that RAG
/// context reads as prose.
fn user_content(part: UserContent) -> Result<Value, EncodeError> {
    match part {
        UserContent::Text(text) => Ok(json!({ "type": "text", "text": text.text })),
        UserContent::Image(image) => media("image", image.media_type, image.data),
        UserContent::Audio(audio) => media("audio", audio.media_type, audio.data),
        UserContent::Video(video) => media("video", video.media_type, video.data),
        UserContent::Document(document) => match (document.media_type, document.data) {
            (Some(media_type), Source::String(text)) if media_type != DocumentMediaType::PDF => {
                Ok(json!({ "type": "text", "text": text }))
            }
            (media_type, data) => media("document", media_type, data),
        },
        UserContent::ToolResult(_) => {
            Err(EncodeError::request("a tool result is a step of its own"))
        }
    }
}

/// A media content item of `kind`: a URL as its `uri`, and base64 data, or a
/// string's bytes in base64, as its `data`. [`encodes`] refuses every other
/// form, so the adapter passes none.
fn media<M: MimeType>(
    kind: &str,
    media_type: Option<M>,
    source: Source,
) -> Result<Value, EncodeError> {
    let unsendable = || {
        EncodeError::request(format!(
            "Gemini Interactions cannot receive this {kind} in its form"
        ))
    };
    let mime_type = media_type.ok_or_else(unsendable)?.to_mime_type().to_owned();
    let key = match super::completion::carried(source, false)? {
        (true, uri) => ("uri", uri),
        (false, data) => ("data", data),
    };
    Ok(json!({ "type": kind, key.0: key.1, "mime_type": mime_type }))
}

/// An assistant block as its step: the provider's step while the block is
/// current, else one rebuilt from its canonical fields, keeping the keys
/// its edited step names; `None` for reasoning with nothing to send.
fn assistant_step(
    block: &AssistantContent,
    target: &dyn ReplayTarget,
    ids: &WireIds,
) -> Result<Option<Value>, EncodeError> {
    let identity = match block.replay(target, ids) {
        Replay::Item(item) => return Ok(Some(item.into_owned())),
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    let output = |content: Value| json!({ "type": "model_output", "content": [content] });
    let mut step = match block {
        AssistantContent::Text(text) => output(json!({ "type": "text", "text": text.text })),
        AssistantContent::Reasoning(reasoning)
            if reasoning.redacted || reasoning.text.trim().is_empty() =>
        {
            return Ok(None);
        }
        AssistantContent::Reasoning(reasoning) => {
            json!({ "type": "thought", "summary": [{ "type": "text", "text": reasoning.text }] })
        }
        AssistantContent::ToolCall(call) => json!({
            "type": "function_call",
            "name": call.function.name,
            "arguments": call.function.arguments_value(),
        }),
        AssistantContent::Image(image) => output(media(
            "image",
            image.media_type.clone(),
            image.data.clone(),
        )?),
        AssistantContent::Opaque(opaque) => return Ok(Some(opaque.item.clone())),
    };
    if let Some(fields) = step.as_object_mut() {
        fields.extend(identity);
        if let AssistantContent::ToolCall(call) = block {
            fields.insert("id".to_owned(), json!(ids.of(&call.id)));
        }
    }
    Ok(Some(step))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod interaction_usage_tests;

#[cfg(test)]
mod history_tests;
