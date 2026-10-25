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
    AssistantContent, DocumentData, DocumentMediaType, DocumentSourceKind as Source, Message,
    MimeType, ToolChoice as Choice, ToolResultContent, UserContent,
};
use crate::providers::internal::wire_ids::WireIds;
use crate::telemetry::GenAiOperation;
use crate::wire::{Descriptor, Mode};

/// Streaming helpers for the Interactions API.
pub mod streaming;

use super::completion::PROVIDER_NAME;

/// `deep-research-preview-04-2026`, the Deep Research agent, built for
/// speed. Name it with
/// [`InteractionsOptions::agent`](super::extension::InteractionsOptions::agent).
pub const DEEP_RESEARCH_PREVIEW_04_2026: &str = "deep-research-preview-04-2026";
/// `deep-research-max-preview-04-2026`, Deep Research Max, built for
/// thoroughness.
pub const DEEP_RESEARCH_MAX_PREVIEW_04_2026: &str = "deep-research-max-preview-04-2026";

/// The wire format both Interactions wires speak.
const API: crate::message::Api = crate::message::Api::from_static("gemini.interactions");

/// Where a thought step holds its signature.
const BLOB_SLOTS: &[&str] = &["/signature"];

/// The Interactions `speech_config`: one voice, or a voice per named
/// speaker.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(untagged)]
pub enum InteractionSpeechConfig {
    /// An array of one voice.
    Voice([InteractionSpeech; 1]),
    /// `{"speakers": [...]}`: each speaker's voice, matched by the name the
    /// input's `speech_metadata` gives the speaker.
    Speakers {
        /// `speakers`.
        speakers: Vec<InteractionSpeech>,
    },
}

impl InteractionSpeechConfig {
    /// Speak with `voice`.
    pub fn voice(voice: InteractionSpeech) -> Self {
        Self::Voice([voice])
    }

    /// Speak each `(speaker, voice)` pair's turns in its voice.
    pub fn speakers<S: Into<String>, V: Into<String>>(
        speakers: impl IntoIterator<Item = (S, V)>,
    ) -> Self {
        Self::Speakers {
            speakers: speakers
                .into_iter()
                .map(|(speaker, voice)| InteractionSpeech::voice(voice).speaker(speaker))
                .collect(),
        }
    }
}

/// One voice of the Interactions `speech_config`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct InteractionSpeech {
    /// `voice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// `language`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// `speaker`: the name the prompt gives the speaker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

impl InteractionSpeech {
    /// Speak with `voice`.
    pub fn voice(voice: impl Into<String>) -> Self {
        Self {
            voice: Some(voice.into()),
            ..Self::default()
        }
    }

    /// `language`.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// `speaker`.
    pub fn speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }
}

/// Create interactions with `POST /v1beta/interactions`.
/// Streaming mode sets `alt=sse` and `stream: true`; unary mode reads a whole resource.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interactions {
    /// The key and the API root.
    pub provider: crate::providers::gemini::GeminiConfig,
    /// The model to address.
    pub model: String,
    /// The model facts the encoder reads and replies are priced by.
    #[serde(skip)]
    pub facts: crate::catalog::ModelFacts,
}

impl Interactions {
    /// The wire for `model`.
    pub fn new(provider: crate::providers::gemini::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            facts: crate::catalog::ModelFacts::default(),
        }
    }

    /// The same wire, encoding with `facts` and pricing its replies by
    /// them.
    pub fn with_facts(mut self, facts: crate::catalog::ModelFacts) -> Self {
        self.facts = facts;
        self
    }
}

impl crate::wire::Wire for Interactions {
    type Op = crate::operation::Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = streaming::InteractionsEvent;
    type Decoder<'id> = streaming::InteractionsDecoder;
    type Reassembler = streaming::document::Interaction;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .facts(&self.facts)
            .telemetry(|_| GenAiOperation::Chat)
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
        let body = create_request_body(self, &request, Some(streaming))?;
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
            .body(body.into_body())?;
        // Gemini supplies no transport request-id response header.
        Ok(crate::wire::Encoded::new(request, framing))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        streaming::InteractionsDecoder::default()
    }
}

impl ReplayTarget for Interactions {
    /// Section 6.4 of the typed-options design, for Interactions.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        super::options::interactions(&self.facts, model, fields)
    }

    fn facts(&self) -> Option<&crate::catalog::ModelFacts> {
        Some(&self.facts)
    }

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
        crate::completion::options::param(self, request, "previous_interaction_id")
            .is_some_and(|id| !id.is_null())
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        crate::providers::internal::wire_ids::legal_call_id(id, 64)
    }

    /// Gemini takes system text only in `systemInstruction`: later system
    /// messages fold into the leading one.
    fn later_system(&self, _model: &str) -> crate::completion::LaterSystem {
        crate::completion::LaterSystem::Leading
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn blob_slots(&self) -> &'static [&'static str] {
        BLOB_SLOTS
    }

    /// A step's `id` survives an edit of its block.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        item.get("id")
            .map(|id| Map::from_iter([("id".to_owned(), id.clone())]))
            .unwrap_or_default()
    }
}

/// Whether this API takes `media`: data or a URL with a media type, an
/// image of a type Gemini reads, a PDF, and a text document, which is sent
/// as text. A file id is never taken.
fn encodes(media: Media<'_>) -> bool {
    let carried = |source: &Source| matches!(source, Source::Url(_) | Source::Base64(_));
    match media {
        Media::Image(image, place) => {
            super::completion::reads_image(image.media_type.as_ref(), place) && carried(&image.data)
        }
        Media::Audio(audio) => audio.media_type.is_some() && carried(&audio.data),
        Media::Video(video) => video.media_type.is_some() && carried(&video.data),
        Media::Document(document) => match (&document.media_type, &document.data) {
            (_, DocumentData::Text(_)) => true,
            (Some(DocumentMediaType::PDF), DocumentData::File(data)) => carried(data),
            _ => false,
        },
    }
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
    /// The model facts the reply, from the model the interaction names, is
    /// priced by.
    #[serde(skip)]
    pub facts: crate::catalog::ModelFacts,
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
            facts: crate::catalog::ModelFacts::default(),
        }
    }

    /// The same wire, pricing its replies by `facts`.
    pub fn with_facts(mut self, facts: crate::catalog::ModelFacts) -> Self {
        self.facts = facts;
        self
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
    type Event = streaming::InteractionsEvent;
    type Decoder<'id> = streaming::InteractionsDecoder;
    type Reassembler = streaming::document::Interaction;

    /// The interaction names its own model; this wire addresses no model id.
    /// The decoder reports the model the interaction names, which the turn's
    /// origin takes.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .facts(&self.facts)
            .telemetry(|_| GenAiOperation::Chat)
            .replay(self)
    }

    /// Reads an existing interaction, so the request carries no body: what
    /// to read is the wire's own data. A request that sets options,
    /// provider options or `additional_params` is refused, since nothing
    /// would send them.
    fn encode(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        let params = crate::completion::options::request_params(
            self,
            &request,
            // `request_params` takes `additional_params.tools` (provider
            // tools included) out of the raw layer for the base to append,
            // so the base refuses them here or they would vanish.
            |input| {
                if input.raw_tools()?.is_empty() {
                    Ok(Map::new())
                } else {
                    Err(EncodeError::request(
                        "a resumed interaction takes no `additional_params.tools` or provider \
                         tools: it is read, not created",
                    ))
                }
            },
            crate::completion::options::RawAt::Top,
            &[],
        )?;
        if !params.is_empty() {
            return Err(EncodeError::request(
                "a resumed interaction takes no provider options or `additional_params`: it is \
                 read, not created",
            ));
        }
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
        streaming::InteractionsDecoder::resuming(self.interaction_id.as_str())
    }
}

impl ReplayTarget for InteractionResume {
    /// A resumed interaction is read, not created, so every set option is
    /// refused.
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap {
        crate::providers::internal::refuse_all(
            fields,
            "a resumed interaction is read, not created; its options were fixed when it was \
             created",
        )
    }

    fn facts(&self) -> Option<&crate::catalog::ModelFacts> {
        Some(&self.facts)
    }

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
        crate::providers::internal::wire_ids::legal_call_id(id, 64)
    }

    /// Gemini takes system text only in `systemInstruction`: later system
    /// messages fold into the leading one.
    fn later_system(&self, _model: &str) -> crate::completion::LaterSystem {
        crate::completion::LaterSystem::Leading
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn blob_slots(&self) -> &'static [&'static str] {
        BLOB_SLOTS
    }
}

/// The create-interaction body `request` sends on `wire`: the wire's own
/// encoding, then the mapped options, then `additional_params`, merged by
/// [`request_params`](crate::completion::options::request_params). Its
/// `generation_config` merges over the typed fields key by key, its `tools`
/// add to the request's, and an `agent` takes the place of the model.
/// System messages become the system instruction. `stream`, when set,
/// overrides the one the parameters name.
///
/// # Errors
///
/// When an option is refused, `additional_params` is not an object, its
/// `tools` are not an array, or it sets `response_format` without
/// `response_mime_type`.
pub(crate) fn create_request_body(
    wire: &Interactions,
    request: &CompletionRequest,
    stream: Option<bool>,
) -> Result<crate::completion::options::FinalBody, EncodeError> {
    let rewrites: Vec<_> = stream
        .map(crate::completion::options::Rewrite::Stream)
        .into_iter()
        .collect();
    crate::completion::options::request_params(
        wire,
        request,
        |input| create_base(wire, request, input),
        crate::completion::options::RawAt::Top,
        &rewrites,
    )
}

/// The wire's own encoding of `request`: model, input steps, system
/// instruction, tools and the typed fields in `generation_config`.
fn create_base(
    wire: &Interactions,
    request: &CompletionRequest,
    input: &mut crate::completion::options::BaseInput<'_>,
) -> Result<Map<String, Value>, EncodeError> {
    let model = request.model.clone().unwrap_or_else(|| wire.model.clone());
    let set = |key: &str| input.param(key).is_some_and(|value| !value.is_null());
    let agent = set("agent");
    if set("response_format") && !set("response_mime_type") {
        return Err(EncodeError::request(
            "response_mime_type is required when response_format is set",
        ));
    }
    let choice = request.tool_choice.clone().map(|choice| match choice {
        Choice::Auto => json!("auto"),
        Choice::None => json!("none"),
        Choice::Required => json!("any"),
        Choice::Specific { function_names } => {
            json!({ "allowed_tools": { "mode": "validated", "tools": function_names } })
        }
    });
    let config = crate::providers::internal::set_fields([
        ("temperature", request.temperature.map(Value::from)),
        ("max_output_tokens", request.max_tokens.map(Value::from)),
        ("tool_choice", choice),
    ]);
    let mut tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| json!({ "type": "function", "name": tool.name, "description": tool.description, "parameters": tool.parameters }))
        .collect();
    tools.extend(input.raw_tools()?);
    let (mut system, mut history) = (Vec::new(), Vec::new());
    for message in &request.chat_history {
        match &**message {
            Message::System { content } => system.push(content.as_str()),
            message => history.push(message),
        }
    }
    let mut body = Map::new();
    if !config.is_empty() {
        body.insert("generation_config".to_owned(), Value::Object(config));
    }
    if !tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(tools));
    }
    if !system.is_empty() {
        body.insert("system_instruction".to_owned(), json!(system.join("\n\n")));
    }
    if !agent {
        body.insert("model".to_owned(), json!(model));
    }
    body.insert(
        "input".to_owned(),
        Value::Array(steps(&history, wire, &model)?),
    );
    Ok(body)
}

/// The steps `history` sends: a user message's content grouped into
/// `user_input` steps around its function results, each a step of its own,
/// and one step per assistant block.
fn steps(
    history: &[&Message],
    target: &dyn ReplayTarget,
    model: &str,
) -> Result<Vec<Value>, EncodeError> {
    let ids = WireIds::for_target(history, target, model);
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
                        steps.push(user_input(std::mem::take(&mut run)));
                    }
                    let value = match result.content.as_slice() {
                        [ToolResultContent::Text(text)] => Value::from(text.text.as_str()),
                        // A scalar or array JSON result is wrapped as the
                        // generate wire wraps it: sent as a text block it is a
                        // multimodal response, which the models refuse.
                        [
                            ToolResultContent::Json {
                                value: value @ (Value::String(_) | Value::Object(_)),
                            },
                        ] => value.clone(),
                        [ToolResultContent::Json { value }] => json!({ "result": value }),
                        contents => Value::Array(
                            contents
                                .iter()
                                .map(result_content)
                                .collect::<Result<_, _>>()?,
                        ),
                    };
                    let mut step = Map::from_iter([
                        ("type".to_owned(), Value::from("function_result")),
                        ("name".to_owned(), Value::from(result.name.as_str())),
                        ("call_id".to_owned(), Value::from(ids.of(&result.call))),
                        ("result".to_owned(), value),
                    ]);
                    if result.is_error {
                        step.insert("is_error".to_owned(), Value::Bool(true));
                    }
                    steps.push(Value::Object(step));
                }
                if !run.is_empty() {
                    steps.push(user_input(run));
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

/// A `user_input` step of `content`.
fn user_input(content: Vec<Value>) -> Value {
    Value::Object(Map::from_iter([
        ("type".to_owned(), Value::from("user_input")),
        ("content".to_owned(), Value::Array(content)),
    ]))
}

/// A block of a tool result as a content item.
fn result_content(content: &ToolResultContent) -> Result<Value, EncodeError> {
    Ok(match content {
        ToolResultContent::Text(text) => json!({ "type": "text", "text": text.text }),
        ToolResultContent::Json { value } => json!({ "type": "text", "text": value.to_string() }),
        ToolResultContent::Image(image) => media("image", image.media_type.clone(), &image.data)?,
    })
}

/// A user part as a content item. A text document goes as text, so that RAG
/// context reads as prose.
fn user_content(part: &UserContent) -> Result<Value, EncodeError> {
    match part {
        UserContent::Text(text) => Ok(json!({ "type": "text", "text": text.text })),
        UserContent::Image(image) => media("image", image.media_type.clone(), &image.data),
        UserContent::Audio(audio) => media("audio", audio.media_type.clone(), &audio.data),
        UserContent::Video(video) => media("video", video.media_type.clone(), &video.data),
        UserContent::Document(document) => match &document.data {
            DocumentData::Text(text) => Ok(json!({ "type": "text", "text": text })),
            DocumentData::File(data) => media("document", document.media_type.clone(), data),
        },
        UserContent::ToolResult(_) => {
            Err(EncodeError::request("a tool result is a step of its own"))
        }
    }
}

/// A media content item of `kind`: a URL as its `uri`, and base64 data as its
/// `data`. [`encodes`] refuses every other form, so the adapter passes none.
fn media<M: MimeType>(
    kind: &str,
    media_type: Option<M>,
    source: &Source,
) -> Result<Value, EncodeError> {
    let unsendable = || {
        EncodeError::request(format!(
            "Gemini Interactions cannot receive this {kind} in its form"
        ))
    };
    let mime_type = media_type.ok_or_else(unsendable)?.to_mime_type().to_owned();
    let key = match super::completion::carried(source)? {
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
    let output = |content: Value| {
        Value::Object(Map::from_iter([
            ("type".to_owned(), Value::from("model_output")),
            ("content".to_owned(), Value::Array(vec![content])),
        ]))
    };
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
            "arguments": call.function.arguments,
        }),
        AssistantContent::Image(image) => {
            output(media("image", image.media_type.clone(), &image.data)?)
        }
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
mod history_tests;
