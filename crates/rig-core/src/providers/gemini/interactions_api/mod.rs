//! The [Gemini Interactions API](https://ai.google.dev/api/interactions-api)
//! through the same [`edge`] as GenerateContent. Requests are a
//! list of steps; options rig does not own are the model's
//! [`api::RequestSettings`].
//!
//! ```no_run
//! use rig_core::providers::gemini::{self, Gemini};
//! use rig_core::providers::gemini::interactions_api::api;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Gemini::from_env()?
//!     .interactions(gemini::GEMINI_3_8_FLASH)
//!     .settings(api::RequestSettings {
//!         store: Some(false),
//!         ..Default::default()
//!     });
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod api;
pub mod streaming;

use serde::Serialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use url::form_urlencoded;

use super::edge::{self, Dialect, Encoded, Media, Source, Unit};
use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::message::{MediaDetail, Message, NativePart, ToolChoice};
use crate::providers::internal::wire_ids::WireIds;
use crate::telemetry::GenAiOperation;
use crate::wire::{Descriptor, Mode};

pub use api::{Interaction, InteractionStatus, Step};

/// Gemini provider name used in normalized records and telemetry.
pub(crate) const PROVIDER_NAME: &str = super::PROVIDER_NAME;

/// Create interactions with `POST /v1beta/interactions`. Streaming mode sets
/// `alt=sse` and `stream: true`; unary mode reads the whole resource.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interactions {
    /// The key and the API root.
    pub provider: crate::providers::gemini::GeminiConfig,
    /// The model to address.
    pub model: String,
    /// Every Interactions option rig does not own.
    #[serde(default)]
    pub settings: api::RequestSettings,
}

impl Interactions {
    /// The wire for `model`.
    pub fn new(provider: crate::providers::gemini::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            settings: api::RequestSettings::default(),
        }
    }

    /// Send `settings` with every request.
    pub fn with_settings(mut self, settings: api::RequestSettings) -> Self {
        self.settings = settings;
        self
    }
}

impl<T> crate::driver::Model<Interactions, T> {
    /// Send `settings` with every request.
    pub fn settings(mut self, settings: api::RequestSettings) -> Self {
        self.wire.settings = settings;
        self
    }
}

impl crate::wire::Wire for Interactions {
    type Op = crate::operation::Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = streaming::InteractionsDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .telemetry(|mode| match mode {
                Mode::Unary => GenAiOperation::Interactions,
                Mode::Streaming => GenAiOperation::InteractionsStreaming,
            })
    }

    fn encode(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        let request = request.replayable_to(&[super::ISSUER])?;
        let streaming = matches!(mode, Mode::Streaming);
        let body = body(&self.model, request, &self.settings, streaming)?;
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

/// Read an existing interaction or resume its event stream. Unary mode
/// retrieves the resource once; streaming mode resumes after
/// `last_event_id`, or from the beginning.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionResume {
    /// The key and the API root.
    pub provider: crate::providers::gemini::GeminiConfig,
    /// The interaction to read.
    pub interaction_id: String,
    /// The last event the consumer saw, so a resumed stream does not
    /// redeliver it.
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
    type Decoder<'id> = streaming::InteractionsDecoder<'id>;

    /// The interaction names its own model.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME).telemetry(|mode| match mode {
            Mode::Unary => GenAiOperation::Interactions,
            Mode::Streaming => GenAiOperation::InteractionsStreaming,
        })
    }

    /// Reads an existing interaction: the request contributes nothing.
    fn encode(
        &self,
        _request: CompletionRequest,
        mode: Mode,
    ) -> Result<crate::wire::Encoded, EncodeError> {
        let (path, framing) = match mode {
            Mode::Unary => (
                format!("/v1beta/interactions/{}", self.interaction_id),
                crate::http_client::framing::Framing::Whole,
            ),
            Mode::Streaming => (
                format!(
                    "{}&alt=sse",
                    stream_path(&self.interaction_id, self.last_event_id.as_deref())
                ),
                crate::http_client::framing::Framing::Sse,
            ),
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

fn stream_path(interaction_id: &str, last_event_id: Option<&str>) -> String {
    let mut serializer = form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("stream", "true");
    if let Some(last_event_id) = last_event_id {
        serializer.append_pair("last_event_id", last_event_id);
    }
    format!(
        "/v1beta/interactions/{interaction_id}?{}",
        serializer.finish()
    )
}

/// The Interactions API's parts: steps, and the content items of
/// `user_input` and `model_output` steps.
pub(crate) struct InteractionsDialect;

/// One element of a request's `input`.
#[derive(Debug)]
pub(crate) enum Element {
    /// A step of its own.
    Step(api::Step),
    /// A content item of the surrounding `user_input` or `model_output`.
    Content(api::Content),
    /// A native step, sent as Gemini sent it.
    RawStep(Box<RawValue>),
    /// A native content item.
    RawContent(Box<RawValue>),
    /// Annotations of the text item before them.
    Annotations(Vec<Value>),
}

impl Serialize for Element {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Step(step) => step.serialize(serializer),
            Self::Content(content) => content.serialize(serializer),
            Self::RawStep(raw) | Self::RawContent(raw) => raw.serialize(serializer),
            Self::Annotations(annotations) => annotations.serialize(serializer),
        }
    }
}

impl Dialect for InteractionsDialect {
    const SCHEMA: &'static str = api::STEP_SCHEMA;
    type Part = Element;

    fn units(raw: &RawValue) -> Result<Vec<Unit>, serde_json::Error> {
        let kind = element_kind(raw)?;
        Ok(match kind.as_str() {
            "thought" => {
                let api::Step::Thought(api::Thought {
                    signature,
                    summary,
                    unmodeled,
                }) = serde_json::from_str(raw.get())?
                else {
                    return Ok(vec![edge::native(raw, api::STEP_SCHEMA)]);
                };
                let texts: Option<Vec<&str>> = summary
                    .iter()
                    .map(|content| match content {
                        api::Content::Text(text)
                            if text.annotations.is_empty() && text.unmodeled.is_empty() =>
                        {
                            Some(text.text.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                match texts {
                    Some(texts) if unmodeled.is_empty() => vec![Unit::Thought {
                        text: texts.concat(),
                        signature,
                    }],
                    _ => vec![edge::native(raw, api::STEP_SCHEMA)],
                }
            }
            "function_call" => {
                let api::Step::FunctionCall(call) = serde_json::from_str(raw.get())? else {
                    return Ok(vec![edge::native(raw, api::STEP_SCHEMA)]);
                };
                match (call.name, call.unmodeled.is_empty()) {
                    (Some(name), true) => vec![Unit::Call {
                        id: call.id,
                        name,
                        args: call.arguments.unwrap_or_default(),
                        signature: call.signature,
                    }],
                    _ => vec![edge::native(raw, api::STEP_SCHEMA)],
                }
            }
            "model_output" => {
                #[derive(serde::Deserialize)]
                struct Output {
                    #[serde(default)]
                    content: Vec<Box<RawValue>>,
                }
                let output: Output = serde_json::from_str(raw.get())?;
                let mut units = Vec::new();
                for content in &output.content {
                    units.extend(content_units(content)?);
                }
                units
            }
            kind if api::is_hosted(kind) => {
                // A hosted step returns re-encoded, with what a streamed step
                // leaves out restored.
                let mut step: api::HostedStep = serde_json::from_str(raw.get())?;
                step.restore();
                vec![Unit::Native(NativePart::new(
                    api::STEP_SCHEMA,
                    serde_json::value::to_raw_value(&step)?,
                ))]
            }
            "text" | "image" | "audio" | "document" | "video" => content_units(raw)?,
            _ => vec![edge::native(raw, api::STEP_SCHEMA)],
        })
    }

    fn part(unit: Unit) -> Result<Encoded<Element>, EncodeError> {
        Ok(Encoded::Typed(match unit {
            Unit::Text { text, signature } => {
                if signature.is_some() {
                    return Err(EncodeError::request(
                        "an Interactions text item cannot carry a thought signature",
                    ));
                }
                Element::Content(api::Content::Text(api::TextContent {
                    text,
                    ..Default::default()
                }))
            }
            Unit::Thought { text, signature } => Element::Step(api::Step::Thought(api::Thought {
                signature,
                summary: if text.is_empty() {
                    Vec::new()
                } else {
                    vec![api::Content::Text(api::TextContent {
                        text,
                        ..Default::default()
                    })]
                },
                ..Default::default()
            })),
            Unit::Call {
                id,
                name,
                args,
                signature,
            } => Element::Step(api::Step::FunctionCall(api::FunctionCall {
                id,
                name: Some(name),
                arguments: Some(args),
                signature,
                ..Default::default()
            })),
            Unit::Result {
                id,
                name,
                response,
                media,
            } => {
                let mut items: Vec<Value> = Vec::new();
                for media in media {
                    items.push(serde_json::to_value(media_content(media)?)?);
                }
                let result = match (response, items.is_empty()) {
                    (Some(response), true) => response.get("result").cloned().map_or(
                        Value::Object(response.clone()),
                        |result| match result {
                            // A string or object result goes as itself; any
                            // other JSON stays wrapped, as the models require.
                            value @ (Value::String(_) | Value::Object(_)) => value,
                            _ => Value::Object(response),
                        },
                    ),
                    (response, false) => {
                        if let Some(response) = response {
                            items.insert(
                                0,
                                serde_json::to_value(api::Content::Text(api::TextContent {
                                    text: Value::Object(response).to_string(),
                                    ..Default::default()
                                }))?,
                            );
                        }
                        Value::Array(items)
                    }
                    (None, true) => Value::Object(Map::new()),
                };
                Element::Step(api::Step::FunctionResult(api::FunctionResult {
                    call_id: id,
                    name: Some(name),
                    result: Some(result),
                    ..Default::default()
                }))
            }
            Unit::Media(media) => Element::Content(media_content(media)?),
            Unit::Native(native) => match native.schema.as_ref() {
                api::STEP_SCHEMA => Element::RawStep(native.part),
                api::CONTENT_SCHEMA => Element::RawContent(native.part),
                api::ANNOTATIONS_SCHEMA => {
                    Element::Annotations(serde_json::from_str(native.json())?)
                }
                other => {
                    return Err(EncodeError::request(format!(
                        "a native `{other}` part cannot be sent to Interactions"
                    )));
                }
            },
        }))
    }
}

/// A raw element's `type`.
fn element_kind(raw: &RawValue) -> Result<String, serde_json::Error> {
    #[derive(serde::Deserialize)]
    struct Kind {
        #[serde(rename = "type", default)]
        kind: String,
    }
    Ok(serde_json::from_str::<Kind>(raw.get())?.kind)
}

/// The units of one content item of a `model_output` step.
fn content_units(raw: &RawValue) -> Result<Vec<Unit>, serde_json::Error> {
    if element_kind(raw)? != "text" {
        return Ok(vec![edge::native(raw, api::CONTENT_SCHEMA)]);
    }
    let api::Content::Text(text) = serde_json::from_str(raw.get())? else {
        return Ok(vec![edge::native(raw, api::CONTENT_SCHEMA)]);
    };
    if !text.unmodeled.is_empty() {
        return Ok(vec![edge::native(raw, api::CONTENT_SCHEMA)]);
    }
    let mut units = vec![Unit::Text {
        text: text.text,
        signature: None,
    }];
    if !text.annotations.is_empty() {
        units.push(annotations(text.annotations)?);
    }
    Ok(units)
}

/// The native unit holding a text item's annotations.
pub(crate) fn annotations(annotations: Vec<Value>) -> Result<Unit, serde_json::Error> {
    Ok(Unit::Native(NativePart::new(
        api::ANNOTATIONS_SCHEMA,
        serde_json::value::to_raw_value(&annotations)?,
    )))
}

fn media_content(media: Media) -> Result<api::Content, EncodeError> {
    let mime_type = media
        .mime_type
        .ok_or_else(|| EncodeError::request("Gemini Interactions media needs a media type"))?;
    let (data, uri) = match media.source {
        Source::Inline(data) => (Some(data), None),
        Source::Uri(uri) => (None, Some(uri)),
    };
    let resolution = media.detail.and_then(|detail| match detail {
        MediaDetail::Auto => None,
        MediaDetail::Low => Some("low"),
        MediaDetail::Medium => Some("medium"),
        MediaDetail::High => Some("high"),
        MediaDetail::UltraHigh => Some("ultra_high"),
    });
    let item = api::MediaContent {
        data,
        uri,
        resolution: resolution.map(str::to_owned),
        ..Default::default()
    };
    Ok(match mime_type.split('/').next().unwrap_or_default() {
        "image" => api::Content::Image(api::MediaContent {
            mime_type: Some(mime_type),
            ..item
        }),
        "audio" => api::Content::Audio(api::MediaContent {
            mime_type: Some(mime_type),
            ..item
        }),
        "video" => api::Content::Video(api::MediaContent {
            mime_type: Some(mime_type),
            ..item
        }),
        _ => api::Content::Document(api::MediaContent {
            mime_type: Some(mime_type),
            ..item
        }),
    })
}

/// A request body: rig's fields, then the model's settings.
#[derive(Debug, Serialize)]
pub(crate) struct Body {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    input: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<api::GenerationConfig>,
    stream: bool,
    #[serde(flatten)]
    settings: api::RequestSettings,
}

/// The step list and body for `request` on `model` with `settings`.
pub(crate) fn body(
    model: &str,
    request: CompletionRequest,
    settings: &api::RequestSettings,
    stream: bool,
) -> Result<Body, EncodeError> {
    if request.additional_params.is_some() {
        return Err(EncodeError::request(
            "additional_params is not read by Gemini; use Interactions::settings",
        ));
    }
    let history = request
        .chat_history_with_documents()
        .into_iter()
        .collect::<Vec<_>>();
    let CompletionRequest {
        tools,
        temperature,
        max_tokens,
        tool_choice,
        output_schema,
        ..
    } = request;
    if output_schema.is_some() {
        return Err(EncodeError::request(
            "Interactions takes an output schema through settings.response_format",
        ));
    }

    let ids = WireIds::new(&history);
    let mut system = Vec::new();
    let mut input = Input::default();
    for (position, message) in history.into_iter().enumerate() {
        match message {
            Message::System { content } => system.push(content),
            Message::User { content } => {
                input.open("user_input");
                for (index, content) in content.into_iter().enumerate() {
                    let unit = with_id(edge::user_unit(content)?, ids.get(position, index));
                    input.push(InteractionsDialect::part(unit)?)?;
                }
            }
            Message::Assistant { content, .. } => {
                input.open("model_output");
                for (index, content) in content.into_iter().enumerate() {
                    let id = ids.get(position, index);
                    for unit in edge::assistant_units(content, &super::ISSUER)? {
                        input.push(InteractionsDialect::part(with_id(unit, id))?)?;
                    }
                }
            }
        }
    }

    let mut settings = settings.clone();
    let mut hosted = std::mem::take(&mut settings.tools);
    let mut tool_list: Vec<Value> = tools
        .into_iter()
        .map(|tool| {
            serde_json::to_value(api::FunctionTool {
                kind: "function".to_owned(),
                name: tool.name,
                description: tool.description,
                parameters: (!tool.parameters.is_null()).then_some(tool.parameters),
            })
        })
        .collect::<Result<_, _>>()?;
    for tool in hosted.drain(..) {
        tool_list.push(serde_json::to_value(tool)?);
    }

    let generation = api::GenerationConfig {
        temperature,
        max_output_tokens: max_tokens,
        tool_choice: tool_choice.map(tool_choice_value),
        settings: std::mem::take(&mut settings.generation_config),
    };
    let generation_config = (generation != api::GenerationConfig::default()).then_some(generation);

    Ok(Body {
        model: settings.agent.is_none().then(|| model.to_owned()),
        input: input.finish()?,
        system_instruction: (!system.is_empty()).then(|| system.join("\n\n")),
        tools: tool_list,
        generation_config,
        stream,
        settings,
    })
}

/// A call or result unit with the id the wire spells for it.
fn with_id(unit: Unit, id: Option<&str>) -> Unit {
    let id = id.map(str::to_owned);
    match unit {
        Unit::Call {
            name,
            args,
            signature,
            ..
        } => Unit::Call {
            id,
            name,
            args,
            signature,
        },
        Unit::Result {
            name,
            response,
            media,
            ..
        } => Unit::Result {
            id,
            name,
            response,
            media,
        },
        unit => unit,
    }
}

fn tool_choice_value(choice: ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Required => Value::String("any".to_owned()),
        ToolChoice::Specific { function_names } => serde_json::json!({
            "allowed_tools": {"mode": "validated", "tools": function_names}
        }),
    }
}

/// The `input` step list under construction: content items gather into the
/// open `user_input` or `model_output` step.
#[derive(Default)]
struct Input {
    steps: Vec<Value>,
    group: Option<(&'static str, Vec<Value>)>,
}

impl Input {
    /// Start gathering content items of `kind`.
    fn open(&mut self, kind: &'static str) {
        self.close();
        self.group = Some((kind, Vec::new()));
    }

    fn close(&mut self) {
        if let Some((kind, content)) = self.group.take()
            && !content.is_empty()
        {
            self.steps
                .push(serde_json::json!({"type": kind, "content": content}));
        }
    }

    fn push(&mut self, element: Encoded<Element>) -> Result<(), EncodeError> {
        let Encoded::Typed(element) = element else {
            return Err(EncodeError::request(
                "an Interactions part must be an element",
            ));
        };
        match element {
            Element::Content(_) | Element::RawContent(_) => {
                let value = serde_json::to_value(&element)?;
                match &mut self.group {
                    Some((_, content)) => content.push(value),
                    None => self.group = Some(("user_input", vec![value])),
                }
            }
            Element::Annotations(annotations) => {
                let last = self
                    .group
                    .as_mut()
                    .and_then(|(_, content)| content.last_mut())
                    .and_then(Value::as_object_mut)
                    .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"));
                let Some(item) = last else {
                    return Err(EncodeError::request(
                        "Interactions annotations must follow their text",
                    ));
                };
                item.insert("annotations".to_owned(), Value::Array(annotations));
            }
            Element::Step(_) | Element::RawStep(_) => {
                let kind = self.group.as_ref().map(|(kind, _)| *kind);
                self.close();
                self.steps.push(serde_json::to_value(&element)?);
                if let Some(kind) = kind {
                    self.group = Some((kind, Vec::new()));
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Value>, EncodeError> {
        self.close();
        Ok(self.steps)
    }
}

#[cfg(test)]
mod tests;
