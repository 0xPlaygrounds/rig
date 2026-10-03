//! The [Gemini GenerateContent API](https://ai.google.dev/api/generate-content)
//! completion wire. Its request is built as REST JSON by [`request_body`],
//! which the Vertex AI and gRPC wires transcode, and its replies are read by
//! [`GenerateContentDecoder`](super::streaming::GenerateContentDecoder).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, completion::GEMINI_2_5_FLASH};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Gemini::from_env()?.completion(GEMINI_2_5_FLASH);
//! # Ok(())
//! # }
//! ```

/// `gemini-3.8-flash` completion model
pub const GEMINI_3_8_FLASH: &str = "gemini-3.8-flash";
/// `gemini-3.1-flash-lite-preview` completion model
pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
/// `gemini-3-flash-preview` completion model
pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use serde_json::{Map, Value, json};

pub use super::cached_content::with_cached_content;
use crate::completion::{Accepts, CompletionRequest, Media, Place, Replay, ReplayTarget};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, DocumentMediaType, DocumentSourceKind as Source, Message, MimeType,
    ToolChoice, ToolResultContent, UserContent,
};
use crate::operation::Completion;
use crate::providers::internal::wire_ids::WireIds;
use crate::telemetry::GenAiOperation;
use crate::wire::{Body, Descriptor, Encoded, Framing, Mode, Wire};

/// Provider name used in normalized responses, streams, and telemetry.
pub const PROVIDER_NAME: &str = "gcp.gemini";

/// Completion wire for unary `generateContent` and SSE `streamGenerateContent`.
/// Both modes use [`GenerateContentDecoder`](super::streaming::GenerateContentDecoder).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenerateContent {
    /// The key and the API root.
    pub provider: super::GeminiConfig,
    /// The model to address, e.g. [`GEMINI_2_5_FLASH`].
    pub model: String,
    /// Handle of a `cachedContents` resource every request reads its prefix
    /// from. See [`Self::with_cached_content`].
    pub cached_content: Option<String>,
    /// Which thought signatures requests re-send. See [`ThoughtReplay`].
    #[serde(default)]
    pub thought_replay: ThoughtReplay,
}

/// Which earlier thought signatures a request re-sends.
///
/// On Gemini 3, every thought signature a request carries is expanded back
/// into the reasoning it came from, and that reasoning is billed again as
/// input on every later call. No explicit cache holds it. Measured on
/// gemini-3.8-flash: a history whose earlier turns carried their signatures
/// cost 6,708 prompt tokens; the same history without them cost 2,681, and
/// Gemini answered normally. Over a 30-turn chat, dropping them cut input by
/// 42%, and by 83% together with automatic caching.
///
/// Google's thinking guide says to send thought signatures back exactly as
/// received. [`ThoughtReplay::CurrentTurn`] departs from that for turns the
/// model has finished; its effect on answer quality is unmeasured. Hence
/// [`ThoughtReplay::All`] is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ThoughtReplay {
    /// Re-send every signature, as Google's guidance asks.
    #[default]
    All,
    /// Re-send only the signatures at or after the newest user message: the
    /// current turn, whose function calls Gemini validates. Parts before it
    /// go out without their signature. The history itself is unchanged.
    CurrentTurn,
}

impl GenerateContent {
    /// The wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            cached_content: None,
            thought_replay: ThoughtReplay::All,
        }
    }

    /// Re-send thought signatures as `replay` says. See [`ThoughtReplay`].
    pub fn thought_replay(mut self, replay: ThoughtReplay) -> Self {
        self.thought_replay = replay;
        self
    }

    /// Use an explicit `cachedContents/<id>` handle as every request's prefix.
    /// Encoding rejects requests with their own system instruction, tools, or
    /// tool choice. See [`crate::providers::gemini::cached_content`] for cache ownership.
    pub fn with_cached_content(mut self, name: impl Into<String>) -> Self {
        self.cached_content = Some(name.into());
        self
    }
}

impl<T> crate::driver::Model<GenerateContent, T> {
    /// This model, re-sending thought signatures as `replay` says. See
    /// [`ThoughtReplay`]. Call it before [`caching`](Self::caching), which
    /// reads it.
    pub fn thought_replay(mut self, replay: ThoughtReplay) -> Self {
        self.wire.thought_replay = replay;
        self
    }
}

/// Remove thought signatures from every part before the newest user text
/// content: the turns the model has finished. Only the signature keys go;
/// every other field is unchanged.
fn drop_finished_signatures(contents: &mut [Value]) {
    let current = contents.iter().rposition(|content| {
        let parts = content.arr("parts");
        content.str("role") == Some("user")
            && parts.iter().any(|part| part.get("text").is_some())
            && !parts
                .iter()
                .any(|part| part.get("functionResponse").is_some())
    });
    for content in contents.iter_mut().take(current.unwrap_or(0)) {
        let parts = content.get_mut("parts").and_then(Value::as_array_mut);
        for part in parts.into_iter().flatten().filter_map(Value::as_object_mut) {
            part.shift_remove("thoughtSignature");
        }
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = super::streaming::GenerateContentDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .telemetry(|mode| match mode {
                Mode::Unary => GenAiOperation::GenerateContent,
                Mode::Streaming => GenAiOperation::ChatStreaming,
            })
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        // The request may name a model of its own; the wire's is the default.
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let mut body = request_body(request, self, &model)?;
        if let Some(name) = self.cached_content.as_deref() {
            with_cached_content(&mut body, name)?;
        }
        if let (ThoughtReplay::CurrentTurn, Some(Value::Array(contents))) =
            (self.thought_replay, body.get_mut("contents"))
        {
            drop_finished_signatures(contents);
        }
        use crate::providers::internal::LogTarget;
        // `alt=sse` is what makes the streamed reply an event stream rather
        // than a JSON array of the same chunks.
        let (verb, framing, target) = match mode {
            Mode::Unary => ("generateContent", Framing::Whole, LogTarget::Completions),
            Mode::Streaming => (
                "streamGenerateContent?alt=sse",
                Framing::Sse,
                LogTarget::Streaming,
            ),
        };
        crate::providers::internal::trace_json(target, "Gemini completion request", &body);
        let uri = self.provider.uri(&format!("/v1beta/models/{model}:{verb}"));
        let request = http::Request::post(uri)
            .header("Content-Type", "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        // Gemini supplies no transport request-id response header.
        Ok(Encoded::new(request, framing)
            .with_projection(super::streaming::GenerateContentDecoder::project)
            .with_analysis_only(super::streaming::GenerateContentDecoder::is_analysis_only))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        super::streaming::GenerateContentDecoder::default()
    }
}

impl ReplayTarget for GenerateContent {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("gemini.generate_content")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn accepts(&self, model: &str) -> Accepts {
        accepts(model)
    }

    fn encodes(&self, _model: &str, media: Media<'_>) -> bool {
        encodes(media, false)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&crate::message::Origin>,
    ) -> String {
        normalize_tool_call_id(model, id)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        CALL_ID_SLOT
    }

    fn declares_tools(&self, request: &crate::completion::CompletionRequest) -> bool {
        self.cached_content.is_some() || declares_tools(request)
    }
}

/// Whether `request` declares tools on a GenerateContent wire: in `tools`,
/// in the `tools` of its `additional_params`, or through a `cachedContent`
/// handle there, since a cache holds its function declarations.
pub fn declares_tools(request: &crate::completion::CompletionRequest) -> bool {
    let params = request.additional_params.as_ref();
    !request.tools.is_empty()
        || params.is_some_and(|params| !params.arr("tools").is_empty())
        || params
            .and_then(Value::as_object)
            .is_some_and(|params| present(params, &CACHED_CONTENT).is_some())
}

/// Where a GenerateContent part holds its call's id, on every wire that
/// speaks it.
pub const CALL_ID_SLOT: Option<&str> = Some("/functionCall/id");

/// The major version of a Gemini model id, read past a `models/` prefix:
/// `gemini-<major>…` or `gemini-live-<major>…`, or `None` for an id that is
/// not Gemini's. A Gemini id that names no version, such as the alias
/// `gemini-flash-latest`, is the current generation (`u32::MAX`), so every
/// rule reads an alias as the newest model.
fn gemini_major(model: &str) -> Option<u32> {
    let model = model.to_ascii_lowercase();
    let model = model.strip_prefix("models/").unwrap_or(&model);
    let rest = model.strip_prefix("gemini-")?;
    let rest = rest.strip_prefix("live-").unwrap_or(rest);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    Some(
        rest.get(..end)
            .and_then(|major| major.parse().ok())
            .unwrap_or(u32::MAX),
    )
}

/// Whether `model` is Gemini 3 or later, which validates the thought
/// signatures of function calls. A Gemini alias counts as current.
pub(super) fn gemini_3_or_later(model: &str) -> bool {
    gemini_major(model).is_some_and(|major| major >= 3)
}

/// Whether `model` takes function-call ids on its calls and their
/// responses: Claude, gpt-oss, and Gemini 3 or later (pi's rule).
pub fn requires_tool_call_id(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.starts_with("claude-") || model.starts_with("gpt-oss-") || gemini_3_or_later(&model)
}

/// What `model` reads on every Gemini wire (REST, Vertex AI, gRPC and
/// Interactions): images in every role, and images inside function
/// responses only from Gemini 3 on. A Gemini alias reads them as the
/// current generation, and a model that is not Gemini (Claude behind
/// Vertex AI) is assumed to read them, as pi assumes.
pub fn accepts(model: &str) -> Accepts {
    Accepts {
        tool_result_images: gemini_major(model).is_none_or(|major| major >= 3),
        ..Accepts::ALL
    }
}

/// Whether a GenerateContent wire takes `media`: data or a URL of a media
/// type Gemini reads. A YouTube video needs no media type, a file id is
/// never taken, and a text document's data is left to the adapter, which
/// sends its text. A function response takes image data, and a URL only when
/// `response_files`: Vertex AI declares `fileData` there and the Gemini API
/// does not.
pub fn encodes(media: Media<'_>, response_files: bool) -> bool {
    match media {
        Media::Image(image, place) => {
            reads_image(image.media_type.as_ref(), place)
                && match image.data {
                    Source::Base64(_) | Source::String(_) => true,
                    Source::Url(_) => place != Place::ToolResult || response_files,
                    _ => false,
                }
        }
        Media::Audio(audio) => {
            audio.media_type.is_some() && matches!(audio.data, Source::Url(_) | Source::Base64(_))
        }
        Media::Video(video) => match &video.data {
            Source::Url(url) if url.starts_with("https://www.youtube.com") => true,
            data => {
                video.media_type.is_some() && matches!(data, Source::Url(_) | Source::Base64(_))
            }
        },
        Media::Document(document) => match (&document.media_type, &document.data) {
            (None, _) => false,
            (Some(_), Source::Url(_) | Source::String(_)) => true,
            (Some(media_type), Source::Base64(_)) => *media_type == DocumentMediaType::PDF,
            (Some(_), _) => false,
        },
    }
}

/// Whether Gemini reads an image of `media_type` at `place`: JPEG, PNG,
/// WEBP, HEIC or HEIF, and only the first three in a function response.
pub(crate) fn reads_image(
    media_type: Option<&crate::message::ImageMediaType>,
    place: Place,
) -> bool {
    use crate::message::ImageMediaType::{HEIC, HEIF, JPEG, PNG, WEBP};
    match place {
        Place::ToolResult => matches!(media_type, Some(JPEG | PNG | WEBP)),
        _ => matches!(media_type, Some(JPEG | PNG | WEBP | HEIC | HEIF)),
    }
}

/// `id` as `model` takes another model's call id: when the model takes ids
/// at all, characters outside `[a-zA-Z0-9_-]` become `_` and it keeps at
/// most 64 of them.
pub fn normalize_tool_call_id(model: &str, id: &str) -> String {
    if !requires_tool_call_id(model) {
        return id.to_owned();
    }
    id.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => c,
            _ => '_',
        })
        .take(64)
        .collect()
}

/// The REST `GenerateContentRequest` body `request` sends to `model` on
/// `target`, a GenerateContent wire. The REST wire sends it; the Vertex AI
/// and gRPC wires transcode it into their SDK and protobuf requests.
///
/// System messages become the system instruction, and the rest the
/// contents. `additional_params` is merged into the body: its `tools` add
/// to the request's, its `generationConfig` is the base the typed fields
/// override, and a `cachedContent` handle is checked as
/// [`GenerateContent::with_cached_content`] checks one. The fields Gemini's
/// request shape leaves unset are sent as `null`, as recorded requests
/// spell them.
///
/// # Errors
///
/// When `additional_params` is malformed, a tool's parameters are no
/// schema Gemini reads, or a field is set two ways.
pub fn request_body(
    request: CompletionRequest,
    target: &dyn ReplayTarget,
    model: &str,
) -> Result<Map<String, Value>, EncodeError> {
    let mut params = match request.additional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(other) => return Err(invalid("additional_params", "an object", &other)),
    };
    let extra_tools = match params.shift_remove("tools") {
        None => Vec::new(),
        Some(Value::Array(tools)) => tools,
        Some(other) => return Err(invalid("additional_params.tools", "a list", &other)),
    };
    let mut handles = Vec::new();
    for spelling in CACHED_CONTENT {
        match params.shift_remove(spelling) {
            None => {}
            Some(Value::String(name)) => handles.push(name),
            Some(other) => {
                return Err(invalid(
                    &format!("additional_params.{spelling}"),
                    "a string",
                    &other,
                ));
            }
        }
    }
    let mut config = match params.shift_remove("generationConfig") {
        None | Some(Value::Null) => None,
        Some(Value::Object(config)) => Some(config),
        Some(other) => return Err(invalid("generationConfig", "an object", &other)),
    };
    let schema = request.output_schema.map(|schema| schema.to_value());
    let mime = schema.as_ref().map(|_| json!("application/json"));
    let (temperature, max_tokens) = (
        request.temperature.map(Value::from),
        request.max_tokens.map(Value::from),
    );
    let typed = [
        ("responseMimeType", mime),
        ("responseJsonSchema", schema),
        ("temperature", temperature),
    ];
    for (key, value) in typed.into_iter().chain([("maxOutputTokens", max_tokens)]) {
        if let Some(value) = value {
            config.get_or_insert_default().insert(key.to_owned(), value);
        }
    }
    let (mut system, mut history) = (Vec::new(), Vec::new());
    for message in request.chat_history {
        match message {
            Message::System { content } if content.is_empty() => {}
            Message::System { content } => system.push(text_part(content)),
            message => history.push(message),
        }
    }
    // Gemini rejects a field set twice, and merges a tool choice set twice,
    // which unions the allowed functions.
    let preamble = "system instruction twice: as a preamble or system message";
    let choice = "tool choice twice: as `tool_choice`, which Gemini merges,";
    let twice = [
        (!system.is_empty(), &SYSTEM_INSTRUCTION, preamble),
        (request.tool_choice.is_some(), &TOOL_CONFIG, choice),
    ];
    for (set, spellings, what) in twice {
        if let (true, Some(spelling)) = (set, present(&params, spellings)) {
            return Err(EncodeError::request(format!(
                "a Gemini request set the {what} and as `additional_params.{spelling}`. Set it \
                 one way or the other"
            )));
        }
    }
    let declared = request
        .tools
        .into_iter()
        .map(declaration)
        .collect::<Result<Vec<_>, _>>()?;
    let declared = (!declared.is_empty())
        .then(|| json!({ "functionDeclarations": declared, "codeExecution": null }));
    let tools: Vec<Value> = declared.into_iter().chain(extra_tools).collect();
    let contents = Value::Array(contents(history, target, model)?);
    let system = (!system.is_empty()).then(|| json!({ "parts": system, "role": "model" }));
    let tool_config = request.tool_choice.map_or(Value::Null, calling_config);
    let mut body = object([
        ("contents", Some(contents)),
        (
            "generationConfig",
            Some(config.map_or(Value::Null, Value::Object)),
        ),
        ("safetySettings", Some(Value::Null)),
        ("toolConfig", Some(tool_config)),
        ("systemInstruction", Some(system.unwrap_or(Value::Null))),
        ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
    ]);
    body.extend(params);
    for name in handles {
        with_cached_content(&mut body, &name)?;
    }
    Ok(body)
}

/// A tool as its function declaration.
fn declaration(tool: crate::completion::ToolDefinition) -> Result<Value, EncodeError> {
    let parameters = tool_parameters_to_schema(tool.parameters).map_err(|error| {
        let reason = std::error::Error::source(&error)
            .map_or_else(|| error.to_string(), ToString::to_string);
        EncodeError::request(format!(
            "Tool '{}' could not be converted to a schema: {reason}",
            tool.name
        ))
    })?;
    let (name, description) = (Some(json!(tool.name)), Some(json!(tool.description)));
    Ok(Value::Object(object([
        ("name", name),
        ("description", description),
        ("parameters", parameters),
    ])))
}

/// An object of the entries that are set.
fn object<const N: usize>(entries: [(&str, Option<Value>); N]) -> Map<String, Value> {
    let entries = entries.into_iter();
    entries
        .filter_map(|(key, value)| Some((key.to_owned(), value?)))
        .collect()
}

fn invalid(field: &str, expected: &str, got: &Value) -> EncodeError {
    EncodeError::request(format!("Gemini `{field}` should be {expected}, got {got}"))
}

/// Proto3 JSON accepts both lowerCamelCase and original proto field names.
pub(super) const SYSTEM_INSTRUCTION: [&str; 2] = ["systemInstruction", "system_instruction"];
pub(super) const TOOL_CONFIG: [&str; 2] = ["toolConfig", "tool_config"];
const CACHED_CONTENT: [&str; 2] = ["cachedContent", "cached_content"];

/// The first of `spellings` set to a value other than `null` in `body`.
pub(super) fn present<'a>(body: &Map<String, Value>, spellings: &[&'a str]) -> Option<&'a str> {
    let set = |spelling: &&&str| body.get(**spelling).is_some_and(|value| !value.is_null());
    spellings.iter().find(set).copied()
}

/// The `functionCallingConfig` of `choice`.
fn calling_config(choice: ToolChoice) -> Value {
    let (mode, names) = match choice {
        ToolChoice::Auto => ("AUTO", None),
        ToolChoice::None => ("NONE", None),
        ToolChoice::Required => ("ANY", None),
        ToolChoice::Specific { function_names } => ("ANY", Some(json!(function_names))),
    };
    let config = object([
        ("mode", Some(json!(mode))),
        ("allowed_function_names", names),
    ]);
    json!({ "functionCallingConfig": config })
}

/// The Gemini contents for `history`, a history without system messages,
/// sent to `model` on `target`. A user message's function responses and
/// its other parts go in contents of their own, in order, as pi sends
/// them: Gemini answers text that shares a content with function responses
/// poorly. Calls and their responses carry the ids [`WireIds`] spells when
/// `model` takes ids.
fn contents(
    history: Vec<Message>,
    target: &dyn ReplayTarget,
    model: &str,
) -> Result<Vec<Value>, EncodeError> {
    let ids = WireIds::for_target(&history, target, model);
    let with_ids = requires_tool_call_id(model);
    let mut contents = Vec::with_capacity(history.len());
    let mut push = |role: &str, parts: Vec<Value>| {
        // Gemini rejects a content with no parts, as pi skips one.
        if !parts.is_empty() {
            contents.push(json!({ "parts": parts, "role": role }));
        }
    };
    for message in history {
        match message {
            Message::System { content } => push("user", vec![text_part(content)]),
            Message::User { content } => {
                let mut run = Vec::new();
                let mut responses = false;
                for part in content {
                    let id = match &part {
                        UserContent::ToolResult(result) => {
                            Some(ids.of(&result.call).filter(|_| with_ids))
                        }
                        _ => None,
                    };
                    if id.is_some() != responses {
                        push("user", std::mem::take(&mut run));
                    }
                    responses = id.is_some();
                    run.push(user_part(part, id.flatten())?);
                }
                push("user", run);
            }
            Message::Assistant(turn) => {
                let parts = turn
                    .content
                    .iter()
                    .map(|block| assistant_part(block, target, &ids, model));
                push(
                    "model",
                    parts
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                );
            }
        }
    }
    Ok(contents)
}

fn text_part(text: String) -> Value {
    json!({ "text": text, "thought": false })
}

fn mime<M: MimeType>(media_type: Option<M>) -> Option<String> {
    media_type.map(|media_type| media_type.to_mime_type().to_owned())
}

/// Where `source` puts media on every Gemini wire, REST and Interactions
/// alike: a URL by reference (`true`) and base64 data inline. A string is
/// inline as it stands when `verbatim`, else as the base64 of its bytes.
/// Each wire's `encodes` refuses every other form, so the adapter passes
/// none.
pub(super) fn carried(source: Source, verbatim: bool) -> Result<(bool, String), EncodeError> {
    use base64::Engine as _;
    match source {
        Source::Url(uri) => Ok((true, uri)),
        Source::Base64(data) => Ok((false, data)),
        Source::String(data) if verbatim => Ok((false, data)),
        Source::String(data) => Ok((false, base64::prelude::BASE64_STANDARD.encode(data))),
        _ => Err(EncodeError::request(
            "Gemini cannot receive this media in its form",
        )),
    }
}

/// `source` of `mime_type` as GenerateContent part data, file data by URI
/// or inline data, which needs a media type; a whole `part` is marked as no
/// thought.
fn media(
    mime_type: Option<String>,
    source: Source,
    string_is_data: bool,
    part: bool,
) -> Result<Value, EncodeError> {
    let (uri, data) = carried(source, string_is_data)?;
    let data = match (uri, mime_type) {
        (true, mime) => ("fileData", json!({ "mimeType": mime, "fileUri": data })),
        (false, Some(mime)) => ("inlineData", json!({ "mimeType": mime, "data": data })),
        (false, None) => {
            return Err(EncodeError::request(
                "Gemini cannot receive media without its type",
            ));
        }
    };
    Ok(Value::Object(object([
        (data.0, Some(data.1)),
        ("thought", part.then_some(Value::Bool(false))),
    ])))
}

/// A user part as Gemini takes it. A function response carries `id`, its
/// call's wire spelling, when the model takes ids, and the error key when
/// the tool failed, as the Gemini SDKs spell a function's failure. Images
/// the adapter leaves in a result (a model that reads them) go in its
/// `parts`, in order.
fn user_part(part: UserContent, id: Option<&str>) -> Result<Value, EncodeError> {
    Ok(match part {
        UserContent::Text(text) => text_part(text.text),
        UserContent::ToolResult(result) => {
            let (mut values, mut parts) = (Vec::new(), Vec::new());
            for item in result.content {
                match item {
                    ToolResultContent::Text(text) => values.push(Value::String(text.text)),
                    ToolResultContent::Json { value } => values.push(value),
                    ToolResultContent::Image(image) => {
                        parts.push(media(mime(image.media_type), image.data, true, false)?);
                    }
                }
            }
            let key = if result.is_error { "error" } else { "result" };
            let value = match values.len() {
                0 => None,
                1 => values.pop(),
                _ => Some(Value::Array(values)),
            };
            let response = object([
                ("name", Some(json!(result.name))),
                ("id", id.map(Value::from)),
                ("response", value.map(|value| json!({ key: value }))),
                ("parts", (!parts.is_empty()).then_some(Value::Array(parts))),
            ]);
            json!({ "functionResponse": response, "thought": false })
        }
        UserContent::Image(image) => media(mime(image.media_type), image.data, true, true)?,
        // A text document goes as text, so that RAG context reads as prose.
        UserContent::Document(document) => match (document.media_type, document.data) {
            (Some(media_type), Source::String(text)) if media_type != DocumentMediaType::PDF => {
                text_part(text)
            }
            (media_type, data) => media(mime(media_type), data, true, true)?,
        },
        UserContent::Audio(audio) => media(mime(audio.media_type), audio.data, false, true)?,
        UserContent::Video(video) => {
            let mut part = media(mime(video.media_type), video.data, false, true)?;
            if let (Some(Value::Object(extra)), Some(part)) =
                (video.additional_params, part.as_object_mut())
            {
                part.extend(extra);
            }
            part
        }
    })
}

/// One assistant block as a Gemini part: its provider item while that is
/// current, else a part rebuilt from its canonical fields, or `None` when
/// it has nothing to send. Calls carry ids only when `model` takes them. A
/// rebuilt call on Gemini 3 carries Google's placeholder signature, since
/// Gemini 3 rejects a call it did not sign without one ("Function call is
/// missing a thought_signature in functionCall parts").
fn assistant_part(
    block: &AssistantContent,
    target: &dyn ReplayTarget,
    ids: &WireIds,
    model: &str,
) -> Result<Option<Value>, EncodeError> {
    let with_ids = requires_tool_call_id(model);
    if let Replay::Item(item) = block.replay(target, ids) {
        let mut item = item.into_owned();
        if let Some(part) = item.as_object_mut() {
            if let (false, Some(Value::Object(call))) = (with_ids, part.get_mut("functionCall")) {
                call.shift_remove("id");
            }
            // Gemini rejects the whole request over a signature that is
            // not base64 ("Base64 decoding failed").
            if part
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .is_some_and(|signature| !is_base64(signature))
            {
                part.shift_remove("thoughtSignature");
            }
        }
        return Ok(Some(item));
    }
    Ok(Some(match block {
        AssistantContent::Text(text) => json!({ "text": text.text }),
        AssistantContent::Reasoning(reasoning)
            if reasoning.redacted || reasoning.text.trim().is_empty() =>
        {
            return Ok(None);
        }
        AssistantContent::Reasoning(reasoning) => {
            json!({ "thought": true, "text": reasoning.text })
        }
        AssistantContent::ToolCall(call) => {
            let id = ids.of(&call.id).filter(|_| with_ids).map(Value::from);
            let name = Some(json!(call.function.name));
            let args = Some(json!(call.function.arguments));
            let signature =
                gemini_3_or_later(model).then(|| json!("skip_thought_signature_validator"));
            let call = Value::Object(object([("name", name), ("args", args), ("id", id)]));
            Value::Object(object([
                ("functionCall", Some(call)),
                ("thoughtSignature", signature),
            ]))
        }
        AssistantContent::Image(image) => media(
            mime(image.media_type.clone()),
            image.data.clone(),
            true,
            true,
        )?,
        AssistantContent::Opaque(opaque) => opaque.item.clone(),
    }))
}

/// Whether `text` is padded standard base64.
fn is_base64(text: &str) -> bool {
    let body = text.trim_end_matches('=');
    let alphabet = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/');
    !body.is_empty()
        && text.len().is_multiple_of(4)
        && text.len() - body.len() <= 2
        && body.bytes().all(alphabet)
}

/// Convert a specified prompt block reason into a provider error with its
/// safety ratings. `feedback` is the reply's `promptFeedback` JSON, read
/// leniently. Content refusals are final; `OTHER` and unknown reasons are
/// transient. The zero value names no block: Gemini spells it
/// `BLOCK_REASON_`, Vertex AI `BLOCKED_REASON_`.
pub(crate) fn blocked_prompt_error(feedback: &Value) -> Option<ProviderError> {
    let reason = match feedback.get("blockReason")? {
        Value::String(reason) => reason.clone(),
        Value::Number(number) => format!("BLOCK_REASON_{number}"),
        _ => return None,
    };
    if matches!(
        reason.as_str(),
        "BLOCK_REASON_UNSPECIFIED" | "BLOCKED_REASON_UNSPECIFIED"
    ) {
        return None;
    }
    let spelled = |value: Option<&Value>| match value {
        Some(Value::String(name)) => name.clone(),
        Some(other) => other.to_string(),
        None => "<unset>".to_owned(),
    };
    let ratings = feedback.arr("safetyRatings").iter();
    let ratings = ratings.map(|rating| {
        format!(
            "{}={}",
            spelled(rating.get("category")),
            spelled(rating.get("probability"))
        )
    });
    let ratings: Vec<String> = ratings.collect();
    let ratings = match ratings.is_empty() {
        true => String::new(),
        false => format!(", safety_ratings=[{}]", ratings.join(", ")),
    };
    let message = format!("Gemini blocked the prompt: block_reason={reason}{ratings}");
    let error = crate::provider_response::ProviderResponseError::without_status(message)
        .with_code(Some(reason.clone()));
    Some(ProviderError::ProviderResponse(match reason.as_str() {
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY" | "MODEL_ARMOR"
        | "JAILBREAK" => error.with_refusal(true),
        _ => error.with_transient(Some(true)),
    }))
}

/// `parameters`, a tool's JSON Schema, as the schema Gemini's `parameters`
/// reads, or `None` for a tool that takes no arguments.
///
/// # Errors
///
/// When the schema is not an object or a reference does not resolve.
pub fn tool_parameters_to_schema(parameters: Value) -> Result<Option<Value>, EncodeError> {
    if parameters.is_null() || parameters == json!({"type": "object", "properties": {}}) {
        return Ok(None);
    }
    schema(parameters).map(Some)
}

/// `value`, a JSON Schema, as the OpenAPI subset Gemini reads: references
/// inlined, the type inferred from a composition or the keys present, a
/// union with `null` as `nullable`, and an array without `items` given
/// string items, which Gemini requires.
fn schema(value: Value) -> Result<Value, EncodeError> {
    const COMPOSITIONS: [&str; 3] = ["anyOf", "oneOf", "allOf"];
    let own = |schema: &Value| match schema.get("type") {
        Some(Value::String(name)) => Some(name.clone()),
        Some(Value::Array(names)) => {
            let mut names = names.iter().filter_map(Value::as_str);
            let first = names.clone().next();
            names
                .find(|name| *name != "null")
                .or(first)
                .map(str::to_owned)
        }
        _ => None,
    };
    let is_null = |schema: &Value| schema.is_object() && own(schema).as_deref() == Some("null");
    let shape = |schema: &Value| match (schema.get("properties"), schema.get("enum")) {
        (Some(_), _) => Some("object".to_owned()),
        (None, Some(_)) => Some("string".to_owned()),
        (None, None) => None,
    };
    let nullable = |schema: &Value| {
        schema.bool("nullable") == Some(true)
            || schema.arr("type").contains(&json!("null"))
            || COMPOSITIONS
                .iter()
                .any(|key| schema.arr(key).iter().any(is_null))
    };
    let value = flatten_schema(value)?;
    if !value.is_object() {
        return Err(EncodeError::request("Expected a JSON object for Schema"));
    }
    let alternatives = COMPOSITIONS.iter().flat_map(|key| value.arr(key));
    let alternatives: Vec<&Value> = alternatives
        .filter(|alt| alt.is_object() && !is_null(alt))
        .collect();
    let composed = alternatives.first().copied();
    let source = match (value.get("properties"), composed) {
        (None, Some(composed)) => composed,
        _ => &value,
    };
    let kind = own(&value)
        .or_else(|| {
            alternatives
                .iter()
                .find_map(|alt| own(alt).or_else(|| shape(alt)))
        })
        .or_else(|| shape(&value))
        .unwrap_or_default();
    let get = |key: &str| value.get(key).or_else(|| source.get(key));
    let strings = |value: Option<&Value>| {
        Some(json!(
            value?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
        ))
    };
    let text = |key: &str| get(key).and_then(Value::as_str).map(Value::from);
    let count = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_i64)
            .map(|count| json!(count as i32))
    };
    // Properties sorted, so the bytes and the cache prefix they key stay stable.
    let properties = source.obj("properties").map(|properties| {
        let properties = properties.iter();
        let properties =
            properties.filter_map(|(name, value)| Some((name, self::schema(value.clone()).ok()?)));
        json!(properties.collect::<std::collections::BTreeMap<_, _>>())
    });
    let items = get("items").and_then(|items| self::schema(items.clone()).ok());
    let items = items.or_else(|| (kind == "array").then(|| json!({ "type": "string" })));
    let nullable = nullable(&value) || composed.is_some_and(nullable);
    Ok(Value::Object(object([
        ("type", Some(json!(kind))),
        ("format", text("format")),
        ("description", text("description")),
        ("nullable", nullable.then_some(Value::Bool(true))),
        ("enum", strings(get("enum"))),
        ("maxItems", count("maxItems")),
        ("minItems", count("minItems")),
        ("properties", properties),
        ("required", strings(source.get("required"))),
        ("items", items),
    ])))
}

/// Inline references from `$defs` or `definitions` and remove those sections.
/// Return unchanged input if neither section exists. Callers must supply
/// acyclic references.
///
/// # Errors
///
/// For a non-object definitions section, a reference path other than
/// `#/$defs/` or `#/definitions/`, or a missing definition.
pub fn flatten_schema(mut schema: Value) -> Result<Value, EncodeError> {
    fn resolve(value: &mut Value, defs: &Map<String, Value>) -> Result<(), EncodeError> {
        match value {
            Value::Object(object) => match object.get("$ref").and_then(Value::as_str) {
                Some(reference) => {
                    let missing = |what: &str| EncodeError::request(format!("{what}: {reference}"));
                    let name = reference
                        .strip_prefix("#/$defs/")
                        .or_else(|| reference.strip_prefix("#/definitions/"));
                    let name = name.ok_or_else(|| missing("Unsupported reference format"))?;
                    let mut resolved = defs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| missing("Reference not found"))?;
                    resolve(&mut resolved, defs)?;
                    *value = resolved;
                    Ok(())
                }
                None => object
                    .values_mut()
                    .try_for_each(|value| resolve(value, defs)),
            },
            Value::Array(items) => items.iter_mut().try_for_each(|value| resolve(value, defs)),
            _ => Ok(()),
        }
    }
    let defs = schema
        .as_object()
        .and_then(|object| object.get("$defs").or_else(|| object.get("definitions")));
    let Some(defs) = defs.cloned() else {
        return Ok(schema);
    };
    let Value::Object(defs) = defs else {
        return Err(EncodeError::request("$defs must be an object"));
    };
    resolve(&mut schema, &defs)?;
    if let Some(object) = schema.as_object_mut() {
        object.shift_remove("$defs");
        object.shift_remove("definitions");
    }
    Ok(schema)
}

/// Map a Google `finishReason` in its SCREAMING_SNAKE_CASE wire spelling.
/// `STOP` is a stop (a tool use when the turn holds calls) and `MAX_TOKENS`
/// a length stop. Every other reason, documented or not, is a failure: the
/// content filters as
/// [`ContentFilter`](crate::completion::FinishReason::ContentFilter), the
/// rest as [`Other`](crate::completion::FinishReason::Other) with the
/// reason's name.
pub fn map_google_finish_reason(wire_name: &str) -> crate::completion::FinishReason {
    use crate::completion::FinishReason;
    match wire_name {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "MODEL_ARMOR" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Rig's usage for a `usageMetadata` document, read leniently: a count that
/// is absent or not a non-negative integer is unreported, and no other field
/// is read, so usage never fails a reply. Rig's input is the prompt plus the
/// tool-use prompt, its output the candidates plus the thoughts, and its
/// total their sum. A count Gemini leaves out is zero in those sums, as its
/// JSON omits zeros. The REST, Vertex AI and gRPC wires all read usage here.
pub fn usage_of(usage: &Value) -> crate::completion::Usage {
    let count = |key: &str| usage.get(key).and_then(Value::as_u64);
    let tool_use = count("toolUsePromptTokenCount");
    let thoughts = count("thoughtsTokenCount");
    let input = count("promptTokenCount")
        .unwrap_or(0)
        .saturating_add(tool_use.unwrap_or(0));
    let output = count("candidatesTokenCount")
        .unwrap_or(0)
        .saturating_add(thoughts.unwrap_or(0));
    crate::completion::Usage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cached_input_tokens: count("cachedContentTokenCount"),
        reasoning_tokens: thoughts,
        tool_use_prompt_tokens: tool_use,
        total_tokens: Some(input.saturating_add(output)),
        cache_creation_input_tokens: None,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod cached_content_conflict_matrix;
#[cfg(test)]
mod cached_content_request_tests;
