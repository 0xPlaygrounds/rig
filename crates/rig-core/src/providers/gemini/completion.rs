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
}

/// Where a GenerateContent part holds its call's id, on every wire that
/// speaks it.
pub const CALL_ID_SLOT: Option<&str> = Some("/functionCall/id");

/// The major version of a Gemini model id, read past a `models/` prefix:
/// `gemini-<major>…` or `gemini-live-<major>…`. An alias such as
/// `gemini-flash-latest` names none.
fn gemini_major(model: &str) -> Option<u32> {
    let model = model.to_ascii_lowercase();
    let model = model.strip_prefix("models/").unwrap_or(&model);
    let rest = model.strip_prefix("gemini-")?;
    let rest = rest.strip_prefix("live-").unwrap_or(rest);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest.get(..end)?.parse().ok()
}

/// Whether `model` is Gemini 3 or later, which validates the thought
/// signatures of function calls.
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
/// responses only from Gemini 3 on. A model that names no Gemini version
/// (an alias, or Claude behind Vertex AI) is assumed to read them, as pi
/// assumes.
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
/// [`GenerateContent::with_cached_content`] checks one.
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
    let CompletionRequest {
        chat_history,
        tools,
        temperature,
        max_tokens,
        tool_choice,
        additional_params,
        output_schema,
        ..
    } = request;
    let mut params = match additional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(other) => return Err(invalid("additional_params", "an object", &other)),
    };
    let mut extra_tools = match params.shift_remove("tools") {
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
    if let Some(schema) = output_schema {
        let config = config.get_or_insert_default();
        config.insert("responseMimeType".into(), json!("application/json"));
        config.insert("responseJsonSchema".into(), schema.to_value());
    }
    if let Some(temperature) = temperature {
        config
            .get_or_insert_default()
            .insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = max_tokens {
        config
            .get_or_insert_default()
            .insert("maxOutputTokens".into(), json!(max_tokens));
    }
    let (system, history): (Vec<Message>, Vec<Message>) = chat_history
        .into_iter()
        .partition(|message| matches!(message, Message::System { .. }));
    let system: Vec<Value> = system
        .into_iter()
        .filter_map(|message| match message {
            Message::System { content } if !content.is_empty() => Some(text_part(content)),
            _ => None,
        })
        .collect();
    // Gemini rejects a field set twice, and merges a tool choice set twice,
    // which unions the allowed functions.
    if !system.is_empty()
        && let Some(spelling) = present(&params, &SYSTEM_INSTRUCTION)
    {
        return Err(EncodeError::request(format!(
            "a Gemini request set the system instruction twice: as a preamble or system \
             message and as `additional_params.{spelling}`. Set it one way or the other"
        )));
    }
    if tool_choice.is_some()
        && let Some(spelling) = present(&params, &TOOL_CONFIG)
    {
        return Err(EncodeError::request(format!(
            "a Gemini request set the tool choice twice: as `tool_choice` and as \
             `additional_params.{spelling}`, which Gemini merges. Set it one way or the other"
        )));
    }
    let mut declared = Vec::with_capacity(tools.len());
    for tool in tools {
        let mut declaration = json!({ "name": tool.name, "description": tool.description });
        let parameters = tool_parameters_to_schema(tool.parameters).map_err(|error| {
            let reason = std::error::Error::source(&error)
                .map_or_else(|| error.to_string(), ToString::to_string);
            EncodeError::request(format!(
                "Tool '{}' could not be converted to a schema: {reason}",
                tool.name
            ))
        })?;
        if let (Some(parameters), Some(declaration)) = (parameters, declaration.as_object_mut()) {
            declaration.insert("parameters".into(), parameters);
        }
        declared.push(declaration);
    }
    let mut all_tools = Vec::new();
    if !declared.is_empty() {
        all_tools.push(json!({ "functionDeclarations": declared, "codeExecution": null }));
    }
    all_tools.append(&mut extra_tools);
    let mut body = Map::from_iter([
        (
            "contents".to_owned(),
            Value::Array(contents(history, target, model)?),
        ),
        (
            "generationConfig".to_owned(),
            config.map_or(Value::Null, Value::Object),
        ),
        ("safetySettings".to_owned(), Value::Null),
        (
            "toolConfig".to_owned(),
            tool_choice.map_or(Value::Null, calling_config),
        ),
        (
            "systemInstruction".to_owned(),
            match system.is_empty() {
                true => Value::Null,
                false => json!({ "parts": system, "role": "model" }),
            },
        ),
    ]);
    if !all_tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(all_tools));
    }
    body.extend(params);
    for name in handles {
        with_cached_content(&mut body, &name)?;
    }
    Ok(body)
}

fn invalid(field: &str, expected: &str, got: &Value) -> EncodeError {
    EncodeError::request(format!("Gemini `{field}` should be {expected}, got {got}"))
}

/// Proto3 JSON accepts both lowerCamelCase and original proto field names.
const SYSTEM_INSTRUCTION: [&str; 2] = ["systemInstruction", "system_instruction"];
const TOOL_CONFIG: [&str; 2] = ["toolConfig", "tool_config"];
const CACHED_CONTENT: [&str; 2] = ["cachedContent", "cached_content"];

/// The first of `spellings` set to a value other than `null` in `body`.
fn present<'a>(body: &Map<String, Value>, spellings: &[&'a str]) -> Option<&'a str> {
    spellings
        .iter()
        .find(|spelling| body.get(**spelling).is_some_and(|value| !value.is_null()))
        .copied()
}

/// The `functionCallingConfig` of `choice`.
fn calling_config(choice: ToolChoice) -> Value {
    let config = match choice {
        ToolChoice::Auto => json!({ "mode": "AUTO" }),
        ToolChoice::None => json!({ "mode": "NONE" }),
        ToolChoice::Required => json!({ "mode": "ANY" }),
        ToolChoice::Specific { function_names } => {
            json!({ "mode": "ANY", "allowed_function_names": function_names })
        }
    };
    json!({ "functionCallingConfig": config })
}

/// Set `name`, a `cachedContents/<id>` handle, as the prefix `body` reads.
///
/// # Errors
///
/// When `name` is not a handle, `body` names another one, or it sets a
/// system instruction, tools or a tool choice, which the cache owns.
pub fn with_cached_content(body: &mut Map<String, Value>, name: &str) -> Result<(), EncodeError> {
    if !name.starts_with("cachedContents/") {
        return Err(EncodeError::request(format!(
            "gemini cached content handle should look like `cachedContents/<id>`, got `{name}`"
        )));
    }
    if let Some(existing) = body.get("cachedContent").and_then(Value::as_str)
        && existing != name
    {
        return Err(EncodeError::request(format!(
            "a Gemini request set cached content twice, to `{existing}` and `{name}`: set it \
             one way or the other"
        )));
    }
    let mut conflicts = Vec::new();
    if present(body, &SYSTEM_INSTRUCTION).is_some() {
        conflicts.push("a system instruction (preamble)");
    }
    if present(body, &["tools"]).is_some() {
        conflicts.push("tools");
    }
    if present(body, &TOOL_CONFIG).is_some() {
        conflicts.push("a tool choice");
    }
    if !conflicts.is_empty() {
        let tools = body
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        let declares_functions = tools.into_iter().any(|tool| {
            ["functionDeclarations", "function_declarations"]
                .iter()
                .any(|spelling| !tool.arr(spelling).is_empty())
        });
        // Cached function declarations need caller-side dispatch; hosted
        // tools run on Gemini's side.
        let caveat = if declares_functions {
            " Function declarations in a cache are declarations only: an `Agent` dispatches \
             only tools it advertised, so a cached function tool runs only when you drive \
             `GenerateContent` yourself. Hosted tools such as `codeExecution` are fine to cache."
        } else {
            ""
        };
        return Err(EncodeError::request(format!(
            "a Gemini request using cached content `{name}` also set {}. The cached content \
             owns the system instruction, tools and tool choice of every request that uses \
             it: move them into the cache, or drop the cache handle.{caveat}",
            conflicts.join(" and ")
        )));
    }
    body.insert("cachedContent".to_owned(), Value::String(name.to_owned()));
    Ok(())
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
    let content = |role: &str, parts: Vec<Value>| json!({ "parts": parts, "role": role });
    let mut contents = Vec::with_capacity(history.len());
    for message in history {
        match message {
            Message::System { content: text } => {
                contents.push(content("user", vec![text_part(text)]))
            }
            Message::User { content: parts } => {
                let mut run = Vec::new();
                let mut responses = false;
                for part in parts {
                    let response = matches!(part, UserContent::ToolResult(_));
                    if response != responses && !run.is_empty() {
                        contents.push(content("user", std::mem::take(&mut run)));
                    }
                    responses = response;
                    let id = match &part {
                        UserContent::ToolResult(result) if with_ids => ids.of(&result.call),
                        _ => None,
                    };
                    run.push(user_part(part, id)?);
                }
                if !run.is_empty() {
                    contents.push(content("user", run));
                }
            }
            Message::Assistant(turn) => {
                let mut parts = Vec::with_capacity(turn.content.len());
                for block in &turn.content {
                    parts.extend(assistant_part(block, target, &ids, model)?);
                }
                contents.push(content("model", parts));
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
    match source {
        Source::Url(uri) => Ok((true, uri)),
        Source::Base64(data) => Ok((false, data)),
        Source::String(data) if verbatim => Ok((false, data)),
        Source::String(data) => Ok((
            false,
            base64::Engine::encode(&base64::prelude::BASE64_STANDARD, data),
        )),
        _ => Err(EncodeError::request(
            "Gemini cannot receive this media in its form",
        )),
    }
}

/// `source` of `mime_type` as GenerateContent part data: file data by URI,
/// or inline data, which needs a media type.
fn media(
    mime_type: Option<String>,
    source: Source,
    string_is_data: bool,
) -> Result<Value, EncodeError> {
    let (uri, data) = carried(source, string_is_data)?;
    Ok(match (uri, mime_type) {
        (true, mime_type) => json!({ "fileData": { "mimeType": mime_type, "fileUri": data } }),
        (false, Some(mime_type)) => {
            json!({ "inlineData": { "mimeType": mime_type, "data": data } })
        }
        (false, None) => {
            return Err(EncodeError::request(
                "Gemini cannot receive media without its type",
            ));
        }
    })
}

/// [`media`] as a whole part.
fn media_part(
    mime_type: Option<String>,
    source: Source,
    string_is_data: bool,
) -> Result<Value, EncodeError> {
    let mut part = media(mime_type, source, string_is_data)?;
    if let Some(part) = part.as_object_mut() {
        part.insert("thought".into(), Value::Bool(false));
    }
    Ok(part)
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
            let mut values = Vec::new();
            let mut parts = Vec::new();
            for item in result.content {
                match item {
                    ToolResultContent::Text(text) => values.push(Value::String(text.text)),
                    ToolResultContent::Json { value } => values.push(value),
                    ToolResultContent::Image(image) => {
                        parts.push(media(mime(image.media_type), image.data, true)?);
                    }
                }
            }
            let mut response = Map::from_iter([("name".to_owned(), json!(result.name))]);
            if let Some(id) = id {
                response.insert("id".to_owned(), json!(id));
            }
            let key = if result.is_error { "error" } else { "result" };
            let value = match values.len() {
                0 => None,
                1 => values.pop(),
                _ => Some(Value::Array(values)),
            };
            if let Some(value) = value {
                response.insert("response".to_owned(), json!({ key: value }));
            }
            if !parts.is_empty() {
                response.insert("parts".to_owned(), Value::Array(parts));
            }
            json!({ "functionResponse": response, "thought": false })
        }
        UserContent::Image(image) => media_part(mime(image.media_type), image.data, true)?,
        // A text document goes as text, so that RAG context reads as prose.
        UserContent::Document(document) => match (document.media_type, document.data) {
            (Some(media_type), Source::String(text)) if media_type != DocumentMediaType::PDF => {
                text_part(text)
            }
            (media_type, data) => media_part(mime(media_type), data, true)?,
        },
        UserContent::Audio(audio) => media_part(mime(audio.media_type), audio.data, false)?,
        UserContent::Video(video) => {
            let mut part = media_part(mime(video.media_type), video.data, false)?;
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
            if !with_ids && let Some(Value::Object(call)) = part.get_mut("functionCall") {
                call.shift_remove("id");
            }
            // Gemini rejects the whole request over a signature that is
            // not base64 ("Base64 decoding failed").
            if part
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .is_some_and(|sig| !is_base64(sig))
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
            let mut function_call =
                json!({ "name": call.function.name, "args": call.function.arguments });
            if let (true, Some(id), Some(fields)) =
                (with_ids, ids.of(&call.id), function_call.as_object_mut())
            {
                fields.insert("id".to_owned(), json!(id));
            }
            let mut part = json!({ "functionCall": function_call });
            if let (true, Some(part)) = (gemini_3_or_later(model), part.as_object_mut()) {
                part.insert(
                    "thoughtSignature".into(),
                    json!("skip_thought_signature_validator"),
                );
            }
            part
        }
        AssistantContent::Image(image) => {
            media_part(mime(image.media_type.clone()), image.data.clone(), true)?
        }
        AssistantContent::Opaque(opaque) => opaque.item.clone(),
    }))
}

/// Whether `text` is padded standard base64.
fn is_base64(text: &str) -> bool {
    let body = text.trim_end_matches('=');
    !body.is_empty()
        && text.len().is_multiple_of(4)
        && text.len() - body.len() <= 2
        && body
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
}

/// Convert a specified prompt block reason into a provider error with its
/// safety ratings. `feedback` is the reply's `promptFeedback` JSON, read
/// leniently. Content refusals are final; `OTHER` and unknown reasons are
/// transient.
pub(crate) fn blocked_prompt_error(feedback: &Value) -> Option<ProviderError> {
    let reason = match feedback.get("blockReason")? {
        Value::String(reason) => reason.clone(),
        Value::Number(number) => format!("BLOCK_REASON_{number}"),
        _ => return None,
    };
    // The zero value names no block. Gemini spells it `BLOCK_REASON_`,
    // Vertex AI `BLOCKED_REASON_`.
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
    let ratings: Vec<String> = feedback
        .arr("safetyRatings")
        .iter()
        .map(|rating| {
            format!(
                "{}={}",
                spelled(rating.get("category")),
                spelled(rating.get("probability"))
            )
        })
        .collect();
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
        Ok(None)
    } else {
        schema(parameters).map(Some)
    }
}

/// `value`, a JSON Schema, as the OpenAPI subset Gemini reads: references
/// inlined, the type inferred from a composition or the keys present, a
/// union with `null` as `nullable`, and an array without `items` given
/// string items, which Gemini requires.
fn schema(value: Value) -> Result<Value, EncodeError> {
    const COMPOSITIONS: [&str; 3] = ["anyOf", "oneOf", "allOf"];
    let value = flatten_schema(value)?;
    let Some(object) = value.as_object() else {
        return Err(EncodeError::request("Expected a JSON object for Schema"));
    };
    let alternatives = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_object)
            .filter(|alternative| !is_null(alternative))
    };
    let composed = COMPOSITIONS.iter().find_map(|key| alternatives(key).next());
    let source = match object.contains_key("properties") {
        true => object,
        false => composed.unwrap_or(object),
    };
    let kind = object
        .get("type")
        .and_then(type_name)
        .or_else(|| {
            COMPOSITIONS.iter().find_map(|key| {
                alternatives(key).find_map(|alternative| {
                    alternative
                        .get("type")
                        .and_then(type_name)
                        .or_else(|| shape_type(alternative))
                })
            })
        })
        .or_else(|| shape_type(object))
        .unwrap_or_default();
    let get = |key: &str| object.get(key).or_else(|| source.get(key));
    fn strings(value: Option<&Value>) -> Option<Vec<&str>> {
        Some(
            value?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .collect(),
        )
    }
    let mut schema = Map::from_iter([("type".to_owned(), json!(kind))]);
    for key in ["format", "description"] {
        if let Some(text) = get(key).and_then(Value::as_str) {
            schema.insert(key.to_owned(), json!(text));
        }
    }
    if nullable(object) || composed.is_some_and(nullable) {
        schema.insert("nullable".to_owned(), Value::Bool(true));
    }
    if let Some(values) = strings(get("enum")) {
        schema.insert("enum".to_owned(), json!(values));
    }
    for key in ["maxItems", "minItems"] {
        if let Some(count) = object.get(key).and_then(Value::as_i64) {
            schema.insert(key.to_owned(), json!(count as i32));
        }
    }
    if let Some(properties) = source.get("properties").and_then(Value::as_object) {
        // Sorted, so the bytes and the cache prefix they key stay stable.
        let properties: std::collections::BTreeMap<&String, Value> = properties
            .iter()
            .filter_map(|(name, value)| Some((name, self::schema(value.clone()).ok()?)))
            .collect();
        schema.insert("properties".to_owned(), json!(properties));
    }
    if let Some(required) = strings(source.get("required")) {
        schema.insert("required".to_owned(), json!(required));
    }
    let items = get("items").and_then(|items| self::schema(items.clone()).ok());
    if let Some(items) = items.or_else(|| (kind == "array").then(|| json!({ "type": "string" }))) {
        schema.insert("items".to_owned(), items);
    }
    Ok(Value::Object(schema))
}

/// The type a schema's `type` names: the string, or in a list the first
/// name other than `null`.
fn type_name(value: &Value) -> Option<String> {
    if let Some(name) = value.as_str() {
        return Some(name.to_owned());
    }
    let names: Vec<&str> = value.as_array()?.iter().filter_map(Value::as_str).collect();
    names
        .iter()
        .find(|name| **name != "null")
        .or(names.first())
        .map(|name| (*name).to_owned())
}

fn is_null(schema: &Map<String, Value>) -> bool {
    schema.get("type").and_then(type_name).as_deref() == Some("null")
}

fn shape_type(schema: &Map<String, Value>) -> Option<String> {
    if schema.contains_key("properties") {
        Some("object".to_owned())
    } else if schema.contains_key("enum") {
        Some("string".to_owned())
    } else {
        None
    }
}

fn nullable(schema: &Map<String, Value>) -> bool {
    schema
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || schema
            .get("type")
            .and_then(Value::as_array)
            .is_some_and(|names| names.iter().any(|name| name.as_str() == Some("null")))
        || ["anyOf", "oneOf", "allOf"].iter().any(|key| {
            schema
                .get(*key)
                .and_then(Value::as_array)
                .is_some_and(|items| items.iter().filter_map(Value::as_object).any(is_null))
        })
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
    let Some(defs) = schema
        .as_object()
        .and_then(|object| object.get("$defs").or_else(|| object.get("definitions")))
        .cloned()
    else {
        return Ok(schema);
    };
    let Some(defs) = defs.as_object() else {
        return Err(EncodeError::request("$defs must be an object"));
    };
    resolve_refs(&mut schema, defs)?;
    if let Some(object) = schema.as_object_mut() {
        object.shift_remove("$defs");
        object.shift_remove("definitions");
    }
    Ok(schema)
}

fn resolve_refs(value: &mut Value, defs: &Map<String, Value>) -> Result<(), EncodeError> {
    match value {
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                let name = reference
                    .strip_prefix("#/$defs/")
                    .or_else(|| reference.strip_prefix("#/definitions/"))
                    .ok_or_else(|| {
                        EncodeError::request(format!("Unsupported reference format: {reference}"))
                    })?;
                let mut resolved = defs.get(name).cloned().ok_or_else(|| {
                    EncodeError::request(format!("Reference not found: {reference}"))
                })?;
                resolve_refs(&mut resolved, defs)?;
                *value = resolved;
                return Ok(());
            }
            object
                .values_mut()
                .try_for_each(|value| resolve_refs(value, defs))
        }
        Value::Array(items) => items
            .iter_mut()
            .try_for_each(|value| resolve_refs(value, defs)),
        _ => Ok(()),
    }
}

/// The configuration types a Gemini request's `additional_params` takes,
/// and the readers of Gemini's finish reasons and usage.
pub mod gemini_api_types {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// The `additional_params` of a Gemini request.
    #[derive(Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct AdditionalParameters {
        /// Change your Gemini request configuration.
        pub generation_config: Option<GenerationConfig>,
        /// Any additional parameters that you want.
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<Value>,
    }

    impl AdditionalParameters {
        /// These parameters with `cfg` as the generation config.
        pub fn with_config(mut self, cfg: GenerationConfig) -> Self {
            self.generation_config = Some(cfg);
            self
        }

        /// These parameters with `params` merged into the request.
        pub fn with_params(mut self, params: Value) -> Self {
            self.additional_params = Some(params);
            self
        }
    }

    /// Model generation options from the [Gemini API](https://ai.google.dev/api/generate-content#generationconfig).
    /// Supported fields depend on the model. All fields default to `None` and
    /// are omitted when unset, preserving provider defaults.
    #[derive(Debug, Default, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerationConfig {
        /// Up to five stop sequences.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        /// Output MIME type: `text/plain` by default, `application/json` for JSON.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        /// OpenAPI-subset output schema.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_schema: Option<Value>,
        /// The legacy spelling of `response_json_schema`.
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "_responseJsonSchema"
        )]
        pub _response_json_schema: Option<Value>,
        /// A standard JSON Schema for the output; excludes `response_schema`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_json_schema: Option<Value>,
        /// Number of generated responses to return; only 1 is supported.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidate_count: Option<i32>,
        /// Maximum output tokens per candidate.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u64>,
        /// Sampling temperature in `[0.0, 2.0]`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        /// Maximum cumulative token probability for nucleus sampling.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        /// Maximum number of likely tokens considered for sampling.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<i32>,
        /// Penalty for tokens already present in the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f64>,
        /// Penalty scaled by each token's frequency in the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f64>,
        /// Whether to return log probabilities.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_logprobs: Option<bool>,
        /// Top log probabilities per step, with `response_logprobs`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<i32>,
        /// Configuration for thinking.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_config: Option<ThinkingConfig>,
        /// Response modalities of multimodal output models.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_modalities: Option<Vec<ResponseModality>>,
        /// Image output configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub image_config: Option<ImageConfig>,
    }

    /// Response modalities supported by Gemini multimodal output models.
    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum ResponseModality {
        Text,
        Image,
        Audio,
    }

    /// Thinking depth level for Gemini 3 models.
    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum ThinkingLevel {
        Minimal,
        Low,
        Medium,
        High,
    }

    /// Configuration for the model's thinking. `thinking_budget` (Gemini
    /// 2.5) and `thinking_level` (Gemini 3) are mutually exclusive.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThinkingConfig {
        /// Token budget for thinking, 0 to 32768.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_budget: Option<u32>,
        /// Thinking depth level.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_level: Option<ThinkingLevel>,
        /// Whether the response includes summaries of the model's reasoning.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub include_thoughts: Option<bool>,
    }

    /// Image output configuration.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ImageConfig {
        /// The output aspect ratio, such as `16:9`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aspect_ratio: Option<String>,
        /// The output size, such as `2K`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub image_size: Option<String>,
    }

    /// Map a Google `finishReason` in its SCREAMING_SNAKE_CASE wire spelling.
    /// `STOP` is a stop (a tool use when the turn holds calls) and
    /// `MAX_TOKENS` a length stop. Every other reason, documented or not, is
    /// a failure: the content filters as
    /// [`ContentFilter`](crate::completion::FinishReason::ContentFilter),
    /// the rest as [`Other`](crate::completion::FinishReason::Other) with
    /// the reason's name.
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

    /// Rig's usage for a `usageMetadata` document, read leniently: a count
    /// that is absent or not a non-negative integer is unreported, and no
    /// other field is read, so usage never fails a reply. Rig's input is the
    /// prompt plus the tool-use prompt, its output the candidates plus the
    /// thoughts, and its total their sum. A count Gemini leaves out is zero
    /// in those sums, as its JSON omits zeros. The REST, Vertex AI and gRPC
    /// wires all read usage here.
    pub fn usage_of(usage: &Value) -> crate::completion::Usage {
        let count = |key: &str| usage.get(key).and_then(Value::as_u64);
        let tool_use = count("toolUsePromptTokenCount");
        let input = count("promptTokenCount")
            .unwrap_or(0)
            .saturating_add(tool_use.unwrap_or(0));
        let thoughts = count("thoughtsTokenCount");
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
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod cached_content_conflict_matrix;
#[cfg(test)]
mod cached_content_request_tests;
