//! Shaping a conversation for the model it is sent to. [`adapt`] is the one
//! place history meets its target: a turn the same model produced keeps its
//! provider items, any other turn replays from its canonical fields, failed
//! turns are skipped, and every tool call gets an answer.
//!
//! ```
//! use rig_core::completion::history::{Accepts, ReplayTarget, adapt};
//! use rig_core::message::{Api, Message};
//!
//! #[derive(Debug)]
//! struct Target;
//!
//! impl ReplayTarget for Target {
//!     fn api(&self) -> Api {
//!         Api::from_static("example.chat")
//!     }
//!     fn provider(&self) -> &str {
//!         "example"
//!     }
//!     fn model(&self) -> &str {
//!         "example-1"
//!     }
//!     fn accepts(&self, _model: &str) -> Accepts {
//!         Accepts::ALL
//!     }
//! }
//!
//! let history = vec![Message::user("hi"), Message::assistant("hello")];
//! assert_eq!(adapt(&history, &Target), history);
//! ```

use std::collections::{HashMap, HashSet};

use base64::Engine as _;
use base64::prelude::{BASE64_STANDARD, BASE64_STANDARD_NO_PAD};

use crate::message::{
    Api, AssistantContent, AssistantMessage, CallId, DocumentMediaType, DocumentSourceKind,
    ImageMediaType, Message, Origin, Text, ToolCall, ToolResult, ToolResultContent, UserContent,
};
use crate::wasm_compat::WasmCompatSync;

/// The text of the result rig sends for a tool call nothing answered.
pub const NO_RESULT_PROVIDED: &str = "No result provided";

/// What replaces a user image for a model without image input.
pub const USER_IMAGE_OMITTED: &str = "(image omitted: model does not support images)";

/// What replaces another model's assistant image for a model that reads no
/// images in assistant turns.
pub const ASSISTANT_IMAGE_OMITTED: &str = "(image omitted: model does not support images)";

/// What replaces a tool-result image for a model without image input.
pub const TOOL_IMAGE_OMITTED: &str = "(tool image omitted: model does not support images)";

/// What replaces an image the provider cannot receive in its form.
pub const IMAGE_UNSENDABLE: &str = "(image omitted: the provider cannot receive it in this form)";

/// What replaces audio the provider cannot receive.
pub const AUDIO_UNSENDABLE: &str = "(audio omitted: the provider cannot receive it in this form)";

/// What replaces video the provider cannot receive.
pub const VIDEO_UNSENDABLE: &str = "(video omitted: the provider cannot receive it in this form)";

/// What replaces a document the provider cannot receive, when it holds no
/// text to send instead.
pub const DOCUMENT_UNSENDABLE: &str =
    "(document omitted: the provider cannot receive it in this form)";

/// What a tool-result image becomes in the result when the image moves to
/// the user message that follows.
pub const TOOL_IMAGE_ATTACHED: &str = "(see attached image)";

/// The text heading the user message that carries tool-result images.
pub const TOOL_IMAGES_HEADING: &str = "Attached image(s) from tool result:";

/// What a model reads, as replay sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accepts {
    /// Images in user messages.
    pub user_images: bool,
    /// Images in assistant turns another model produced.
    pub assistant_images: bool,
    /// Images inside tool results.
    pub tool_result_images: bool,
    /// Tool calls and their results.
    pub tools: bool,
}

impl Accepts {
    /// A model that reads every kind of content.
    pub const ALL: Self = Self {
        user_images: true,
        assistant_images: true,
        tool_result_images: true,
        tools: true,
    };

    /// A model that reads text and tools but no images.
    pub const TEXT: Self = Self {
        user_images: false,
        assistant_images: false,
        tool_result_images: false,
        tools: true,
    };
}

/// Where an image sits in a history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// In a user message.
    User,
    /// Inside a tool result.
    ToolResult,
    /// In an assistant turn another model produced.
    Assistant,
}

/// One media part, as [`ReplayTarget::encodes`] sees it. Raw bytes are
/// already base64, and an inline image already has the media type its bytes
/// name.
#[derive(Debug, Clone, Copy)]
pub enum Media<'a> {
    /// An image, and where it sits.
    Image(&'a crate::message::Image, Place),
    /// Audio in a user message.
    Audio(&'a crate::message::Audio),
    /// Video in a user message.
    Video(&'a crate::message::Video),
    /// A document in a user message.
    Document(&'a crate::message::Document),
}

/// The model a request is sent to, as replay sees it. Every completion wire
/// implements it.
pub trait ReplayTarget: std::fmt::Debug + WasmCompatSync {
    /// The wire format.
    fn api(&self) -> Api;

    /// The provider descriptor name.
    fn provider(&self) -> &str;

    /// The model id the wire addresses.
    fn model(&self) -> &str;

    /// What `model`, the model a request addresses on this wire, reads.
    /// [`adapt`] downgrades everything else, so the encoder never sees it.
    fn accepts(&self, model: &str) -> Accepts;

    /// Whether the encoder carries `media` to `model`: its source (data,
    /// URL, file id or string), its media type, and where it sits. [`adapt`]
    /// replaces every part this refuses with a placeholder, or a text
    /// document with its text, so the encoder never refuses canonical
    /// content. Row H9 of the history conformance suite holds every wire to
    /// it. By default every form is carried.
    fn encodes(&self, model: &str, media: Media<'_>) -> bool {
        let _ = (model, media);
        true
    }

    /// The id `id` takes when a call another model made is sent to `model`
    /// on this wire. `source` is the call's origin, `None` for a hand-built
    /// turn. Results answering the call are rewritten to match, and an id
    /// another call already took is made distinct.
    fn normalize_tool_call_id(&self, id: &str, model: &str, source: Option<&Origin>) -> String {
        let _ = (model, source);
        id.to_owned()
    }

    /// Whether `request` continues a conversation the provider stores, such
    /// as one naming a previous interaction. Results before the history's
    /// first turn then answer calls the provider holds, so [`adapt`] keeps
    /// them.
    fn continues_stored(&self, request: &crate::completion::CompletionRequest) -> bool {
        let _ = request;
        false
    }
}

/// `history` shaped for `target`.
///
/// - Blank system messages are dropped.
/// - A turn whose origin names `target`'s API, provider and model keeps its
///   provider items. Its reasoning and blank text are dropped unless they
///   still hold a current provider item.
/// - Any other turn keeps only canonical fields: provider items are cleared,
///   reasoning becomes plain text (redacted or empty reasoning is dropped),
///   [`Opaque`](crate::message::Opaque) items are dropped and call ids go
///   through [`ReplayTarget::normalize_tool_call_id`], made distinct when two
///   calls would share one.
/// - Content the model does not read ([`ReplayTarget::accepts`]) is
///   downgraded: images become placeholder text, a tool-result image moves
///   to a user message after the results when only user images are read,
///   and without tools calls and results become text.
/// - Opaque items marked not to replay are always dropped, and so is a turn
///   that ended in an error or was aborted, with the results answering it.
/// - Every call left unanswered when the next user or assistant message
///   arrives, or when the history ends, gets a [`NO_RESULT_PROVIDED`] error
///   result; a result no preceding call asked for is dropped. A system
///   message that arrives while calls wait is held until they are answered.
/// - Adjacent user messages become one. A turn left empty is dropped, and so
///   is a user message left empty. A message that was empty to begin with is
///   kept, for the request boundary to reject.
pub fn adapt(history: &[Message], target: &dyn ReplayTarget) -> Vec<Message> {
    adapt_for_model(history, target, None, false)
}

/// [`adapt`] for a request that names `model` in place of the wire's own,
/// and that continues a conversation the provider stores when `stored`.
pub(crate) fn adapt_for_model(
    history: &[Message],
    target: &dyn ReplayTarget,
    model: Option<&str>,
    stored: bool,
) -> Vec<Message> {
    let model = model.unwrap_or(target.model());
    let same = (target.api(), target.provider(), model);
    let accepts = target.accepts(model);
    let mut ids = Renamed::default();
    let mut shaped = Vec::with_capacity(history.len());
    for message in history {
        match message {
            Message::System { content } => {
                if !content.trim().is_empty() {
                    shaped.push(Some(message.clone()));
                }
            }
            Message::User { content } => {
                let form = Form {
                    target,
                    model,
                    accepts,
                };
                shaped.extend(user(content, &ids, &form).into_iter().map(Some));
            }
            Message::Assistant(turn) => {
                let adapted = assistant(turn, target, &same, accepts, &mut ids);
                let adapted = AssistantMessage {
                    content: adapted
                        .content
                        .into_iter()
                        .map(|block| match block {
                            AssistantContent::Image(image) if image.native.is_none() => {
                                let image = sendable_image(image);
                                if matches!(image.data, DocumentSourceKind::Unknown)
                                    || !target
                                        .encodes(model, Media::Image(&image, Place::Assistant))
                                {
                                    AssistantContent::Text(Text::new(IMAGE_UNSENDABLE))
                                } else {
                                    AssistantContent::Image(image)
                                }
                            }
                            block => block,
                        })
                        .collect(),
                    ..adapted
                };
                let emptied = adapted.content.is_empty() && !turn.content.is_empty();
                shaped.push((!emptied).then_some(Message::Assistant(adapted)));
            }
        }
    }
    merge_users(answer_calls(shaped, stored))
}

/// Call ids renamed for the target: each source id's new id, and the ids
/// taken, so two calls never share one.
#[derive(Default)]
struct Renamed {
    to: HashMap<CallId, CallId>,
    taken: HashMap<String, CallId>,
}

impl Renamed {
    /// The id `call` takes: `normalized`, or, when another call took it, the
    /// same id with its tail replaced by a counter until it is free. A
    /// counter of lowercase alphanumerics keeps the id's length and is legal
    /// on every wire.
    fn claim(&mut self, call: &CallId, normalized: String) -> String {
        if let Some(id) = self.to.get(call) {
            return id.wire().into_owned();
        }
        let mut id = normalized.clone();
        let mut attempt: u64 = 1;
        while self.taken.get(&id).is_some_and(|owner| owner != call) {
            id = with_counter(&normalized, attempt);
            attempt += 1;
        }
        self.taken.insert(id.clone(), call.clone());
        id
    }
}

/// `id` with its last characters replaced by `attempt` in base 36.
fn with_counter(id: &str, attempt: u64) -> String {
    let mut digits = Vec::new();
    let mut value = attempt;
    while value > 0 {
        digits.push(char::from_digit((value % 36) as u32, 36).unwrap_or('0'));
        value /= 36;
    }
    digits.reverse();
    let keep = id.chars().count().saturating_sub(digits.len());
    id.chars().take(keep).chain(digits).collect()
}

/// `turn` shaped for the target `(api, provider, model)`.
fn assistant(
    turn: &AssistantMessage,
    target: &dyn ReplayTarget,
    (api, provider, model): &(Api, &str, &str),
    accepts: Accepts,
    ids: &mut Renamed,
) -> AssistantMessage {
    let same = turn
        .origin
        .as_ref()
        .is_some_and(|origin| origin.same_model(api, provider, model));
    let content = turn
        .content
        .iter()
        .filter_map(|block| {
            let block = if same {
                block.clone()
            } else {
                match block.canonical() {
                    AssistantContent::Reasoning(reasoning) => {
                        if reasoning.redacted || reasoning.text.trim().is_empty() {
                            return None;
                        }
                        AssistantContent::Text(Text::new(reasoning.text))
                    }
                    AssistantContent::Opaque(_) => return None,
                    AssistantContent::Image(_) if !accepts.assistant_images => {
                        AssistantContent::Text(Text::new(ASSISTANT_IMAGE_OMITTED))
                    }
                    AssistantContent::ToolCall(mut call) => {
                        let wire = call.id.wire();
                        let normalized =
                            target.normalize_tool_call_id(&wire, model, turn.origin.as_ref());
                        let claimed = ids.claim(&call.id, normalized);
                        if claimed != wire {
                            let id = CallId::from_wire(claimed);
                            ids.to.insert(call.id.clone(), id.clone());
                            call.id = id;
                        }
                        AssistantContent::ToolCall(call)
                    }
                    block => block,
                }
            };
            let block = match block {
                AssistantContent::ToolCall(call) if !accepts.tools => {
                    AssistantContent::Text(Text::new(format!(
                        "[called tool {} with {}]",
                        call.function.name,
                        call.function.arguments_value()
                    )))
                }
                block => block,
            };
            kept(&block).then_some(block)
        })
        .collect();
    AssistantMessage {
        content,
        origin: turn.origin.clone(),
        stop: turn.stop.clone(),
        native: if same { turn.native.clone() } else { None },
    }
}

/// Whether a block has anything to send: blank text survives only with a
/// provider item that is still current, empty reasoning with any provider
/// item (its identity pairs it with what follows, pi replays it whatever
/// its text), and an opaque item only when it replays.
fn kept(block: &AssistantContent) -> bool {
    let current = block.native_item().is_some();
    match block {
        AssistantContent::Text(text) => !text.text.trim().is_empty() || current,
        AssistantContent::Reasoning(reasoning) => {
            !reasoning.text.trim().is_empty() || reasoning.native.is_some()
        }
        AssistantContent::Opaque(opaque) => opaque.replay,
        AssistantContent::ToolCall(_) | AssistantContent::Image(_) => true,
    }
}

/// The target a user message is shaped for.
struct Form<'a> {
    target: &'a dyn ReplayTarget,
    model: &'a str,
    accepts: Accepts,
}

impl Form<'_> {
    /// Whether an image the model reads at `place` can be sent there.
    fn sends(&self, image: &crate::message::Image, place: Place) -> bool {
        let reads = match place {
            Place::User => self.accepts.user_images,
            Place::ToolResult => self.accepts.tool_result_images,
            Place::Assistant => self.accepts.assistant_images,
        };
        reads
            && !matches!(image.data, DocumentSourceKind::Unknown)
            && self.target.encodes(self.model, Media::Image(image, place))
    }
}

/// The user message `content` shaped for `form`: one message, or two when
/// tool-result images move to a message of their own.
fn user(content: &[UserContent], ids: &Renamed, form: &Form<'_>) -> Vec<Message> {
    let mut shaped: Vec<UserContent> = Vec::with_capacity(content.len());
    let mut attached = Vec::new();
    for part in content {
        let placeholder = match part {
            UserContent::Image(image) => {
                let image = sendable_image(image.clone());
                if form.sends(&image, Place::User) {
                    shaped.push(UserContent::Image(image));
                    continue;
                }
                if form.accepts.user_images {
                    IMAGE_UNSENDABLE
                } else {
                    USER_IMAGE_OMITTED
                }
            }
            UserContent::Audio(audio) => {
                let mut audio = audio.clone();
                audio.data = sendable(audio.data);
                if !matches!(audio.data, DocumentSourceKind::Unknown)
                    && form.target.encodes(form.model, Media::Audio(&audio))
                {
                    shaped.push(UserContent::Audio(audio));
                    continue;
                }
                AUDIO_UNSENDABLE
            }
            UserContent::Video(video) => {
                let mut video = video.clone();
                video.data = sendable(video.data);
                if !matches!(video.data, DocumentSourceKind::Unknown)
                    && form.target.encodes(form.model, Media::Video(&video))
                {
                    shaped.push(UserContent::Video(video));
                    continue;
                }
                VIDEO_UNSENDABLE
            }
            UserContent::Document(document) => {
                let mut document = document.clone();
                document.data = sendable(document.data);
                if !matches!(document.data, DocumentSourceKind::Unknown)
                    && form.target.encodes(form.model, Media::Document(&document))
                {
                    shaped.push(UserContent::Document(document));
                    continue;
                }
                if let Some(text) = document_text(&document) {
                    shaped.push(UserContent::text(text));
                    continue;
                }
                DOCUMENT_UNSENDABLE
            }
            UserContent::ToolResult(result) => {
                let mut result = result.clone();
                if let Some(id) = ids.to.get(&result.call) {
                    result.call = id.clone();
                }
                result.content = result_images(result.content, form, &mut attached);
                if form.accepts.tools {
                    shaped.push(UserContent::ToolResult(result));
                } else {
                    shaped.push(UserContent::text(result_text(&result)));
                }
                continue;
            }
            UserContent::Text(_) => {
                shaped.push(part.clone());
                continue;
            }
        };
        let repeated = matches!(
            shaped.last(),
            Some(UserContent::Text(text)) if text.text == placeholder
        );
        if !repeated {
            shaped.push(UserContent::text(placeholder));
        }
    }
    let mut messages = vec![Message::User { content: shaped }];
    if !attached.is_empty() {
        let mut content = vec![UserContent::text(TOOL_IMAGES_HEADING)];
        content.extend(attached.into_iter().map(UserContent::Image));
        messages.push(Message::User { content });
    }
    messages
}

/// A tool result as text, for a model without tools.
fn result_text(result: &ToolResult) -> String {
    let text: Vec<String> = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => text.text.clone(),
            ToolResultContent::Json { value } => value.to_string(),
            ToolResultContent::Image(_) => TOOL_IMAGE_OMITTED.to_owned(),
        })
        .collect();
    let kind = if result.is_error { "error" } else { "result" };
    format!("[tool {} {kind}] {}", result.name, text.join("\n"))
}

/// `content` with each image the result cannot carry replaced: it becomes
/// [`TOOL_IMAGE_ATTACHED`] and moves to `attached` when it can go in a user
/// message, else [`TOOL_IMAGE_OMITTED`].
fn result_images(
    content: Vec<ToolResultContent>,
    form: &Form<'_>,
    attached: &mut Vec<crate::message::Image>,
) -> Vec<ToolResultContent> {
    let mut shaped: Vec<ToolResultContent> = Vec::with_capacity(content.len());
    for part in content {
        match part {
            ToolResultContent::Image(image) => {
                let image = sendable_image(image);
                if form.sends(&image, Place::ToolResult) {
                    shaped.push(ToolResultContent::Image(image));
                    continue;
                }
                let placeholder = if form.sends(&image, Place::User) {
                    attached.push(image);
                    TOOL_IMAGE_ATTACHED
                } else {
                    TOOL_IMAGE_OMITTED
                };
                if shaped.last().and_then(ToolResultContent::as_text) != Some(placeholder) {
                    shaped.push(ToolResultContent::text(placeholder));
                }
            }
            part => shaped.push(part),
        }
    }
    shaped
}

/// `source` with raw bytes given as base64, which every wire that takes
/// inline data reads.
fn sendable(source: DocumentSourceKind) -> DocumentSourceKind {
    match source {
        DocumentSourceKind::Raw(bytes) => DocumentSourceKind::Base64(BASE64_STANDARD.encode(bytes)),
        source => source,
    }
}

/// `image` with raw bytes as base64, and, when it is inline data without a
/// media type, the type its bytes name.
fn sendable_image(mut image: crate::message::Image) -> crate::message::Image {
    image.data = sendable(image.data);
    if image.media_type.is_none()
        && let DocumentSourceKind::Base64(data) = &image.data
    {
        image.media_type = sniffed(data);
    }
    image
}

/// The image type the base64 `data` starts with.
fn sniffed(data: &str) -> Option<ImageMediaType> {
    let head: String = data.chars().take(24).collect();
    let bytes = BASE64_STANDARD
        .decode(head.as_bytes())
        .or_else(|_| BASE64_STANDARD_NO_PAD.decode(head.as_bytes()))
        .ok()?;
    match bytes.as_slice() {
        [0x89, b'P', b'N', b'G', ..] => Some(ImageMediaType::PNG),
        [0xFF, 0xD8, 0xFF, ..] => Some(ImageMediaType::JPEG),
        [b'G', b'I', b'F', b'8', ..] => Some(ImageMediaType::GIF),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => Some(ImageMediaType::WEBP),
        _ => None,
    }
}

/// The text of a document that holds text: a string, or base64 data of a
/// media type other than PDF that decodes as UTF-8.
fn document_text(document: &crate::message::Document) -> Option<String> {
    match &document.data {
        DocumentSourceKind::String(text) => Some(text.clone()),
        DocumentSourceKind::Base64(data)
            if document
                .media_type
                .as_ref()
                .is_some_and(|media_type| *media_type != DocumentMediaType::PDF) =>
        {
            let bytes = BASE64_STANDARD.decode(data.as_bytes()).ok()?;
            String::from_utf8(bytes).ok()
        }
        _ => None,
    }
}

/// `history` with each run of adjacent user messages made one: a wire that
/// requires alternating roles would otherwise reject it. An empty user
/// message stays on its own, for the request boundary to reject.
fn merge_users(history: Vec<Message>) -> Vec<Message> {
    let mut merged: Vec<Message> = Vec::with_capacity(history.len());
    for message in history {
        match (merged.last_mut(), message) {
            (Some(Message::User { content: previous }), Message::User { content })
                if !previous.is_empty() && !content.is_empty() =>
            {
                previous.extend(content);
            }
            (_, message) => merged.push(message),
        }
    }
    merged
}

/// pi's second pass over `history`, where `None` is a turn the first pass
/// emptied: skip failed turns, answer every unanswered call, drop results no
/// call waits for, and hold system messages that arrive while calls wait.
/// A skipped turn's results go with it, and the user messages it separated
/// become one, since a wire that requires alternating roles would otherwise
/// reject the history.
fn answer_calls(history: Vec<Option<Message>>, stored: bool) -> Vec<Message> {
    // Results before the first turn of a stored conversation answer calls
    // the provider holds.
    let mut stored = stored;
    let mut shaped = Vec::with_capacity(history.len());
    let mut waiting: Vec<ToolCall> = Vec::new();
    let mut held = Vec::new();
    let mut gap = false;
    for message in adjacent_users_merged(history) {
        let Some(message) = message else {
            close(&mut shaped, &mut waiting, &mut held, Vec::new(), false);
            stored = false;
            gap = true;
            continue;
        };
        match message {
            Message::Assistant(turn) => {
                close(&mut shaped, &mut waiting, &mut held, Vec::new(), false);
                stored = false;
                if turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
                    gap = true;
                    continue;
                }
                gap = false;
                waiting = distinct(turn.tool_calls());
                shaped.push(Message::Assistant(turn));
            }
            Message::User { mut content } => {
                if content.is_empty() {
                    close(&mut shaped, &mut waiting, &mut held, Vec::new(), false);
                    shaped.push(Message::User { content });
                    gap = false;
                    continue;
                }
                // A result answers a call of the turn just before it, once.
                let mut answered = HashSet::new();
                content.retain(|part| match part {
                    UserContent::ToolResult(result) => {
                        (stored || waiting.iter().any(|call| call.id == result.call))
                            && answered.insert(result.call.clone())
                    }
                    UserContent::Text(_)
                    | UserContent::Image(_)
                    | UserContent::Audio(_)
                    | UserContent::Video(_)
                    | UserContent::Document(_) => true,
                });
                if content.is_empty() && waiting.is_empty() {
                    // Only results nothing waits for: the gap stays open.
                    continue;
                }
                close(&mut shaped, &mut waiting, &mut held, content, gap);
                gap = false;
            }
            Message::System { .. } if !waiting.is_empty() => held.push(message),
            system => {
                gap = false;
                shaped.push(system);
            }
        }
    }
    close(&mut shaped, &mut waiting, &mut held, Vec::new(), false);
    shaped
}

/// `history` with each run of adjacent non-empty user messages made one, so
/// results split over several messages answer the turn before them.
fn adjacent_users_merged(history: Vec<Option<Message>>) -> Vec<Option<Message>> {
    let mut merged: Vec<Option<Message>> = Vec::with_capacity(history.len());
    for message in history {
        match (merged.last_mut(), message) {
            (Some(Some(Message::User { content: previous })), Some(Message::User { content }))
                if !previous.is_empty() && !content.is_empty() =>
            {
                previous.extend(content)
            }
            (_, message) => merged.push(message),
        }
    }
    merged
}

/// Push the user message `content` with a result added for each `waiting`
/// call it does not answer, then the held system messages. The results go
/// before its first non-result part, in call order. Nothing is pushed for
/// an empty message, and `merge` appends `content` to a user message that
/// ends `shaped`.
fn close(
    shaped: &mut Vec<Message>,
    waiting: &mut Vec<ToolCall>,
    held: &mut Vec<Message>,
    mut content: Vec<UserContent>,
    merge: bool,
) {
    let missing: Vec<UserContent> = waiting
        .drain(..)
        .filter(|call| {
            !content.iter().any(
                |part| matches!(part, UserContent::ToolResult(result) if result.call == call.id),
            )
        })
        .map(|call| {
            UserContent::ToolResult(ToolResult {
                call: call.id,
                name: call.function.name,
                content: vec![ToolResultContent::text(NO_RESULT_PROVIDED)],
                is_error: true,
            })
        })
        .collect();
    let at = content
        .iter()
        .position(|part| !matches!(part, UserContent::ToolResult(_)))
        .unwrap_or(content.len());
    content.splice(at..at, missing);
    // Results come first: Anthropic requires it, and Chat sends them as tool
    // messages that must follow the turn directly.
    content.sort_by_key(|part| !matches!(part, UserContent::ToolResult(_)));
    if !content.is_empty() {
        match shaped.last_mut() {
            Some(Message::User { content: previous }) if merge => previous.extend(content),
            _ => shaped.push(Message::User { content }),
        }
    }
    shaped.append(held);
}

/// `calls`, keeping the first of each id.
fn distinct<'a>(calls: impl Iterator<Item = &'a ToolCall>) -> Vec<ToolCall> {
    let mut seen = HashSet::new();
    calls
        .filter(|call| seen.insert(call.id.clone()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests;
