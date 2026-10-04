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

    /// Whether `request` declares tools to the provider, in `tools`, in the
    /// `tools` of its `additional_params`, or through tools the wire adds.
    /// A request that declares none gets its history's calls and results as
    /// text. `ToolChoice::None` still declares them: the wire sends its own
    /// `none` choice beside the tool history.
    fn declares_tools(&self, request: &crate::completion::CompletionRequest) -> bool {
        declares_tools(request)
    }

    /// The keys of a provider item that survive an edit of its block: an
    /// encoder rebuilding the edited block keeps them ([`Replay::Identity`]).
    /// By default none do.
    fn identity(&self, item: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        let _ = item;
        serde_json::Map::new()
    }

    /// Where a tool call's id sits in this wire's call item, as a JSON
    /// pointer, when it has one. Replay writes the call's wire id there, so
    /// a replayed item and its result always agree, and a call the provider
    /// sent without an id keeps its item only on a wire with a slot.
    fn call_id_slot(&self) -> Option<&'static str> {
        None
    }

    /// Whether every reply on this wire states why it stopped. A reply that
    /// then names no reason failed (pi's `supportsFinishReason`); a wire
    /// with no finish vocabulary, such as a local runtime, says `false`.
    fn states_finish_reason(&self) -> bool {
        true
    }

    /// Whether `model` binds its provider items to the request's tools and
    /// system prompt, so a turn made under another context replays as if
    /// from another model.
    fn binds_context(&self, model: &str) -> bool {
        let _ = model;
        false
    }

    /// Whether `request` asks the provider to drop an item bound to another
    /// context itself (Anthropic's `drop_block`), so a turn made under
    /// another context replays verbatim rather than as another model's.
    fn drops_unbound_items(&self, request: &crate::completion::CompletionRequest) -> bool {
        let _ = request;
        false
    }

    /// Whether this provider item requires the item after it in its turn
    /// (Responses reasoning): when that one is not replayed, neither is
    /// this.
    fn needs_next(&self, item: &serde_json::Value) -> bool {
        let _ = item;
        false
    }

    /// The hosted-tool pair this opaque item belongs to: whether it is the
    /// use or the result, and the id they share. A use and its result
    /// replay only together, in one turn or across the model's turns. A use
    /// still running when the last turn ended (a paused turn, or one whose
    /// client call the hosted tool made) replays alone.
    fn hosted_pair(&self, item: &serde_json::Value) -> Option<(Pairing, String)> {
        let _ = item;
        None
    }

    /// Whether the wire rejects a conversation whose first message after
    /// the system prompt is not a user message. [`adapt`] then drops the
    /// assistant turns before the first user message, with their results.
    fn starts_with_user(&self) -> bool {
        false
    }

    /// Where `model` takes system messages that come after the
    /// conversation begins ([`LaterSystem`]). By default where the history
    /// has them.
    fn later_system(&self, model: &str) -> LaterSystem {
        let _ = model;
        LaterSystem::InPlace
    }

    /// Whether the wire requires user and assistant messages to alternate.
    /// [`adapt`] then joins two messages of one role that only system
    /// messages separate, which move after them.
    fn alternates_roles(&self) -> bool {
        false
    }

    /// Whether a hosted tool's use and result need the request to declare
    /// tools. A request that declares none then sends neither.
    fn hosted_needs_tools(&self) -> bool {
        false
    }

    /// Whether `model` reads a tool result as several parts. When it reads
    /// neither parts nor result images, [`adapt`] joins a result's text
    /// parts into one.
    fn result_parts(&self, model: &str) -> bool {
        let _ = model;
        false
    }

    /// Whether `block` on its own makes an assistant message this wire
    /// sends. A turn with no such block carries nothing, so [`adapt`] drops
    /// it and the user messages around it become one. By default every
    /// block does.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        let _ = block;
        true
    }
}

/// Where a target takes the system messages that come after the
/// conversation begins ([`ReplayTarget::later_system`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaterSystem {
    /// Where the history has them.
    InPlace,
    /// Joined with the leading ones into one system message, as pi's
    /// `collapseSystemMessages` does.
    Leading,
    /// As user text where the history has them, so adding one never moves
    /// the prefix before it.
    UserText,
}

/// Which side of a hosted-tool pair an opaque item is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pairing {
    /// The hosted tool's use.
    Use,
    /// Its result.
    Result,
}

/// What an encoder sends for one assistant block.
#[derive(Debug, Clone, PartialEq)]
pub enum Replay<'a> {
    /// The provider item, still current, with a call's id slot spelled.
    Item(std::borrow::Cow<'a, serde_json::Value>),
    /// The block was edited: rebuild it from its canonical fields, keeping
    /// these keys of its stale item.
    Identity(serde_json::Map<String, serde_json::Value>),
    /// Rebuild it from its canonical fields.
    Rebuild,
}

impl AssistantContent {
    /// What `target`'s encoder sends for this block, with call ids spelled
    /// by `ids`. Encoders read provider items only through this.
    pub fn replay(
        &self,
        target: &dyn ReplayTarget,
        ids: &crate::providers::internal::wire_ids::WireIds,
    ) -> Replay<'_> {
        if let Some(item) = self.native_item() {
            return match (self, target.call_id_slot()) {
                (AssistantContent::ToolCall(call), Some(slot)) => {
                    let mut item = item.clone();
                    if let Some(id) = ids.of(&call.id) {
                        set_pointer(&mut item, slot, serde_json::Value::String(id.to_owned()));
                    }
                    Replay::Item(std::borrow::Cow::Owned(item))
                }
                _ => Replay::Item(std::borrow::Cow::Borrowed(item)),
            };
        }
        let identity = self
            .stale_item()
            .map(|item| target.identity(item))
            .unwrap_or_default();
        if identity.is_empty() {
            Replay::Rebuild
        } else {
            Replay::Identity(identity)
        }
    }
}

/// Set the value at `pointer` in `item`, creating objects on the way.
fn set_pointer(item: &mut serde_json::Value, pointer: &str, value: serde_json::Value) {
    let mut at = item;
    let mut keys = pointer.split('/').skip(1).peekable();
    while let Some(key) = keys.next() {
        if !at.is_object() {
            *at = serde_json::Value::Object(serde_json::Map::new());
        }
        let serde_json::Value::Object(fields) = at else {
            return;
        };
        if keys.peek().is_none() {
            fields.insert(key.to_owned(), value);
            return;
        }
        at = fields
            .entry(key.to_owned())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
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
/// - Every call left unanswered when a user message with more than results
///   or the next assistant message arrives, or when the history ends, gets a
///   [`NO_RESULT_PROVIDED`] error result; a result no preceding call asked
///   for is dropped. A system message that arrives while calls wait is held
///   until they are answered.
/// - Adjacent user messages become one. A turn left empty is dropped, and so
///   is a user message left empty. A message that was empty to begin with is
///   kept, for the request boundary to reject.
pub fn adapt(history: &[Message], target: &dyn ReplayTarget) -> Vec<Message> {
    adapt_for(history, target, &Request::default())
}

/// What `adapt` reads of the request a history is sent with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Request<'a> {
    /// The model it names in place of the wire's own.
    pub(crate) model: Option<&'a str>,
    /// Whether it continues a conversation the provider stores.
    pub(crate) stored: bool,
    /// Whether it carries tools ([`ReplayTarget::declares_tools`]): a
    /// request with none gets calls and results as text.
    pub(crate) tools: bool,
    /// The fingerprint of its tools and system prompt ([`context_of`]).
    pub(crate) context: Option<crate::message::Fingerprint>,
}

impl Default for Request<'_> {
    fn default() -> Self {
        Self {
            model: None,
            stored: false,
            tools: true,
            context: None,
        }
    }
}

/// The fingerprint of `request`'s tool definitions, by name, and of its
/// system prompt as [`adapt`] sends it to `model` on `target`: what a
/// context-bound item was made under. The prompt is the leading non-blank
/// system messages, or every one when the target folds them into one, so
/// the request before `adapt` and after it give the same fingerprint.
pub(crate) fn context_of(
    request: &crate::completion::CompletionRequest,
    target: &dyn ReplayTarget,
    model: &str,
) -> crate::message::Fingerprint {
    let mut tools: Vec<_> = request.tools.iter().collect();
    tools.sort_by(|left, right| left.name.cmp(&right.name));
    let raw = raw_tools(request);
    let folds = target.later_system(model) == LaterSystem::Leading;
    let mut system: Vec<&str> = request
        .chat_history
        .iter()
        .map_while(|message| match message {
            Message::System { content } => Some(Some(content.as_str())),
            Message::User { .. } | Message::Assistant(_) => folds.then_some(None),
        })
        .flatten()
        .filter(|content| !content.trim().is_empty())
        .collect();
    let joined;
    if folds && system.len() > 1 {
        joined = system.join("\n\n");
        system = vec![joined.as_str()];
    }
    let mut fields = vec![serde_json::json!("context"), serde_json::json!(tools)];
    fields.push(serde_json::json!(system));
    if !raw.is_empty() {
        fields.push(serde_json::json!(raw));
    }
    crate::message::Fingerprint::of(&serde_json::Value::Array(fields))
}

/// Whether `request` declares tools in `tools` or in the `tools` of its
/// `additional_params`.
pub(crate) fn declares_tools(request: &crate::completion::CompletionRequest) -> bool {
    !request.tools.is_empty() || !raw_tools(request).is_empty()
}

/// The tools `request` passes as is in the `tools` of its
/// `additional_params`, such as a provider's hosted tools.
pub(crate) fn raw_tools(request: &crate::completion::CompletionRequest) -> &[serde_json::Value] {
    use crate::json_utils::Lenient;
    request
        .additional_params
        .as_ref()
        .map_or(&[][..], |params| params.arr("tools"))
}

/// [`adapt`] for `request`.
pub(crate) fn adapt_for(
    history: &[Message],
    target: &dyn ReplayTarget,
    request: &Request<'_>,
) -> Vec<Message> {
    let model = request.model.unwrap_or(target.model());
    let stored = request.stored;
    let same = Same {
        api: target.api(),
        provider: target.provider(),
        model,
        context: request.context.filter(|_| target.binds_context(model)),
    };
    let mut accepts = target.accepts(model);
    accepts.tools &= request.tools;
    let hosted = hosted_pairs(history, target, &same);
    let last_turn = history
        .iter()
        .rposition(|message| matches!(message, Message::Assistant(_)));
    let mut ids = Renamed::default();
    let mut shaped = Vec::with_capacity(history.len());
    for (at, message) in history.iter().enumerate() {
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
                shaped.extend(user(content, &mut ids, &form).into_iter().map(Some));
            }
            Message::Assistant(turn) => {
                // A use can still be running only in the last turn, and only
                // while nothing new follows it or the turn awaits a client call.
                let last = Some(at) == last_turn
                    && (turn
                        .content
                        .iter()
                        .any(|block| matches!(block, AssistantContent::ToolCall(_)))
                        || history
                            .get(at + 1..)
                            .into_iter()
                            .flatten()
                            .all(|message| matches!(message, Message::System { .. })));
                let here: HashSet<usize> = hosted
                    .iter()
                    .filter(|(message, _)| *message == at)
                    .map(|(_, block)| *block)
                    .collect();
                let adapted = assistant(turn, target, &same, accepts, &mut ids, &here, last);
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
                let emptied = !turn.content.is_empty()
                    && !adapted
                        .content
                        .iter()
                        .any(|block| target.sends_alone(block));
                shaped.push((!emptied).then_some(Message::Assistant(adapted)));
            }
        }
    }
    // Calls and results pair first, so an orphan result goes whether or not
    // the request declares tools; only then does a request without tools
    // get them as text, with no result made up for an unanswered call.
    let shaped = merge_users(answer_calls(shaped, stored, accepts.tools));
    let shaped = if accepts.tools {
        shaped
    } else {
        tools_as_text(shaped)
    };
    let shaped = match target.later_system(model) {
        LaterSystem::InPlace => shaped,
        LaterSystem::Leading => leading_system(shaped),
        LaterSystem::UserText => system_as_user_text(shaped),
    };
    let shaped = if target.starts_with_user() && !stored {
        from_first_user(shaped)
    } else {
        shaped
    };
    if target.alternates_roles() {
        alternated(shaped)
    } else {
        shaped
    }
}

/// `history` with its calls and results as text, for a request that
/// declares no tools. Keys are sorted, so the text is the same however the
/// provider ordered the arguments.
fn tools_as_text(history: Vec<Message>) -> Vec<Message> {
    let history = history
        .into_iter()
        .map(|message| match message {
            Message::Assistant(mut turn) => {
                for block in &mut turn.content {
                    if let AssistantContent::ToolCall(call) = block {
                        *block = AssistantContent::Text(Text::new(format!(
                            "[called tool {} with {}]",
                            call.function.name,
                            crate::json_utils::to_canonical_string(
                                &call.function.arguments_value()
                            )
                        )));
                    }
                }
                Message::Assistant(turn)
            }
            Message::User { content } => Message::User {
                content: content
                    .into_iter()
                    .map(|part| match part {
                        UserContent::ToolResult(result) => UserContent::text(result_text(&result)),
                        part => part,
                    })
                    .collect(),
            },
            message => message,
        })
        .collect();
    merge_users(history)
}

/// `history` from its first user message, after the leading system
/// messages: an assistant turn before it goes, with the results answering
/// it, until a user message leads.
fn from_first_user(mut history: Vec<Message>) -> Vec<Message> {
    let lead = history
        .iter()
        .take_while(|message| matches!(message, Message::System { .. }))
        .count();
    while let Some(Message::Assistant(turn)) = history.get(lead) {
        let calls: HashSet<CallId> = turn.tool_calls().map(|call| call.id.clone()).collect();
        history.remove(lead);
        if let Some(Message::User { content }) = history.get_mut(lead) {
            content.retain(|part| {
                !matches!(part, UserContent::ToolResult(result) if calls.contains(&result.call))
            });
            if content.is_empty() {
                history.remove(lead);
            }
        }
    }
    history
}

/// Whether `history` has a system message after its first user or
/// assistant message.
fn has_later_system(history: &[Message]) -> bool {
    history
        .iter()
        .skip_while(|message| matches!(message, Message::System { .. }))
        .any(|message| matches!(message, Message::System { .. }))
}

/// `history` with every later system message as user text where it stands.
fn system_as_user_text(history: Vec<Message>) -> Vec<Message> {
    let mut leading = true;
    let history = history
        .into_iter()
        .map(|message| match message {
            Message::System { content } if !leading => Message::User {
                content: vec![UserContent::text(content)],
            },
            message => {
                leading &= matches!(message, Message::System { .. });
                message
            }
        })
        .collect();
    merge_users(history)
}

/// `history` with no two user or assistant messages of one role in a row:
/// two that only system messages separate become one, and the system
/// messages move after them.
fn alternated(history: Vec<Message>) -> Vec<Message> {
    let mut alternated: Vec<Message> = Vec::with_capacity(history.len());
    let mut held: Vec<Message> = Vec::new();
    for message in history {
        let started = alternated
            .iter()
            .any(|message| !matches!(message, Message::System { .. }));
        match (message, alternated.last_mut()) {
            (message @ Message::System { .. }, _) if started => held.push(message),
            (Message::User { content }, Some(Message::User { content: previous })) => {
                previous.extend(content);
            }
            (Message::Assistant(turn), Some(Message::Assistant(previous))) => {
                if previous.origin != turn.origin {
                    previous.origin = None;
                }
                previous.stop = turn.stop;
                previous.content.extend(turn.content);
            }
            (message, _) => {
                alternated.append(&mut held);
                alternated.push(message);
            }
        }
    }
    alternated.append(&mut held);
    alternated
}

/// `history` with its system messages joined into one leading message, when
/// any comes after the conversation begins.
fn leading_system(history: Vec<Message>) -> Vec<Message> {
    if !has_later_system(&history) {
        return history;
    }
    let (system, rest): (Vec<Message>, Vec<Message>) = history
        .into_iter()
        .partition(|message| matches!(message, Message::System { .. }));
    let prompt: Vec<String> = system
        .into_iter()
        .filter_map(|message| match message {
            Message::System { content } => Some(content),
            Message::User { .. } | Message::Assistant(_) => None,
        })
        .collect();
    let mut history = Vec::with_capacity(rest.len() + 1);
    if !prompt.is_empty() {
        history.push(Message::system(prompt.join("\n\n")));
    }
    // Moving a system message out may leave two user messages adjacent.
    history.extend(merge_users(rest));
    history
}

/// Call ids as the target sends them. Every provider id in the adapted
/// history is distinct: a repeated one, in one turn or across turns, takes a
/// counter. The renames of the latest turn map its results, in call order,
/// so two calls that shared an id are answered by the results that followed
/// them in turn.
#[derive(Default)]
struct Renamed {
    /// For the latest turn: each source id's ids, in call order.
    to: HashMap<CallId, std::collections::VecDeque<CallId>>,
    taken: HashSet<String>,
}

impl Renamed {
    /// A new turn begins: its results answer only its own calls.
    fn turn(&mut self) {
        self.to.clear();
    }

    /// The id the call `source` takes, wanting `wanted`: `wanted`, or, when
    /// an earlier call took it, the same id with its tail replaced by a
    /// counter until it is free. A counter of lowercase alphanumerics keeps
    /// the id's length and is legal on every wire. An id rig issued stays
    /// as it is: [`WireIds`] spells it.
    ///
    /// [`WireIds`]: crate::providers::internal::wire_ids::WireIds
    fn claim(&mut self, source: &CallId, wanted: String) -> CallId {
        let id = match source {
            CallId::Local(_) => source.clone(),
            CallId::Provider(_) => {
                let mut id = wanted.clone();
                let mut attempt: u64 = 1;
                while self.taken.contains(&id) {
                    id = with_counter(&wanted, attempt);
                    attempt += 1;
                }
                self.taken.insert(id.clone());
                if id == source.wire() {
                    source.clone()
                } else {
                    CallId::from_wire(id)
                }
            }
        };
        self.to
            .entry(source.clone())
            .or_default()
            .push_back(id.clone());
        id
    }

    /// The id a result for `source` answers: the next call of the latest
    /// turn that had it, the last one once each was answered.
    fn answer(&mut self, source: &CallId) -> Option<CallId> {
        let ids = self.to.get_mut(source)?;
        if ids.len() > 1 {
            ids.pop_front()
        } else {
            ids.front().cloned()
        }
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

/// The model a history is sent to, as a turn's origin is compared with it.
struct Same<'a> {
    api: Api,
    provider: &'a str,
    model: &'a str,
    /// The request's context, when the target binds items to it.
    context: Option<crate::message::Fingerprint>,
}

impl Same<'_> {
    /// Whether `origin` is this model, made under this context where the
    /// target binds items to it.
    fn is(&self, origin: &Origin) -> bool {
        origin.same_model(&self.api, self.provider, self.model)
            && self
                .context
                .is_none_or(|context| origin.context == Some(context))
    }
}

/// `turn` shaped for the target named by `same`.
fn assistant(
    turn: &AssistantMessage,
    target: &dyn ReplayTarget,
    same_model: &Same<'_>,
    accepts: Accepts,
    ids: &mut Renamed,
    hosted: &HashSet<usize>,
    last: bool,
) -> AssistantMessage {
    let model = same_model.model;
    let same = turn
        .origin
        .as_ref()
        .is_some_and(|origin| same_model.is(origin));
    // A failed turn is skipped, so it claims no ids a later turn may use.
    if turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
        return turn.clone();
    }
    ids.turn();
    let content: Vec<Option<AssistantContent>> = turn
        .content
        .iter()
        .map(|block| {
            let block = if same {
                match block.clone() {
                    AssistantContent::ToolCall(mut call) => {
                        let wanted = call.id.wire().into_owned();
                        let item = AssistantContent::ToolCall(call.clone())
                            .native_item()
                            .cloned()
                            .filter(|_| target.call_id_slot().is_some());
                        call.id = ids.claim(&call.id, wanted);
                        let block = AssistantContent::ToolCall(call);
                        // A rename is rig's, not an edit: replay spells the
                        // new id into the item's slot, so the item stays.
                        match item {
                            Some(item) if block.native_item().is_none() => {
                                block.canonical().with_native(item)
                            }
                            _ => block,
                        }
                    }
                    // The capability holds for the model's own turns too: an
                    // image it made but does not read back is left out.
                    AssistantContent::Image(_) if !accepts.assistant_images => return None,
                    block => block,
                }
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
                        let normalized = target.normalize_tool_call_id(
                            &call.id.wire(),
                            model,
                            turn.origin.as_ref(),
                        );
                        call.id = ids.claim(&call.id, normalized);
                        AssistantContent::ToolCall(call)
                    }
                    block => block,
                }
            };
            // Keys sorted, so the text is the same however the provider
            // ordered the arguments.
            let block = match block {
                AssistantContent::Opaque(opaque)
                    if !accepts.tools
                        && target.hosted_needs_tools()
                        && target.hosted_pair(&opaque.item).is_some() =>
                {
                    return None;
                }
                block => block,
            };
            (!block.is_blank()).then_some(block)
        })
        .collect();
    let content = if same {
        paired(content, target, accepts.tools, hosted, last)
    } else {
        content.into_iter().flatten().collect()
    };
    AssistantMessage {
        content,
        origin: turn.origin.clone(),
        stop: turn.stop.clone(),
    }
}

/// The hosted uses and results that pair, as (message, block) positions in
/// the model's own replayed turns. A use pairs with the next result of its id,
/// in its turn or a later one: a programmatic tool call's code execution
/// returns its result in the turn after the client call it made. A second use
/// of an id before its result leaves the first unpaired.
fn hosted_pairs(
    history: &[Message],
    target: &dyn ReplayTarget,
    same: &Same<'_>,
) -> HashSet<(usize, usize)> {
    let mut open: HashMap<String, (usize, usize)> = HashMap::new();
    let mut paired = HashSet::new();
    for (at, message) in history.iter().enumerate() {
        let Message::Assistant(turn) = message else {
            continue;
        };
        if turn.stop.as_ref().is_some_and(|stop| stop.is_failure())
            || !turn.origin.as_ref().is_some_and(|origin| same.is(origin))
        {
            continue;
        }
        for (index, block) in turn.content.iter().enumerate() {
            let AssistantContent::Opaque(opaque) = block else {
                continue;
            };
            if !opaque.replay {
                continue;
            }
            match target.hosted_pair(&opaque.item) {
                Some((Pairing::Use, id)) => {
                    open.insert(id, (at, index));
                }
                Some((Pairing::Result, id)) => {
                    if let Some(used) = open.remove(&id) {
                        paired.insert(used);
                        paired.insert((at, index));
                    }
                }
                None => {}
            }
        }
    }
    paired
}

/// A same-model turn's kept blocks (`None` where `adapt` dropped one), with
/// every block whose partner is gone dropped too: an item that needs the one
/// after it ([`ReplayTarget::needs_next`]), edited or not, when that one is
/// dropped or only rebuilt, and a hosted use or result whose block position
/// is not in `hosted` ([`ReplayTarget::hosted_pair`]). In the `last` turn,
/// one that ends the history or awaits a client call, a use followed only by
/// calls and opaque items is still running and stays.
fn paired(
    mut content: Vec<Option<AssistantContent>>,
    target: &dyn ReplayTarget,
    tools: bool,
    hosted: &HashSet<usize>,
    last: bool,
) -> Vec<AssistantContent> {
    let pair = |block: &AssistantContent| match block {
        AssistantContent::Opaque(opaque) if opaque.replay => target.hosted_pair(&opaque.item),
        _ => None,
    };
    for at in 0..content.len() {
        let Some((side, _)) = content.get(at).and_then(Option::as_ref).and_then(pair) else {
            continue;
        };
        let running = last
            && side == Pairing::Use
            && content
                .get(at + 1..)
                .into_iter()
                .flatten()
                .flatten()
                .all(|block| {
                    matches!(
                        block,
                        AssistantContent::ToolCall(_) | AssistantContent::Opaque(_)
                    )
                });
        if !hosted.contains(&at)
            && !running
            && let Some(slot) = content.get_mut(at)
        {
            *slot = None;
        }
    }
    for at in (0..content.len()).rev() {
        let needs = content
            .get(at)
            .and_then(Option::as_ref)
            .and_then(|block| match block {
                AssistantContent::Opaque(opaque) if opaque.replay => Some(&opaque.item),
                // An edited block rebuilt under its item's identity needs its
                // partner as much as the item itself.
                block => block.native_item().or_else(|| {
                    block
                        .stale_item()
                        .filter(|item| !target.identity(item).is_empty())
                }),
            })
            .is_some_and(|item| target.needs_next(item));
        // The partner must go back as the item the provider issued: its item,
        // or a rebuild that keeps its identity, never a block with neither.
        let next_gone =
            content
                .get(at + 1)
                .and_then(Option::as_ref)
                .is_none_or(|next| match next {
                    AssistantContent::Opaque(opaque) => !opaque.replay,
                    // A request without tools gets the call as text.
                    AssistantContent::ToolCall(_) if !tools => true,
                    next => {
                        next.native_item().is_none()
                            && next
                                .stale_item()
                                .is_none_or(|item| target.identity(item).is_empty())
                    }
                });
        if needs
            && next_gone
            && let Some(slot) = content.get_mut(at)
        {
            *slot = None;
        }
    }
    content.into_iter().flatten().collect()
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
fn user(content: &[UserContent], ids: &mut Renamed, form: &Form<'_>) -> Vec<Message> {
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
                if let Some(id) = ids.answer(&result.call) {
                    result.call = id;
                }
                result.content = result_images(result.content, form, &mut attached);
                let parts = form.accepts.tool_result_images || form.target.result_parts(form.model);
                result.content = result_text_parts(result.content, parts, result.is_error);
                shaped.push(UserContent::ToolResult(result));
                continue;
            }
            // Blank text says nothing, and several providers reject it.
            UserContent::Text(text) if text.text.trim().is_empty() => continue,
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
    let mut messages = Vec::new();
    if !shaped.is_empty() {
        messages.push(Message::User { content: shaped });
    }
    if !attached.is_empty() {
        let mut content = vec![UserContent::text(TOOL_IMAGES_HEADING)];
        content.extend(attached.into_iter().map(UserContent::Image));
        messages.push(Message::User { content });
    }
    messages
}

/// What replaces an empty tool result: models answer a call whose result
/// says nothing better than one with no content (pi).
pub const NO_TOOL_OUTPUT: &str = "(no tool output)";

/// `content` of a result that has nothing to say, said plainly, and joined
/// into one text unless the model reads several `parts`: Gemini 2 rejects a
/// result of several parts (pi `google-shared.js`).
fn result_text_parts(
    content: Vec<ToolResultContent>,
    parts: bool,
    is_error: bool,
) -> Vec<ToolResultContent> {
    let blank = content.iter().all(|part| match part {
        ToolResultContent::Text(text) => text.text.trim().is_empty(),
        ToolResultContent::Json { .. } | ToolResultContent::Image(_) => false,
    });
    if blank {
        let text = if is_error {
            format!("[tool error] {NO_TOOL_OUTPUT}")
        } else {
            NO_TOOL_OUTPUT.to_owned()
        };
        return vec![ToolResultContent::text(text)];
    }
    let texts = content
        .iter()
        .filter(|part| !matches!(part, ToolResultContent::Image(_)))
        .count();
    if parts || texts < 2 {
        return content;
    }
    let mut joined: Vec<String> = Vec::new();
    let mut images = Vec::new();
    for part in content {
        match part {
            ToolResultContent::Text(text) => joined.push(text.text),
            ToolResultContent::Json { value } => joined.push(value.to_string()),
            image @ ToolResultContent::Image(_) => images.push(image),
        }
    }
    std::iter::once(ToolResultContent::text(joined.join("\n")))
        .chain(images)
        .collect()
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
fn answer_calls(history: Vec<Option<Message>>, stored: bool, answers: bool) -> Vec<Message> {
    // Results before the first turn of a stored conversation answer calls
    // the provider holds.
    let mut stored = stored;
    let mut shaped = Vec::with_capacity(history.len());
    let mut waiting: Vec<ToolCall> = Vec::new();
    let mut held = Vec::new();
    let mut gap = false;
    // Results that answer some waiting calls while others still wait: a
    // system message between them must not end the turn's results.
    let mut pending: Vec<UserContent> = Vec::new();
    let mut pending_gap = false;
    for message in adjacent_users_merged(history) {
        let Some(message) = message else {
            close(
                &mut shaped,
                &mut waiting,
                &mut held,
                answers,
                std::mem::take(&mut pending),
                std::mem::take(&mut pending_gap),
            );
            stored = false;
            gap = true;
            continue;
        };
        match message {
            Message::Assistant(turn) => {
                close(
                    &mut shaped,
                    &mut waiting,
                    &mut held,
                    answers,
                    std::mem::take(&mut pending),
                    std::mem::take(&mut pending_gap),
                );
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
                    close(
                        &mut shaped,
                        &mut waiting,
                        &mut held,
                        answers,
                        std::mem::take(&mut pending),
                        std::mem::take(&mut pending_gap),
                    );
                    shaped.push(Message::User { content });
                    gap = false;
                    continue;
                }
                // A result answers a call of the turn just before it, once.
                let mut answered: HashSet<CallId> = pending
                    .iter()
                    .filter_map(|part| match part {
                        UserContent::ToolResult(result) => Some(result.call.clone()),
                        _ => None,
                    })
                    .collect();
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
                let only_results = !content.is_empty()
                    && content
                        .iter()
                        .all(|part| matches!(part, UserContent::ToolResult(_)));
                if pending.is_empty() {
                    pending_gap = gap;
                }
                pending.extend(content);
                gap = false;
                // pi holds a system message while calls wait, so results
                // split around one still answer the turn.
                if only_results && waiting.iter().any(|call| !answered.contains(&call.id)) {
                    continue;
                }
                close(
                    &mut shaped,
                    &mut waiting,
                    &mut held,
                    answers,
                    std::mem::take(&mut pending),
                    std::mem::take(&mut pending_gap),
                );
            }
            Message::System { .. } if !waiting.is_empty() => held.push(message),
            system => {
                gap = false;
                shaped.push(system);
            }
        }
    }
    close(
        &mut shaped,
        &mut waiting,
        &mut held,
        answers,
        pending,
        pending_gap,
    );
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
    answers: bool,
    mut content: Vec<UserContent>,
    merge: bool,
) {
    if !answers {
        waiting.clear();
    }
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
    // A held system message goes right after the results, before the user's
    // own text, as pi places it.
    let at = content
        .iter()
        .position(|part| !matches!(part, UserContent::ToolResult(_)))
        .unwrap_or(content.len());
    let text = if held.is_empty() {
        Vec::new()
    } else {
        content.split_off(at)
    };
    if !content.is_empty() {
        match shaped.last_mut() {
            Some(Message::User { content: previous }) if merge => previous.extend(content),
            _ => shaped.push(Message::User { content }),
        }
    }
    shaped.append(held);
    if !text.is_empty() {
        shaped.push(Message::User { content: text });
    }
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
