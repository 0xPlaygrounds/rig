//! Shaping a conversation for the model it is sent to. [`adapt`] is the one
//! place history meets its target: a turn the same model produced keeps its
//! provider items, any other turn replays from its canonical fields, failed
//! turns are skipped, and every tool call gets an answer.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_core::completion::history::{Accepts, ReplayTarget, adapt};
//! use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
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
//!     fn map_options(&self, _request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
//!         let OptionFields {
//!             reasoning,
//!             cache,
//!             service_tier,
//!             verbosity,
//!             parallel_tool_calls,
//!             top_p,
//!             seed,
//!             stop,
//!             cache_key,
//!         } = fields;
//!         let none = |set: bool| match set {
//!             true => Mapping::unsupported("the example takes no options"),
//!             false => Mapping::Nothing,
//!         };
//!         OptionMap {
//!             reasoning: none(reasoning.is_some()),
//!             cache: none(cache.is_some()),
//!             service_tier: none(service_tier.is_some()),
//!             verbosity: none(verbosity.is_some()),
//!             parallel_tool_calls: none(parallel_tool_calls.is_some()),
//!             top_p: none(top_p.is_some()),
//!             seed: none(seed.is_some()),
//!             stop: none(!stop.is_empty()),
//!             cache_key: Mapping::unrouted(cache_key),
//!         }
//!     }
//! }
//!
//! let history = vec![Message::user("hi"), Message::assistant("hello")];
//! assert_eq!(adapt(&history, &Target), history);
//! ```

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use base64::Engine as _;
use base64::prelude::{BASE64_STANDARD, BASE64_STANDARD_NO_PAD};

use crate::conversation::pairing::is_result;
use crate::message::{
    Api, AssistantContent, AssistantMessage, CallId, DocumentData, DocumentMediaType,
    DocumentSourceKind, ImageMediaType, Message, Origin, Text, ToolResult, ToolResultContent,
    UserContent,
};
use crate::wasm_compat::WasmCompatSync;

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
    /// Another model's reasoning text as reasoning: the wire has a place for
    /// reasoning that carries no provider item or signature, such as Chat
    /// Completions' `reasoning_content` field. Read only for a request that
    /// asks for it ([`CompletionRequest::foreign_reasoning`]).
    ///
    /// [`CompletionRequest::foreign_reasoning`]: field@crate::completion::CompletionRequest::foreign_reasoning
    pub foreign_reasoning: bool,
}

impl Accepts {
    /// Every kind of content. Another model's reasoning is replayed as plain
    /// text (`foreign_reasoning` is off).
    pub const ALL: Self = Self {
        user_images: true,
        assistant_images: true,
        tool_result_images: true,
        tools: true,
        foreign_reasoning: false,
    };

    /// Text and tools, no images. Another model's reasoning is replayed as
    /// plain text.
    pub const TEXT: Self = Self {
        user_images: false,
        assistant_images: false,
        tool_result_images: false,
        tools: true,
        foreign_reasoning: false,
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

    /// How this wire answers each option of `request`, whose resolved model
    /// and typed fields an answer may depend on. No default: every
    /// completion wire writes one, destructuring `fields` with no `..`, so a
    /// new option fails to compile until the wire answers for it. An unset
    /// option answers `Mapping::Nothing`, so wrap each answer in
    /// `Mapping::of` (`Mapping::of_stop` for `stop`), as in
    /// `seed: Mapping::of(seed, |_| Mapping::unsupported("no such field"))`;
    /// any other answer for an unset option fails every request.
    /// `Completion::prepare` reports each refusal before the wire encodes.
    fn map_options(
        &self,
        request: &crate::completion::CompletionRequest,
        fields: crate::completion::options::OptionFields<'_>,
    ) -> crate::completion::options::OptionMap;

    /// The model facts this wire encodes with: the spec it was connected
    /// to and the catalog that answers for other models. `None`, the
    /// default, answers from the built-in catalog.
    fn facts(&self) -> Option<&crate::catalog::ModelFacts> {
        None
    }

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
    /// then names no reason failed; a wire
    /// with no finish vocabulary, such as a local runtime, says `false`.
    fn states_finish_reason(&self) -> bool {
        true
    }

    /// Whether the catalog's per-token API prices apply to this wire's
    /// replies. A subscription backend, or a server that is usually local,
    /// says `false`: its replies then carry only a cost they report.
    fn catalog_priced(&self) -> bool {
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

    /// The target `request` is sent to, for a wire that picks its API per
    /// request. Preparing the request and folding its reply both read the
    /// target this names, so a turn records the API it was made on. `None`,
    /// the default, is this target.
    fn route(&self, request: &crate::completion::CompletionRequest) -> Option<&dyn ReplayTarget> {
        let _ = request;
        None
    }

    /// Whether the encoder sends a request's documents itself. Otherwise
    /// they join the history as text before it is adapted.
    fn takes_documents(&self) -> bool {
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
    /// Joined with the leading ones into one system message.
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
///   reasoning becomes plain text, or stays reasoning when the request asks
///   for it and the model reads it ([`Accepts::foreign_reasoning`]);
///   redacted or empty reasoning is dropped;
///   [`Opaque`](crate::message::Opaque) items are dropped and call ids go
///   through [`ReplayTarget::normalize_tool_call_id`], made distinct when two
///   calls would share one.
/// - Content the model does not read ([`ReplayTarget::accepts`]) is
///   downgraded: images become placeholder text, a tool-result image moves
///   to a user message after the results when only user images are read,
///   and without tools calls and results become text.
/// - Opaque items marked not to replay are always dropped, and so is a turn
///   that ended in an error or was aborted, with the results answering it.
/// - Calls and results pair by [`crate::conversation::pairing::pair`]: an
///   unanswered call gets a
///   [`NO_RESULT_PROVIDED`](crate::conversation::pairing::NO_RESULT_PROVIDED)
///   result and an orphan result is dropped.
/// - Adjacent user messages become one. A turn left empty is dropped, and so
///   is a user message left empty. A message that was empty to begin with is
///   kept, for the request boundary to reject.
pub fn adapt(history: &[Message], target: &dyn ReplayTarget) -> Vec<Message> {
    let history = history.iter().cloned().map(Arc::new).collect();
    let adapted = adapt_for(history, target, &Request::default(), &mut Memo::default());
    adapted.into_iter().map(Arc::unwrap_or_clone).collect()
}

/// What [`adapt_for`] keeps of a settled history for the next request
/// adapted the same way: the history up to its last turn, what the messages
/// before that turn became, and the call ids they claimed. A message's
/// shape depends only on the messages before it, but for the last turn's,
/// so a history that starts with the same messages (the same `Arc`s, or
/// equal ones) is adapted from that turn on.
#[derive(Debug, Default)]
pub(crate) struct Memo {
    shape: Option<Shape>,
    given: Vec<Arc<Message>>,
    shaped: Vec<Option<Arc<Message>>>,
    taken: HashSet<String>,
}

/// What adapting reads besides the messages.
#[derive(Debug, PartialEq)]
struct Shape {
    api: Api,
    provider: String,
    model: String,
    context: Option<crate::message::Fingerprint>,
    accepts: Accepts,
    from_user: bool,
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
    /// Whether it asks for other models' reasoning as reasoning
    /// ([`CompletionRequest::foreign_reasoning`]).
    ///
    /// [`CompletionRequest::foreign_reasoning`]: field@crate::completion::CompletionRequest::foreign_reasoning
    pub(crate) foreign_reasoning: bool,
}

impl Default for Request<'_> {
    fn default() -> Self {
        Self {
            model: None,
            stored: false,
            tools: true,
            context: None,
            foreign_reasoning: false,
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
        .map_while(|message| match &**message {
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
    // Hashed as built: `Fingerprint::of` would copy it first.
    let fields = crate::message::Canonical(&serde_json::Value::Array(fields));
    crate::message::Fingerprint::of_canonical(&fields)
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
    history: Vec<Arc<Message>>,
    target: &dyn ReplayTarget,
    request: &Request<'_>,
    memo: &mut Memo,
) -> Vec<Arc<Message>> {
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
    accepts.foreign_reasoning &= request.foreign_reasoning;
    let starts_with_user = target.starts_with_user() && !stored;
    let shape = Shape {
        api: same.api.clone(),
        provider: same.provider.to_owned(),
        model: model.to_owned(),
        context: same.context,
        accepts,
        from_user: starts_with_user,
    };
    let resumed = memo.shape.as_ref() == Some(&shape)
        && history.get(..memo.given.len()).is_some_and(|given| {
            given
                .iter()
                .zip(&memo.given)
                .all(|(message, was)| Arc::ptr_eq(message, was) || message == was)
        });
    if !resumed {
        *memo = Memo::default();
    }
    // The last message kept was the last turn, so it is adapted again.
    let from = memo.given.len().saturating_sub(1);
    let last_turn = history
        .iter()
        .rposition(|message| matches!(&**message, Message::Assistant(_)));
    let (hosted, open) = hosted_pairs(&history, from, last_turn, target, &same);
    let mut ids = Renamed {
        to: Vec::new(),
        before: &memo.taken,
        taken: HashSet::new(),
    };
    let mut claimed = Vec::new();
    let mut kept = 0;
    let form = Form {
        target,
        model,
        accepts,
    };
    // A message adapting leaves as it is goes on as the same `Arc`.
    let mut shaped: Vec<Option<Arc<Message>>> = Vec::with_capacity(history.len());
    shaped.extend(memo.shaped.iter().cloned());
    for (at, message) in history.iter().enumerate().skip(from) {
        if Some(at) == last_turn {
            claimed = ids.taken.iter().map(|id| id.to_string()).collect();
            kept = shaped.len();
        }
        match &**message {
            Message::System { content } => {
                if !content.trim().is_empty() {
                    shaped.push(Some(message.clone()));
                }
            }
            Message::User { content } => {
                shaped.extend(
                    user(message, content, &mut ids, &form)
                        .into_iter()
                        .map(Some),
                );
            }
            // A failed turn is skipped, with the results answering it.
            Message::Assistant(turn)
                if turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) =>
            {
                shaped.push(None);
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
                            .all(|message| matches!(&**message, Message::System { .. })));
                let here: HashSet<usize> = hosted
                    .iter()
                    .filter(|(message, _)| *message == at)
                    .map(|(_, block)| *block)
                    .collect();
                shaped.push(assistant(
                    message, turn, &form, &same, &mut ids, &here, last,
                ));
            }
        }
    }
    let (before, tail) = shaped.split_at(memo.shaped.len().min(shaped.len()));
    let last = before.last().and_then(Option::as_deref);
    if settled(tail, last, starts_with_user, accepts.tools) {
        drop(ids);
        // The next history adapts from the last turn on unless a hosted use
        // before it may yet pair.
        match last_turn.filter(|_| !open) {
            Some(upto) => {
                let given = history.get(memo.given.len()..=upto).unwrap_or_default();
                memo.given.extend(given.iter().cloned());
                let shaped = shaped.get(memo.shaped.len()..kept).unwrap_or_default();
                memo.shaped.extend(shaped.iter().cloned());
                memo.taken.extend(claimed);
                memo.shape = Some(shape);
            }
            None => *memo = Memo::default(),
        }
        let settled: Vec<Arc<Message>> = shaped.into_iter().flatten().collect();
        #[cfg(test)]
        {
            let shaped = settled.iter().map(|message| Some(Message::clone(message)));
            let rebuilt = reshaped(shaped.collect(), target, model, stored, accepts.tools);
            let settled: Vec<&Message> = settled.iter().map(Arc::as_ref).collect();
            assert_eq!(
                settled,
                rebuilt.iter().collect::<Vec<_>>(),
                "a settled history changed"
            );
        }
        return settled;
    }
    drop(ids);
    *memo = Memo::default();
    let shaped = shaped
        .into_iter()
        .map(|message| message.map(Arc::unwrap_or_clone));
    let shaped = reshaped(shaped.collect(), target, model, stored, accepts.tools);
    shaped.into_iter().map(Arc::new).collect()
}

/// Whether `history`, shaped message by message, is already what pairing
/// and the role rules make of it, so they can be skipped: no turn was
/// dropped, the system messages all lead, user and assistant messages
/// alternate (from a user message when `starts_with_user`), and each turn's
/// calls are answered exactly, results first, by the user message after it.
/// Without `tools`, it holds no call or result at all.
fn settled<'a>(
    history: &'a [Option<Arc<Message>>],
    mut last: Option<&'a Message>,
    starts_with_user: bool,
    tools: bool,
) -> bool {
    let mut waiting: Vec<&CallId> = Vec::new();
    for message in history {
        let Some(message) = message.as_deref() else {
            return false;
        };
        match (message, last) {
            (Message::System { .. }, None) => continue,
            (Message::System { .. }, Some(_))
            | (Message::User { .. }, Some(Message::User { .. }))
            | (Message::Assistant(_), Some(Message::Assistant(_))) => return false,
            (Message::Assistant(_), None) if starts_with_user => return false,
            // A turn follows nothing or a user message, which answered
            // every call before it, so nothing waits here.
            (Message::Assistant(turn), _) => {
                waiting.extend(turn.tool_calls().map(|call| &call.id));
                if !tools && !waiting.is_empty() {
                    return false;
                }
            }
            (Message::User { content }, _) => {
                let results = content.iter().take_while(|part| is_result(part)).count();
                let answered = content.iter().take(results).all(|part| match part {
                    UserContent::ToolResult(result) => {
                        let at = waiting.iter().position(|id| **id == result.call);
                        at.map(|at| waiting.remove(at)).is_some()
                    }
                    _ => false,
                });
                if content.is_empty()
                    || (!tools && results > 0)
                    || !answered
                    || !waiting.is_empty()
                    || content.iter().skip(results).any(is_result)
                {
                    return false;
                }
            }
        }
        last = Some(message);
    }
    waiting.is_empty()
}

/// `shaped`, adapted message by message, paired and put in the roles
/// `target` takes.
fn reshaped(
    shaped: Vec<Option<Message>>,
    target: &dyn ReplayTarget,
    model: &str,
    stored: bool,
    tools: bool,
) -> Vec<Message> {
    // Calls and results pair first, so an orphan result goes whether or not
    // the request declares tools; only then does a request without tools
    // get them as text, with no result made up for an unanswered call.
    let pairing = crate::conversation::pairing::Pairing {
        stored,
        answers: tools,
    };
    let shaped = merge_users(crate::conversation::pairing::pair(shaped, pairing).messages);
    let shaped = if tools { shaped } else { tools_as_text(shaped) };
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

/// `points`, which name `written` (a request's history before its
/// `documents` join it), as they fall in `adapted`, what [`adapt_for`] made
/// of it with `request`. A message point becomes the longest part of
/// `adapted` that the adaption of its prefix alone begins with: the point
/// never covers what follows the prefix, however the adaption dropped,
/// added or joined messages around it. A point past the end of `written`
/// goes.
pub(crate) fn cache_points(
    points: &[crate::completion::CachePoint],
    written: &[Arc<Message>],
    documents: Option<&Message>,
    adapted: &[Arc<Message>],
    target: &dyn ReplayTarget,
    request: &Request<'_>,
) -> Vec<crate::completion::CachePoint> {
    use crate::completion::CachePoint;
    let prefix_of = |turns: usize, items: Option<usize>| -> Option<Vec<Arc<Message>>> {
        let mut prefix = Vec::new();
        let mut counted = 0;
        for message in written {
            if counted == turns {
                break;
            }
            let mut message = Arc::clone(message);
            if !matches!(*message, Message::System { .. }) {
                counted += 1;
            }
            // Only the message the point cuts inside is copied.
            if let Some(items) = items.filter(|_| counted == turns) {
                match Arc::make_mut(&mut message) {
                    Message::User { content } => content.truncate(items),
                    Message::Assistant(turn) => turn.content.truncate(items),
                    Message::System { .. } => {}
                }
            }
            prefix.push(message);
        }
        (counted == turns).then_some(prefix)
    };
    points
        .iter()
        .filter_map(|point| {
            let prefix = match *point {
                CachePoint::Tools | CachePoint::System | CachePoint::Messages(0) => {
                    return Some(*point);
                }
                CachePoint::Messages(turns) => prefix_of(turns, None),
                CachePoint::Within { message, items } => message
                    .checked_add(1)
                    .and_then(|turns| prefix_of(turns, Some(items))),
            };
            let mut prefix = prefix?;
            if let Some(documents) = documents {
                crate::completion::request::insert_after_leading_system(
                    &mut prefix,
                    documents.clone(),
                );
            }
            let prefix = adapt_for(prefix, target, request, &mut Memo::default());
            Some(shared_prefix(adapted, &prefix))
        })
        .collect()
}

/// Where `prefix` stops being the start of `history`, both adapted, as a
/// cache point over `history`. System messages are not counted.
fn shared_prefix(
    history: &[Arc<Message>],
    prefix: &[Arc<Message>],
) -> crate::completion::CachePoint {
    use crate::completion::CachePoint;
    fn turns(history: &[Arc<Message>]) -> impl Iterator<Item = &Message> {
        history
            .iter()
            .map(Arc::as_ref)
            .filter(|message| !matches!(message, Message::System { .. }))
    }
    fn lead<T: PartialEq>(whole: &[T], part: &[T]) -> (usize, usize, usize) {
        let shared = whole
            .iter()
            .zip(part)
            .take_while(|(whole, part)| whole == part)
            .count();
        (shared, whole.len(), part.len())
    }
    let mut message = 0;
    for (whole, part) in turns(history).zip(turns(prefix)) {
        let (shared, whole, part) = match (whole, part) {
            (Message::User { content: whole }, Message::User { content: part }) => {
                lead(whole, part)
            }
            (Message::Assistant(whole), Message::Assistant(part)) => {
                lead(&whole.content, &part.content)
            }
            _ => (0, 1, 1),
        };
        if shared == whole && shared == part {
            message += 1;
            continue;
        }
        return if shared == 0 {
            CachePoint::Messages(message)
        } else {
            CachePoint::Within {
                message,
                items: shared,
            }
        };
    }
    CachePoint::Messages(message)
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
struct Renamed<'a> {
    /// For the latest turn: each call's source id and the id it took, in
    /// call order.
    to: Vec<(Cow<'a, CallId>, Cow<'a, CallId>)>,
    /// The ids an earlier adaption of the same messages claimed ([`Memo`]).
    before: &'a HashSet<String>,
    taken: HashSet<Cow<'a, str>>,
}

impl<'a> Renamed<'a> {
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
    fn claim(&mut self, source: Cow<'a, CallId>, wanted: Cow<'a, str>) -> Cow<'a, CallId> {
        let id = match &*source {
            CallId::Local(_) => source.clone(),
            CallId::Provider(_) => {
                let mut id = wanted.clone();
                let mut attempt: u64 = 1;
                while self.before.contains(id.as_ref()) || self.taken.contains(&id) {
                    id = Cow::Owned(with_counter(&wanted, attempt));
                    attempt += 1;
                }
                self.taken.insert(id.clone());
                if id == source.wire() {
                    source.clone()
                } else {
                    Cow::Owned(CallId::from_wire(id))
                }
            }
        };
        self.to.push((source, id.clone()));
        id
    }

    /// The id a result for `source` answers: the next call of the latest
    /// turn that had it, the last one once each was answered.
    fn answer(&mut self, source: &CallId) -> Option<Cow<'a, CallId>> {
        let mut calls = (self.to.iter().enumerate())
            .filter(|(_, (from, _))| from.as_ref() == source)
            .map(|(at, _)| at);
        let first = calls.next()?;
        if calls.next().is_some() {
            Some(self.to.remove(first).1)
        } else {
            self.to.get(first).map(|(_, id)| id.clone())
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

/// The turn `message`, holding `turn`, shaped for `form` and the target
/// named by `same`: `message` itself when nothing in it changes, nothing
/// when no block it sends alone is left.
fn assistant<'a>(
    message: &Arc<Message>,
    turn: &'a AssistantMessage,
    form: &Form<'_>,
    same_model: &Same<'_>,
    ids: &mut Renamed<'a>,
    hosted: &HashSet<usize>,
    last: bool,
) -> Option<Arc<Message>> {
    let Form {
        target,
        model,
        accepts,
    } = *form;
    let same = turn
        .origin
        .as_ref()
        .is_some_and(|origin| same_model.is(origin));
    ids.turn();
    let content: Vec<Option<Cow<'a, AssistantContent>>> = turn
        .content
        .iter()
        .map(|block| {
            let block = if same {
                match block {
                    AssistantContent::ToolCall(call) => {
                        let id = ids.claim(Cow::Borrowed(&call.id), call.id.wire());
                        if *id == call.id {
                            Cow::Borrowed(block)
                        } else {
                            // A rename is rig's, not an edit: replay spells
                            // the new id into the item's slot, so the item
                            // stays.
                            let item = block
                                .native_item()
                                .filter(|_| target.call_id_slot().is_some());
                            let id = id.into_owned();
                            let renamed = crate::message::ToolCall { id, ..call.clone() };
                            let renamed = AssistantContent::ToolCall(renamed);
                            Cow::Owned(match item {
                                Some(item) if renamed.native_item().is_none() => {
                                    renamed.into_canonical().with_native(item.clone())
                                }
                                _ => renamed,
                            })
                        }
                    }
                    // The capability holds for the model's own turns too: an
                    // image it made but does not read back is left out.
                    AssistantContent::Image(_) if !accepts.assistant_images => return None,
                    block => Cow::Borrowed(block),
                }
            } else {
                Cow::Owned(match block.canonical() {
                    AssistantContent::Reasoning(reasoning) => {
                        if reasoning.redacted || reasoning.text.trim().is_empty() {
                            return None;
                        }
                        if accepts.foreign_reasoning {
                            AssistantContent::Reasoning(reasoning)
                        } else {
                            AssistantContent::Text(Text::new(reasoning.text))
                        }
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
                        let source = Cow::Owned(call.id);
                        call.id = ids.claim(source, Cow::Owned(normalized)).into_owned();
                        AssistantContent::ToolCall(call)
                    }
                    block => block,
                })
            };
            // Keys sorted, so the text is the same however the provider
            // ordered the arguments.
            let block = match &*block {
                AssistantContent::Opaque(opaque)
                    if !accepts.tools
                        && target.hosted_needs_tools()
                        && target.hosted_pair(&opaque.item).is_some() =>
                {
                    return None;
                }
                _ => block,
            };
            (!block.is_blank()).then_some(block)
        })
        .collect();
    let content: Vec<Cow<'a, AssistantContent>> = if same {
        paired(content, target, accepts.tools, hosted, last)
    } else {
        content.into_iter().flatten().collect()
    };
    let content: Vec<Cow<'a, AssistantContent>> = content
        .into_iter()
        .map(|block| match &*block {
            AssistantContent::Image(image) if image.native.is_none() => {
                if sendable_as_is(image) && form.sends(image, Place::Assistant) {
                    return block;
                }
                let image = sendable_image(image.clone());
                Cow::Owned(
                    if matches!(image.data, DocumentSourceKind::Unknown)
                        || !target.encodes(model, Media::Image(&image, Place::Assistant))
                    {
                        AssistantContent::Text(Text::new(IMAGE_UNSENDABLE))
                    } else {
                        AssistantContent::Image(image)
                    },
                )
            }
            _ => block,
        })
        .collect();
    if !turn.content.is_empty() && !content.iter().any(|block| target.sends_alone(block)) {
        return None;
    }
    if content.len() == turn.content.len()
        && content
            .iter()
            .all(|block| matches!(block, Cow::Borrowed(_)))
    {
        return Some(message.clone());
    }
    Some(Arc::new(Message::Assistant(AssistantMessage {
        content: content.into_iter().map(Cow::into_owned).collect(),
        origin: turn.origin.clone(),
        stop: turn.stop.clone(),
    })))
}

/// The hosted uses and results that pair, as (message, block) positions in
/// the model's own replayed turns. A use pairs with the next result of its id,
/// in its turn or a later one: a programmatic tool call's code execution
/// returns its result in the turn after the client call it made. A second use
/// of an id before its result leaves the first unpaired. Read from message
/// `from` on, with whether a use was still open when message `boundary`
/// began.
fn hosted_pairs(
    history: &[Arc<Message>],
    from: usize,
    boundary: Option<usize>,
    target: &dyn ReplayTarget,
    same: &Same<'_>,
) -> (HashSet<(usize, usize)>, bool) {
    let mut open: HashMap<String, (usize, usize)> = HashMap::new();
    let mut paired = HashSet::new();
    let mut open_at_boundary = false;
    for (at, message) in history.iter().enumerate().skip(from) {
        if Some(at) == boundary {
            open_at_boundary = !open.is_empty();
        }
        let Message::Assistant(turn) = &**message else {
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
    (paired, open_at_boundary)
}

/// A same-model turn's kept blocks (`None` where `adapt` dropped one), with
/// every block whose partner is gone dropped too: an item that needs the one
/// after it ([`ReplayTarget::needs_next`]), edited or not, when that one is
/// dropped or only rebuilt, and a hosted use or result whose block position
/// is not in `hosted` ([`ReplayTarget::hosted_pair`]). In the `last` turn,
/// one that ends the history or awaits a client call, a use followed only by
/// calls and opaque items is still running and stays.
fn paired<'a>(
    mut content: Vec<Option<Cow<'a, AssistantContent>>>,
    target: &dyn ReplayTarget,
    tools: bool,
    hosted: &HashSet<usize>,
    last: bool,
) -> Vec<Cow<'a, AssistantContent>> {
    let pair = |block: &AssistantContent| match block {
        AssistantContent::Opaque(opaque) if opaque.replay => target.hosted_pair(&opaque.item),
        _ => None,
    };
    for at in 0..content.len() {
        let Some((side, _)) = content.get(at).and_then(Option::as_deref).and_then(pair) else {
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
                        **block,
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
    // The item each block goes back as, read once: its item while current,
    // or a rebuild that keeps the item's identity (an edited block rebuilt
    // under it needs its partner as much as the item itself); whether it
    // goes back as one, and whether that item needs the block after it.
    let (kept, needs): (Vec<bool>, Vec<bool>) = content
        .iter()
        .map(|block| {
            let item = block.as_deref().and_then(|block| match block {
                AssistantContent::Opaque(opaque) => opaque.replay.then_some(&opaque.item),
                block => block
                    .native()
                    .filter(|(item, current)| *current || !target.identity(item).is_empty())
                    .map(|(item, _)| item),
            });
            (
                item.is_some(),
                item.is_some_and(|item| target.needs_next(item)),
            )
        })
        .unzip();
    for at in (0..content.len()).rev() {
        // The partner must go back as the item the provider issued: its item,
        // or a rebuild that keeps its identity, never a block with neither.
        // A request without tools gets a call as text.
        let next_gone = content
            .get(at + 1)
            .and_then(Option::as_deref)
            .is_none_or(|next| {
                !kept.get(at + 1).copied().unwrap_or_default()
                    || (!tools && matches!(next, AssistantContent::ToolCall(_)))
            });
        if needs.get(at).copied().unwrap_or_default()
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

/// The user message `message`, holding `content`, shaped for `form`: one
/// message, or two when tool-result images move to a message of their own,
/// or `message` itself when nothing in it changes.
fn user<'a>(
    message: &Arc<Message>,
    content: &'a [UserContent],
    ids: &mut Renamed<'a>,
    form: &Form<'_>,
) -> Vec<Arc<Message>> {
    let mut shaped: Vec<Cow<'_, UserContent>> = Vec::with_capacity(content.len());
    let mut attached = Vec::new();
    for part in content {
        let placeholder = match part {
            UserContent::Image(image)
                if sendable_as_is(image) && form.sends(image, Place::User) =>
            {
                shaped.push(Cow::Borrowed(part));
                continue;
            }
            UserContent::Image(image) => {
                let image = sendable_image(image.clone());
                if form.sends(&image, Place::User) {
                    shaped.push(Cow::Owned(UserContent::Image(image)));
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
                    shaped.push(Cow::Owned(UserContent::Audio(audio)));
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
                    shaped.push(Cow::Owned(UserContent::Video(video)));
                    continue;
                }
                VIDEO_UNSENDABLE
            }
            UserContent::Document(document) => {
                let mut document = document.clone();
                if let DocumentData::File(source) = document.data {
                    document.data = DocumentData::File(sendable(source));
                }
                if document.data != DocumentData::File(DocumentSourceKind::Unknown)
                    && form.target.encodes(form.model, Media::Document(&document))
                {
                    shaped.push(Cow::Owned(UserContent::Document(document)));
                    continue;
                }
                // Base64 of a media type other than PDF that decodes as
                // UTF-8 is text the wire can still read.
                let text = match document.data {
                    DocumentData::Text(text) => Some(text),
                    DocumentData::File(DocumentSourceKind::Base64(data))
                        if !matches!(document.media_type, None | Some(DocumentMediaType::PDF)) =>
                    {
                        BASE64_STANDARD
                            .decode(data.as_bytes())
                            .ok()
                            .and_then(|bytes| String::from_utf8(bytes).ok())
                    }
                    DocumentData::File(_) => None,
                };
                if let Some(text) = text {
                    shaped.push(Cow::Owned(UserContent::text(text)));
                    continue;
                }
                DOCUMENT_UNSENDABLE
            }
            UserContent::ToolResult(result) => {
                let answer = ids.answer(&result.call);
                let parts = form.accepts.tool_result_images || form.target.result_parts(form.model);
                if answer.as_ref().is_none_or(|id| **id == result.call)
                    && result_as_is(result, form, parts)
                {
                    shaped.push(Cow::Borrowed(part));
                    continue;
                }
                let mut result = result.clone();
                if let Some(id) = answer {
                    result.call = id.into_owned();
                }
                result.content = result_images(result.content, form, &mut attached);
                result.content = result_text_parts(result.content, parts, result.is_error);
                shaped.push(Cow::Owned(UserContent::ToolResult(result)));
                continue;
            }
            // Blank text says nothing, and several providers reject it.
            UserContent::Text(text) if text.text.trim().is_empty() => continue,
            UserContent::Text(_) => {
                shaped.push(Cow::Borrowed(part));
                continue;
            }
        };
        let repeated = matches!(
            shaped.last().map(|part| &**part),
            Some(UserContent::Text(text)) if text.text == placeholder
        );
        if !repeated {
            shaped.push(Cow::Owned(UserContent::text(placeholder)));
        }
    }
    let unchanged =
        shaped.len() == content.len() && shaped.iter().all(|part| matches!(part, Cow::Borrowed(_)));
    if !shaped.is_empty() && unchanged && attached.is_empty() {
        return vec![message.clone()];
    }
    let mut messages = Vec::new();
    if !shaped.is_empty() {
        let content = shaped.into_iter().map(Cow::into_owned).collect();
        messages.push(Arc::new(Message::User { content }));
    }
    if !attached.is_empty() {
        let mut content = vec![UserContent::text(TOOL_IMAGES_HEADING)];
        content.extend(attached.into_iter().map(UserContent::Image));
        messages.push(Arc::new(Message::User { content }));
    }
    messages
}

/// Whether `result` is sent as it is: every image in it is sent in the
/// result as it is ([`result_images`]), and its text needs no placeholder
/// or joining ([`result_text_parts`]).
fn result_as_is(result: &ToolResult, form: &Form<'_>, parts: bool) -> bool {
    let images = result.content.iter().all(|part| match part {
        ToolResultContent::Image(image) => {
            sendable_as_is(image) && form.sends(image, Place::ToolResult)
        }
        ToolResultContent::Text(_) | ToolResultContent::Json { .. } => true,
    });
    images && !says_nothing(&result.content) && (parts || texts(&result.content) < 2)
}

/// What replaces an empty tool result: models answer a call whose result
/// says nothing better than one with no content.
pub const NO_TOOL_OUTPUT: &str = "(no tool output)";

/// `content` of a result that has nothing to say, said plainly, and joined
/// into one text unless the model reads several `parts`: Gemini 2 rejects a
/// result of several parts.
fn result_text_parts(
    content: Vec<ToolResultContent>,
    parts: bool,
    is_error: bool,
) -> Vec<ToolResultContent> {
    if says_nothing(&content) {
        let text = if is_error {
            format!("[tool error] {NO_TOOL_OUTPUT}")
        } else {
            NO_TOOL_OUTPUT.to_owned()
        };
        return vec![ToolResultContent::text(text)];
    }
    if parts || texts(&content) < 2 {
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

/// Whether a result's `content` is only blank text, or nothing.
fn says_nothing(content: &[ToolResultContent]) -> bool {
    content.iter().all(|part| match part {
        ToolResultContent::Text(text) => text.text.trim().is_empty(),
        ToolResultContent::Json { .. } | ToolResultContent::Image(_) => false,
    })
}

/// How many parts of a result's `content` are text or JSON.
fn texts(content: &[ToolResultContent]) -> usize {
    content
        .iter()
        .filter(|part| !matches!(part, ToolResultContent::Image(_)))
        .count()
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

/// Whether [`sendable_image`] leaves `image` as it is: no raw bytes to
/// encode, and no media type its bytes would name.
fn sendable_as_is(image: &crate::message::Image) -> bool {
    match &image.data {
        DocumentSourceKind::Raw(_) => false,
        DocumentSourceKind::Base64(data) => image.media_type.is_some() || sniffed(data).is_none(),
        _ => true,
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
    ImageMediaType::sniff(&bytes)
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

#[cfg(test)]
mod tests;
