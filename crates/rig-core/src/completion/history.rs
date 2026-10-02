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

use crate::message::{
    Api, AssistantContent, AssistantMessage, CallId, Message, Origin, Text, ToolCall, ToolResult,
    ToolResultContent, UserContent,
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
                shaped.extend(user(content, &ids, accepts).into_iter().map(Some));
            }
            Message::Assistant(turn) => {
                let adapted = assistant(turn, target, &same, accepts, &mut ids);
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

/// Whether a block has anything to send: blank text and empty reasoning
/// survive only with a provider item that is still current, and an opaque
/// item only when it replays.
fn kept(block: &AssistantContent) -> bool {
    let current = block.native_item().is_some();
    match block {
        AssistantContent::Text(text) => !text.text.trim().is_empty() || current,
        AssistantContent::Reasoning(reasoning) => !reasoning.text.trim().is_empty() || current,
        AssistantContent::Opaque(opaque) => opaque.replay,
        AssistantContent::ToolCall(_) | AssistantContent::Image(_) => true,
    }
}

/// The user message `content` shaped for a model that reads `accepts`: one
/// message, or two when tool-result images move to a message of their own.
fn user(content: &[UserContent], ids: &Renamed, accepts: Accepts) -> Vec<Message> {
    let mut shaped: Vec<UserContent> = Vec::with_capacity(content.len());
    let mut attached = Vec::new();
    for part in content {
        match part {
            UserContent::Image(_) if !accepts.user_images => {
                let omitted = matches!(
                    shaped.last(),
                    Some(UserContent::Text(text)) if text.text == USER_IMAGE_OMITTED
                );
                if !omitted {
                    shaped.push(UserContent::text(USER_IMAGE_OMITTED));
                }
            }
            UserContent::ToolResult(result) => {
                let mut result = result.clone();
                if let Some(id) = ids.to.get(&result.call) {
                    result.call = id.clone();
                }
                if !accepts.tool_result_images {
                    result.content =
                        without_images(result.content, accepts.user_images, &mut attached);
                }
                if accepts.tools {
                    shaped.push(UserContent::ToolResult(result));
                } else {
                    shaped.push(UserContent::text(result_text(&result)));
                }
            }
            part => shaped.push(part.clone()),
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

/// `content` without images: each becomes [`TOOL_IMAGE_ATTACHED`] and moves
/// to `attached` when the model reads user images, else
/// [`TOOL_IMAGE_OMITTED`].
fn without_images(
    content: Vec<ToolResultContent>,
    user_images: bool,
    attached: &mut Vec<crate::message::Image>,
) -> Vec<ToolResultContent> {
    let mut shaped: Vec<ToolResultContent> = Vec::with_capacity(content.len());
    for part in content {
        match part {
            ToolResultContent::Image(image) => {
                let placeholder = if user_images {
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
