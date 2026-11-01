//! The rules pairing a conversation's tool calls with their results:
//! which result answers which call ([`answered_by`]), validation and repair
//! of a message list, the same repair over entries that keeps their ids
//! ([`normalize`]), the calls a cut-short conversation still owes, and
//! where a window of it may start.
//!
//! ```
//! use rig_core::{conversation::pairing, message::Message};
//!
//! pairing::validate(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), pairing::PairingError>(())
//! ```

use std::borrow::Borrow;
use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{Author, Entry, MessageId, ResultFacts};
use crate::message::{AssistantMessage, CallId, Message, ToolCall, ToolName, UserContent};
use crate::tool::feedback::tool_result_message;

/// Why a history is not a canonical conversation, or what [`pair`] repaired.
/// See [`validate`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PairingError {
    /// Two assistant messages in a row (index of the second).
    #[error("consecutive assistant messages at index {index}")]
    ConsecutiveAssistant {
        /// Index of the offending (second) assistant message.
        index: usize,
    },
    /// An assistant tool call nothing answered before the next assistant
    /// message, a user message with more than results, or the end.
    #[error("tool call `{call_id}` at index {index} has no result in the following message")]
    UnansweredToolCall {
        /// Index of the assistant message carrying the call.
        index: usize,
        /// The unanswered call id.
        call_id: CallId,
    },
    /// A tool result that answers no call still waiting for one.
    #[error(
        "tool result `{call_id}` at index {index} answers no call from the preceding assistant message"
    )]
    OrphanToolResult {
        /// Index of the user message carrying the result.
        index: usize,
        /// The orphan result's call id.
        call_id: CallId,
    },
}

/// A tool call's place: its reply's entry and its index among the reply's
/// tool calls. Unlike a provider's call id, no two calls of one
/// conversation share one; across agents, pair it with the agent.
///
/// ```
/// use rig_core::conversation::{CallRef, Conversation};
/// use rig_core::message::{AssistantContent, AssistantMessage, Message, ToolCall, ToolFunction, ToolName};
///
/// let look = ToolFunction::new(ToolName::new("look")?, serde_json::json!({}));
/// let call = AssistantContent::ToolCall(ToolCall::from_wire("c1", look));
/// let mut conversation = Conversation::new();
/// let reply = conversation.push(Message::Assistant(AssistantMessage::new(vec![call.clone(), call])));
/// let second = CallRef { reply, index: 1 };
/// assert_eq!(second.wire(), format!("{reply}/1"));
/// let pending = conversation.pending_calls();
/// assert_eq!(pending.get(1).map(|(at, _)| *at), Some(second));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CallRef {
    pub reply: MessageId,
    pub index: u32,
}

impl CallRef {
    /// Its text form, such as `m4/1`.
    pub fn wire(&self) -> String {
        format!("{}/{}", self.reply, self.index)
    }

    /// The tool calls of `message`, the entry `reply`, with their places.
    pub fn calls(
        reply: MessageId,
        message: &AssistantMessage,
    ) -> impl Iterator<Item = (Self, &ToolCall)> {
        (0..).zip(message.tool_calls()).map(move |(index, call)| {
            let at = Self { reply, index };
            (at, call)
        })
    }
}

/// The answer of each of `calls`, in call order: the n-th answer with an id
/// answers the n-th call with that id, as [`pair`] pairs them. `None` for a
/// call nothing answers; an answer no call waits for is dropped.
///
/// ```
/// use rig_core::conversation::pairing::answered_by;
/// use rig_core::message::CallId;
///
/// let id = CallId::from_wire("c1");
/// let answers = [(&id, "first"), (&id, "second")];
/// assert_eq!(answered_by([&id, &id, &id], answers), [Some("first"), Some("second"), None]);
/// ```
pub fn answered_by<'c, 'a, T>(
    calls: impl IntoIterator<Item = &'c CallId>,
    answers: impl IntoIterator<Item = (&'a CallId, T)>,
) -> Vec<Option<T>> {
    let mut waiting: Vec<(usize, &CallId)> = calls.into_iter().enumerate().collect();
    let mut found: Vec<Option<T>> = waiting.iter().map(|_| None).collect();
    for (id, answer) in answers {
        if let Some((at, _)) = take(&mut waiting, id, |(_, id)| id)
            && let Some(slot) = found.get_mut(at)
        {
            *slot = Some(answer);
        }
    }
    found
}

/// The item of each of `calls`, in call order: the one kept at its place,
/// else the placeless one [`answered_by`] pairs with it.
pub fn placed<T: Copy>(
    calls: &[(CallRef, &CallId)],
    items: &[(&CallId, Option<CallRef>, T)],
) -> Vec<Option<T>> {
    let placeless = items.iter().filter(|(_, at, _)| at.is_none());
    let paired = answered_by(
        calls.iter().map(|(_, id)| *id),
        placeless.map(|&(id, _, item)| (id, item)),
    );
    let at = |place: CallRef| items.iter().find(|(_, at, _)| *at == Some(place));
    let calls = calls.iter().zip(paired);
    calls
        .map(|((place, _), paired)| at(*place).map(|&(.., item)| item).or(paired))
        .collect()
}

/// Rejects what [`pair`] would repair: consecutive assistant messages, a
/// call nothing answered, and a result no waiting call asked for.
///
/// Call ids are matched as a multiset in call order, so a turn that repeats
/// an id is answered by one result per occurrence. A turn that ended in an
/// error or was aborted owes no results. A turn's results may be split over
/// adjacent user messages or around system messages, which reset the
/// consecutive-assistant check but keep calls waiting.
pub fn validate(messages: &[impl Borrow<Message>]) -> Result<(), PairingError> {
    repair(
        messages
            .iter()
            .map(|message| message.borrow().clone())
            .collect(),
    )
    .repairs
    .into_iter()
    .next()
    .map_or(Ok(()), Err)
}

/// The text of the result rig sends for a tool call nothing answered.
pub const NO_RESULT_PROVIDED: &str = "No result provided";

/// How [`pair`] reads a history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pairing {
    /// Results before the first assistant turn answer calls the provider
    /// holds for a stored conversation, so they are kept, one per id.
    pub stored: bool,
    /// Every unanswered call gets a [`NO_RESULT_PROVIDED`] result. Without
    /// it, as for a request that sends calls as text, none is made up.
    pub answers: bool,
}

impl Pairing {
    /// A history read on its own: nothing is stored, every call is answered.
    pub const fn canonical() -> Self {
        Self {
            stored: false,
            answers: true,
        }
    }
}

/// A history with its calls and results paired.
#[derive(Debug, Clone, PartialEq)]
pub struct Paired {
    /// The paired history.
    pub messages: Vec<Message>,
    /// What was repaired, in order, by input index; empty for a canonical one.
    pub repairs: Vec<PairingError>,
}

/// `messages` read as a canonical conversation and repaired into one
/// [`validate`] accepts, losing no turn: the lenient load of a
/// history from outside the protocol, such as a stored transcript, and the
/// close of a run's history at cancellation. Beyond [`pair`], an assistant
/// turn left directly after another is merged into it; `repairs` reports
/// the input as [`pair`] read it.
pub fn repair(messages: Vec<Message>) -> Paired {
    let shaped = shape(messages, true);
    Paired {
        messages: shaped.messages,
        repairs: shaped.repairs,
    }
}

/// `entries` repaired as [`repair`] repairs messages, keeping ids: what
/// the conversation holds after [`Conversation::normalize`].
///
/// [`Conversation::normalize`]: super::Conversation::normalize
#[derive(Debug, Clone, PartialEq)]
pub struct Normalized {
    /// The repaired entries, each with the id, author and made-time of the
    /// first entry it holds anything of; made-up results that no user
    /// entry follows to take them are a new entry by [`Author::rig`].
    pub entries: Vec<Entry>,
    /// What was repaired, as [`repair`] reports it.
    pub repairs: Vec<PairingError>,
    /// Each entry merged into another: the id kept, and the id merged into
    /// it.
    pub merged: Vec<(MessageId, MessageId)>,
    /// The ids of entries nothing was kept of, such as one holding only
    /// results no call asked for.
    pub dropped: Vec<MessageId>,
}

/// `entries` repaired as [`repair`] repairs their messages, but that user
/// entries side by side stay apart: an assistant entry directly after
/// another is merged into it (the later stop standing) and keeps the
/// first's id, author and made-time; a result no call waits for is
/// dropped, and an entry left empty with it; a made-up result for a call
/// nothing answered goes into the next user entry, else a new entry by
/// [`Author::rig`]; a system entry arriving while calls wait moves after
/// the entries that answer them. A list [`repair`] leaves as it is comes
/// back unchanged.
pub fn normalize(entries: Vec<Entry>) -> Normalized {
    let (messages, metas): (Vec<Message>, Vec<Meta>) = entries
        .into_iter()
        .map(|entry| {
            let message = Arc::unwrap_or_clone(entry.message);
            (message, (entry.author, entry.id, entry.made, entry.facts))
        })
        .unzip();
    let shaped = shape(messages, false);
    let mut used = HashSet::new();
    let mut merged = Vec::new();
    let mut entries = Vec::with_capacity(shaped.messages.len());
    for (message, sources) in shaped.messages.into_iter().zip(shaped.sources) {
        // The facts of every entry it holds anything of, found by call.
        let facts = sources
            .iter()
            .filter_map(|source| metas.get(*source))
            .fold(ResultFacts::default(), |facts, (.., more)| facts.and(more));
        let mut free = sources.iter().filter(|source| used.insert(**source));
        let kept = free.next().and_then(|source| metas.get(*source));
        let into = kept.and_then(|(_, id, ..)| *id);
        for gone in free.filter_map(|source| metas.get(*source)?.1) {
            if let Some(into) = into {
                merged.push((into, gone));
            }
        }
        let entry = match kept {
            Some((author, id, made, _)) => Entry {
                message: Arc::new(message),
                author: author.clone(),
                id: *id,
                made: *made,
                facts,
            },
            // Part of an entry already kept, by its author, or results
            // made up, by rig.
            None => {
                let first = sources.first().and_then(|source| metas.get(*source));
                Entry {
                    message: Arc::new(message),
                    author: first
                        .map_or_else(|| Some(Author::rig()), |(author, ..)| author.clone()),
                    id: None,
                    made: None,
                    facts,
                }
            }
        };
        entries.push(entry);
    }
    let dropped = metas.iter().enumerate();
    let dropped = dropped.filter(|(source, _)| !used.contains(source));
    Normalized {
        entries,
        repairs: shaped.repairs,
        merged,
        dropped: dropped.filter_map(|(_, (_, id, ..))| *id).collect(),
    }
}

/// An entry without its message: author, id, made-time and facts.
type Meta = (Option<Author>, Option<MessageId>, Option<u64>, ResultFacts);

/// Messages after [`pair`]'s walk and [`merge_turns`], each with the input
/// indices it holds anything of (the first is the one it stands for), and
/// what the first walk repaired.
struct Shaped {
    messages: Vec<Message>,
    sources: Vec<Vec<usize>>,
    repairs: Vec<PairingError>,
}

/// `messages` read as a canonical conversation and paired, with adjacent
/// user messages made one when `merge_users`, then adjacent turns merged,
/// paired again until none is left.
fn shape(messages: Vec<Message>, merge_users: bool) -> Shaped {
    let canonical = |messages: Vec<Message>| {
        let history = messages.into_iter().map(Some).collect();
        walk(history, Pairing::canonical(), merge_users, true)
    };
    let mut shaped = canonical(messages);
    // Each merge removes a turn, so this ends.
    while merge_turns(&mut shaped) {
        let again = canonical(std::mem::take(&mut shaped.messages));
        let sources = again.sources.iter().map(|sources| {
            let earlier = sources.iter().filter_map(|at| shaped.sources.get(*at));
            earlier.flatten().copied().collect()
        });
        shaped.sources = sources.collect();
        shaped.messages = again.messages;
    }
    shaped
}

/// Merge every assistant turn directly after another into it, the later
/// turn's stop standing; whether any was merged.
fn merge_turns(shaped: &mut Shaped) -> bool {
    let before = shaped.messages.len();
    let mut messages: Vec<Message> = Vec::with_capacity(before);
    let mut sources: Vec<Vec<usize>> = Vec::with_capacity(before);
    let all = shaped.messages.drain(..).zip(shaped.sources.drain(..));
    for (mut message, mut from) in all {
        if let Message::Assistant(turn) = &mut message
            && let Some(Message::Assistant(previous)) = messages.last_mut()
        {
            previous.content.append(&mut turn.content);
            previous.stop = turn.stop.take();
            if let Some(into) = sources.last_mut() {
                into.append(&mut from);
            }
            continue;
        }
        messages.push(message);
        sources.push(from);
    }
    shaped.messages = messages;
    shaped.sources = sources;
    shaped.messages.len() < before
}

/// The one rule for when a tool call is answered.
/// `None` is a turn the caller emptied; it ends waiting calls like a turn.
///
/// - A result answers the first call of the turn before it still waiting
///   with its id; a result nothing waits for is dropped.
/// - A turn that ended in an error or was aborted is kept; its calls wait
///   for nothing.
/// - A call still waiting at a user message with more than results, the
///   next turn or the end gets a [`close_pending`] result, before the user
///   message's own content.
/// - Adjacent non-empty user messages become one, and a system message
///   arriving while calls wait is held until they are answered.
pub fn pair(history: Vec<Option<Message>>, pairing: Pairing) -> Paired {
    let shaped = walk(history, pairing, true, false);
    Paired {
        messages: shaped.messages,
        repairs: shaped.repairs,
    }
}

/// [`pair`]'s walk, noting where each message came from when `sourced`.
/// Without `merge_users`, the content of adjacent user messages answers
/// calls as one but stays in the message it came from, and a held system
/// message goes after them whole.
fn walk(
    history: Vec<Option<Message>>,
    pairing: Pairing,
    merge_users: bool,
    sourced: bool,
) -> Shaped {
    // The ids results before the first turn took, when calls are stored.
    let mut stored = pairing.stored.then(Vec::new);
    let mut walk = Walk {
        answers: pairing.answers,
        merge_users,
        sourced,
        ..Walk::default()
    };
    for step in steps(history) {
        match step {
            Step::Turn(index, turn) => {
                if walk.after_assistant && turn.is_some() {
                    walk.repairs
                        .push(PairingError::ConsecutiveAssistant { index });
                }
                walk.close();
                stored = None;
                walk.gap = turn.is_none();
                walk.after_assistant = turn.is_some();
                let Some(turn) = turn else { continue };
                if !turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
                    let calls = turn.tool_calls();
                    let calls = calls.map(|call| (call.id.clone(), call.function.name.clone()));
                    walk.waiting = calls.collect();
                    walk.waiting_at = index;
                }
                walk.put(Message::Assistant(turn), [index]);
            }
            Step::System(index, system) => {
                walk.after_assistant = false;
                if walk.waiting.is_empty() {
                    walk.gap = false;
                    walk.put(system, [index]);
                } else {
                    walk.held.push((index, system));
                }
            }
            Step::User(first, parts) => {
                walk.after_assistant = false;
                if parts.is_empty() {
                    walk.close();
                    let empty = Message::User {
                        content: Vec::new(),
                    };
                    walk.put(empty, [first]);
                    walk.gap = false;
                    continue;
                }
                let mut content = Vec::with_capacity(parts.len());
                for (index, part) in parts {
                    let UserContent::ToolResult(result) = part else {
                        content.push((Some(index), part));
                        continue;
                    };
                    let answers = match &mut stored {
                        Some(taken) if !taken.contains(&result.call) => {
                            taken.push(result.call.clone());
                            true
                        }
                        Some(_) => false,
                        None => take(&mut walk.waiting, &result.call, |(id, _)| id).is_some(),
                    };
                    if answers {
                        content.push((Some(index), UserContent::ToolResult(result)));
                    } else {
                        walk.repairs.push(PairingError::OrphanToolResult {
                            index,
                            call_id: result.call,
                        });
                    }
                }
                if content.is_empty() && walk.waiting.is_empty() {
                    // Only results nothing waits for: the gap stays open.
                    continue;
                }
                let only_results =
                    !content.is_empty() && content.iter().all(|(_, part)| is_result(part));
                if walk.pending.is_empty() {
                    walk.pending_gap = walk.gap;
                }
                walk.pending.extend(content);
                walk.gap = false;
                // pi holds a system message while calls wait, so results
                // split around one still answer the turn.
                if only_results && !walk.waiting.is_empty() {
                    continue;
                }
                walk.close();
            }
        }
    }
    walk.close();
    Shaped {
        messages: walk.shaped,
        sources: walk.sources,
        repairs: walk.repairs,
    }
}

pub(crate) fn is_result(part: &UserContent) -> bool {
    matches!(part, UserContent::ToolResult(_))
}

/// Why a batch of results does not answer a turn's calls ([`answers`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnswerError {
    /// A part of the batch is not a tool result.
    #[error("received content that is not a tool result")]
    NotAResult,
    /// A result for an id no call has, or whose every call was answered.
    #[error("received a result for unknown or already-answered tool call id `{0}`")]
    Unknown(CallId),
    /// The calls the batch left unanswered, in call order.
    #[error("left pending tool call id(s) unanswered: {0:?}")]
    Unanswered(Vec<CallId>),
}

/// Whether `results` answer `pending` exactly, one result per occurrence of
/// an id in any order, matched as [`pair`] matches them.
pub fn answers(pending: &[CallId], results: &[UserContent]) -> Result<(), AnswerError> {
    let mut waiting = pending.to_vec();
    for result in results {
        let UserContent::ToolResult(result) = result else {
            return Err(AnswerError::NotAResult);
        };
        if take(&mut waiting, &result.call, |id| id).is_none() {
            return Err(AnswerError::Unknown(result.call.clone()));
        }
    }
    if waiting.is_empty() {
        Ok(())
    } else {
        Err(AnswerError::Unanswered(waiting))
    }
}

/// The user message closing `calls` nothing answered: a [`NO_RESULT_PROVIDED`]
/// error result per occurrence, in call order, exactly as [`pair`] closes them.
pub fn close_pending<'a>(calls: impl IntoIterator<Item = &'a ToolCall>) -> Message {
    close_pending_with(calls, NO_RESULT_PROVIDED)
}

/// [`close_pending`], each error result saying `why`, such as why the
/// calls never ran.
pub fn close_pending_with<'a>(calls: impl IntoIterator<Item = &'a ToolCall>, why: &str) -> Message {
    let calls = calls.into_iter();
    Message::User {
        content: unanswered(calls.map(|call| (&call.id, &call.function.name)), why),
    }
}

fn unanswered<'a>(
    calls: impl IntoIterator<Item = (&'a CallId, &'a ToolName)>,
    why: &str,
) -> Vec<UserContent> {
    calls
        .into_iter()
        .map(|(id, name)| tool_result_message(id.clone(), name.clone(), why.to_owned()))
        .collect()
}

/// Takes the first of `waiting` with the id `id`.
fn take<T>(waiting: &mut Vec<T>, id: &CallId, id_of: impl Fn(&T) -> &CallId) -> Option<T> {
    let at = waiting.iter().position(|call| id_of(call) == id)?;
    Some(waiting.remove(at))
}

/// One step of [`pair`]'s walk, with indices into its input.
enum Step {
    /// An assistant turn, or `None` for one the caller emptied.
    Turn(usize, Option<AssistantMessage>),
    System(usize, Message),
    /// A run of adjacent non-empty user messages, or one empty one: the
    /// index of the first, and the index each part came from.
    User(usize, Vec<(usize, UserContent)>),
}

/// `history` as steps, each run of adjacent non-empty user messages made
/// one, so results split over several messages answer the turn before them.
fn steps(history: Vec<Option<Message>>) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::with_capacity(history.len());
    for (index, message) in history.into_iter().enumerate() {
        match message {
            None => steps.push(Step::Turn(index, None)),
            Some(Message::Assistant(turn)) => steps.push(Step::Turn(index, Some(turn))),
            Some(system @ Message::System { .. }) => steps.push(Step::System(index, system)),
            Some(Message::User { content }) => {
                let joins = !content.is_empty();
                let parts = content.into_iter().map(|part| (index, part));
                match steps.last_mut() {
                    Some(Step::User(_, previous)) if joins && !previous.is_empty() => {
                        previous.extend(parts)
                    }
                    _ => steps.push(Step::User(index, parts.collect())),
                }
            }
        }
    }
    steps
}

/// The state of [`pair`]'s walk.
#[derive(Default)]
struct Walk {
    answers: bool,
    /// Whether the content of adjacent user messages becomes one message.
    merge_users: bool,
    /// Whether `sources` is kept.
    sourced: bool,
    shaped: Vec<Message>,
    /// Parallel to `shaped` when `sourced`: the input indices each holds
    /// anything of.
    sources: Vec<Vec<usize>>,
    repairs: Vec<PairingError>,
    /// Every occurrence of the last turn's calls still waiting, in call
    /// order, and the turn's index.
    waiting: Vec<(CallId, ToolName)>,
    waiting_at: usize,
    /// System messages that arrived while calls waited, with their indices.
    held: Vec<(usize, Message)>,
    /// Whether a turn was emptied since the last message pushed.
    gap: bool,
    /// Results answering some waiting calls while others still wait, and
    /// content after them, each with its index (none for one made up).
    pending: Vec<(Option<usize>, UserContent)>,
    pending_gap: bool,
    after_assistant: bool,
}

impl Walk {
    fn put(&mut self, message: Message, sources: impl IntoIterator<Item = usize>) {
        self.shaped.push(message);
        if self.sourced {
            self.sources.push(sources.into_iter().collect());
        }
    }

    /// Push the pending user content with a result added for each waiting
    /// call, then the held system messages. The results go before its first
    /// non-result part, in call order. Nothing is pushed for empty content,
    /// and content after a gap is appended to a user message that ends
    /// `shaped`. Without `merge_users`, each message's content stays its
    /// own, the made-up results go into the first, and the held system
    /// messages follow them all.
    fn close(&mut self) {
        let mut content = std::mem::take(&mut self.pending);
        let merge = std::mem::take(&mut self.pending_gap);
        let waiting = std::mem::take(&mut self.waiting);
        let missing =
            if self.answers {
                let index = self.waiting_at;
                self.repairs.extend(waiting.iter().map(|(id, _)| {
                    PairingError::UnansweredToolCall {
                        index,
                        call_id: id.clone(),
                    }
                }));
                unanswered(
                    waiting.iter().map(|(id, name)| (id, name)),
                    NO_RESULT_PROVIDED,
                )
            } else {
                Vec::new()
            };
        if !self.merge_users {
            self.close_apart(content, missing);
            return;
        }
        let at = content
            .iter()
            .position(|(_, part)| !is_result(part))
            .unwrap_or(content.len());
        content.splice(at..at, missing.into_iter().map(|part| (None, part)));
        // Results come first: Anthropic requires it, and Chat sends them as
        // tool messages that must follow the turn directly.
        content.sort_by_key(|(_, part)| !is_result(part));
        // A held system message goes right after the results, before the
        // user's own text, as pi places it.
        let at = content
            .iter()
            .position(|(_, part)| !is_result(part))
            .unwrap_or(content.len());
        let text = if self.held.is_empty() {
            Vec::new()
        } else {
            content.split_off(at)
        };
        if !content.is_empty() {
            let (sources, content) = sourced(content);
            match self.shaped.last_mut() {
                // Only `pair` leaves a gap to merge across, and it keeps no
                // sources.
                Some(Message::User { content: previous }) if merge => previous.extend(content),
                _ => self.put(Message::User { content }, sources),
            }
        }
        for (index, system) in std::mem::take(&mut self.held) {
            self.put(system, [index]);
        }
        if !text.is_empty() {
            let (sources, content) = sourced(text);
            self.put(Message::User { content }, sources);
        }
    }

    /// [`close`](Self::close) without `merge_users`: each input message's
    /// pending content as its own message, results first, the `missing`
    /// results in the first (else a message of their own), then the held
    /// system messages.
    fn close_apart(
        &mut self,
        content: Vec<(Option<usize>, UserContent)>,
        missing: Vec<UserContent>,
    ) {
        let mut messages: Vec<(Option<usize>, Vec<UserContent>)> = Vec::new();
        for (source, part) in content {
            match messages.last_mut() {
                Some((last, parts)) if *last == source => parts.push(part),
                _ => messages.push((source, vec![part])),
            }
        }
        match messages.first_mut() {
            Some((_, parts)) => {
                let at = parts.iter().position(|part| !is_result(part));
                let at = at.unwrap_or(parts.len());
                parts.splice(at..at, missing);
            }
            None if !missing.is_empty() => messages.push((None, missing)),
            None => {}
        }
        for (source, mut content) in messages {
            content.sort_by_key(|part| !is_result(part));
            self.put(Message::User { content }, source);
        }
        for (index, system) in std::mem::take(&mut self.held) {
            self.put(system, [index]);
        }
    }
}

/// `content` without its indices, and the indices it holds, each once in
/// the order first met.
fn sourced(content: Vec<(Option<usize>, UserContent)>) -> (Vec<usize>, Vec<UserContent>) {
    let mut sources: Vec<usize> = Vec::new();
    let parts = content.into_iter().map(|(source, part)| {
        if let Some(source) = source
            && !sources.contains(&source)
        {
            sources.push(source);
        }
        part
    });
    let parts = parts.collect();
    (sources, parts)
}

/// The tool calls of the last assistant message in `messages` that no
/// later tool result answers, in call order: what a conversation cut short
/// still owes the model. A turn that ended in an
/// error or was aborted owes none, as its calls never run.
pub fn pending_calls(messages: &[impl Borrow<Message>]) -> Vec<ToolCall> {
    let pending = pending(messages);
    let calls = pending.into_iter().flat_map(|(_, calls)| calls);
    calls.map(|(_, call)| call.clone()).collect()
}

/// [`pending_calls`], with the index of the message asking for them and
/// each call's place among its tool calls.
pub(crate) fn pending<M: Borrow<Message>>(
    messages: &[M],
) -> Option<(usize, Vec<(u32, &ToolCall)>)> {
    let (at, reply) = messages
        .iter()
        .map(Borrow::borrow)
        .enumerate()
        .rev()
        .find_map(|(at, message)| match message {
            Message::Assistant(reply) => Some((at, reply)),
            _ => None,
        })?;
    if reply.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
        return None;
    }
    let results = messages
        .iter()
        .skip(at + 1)
        .flat_map(|message| match message.borrow() {
            Message::User { content } => content.as_slice(),
            _ => &[],
        })
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some((&result.call, ())),
            _ => None,
        });
    let answered = answered_by(reply.tool_calls().map(|call| &call.id), results);
    let calls = (0..).zip(reply.tool_calls()).zip(answered);
    let calls = calls.filter_map(|(call, answer)| answer.is_none().then_some(call));
    Some((at, calls.collect()))
}

/// The first index at or after `at` a request may start `messages` from:
/// past every tool result before the first assistant turn from `at`, since
/// its call is cut off, with any text or system message among them. Paired
/// results later on are left alone. `at` past the end is the end.
pub fn safe_start(messages: &[impl Borrow<Message>], at: usize) -> usize {
    let mut start = at.min(messages.len());
    for (index, message) in messages.iter().enumerate().skip(at) {
        match message.borrow() {
            Message::Assistant(_) => break,
            Message::User { content }
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_))) =>
            {
                start = index + 1;
            }
            _ => {}
        }
    }
    start
}

#[cfg(test)]
mod tests;
