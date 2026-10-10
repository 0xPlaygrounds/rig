//! Conversation validation and repair, assistant-turn classification, and
//! constructors for real or synthetic tool results.
//!
//! ```
//! use rig_core::{message::Message, transcript::validate_canonical};
//!
//! validate_canonical(&[Message::user("Hello"), Message::assistant("Hi")])?;
//! # Ok::<(), rig_core::transcript::TranscriptError>(())
//! ```

use crate::message::{
    self, AssistantContent, AssistantMessage, CallId, Message, ToolCall, ToolName,
    ToolResultContent, UserContent,
};
use crate::tool::ToolResult;

/// Why a history is not a canonical transcript, or what [`pair`] repaired.
/// See [`validate_canonical`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TranscriptError {
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

/// Rejects what [`pair`] would repair: consecutive assistant messages, a
/// call nothing answered, and a result no waiting call asked for.
///
/// Call ids are matched as a multiset in call order, so a turn that repeats
/// an id is answered by one result per occurrence. A turn that ended in an
/// error or was aborted owes no results. A turn's results may be split over
/// adjacent user messages or around system messages, which reset the
/// consecutive-assistant check but keep calls waiting.
pub fn validate_canonical(messages: &[Message]) -> Result<(), TranscriptError> {
    repair(messages.to_vec())
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
    pub repairs: Vec<TranscriptError>,
}

/// `messages` read as a canonical transcript and repaired into one
/// [`validate_canonical`] accepts, losing no turn: the lenient load of a
/// history from outside the protocol, such as a memory backend, and the
/// close of a run's history at cancellation. Beyond [`pair`], an assistant
/// turn left directly after another is merged into it; `repairs` reports
/// the input as [`pair`] read it.
pub fn repair(messages: Vec<Message>) -> Paired {
    let canonical = |messages: Vec<Message>| {
        pair(
            messages.into_iter().map(Some).collect(),
            Pairing::canonical(),
        )
    };
    let mut paired = canonical(messages);
    // Each merge removes a turn, so this ends.
    while merge_turns(&mut paired.messages) {
        paired.messages = canonical(std::mem::take(&mut paired.messages)).messages;
    }
    paired
}

/// Merge every assistant turn directly after another into it, the later
/// turn's stop standing; whether any was merged.
fn merge_turns(messages: &mut Vec<Message>) -> bool {
    let before = messages.len();
    let mut merged: Vec<Message> = Vec::with_capacity(before);
    for mut message in messages.drain(..) {
        if let Message::Assistant(turn) = &mut message
            && let Some(Message::Assistant(previous)) = merged.last_mut()
        {
            previous.content.append(&mut turn.content);
            previous.stop = turn.stop.take();
            continue;
        }
        merged.push(message);
    }
    *messages = merged;
    messages.len() < before
}

/// The one rule for when a tool call is answered, after pi's pairing pass.
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
    // The ids results before the first turn took, when calls are stored.
    let mut stored = pairing.stored.then(Vec::new);
    let mut walk = Walk {
        answers: pairing.answers,
        ..Walk::default()
    };
    for step in steps(history) {
        match step {
            Step::Turn(index, turn) => {
                if walk.after_assistant && turn.is_some() {
                    walk.repairs
                        .push(TranscriptError::ConsecutiveAssistant { index });
                }
                walk.close();
                stored = None;
                walk.gap = turn.is_none();
                walk.after_assistant = turn.is_some();
                let Some(turn) = turn else { continue };
                if !turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
                    walk.waiting = turn.tool_calls().cloned().collect();
                    walk.waiting_at = index;
                }
                walk.shaped.push(Message::Assistant(turn));
            }
            Step::System(system) => {
                walk.after_assistant = false;
                if walk.waiting.is_empty() {
                    walk.gap = false;
                    walk.shaped.push(system);
                } else {
                    walk.held.push(system);
                }
            }
            Step::User(parts) => {
                walk.after_assistant = false;
                if parts.is_empty() {
                    walk.close();
                    walk.shaped.push(Message::User {
                        content: Vec::new(),
                    });
                    walk.gap = false;
                    continue;
                }
                let mut content = Vec::with_capacity(parts.len());
                for (index, part) in parts {
                    let UserContent::ToolResult(result) = part else {
                        content.push(part);
                        continue;
                    };
                    let answers = match &mut stored {
                        Some(taken) if !taken.contains(&result.call) => {
                            taken.push(result.call.clone());
                            true
                        }
                        Some(_) => false,
                        None => answer(&mut walk.waiting, &result.call, |call| &call.id),
                    };
                    if answers {
                        content.push(UserContent::ToolResult(result));
                    } else {
                        walk.repairs.push(TranscriptError::OrphanToolResult {
                            index,
                            call_id: result.call,
                        });
                    }
                }
                if content.is_empty() && walk.waiting.is_empty() {
                    // Only results nothing waits for: the gap stays open.
                    continue;
                }
                let only_results = !content.is_empty()
                    && content
                        .iter()
                        .all(|part| matches!(part, UserContent::ToolResult(_)));
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
    Paired {
        messages: walk.shaped,
        repairs: walk.repairs,
    }
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
        if !answer(&mut waiting, &result.call, |id| id) {
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
    Message::User {
        content: unanswered(calls, why),
    }
}

fn unanswered<'a>(calls: impl IntoIterator<Item = &'a ToolCall>, why: &str) -> Vec<UserContent> {
    calls
        .into_iter()
        .map(|call| {
            tool_result_message(call.id.clone(), call.function.name.clone(), why.to_owned())
        })
        .collect()
}

/// Take the first of `waiting` with the id `id`; false when none has it.
fn answer<T>(waiting: &mut Vec<T>, id: &CallId, id_of: impl Fn(&T) -> &CallId) -> bool {
    match waiting.iter().position(|call| id_of(call) == id) {
        Some(at) => {
            waiting.remove(at);
            true
        }
        None => false,
    }
}

/// One step of [`pair`]'s walk, with indices into its input.
enum Step {
    /// An assistant turn, or `None` for one the caller emptied.
    Turn(usize, Option<AssistantMessage>),
    System(Message),
    /// A run of adjacent non-empty user messages, or one empty one, with the
    /// index each part came from.
    User(Vec<(usize, UserContent)>),
}

/// `history` as steps, each run of adjacent non-empty user messages made
/// one, so results split over several messages answer the turn before them.
fn steps(history: Vec<Option<Message>>) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::with_capacity(history.len());
    for (index, message) in history.into_iter().enumerate() {
        match message {
            None => steps.push(Step::Turn(index, None)),
            Some(Message::Assistant(turn)) => steps.push(Step::Turn(index, Some(turn))),
            Some(system @ Message::System { .. }) => steps.push(Step::System(system)),
            Some(Message::User { content }) => {
                let joins = !content.is_empty();
                let parts = content.into_iter().map(|part| (index, part));
                match steps.last_mut() {
                    Some(Step::User(previous)) if joins && !previous.is_empty() => {
                        previous.extend(parts)
                    }
                    _ => steps.push(Step::User(parts.collect())),
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
    shaped: Vec<Message>,
    repairs: Vec<TranscriptError>,
    /// Every occurrence of the last turn's calls still waiting, in call
    /// order, and the turn's index.
    waiting: Vec<ToolCall>,
    waiting_at: usize,
    /// System messages that arrived while calls waited.
    held: Vec<Message>,
    /// Whether a turn was emptied since the last message pushed.
    gap: bool,
    /// Results answering some waiting calls while others still wait.
    pending: Vec<UserContent>,
    pending_gap: bool,
    after_assistant: bool,
}

impl Walk {
    /// Push the pending user content with a result added for each waiting
    /// call, then the held system messages. The results go before its first
    /// non-result part, in call order. Nothing is pushed for empty content,
    /// and content after a gap is appended to a user message that ends
    /// `shaped`.
    fn close(&mut self) {
        let mut content = std::mem::take(&mut self.pending);
        let merge = std::mem::take(&mut self.pending_gap);
        let waiting = std::mem::take(&mut self.waiting);
        let missing =
            if self.answers {
                let index = self.waiting_at;
                self.repairs.extend(waiting.iter().map(|call| {
                    TranscriptError::UnansweredToolCall {
                        index,
                        call_id: call.id.clone(),
                    }
                }));
                unanswered(&waiting, NO_RESULT_PROVIDED)
            } else {
                Vec::new()
            };
        let at = content
            .iter()
            .position(|part| !matches!(part, UserContent::ToolResult(_)))
            .unwrap_or(content.len());
        content.splice(at..at, missing);
        // Results come first: Anthropic requires it, and Chat sends them as
        // tool messages that must follow the turn directly.
        content.sort_by_key(|part| !matches!(part, UserContent::ToolResult(_)));
        // A held system message goes right after the results, before the
        // user's own text, as pi places it.
        let at = content
            .iter()
            .position(|part| !matches!(part, UserContent::ToolResult(_)))
            .unwrap_or(content.len());
        let text = if self.held.is_empty() {
            Vec::new()
        } else {
            content.split_off(at)
        };
        if !content.is_empty() {
            match self.shaped.last_mut() {
                Some(Message::User { content: previous }) if merge => previous.extend(content),
                _ => self.shaped.push(Message::User { content }),
            }
        }
        self.shaped.append(&mut self.held);
        if !text.is_empty() {
            self.shaped.push(Message::User { content: text });
        }
    }
}

/// Shape a tool's result as the tool result the model reads, without
/// reparsing text. Anything but a success is an error result.
pub fn tool_result_output(call: CallId, name: ToolName, result: &ToolResult) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: result.output().clone().into_content(),
        is_error: !result.is_success(),
    })
}

/// Constructs the error result of a call that never ran, its text verbatim,
/// such as recovery feedback or a skip reason. JSON-shaped text is not
/// reinterpreted as structured or multimodal output.
pub fn tool_result_message(call: CallId, name: ToolName, message: String) -> UserContent {
    UserContent::ToolResult(message::ToolResult {
        call,
        name,
        content: vec![ToolResultContent::text(message)],
        is_error: true,
    })
}

/// What the model reads when its call to `tool` sent `raw`, arguments that
/// are not a JSON object: the tool never ran, and the model calls it again.
pub fn invalid_arguments_feedback(tool: &str, raw: &str) -> String {
    format!(
        "The arguments for tool `{tool}` are not a JSON object: {raw}. \
         Call the tool again with a JSON object as its arguments."
    )
}

/// The result every other call of a turn gets when one call was retried or
/// skipped: none of the turn's calls ran.
pub const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

/// The tool results answering a turn with an invalid call, in call order:
/// `feedback` for the call `invalid`, [`TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER`]
/// for every other call. Empty when `content` has no tool calls.
pub fn invalid_call_feedback(
    content: &[AssistantContent],
    invalid: &CallId,
    feedback: &str,
) -> Vec<UserContent> {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) if &call.id == invalid => Some(tool_result_message(
                call.id.clone(),
                call.function.name.clone(),
                feedback.to_owned(),
            )),
            AssistantContent::ToolCall(call) => Some(not_executed(call)),
            _ => None,
        })
        .collect()
}

/// The result of a call the run did not execute because a call beside it
/// was invalid.
pub fn not_executed(call: &ToolCall) -> UserContent {
    tool_result_message(
        call.id.clone(),
        call.function.name.clone(),
        TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_owned(),
    )
}

/// Whether a generated assistant turn is empty: every part is blank
/// ([`AssistantContent::is_blank`], the rule replay drops parts by). An
/// empty turn must not enter history.
pub fn is_empty_assistant_turn(content: &[AssistantContent]) -> bool {
    content.iter().all(AssistantContent::is_blank)
}

/// The text of a final answer: the text parts of the model's message,
/// joined by blank lines and trimmed. `None` when `message` is not the
/// model's, still asks for tool calls, or has no text.
///
/// ```
/// use rig_core::{message::Message, transcript::final_answer};
///
/// assert_eq!(final_answer(&Message::assistant(" Done. ")).as_deref(), Some("Done."));
/// assert_eq!(final_answer(&Message::user("Done.")), None);
/// ```
pub fn final_answer(message: &Message) -> Option<String> {
    let Message::Assistant(reply) = message else {
        return None;
    };
    let mut parts = Vec::new();
    for item in reply.content.iter() {
        match item {
            AssistantContent::Text(text) => parts.push(text.text.as_str()),
            AssistantContent::ToolCall(_) => return None,
            _ => {}
        }
    }
    let text = parts.join("\n\n");
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The text parts of an assistant turn, concatenated.
pub fn assistant_text_from_choice(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Why `call`'s arguments do not fit a tool whose JSON schema is
/// `parameters`, for the model to read as the call's error: they are not a
/// JSON object ([`invalid_arguments_feedback`]), or name an argument the
/// schema does not declare. `None` when they fit as far as that goes; the
/// tool checks their values.
pub fn arguments_refusal(parameters: &serde_json::Value, call: &ToolCall) -> Option<String> {
    let name = call.function.name.as_str();
    if let Some(raw) = &call.function.invalid_arguments {
        return Some(invalid_arguments_feedback(name, raw));
    }
    let declared = parameters
        .get("properties")
        .and_then(serde_json::Value::as_object);
    fn quoted<'a>(names: impl Iterator<Item = &'a String>) -> String {
        names
            .map(|arg| format!("`{arg}`"))
            .collect::<Vec<_>>()
            .join(", ")
    }
    let unknown = quoted(
        call.function
            .arguments
            .keys()
            .filter(|arg| !declared.is_some_and(|declared| declared.contains_key(arg.as_str()))),
    );
    if unknown.is_empty() {
        return None;
    }
    let known = declared
        .map(|declared| quoted(declared.keys()))
        .filter(|known| !known.is_empty())
        .unwrap_or_else(|| "none".to_owned());
    Some(format!(
        "`{name}` has no argument {unknown}. Its arguments are: {known}. Call it again with only \
         those."
    ))
}

/// The tool calls of the last assistant message in `messages` that no
/// later tool result answers, in call order: what a conversation cut short
/// still owes the model.
pub fn pending_calls(messages: &[Message]) -> Vec<ToolCall> {
    let Some(at) = messages
        .iter()
        .rposition(|message| matches!(message, Message::Assistant(_)))
    else {
        return Vec::new();
    };
    let mut later = messages.iter().skip(at);
    let Some(Message::Assistant(reply)) = later.next() else {
        return Vec::new();
    };
    let answered: Vec<&CallId> = later
        .flat_map(|message| match message {
            Message::User { content } => content.as_slice(),
            _ => &[],
        })
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(&result.call),
            _ => None,
        })
        .collect();
    reply
        .content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) if !answered.contains(&&call.id) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod validator_tests;
