//! Shaping a conversation for the model it is sent to. [`adapt`] is the one
//! place history meets its target: a turn the same model produced keeps its
//! provider items, any other turn replays from its canonical fields, failed
//! turns are skipped, and every tool call gets an answer.
//!
//! ```
//! use rig_core::completion::history::{ReplayTarget, adapt};
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

/// What replaces a tool-result image for a model without image input.
pub const TOOL_IMAGE_OMITTED: &str = "(tool image omitted: model does not support images)";

/// The model a request is sent to, as replay sees it. Every completion wire
/// implements it.
pub trait ReplayTarget: std::fmt::Debug + WasmCompatSync {
    /// The wire format.
    fn api(&self) -> Api;

    /// The provider descriptor name.
    fn provider(&self) -> &str;

    /// The model id the wire addresses.
    fn model(&self) -> &str;

    /// Whether the model reads images. When it does not, user and
    /// tool-result images are replaced by a placeholder text.
    fn accepts_images(&self) -> bool {
        true
    }

    /// The id `id` takes when a call another model made is sent to this
    /// wire. `source` is the call's origin, `None` for a hand-built turn.
    /// Results answering the call are rewritten to match.
    fn normalize_tool_call_id(&self, id: &str, source: Option<&Origin>) -> String {
        let _ = source;
        id.to_owned()
    }
}

/// `history` shaped for `target`.
///
/// - A turn whose origin names `target`'s API, provider and model keeps its
///   provider items. Its reasoning with neither text nor an item, and its
///   blank text with no item, are dropped.
/// - Any other turn keeps only canonical fields: provider items are cleared,
///   reasoning becomes plain text (redacted or empty reasoning is dropped),
///   [`Opaque`](crate::message::Opaque) items are dropped and call ids go
///   through [`ReplayTarget::normalize_tool_call_id`].
/// - Opaque items marked not to replay are always dropped, and so is a turn
///   that ended in an error or was aborted, with the results answering it.
/// - Every call left unanswered when the next user or assistant message
///   arrives, or when the history ends, gets a [`NO_RESULT_PROVIDED`] result.
///   A system message that arrives while calls wait is held until they are
///   answered.
/// - A turn left empty is dropped, and so is a user message left empty. A
///   message that was empty to begin with is kept, for the request
///   boundary to reject.
pub fn adapt(history: &[Message], target: &dyn ReplayTarget) -> Vec<Message> {
    adapt_for_model(history, target, None)
}

/// [`adapt`] for a request that names `model` in place of the wire's own.
pub(crate) fn adapt_for_model(
    history: &[Message],
    target: &dyn ReplayTarget,
    model: Option<&str>,
) -> Vec<Message> {
    let same = (
        target.api(),
        target.provider(),
        model.unwrap_or(target.model()),
    );
    let images = target.accepts_images();
    let mut renamed = HashMap::new();
    let shaped = history
        .iter()
        .filter_map(|message| match message {
            Message::System { .. } => Some(message.clone()),
            Message::User { content } => Some(Message::User {
                content: user_content(content, &renamed, images),
            }),
            Message::Assistant(turn) => {
                let shaped = assistant(turn, target, &same, &mut renamed);
                let emptied = shaped.content.is_empty() && !turn.content.is_empty();
                (!emptied).then_some(Message::Assistant(shaped))
            }
        })
        .collect();
    answer_calls(shaped)
}

/// `turn` shaped for the target `(api, provider, model)`.
fn assistant(
    turn: &AssistantMessage,
    target: &dyn ReplayTarget,
    (api, provider, model): &(Api, &str, &str),
    renamed: &mut HashMap<CallId, CallId>,
) -> AssistantMessage {
    let same = turn
        .origin
        .as_ref()
        .is_some_and(|origin| origin.same_model(api, provider, model));
    let content =
        turn.content
            .iter()
            .filter_map(|block| {
                if same {
                    return kept(block).then(|| block.clone());
                }
                let block =
                    match block.canonical() {
                        AssistantContent::Reasoning(reasoning) => {
                            if reasoning.redacted || reasoning.text.trim().is_empty() {
                                return None;
                            }
                            AssistantContent::Text(Text::new(reasoning.text))
                        }
                        AssistantContent::Opaque(_) => return None,
                        AssistantContent::ToolCall(call) => AssistantContent::ToolCall(
                            renamed_call(call, target, turn.origin.as_ref(), renamed),
                        ),
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
/// survive only with a provider item, and an opaque item only when it
/// replays.
fn kept(block: &AssistantContent) -> bool {
    match block {
        AssistantContent::Text(text) => !text.text.trim().is_empty() || text.native.is_some(),
        AssistantContent::Reasoning(reasoning) => {
            !reasoning.text.trim().is_empty() || reasoning.native.is_some()
        }
        AssistantContent::Opaque(opaque) => opaque.replay,
        AssistantContent::ToolCall(_) | AssistantContent::Image(_) => true,
    }
}

fn renamed_call(
    mut call: ToolCall,
    target: &dyn ReplayTarget,
    origin: Option<&Origin>,
    renamed: &mut HashMap<CallId, CallId>,
) -> ToolCall {
    let wire = call.id.wire();
    let normalized = target.normalize_tool_call_id(&wire, origin);
    if normalized != wire {
        let id = CallId::from_wire(normalized);
        renamed.insert(call.id.clone(), id.clone());
        call.id = id;
    }
    call
}

fn user_content(
    content: &[UserContent],
    renamed: &HashMap<CallId, CallId>,
    images: bool,
) -> Vec<UserContent> {
    let mut shaped: Vec<UserContent> = Vec::with_capacity(content.len());
    for part in content {
        match part {
            UserContent::Image(_) if !images => {
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
                if let Some(id) = renamed.get(&result.call) {
                    result.call = id.clone();
                }
                if !images {
                    result.content = without_images(result.content);
                }
                shaped.push(UserContent::ToolResult(result));
            }
            part => shaped.push(part.clone()),
        }
    }
    shaped
}

fn without_images(content: Vec<ToolResultContent>) -> Vec<ToolResultContent> {
    let mut shaped: Vec<ToolResultContent> = Vec::with_capacity(content.len());
    for part in content {
        match part {
            ToolResultContent::Image(_) => {
                if shaped.last().and_then(ToolResultContent::as_text) != Some(TOOL_IMAGE_OMITTED) {
                    shaped.push(ToolResultContent::text(TOOL_IMAGE_OMITTED));
                }
            }
            part => shaped.push(part),
        }
    }
    shaped
}

/// pi's second pass: skip failed turns, answer every unanswered call, and
/// hold system messages that arrive while calls wait.
fn answer_calls(history: Vec<Message>) -> Vec<Message> {
    let mut shaped = Vec::with_capacity(history.len());
    let mut waiting: Vec<ToolCall> = Vec::new();
    let mut held = Vec::new();
    let mut skipped = HashSet::new();
    for message in history {
        match message {
            Message::Assistant(turn) => {
                close(&mut shaped, &mut waiting, &mut held, Vec::new());
                if turn.stop.as_ref().is_some_and(|stop| stop.is_failure()) {
                    skipped.extend(turn.tool_calls().map(|call| call.id.clone()));
                    continue;
                }
                waiting = distinct(turn.tool_calls());
                shaped.push(Message::Assistant(turn));
            }
            Message::User { mut content } => {
                if content.is_empty() {
                    close(&mut shaped, &mut waiting, &mut held, Vec::new());
                    shaped.push(Message::User { content });
                    continue;
                }
                content.retain(|part| {
                    !matches!(part, UserContent::ToolResult(result) if skipped.contains(&result.call))
                });
                close(&mut shaped, &mut waiting, &mut held, content);
            }
            Message::System { .. } if !waiting.is_empty() => held.push(message),
            system => shaped.push(system),
        }
    }
    close(&mut shaped, &mut waiting, &mut held, Vec::new());
    shaped
}

/// Push the user message `content` with a result added for each `waiting`
/// call it does not answer, then the held system messages. The results go
/// before its first non-result part, in call order. Nothing is pushed for
/// an empty message.
fn close(
    shaped: &mut Vec<Message>,
    waiting: &mut Vec<ToolCall>,
    held: &mut Vec<Message>,
    mut content: Vec<UserContent>,
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
            })
        })
        .collect();
    let at = content
        .iter()
        .position(|part| !matches!(part, UserContent::ToolResult(_)))
        .unwrap_or(content.len());
    content.splice(at..at, missing);
    if !content.is_empty() {
        shaped.push(Message::User { content });
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
