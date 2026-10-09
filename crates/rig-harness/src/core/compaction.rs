//! Compaction: when the conversation nears the model's context window, or
//! the user asks with `/compact`, its older messages are replaced in
//! requests by a structured summary the model writes. The messages stay in
//! the [`Conversation`](super::agent::Conversation), so views still show
//! them: an agent's [`Compacted`] says how many of them requests leave
//! out, and what goes in their place. Its log records the compaction, and a
//! restored session starts from the first message it kept.
//!
//! Automatic compaction first clears old tool outputs, which costs no model
//! call; only when that is not enough does it summarize. The summary is a
//! model call of its own, dispatched and recorded like any other, on a call
//! entity of the turn with a [`Summarizing`], so interrupting the turn
//! cancels it. The format and prompts follow pi's
//! (`references/pi/packages/coding-agent/src/core/compaction/compaction.ts:507-579`).

use std::collections::BTreeSet;
use std::fmt::Write as _;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, Message,
};
use rig_core::effect::Outcome;
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolResultContent, UserContent};
use rig_core::serve::Reply;
use rig_memory::{HeuristicTokenCounter, TokenCounter};
use serde::{Deserialize, Serialize};

/// Tokens left free below the model's window: past `window - RESERVE` the
/// conversation is compacted before the next call (pi's `reserveTokens`).
pub const RESERVE: u64 = 16_384;
/// Tokens of the newest messages a compaction keeps as they are, at most a
/// quarter of the window (pi's `keepRecentTokens`).
const KEEP_RECENT: u64 = 20_000;
/// Compactions a turn may make, asked for or not.
pub const MAX_COMPACTIONS: u32 = 2;
/// The longest summary asked for.
const SUMMARY_TOKENS: u64 = 12_000;
/// Characters of a tool's output or arguments the summarizer sees.
const SNIPPET_CHARS: usize = 2_000;
/// Characters of conversation sent to the summarizer when the model's
/// window is not known.
const DEFAULT_SUMMARIZED_CHARS: usize = 400_000;

/// Which of the agent's messages requests leave out, and the summary sent
/// in their place. The default leaves out none.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Default, Clone, Debug, Serialize, Deserialize)]
pub struct Compacted {
    /// How many of the conversation's first messages the summary replaces.
    pub upto: usize,
    /// The summary, in the sections the summarizer is asked for; empty before the
    /// first compaction.
    pub summary: String,
    /// Files the summarized messages read and did not change.
    pub read: BTreeSet<String>,
    /// Files the summarized messages changed.
    pub modified: BTreeSet<String>,
}

impl Compacted {
    /// The messages requests still send as they are.
    pub fn live<'a>(&self, messages: &'a [Message]) -> &'a [Message] {
        messages
            .get(self.upto.min(messages.len())..)
            .unwrap_or_default()
    }

    /// [`Self::live`], to change in place.
    pub fn live_mut<'a>(&self, messages: &'a mut [Message]) -> &'a mut [Message] {
        let from = self.upto.min(messages.len());
        messages.get_mut(from..).unwrap_or_default()
    }

    /// The messages a request sends: the summary, then the live messages.
    /// The summary goes into the first live message when that is the
    /// user's, so user and assistant messages still alternate.
    pub fn request(&self, messages: &[Message]) -> Vec<Message> {
        let mut live = self.live(messages).to_vec();
        if self.summary.is_empty() {
            return live;
        }
        let summary = UserContent::text(self.message());
        match live.first_mut() {
            Some(Message::User { content }) => content.insert(0, summary),
            _ => live.insert(
                0,
                Message::User {
                    content: vec![summary],
                },
            ),
        }
        live
    }

    /// The estimated tokens of [`Self::request`]'s messages.
    pub fn estimate(&self, messages: &[Message]) -> u64 {
        let summary = if self.summary.is_empty() {
            0
        } else {
            estimate_content(&UserContent::text(self.message()))
        };
        summary + estimate(self.live(messages))
    }

    /// The text that stands for the summarized messages.
    fn message(&self) -> String {
        let mut text = format!(
            "The conversation before this point was compacted into this summary:\n\n\
             <summary>\n{}\n</summary>",
            self.summary.trim()
        );
        let read: Vec<&str> = self
            .read
            .difference(&self.modified)
            .map(String::as_str)
            .collect();
        if !read.is_empty() {
            let _ = write!(text, "\n\n<read-files>\n{}\n</read-files>", read.join("\n"));
        }
        if !self.modified.is_empty() {
            let modified: Vec<&str> = self.modified.iter().map(String::as_str).collect();
            let _ = write!(
                text,
                "\n\n<modified-files>\n{}\n</modified-files>",
                modified.join("\n")
            );
        }
        text
    }

    /// Where a new compaction should end: the first message it keeps.
    /// Walking back from the newest message, it keeps at least
    /// 20k tokens (a quarter of the window at most) and cuts
    /// before a reply or a user's own message, never between a tool call
    /// and its result. With `force`, a shorter conversation still has its
    /// older messages summarized, all but the newest reply or message.
    /// `None` when nothing new would be summarized.
    pub fn cut(&self, messages: &[Message], spec: &ModelSpec, force: bool) -> Option<usize> {
        let keep = spec
            .context_window
            .map_or(KEEP_RECENT, |window| KEEP_RECENT.min(u64::from(window) / 4));
        let mut kept = 0u64;
        let mut newest = None;
        for (index, message) in messages.iter().enumerate().skip(self.upto + 1).rev() {
            kept += count(message);
            if !can_start_live(message) {
                continue;
            }
            newest.get_or_insert(index);
            if kept >= keep {
                return Some(index);
            }
        }
        newest.filter(|_| force)
    }
}

/// Whether the live messages may start with `message`: a reply, or a user
/// message that answers no tool call.
fn can_start_live(message: &Message) -> bool {
    match message {
        Message::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        Message::Assistant(_) | Message::System { .. } => true,
    }
}

/// The estimated tokens of a message, by rig-memory's heuristic (four bytes
/// a token).
fn count(message: &Message) -> u64 {
    HeuristicTokenCounter::default().count(message) as u64
}

/// The estimated tokens of `messages`.
pub fn estimate(messages: &[Message]) -> u64 {
    messages.iter().map(count).sum()
}

/// The estimated tokens of one item of a user message, by the same
/// heuristic.
pub(crate) fn estimate_content(content: &UserContent) -> u64 {
    HeuristicTokenCounter::default().count_user(content) as u64
}

/// Whether a request of `tokens` leaves less than [`RESERVE`] of `spec`'s
/// window free. Never, when the window is not known.
pub fn over_threshold(tokens: u64, spec: &ModelSpec) -> bool {
    spec.context_window
        .is_some_and(|window| tokens > u64::from(window).saturating_sub(RESERVE))
}

/// Why a conversation is compacted.
#[derive(Clone, Debug, PartialEq, Eq, Reflect)]
pub enum CompactReason {
    /// The user asked, with what to focus on, if anything. The turn ends
    /// with the compaction.
    Asked {
        /// What the summary should keep above all; may be empty.
        focus: String,
    },
    /// The next request would leave less than [`RESERVE`] free; the turn
    /// carries on with the model call.
    Threshold,
    /// The model refused the request as too long; the turn carries on with
    /// the model call.
    Overflow,
}

/// Summarize the older messages of the turn's agent.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Summarize {
    /// The turn.
    pub entity: Entity,
    /// Why.
    pub reason: CompactReason,
}

/// A summary model call, on a call entity of the turn; its task is a
/// [`Running<Summary>`](super::calls::Running).
#[derive(Component, Clone, Debug)]
pub struct Summarizing {
    /// Why.
    pub reason: CompactReason,
    /// The first message the compaction keeps.
    pub upto: usize,
    /// How many messages the summary replaces anew.
    pub messages: usize,
    /// Their estimated tokens.
    pub tokens: u64,
    /// The files read, with those of earlier compactions.
    pub read: BTreeSet<String>,
    /// The files changed, with those of earlier compactions.
    pub modified: BTreeSet<String>,
}

/// What a summary call's task returns.
pub struct Summary(pub Result<CompletionResponse, ErrorReport>);

/// Waits for the summary call's reply, streamed or not, as one response.
pub(crate) async fn summarize(
    reply: impl Future<Output = Reply>,
) -> Result<CompletionResponse, ErrorReport> {
    match reply.await.into_outcome().await? {
        Outcome::Completion(response) => Ok(response),
        _ => Err(ErrorReport::new(
            ErrorKind::Internal,
            "the model answered the summary request with something other than a completion",
        )),
    }
}

/// The summary in a finished reply, or why it cannot be used: a summary
/// cut at the output limit is incomplete and must not replace anything.
pub fn summary_text(response: &CompletionResponse) -> Result<String, String> {
    if matches!(response.finish_reason(), Some(FinishReason::Length)) {
        return Err("the summary hit the model's output limit".to_owned());
    }
    if response.tool_calls().next().is_some() {
        return Err("the model called a tool instead of summarizing".to_owned());
    }
    let text = response.text();
    let text = text.trim();
    if text.is_empty() {
        return Err("the model returned no summary".to_owned());
    }
    Ok(text.to_owned())
}

/// What a compaction ending at `upto` summarizes anew: the messages from
/// the last compaction to `upto`, the files they read and changed, and the
/// request for the summary to `spec`. `focus` is what the user asked to
/// keep above all.
pub fn plan(
    compacted: &Compacted,
    messages: &[Message],
    upto: usize,
    spec: &ModelSpec,
    reason: CompactReason,
) -> Result<(Summarizing, CompletionRequest), String> {
    let older = messages
        .get(compacted.upto.min(upto)..upto)
        .ok_or("the conversation is shorter than the compaction")?;
    let mut read = compacted.read.clone();
    let mut modified = compacted.modified.clone();
    touched_files(older, &mut read, &mut modified);
    let focus = match &reason {
        CompactReason::Asked { focus } => focus.trim(),
        _ => "",
    };
    let request = summary_request(older, &compacted.summary, focus, spec)?;
    let summarizing = Summarizing {
        reason,
        upto,
        messages: older.len(),
        tokens: estimate(older),
        read,
        modified,
    };
    Ok((summarizing, request))
}

/// The request for a summary of `older`, merged into `previous` when there
/// is one. The conversation goes as text in one user message, so the
/// model reads it rather than continues it.
fn summary_request(
    older: &[Message],
    previous: &str,
    focus: &str,
    spec: &ModelSpec,
) -> Result<CompletionRequest, String> {
    let budget = spec
        .context_window
        .map_or(DEFAULT_SUMMARIZED_CHARS, |window| {
            let tokens = u64::from(window).saturating_sub(RESERVE + SUMMARY_TOKENS);
            // Three characters a token, to stay under the window with code.
            usize::try_from(tokens.saturating_mul(3)).unwrap_or(usize::MAX)
        });
    let conversation = serialize(older);
    let conversation = keep_end(&conversation, budget.saturating_sub(previous.len()));
    let mut prompt = format!("<conversation>\n{conversation}\n</conversation>\n\n");
    if previous.is_empty() {
        prompt.push_str(INITIAL_PROMPT);
    } else {
        let _ = write!(
            prompt,
            "<previous-summary>\n{previous}\n</previous-summary>\n\n{UPDATE_PROMPT}"
        );
    }
    if !focus.is_empty() {
        let _ = write!(prompt, "\n\nAdditional focus: {focus}");
    }
    prompt.push_str(FORMAT);
    let options = spec.default_options(None);
    spec.validate(&options)
        .map_err(|refusal| refusal.to_string())?;
    let max_tokens = spec
        .max_output_tokens
        .map_or(SUMMARY_TOKENS, |most| SUMMARY_TOKENS.min(u64::from(most)));
    Ok(CompletionRequest::new(Message::user(prompt))
        .preamble(SYSTEM_PROMPT.to_owned())
        .options(options)
        .max_tokens(max_tokens))
}

/// The last `budget` bytes of `text`, cut at a character, with a note when
/// the start is left out.
fn keep_end(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_owned();
    }
    let start = text.ceil_char_boundary(text.len() - budget);
    let rest = text.get(start..).unwrap_or_default();
    format!("[… {start} earlier characters left out]\n{rest}")
}

/// `messages` as text, one labelled block per part, with long tool outputs
/// and arguments cut (pi's `serializeConversation`,
/// `references/pi/packages/coding-agent/src/core/compaction/utils.ts:114-159`).
fn serialize(messages: &[Message]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for message in messages {
        match message {
            Message::System { content } => parts.push(format!("[System]: {content}")),
            Message::User { content } => {
                for item in content {
                    match item {
                        UserContent::Text(text) => parts.push(format!("[User]: {}", text.text)),
                        UserContent::ToolResult(result) => {
                            let output: Vec<String> = result
                                .content
                                .iter()
                                .map(|part| match part {
                                    ToolResultContent::Text(text) => text.text.clone(),
                                    ToolResultContent::Json { value } => value.to_string(),
                                    ToolResultContent::Image(_) => "[image]".to_owned(),
                                })
                                .collect();
                            let label = if result.is_error {
                                "Tool error"
                            } else {
                                "Tool result"
                            };
                            parts.push(format!("[{label}]: {}", snippet(&output.join("\n"))));
                        }
                        _ => parts.push("[User]: [attachment]".to_owned()),
                    }
                }
            }
            Message::Assistant(reply) => {
                for item in &reply.content {
                    match item {
                        AssistantContent::Text(text) => {
                            parts.push(format!("[Assistant]: {}", text.text));
                        }
                        AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                            parts.push(format!("[Assistant thinking]: {}", reasoning.text));
                        }
                        AssistantContent::ToolCall(call) => {
                            let arguments =
                                serde_json::Value::Object(call.function.arguments.clone());
                            parts.push(format!(
                                "[Assistant tool call]: {}({})",
                                call.function.name.as_str(),
                                snippet(&arguments.to_string())
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    parts.join("\n\n")
}

/// `text` cut to [`SNIPPET_CHARS`] characters, saying how much was cut.
fn snippet(text: &str) -> String {
    match text.char_indices().nth(SNIPPET_CHARS) {
        Some((end, _)) => format!(
            "{}\n[… {} more characters]",
            text.get(..end).unwrap_or_default(),
            text.len() - end
        ),
        None => text.to_owned(),
    }
}

/// Adds the paths the built-in file tools were called with in `messages`:
/// `read`'s to `read`, `edit`'s and `write`'s to `modified`.
fn touched_files(
    messages: &[Message],
    read: &mut BTreeSet<String>,
    modified: &mut BTreeSet<String>,
) {
    let calls = messages.iter().flat_map(|message| match message {
        Message::Assistant(reply) => reply.content.as_slice(),
        _ => &[],
    });
    for item in calls {
        let AssistantContent::ToolCall(call) = item else {
            continue;
        };
        let Some(path) = call
            .function
            .arguments
            .get("path")
            .and_then(|path| path.as_str())
        else {
            continue;
        };
        match call.function.name.as_str() {
            "read" => {
                read.insert(path.to_owned());
            }
            "edit" | "write" => {
                modified.insert(path.to_owned());
            }
            _ => {}
        }
    }
}

/// The summarizer's system prompt.
const SYSTEM_PROMPT: &str = "You summarize a conversation between a user and a coding agent \
    so that another model can continue the work from the summary alone. Read the \
    conversation and write the summary in the exact format asked for.\n\n\
    Do not continue the conversation. Do not answer questions in it. Output only the summary.";

/// The request for a first summary.
const INITIAL_PROMPT: &str = "The conversation above is to be summarized. Write a structured \
    checkpoint of it that another model will use to continue the work.";

/// The request to fold new messages into an earlier summary.
const UPDATE_PROMPT: &str = "The conversation above is the NEW part of a conversation whose \
    earlier part is summarized in <previous-summary>. Update that summary with it:\n\
    - keep everything in the previous summary that still holds;\n\
    - add the new progress, decisions and context;\n\
    - move items from \"In progress\" to \"Done\" when they were completed;\n\
    - update \"Next steps\" to what is left;\n\
    - drop what is no longer relevant.";

/// The summary's format, after either request.
const FORMAT: &str = "\n\nUse exactly this format:\n\n\
    ## Goal\n\
    [What the user is trying to get done; several items if the session covers several tasks.]\n\n\
    ## Constraints and preferences\n\
    - [What the user asked for or ruled out, or \"(none)\"]\n\n\
    ## Progress\n\
    ### Done\n\
    - [x] [Completed tasks and changes]\n\
    ### In progress\n\
    - [ ] [Current work]\n\
    ### Blocked\n\
    - [What prevents progress, if anything]\n\n\
    ## Key decisions\n\
    - **[Decision]**: [Why]\n\n\
    ## Next steps\n\
    1. [What should happen next, in order]\n\n\
    ## Critical context\n\
    - [Data, examples, commands or references needed to continue, or \"(none)\"]\n\n\
    Keep each section short. Keep exact file paths, function names, commands and error \
    messages.";
