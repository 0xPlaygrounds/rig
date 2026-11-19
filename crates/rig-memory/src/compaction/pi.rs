// The design, prompts and logic of this module are ported from the pi coding agent's compaction
// (https://github.com/badlogic/pi-mono, packages/coding-agent/src/core/compaction), under its
// license:
//
// MIT License
//
// Copyright (c) 2025 Mario Zechner
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! When to compact, where to cut, and the summary requests: pure functions over a conversation's
//! messages. The caller sends the requests, reads the replies with [`summary_text`] and puts the
//! summary in with [`apply_summary`](crate::apply_summary).

use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, Message, UnsupportedOption,
};
use rig_core::conversation::Conversation;
use rig_core::message::{StopReason, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};

use super::{SummaryState, summarized_prefix};

/// When compaction runs and how much it keeps: `[compaction]` in settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompactionSettings {
    /// Whether a conversation is compacted on its own near the window. `/compact` works either
    /// way.
    pub enabled: bool,
    /// Tokens of the model's input limit kept free for the reply.
    pub reserve_tokens: u64,
    /// About how many tokens of the newest messages stay as they are.
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    /// Enabled, 16384 tokens reserved, 20000 kept.
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

impl CompactionSettings {
    /// Whether a context of `context_tokens` in a window of `context_window` tokens is compacted
    /// on its own: when enabled and it leaves less than the reserve free.
    pub fn should_compact(&self, context_tokens: u64, context_window: u64) -> bool {
        self.enabled && context_tokens > context_window.saturating_sub(self.reserve_tokens)
    }
}

/// Characters an image counts for.
const ESTIMATED_IMAGE_CHARS: u64 = 4_800;

/// Characters of user or tool-result content, an image as [`ESTIMATED_IMAGE_CHARS`].
fn user_chars(content: &[UserContent]) -> u64 {
    content
        .iter()
        .map(|item| match item {
            UserContent::Text(text) => chars(&text.text),
            UserContent::ToolResult(result) => result
                .content
                .iter()
                .map(|part| match part {
                    ToolResultContent::Text(text) => chars(&text.text),
                    ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
                    ToolResultContent::Json { value } => chars(&value.to_string()),
                })
                .fold(0, u64::saturating_add),
            UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            _ => 0,
        })
        .fold(0, u64::saturating_add)
}

fn chars(text: &str) -> u64 {
    text.chars().count() as u64
}

/// The estimated tokens of `message`, at four characters a token, rounded up: on the high side.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = match message {
        Message::System { content } => chars(content),
        Message::User { content } => user_chars(content),
        Message::Assistant(reply) => reply
            .content
            .iter()
            .map(|item| match item {
                AssistantContent::Text(text) => chars(&text.text),
                AssistantContent::Reasoning(reasoning) => chars(&reasoning.text),
                AssistantContent::ToolCall(call) => chars(call.function.name.as_str())
                    .saturating_add(chars(
                        &serde_json::to_string(&call.function.arguments).unwrap_or_default(),
                    )),
                _ => 0,
            })
            .fold(0, u64::saturating_add),
    };
    chars.div_ceil(4)
}

/// The estimated tokens of `messages` ([`estimate_message_tokens`]).
pub fn estimate_tokens(messages: &[impl Borrow<Message>]) -> u64 {
    messages
        .iter()
        .map(|message| estimate_message_tokens(message.borrow()))
        .fold(0, u64::saturating_add)
}

/// Whether `message` answers tool calls: a user message holding a tool result.
fn is_tool_result(message: &Message) -> bool {
    matches!(message, Message::User { content } if content.iter().any(|item| matches!(item, UserContent::ToolResult(_))))
}

/// Whether a compaction may keep from `message`: anything but a tool result, which must follow
/// its call.
fn is_cut_point(message: &Message) -> bool {
    !is_tool_result(message)
}

/// Whether `message` starts a turn: a user or system message that is no tool result.
fn is_turn_start(message: &Message) -> bool {
    !matches!(message, Message::Assistant(_)) && !is_tool_result(message)
}

/// Where a compaction cuts ([`find_cut_point`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPoint {
    /// The index of the first message kept.
    pub first_kept: usize,
    /// When the cut splits a turn (it keeps from a reply), the index of the message that started
    /// that turn.
    pub turn_start: Option<usize>,
}

/// The cut of `messages[start..end]` that keeps about `keep_recent_tokens`: walking back from the
/// newest message until that many are counted, it cuts at the nearest message at or after that
/// point that is no tool result. Cut at a reply, its results stay after it, and the turn it is
/// part of is split. With nothing to cut at, it keeps from `start`.
pub fn find_cut_point(
    messages: &[Arc<Message>],
    start: usize,
    end: usize,
    keep_recent_tokens: u64,
) -> CutPoint {
    let end = end.min(messages.len());
    let cut_points: Vec<usize> = (start..end)
        .filter(|&at| {
            messages
                .get(at)
                .is_some_and(|message| is_cut_point(message))
        })
        .collect();
    let Some(&first) = cut_points.first() else {
        return CutPoint {
            first_kept: start,
            turn_start: None,
        };
    };
    let mut cut = first;
    let mut accumulated = 0u64;
    for at in (start..end).rev() {
        let tokens = messages
            .get(at)
            .map_or(0, |message| estimate_message_tokens(message));
        if tokens == 0 {
            continue;
        }
        accumulated = accumulated.saturating_add(tokens);
        if accumulated >= keep_recent_tokens {
            // Trailing tool results over the budget keep the reply that called them.
            cut = cut_points
                .iter()
                .copied()
                .find(|&candidate| candidate >= at)
                .or(cut_points.last().copied())
                .unwrap_or(first);
            break;
        }
    }
    let starts_turn = messages
        .get(cut)
        .is_some_and(|message| is_turn_start(message));
    let turn_start = if starts_turn {
        None
    } else {
        (start..=cut).rev().find(|&at| {
            messages
                .get(at)
                .is_some_and(|message| is_turn_start(message))
        })
    };
    CutPoint {
        first_kept: cut,
        turn_start,
    }
}

/// A compaction ready to summarize ([`prepare_compaction`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionPreparation {
    /// The index of the first message kept.
    pub first_kept: usize,
    /// The messages summarized into the history summary, after the earlier summaries.
    pub to_summarize: Vec<Arc<Message>>,
    /// When the cut splits a turn, that turn's messages before the cut, summarized apart.
    pub turn_prefix: Vec<Arc<Message>>,
    /// The newest compaction's summary state, which the summary updates.
    pub previous: SummaryState,
    /// The files read and not changed, and the files changed, by every summarized message and
    /// earlier compaction.
    pub files: FileLists,
}

/// The files a compaction lists after its summary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileLists {
    /// Files only read, sorted.
    pub read_files: Vec<String>,
    /// Files edited or written, sorted.
    pub modified_files: Vec<String>,
}

/// What a compaction of `conversation`, keeping about `keep_recent_tokens`, summarizes, or `None`
/// when nothing would be. The earlier summaries ([`summarized_prefix`]) are never cut again: the
/// new summary updates the newest one.
pub fn prepare_compaction(
    conversation: &Conversation,
    keep_recent_tokens: u64,
) -> Option<CompactionPreparation> {
    let messages = conversation.messages();
    let start = summarized_prefix(conversation);
    let cut = find_cut_point(messages, start, messages.len(), keep_recent_tokens);
    prepare_compaction_at(conversation, cut)
}

/// What a compaction of `conversation` cut at `cut` summarizes, or `None` when nothing would be:
/// the messages after the earlier summaries and before the cut (or before the split turn's start),
/// and a split turn's earlier part.
pub fn prepare_compaction_at(
    conversation: &Conversation,
    cut: CutPoint,
) -> Option<CompactionPreparation> {
    let messages = conversation.messages();
    let start = summarized_prefix(conversation);
    let history_end = cut.turn_start.unwrap_or(cut.first_kept);
    let to_summarize = messages
        .get(start..history_end)
        .unwrap_or_default()
        .to_vec();
    let turn_prefix = match cut.turn_start {
        Some(turn_start) => messages
            .get(turn_start..cut.first_kept)
            .unwrap_or_default()
            .to_vec(),
        None => Vec::new(),
    };
    if to_summarize.is_empty() && turn_prefix.is_empty() {
        return None;
    }
    let previous = super::summary_state(conversation);
    let mut ops = FileOperations::default();
    ops.read.extend(previous.read_files.iter().cloned());
    ops.edited.extend(previous.modified_files.iter().cloned());
    for message in to_summarize.iter().chain(&turn_prefix) {
        ops.extract(message);
    }
    Some(CompactionPreparation {
        first_kept: cut.first_kept,
        to_summarize,
        turn_prefix,
        previous,
        files: ops.lists(),
    })
}

/// The files the built-in file tools were called with.
#[derive(Default)]
struct FileOperations {
    read: BTreeSet<String>,
    written: BTreeSet<String>,
    edited: BTreeSet<String>,
}

impl FileOperations {
    /// Adds the `path` of each `read`, `write` and `edit` call of `message`.
    fn extract(&mut self, message: &Message) {
        let Message::Assistant(reply) = message else {
            return;
        };
        for call in reply.tool_calls() {
            let Some(path) = call
                .function
                .arguments
                .get("path")
                .and_then(|path| path.as_str())
            else {
                continue;
            };
            let set = match call.function.name.as_str() {
                "read" => &mut self.read,
                "write" => &mut self.written,
                "edit" => &mut self.edited,
                _ => continue,
            };
            set.insert(path.to_owned());
        }
    }

    /// The files only read, and the files changed.
    fn lists(&self) -> FileLists {
        let modified: BTreeSet<&String> = self.edited.iter().chain(&self.written).collect();
        FileLists {
            read_files: self
                .read
                .iter()
                .filter(|path| !modified.contains(path))
                .cloned()
                .collect(),
            modified_files: modified.into_iter().cloned().collect(),
        }
    }
}

/// The file lists as tags after a summary, or nothing when both are empty.
pub(crate) fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

/// The most characters of a tool result a summary request shows.
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// `text` cut to `max_chars` characters, saying how many more there were.
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((end, _)) => format!(
            "{}\n\n[... {} more characters truncated]",
            text.get(..end).unwrap_or_default(),
            text.get(end..).unwrap_or_default().chars().count()
        ),
        None => text.to_owned(),
    }
}

/// `messages` as labeled text, so the model reads them rather than continues them. Thinking is
/// left out, and tool results are cut to 2000 characters.
pub fn serialize_conversation(messages: &[impl Borrow<Message>]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message.borrow() {
            Message::System { content } if !content.is_empty() => {
                parts.push(format!("[User]: {content}"));
            }
            Message::System { .. } => {}
            Message::User { content } => {
                let text: Vec<&str> = content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect();
                let text = text.join("\n");
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
                }
                for result in content.iter().filter_map(UserContent::as_tool_result) {
                    let text: Vec<String> = result
                        .content
                        .iter()
                        .map(|part| part.to_text(""))
                        .filter(|text| !text.is_empty())
                        .collect();
                    let text = text.join("\n");
                    if !text.is_empty() {
                        parts.push(format!(
                            "[Tool result]: {}",
                            truncate_for_summary(&text, TOOL_RESULT_MAX_CHARS)
                        ));
                    }
                }
            }
            Message::Assistant(reply) => {
                let calls: Vec<String> = reply
                    .tool_calls()
                    .map(|call| {
                        let arguments: Vec<String> = call
                            .function
                            .arguments
                            .iter()
                            .map(|(key, value)| format!("{key}={value}"))
                            .collect();
                        format!("{}({})", call.function.name.as_str(), arguments.join(", "))
                    })
                    .collect();
                let texts: Vec<&str> = reply
                    .content
                    .iter()
                    .filter_map(AssistantContent::as_text)
                    .collect();
                if !texts.is_empty() {
                    parts.push(format!("[Assistant]: {}", texts.join("\n")));
                }
                if !calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
                }
            }
        }
    }
    parts.join("\n\n")
}

/// The system prompt of every summary request.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or \"(none)\" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or \"(none)\" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed
- UPDATE \"Next Steps\" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or recreate later messages.";

/// A summary request to `spec` asking `prompt` alone, under [`SUMMARIZATION_SYSTEM_PROMPT`], with
/// the model's output limit when known.
///
/// # Errors
///
/// When `spec` does not take its default options.
pub fn summarization_request(
    prompt: String,
    spec: &ModelSpec,
) -> Result<CompletionRequest, UnsupportedOption> {
    let options = spec.default_options(None);
    spec.validate(&options)?;
    let request = CompletionRequest::new(Message::user(prompt))
        .preamble(SUMMARIZATION_SYSTEM_PROMPT.to_owned())
        .options(options);
    Ok(match spec.output_limit() {
        Some(most) => request.max_tokens(u64::from(most)),
        None => request,
    })
}

/// `prompt` followed by the user's `instructions`, when there are any.
fn with_focus(prompt: &str, instructions: &str) -> String {
    let instructions = instructions.trim();
    if instructions.is_empty() {
        prompt.to_owned()
    } else {
        format!("{prompt}\n\nAdditional focus: {instructions}")
    }
}

/// The request for the history summary of `messages` to `spec`, updating `previous` (the newest
/// summary, with its file lists) when there is one, focused on `instructions`.
///
/// # Errors
///
/// As [`summarization_request`].
pub fn summary_request(
    messages: &[impl Borrow<Message>],
    previous: Option<&str>,
    instructions: &str,
    spec: &ModelSpec,
) -> Result<CompletionRequest, UnsupportedOption> {
    let base = match previous {
        Some(_) => UPDATE_SUMMARIZATION_PROMPT,
        None => SUMMARIZATION_PROMPT,
    };
    let mut prompt = format!(
        "<conversation>\n{}\n</conversation>\n\n",
        serialize_conversation(messages)
    );
    if let Some(previous) = previous {
        prompt.push_str(&format!(
            "<previous-summary>\n{previous}\n</previous-summary>\n\n"
        ));
    }
    prompt.push_str(&with_focus(base, instructions));
    summarization_request(prompt, spec)
}

/// The request for the summary of a split turn's earlier part, `messages`, to `spec`, focused on
/// `instructions`.
///
/// # Errors
///
/// As [`summarization_request`].
pub fn turn_prefix_request(
    messages: &[impl Borrow<Message>],
    instructions: &str,
    spec: &ModelSpec,
) -> Result<CompletionRequest, UnsupportedOption> {
    let prompt = format!(
        "# Conversation\n{}\n\n# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}",
        serialize_conversation(messages)
    );
    summarization_request(with_focus(&prompt, instructions), spec)
}

/// The history summary when a split turn's earlier part has no messages of its own to summarize:
/// the previous summary, or a note that there is none.
pub fn no_history(previous: Option<&str>) -> String {
    previous.unwrap_or("No prior history.").to_owned()
}

/// One summary of a split turn: the `history` summary, then its earlier part's `prefix`.
pub fn merge_split_turn(history: &str, prefix: &str) -> String {
    format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}")
}

/// Why a summary reply cannot be used.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SummaryFailure {
    /// The reply failed, for this reason.
    #[error("{0} failed: {1}")]
    Failed(&'static str, String),
    /// The reply hit the output limit, so the summary is incomplete.
    #[error("{0} failed: generation hit the token cap and the summary is incomplete")]
    Truncated(&'static str),
    /// The model called a tool.
    #[error("{0} attempted to call a tool")]
    CalledTool(&'static str),
}

/// The summary in `response`, a reply to the request `label` names, or why it cannot become a
/// session's checkpoint: an errored reply, one cut at the output limit, or one that calls a tool.
///
/// # Errors
///
/// [`SummaryFailure`].
pub fn summary_text(
    response: &CompletionResponse,
    label: &'static str,
) -> Result<String, SummaryFailure> {
    match response.stop() {
        StopReason::Error(why) | StopReason::Aborted(why) => {
            Err(SummaryFailure::Failed(label, why))
        }
        StopReason::Length => Err(SummaryFailure::Truncated(label)),
        _ if response.tool_calls().next().is_some() => Err(SummaryFailure::CalledTool(label)),
        _ => Ok(response.text()),
    }
}

#[cfg(test)]
mod tests;
