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

use std::borrow::{Borrow, Cow};
use std::collections::BTreeSet;
use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, Message, Reasoning, UnsupportedOption,
};
use rig_core::conversation::Conversation;
use rig_core::message::{StopReason, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};

use super::summarized_prefix;

/// When compaction runs and how much it keeps: `[compaction]` in settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompactionSettings {
    /// Whether a conversation is compacted on its own: near the window, and after a request
    /// refused as too long or a reply the window cut short. `/compact` works either way.
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

/// The length of `text` in UTF-16 code units, as JavaScript counts it.
fn chars(text: &str) -> u64 {
    text.encode_utf16().count() as u64
}

/// Characters of tool-result content, an image as [`ESTIMATED_IMAGE_CHARS`].
fn result_chars(content: &[ToolResultContent]) -> u64 {
    content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => chars(&text.text),
            ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            ToolResultContent::Json { value } => chars(&value.to_string()),
        })
        .fold(0, u64::saturating_add)
}

/// The estimated tokens of `message`, at four characters a token, rounded up: on the high side.
/// Each tool result counts as a message of its own, rounded apart, beside the rest of a user
/// message.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::System { content } => chars(content).div_ceil(4),
        Message::User { content } => {
            let mut results = 0u64;
            let mut rest = 0u64;
            for item in content {
                match item {
                    UserContent::ToolResult(result) => {
                        results = results.saturating_add(result_chars(&result.content).div_ceil(4));
                    }
                    UserContent::Text(text) => rest = rest.saturating_add(chars(&text.text)),
                    UserContent::Image(_) => rest = rest.saturating_add(ESTIMATED_IMAGE_CHARS),
                    _ => {}
                }
            }
            results.saturating_add(rest.div_ceil(4))
        }
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
            .fold(0, u64::saturating_add)
            .div_ceil(4),
    }
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

/// Whether `message` starts a turn: a user or system message that is no tool result.
fn is_turn_start(message: &Message) -> bool {
    !matches!(message, Message::Assistant(_)) && !is_tool_result(message)
}

/// For each of `messages`, whether a compaction may keep from it: no tool result, and nothing
/// between a call and its result, so a result is never kept without its call. Keeping from the
/// reply that made a call keeps its results too.
fn cut_points(messages: &[Arc<Message>]) -> Vec<bool> {
    let mut allowed: Vec<bool> = messages
        .iter()
        .map(|message| !is_tool_result(message))
        .collect();
    let mut calls: Vec<(usize, &rig_core::message::CallId)> = Vec::new();
    for (at, message) in messages.iter().enumerate() {
        match &**message {
            Message::Assistant(reply) => {
                calls.extend(reply.tool_calls().map(|call| (at, &call.id)))
            }
            Message::User { content } => {
                for result in content.iter().filter_map(UserContent::as_tool_result) {
                    let made = calls.iter().rev().find(|(_, id)| **id == result.call);
                    if let Some(&(made, _)) = made {
                        for inside in allowed.iter_mut().take(at.saturating_add(1)).skip(made + 1) {
                            *inside = false;
                        }
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    allowed
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
/// point that is no tool result and lies outside every tool exchange. Cut at a reply, its results
/// stay after it, and the turn it is part of is split. With nothing to cut at, it keeps from
/// `start`. After an overflow (`overflowed`), a last message over the budget alone is summarized
/// too, as the turn's earlier part, since sending it again would overflow again.
pub fn find_cut_point(
    messages: &[Arc<Message>],
    (start, end): (usize, usize),
    keep_recent_tokens: u64,
    overflowed: bool,
) -> CutPoint {
    let end = end.min(messages.len());
    let allowed = cut_points(messages);
    let cut_points: Vec<usize> = (start..end)
        .filter(|&at| allowed.get(at).copied().unwrap_or(false))
        .collect();
    let Some(&first) = cut_points.first() else {
        return CutPoint {
            first_kept: start,
            turn_start: None,
        };
    };
    let mut cut = first;
    let mut exceeded = false;
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
            exceeded = true;
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
    if overflowed && exceeded && cut.saturating_add(1) == end {
        cut = end;
    }
    let starts_turn = messages
        .get(cut)
        .is_some_and(|message| is_turn_start(message));
    let turn_start = if starts_turn {
        None
    } else {
        (start..cut.min(end.saturating_sub(1)).saturating_add(1))
            .rev()
            .find(|&at| {
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
    /// The newest compaction's summary and file lists, as the update prompt sends it verbatim;
    /// `None` before the first compaction.
    pub previous: Option<String>,
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
/// when nothing would be; `overflowed` after the provider refused a request as too long
/// ([`find_cut_point`]). The earlier summaries ([`summarized_prefix`]) are never cut again: the
/// new summary updates the newest one.
pub fn prepare_compaction(
    conversation: &Conversation,
    keep_recent_tokens: u64,
    overflowed: bool,
) -> Option<CompactionPreparation> {
    let messages = conversation.messages();
    let range = (summarized_prefix(conversation), messages.len());
    let cut = find_cut_point(messages, range, keep_recent_tokens, overflowed);
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
    let state = super::summary_state(conversation);
    let mut ops = FileOperations::default();
    ops.read.extend(state.read_files.iter().cloned());
    ops.edited.extend(state.modified_files.iter().cloned());
    for message in to_summarize.iter().chain(&turn_prefix) {
        ops.extract(message);
    }
    Some(CompactionPreparation {
        first_kept: cut.first_kept,
        to_summarize,
        turn_prefix,
        previous: state.previous_summary(),
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
                .filter(|path| !path.is_empty())
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

/// `text` cut to `max_chars` UTF-16 code units, saying how many more there were. A character
/// that would straddle the cut is left out whole.
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let total = text.encode_utf16().count();
    if total <= max_chars {
        return text.to_owned();
    }
    let mut units = 0;
    let mut end = 0;
    for (at, character) in text.char_indices() {
        units += character.len_utf16();
        if units > max_chars {
            break;
        }
        end = at + character.len_utf8();
    }
    format!(
        "{}\n\n[... {} more characters truncated]",
        text.get(..end).unwrap_or_default(),
        total - max_chars
    )
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
                for result in content.iter().filter_map(UserContent::as_tool_result) {
                    let text: String = result.content.iter().map(|part| part.to_text("")).collect();
                    if !text.is_empty() {
                        parts.push(format!(
                            "[Tool result]: {}",
                            truncate_for_summary(&text, TOOL_RESULT_MAX_CHARS)
                        ));
                    }
                }
                let text: String = content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect();
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
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

/// Tokens a request leaves free below the window besides its output, for an input counted higher
/// than estimated.
const CONTEXT_SAFETY_TOKENS: u64 = 4_096;

/// The share of the input also left free while that leaves [`MIN_OUTPUT_WITH_RATIO_MARGIN`].
const CONTEXT_SAFETY_PERCENT: u64 = 15;

/// The least output the ratio margin is kept for; below it only the fixed margin is.
const MIN_OUTPUT_WITH_RATIO_MARGIN: u64 = 32_768;

/// `max_tokens` fitted into `spec`'s context window after an input of `input_tokens`, leaving a
/// margin for an input counted higher than estimated: unchanged when the window is not known, and
/// never below 1.
pub fn clamp_max_tokens(spec: &ModelSpec, input_tokens: u64, max_tokens: u64) -> u64 {
    let Some(window) = spec
        .context_window
        .map(u64::from)
        .filter(|&window| window > 0)
    else {
        return max_tokens.max(1);
    };
    let ratio = input_tokens
        .saturating_mul(CONTEXT_SAFETY_PERCENT)
        .div_ceil(100);
    let margin = CONTEXT_SAFETY_TOKENS.max(ratio);
    let mut room = window.saturating_sub(input_tokens).saturating_sub(margin);
    if room < MIN_OUTPUT_WITH_RATIO_MARGIN {
        let fixed = window
            .saturating_sub(input_tokens)
            .saturating_sub(CONTEXT_SAFETY_TOKENS);
        room = MIN_OUTPUT_WITH_RATIO_MARGIN.min(fixed);
    }
    max_tokens.min(room.max(1))
}

/// A one-off summary request to `spec`: `system`, then `prompt` as the user's message, with the
/// `reasoning` effort when the model takes it, no prompt-cache writes, and an output limit of
/// `max_output` (the model's output limit when `None`) fitted to the room the input leaves.
///
/// # Errors
///
/// When `spec` does not take its default options.
pub fn summarization_request(
    (system, prompt): (&str, String),
    spec: &ModelSpec,
    reasoning: Option<Reasoning>,
    max_output: Option<u64>,
) -> Result<CompletionRequest, UnsupportedOption> {
    let mut options = spec.default_options(reasoning);
    options.cache = None;
    if reasoning.is_some() && spec.validate(&options).is_err() {
        options.reasoning = None;
    }
    spec.validate(&options)?;
    let input = estimate_tokens(&[Message::system(system), Message::user(prompt.as_str())]);
    let request = CompletionRequest::new(Message::user(prompt))
        .preamble(system.to_owned())
        .options(options);
    Ok(match max_output.or(spec.output_limit().map(u64::from)) {
        Some(most) => request.max_tokens(clamp_max_tokens(spec, input, most)),
        None => request,
    })
}

/// The end of `text`, cut to fit `spec`'s input limit less `reserved` tokens at three characters a
/// token, for a request of another plugin that reads a whole conversation; the cut start says how
/// many characters were left out. Unchanged when the limit is not known or it fits.
pub fn keep_end_to_fit<'t>(text: &'t str, spec: &ModelSpec, reserved: u64) -> Cow<'t, str> {
    let Some(limit) = spec.input_limit().map(u64::from) else {
        return Cow::Borrowed(text);
    };
    let budget =
        usize::try_from(limit.saturating_sub(reserved).saturating_mul(3)).unwrap_or(usize::MAX);
    let left_out = text.chars().count().saturating_sub(budget);
    if left_out == 0 {
        return Cow::Borrowed(text);
    }
    let start = text
        .char_indices()
        .nth(left_out)
        .map_or(text.len(), |(start, _)| start);
    let rest = text.get(start..).unwrap_or_default();
    Cow::Owned(format!(
        "[... {left_out} earlier characters left out]\n{rest}"
    ))
}

/// Tokens of the input limit a summary request keeps for its answer when it cuts what it reads.
const SUMMARY_OUTPUT_ROOM: u64 = 16_384;

/// `messages` as text ([`serialize_conversation`]), its oldest part left out, as the cut start
/// says, when the request around it (`around`) and room for the answer would not fit `spec`'s
/// input limit otherwise: so a summary still fits a smaller model the agent switched to. Beyond
/// pi, which sends it whole.
fn fitted_conversation(
    messages: &[impl Borrow<Message>],
    spec: &ModelSpec,
    around: &str,
) -> String {
    let text = serialize_conversation(messages);
    let reserved = estimate_tokens(&[
        Message::system(SUMMARIZATION_SYSTEM_PROMPT),
        Message::user(around),
    ])
    .saturating_add(SUMMARY_OUTPUT_ROOM)
    .saturating_add(CONTEXT_SAFETY_TOKENS);
    keep_end_to_fit(&text, spec, reserved).into_owned()
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

/// The request for the history summary of `messages` to `spec` with the `reasoning` effort,
/// updating `previous` (the newest summary, with its file lists) when there is one, focused on
/// `instructions`.
///
/// # Errors
///
/// As [`summarization_request`].
pub fn summary_request(
    messages: &[impl Borrow<Message>],
    (previous, instructions): (Option<&str>, &str),
    spec: &ModelSpec,
    reasoning: Option<Reasoning>,
) -> Result<CompletionRequest, UnsupportedOption> {
    let base = match previous {
        Some(_) => UPDATE_SUMMARIZATION_PROMPT,
        None => SUMMARIZATION_PROMPT,
    };
    let around = format!("{}{base}{instructions}", previous.unwrap_or_default());
    let mut prompt = format!(
        "<conversation>\n{}\n</conversation>\n\n",
        fitted_conversation(messages, spec, &around)
    );
    if let Some(previous) = previous {
        prompt.push_str(&format!(
            "<previous-summary>\n{previous}\n</previous-summary>\n\n"
        ));
    }
    prompt.push_str(&with_focus(base, instructions));
    summarization_request((SUMMARIZATION_SYSTEM_PROMPT, prompt), spec, reasoning, None)
}

/// The request for the summary of a split turn's earlier part, `messages`, to `spec` with the
/// `reasoning` effort, focused on `instructions`.
///
/// # Errors
///
/// As [`summarization_request`].
pub fn turn_prefix_request(
    messages: &[impl Borrow<Message>],
    instructions: &str,
    spec: &ModelSpec,
    reasoning: Option<Reasoning>,
) -> Result<CompletionRequest, UnsupportedOption> {
    let around = format!("{TURN_PREFIX_SUMMARIZATION_PROMPT}{instructions}");
    let prompt = format!(
        "# Conversation\n{}\n\n# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}",
        fitted_conversation(messages, spec, &around)
    );
    let prompt = with_focus(&prompt, instructions);
    summarization_request((SUMMARIZATION_SYSTEM_PROMPT, prompt), spec, reasoning, None)
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

/// The summary in `response`, a reply to the request `label` names, its text parts joined by
/// newlines, or why it cannot become a session's checkpoint: an errored reply, one cut at the
/// output limit, or one that calls a tool.
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
            let why = if why.is_empty() {
                "Unknown error".to_owned()
            } else {
                why
            };
            Err(SummaryFailure::Failed(label, why))
        }
        StopReason::Length => Err(SummaryFailure::Truncated(label)),
        _ if response.tool_calls().next().is_some() => Err(SummaryFailure::CalledTool(label)),
        _ => {
            let texts: Vec<&str> = response
                .choice
                .iter()
                .filter_map(AssistantContent::as_text)
                .collect();
            Ok(texts.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests;
