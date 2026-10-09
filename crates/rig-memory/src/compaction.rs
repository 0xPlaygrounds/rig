//! Building blocks for compacting a long conversation: a [`SummaryState`]
//! that says which leading messages a summary replaces, a [`Summarizer`]
//! that asks a model for that summary, a [`ModelCompactor`] that runs it as
//! a [`Compactor`], and [`ClearToolOutputs`], a [`MemoryPolicy`] that frees
//! context without a model call.
//!
//! Nothing here knows a tool or a domain: the summarizer's prompts, the
//! tool arguments a summary keeps track of and the cleared-output text are
//! the caller's.
//!
//! ```
//! use rig_memory::{ClearToolOutputs, HeuristicTokenCounter, SummaryState};
//!
//! let state = SummaryState::default();
//! let counter = HeuristicTokenCounter::default();
//! assert_eq!(state.estimate(&[], &counter), 0);
//! let _policy = ClearToolOutputs::new(40_000);
//! ```

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};

use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, Message,
    UnsupportedOption,
};
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::id::ConversationId;
use rig_core::message::{ToolResultContent, UserContent};
use rig_core::serve::{Dispatch, ErasedHandler, Reply};
use rig_core::wasm_compat::WasmBoxedFuture;
use serde::{Deserialize, Serialize};

use crate::{
    Compactor, HeuristicTokenCounter, MemoryError, MemoryPolicy, TextSummary, TokenCounter,
};

/// Which of a conversation's first messages a summary replaces, the
/// summary, and the values tracked across compactions (such as the files a
/// coding agent read). The default replaces none.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryState {
    /// How many of the conversation's first messages the summary replaces.
    pub upto: usize,
    /// The summary; empty before the first compaction.
    pub summary: String,
    /// Named sets of values, in the order they were first tracked.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tracked: Vec<TrackedSet>,
}

/// One named set of a [`SummaryState`], sent after the summary inside a
/// `<name>` tag.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackedSet {
    /// The set's name, also its tag.
    pub name: String,
    /// Its values.
    pub values: BTreeSet<String>,
}

/// A rule of [`SummaryState::track`]: the string argument `argument` of
/// every call of the tool `tool` goes into the set `set`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackArgument<'a> {
    /// The tool's name.
    pub tool: &'a str,
    /// The argument's name.
    pub argument: &'a str,
    /// The set's name.
    pub set: &'a str,
}

impl SummaryState {
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
        let Some(summary) = self.message() else {
            return live;
        };
        let summary = UserContent::text(summary);
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

    /// The tokens of [`Self::request`]'s messages, by `counter`.
    pub fn estimate(&self, messages: &[Message], counter: &impl TokenCounter) -> usize {
        let summary = self.message().map_or(0, |summary| {
            counter.count(&Message::User {
                content: vec![UserContent::text(summary)],
            })
        });
        summary + counter.count_all(self.live(messages))
    }

    /// The text that stands for the summarized messages, `None` before the
    /// first compaction. A value appears only in the last set that holds
    /// it, so a later set (files changed) overrides an earlier one (files
    /// read).
    pub fn message(&self) -> Option<String> {
        if self.summary.is_empty() {
            return None;
        }
        let mut text = format!(
            "The conversation before this point was compacted into this summary:\n\n\
             <summary>\n{}\n</summary>",
            self.summary.trim()
        );
        for (index, set) in self.tracked.iter().enumerate() {
            let later = self.tracked.get(index + 1..).unwrap_or_default();
            let values: Vec<&str> = set
                .values
                .iter()
                .filter(|value| !later.iter().any(|later| later.values.contains(*value)))
                .map(String::as_str)
                .collect();
            if !values.is_empty() {
                let _ = write!(
                    text,
                    "\n\n<{name}>\n{}\n</{name}>",
                    values.join("\n"),
                    name = set.name
                );
            }
        }
        Some(text)
    }

    /// Where a new compaction should end: the first message it keeps.
    /// Walking back from the newest message, it keeps at least
    /// `keep_tokens` and cuts before a reply or a user's own message, never
    /// between a tool call and its result. With `force`, a shorter
    /// conversation still has its older messages summarized, all but the
    /// newest reply or message. `None` when nothing new would be
    /// summarized.
    pub fn cut(
        &self,
        messages: &[Message],
        keep_tokens: usize,
        force: bool,
        counter: &impl TokenCounter,
    ) -> Option<usize> {
        let mut kept = 0;
        let mut newest = None;
        for (index, message) in messages.iter().enumerate().skip(self.upto + 1).rev() {
            kept += counter.count(message);
            if !can_start_live(message) {
                continue;
            }
            newest.get_or_insert(index);
            if kept >= keep_tokens {
                return Some(index);
            }
        }
        newest.filter(|_| force)
    }

    /// Adds the arguments `rules` name of the tool calls in `messages` to
    /// the tracked sets, creating the rules' sets in rule order first.
    pub fn track(&mut self, messages: &[Message], rules: &[TrackArgument<'_>]) {
        for rule in rules {
            if !self.tracked.iter().any(|set| set.name == rule.set) {
                self.tracked.push(TrackedSet {
                    name: rule.set.to_owned(),
                    values: BTreeSet::new(),
                });
            }
        }
        let calls = messages.iter().flat_map(|message| match message {
            Message::Assistant(reply) => reply.content.as_slice(),
            _ => &[],
        });
        for item in calls {
            let AssistantContent::ToolCall(call) = item else {
                continue;
            };
            for rule in rules
                .iter()
                .filter(|rule| rule.tool == call.function.name.as_str())
            {
                let value = call
                    .function
                    .arguments
                    .get(rule.argument)
                    .and_then(|value| value.as_str());
                let set = self.tracked.iter_mut().find(|set| set.name == rule.set);
                if let (Some(value), Some(set)) = (value, set) {
                    set.values.insert(value.to_owned());
                }
            }
        }
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

/// What a [`Summarizer`] asks the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryPrompts {
    /// The summarizer's system prompt.
    pub system: Cow<'static, str>,
    /// The request for a first summary, after the conversation.
    pub initial: Cow<'static, str>,
    /// The request to fold the conversation into `<previous-summary>`.
    pub update: Cow<'static, str>,
    /// The summary's format, after either request.
    pub format: Cow<'static, str>,
}

/// The sizes a [`Summarizer`] works within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SummaryLimits {
    /// Tokens left free below the model's window.
    pub reserve: u64,
    /// The longest summary asked for, in tokens.
    pub summary_tokens: u64,
    /// Characters of a tool's output or arguments the model sees.
    pub snippet_chars: usize,
    /// Characters of conversation sent when the model's window is unknown.
    pub unknown_window_chars: usize,
}

impl SummaryLimits {
    /// 16k tokens reserved, summaries up to 12k tokens, 2,000-character
    /// snippets and 400k characters for an unknown window (pi's values).
    pub const DEFAULT: Self = Self {
        reserve: 16_384,
        summary_tokens: 12_000,
        snippet_chars: 2_000,
        unknown_window_chars: 400_000,
    };
}

impl Default for SummaryLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Builds the request for a summary of a conversation's older messages and
/// reads the summary from the model's answer. The conversation goes as
/// text in one user message, so the model reads it rather than continues
/// it. The format follows pi's
/// (`references/pi/packages/coding-agent/src/core/compaction/compaction.ts:507-579`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summarizer {
    /// What the model is asked.
    pub prompts: SummaryPrompts,
    /// Sizes.
    pub limits: SummaryLimits,
}

/// Why a summary cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SummaryError {
    /// The summary hit the model's output limit, so it is incomplete.
    Truncated,
    /// The model called a tool instead of summarizing.
    CalledTool,
    /// The model returned no text.
    Empty,
}

impl fmt::Display for SummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Truncated => "the summary hit the model's output limit",
            Self::CalledTool => "the model called a tool instead of summarizing",
            Self::Empty => "the model returned no summary",
        })
    }
}

impl std::error::Error for SummaryError {}

impl Summarizer {
    /// The request to `spec` for a summary of `older`, merged into
    /// `previous` when that is not empty. `focus` is what the summary
    /// should keep above all; it may be empty.
    pub fn request(
        &self,
        older: &[Message],
        previous: &str,
        focus: &str,
        spec: &ModelSpec,
    ) -> Result<CompletionRequest, UnsupportedOption> {
        let limits = &self.limits;
        let budget = spec
            .context_window
            .map_or(limits.unknown_window_chars, |window| {
                let tokens =
                    u64::from(window).saturating_sub(limits.reserve + limits.summary_tokens);
                // Three characters a token, to stay under the window with code.
                usize::try_from(tokens.saturating_mul(3)).unwrap_or(usize::MAX)
            });
        let conversation = self.serialize(older);
        let conversation = keep_end(&conversation, budget.saturating_sub(previous.len()));
        let mut prompt = format!("<conversation>\n{conversation}\n</conversation>\n\n");
        if previous.is_empty() {
            prompt.push_str(&self.prompts.initial);
        } else {
            let _ = write!(
                prompt,
                "<previous-summary>\n{previous}\n</previous-summary>\n\n{}",
                self.prompts.update
            );
        }
        if !focus.is_empty() {
            let _ = write!(prompt, "\n\nAdditional focus: {focus}");
        }
        prompt.push_str(&self.prompts.format);
        let options = spec.default_options(None);
        spec.validate(&options)?;
        let max_tokens = spec
            .max_output_tokens
            .map_or(limits.summary_tokens, |most| {
                limits.summary_tokens.min(u64::from(most))
            });
        Ok(CompletionRequest::new(Message::user(prompt))
            .preamble(self.prompts.system.clone().into_owned())
            .options(options)
            .max_tokens(max_tokens))
    }

    /// The summary in a finished answer, or why it cannot be used.
    pub fn summary_text(response: &CompletionResponse) -> Result<String, SummaryError> {
        if matches!(response.finish_reason(), Some(FinishReason::Length)) {
            return Err(SummaryError::Truncated);
        }
        if response.tool_calls().next().is_some() {
            return Err(SummaryError::CalledTool);
        }
        let text = response.text();
        let text = text.trim();
        if text.is_empty() {
            return Err(SummaryError::Empty);
        }
        Ok(text.to_owned())
    }

    /// `messages` as text, one labelled block per part, with long tool
    /// outputs and arguments cut (pi's `serializeConversation`,
    /// `references/pi/packages/coding-agent/src/core/compaction/utils.ts:114-159`).
    fn serialize(&self, messages: &[Message]) -> String {
        let snippet = |text: &str| snippet(text, self.limits.snippet_chars);
        let mut parts: Vec<String> = Vec::new();
        for message in messages {
            match message {
                Message::System { content } => parts.push(format!("[System]: {content}")),
                Message::User { content } => {
                    for item in content {
                        match item {
                            UserContent::Text(text) => {
                                parts.push(format!("[User]: {}", text.text));
                            }
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
                            AssistantContent::Reasoning(reasoning)
                                if !reasoning.text.is_empty() =>
                            {
                                parts.push(format!("[Assistant thinking]: {}", reasoning.text));
                            }
                            AssistantContent::ToolCall(call) => {
                                let arguments = rig_core::serde_json::Value::Object(
                                    call.function.arguments.clone(),
                                );
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

/// `text` cut to `chars` characters, saying how much was cut.
fn snippet(text: &str, chars: usize) -> String {
    match text.char_indices().nth(chars) {
        Some((end, _)) => format!(
            "{}\n[… {} more characters]",
            text.get(..end).unwrap_or_default(),
            text.len() - end
        ),
        None => text.to_owned(),
    }
}

/// Waits for a completion's reply, streamed or not, as one response.
pub async fn completion_of(
    reply: impl Future<Output = Reply>,
) -> Result<CompletionResponse, ErrorReport> {
    match reply.await.into_outcome().await? {
        Outcome::Completion(response) => Ok(response),
        _ => Err(ErrorReport::new(
            ErrorKind::Internal,
            "the model answered a completion request with something other than a completion",
        )),
    }
}

/// A [`Compactor`] that asks a model for the summary through its handler,
/// unrecorded. A host that records its effects builds the request with
/// [`Summarizer::request`] and dispatches it itself.
#[derive(Clone, Debug)]
pub struct ModelCompactor {
    summarizer: Summarizer,
    handler: ErasedHandler,
    spec: ModelSpec,
}

impl ModelCompactor {
    /// Summarize with the model `spec`, served by `handler`.
    pub fn new(summarizer: Summarizer, handler: ErasedHandler, spec: ModelSpec) -> Self {
        Self {
            summarizer,
            handler,
            spec,
        }
    }
}

impl Compactor for ModelCompactor {
    type Artifact = TextSummary;

    fn compact<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            let previous = carry_over.map_or("", TextSummary::as_str);
            let request = self
                .summarizer
                .request(evicted, previous, "", &self.spec)
                .map_err(|refusal| MemoryError::Policy(refusal.to_string()))?;
            let kind = EffectKind::Completion {
                request,
                stream: false,
            };
            let reply = self
                .handler
                .handle(kind, Dispatch::new(EffectId::from_raw(0), false));
            let response = completion_of(reply)
                .await
                .map_err(|report| MemoryError::Policy(report.to_string()))?;
            Summarizer::summary_text(&response)
                .map(TextSummary)
                .map_err(|why| MemoryError::Policy(why.to_string()))
        })
    }
}

/// What [`ClearToolOutputs::clear`] took out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cleared {
    /// Tool results cleared.
    pub results: usize,
    /// Their estimated tokens.
    pub tokens: usize,
}

/// A [`MemoryPolicy`] that frees context by clearing the outputs of older
/// tool calls, newest kept first: walking back from the end, the outputs
/// within the first `keep_tokens` stay, and every older one is replaced by
/// a placeholder. The last message is never touched: it is what the model
/// must answer. Error results stay, being short and telling the model what
/// went wrong. The calls themselves stay, so the conversation keeps its
/// shape (opencode's `prune`,
/// `references/opencode/packages/opencode/src/session/compaction.ts:271-316`).
#[derive(Clone, Debug)]
pub struct ClearToolOutputs {
    keep_tokens: usize,
    placeholder: Cow<'static, str>,
    counter: HeuristicTokenCounter,
}

impl ClearToolOutputs {
    /// What a cleared output says by default.
    pub const PLACEHOLDER: &'static str =
        "[output cleared to fit the context window; run the tool again if needed]";

    /// Keep the newest `keep_tokens` of tool output, counted by the default
    /// [`HeuristicTokenCounter`], and say [`Self::PLACEHOLDER`] instead of
    /// the rest.
    pub fn new(keep_tokens: usize) -> Self {
        Self {
            keep_tokens,
            placeholder: Cow::Borrowed(Self::PLACEHOLDER),
            counter: HeuristicTokenCounter::default(),
        }
    }

    /// Say `placeholder` instead of a cleared output.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_placeholder(mut self, placeholder: impl Into<Cow<'static, str>>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Count tool outputs with `counter`.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_counter(mut self, counter: HeuristicTokenCounter) -> Self {
        self.counter = counter;
        self
    }

    /// Clears older tool outputs of `messages` in place.
    pub fn clear(&self, messages: &mut [Message]) -> Cleared {
        let mut cleared = Cleared::default();
        let mut kept = 0;
        let Some((_, earlier)) = messages.split_last_mut() else {
            return cleared;
        };
        for message in earlier.iter_mut().rev() {
            let Message::User { content } = message else {
                continue;
            };
            for item in content.iter_mut().rev() {
                let tokens = self.counter.count_user(item);
                let UserContent::ToolResult(result) = item else {
                    continue;
                };
                if result.is_error || self.is_cleared(&result.content) {
                    continue;
                }
                kept += tokens;
                if kept <= self.keep_tokens {
                    continue;
                }
                result.content = vec![ToolResultContent::text(self.placeholder.clone())];
                cleared.results += 1;
                cleared.tokens += tokens;
            }
        }
        cleared
    }

    fn is_cleared(&self, content: &[ToolResultContent]) -> bool {
        matches!(content, [only] if only.as_text() == Some(&*self.placeholder))
    }
}

impl MemoryPolicy for ClearToolOutputs {
    fn apply(&self, mut messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        self.clear(&mut messages);
        Ok(messages)
    }
}

#[cfg(test)]
mod tests;
