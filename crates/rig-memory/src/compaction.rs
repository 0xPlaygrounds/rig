//! Building blocks for compacting a long conversation: a
//! [`CompactionPolicy`] that says when and where to cut it and plans the
//! summary of its older messages, a [`SummaryState`] with the summary and
//! the values tracked across compactions, a [`Summarizer`] that asks a
//! model for the summary, a [`ModelCompactor`] that runs it as a
//! [`Compactor`], and [`ClearToolOutputs`], a [`MemoryPolicy`] that frees
//! context without a model call.
//!
//! The summarizer's default prompts ask for pi's structured checkpoint;
//! the tool arguments a summary keeps track of are the caller's.
//!
//! ```
//! use rig_memory::{CompactionPolicy, SummaryState};
//!
//! let policy = CompactionPolicy::default();
//! assert_eq!(policy.cut(&[], 0, &rig_memory::CompactReason::Threshold, None), None);
//! assert_eq!(SummaryState::default().message(), None);
//! ```

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};

use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, Message,
    UnsupportedOption,
};
use rig_core::effect::family::Completion;
use rig_core::effect::{EffectId, EffectKind, Family};
use rig_core::id::ConversationId;
use rig_core::message::{ToolResultContent, UserContent};
use rig_core::serve::{Dispatch, ErasedHandler};
use rig_core::wasm_compat::WasmBoxedFuture;
use serde::{Deserialize, Serialize};

use crate::{
    Compactor, HeuristicTokenCounter, MemoryError, MemoryPolicy, TextSummary, TokenCounter,
};

/// A summary of a conversation's older messages and the values tracked
/// across compactions (such as the files a coding agent read). The default
/// is the state before the first compaction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryState {
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

/// Where a new compaction of `messages`, whose first `from` are summarized
/// already, should end: the first message it keeps. Walking back from the
/// newest message, it keeps at least `keep_tokens` and cuts before a reply
/// or a user's own message, never between a tool call and its result.
/// With `force`, a shorter conversation still has its older messages
/// summarized, all but the newest reply or message. `None` when nothing
/// new would be summarized.
fn cut_at(
    messages: &[Message],
    from: usize,
    keep_tokens: usize,
    force: bool,
    counter: &impl TokenCounter,
) -> Option<usize> {
    let mut kept = 0;
    let mut newest = None;
    for (index, message) in messages.iter().enumerate().skip(from + 1).rev() {
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

/// Whether the messages a request sends as they are may start with
/// `message`: a reply, or a user message that answers no tool call.
fn can_start_live(message: &Message) -> bool {
    match message {
        Message::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        Message::Assistant(_) | Message::System { .. } => true,
    }
}

/// Why a conversation is compacted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactReason {
    /// The user asked, with what to focus on, if anything.
    Asked {
        /// What the summary should keep above all; may be empty.
        focus: String,
    },
    /// The next request would leave less than the reserve of the model's
    /// window free.
    Threshold,
    /// The model refused the request as too long.
    Overflow,
}

/// How a conversation is compacted: what the summarizer is asked, the tool
/// arguments a summary keeps track of, such as the files a coding agent's
/// file tools read and changed, how much of the newest conversation stays
/// as it is, and how old tool outputs are cleared first. Tokens are
/// estimated by the default [`HeuristicTokenCounter`].
#[derive(Clone, Debug)]
pub struct CompactionPolicy {
    /// The summarizer; its reserve is how much of the model's window is
    /// kept free.
    pub summarizer: Summarizer,
    /// The tool arguments tracked across compactions.
    pub tracked: Vec<TrackArgument<'static>>,
    /// Tokens of the newest messages an automatic compaction keeps as they
    /// are, a quarter of the model's window at most (pi's
    /// `keepRecentTokens`): the work carries on, so its recent part stays.
    pub keep_recent: usize,
    /// Tokens of the newest messages a compaction the user asked for keeps
    /// as they are; the newest reply or message always stays.
    pub keep_asked: usize,
    /// How old tool outputs are cleared before a summary is asked for.
    pub clearing: ClearToolOutputs,
}

impl Default for CompactionPolicy {
    /// [`Summarizer::DEFAULT`], nothing tracked, the newest 20k tokens kept
    /// by an automatic compaction and none beyond the newest reply or
    /// message by an asked one, and all but the newest 40k tokens of tool
    /// output cleared (opencode's `PRUNE_PROTECT`).
    fn default() -> Self {
        Self {
            summarizer: Summarizer::DEFAULT,
            tracked: Vec::new(),
            keep_recent: 20_000,
            keep_asked: 0,
            clearing: ClearToolOutputs::new(40_000),
        }
    }
}

impl CompactionPolicy {
    /// The estimated tokens of `messages`.
    pub fn estimate(&self, messages: &[Message]) -> u64 {
        HeuristicTokenCounter::default().count_all(messages) as u64
    }

    /// Whether a request of `tokens` leaves less than the summarizer's
    /// reserve of `spec`'s window free. Never, when the window is not known.
    pub fn over_threshold(&self, tokens: u64, spec: &ModelSpec) -> bool {
        spec.context_window.is_some_and(|window| {
            tokens > u64::from(window).saturating_sub(self.summarizer.limits.reserve)
        })
    }

    /// Tokens of the newest messages a compaction for `reason` with `spec`
    /// keeps as they are.
    pub fn keep(&self, reason: &CompactReason, spec: Option<&ModelSpec>) -> usize {
        match reason {
            CompactReason::Asked { .. } => self.keep_asked,
            CompactReason::Threshold | CompactReason::Overflow => spec
                .and_then(|spec| spec.context_window)
                .map_or(self.keep_recent, |window| {
                    self.keep_recent
                        .min(usize::try_from(window / 4).unwrap_or(usize::MAX))
                }),
        }
    }

    /// Where a new compaction for `reason` of `messages`, whose first
    /// `from` are summarized already, should end with `spec`: the first
    /// message it keeps, or `None` when nothing new would be summarized. A
    /// compaction that must happen (the user asked, or the model refused
    /// the request as too long) still summarizes a short conversation's
    /// older messages.
    pub fn cut(
        &self,
        messages: &[Message],
        from: usize,
        reason: &CompactReason,
        spec: Option<&ModelSpec>,
    ) -> Option<usize> {
        let force = !matches!(reason, CompactReason::Threshold);
        cut_at(
            messages,
            from,
            self.keep(reason, spec),
            force,
            &HeuristicTokenCounter::default(),
        )
    }

    /// Plans a compaction for `reason` ending at `upto` (from
    /// [`Self::cut`]), of `messages` whose first `from` were summarized into
    /// `state` already: the state the new summary goes into, with the tool
    /// arguments the messages summarized anew used that the policy tracks,
    /// and the request for that summary to `spec`.
    pub fn plan(
        &self,
        state: &SummaryState,
        messages: &[Message],
        from: usize,
        upto: usize,
        spec: &ModelSpec,
        reason: &CompactReason,
    ) -> Result<(SummaryState, CompletionRequest), UnsupportedOption> {
        let older = messages.get(from.min(upto)..upto).unwrap_or_default();
        let focus = match reason {
            CompactReason::Asked { focus } => focus.trim(),
            _ => "",
        };
        let request = self
            .summarizer
            .request(older, &state.summary, focus, spec)?;
        let mut next = state.clone();
        next.track(older, &self.tracked);
        Ok((next, request))
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

impl SummaryPrompts {
    /// pi's structured checkpoint of an agent's work: goal, constraints,
    /// progress, decisions, next steps and critical context.
    pub const DEFAULT: Self = Self {
        system: Cow::Borrowed(
            "You summarize a conversation between a user and a coding agent so that another \
             model can continue the work from the summary alone. Read the conversation and \
             write the summary in the exact format asked for.\n\n\
             Do not continue the conversation. Do not answer questions in it. Output only the \
             summary.",
        ),
        initial: Cow::Borrowed(
            "The conversation above is to be summarized. Write a structured checkpoint of it \
             that another model will use to continue the work.",
        ),
        update: Cow::Borrowed(
            "The conversation above is the NEW part of a conversation whose earlier part is \
             summarized in <previous-summary>. Update that summary with it:\n\
             - keep everything in the previous summary that still holds;\n\
             - add the new progress, decisions and context;\n\
             - move items from \"In progress\" to \"Done\" when they were completed;\n\
             - update \"Next steps\" to what is left;\n\
             - drop what is no longer relevant.",
        ),
        format: Cow::Borrowed(
            "\n\nUse exactly this format:\n\n\
             ## Goal\n\
             [What the user is trying to get done; several items if the session covers several \
             tasks.]\n\n\
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
             messages.",
        ),
    };
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

impl Summarizer {
    /// [`SummaryPrompts::DEFAULT`] within [`SummaryLimits::DEFAULT`].
    pub const DEFAULT: Self = Self {
        prompts: SummaryPrompts::DEFAULT,
        limits: SummaryLimits::DEFAULT,
    };
}

impl Default for Summarizer {
    fn default() -> Self {
        Self::DEFAULT
    }
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
            let response = self
                .handler
                .handle(kind, Dispatch::new(EffectId::from_raw(0), false))
                .await
                .into_outcome()
                .await
                .and_then(Completion::unwrap)
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
