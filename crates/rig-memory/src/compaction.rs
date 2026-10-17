//! Compacting a long conversation: a [`CompactionPolicy`] says when and where to cut and plans
//! the summary, which [`apply_summary`] puts in place of the older messages; [`ClearToolOutputs`]
//! frees context without a model call. The default prompts and limits follow the
//! [pi coding agent](https://github.com/badlogic/pi-mono).
//!
//! ```
//! use rig_memory::{CompactionPolicy, SummaryState};
//!
//! let policy = CompactionPolicy::default();
//! let reason = rig_memory::CompactReason::Threshold;
//! assert_eq!(policy.cut(&[], 0, &reason, None), None);
//! assert_eq!(SummaryState::default().message(), None);
//! ```

use std::borrow::{Borrow, Cow};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, Message,
    UnsupportedOption,
};
use rig_core::conversation::{
    Aside, AsideId, Author, Change, Conversation, Entry, MessageId, Spliced,
};
use rig_core::message::{ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};

use crate::session::{LogRecord, as_of, history};
use crate::{HeuristicTokenCounter, SessionError, Store, TokenCounter};

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackArgument {
    /// The tool's name.
    pub tool: Cow<'static, str>,
    /// The argument's name.
    pub argument: Cow<'static, str>,
    /// The set's name.
    pub set: Cow<'static, str>,
}

impl TrackArgument {
    /// The rule tracking `argument` of `tool` in the set `set`.
    pub const fn new(tool: &'static str, argument: &'static str, set: &'static str) -> Self {
        Self {
            tool: Cow::Borrowed(tool),
            argument: Cow::Borrowed(argument),
            set: Cow::Borrowed(set),
        }
    }
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
    pub fn track(&mut self, messages: &[impl Borrow<Message>], rules: &[TrackArgument]) {
        for rule in rules {
            if !self.tracked.iter().any(|set| set.name == rule.set) {
                self.tracked.push(TrackedSet {
                    name: rule.set.clone().into_owned(),
                    values: BTreeSet::new(),
                });
            }
        }
        let calls = messages.iter().flat_map(|message| match message.borrow() {
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
                    .get(rule.argument.as_ref())
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
    messages: &[Arc<Message>],
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
///
/// Its serde form is one flat table, the summarizer's `prompts` and
/// `limits` beside the other fields; each field missing is the default's,
/// and an unknown one is refused.
///
/// ```
/// use rig_memory::CompactionPolicy;
///
/// let policy: CompactionPolicy =
///     serde_json::from_value(serde_json::json!({"keep_recent": 1000, "limits": {"reserve": 1}}))?;
/// assert_eq!(policy.keep_recent, 1000);
/// assert_eq!(policy.summarizer.limits.reserve, 1);
/// assert_eq!(policy.keep_asked, CompactionPolicy::default().keep_asked);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "PolicyTable", into = "PolicyTable")]
pub struct CompactionPolicy {
    /// The summarizer; its reserve is how much of the model's window is
    /// kept free.
    pub summarizer: Summarizer,
    /// The tool arguments tracked across compactions.
    pub tracked: Vec<TrackArgument>,
    /// Tokens of the newest messages an automatic compaction keeps as they
    /// are, a quarter of the model's window at most: the work carries on,
    /// so its recent part stays.
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
    /// output cleared.
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

/// The serde form of a [`CompactionPolicy`]: one flat table.
#[derive(Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PolicyTable {
    prompts: SummaryPrompts,
    limits: SummaryLimits,
    tracked: Vec<TrackArgument>,
    keep_recent: usize,
    keep_asked: usize,
    clearing: ClearToolOutputs,
}

impl Default for PolicyTable {
    fn default() -> Self {
        CompactionPolicy::default().into()
    }
}

impl From<PolicyTable> for CompactionPolicy {
    fn from(table: PolicyTable) -> Self {
        Self {
            summarizer: Summarizer {
                prompts: table.prompts,
                limits: table.limits,
            },
            tracked: table.tracked,
            keep_recent: table.keep_recent,
            keep_asked: table.keep_asked,
            clearing: table.clearing,
        }
    }
}

impl From<CompactionPolicy> for PolicyTable {
    fn from(policy: CompactionPolicy) -> Self {
        Self {
            prompts: policy.summarizer.prompts,
            limits: policy.summarizer.limits,
            tracked: policy.tracked,
            keep_recent: policy.keep_recent,
            keep_asked: policy.keep_asked,
            clearing: policy.clearing,
        }
    }
}

/// The estimated tokens of `messages`, by the default
/// [`HeuristicTokenCounter`], as a policy estimates them.
pub fn estimate_tokens(messages: &[impl Borrow<Message>]) -> u64 {
    let counter = HeuristicTokenCounter::default();
    messages
        .iter()
        .map(|message| counter.count(message.borrow()) as u64)
        .fold(0, u64::saturating_add)
}

impl CompactionPolicy {
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

    /// Where a compaction for `reason` should end: the index of the first
    /// message it keeps, after the newest messages within
    /// [`keep`](Self::keep) and never between a tool call and its result,
    /// or `None` when nothing new would be summarized. `from` is how many
    /// leading messages are summaries already ([`summarized_prefix`]). A
    /// short conversation is still summarized when the user asked or the
    /// model overflowed.
    pub fn cut(
        &self,
        messages: &[Arc<Message>],
        from: usize,
        reason: &CompactReason,
        spec: Option<&ModelSpec>,
    ) -> Option<usize> {
        cut_at(
            messages,
            from,
            self.keep(reason, spec),
            !matches!(reason, CompactReason::Threshold),
            &HeuristicTokenCounter::default(),
        )
    }

    /// The summary request for a compaction ending at `upto`, and the
    /// [`SummaryState`] the new summary goes into (with the tracked
    /// arguments of the newly summarized messages).
    pub fn plan(
        &self,
        state: &SummaryState,
        messages: &[Arc<Message>],
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

/// Who writes a compaction's summary: the author of its entry.
pub fn summary_author() -> Author {
    Author::plugin("compaction")
}

/// The summary a compaction put in, when `change` is one: the id of the
/// entry by [`summary_author`] in a splice that took messages out.
/// Clearing old tool outputs, by the same author, keeps every message and
/// is none.
pub fn summary_in(change: &Change) -> Option<MessageId> {
    let Change::Splice { removed, with, .. } = change else {
        return None;
    };
    let kept = |id: &MessageId| with.iter().any(|entry| entry.id == Some(*id));
    if removed.iter().all(kept) {
        return None;
    }
    let summary = summary_author();
    with.iter()
        .find(|entry| entry.author.as_ref() == Some(&summary))
        .and_then(|entry| entry.id)
}

/// How many of the first entries of `conversation` are summaries (by
/// [`summary_author`]): what a new compaction builds on from its
/// [`SummaryState`] instead of reading it again. It is the `from` the
/// policy's [`cut`](CompactionPolicy::cut) and
/// [`plan`](CompactionPolicy::plan) are given.
pub fn summarized_prefix(conversation: &Conversation) -> usize {
    let author = summary_author();
    let summaries = conversation.entries();
    summaries
        .take_while(|entry| entry.author == Some(&author))
        .count()
}

/// A summary [`CompactionPolicy::plan_summary`] planned.
#[derive(Clone, Debug)]
pub struct SummaryPlan {
    /// The first message the compaction keeps; `None` when it summarizes
    /// every message. By id, so a splice made while the summary is written
    /// does not move it.
    pub keep_from: Option<MessageId>,
    /// The state the summary goes into.
    pub state: SummaryState,
    /// The request for the summary.
    pub request: CompletionRequest,
}

/// Where a compaction [`CompactionPolicy::cut_summary`] chose ends, with no
/// request: for a summary asked at the end of the conversation itself
/// ([`Summarizer::instruction`]).
#[derive(Clone, Debug)]
pub struct SummaryCut {
    /// The index of the first message kept: the messages before it are
    /// summarized.
    pub upto: usize,
    /// The first message kept, by id; `None` when it summarizes every
    /// message.
    pub keep_from: Option<MessageId>,
    /// The state the summary goes into, with the tracked arguments of the
    /// newly summarized messages.
    pub state: SummaryState,
}

impl CompactionPolicy {
    /// [`cut`](Self::cut) for `conversation`, with the state the summary
    /// goes into. `None` when nothing new would be summarized.
    pub fn cut_summary(
        &self,
        conversation: &Conversation,
        state: &SummaryState,
        reason: &CompactReason,
        spec: &ModelSpec,
    ) -> Option<SummaryCut> {
        let messages = conversation.messages();
        let from = summarized_prefix(conversation);
        let upto = self.cut(messages, from, reason, Some(spec))?;
        let mut next = state.clone();
        next.track(
            messages.get(from.min(upto)..upto).unwrap_or_default(),
            &self.tracked,
        );
        Some(SummaryCut {
            upto,
            keep_from: conversation.id(upto),
            state: next,
        })
    }

    /// [`cut`](Self::cut) and [`plan`](Self::plan) in one step, for
    /// `conversation`. Returns `None` when nothing new would be summarized.
    ///
    /// # Errors
    ///
    /// When `spec` does not take the summary request's options.
    pub fn plan_summary(
        &self,
        conversation: &Conversation,
        state: &SummaryState,
        reason: &CompactReason,
        spec: &ModelSpec,
    ) -> Result<Option<SummaryPlan>, UnsupportedOption> {
        let messages = conversation.messages();
        let from = summarized_prefix(conversation);
        let Some(upto) = self.cut(messages, from, reason, Some(spec)) else {
            return Ok(None);
        };
        let (state, request) = self.plan(state, messages, from, upto, spec, reason)?;
        Ok(Some(SummaryPlan {
            keep_from: conversation.id(upto),
            state,
            request,
        }))
    }
}

/// Puts the summary of `state` in place of the messages of
/// `conversation` before `keep_from` (every message with `None`), earlier
/// summaries included: one user entry by [`summary_author`], a splice it
/// makes, followed by an aside of kind [`COMPACTED`] that says what it
/// replaced ([`Compacted`]), so [`undo_summary`] can put that back.
/// `None`, changing nothing, when `keep_from` was taken out meanwhile or
/// `state` has no summary.
pub fn apply_summary(
    conversation: &mut Conversation,
    keep_from: Option<MessageId>,
    state: &SummaryState,
) -> Option<Spliced> {
    let end = match keep_from {
        None => conversation.messages().len(),
        Some(kept) => conversation.index_of(kept)?,
    };
    let summary = Entry::new(Message::user(state.message()?)).by(summary_author());
    let spliced = conversation.splice(..end, [summary], summary_author());
    if let Some(&summary) = spliced.changed.first() {
        let compacted = Compacted {
            summary,
            replaced: spliced.removed.clone(),
            state: state.clone(),
        };
        let value = serde_json::to_value(compacted).unwrap_or_default();
        let aside = Aside::new(COMPACTED, summary_author(), value).after(Some(summary));
        conversation.add_aside(aside);
    }
    Some(spliced)
}

/// The kind of the aside [`apply_summary`] puts after each summary.
pub const COMPACTED: &str = "rig.compaction";

/// The summary state the next compaction of `conversation` builds on: the
/// newest compaction's not undone ([`Compacted`]), or the state before any.
pub fn summary_state(conversation: &Conversation) -> SummaryState {
    Compacted::of(conversation)
        .last()
        .map(|(_, compacted)| compacted.state)
        .unwrap_or_default()
}

/// What a compaction replaced, kept with the conversation in an aside of
/// kind [`COMPACTED`] by [`summary_author`] after its summary: the newest
/// one is what [`undo_summary`] puts back.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compacted {
    /// The summary's entry.
    pub summary: MessageId,
    /// The entries it replaced, earlier summaries included, in order.
    pub replaced: Vec<MessageId>,
    /// The state it put in, which the next compaction builds on.
    pub state: SummaryState,
}

impl Compacted {
    /// The record of every compaction of `conversation` not undone, oldest
    /// first, each with its aside.
    pub fn of(conversation: &Conversation) -> impl Iterator<Item = (AsideId, Self)> + '_ {
        let author = summary_author();
        let records = conversation.asides();
        let ours =
            records.filter(move |(_, aside)| aside.kind == COMPACTED && aside.author == author);
        ours.filter_map(|(id, aside)| Some((id, Self::deserialize(&aside.value).ok()?)))
    }
}

/// What [`undo_summary`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Uncompacted {
    /// The entries put back in place of the summary, with their ids.
    pub restored: Vec<MessageId>,
    /// The summary state now: the compaction's before it, or the state
    /// before any.
    pub state: SummaryState,
}

/// Why [`undo_summary`] put nothing back.
#[derive(Debug, thiserror::Error)]
pub enum UndoError {
    /// The conversation has no compaction left to undo.
    #[error("there is no compaction to undo")]
    Nothing,
    /// The newest compaction's summary is no longer in the conversation.
    #[error("its summary is no longer in the conversation")]
    SummaryGone,
    /// The session's log does not hold what it replaced, such as before
    /// the compaction was saved.
    #[error("the session's log does not hold what it replaced")]
    NotInLog,
    /// The session's log could not be read.
    #[error(transparent)]
    Load(#[from] SessionError),
}

/// Undoes the newest compaction of `conversation`, the conversation saved
/// in the log `log` of `store`: the entries its summary replaced, read
/// from that log as they were just before it, are put back in the
/// summary's place with their ids ([`Conversation::restore`] by `by`), and
/// its [`Compacted`] aside goes. Entries added after the summary stay
/// after them. Undone one after another, compactions come back newest
/// first, each with the summary it built on.
///
/// # Errors
///
/// [`UndoError`], changing nothing.
pub fn undo_summary(
    conversation: &mut Conversation,
    store: &dyn Store,
    log: &str,
    by: Author,
) -> Result<Uncompacted, UndoError> {
    let (aside, compacted) = Compacted::of(conversation)
        .last()
        .ok_or(UndoError::Nothing)?;
    let at = conversation
        .index_of(compacted.summary)
        .ok_or(UndoError::SummaryGone)?;
    // The splice that made the summary, the first naming it: later ones
    // put it back.
    let made = history(store, log)?
        .into_iter()
        .find_map(|(stamp, record)| match record {
            LogRecord::Change(recorded)
                if summary_in(&recorded.change) == Some(compacted.summary) =>
            {
                Some(stamp.n)
            }
            _ => None,
        });
    let before = as_of(store, log, made.ok_or(UndoError::NotInLog)?)?;
    let originals: Option<Vec<Entry>> = (compacted.replaced.iter())
        .map(|id| before.entry(before.index_of(*id)?))
        .collect();
    let originals = originals.ok_or(UndoError::NotInLog)?;
    let restored = conversation.restore(at..=at, originals, by).restored;
    conversation.remove_aside(aside);
    let state = Compacted::of(conversation).last().map(|(_, it)| it.state);
    Ok(Uncompacted {
        restored,
        state: state.unwrap_or_default(),
    })
}

/// What a [`Summarizer`] asks the model. Each prompt missing from its serde
/// form is the default's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SummaryPrompts {
    /// The summarizer's system prompt.
    pub system: Cow<'static, str>,
    /// The request for a first summary, after the conversation.
    pub initial: Cow<'static, str>,
    /// The request to fold the conversation into `<previous-summary>`.
    pub update: Cow<'static, str>,
    /// The summary's format, after any request.
    pub format: Cow<'static, str>,
    /// The request for a summary sent at the end of the conversation
    /// itself, after the agent's own system prompt and tools, in place of
    /// `system`, `initial` and `update` ([`Summarizer::instruction`]).
    pub in_place: Cow<'static, str>,
}

impl SummaryPrompts {
    /// A structured checkpoint of an agent's work: goal, constraints,
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
        in_place: Cow::Borrowed(
            "Stop here: do not continue the work, do not answer the user and do not call any \
             tool. Instead write a structured checkpoint of the conversation above. It replaces \
             the older messages, so another model must be able to continue the work from it and \
             the messages kept after it alone. If the conversation opens with a summary of an \
             earlier part, fold it in: keep everything in it that still holds, add the new \
             progress, decisions and context, move items from \"In progress\" to \"Done\" when \
             they were completed, update \"Next steps\" to what is left, and drop what is no \
             longer relevant.",
        ),
    };
}

/// The sizes a [`Summarizer`] works within. Each size missing from its
/// serde form is the default's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    /// snippets and 400k characters for an unknown window.
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

impl Default for SummaryPrompts {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Builds the request for a summary of a conversation's older messages and
/// reads the summary from the model's answer. The conversation goes as
/// text in one user message, so the model reads it rather than continues
/// it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
        older: &[Arc<Message>],
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
        Ok(CompletionRequest::new(Message::user(prompt))
            .preamble(self.prompts.system.clone().into_owned())
            .options(options)
            .max_tokens(self.max_tokens(spec)))
    }

    /// The longest summary asked of `spec`: the limit's, or the model's
    /// output limit when that is lower.
    pub fn max_tokens(&self, spec: &ModelSpec) -> u64 {
        let most = self.limits.summary_tokens;
        spec.max_output_tokens
            .map_or(most, |model| most.min(u64::from(model)))
    }

    /// The message that asks for a summary at the end of the conversation
    /// itself, the agent's own request with its system prompt and tools, so
    /// the provider can read all of it from its prompt cache:
    /// [`SummaryPrompts::in_place`], which messages to summarize, `focus`
    /// when it is not empty, and the format. `kept` is the first message
    /// the compaction keeps, named by the start of its text or the tools it
    /// calls, since the model sees no message ids; `None` summarizes every
    /// message.
    pub fn instruction(&self, kept: Option<&Message>, focus: &str) -> String {
        let mut text = self.prompts.in_place.clone().into_owned();
        match kept {
            Some(kept) => {
                let _ = write!(
                    text,
                    "\n\nSummarize what comes before {}: that message and everything after it \
                     stay after your summary, word for word.",
                    boundary(kept)
                );
            }
            None => text.push_str("\n\nSummarize the whole conversation."),
        }
        if !focus.is_empty() {
            let _ = write!(text, "\n\nAdditional focus: {focus}");
        }
        text.push_str(&self.prompts.format);
        text
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
    /// outputs and arguments cut.
    fn serialize(&self, messages: &[Arc<Message>]) -> String {
        let snippet = |text: &str| snippet(text, self.limits.snippet_chars);
        let mut parts: Vec<String> = Vec::new();
        for message in messages {
            match &**message {
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
                                let arguments = serde_json::to_string(&call.function.arguments);
                                parts.push(format!(
                                    "[Assistant tool call]: {}({})",
                                    call.function.name.as_str(),
                                    snippet(&arguments.unwrap_or_default())
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

/// How [`Summarizer::instruction`] names `message`: by the start of its
/// text, or by the tools a reply without text calls.
fn boundary(message: &Message) -> String {
    let text = match message {
        Message::System { content } => content.clone(),
        Message::User { content } => {
            let text: Vec<&str> = content
                .iter()
                .filter_map(|part| match part {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            text.join(" ")
        }
        Message::Assistant(reply) => reply.text(),
    };
    if !text.trim().is_empty() {
        return format!("the message that begins “{}”", crate::meta::title_of(&text));
    }
    let Message::Assistant(reply) = message else {
        return "the newest message".to_owned();
    };
    let mut names: Vec<&str> = reply
        .tool_calls()
        .map(|call| call.function.name.as_str())
        .collect();
    names.dedup();
    let first = reply.tool_calls().next().map(|call| {
        let arguments = serde_json::to_string(&call.function.arguments).unwrap_or_default();
        crate::meta::title_of(&arguments)
    });
    match (names.as_slice(), first) {
        ([], _) | (_, None) => "your newest reply".to_owned(),
        (names, Some(arguments)) => format!(
            "your reply that calls `{}` (the first call with {arguments})",
            names.join("`, `")
        ),
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

/// What [`ClearToolOutputs::clear`] took out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cleared {
    /// Tool results cleared.
    pub results: usize,
    /// Their estimated tokens.
    pub tokens: usize,
}

/// Frees context by clearing the outputs of older tool calls: walking back
/// from the end, the outputs within the first `keep_tokens` stay, and every
/// older one is replaced by a placeholder. The last message is never
/// touched: it is what the model must answer. Error results stay, being
/// short and telling the model what went wrong. The calls themselves stay,
/// so the conversation keeps its shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClearToolOutputs {
    /// Tokens of the newest tool output kept.
    pub keep_tokens: usize,
}

impl ClearToolOutputs {
    /// What a cleared output says.
    pub const PLACEHOLDER: &'static str =
        "[output cleared to fit the context window; run the tool again if needed]";

    /// Keep the newest `keep_tokens` of tool output, counted by the default
    /// [`HeuristicTokenCounter`], and say [`Self::PLACEHOLDER`] instead of
    /// the rest.
    pub fn new(keep_tokens: usize) -> Self {
        Self { keep_tokens }
    }

    /// Clears older tool outputs of `messages` in place.
    pub fn clear(&self, messages: &mut [Arc<Message>]) -> Cleared {
        let (targets, cleared) = self.targets(messages);
        for (at, items) in targets {
            if let Some(message) = messages.get_mut(at) {
                clear_items(Arc::make_mut(message), &items);
            }
        }
        cleared
    }

    /// Clears older tool outputs of `conversation`, as [`Self::clear`]
    /// does, rewriting the entries that change in place by `by`
    /// ([`Conversation::edit_each`]), so they keep their ids.
    pub fn clear_conversation(&self, conversation: &mut Conversation, by: Author) -> Cleared {
        let (targets, cleared) = self.targets(conversation.messages());
        conversation.edit_each(by, |entry| {
            let items = targets.get(&entry.index)?;
            let mut message = entry.message.clone();
            clear_items(&mut message, items);
            Some(message)
        });
        cleared
    }

    /// The tool results of `messages` to clear, as the indices of their
    /// content items by message, and what clearing them takes out.
    fn targets(&self, messages: &[impl Borrow<Message>]) -> (BTreeMap<usize, Vec<usize>>, Cleared) {
        let counter = HeuristicTokenCounter::default();
        let mut cleared = Cleared::default();
        let mut targets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut kept = 0;
        let Some((_, earlier)) = messages.split_last() else {
            return (targets, cleared);
        };
        for (at, message) in earlier.iter().enumerate().rev() {
            let Message::User { content } = message.borrow() else {
                continue;
            };
            for (item_at, item) in content.iter().enumerate().rev() {
                let tokens = counter.count_user(item);
                let UserContent::ToolResult(result) = item else {
                    continue;
                };
                if result.is_error || is_cleared(&result.content) {
                    continue;
                }
                kept += tokens;
                if kept <= self.keep_tokens {
                    continue;
                }
                targets.entry(at).or_default().push(item_at);
                cleared.results += 1;
                cleared.tokens += tokens;
            }
        }
        (targets, cleared)
    }
}

/// Replaces the tool results at `items` of `message` with the placeholder.
fn clear_items(message: &mut Message, items: &[usize]) {
    let Message::User { content } = message else {
        return;
    };
    for (at, item) in content.iter_mut().enumerate() {
        if let UserContent::ToolResult(result) = item
            && items.contains(&at)
        {
            result.content = vec![ToolResultContent::text(ClearToolOutputs::PLACEHOLDER)];
        }
    }
}

/// Whether `content` is a cleared output.
fn is_cleared(content: &[ToolResultContent]) -> bool {
    matches!(content, [only] if only.as_text() == Some(ClearToolOutputs::PLACEHOLDER))
}

#[cfg(test)]
mod tests;
