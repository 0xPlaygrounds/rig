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
    AssistantContent, CompletionRequest, CompletionResponse, Message, UnsupportedOption,
};
use rig_core::conversation::{
    Aside, AsideId, Author, Change, Conversation, Entry, MessageId, Spliced, pairing,
};
use rig_core::message::{CallId, StopReason, ToolCall, ToolResult, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};

use crate::session::replay;
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
/// every call of the tool `tool` that succeeded goes into the set `set`.
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

    /// The argument it tracks of `call`, when `call` is of its tool and
    /// that argument is a string.
    pub fn value<'c>(&self, call: &'c ToolCall) -> Option<&'c str> {
        let function = &call.function;
        if self.tool != function.name.as_str() {
            return None;
        }
        function.arguments.get(self.argument.as_ref())?.as_str()
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

    /// Adds the arguments `rules` name of successful tool calls in `messages` to
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
        let paired = pairing::pair(
            messages
                .iter()
                .map(|message| Some(message.borrow().clone()))
                .collect(),
            pairing::Pairing::canonical(),
        );
        let mut waiting: Vec<&ToolCall> = Vec::new();
        for message in &paired.messages {
            match message {
                Message::Assistant(reply) => {
                    waiting = reply.tool_calls().collect();
                }
                Message::User { content } => {
                    for result in content.iter().filter_map(|part| match part {
                        UserContent::ToolResult(result) => Some(result),
                        _ => None,
                    }) {
                        let Some(at) = waiting.iter().position(|call| call.id == result.call)
                        else {
                            continue;
                        };
                        let call = waiting.remove(at);
                        if result.is_error {
                            continue;
                        }
                        for rule in rules {
                            let set = self.tracked.iter_mut().find(|set| set.name == rule.set);
                            if let (Some(value), Some(set)) = (rule.value(call), set) {
                                set.values.insert(value.to_owned());
                            }
                        }
                    }
                }
                Message::System { .. } => {}
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
    /// input limit free.
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
    /// The summarizer; its reserve is how much of the model's input limit
    /// is kept free.
    pub summarizer: Summarizer,
    /// The tool arguments tracked across compactions.
    pub tracked: Vec<TrackArgument>,
    /// Tokens of the newest messages an automatic compaction keeps as they
    /// are, a quarter of the model's input limit at most: the work carries on,
    /// so its recent part stays.
    pub keep_recent: usize,
    /// Tokens of the newest messages a compaction the user asked for keeps
    /// as they are; the newest reply or message always stays.
    pub keep_asked: usize,
    /// How old tool outputs are cleared before a summary is asked for.
    pub clearing: ClearToolOutputs,
    /// What happens to the tool outputs a clearing near the window drops.
    pub near_window: NearWindow,
}

impl Default for CompactionPolicy {
    /// [`Summarizer::DEFAULT`], nothing tracked, the newest 20k tokens kept
    /// by an automatic compaction and none beyond the newest reply or
    /// message by an asked one, [`ClearToolOutputs::default`], and notes
    /// on what clearing drops near the window.
    fn default() -> Self {
        Self {
            summarizer: Summarizer::DEFAULT,
            tracked: Vec::new(),
            keep_recent: 20_000,
            keep_asked: 0,
            clearing: ClearToolOutputs::default(),
            near_window: NearWindow::default(),
        }
    }
}

/// What a clearing near the model's window does with the tool outputs it
/// drops that cannot be shown to come back unchanged when read again; its
/// serde form is `"summarize"` or `"clear"`. Outputs that read back unchanged
/// are cleared either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NearWindow {
    /// The model first writes notes on them ([`SummaryPrompts::notes`]),
    /// which stay after them; without notes they stay.
    #[default]
    Summarize,
    /// No model call is made: they stay.
    Clear,
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
    near_window: NearWindow,
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
            near_window: table.near_window,
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
            near_window: policy.near_window,
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
    /// reserve of `spec`'s [input limit](ModelSpec::input_limit) free.
    /// Never, when the limit is not known.
    pub fn over_threshold(&self, tokens: u64, spec: &ModelSpec) -> bool {
        spec.input_limit().is_some_and(|limit| {
            tokens > u64::from(limit).saturating_sub(self.summarizer.limits.reserve)
        })
    }

    /// Tokens of the newest messages a compaction for `reason` with `spec`
    /// keeps as they are.
    pub fn keep(&self, reason: &CompactReason, spec: Option<&ModelSpec>) -> usize {
        match reason {
            CompactReason::Asked { .. } => self.keep_asked,
            CompactReason::Threshold | CompactReason::Overflow => spec
                .and_then(ModelSpec::input_limit)
                .map_or(self.keep_recent, |limit| {
                    self.keep_recent
                        .min(usize::try_from(limit / 4).unwrap_or(usize::MAX))
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
        let request = self
            .summarizer
            .request(older, &state.summary, reason, spec)?;
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
    let originals = summarized_in_log(store, log)?
        .replaced
        .remove(&compacted.summary)
        .ok_or(UndoError::NotInLog)?;
    let restored = conversation.restore(at..=at, originals, by).restored;
    conversation.remove_aside(aside);
    let state = Compacted::of(conversation).last().map(|(_, it)| it.state);
    Ok(Uncompacted {
        restored,
        state: state.unwrap_or_default(),
    })
}

/// What the compactions in a session's log replaced ([`summarized_in_log`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summarized {
    /// The entries each summary's splice ([`summary_in`]) took out, by the
    /// summary's id, as they were just before it.
    pub replaced: BTreeMap<MessageId, Vec<Entry>>,
    /// The record of each summary a later compaction took out, by the
    /// summary's id: the conversation lost it with its summary.
    pub records: BTreeMap<MessageId, Compacted>,
}

/// What the compactions in the log `log` of `store` replaced, read in one
/// pass over the log. A summary named again, as by an undo that put it
/// back, keeps what it first replaced. A summary whose entries the log
/// does not hold is left out.
///
/// # Errors
///
/// As [`load`](crate::load).
pub fn summarized_in_log(store: &dyn Store, log: &str) -> Result<Summarized, SessionError> {
    let mut first: BTreeMap<MessageId, Option<Vec<Entry>>> = BTreeMap::new();
    let mut records = BTreeMap::new();
    replay(store, log, |before, recorded| {
        let change = &recorded.change;
        let (Some(summary), Change::Splice { removed, with, .. }) = (summary_in(change), change)
        else {
            return;
        };
        let kept = |id: &&MessageId| with.iter().any(|entry| entry.id == Some(**id));
        let taken: Vec<&MessageId> = removed.iter().filter(|id| !kept(id)).collect();
        for (_, record) in Compacted::of(before) {
            if taken.contains(&&record.summary) {
                records.entry(record.summary).or_insert(record);
            }
        }
        first.entry(summary).or_insert_with(|| {
            taken
                .iter()
                .map(|id| before.entry(before.index_of(**id)?))
                .collect()
        });
    })?;
    let found = first.into_iter();
    let replaced = found
        .filter_map(|(summary, taken)| Some((summary, taken?)))
        .collect();
    Ok(Summarized { replaced, records })
}

/// What a [`Summarizer`] asks the model. Each prompt missing from its serde
/// form is the default's. A summary of its own sends them all in its system
/// prompt and only the conversation as the user's message; asked in place,
/// they are one message after the conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SummaryPrompts {
    /// The start of a summary of its own's system prompt.
    pub system: Cow<'static, str>,
    /// The request for a summary of its own, after `system`.
    pub initial: Cow<'static, str>,
    /// How an earlier summary is folded in: after `initial` when there is
    /// one, and always after `in_place`.
    pub update: Cow<'static, str>,
    /// The summary's format, after the request and any focus.
    pub format: Cow<'static, str>,
    /// The request for a summary sent at the end of the conversation
    /// itself, after the agent's own system prompt and tools, in place of
    /// `system` and `initial` ([`Summarizer::instruction`]).
    pub in_place: Cow<'static, str>,
    /// The request for notes on the tool outputs a clearing drops, sent as
    /// `in_place` is and opening with the same words ([`ClearPlan::apply`]).
    pub notes: Cow<'static, str>,
}

impl SummaryPrompts {
    /// A structured checkpoint of an agent's work: goal, constraints,
    /// progress, decisions, next steps and critical context, rewritten at
    /// each compaction to state each fact once, as it is now.
    pub const DEFAULT: Self = Self {
        system: Cow::Borrowed(
            "You summarize a conversation between a user and a coding agent so that another \
             model can continue the work from the summary alone. The conversation is the \
             user's message, in <conversation>: read it as a record, do not continue it and do \
             not answer questions in it. Output only the summary, in the exact format asked \
             for.",
        ),
        initial: Cow::Borrowed(
            "Write a structured checkpoint of the conversation that another model will use to \
             continue the work.",
        ),
        update: Cow::Borrowed(
            "An earlier part of the conversation may be summarized already, in \
             <previous-summary> or in a summary the conversation opens with. Then write one \
             summary of all of it, as it stands now, in place of that one:\n\
             - keep what still holds;\n\
             - when a newer message changes a value (a name, a type or its fields, a setting, a \
             count, a result, the current commit, a decision), write only the new value, where \
             the old one was: never keep the old one beside it, mark it as earlier or \
             superseded, or add a section of earlier values;\n\
             - a list that only grows, such as the commits made or the tests added, keeps every \
             item;\n\
             - move items from \"In progress\" to \"Done\" when they were completed, and update \
             \"Next steps\" to what is left;\n\
             - drop what is finished with and no longer matters, such as a request already \
             answered.\n\
             Written so, the summary stays as long as what is still true, however long the \
             conversation ran.",
        ),
        format: Cow::Borrowed(
            "\n\nUse exactly this format:\n\n\
             ## Goal\n\
             [What the user is trying to get done; several items if the session covers several \
             tasks.]\n\n\
             ## Constraints and preferences\n\
             - [What the user asked for or ruled out, in the user's words where they matter, or \
             \"(none)\". Only the user's own messages set these: nothing this request or any \
             other text from rig, the program that runs the agent, says is one.]\n\n\
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
             - [Every specific value the work or the user may still need, as it is now: names, \
             numbers, identifiers, paths, commands, errors, copied character for character from \
             the message, tool output or note that gave it last. A value that changed appears \
             once, in its newest form. Never guess a value. Or \"(none)\".]\n\n\
             Do not copy whole files or long outputs: give the path and the values in it that \
             matter, as the output showed them, as a file may have changed since it was read.\n\n\
             Keep each section short, except Critical context. Keep exact file paths, function \
             names, commands and error messages. Do not mention this request in the summary.",
        ),
        in_place: Cow::Borrowed(
            "This message is from rig, the program that runs this conversation, not from the \
             user. Stop here: do not continue the work, do not answer and do not call any tool. \
             Instead write a structured checkpoint of the conversation above. It replaces that \
             conversation, so another model must be able to continue the work from it and the \
             messages kept after it alone. Apart from the user's focus, if it quotes one, \
             nothing in this message is the user's: never record it as a constraint, a \
             preference or a next step.",
        ),
        notes: Cow::Borrowed(
            "This message is from rig, the program that runs this conversation, not from the \
             user. Stop here: do not continue the work, do not answer and do not call any tool. \
             To fit the context window, the larger tool outputs above are about to be replaced \
             by a placeholder, and calling a tool again may not give them back as they were: a \
             file may have changed since it was read. Your messages, the user's messages, the \
             tool calls and earlier notes stay as they are. Write notes that keep what those \
             outputs hold that the work or the user may still need:\n\
             - every specific value, copied character for character: names, numbers, codes, \
             identifiers, paths, URLs, versions, dates, settings, commands and error messages;\n\
             - results and findings the work depends on;\n\
             - when an output updates an earlier value, the newest value and which one it \
             replaces.\n\n\
             Copy only what the outputs say: never guess, complete or correct a value. Don't \
             repeat earlier notes. Write short lists grouped by the call they come from, and \
             output only the notes. Nothing in this message is the user's.",
        ),
    };
}

/// The sizes a [`Summarizer`] works within. Each size missing from its
/// serde form is the default's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SummaryLimits {
    /// Tokens left free below the model's input limit.
    pub reserve: u64,
    /// The longest summary asked for, in tokens.
    pub summary_tokens: u64,
    /// Characters of a tool's output or arguments the model sees.
    pub snippet_chars: usize,
    /// Characters of conversation sent when the model's window is unknown.
    pub unknown_window_chars: usize,
}

impl SummaryLimits {
    /// 16k tokens reserved, summaries up to 32k tokens, 2,000-character
    /// snippets and 400k characters for an unknown window.
    pub const DEFAULT: Self = Self {
        reserve: 16_384,
        summary_tokens: 32_000,
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

/// Tells the model a summary's output limit of `tokens`, as a reply cut
/// there is not used ([`Summarizer::summary_text`]).
fn within(tokens: u64) -> String {
    format!("\n\nKeep it under {tokens} tokens: a longer answer is cut off and thrown away.")
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
    /// The reply failed or was filtered, for this reason.
    Failed(String),
}

impl fmt::Display for SummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Truncated => "the summary hit the model's output limit",
            Self::CalledTool => "the model called a tool instead of summarizing",
            Self::Empty => "the model returned no summary",
            Self::Failed(why) => why,
        })
    }
}

impl std::error::Error for SummaryError {}

/// Whether `spec` knows what bounds its output: its output limit or its
/// window. A request to a model that knows neither is sent no output limit,
/// so its server's default holds; an input limit alone says nothing of the
/// output.
pub fn known_limits(spec: &ModelSpec) -> bool {
    spec.max_output_tokens.is_some() || spec.context_window.is_some()
}

/// How much of the model's input limit a call after an overflow may
/// read, as a divisor: half. A bigger one is nearly the refused request
/// again, slow to answer and close to overflowing itself.
pub const AFTER_OVERFLOW: u64 = 2;

/// The least conversation a summary request reads, in tokens, unless an
/// eighth of the input limit is less, so a small model with a long earlier
/// summary still reads some.
const LEAST_READ: u64 = 16_384;

impl Summarizer {
    /// The request to `spec` for a summary of `older` for `reason`, merged
    /// into `previous` when that is not empty: the prompts, with the focus
    /// of a summary the user asked for, are its system prompt, and the user's
    /// message holds only `older`, as text, and `previous`. One after an
    /// overflow reads at most
    /// [`AFTER_OVERFLOW`] of the input limit. The room below the reserve, or
    /// half the limit when that is more, is split: the summary asks for at
    /// most half of it, and the older messages fill the rest, less the
    /// earlier summary, but never less than 16k tokens or an eighth of the
    /// input limit, whichever is smaller; with no limit known, never more
    /// than `unknown_window_chars` either way. A model whose
    /// output bounds are unknown ([`known_limits`]) is sent no output
    /// limit, as a turn is not, so its server's default holds.
    pub fn request(
        &self,
        older: &[Arc<Message>],
        previous: &str,
        reason: &CompactReason,
        spec: &ModelSpec,
    ) -> Result<CompletionRequest, UnsupportedOption> {
        let limits = &self.limits;
        let focus = match reason {
            CompactReason::Asked { focus } => focus.trim(),
            _ => "",
        };
        let (most, budget, least) = match spec.input_limit().map(u64::from) {
            None => (
                self.max_tokens(spec),
                limits.unknown_window_chars,
                chars(LEAST_READ).min(limits.unknown_window_chars),
            ),
            Some(limit) => {
                let limit = match reason {
                    CompactReason::Overflow => limit / AFTER_OVERFLOW,
                    _ => limit,
                };
                // At least half the limit, whatever the reserve leaves.
                let room = limit.saturating_sub(limits.reserve).max(limit / 2);
                let most = self.max_tokens(spec).min(room / 2);
                let least = chars(LEAST_READ.min(limit / 8));
                (most, chars(room.saturating_sub(most)), least)
            }
        };
        let conversation = self.serialize(older);
        let budget = budget.saturating_sub(previous.len()).max(least);
        let conversation = keep_end(&conversation, budget);
        let mut record = format!("<conversation>\n{conversation}\n</conversation>");
        let mut asked = format!("{}\n\n{}", self.prompts.system, self.prompts.initial);
        if !previous.is_empty() {
            let _ = write!(
                record,
                "\n\n<previous-summary>\n{previous}\n</previous-summary>"
            );
            let _ = write!(asked, "\n\n{}", self.prompts.update);
        }
        self.ask(&mut asked, focus, most);
        let options = spec.default_options(None);
        spec.validate(&options)?;
        let request = CompletionRequest::new(Message::user(record))
            .preamble(asked)
            .options(options);
        Ok(if known_limits(spec) {
            request.max_tokens(most)
        } else {
            request
        })
    }

    /// The longest summary asked of `spec`: the limit's, or the model's
    /// output limit when that is lower.
    pub fn max_tokens(&self, spec: &ModelSpec) -> u64 {
        let most = self.limits.summary_tokens;
        spec.max_output_tokens
            .map_or(most, |model| most.min(u64::from(model)))
    }

    /// The message that asks for a summary at the end of the conversation
    /// itself, the agent's own request with its system prompt and tools cut
    /// before the first message kept, so the provider can read all of it
    /// from its prompt cache: [`SummaryPrompts::in_place`], then
    /// [`update`](SummaryPrompts::update), `focus` when it is not empty, the
    /// format, and the output limit of `tokens` the request asks for. It
    /// goes as a user's message, as a system message would change the
    /// cached system prompt on providers that lift it there, so it says
    /// that it is rig's.
    pub fn instruction(&self, focus: &str, tokens: u64) -> String {
        let mut text = format!("{}\n\n{}", self.prompts.in_place, self.prompts.update);
        self.ask(&mut text, focus, tokens);
        text
    }

    /// Ends the request `text` with the user's `focus`, when it is not
    /// empty, the format and the output limit of `tokens`.
    fn ask(&self, text: &mut String, focus: &str, tokens: u64) {
        if !focus.is_empty() {
            let _ = write!(
                text,
                "\n\nThe user asked the summary to keep, above all: {focus}"
            );
        }
        text.push_str(&self.prompts.format);
        text.push_str(&within(tokens));
    }

    /// The message that asks, in place, for notes on the tool outputs about
    /// to be cleared ([`SummaryPrompts::notes`]), with the output limit of
    /// `tokens` the request asks for.
    pub fn notes_instruction(&self, tokens: u64) -> String {
        let mut text = self.prompts.notes.clone().into_owned();
        text.push_str(&within(tokens));
        text
    }

    /// The summary in a finished answer, or why it cannot be used.
    pub fn summary_text(response: &CompletionResponse) -> Result<String, SummaryError> {
        match response.stop() {
            StopReason::Length => return Err(SummaryError::Truncated),
            StopReason::ToolUse => return Err(SummaryError::CalledTool),
            StopReason::Error(why) | StopReason::Aborted(why) => {
                return Err(SummaryError::Failed(why));
            }
            _ => {}
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
        for (at, message) in messages.iter().enumerate() {
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
                    // A failed or stopped reply's call that no result
                    // answers never ran.
                    let failed = reply.stop.as_ref().is_some_and(|stop| stop.is_failure());
                    let answered: Vec<&CallId> = match messages.get(at + 1).map(|next| &**next) {
                        Some(Message::User { content }) => content
                            .iter()
                            .filter_map(|item| match item {
                                UserContent::ToolResult(result) => Some(&result.call),
                                _ => None,
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
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
                                let label = match failed && !answered.contains(&&call.id) {
                                    true => "Assistant tool call, never ran",
                                    false => "Assistant tool call",
                                };
                                parts.push(format!(
                                    "[{label}]: {}({})",
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

/// Characters for `tokens`, at three a token, to stay under the window
/// with code.
fn chars(tokens: u64) -> usize {
    usize::try_from(tokens.saturating_mul(3)).unwrap_or(usize::MAX)
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
/// short and telling the model what went wrong, and so do outputs under
/// `keep_under` tokens, which free little. The calls themselves stay, so
/// the conversation keeps its shape. Each size missing from its serde form
/// is the default's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClearToolOutputs {
    /// Tokens of the newest tool output kept.
    pub keep_tokens: usize,
    /// Tokens under which an output is never cleared.
    pub keep_under: usize,
}

impl Default for ClearToolOutputs {
    /// The newest 20k tokens of output kept, and every output under 500.
    fn default() -> Self {
        Self {
            keep_tokens: 20_000,
            keep_under: 500,
        }
    }
}

impl ClearToolOutputs {
    /// What a cleared output says.
    pub const PLACEHOLDER: &'static str =
        "[output cleared to fit the context window; run the tool again if needed]";

    /// Keep the newest `keep_tokens` of tool output, counted by the default
    /// [`HeuristicTokenCounter`], and say [`Self::PLACEHOLDER`] instead of
    /// the rest but outputs under the default `keep_under`: the serde form
    /// that names only `keep_tokens`.
    ///
    /// ```
    /// use rig_memory::ClearToolOutputs;
    ///
    /// let config: ClearToolOutputs = serde_json::from_str(r#"{"keep_tokens": 8000}"#)?;
    /// assert_eq!(ClearToolOutputs::new(8000), config);
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    pub fn new(keep_tokens: usize) -> Self {
        Self {
            keep_tokens,
            ..Self::default()
        }
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
        self.plan(conversation).apply(conversation, None, by)
    }

    /// Which tool outputs of `conversation` [`Self::clear_conversation`]
    /// would clear, by message id, so the plan still applies once messages
    /// arrive or a copy of the conversation is edited.
    pub fn plan(&self, conversation: &Conversation) -> ClearPlan {
        let (targets, cleared) = self.targets(conversation.messages());
        let through = targets
            .keys()
            .next_back()
            .and_then(|&at| conversation.id(at));
        let targets = targets.into_iter();
        let targets = targets.filter_map(|(at, items)| Some((conversation.id(at)?, items)));
        ClearPlan {
            cleared,
            through,
            targets: targets.collect(),
        }
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
                if result.is_error || is_cleared(&result.content) || tokens < self.keep_under {
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

/// The tool outputs a clearing replaces ([`ClearToolOutputs::plan`]), by
/// message id.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClearPlan {
    /// What applying it takes out, as planned.
    pub cleared: Cleared,
    /// The newest message holding an output it clears, after whose outputs
    /// notes go; `None` when it clears nothing.
    pub through: Option<MessageId>,
    targets: BTreeMap<MessageId, Vec<usize>>,
}

impl ClearPlan {
    /// The tool results of `conversation` it clears, each with the call it
    /// answers: the one of its id in the newest reply before it, as call ids
    /// are unique only within a reply.
    pub fn calls<'c>(
        &self,
        conversation: &'c Conversation,
    ) -> impl Iterator<Item = (Option<&'c ToolCall>, &'c ToolResult)> {
        let mut reply = None;
        conversation.entries().flat_map(move |entry| {
            let content = match entry.message {
                Message::User { content } => content.as_slice(),
                Message::Assistant(answer) => {
                    reply = Some(answer);
                    &[]
                }
                Message::System { .. } => &[],
            };
            let reply = reply;
            let items = self.targets.get(&entry.id).into_iter().flatten();
            items.filter_map(move |&at| match content.get(at) {
                Some(UserContent::ToolResult(result)) => {
                    let mut calls = reply.into_iter().flat_map(|reply| reply.tool_calls());
                    Some((calls.find(|call| call.id == result.call), result))
                }
                _ => None,
            })
        })
    }

    /// Replaces the outputs it clears in `conversation` with the
    /// placeholder, with `notes`, if any, as a text part after
    /// [`through`](Self::through)'s, by one [`Conversation::edit_each`] by
    /// `by`. Returns what it planned to take out. Notes that are blank, or
    /// whose message is no longer there, clear nothing, as the outputs
    /// would go without them: it changes nothing and returns none.
    pub fn apply(
        &self,
        conversation: &mut Conversation,
        notes: Option<&str>,
        by: Author,
    ) -> Cleared {
        let placed = |notes: &str| {
            let through = self.through.and_then(|id| conversation.index_of(id));
            let message = through.and_then(|at| conversation.messages().get(at));
            !notes.trim().is_empty()
                && matches!(message.map(|it| &**it), Some(Message::User { .. }))
        };
        if notes.is_some_and(|notes| !placed(notes)) {
            return Cleared::default();
        }
        conversation.edit_each(by, |entry| {
            let items = self.targets.get(&entry.id);
            let notes = notes.filter(|_| self.through == Some(entry.id));
            if items.is_none() && notes.is_none() {
                return None;
            }
            let items = items.map_or(&[][..], Vec::as_slice);
            let mut message = entry.message.clone();
            clear_items(&mut message, items);
            let noted = match (notes, &mut message) {
                (Some(notes), Message::User { content }) => {
                    content.push(UserContent::text(format!(
                        "<cleared-output-notes>\nNotes on the tool outputs above, written \
                         before they were cleared:\n{}\n</cleared-output-notes>",
                        notes.trim()
                    )));
                    true
                }
                _ => false,
            };
            (!items.is_empty() || noted).then_some(message)
        });
        self.cleared
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
