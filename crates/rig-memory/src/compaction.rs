//! Compacting a long conversation: [`prepare_compaction`] says what a compaction summarizes and
//! builds its summary requests, and [`apply_summary`] puts the summary in place of the messages
//! it read, when they are still as it read them.
//!
//! ```
//! use rig_core::conversation::Conversation;
//! use rig_memory::{CompactionSettings, SummaryState, prepare_compaction};
//!
//! let settings = CompactionSettings::default();
//! assert!(settings.should_compact(120_000, 128_000));
//! let conversation = Conversation::default();
//! assert_eq!(prepare_compaction(&conversation, settings.keep_recent_tokens, false), None);
//! assert_eq!(SummaryState::default().message(), None);
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::conversation::{Author, Change, Conversation, Entry, MessageId, Spliced};

use crate::session::replay;
use crate::{SessionError, Store};

mod pi;

pub use pi::{
    CompactionPreparation, CompactionSettings, FileLists, SUMMARIZATION_SYSTEM_PROMPT,
    SummaryFailure, clamp_max_tokens, estimate_message_tokens, estimate_tokens, find_cut_point,
    merge_split_turn, no_history, prepare_compaction, prepare_compaction_at, request_tokens,
    serialize_conversation, summarization_request, summary_request, summary_text,
    turn_prefix_request,
};

/// What the summary's text opens with in the conversation.
const OPENS: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";

/// What the summary's text closes with in the conversation.
const CLOSES: &str = "\n</summary>";

/// A compaction's summary and the files listed after it, which the next compaction carries on.
/// The default is the state before the first compaction. The summary's entry in the
/// conversation is its only record: [`apply_summary`] writes it and [`read`](Self::read) reads
/// it back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SummaryState {
    /// The summary; empty before the first compaction. File lists a model wrote at its end are
    /// not kept: the lists below are the only ones.
    pub summary: String,
    /// The files read and not changed by the summarized messages, sorted.
    pub read_files: Vec<String>,
    /// The files changed by the summarized messages, sorted.
    pub modified_files: Vec<String>,
}

impl SummaryState {
    /// The summary followed by its file lists, as the next compaction's update prompt reads it,
    /// as pi stores it: a summary with no text still carries its lists. `None` when both are
    /// empty, as before the first compaction.
    pub fn previous_summary(&self) -> Option<String> {
        Some(self.with_files()).filter(|text| !text.is_empty())
    }

    /// The text that stands for the summarized messages in requests, `None` before the first
    /// compaction.
    pub fn message(&self) -> Option<String> {
        self.previous_summary().map(|_| self.wrapped())
    }

    /// The state a summary entry's `text` holds: the summary, and the file lists rig put after
    /// it. Text that is no summary of rig's reads as a summary with no lists.
    pub fn read(text: &str) -> Self {
        let framed = text.strip_prefix(OPENS);
        let Some(body) = framed.and_then(|text| text.strip_suffix(CLOSES)) else {
            return Self {
                summary: text.to_owned(),
                ..Self::default()
            };
        };
        let (body, modified_files) = pi::take_file_list(body, "modified-files");
        let (summary, read_files) = pi::take_file_list(body, "read-files");
        Self {
            summary: summary.to_owned(),
            read_files,
            modified_files,
        }
    }

    /// The summary, without file lists a model wrote at its end, followed by its file lists.
    fn with_files(&self) -> String {
        let files = pi::format_file_operations(&self.read_files, &self.modified_files);
        format!("{}{files}", pi::without_file_lists(&self.summary))
    }

    /// [`with_files`](Self::with_files) in the frame requests send it in.
    fn wrapped(&self) -> String {
        format!("{OPENS}{}{CLOSES}", self.with_files())
    }
}

/// The end of `text`, a conversation another plugin summarizes for a model `spec` in at most
/// `summary_tokens`, cut to fit that model as rig's summaries were fitted before compaction
/// followed pi; the cut start says how many characters were left out. Returns the text and the
/// output limit to ask for, `None` when the model's limits are unknown. The room below a reserve
/// of 16384 tokens, or half the input limit when that is more, is split: the answer gets at most
/// half, and the text the rest at three characters a token, but never less than 16384 tokens or an
/// eighth of the input limit, whichever is smaller; with no limit known, 400000 characters.
pub fn fit_to_model<'t>(
    text: &'t str,
    spec: &ModelSpec,
    summary_tokens: u64,
) -> (std::borrow::Cow<'t, str>, Option<u64>) {
    let chars = |tokens: u64| usize::try_from(tokens.saturating_mul(3)).unwrap_or(usize::MAX);
    let most = spec
        .output_limit()
        .map_or(summary_tokens, |limit| summary_tokens.min(u64::from(limit)));
    let (most, budget) = match spec.input_limit().map(u64::from) {
        None => (most, 400_000),
        Some(limit) => {
            let room = limit.saturating_sub(16_384).max(limit / 2);
            let most = most.min(room / 2);
            let least = chars(16_384u64.min(limit / 8));
            (most, chars(room.saturating_sub(most)).max(least))
        }
    };
    let known = spec.max_output_tokens.is_some() || spec.context_window.is_some();
    let most = known.then_some(most);
    let left_out = text.chars().count().saturating_sub(budget);
    if left_out == 0 {
        return (std::borrow::Cow::Borrowed(text), most);
    }
    let start = text
        .char_indices()
        .nth(left_out)
        .map_or(text.len(), |(start, _)| start);
    let rest = text.get(start..).unwrap_or_default();
    let cut = format!("[\u{2026} {left_out} earlier characters left out]\n{rest}");
    (std::borrow::Cow::Owned(cut), most)
}

/// Who writes a compaction's summary: the author of its entry.
pub fn summary_author() -> Author {
    Author::plugin("compaction")
}

/// The summary a compaction put in, when `change` is one: the id of the
/// entry by [`summary_author`] in a splice that took messages out.
/// A splice by the same author that keeps every message, as an earlier
/// version's clearing of old tool outputs did, is none.
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
/// [`SummaryState`] instead of reading it again, so [`prepare_compaction`]
/// never cuts them again.
pub fn summarized_prefix(conversation: &Conversation) -> usize {
    let author = summary_author();
    let summaries = conversation.entries();
    summaries
        .take_while(|entry| entry.author == Some(&author))
        .count()
}

/// Whether `conversation` ends in its last compaction, as pi's session does when nothing was
/// added after it: a summary is in, and every other message is older than it.
pub fn already_compacted(conversation: &Conversation) -> bool {
    let author = summary_author();
    let mut summaries = conversation.entries();
    let summary = summaries
        .find(|entry| entry.author == Some(&author))
        .map(|entry| entry.id);
    summary.is_some_and(|summary| conversation.entries().all(|entry| entry.id <= summary))
}

/// Puts the summary of `state` in place of `replaces`, the first
/// messages of `conversation` as a compaction read them
/// ([`CompactionPreparation::replaces`]), earlier summaries included: one
/// user entry by [`summary_author`], in one splice, which is the
/// compaction's whole record. Messages added after them stay. `None`,
/// changing nothing, when `replaces` is empty, when the conversation no
/// longer starts with those very messages, such as after an edit of one, so
/// a summary never replaces what it did not read, or when what stays would
/// start with a tool result whose call the summary replaced.
pub fn apply_summary(
    conversation: &mut Conversation,
    replaces: &[Arc<Message>],
    state: &SummaryState,
) -> Option<Spliced> {
    let end = replaces.len();
    let now = conversation.messages();
    let read = now.get(..end).filter(|now| !now.is_empty())?;
    let same = read
        .iter()
        .zip(replaces)
        .all(|(now, read)| Arc::ptr_eq(now, read));
    if !same || !pi::may_keep_from(now, end) {
        return None;
    }
    let summary = Entry::new(Message::user(state.wrapped())).by(summary_author());
    Some(conversation.splice(..end, [summary], summary_author()))
}

/// The summary state the next compaction of `conversation` builds on: the
/// one its newest leading summary entry holds ([`summarized_prefix`],
/// [`SummaryState::read`]), or the state before any.
pub fn summary_state(conversation: &Conversation) -> SummaryState {
    let newest = summarized_prefix(conversation).checked_sub(1);
    let newest = newest.and_then(|at| conversation.messages().get(at)?.rag_text());
    newest
        .map(|text| SummaryState::read(&text))
        .unwrap_or_default()
}

/// The entries each compaction in the log `log` of `store` replaced
/// ([`summary_in`]), by the summary's id, as they were just before it, read
/// in one pass over the log. A summary named again keeps what it first
/// replaced. A reset, the log's value replaced by a whole other one, drops
/// them all: nothing says the new value is the same conversation, so its
/// summaries show nothing of the old one's messages.
///
/// # Errors
///
/// As [`load`](crate::load).
pub fn summarized_in_log(
    store: &dyn Store,
    log: &str,
) -> Result<BTreeMap<MessageId, Vec<Entry>>, SessionError> {
    let mut replaced = BTreeMap::new();
    replay(store, log, |before, recorded| {
        let change = &recorded.change;
        let (Some(summary), Change::Splice { removed, with, .. }) = (summary_in(change), change)
        else {
            if matches!(change, Change::Reset { .. }) {
                replaced.clear();
            }
            return;
        };
        let kept = |id: &&MessageId| with.iter().any(|entry| entry.id == Some(**id));
        let taken = removed.iter().filter(|id| !kept(id));
        let taken = taken.filter_map(|id| before.entry(before.index_of(*id)?));
        replaced.entry(summary).or_insert_with(|| taken.collect());
    })?;
    Ok(replaced)
}

#[cfg(test)]
mod tests;
