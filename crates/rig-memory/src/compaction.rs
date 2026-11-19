//! Compacting a long conversation: [`prepare_compaction`] says what a compaction summarizes and
//! builds its summary requests, and [`apply_summary`] puts the summary in place of the older
//! messages, keeping what it replaced so [`undo_summary`] can put that back.
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

use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::conversation::{
    Aside, AsideId, Author, Change, Conversation, Entry, MessageId, Spliced,
};
use rig_core::tool::content_hash;
use serde::{Deserialize, Serialize};

use crate::session::replay;
use crate::{SessionError, Store};

mod pi;

pub use pi::{
    CompactionPreparation, CompactionSettings, CutPoint, FileLists, SUMMARIZATION_SYSTEM_PROMPT,
    SummaryFailure, clamp_max_tokens, estimate_message_tokens, estimate_tokens, find_cut_point,
    merge_split_turn, no_history, prepare_compaction, prepare_compaction_at, request_tokens,
    serialize_conversation, summarization_request, summary_request, summary_text,
    turn_prefix_request,
};

/// A compaction's summary and the files listed after it, which the next compaction carries on.
/// The default is the state before the first compaction. A state saved by an earlier version
/// reads as its summary alone.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryState {
    /// The summary; empty before the first compaction.
    pub summary: String,
    /// The files read and not changed by the summarized messages, sorted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_files: Vec<String>,
    /// The files changed by the summarized messages, sorted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

    /// The summary followed by its file lists.
    fn with_files(&self) -> String {
        let files = pi::format_file_operations(&self.read_files, &self.modified_files);
        format!("{}{files}", self.summary)
    }

    /// [`with_files`](Self::with_files) in the frame requests send it in.
    fn wrapped(&self) -> String {
        format!(
            "The conversation history before this point was compacted into the following summary:\n\n<summary>\n{}\n</summary>",
            self.with_files()
        )
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

/// Puts the summary of `state` in place of the messages of
/// `conversation` before `keep_from` (every message with `None`), earlier
/// summaries included: one user entry by [`summary_author`], a splice it
/// makes, followed by an aside of kind [`COMPACTED`] that says what it
/// replaced ([`Compacted`]), so [`undo_summary`] can put that back.
/// `None`, changing nothing, when `keep_from` was taken out meanwhile.
pub fn apply_summary(
    conversation: &mut Conversation,
    keep_from: Option<MessageId>,
    state: &SummaryState,
) -> Option<Spliced> {
    let end = match keep_from {
        None => conversation.messages().len(),
        Some(kept) => conversation.index_of(kept)?,
    };
    let summary = Entry::new(Message::user(state.wrapped())).by(summary_author());
    let entries: Vec<Entry> = (0..end).filter_map(|at| conversation.entry(at)).collect();
    let replaced_hash = serde_json::to_vec(&entries).ok();
    let spliced = conversation.splice(..end, [summary], summary_author());
    if let Some(&summary) = spliced.changed.first() {
        let compacted = Compacted {
            summary,
            replaced: spliced.removed.clone(),
            replaced_hash: replaced_hash
                .and_then(|bytes| content_hash(&bytes).0.get(..HASH_LEN).map(str::to_owned)),
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

/// The hex digits of [`Compacted::replaced_hash`]: 128 bits, as every
/// reset writes it again for each record.
const HASH_LEN: usize = 32;

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
    /// The start of the SHA-256 of those entries, ids and made-times
    /// included, taken when it was made: what tells the same compaction
    /// from another conversation's with equal ids. `None` in a record saved
    /// before it was kept, which proves nothing across a reset.
    #[serde(default, rename = "hash", skip_serializing_if = "Option::is_none")]
    pub replaced_hash: Option<String>,
    /// The state it put in, which the next compaction builds on.
    pub state: SummaryState,
}

impl Compacted {
    /// The record of every compaction of `conversation` not undone, oldest
    /// first, each with its aside.
    pub fn of(conversation: &Conversation) -> impl Iterator<Item = (AsideId, Self)> + '_ {
        let records = conversation.asides();
        records.filter_map(|(id, aside)| Some((id, Self::read(aside)?)))
    }

    /// The record `aside` holds, if it is one.
    fn read(aside: &Aside) -> Option<Self> {
        let ours = aside.kind == COMPACTED && aside.author == summary_author();
        ours.then(|| rig_core::json_utils::from_value(&aside.value).ok())?
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
    /// The session's log does not hold what it replaced: the compaction
    /// was not saved there, or a reset brought its summary in, or brought
    /// it back with a record that cannot prove it is the same compaction.
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
    let mut known = provenance(store, log)?;
    if known.proofs.get(&compacted.summary) != Some(&compacted) {
        return Err(UndoError::NotInLog);
    }
    let originals = known
        .first
        .remove(&compacted.summary)
        .flatten()
        .ok_or(UndoError::NotInLog)?;
    let restored = conversation.restore(at..=at, originals, by).restored;
    conversation.remove_aside(aside);
    Ok(Uncompacted {
        restored,
        state: summary_state(conversation),
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

/// What a log's compactions replaced, between two resets of it.
#[derive(Default)]
struct Provenance {
    /// What each summary first replaced, `None` for entries the log lacks.
    first: BTreeMap<MessageId, Option<Vec<Entry>>>,
    records: BTreeMap<MessageId, Compacted>,
    /// The record each compaction saved here.
    proofs: BTreeMap<MessageId, Compacted>,
}

impl Provenance {
    /// Copies to `into` what this knows of `summary`, and nothing of the
    /// summaries it replaced, each of which proves itself.
    fn carry(&self, summary: MessageId, into: &mut Self) {
        if let Some(taken) = self.first.get(&summary) {
            into.first.insert(summary, taken.clone());
        }
        if let Some(record) = self.records.get(&summary) {
            into.records.insert(summary, record.clone());
        }
        if let Some(proof) = self.proofs.get(&summary) {
            into.proofs.insert(summary, proof.clone());
        }
    }
}

/// What the compactions in the log `log` of `store` replaced, read in one
/// pass over the log. A summary named again, as by an undo that put it
/// back, keeps what it first replaced. A reset carries each summary over
/// on its own, only when the new value holds a record equal to the one its
/// compaction saved here, [`Compacted::replaced_hash`] included, so a
/// summary of another conversation that reuses an id takes nothing from
/// this one. A summary whose entries the log does not hold is left out.
///
/// # Errors
///
/// As [`load`](crate::load).
pub fn summarized_in_log(store: &dyn Store, log: &str) -> Result<Summarized, SessionError> {
    let known = provenance(store, log)?;
    let found = known.first.into_iter();
    let replaced = found
        .filter_map(|(summary, taken)| Some((summary, taken?)))
        .collect();
    let records = known.records;
    Ok(Summarized { replaced, records })
}

/// The [`Provenance`] of the log `log` of `store` as it ends.
fn provenance(store: &dyn Store, log: &str) -> Result<Provenance, SessionError> {
    let mut known = Provenance::default();
    let mut stretches: Vec<Provenance> = Vec::new();
    replay(store, log, |before, recorded| {
        let change = &recorded.change;
        if let Change::Reset { state } = change {
            stretches.push(std::mem::take(&mut known));
            for (_, record) in Compacted::of(state) {
                let proved = |then: &&Provenance| {
                    record.replaced_hash.is_some()
                        && then.proofs.get(&record.summary) == Some(&record)
                };
                if let Some(then) = stretches.iter().rev().find(proved) {
                    then.carry(record.summary, &mut known);
                }
            }
            return;
        }
        if let Change::Aside {
            aside: Some(aside), ..
        } = change
            && let Some(record) = Compacted::read(aside)
        {
            known.proofs.entry(record.summary).or_insert(record);
            return;
        }
        let (Some(summary), Change::Splice { removed, with, .. }) = (summary_in(change), change)
        else {
            return;
        };
        let kept = |id: &&MessageId| with.iter().any(|entry| entry.id == Some(**id));
        let taken: Vec<&MessageId> = removed.iter().filter(|id| !kept(id)).collect();
        for (_, record) in Compacted::of(before) {
            if taken.contains(&&record.summary) {
                known.records.entry(record.summary).or_insert(record);
            }
        }
        // An entry put back carries its made-time: an undo, which replaced nothing.
        if with
            .iter()
            .any(|entry| entry.id == Some(summary) && entry.made.is_some())
        {
            return;
        }
        known.first.entry(summary).or_insert_with(|| {
            taken
                .iter()
                .map(|id| before.entry(before.index_of(**id)?))
                .collect()
        });
    })?;
    Ok(known)
}

#[cfg(test)]
mod tests;
