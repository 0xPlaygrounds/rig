//! A running session's logs: [`Writer`] queues each agent's changes as a
//! host saves them and writes them on [`flush`](Writer::flush).

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_core::conversation::{Change, ChangeRecord, Conversation, Lineage, MessageId};
use serde_json::Value;

use crate::blobs::{PutBlob, put_blob, read_blob, store_change};
use crate::fork::turn_cut;
use crate::format::{Header, LogLines, Meta, SESSION, write_change, write_meta};
use crate::session::{
    Fold, Folded, fold_all, fold_lines, log_lines, read_header, read_logs, repairs, retired,
    session_of,
};
use crate::store::now;
use crate::{Repair, SavedAgent, Session, SessionError, Stamp, Store};
use rig_core::now_ms;

/// One log as the writer keeps it.
#[derive(Default)]
pub(crate) struct Log {
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    depth: usize,
    next_seq: u64,
    /// Whether the log holds a change or a saved value, so it is written.
    started: bool,
    pending: Vec<u8>,
    /// The newest value of each saved value, so an unchanged one is not
    /// written again.
    state: BTreeMap<String, Value>,
    /// The log's lineage, from its header.
    lineage: Lineage,
    /// The changes it holds.
    changes: u64,
}

/// The records of every log, before and after they are written.
#[derive(Default)]
pub(crate) struct Book {
    /// The next record's `n`.
    next_n: u64,
    /// Records are queued only once started.
    live: bool,
    /// Why the logging stopped: every record after a failed write is
    /// dropped.
    failure: Option<String>,
    /// Whether a flush wrote a record.
    wrote: bool,
    logs: HashMap<String, Log>,
    /// The logs that end in a torn line, each with the bytes before it: the
    /// line is set aside before anything is written after it.
    torn: HashMap<String, u64>,
}

impl Book {
    /// A book that goes on after the logs `folded`, each as it ends.
    pub fn after(folded: &[Folded]) -> Self {
        let parents: HashMap<&str, Option<&str>> = folded
            .iter()
            .map(|log| (log.name.as_str(), log.header.parent.as_deref()))
            .collect();
        let mut book = Self::default();
        for log in folded {
            // Parents come from the logs' headers; the bound stops a corrupt store's loop.
            let ancestors = std::iter::successors(log.header.parent.as_deref(), |parent| {
                parents.get(parent).copied().flatten()
            });
            let known = ancestors.take_while(|parent| parents.contains_key(parent));
            let kept = Log {
                depth: known.take(folded.len()).count(),
                next_seq: log.next_seq,
                started: true,
                state: log.state.clone(),
                lineage: log.header.lineage,
                changes: log.conversation.revision(),
                ..Log::default()
            };
            book.logs.insert(log.name.clone(), kept);
            book.next_n = book.next_n.max(log.next_n);
        }
        book
    }

    /// Sets aside the torn line each log of `torn` ends in, given with the
    /// bytes before it, as [`set_aside_tail`] does, before that log is next
    /// written.
    pub fn set_aside(&mut self, torn: impl IntoIterator<Item = (String, u64)>) {
        self.torn.extend(torn);
    }

    /// Numbers the next record `n` or later.
    pub fn raise_n(&mut self, n: u64) {
        self.next_n = self.next_n.max(n);
    }

    pub fn start(&mut self) {
        self.live = true;
    }

    fn is_live(&self) -> bool {
        self.live && self.failure.is_none()
    }

    /// The log `name`, started first with a header naming `parent` when it
    /// has none; `None` while nothing is logged.
    pub fn opened(&mut self, name: &str, parent: Option<&str>) -> Option<&mut Log> {
        if !self.is_live() {
            return None;
        }
        if !self.logs.contains_key(name) {
            let depth = |parent| self.logs.get(parent).map_or(0, |log: &Log| log.depth + 1);
            let header = Header::new(name, parent);
            let mut log = Log {
                depth: parent.map_or(0, depth),
                lineage: header.lineage,
                ..Log::default()
            };
            let header = Meta::Header(header);
            let stamp = Stamp {
                n: self.next_n,
                seq: 0,
                t: now_ms(),
            };
            if let Err(failure) = write_meta(&mut log.pending, stamp, &header) {
                self.failure = Some(failure.to_string());
                return None;
            }
            log.next_seq += 1;
            self.next_n += 1;
            self.logs.insert(name.to_owned(), log);
        }
        self.logs.get_mut(name)
    }

    /// Queues `meta` for the log `name`; whether it was queued.
    fn meta(&mut self, name: &str, meta: &Meta) -> bool {
        if self.opened(name, None).is_none() {
            return false;
        }
        let n = self.next_n;
        let Some(log) = self.logs.get_mut(name) else {
            return false;
        };
        let stamp = Stamp {
            n,
            seq: log.next_seq,
            t: now_ms(),
        };
        match write_meta(&mut log.pending, stamp, meta) {
            Ok(()) => {
                log.next_seq += 1;
                self.next_n += 1;
                true
            }
            Err(failure) => {
                self.failure = Some(failure.to_string());
                false
            }
        }
    }

    /// Queues the unsaved changes of `conversation` for the log `name`, or
    /// one reset when it does not continue the log
    /// ([`Conversation::continues`]), empties its outbox and marks it as
    /// continuing the log; the changes queued, each stamped with when it
    /// was made. While nothing is logged the outbox is emptied all the
    /// same, and the next save writes a reset. Each image is stored as a
    /// blob through `put`.
    pub fn save(
        &mut self,
        put: &mut PutBlob<'_>,
        name: &str,
        conversation: &mut Conversation,
    ) -> Vec<ChangeRecord> {
        let Some(log) = self.opened(name, None) else {
            return stamp_all(conversation.take_unsaved());
        };
        let (lineage, held) = (log.lineage, log.changes);
        let continues = conversation.continues(lineage, held);
        let unsaved = conversation.take_unsaved();
        let changes = if continues {
            unsaved
        } else {
            vec![ChangeRecord {
                change: Change::Reset {
                    state: Box::new(conversation.view()),
                },
                by: None,
                at: None,
            }]
        };
        if changes.is_empty() {
            conversation.continue_log(lineage, held);
            return changes;
        }
        let mut queued = Vec::with_capacity(changes.len());
        for recorded in stamp_all(changes) {
            if let Err(failure) = self.queue(put, name, &recorded) {
                self.failure = Some(failure);
                break;
            }
            queued.push(recorded);
        }
        let held = held.saturating_add(queued.len() as u64);
        if let Some(log) = self.logs.get_mut(name) {
            log.started = true;
            log.changes = held;
        }
        conversation.continue_log(lineage, held);
        queued
    }

    /// Queues `recorded`, its images stored as blobs, for the log `name`.
    fn queue(
        &mut self,
        put: &mut PutBlob<'_>,
        name: &str,
        recorded: &ChangeRecord,
    ) -> Result<(), String> {
        let stored = store_change(&recorded.change, put)
            .map_err(|failure| format!("storing an image failed: {failure}"))?;
        let change = stored.as_ref().unwrap_or(&recorded.change);
        let n = self.next_n;
        let log = self.logs.get_mut(name).ok_or("no log")?;
        let stamp = Stamp {
            n,
            seq: log.next_seq,
            t: recorded.at.unwrap_or_default(),
        };
        write_change(&mut log.pending, stamp, recorded.by.as_ref(), change)
            .map_err(|failure| failure.to_string())?;
        log.next_seq += 1;
        self.next_n += 1;
        Ok(())
    }

    /// Queues the saved value `key` of the log `name` when it changed;
    /// `None` or `null` when it was removed. With `starts`, the change
    /// starts the log.
    pub fn save_value(&mut self, name: &str, key: &str, value: Option<Value>, starts: bool) {
        let value = value.filter(|value| !value.is_null());
        let known = self.logs.get(name).and_then(|log| log.state.get(key));
        if known == value.as_ref() {
            return;
        }
        let meta = Meta::State {
            key: key.to_owned(),
            value: value.clone(),
        };
        if self.meta(name, &meta)
            && let Some(log) = self.logs.get_mut(name)
        {
            log.started |= starts;
            match value {
                Some(value) => log.state.insert(key.to_owned(), value),
                None => log.state.remove(key),
            };
        }
    }

    /// Queues the saved values of the log `name` that differ from `known`,
    /// and the removal of each of `known` that `values` does not have.
    pub fn states(
        &mut self,
        name: &str,
        known: Option<&BTreeMap<String, Value>>,
        values: &BTreeMap<String, Value>,
    ) {
        let removed = known.into_iter().flat_map(BTreeMap::keys);
        let removed: Vec<String> = removed
            .filter(|key| !values.contains_key(*key))
            .cloned()
            .collect();
        for key in removed {
            self.save_value(name, &key, None, true);
        }
        for (key, value) in values {
            self.save_value(name, key, Some(value.clone()), true);
        }
    }

    /// Starts the log `to` in `store` as a copy of the log `from` up to a
    /// turn boundary at or before `through` ([`turn_cut`]), with a header
    /// of its own and a fresh lineage, and goes on after it; the copy as
    /// read. `from` must be written already. An agent with no log yet gets
    /// an empty one.
    pub fn fork(
        &mut self,
        store: &dyn Store,
        (from, to): (&str, &str),
        through: Option<MessageId>,
    ) -> Result<Folded, SessionError> {
        let refuse = |why: String| SessionError::from(io::Error::other(why));
        if !self.is_live() {
            return Err(refuse("the session is not logged".to_owned()));
        }
        if self.logs.contains_key(to) || log_lines(store, to)?.is_some() {
            return Err(refuse(format!("there is a log `{to}` already")));
        }
        let lines = log_lines(store, from)?.unwrap_or_else(|| LogLines::parse(Vec::new()));
        let mut bytes = Vec::new();
        if let Some((_, start, first)) = lines.iter().next() {
            let (stamp, _) = read_header(from, first)?;
            let cut = turn_cut(from, &lines, through)?;
            let stamp = Stamp {
                t: now_ms(),
                ..stamp
            };
            let header = Meta::Header(Header::new(to, None));
            write_meta(&mut bytes, stamp, &header).map_err(io::Error::from)?;
            let start = usize::try_from(start).unwrap_or(usize::MAX);
            bytes.extend_from_slice(lines.prefix(cut).get(start..).unwrap_or_default());
        } else if through.is_some() {
            return Err(refuse(format!("the log `{from}` holds no message")));
        }
        if bytes.is_empty() {
            let _ = self
                .opened(to, None)
                .ok_or_else(|| refuse("logging stopped".to_owned()))?;
            let folded = Folded {
                name: to.to_owned(),
                ..Folded::default()
            };
            return Ok(folded);
        }
        let copy = LogLines::parse(bytes);
        let read = |blob: &str| read_blob(store, blob);
        let folded = fold_lines(to, &copy, Fold::WHOLE, &read)?
            .ok_or_else(|| refuse(format!("the copy of the log `{from}` is empty")))?;
        store.append_log(to, copy.prefix(u64::MAX))?;
        let log = Log {
            next_seq: folded.next_seq,
            started: true,
            state: folded.state.clone(),
            lineage: folded.header.lineage,
            changes: folded.conversation.revision(),
            ..Log::default()
        };
        self.logs.insert(to.to_owned(), log);
        Ok(folded)
    }

    pub fn retire(&mut self, name: &str) {
        self.meta(name, &Meta::Retired);
    }

    /// Writes every queued record to `store`: each started log's records
    /// with one write, a subagent's log before its parent's, and the
    /// [`SESSION`] log once any is started. A failed write stops the
    /// logging.
    pub fn flush(&mut self, store: &dyn Store) -> Result<(), SessionError> {
        now(self.flush_async(store))
    }

    /// [`Self::flush`] to an [`AsyncStore`](crate::AsyncStore).
    pub async fn flush_async<S: crate::AsyncStore + ?Sized>(
        &mut self,
        store: &S,
    ) -> Result<(), SessionError> {
        if let Some(failure) = &self.failure {
            return Err(io::Error::other(failure.clone()).into());
        }
        let any = self.logs.values().any(|log| log.started);
        let mut due: Vec<(usize, String)> = self
            .logs
            .iter()
            .filter(|(name, log)| log.started || any && name.as_str() == SESSION)
            .filter(|(_, log)| !log.pending.is_empty())
            .map(|(name, log)| (log.depth, name.clone()))
            .collect();
        due.sort_by(|a, b| b.cmp(a));
        for (_, name) in due {
            let Some(log) = self.logs.get_mut(&name) else {
                continue;
            };
            let set_aside = match self.torn.remove(&name) {
                Some(kept) => set_aside_tail_async(store, &name, kept).await.map(drop),
                None => Ok(()),
            };
            let written = match set_aside {
                Ok(()) => store
                    .append_log(&name, &log.pending)
                    .await
                    .map_err(Into::into),
                Err(failure) => Err(failure),
            };
            if let Err(failure) = written {
                let failure = format!("writing the log {name} failed: {failure}");
                self.failure = Some(failure.clone());
                return Err(io::Error::other(failure).into());
            }
            log.pending.clear();
            self.wrote = true;
        }
        Ok(())
    }
}

/// A running session's logs, the harness's appender. Every method takes
/// `&self`, so clones share them. Nothing is logged before
/// [`start`](Self::start), nor once a write failed; a default writer has no
/// store and logs nothing.
#[derive(Clone, Default)]
pub struct Writer(Arc<Mutex<Inner>>);

#[derive(Default)]
struct Inner {
    store: Option<Arc<dyn Store>>,
    book: Book,
}

impl Writer {
    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Opens the session in `store` to go on with it, and returns the
    /// session as read (retired agents and their subagents left out).
    ///
    /// Reads like [`Session::read`], so a broken session, or one of
    /// another format, is refused before anything is written. Then each torn tail
    /// is moved to `recovered/<log>-<k>.part` ([`Repair::TornTail`]).
    /// Nothing is logged until [`start`](Self::start).
    ///
    /// # Errors
    ///
    /// As [`Session::read`], or when setting a torn tail aside fails.
    pub fn open(store: Arc<dyn Store>) -> Result<(Self, Session), SessionError> {
        let mut folded = fold_all(&*store, &read_logs(&*store)?)?;
        let mut repairs = repairs(&*store, &folded)?;
        for repair in &mut repairs {
            if let Repair::TornTail {
                log,
                kept,
                set_aside,
                ..
            } = repair
            {
                *set_aside = Some(set_aside_tail(&*store, log, *kept)?);
            }
        }
        let book = Book::after(&folded);
        let skipped = retired(&folded);
        folded.retain(|log| !skipped.contains(&log.name));
        let writer = Self(Arc::new(Mutex::new(Inner {
            store: Some(store),
            book,
        })));
        Ok((writer, session_of(folded, repairs)))
    }

    /// Starts logging, once the host restored the session: what it did to
    /// restore is not logged.
    pub fn start(&self) {
        let mut inner = self.inner();
        if inner.store.is_some() {
            inner.book.start();
        }
    }

    /// Whether records are logged: the writer was started, and no write
    /// failed.
    pub fn is_live(&self) -> bool {
        self.inner().book.is_live()
    }

    /// Why the logging stopped, if it did.
    pub fn failure(&self) -> Option<String> {
        self.inner().book.failure.clone()
    }

    /// Whether a [`flush`](Self::flush) wrote a record to the store, such
    /// as for a host that marks its session saved once something is.
    pub fn has_written(&self) -> bool {
        self.inner().book.wrote
    }

    /// Starts the log `log`, of an agent spawned by the agent `parent`,
    /// unless it has one.
    pub fn open_log(&self, log: &str, parent: Option<&str>) {
        let _ = self.inner().book.opened(log, parent);
    }

    /// Queues the unsaved changes of `conversation` for the log `log` and
    /// empties its outbox: the changes, each stamped with when it was made
    /// ([`ChangeRecord::at`]). When the conversation does not continue the log
    /// ([`Conversation::continues`]), as when a value made elsewhere
    /// replaced it, one [`Change::Reset`] to the whole value is queued
    /// instead. Either way the conversation then continues the log.
    pub fn save(&self, log: &str, conversation: &mut Conversation) -> Vec<ChangeRecord> {
        let mut inner = self.inner();
        let Inner { store, book } = &mut *inner;
        match store {
            Some(store) => {
                let put = &mut |name: String, bytes: Vec<u8>| put_blob(&**store, &name, &bytes);
                book.save(put, log, conversation)
            }
            None => stamp_all(conversation.take_unsaved()),
        }
    }

    /// Queues the saved value `key` of the log `log`, an agent's id or
    /// [`SESSION`], when it changed; `None` or a `null` value when it was
    /// removed. With `starts`, the change starts the log.
    pub fn save_value(&self, log: &str, key: &str, value: Option<Value>, starts: bool) {
        self.inner().book.save_value(log, key, value, starts);
    }

    /// Starts the log `to` as a copy of the log `from` up to a turn
    /// boundary at or before message `through` (its end with `None`): through
    /// the last record after which no turn runs, no tool call waits for its
    /// result, the model owes no answer and nothing after `through` is in the
    /// conversation. What is queued is written first. The copy keeps the
    /// records' stamps, so [`as_of`](crate::as_of) and
    /// [`undo_summary`](crate::undo_summary) read it as they read `from`,
    /// and has its own header and lineage. Returns the agent of the copy:
    /// its conversation continues the new log, and its saved values are
    /// those `from` had at the cut.
    ///
    /// # Errors
    ///
    /// When nothing is logged, the log `to` exists, `from` does not read or
    /// never held `through`, or a write fails.
    pub fn fork_log(
        &self,
        from: &str,
        to: &str,
        through: Option<MessageId>,
    ) -> Result<SavedAgent, SessionError> {
        let mut inner = self.inner();
        let Inner { store, book } = &mut *inner;
        let Some(store) = store else {
            return Err(io::Error::other("the session is not stored").into());
        };
        book.flush(&**store)?;
        let folded = book.fork(&**store, (from, to), through)?;
        Ok(SavedAgent {
            id: folded.name,
            started_ms: folded.started_ms,
            conversation: folded.conversation,
            saved: folded.state,
            ..SavedAgent::default()
        })
    }

    /// Queues that the agent of the log `log` is gone for good.
    pub fn retire(&self, log: &str) {
        self.inner().book.retire(log);
    }

    /// Writes every queued record: each started log's records with one
    /// write, a subagent's log before its parent's, so a parent never
    /// refers to a subagent message that is not stored, and the [`SESSION`]
    /// log once any is started. No fsync: written is not yet proof against
    /// a power cut. A failed write stops the logging ([`Self::failure`]).
    ///
    /// # Errors
    ///
    /// When the logging stopped, now or before: a write failed, so what
    /// was queued is not stored and nothing more will be. A writer without
    /// a store keeps nothing and never fails.
    pub fn flush(&self) -> Result<(), SessionError> {
        let mut inner = self.inner();
        let Inner { store, book } = &mut *inner;
        match store {
            Some(store) => book.flush(&**store),
            None => Ok(()),
        }
    }
}

/// Copies the bytes of the log `log` after its first `kept` to the free
/// file `recovered/<log>-<k>.part`, then replaces the log with the bytes
/// before them. Returns the name of the file the tail went to. A crash
/// between the two leaves the tail in the log, set aside again by the next
/// open.
pub(crate) fn set_aside_tail(
    store: &dyn Store,
    log: &str,
    kept: u64,
) -> Result<String, SessionError> {
    now(set_aside_tail_async(store, log, kept))
}

/// [`set_aside_tail`] in an [`AsyncStore`](crate::AsyncStore).
async fn set_aside_tail_async<S: crate::AsyncStore + ?Sized>(
    store: &S,
    log: &str,
    kept: u64,
) -> Result<String, SessionError> {
    let bytes = store.read_log(log).await?;
    let kept = usize::try_from(kept).unwrap_or(usize::MAX);
    let (prefix, tail) = bytes.split_at_checked(kept).unwrap_or((&bytes, &[]));
    let files = store.files().await?;
    let name = (1..=files.len() + 1)
        .map(|k| format!("recovered/{log}-{k}.part"))
        .find(|name| !files.contains(name))
        .unwrap_or_default();
    store.write_file(&name, tail).await?;
    store.replace_log(log, prefix).await?;
    Ok(name)
}

/// `changes`, each stamped with now unless it says when it was made.
fn stamp_all(changes: Vec<ChangeRecord>) -> Vec<ChangeRecord> {
    let now = now_ms();
    let stamp = |recorded: ChangeRecord| ChangeRecord {
        at: Some(recorded.at.unwrap_or(now)),
        ..recorded
    };
    changes.into_iter().map(stamp).collect()
}
