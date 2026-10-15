//! The project's provenance records: which tool call wrote each line of
//! each file an agent changed, carried forward through every later change.
//! They are kept as a log that only grows, one [`Logged`] line per call
//! or forget, so sessions working in one project at once never lose each
//! other's, and folded into a [`ProvenanceStore`] when asked.

use std::collections::BTreeMap;

use rig_harness::rig_core::conversation::MessageKey;
use rig_harness::rig_core::tool::{ChangeKind, FileChange, Hunk, HunkLine};
use serde::{Deserialize, Serialize};
use similar::{DiffTag, TextDiff};

/// One tool call that changed files: who made it, where and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The session's id.
    pub session: String,
    /// The agent's name.
    pub agent: String,
    /// The agent's stable id: the name of its log in the session.
    pub agent_id: String,
    /// The turn, counted from 1 over the session.
    pub turn: u64,
    /// The message that asked for the turn, in the agent's conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asked: Option<MessageKey>,
    /// What the user typed to start the turn (the root agent's turn, for
    /// a subagent's call).
    pub prompt: String,
    /// The tool.
    pub tool: String,
    /// The model's id for the call.
    pub call_id: String,
    /// When the call settled, in milliseconds since the Unix epoch.
    pub at_ms: u64,
}

/// Where a line came from: the record of the call that wrote it, and
/// whether the user changed it after.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineOrigin {
    /// The [`Record`]'s key.
    pub record: u64,
    /// Whether the user changed the line after the agent wrote it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub changed_by_you: bool,
}

/// A file with at least one line an agent wrote: its text as last seen,
/// and each line's origin (`None`: not written by an agent).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackedFile {
    /// The file's text when it was last seen.
    pub text: String,
    /// Each line's origin, by line.
    pub lines: Vec<Option<LineOrigin>>,
}

/// A run of lines with one origin, numbered from 1, both ends included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The first line.
    pub from: usize,
    /// The last line.
    pub to: usize,
    /// Where they came from.
    pub origin: Option<LineOrigin>,
}

/// One line of a project's provenance log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Logged {
    /// A tool call changed these files, each by its path from the project
    /// root.
    Changed {
        /// Who made the call, where and why.
        record: Record,
        /// What it changed, with each file's text before and its hunks.
        changes: Vec<(String, FileChange)>,
    },
    /// The user dropped the records of the session this identifies.
    Forgot(String),
}

/// Every provenance record of a project, folded from its log: a record's
/// key is its call's place in the log, so it never changes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceStore {
    /// The tool calls, by key.
    pub records: BTreeMap<u64, Record>,
    /// The files agents wrote lines of, by path from the project root.
    pub files: BTreeMap<String, TrackedFile>,
}

impl ProvenanceStore {
    /// The store a log's lines make, in order. A file a change does not
    /// say the text of (over the tool's read limit, or not UTF-8) is not
    /// tracked from then on; a line that does not read is skipped.
    pub fn fold(log: &[u8]) -> Self {
        let mut store = Self::default();
        let logged = log
            .split(|byte| *byte == b'\n')
            .filter_map(|line| serde_json::from_slice(line).ok());
        for (key, logged) in (0..).zip(logged) {
            match logged {
                Logged::Changed { record, changes } => {
                    for (path, change) in &changes {
                        store.agent_changed(path, change, key);
                    }
                    store.records.insert(key, record);
                    store.prune();
                }
                Logged::Forgot(session) => {
                    store.forget(&session);
                }
            }
        }
        store
    }

    /// The file at `path` is now `now` (`None`: gone), changed by the user:
    /// a line of an agent's they changed keeps its origin, marked; a line
    /// they added is no agent's.
    pub fn user_changed(&mut self, path: &str, now: Option<&str>) {
        let Some(file) = self.files.get_mut(path) else {
            return;
        };
        let Some(now) = now else {
            self.files.remove(path);
            return;
        };
        if file.text == now {
            return;
        }
        file.lines = carry(&file.text, &file.lines, now, &|was| {
            was.map(|origin| LineOrigin {
                changed_by_you: true,
                ..origin
            })
        });
        now.clone_into(&mut file.text);
        self.prune();
    }

    /// The call of `record` made `change` to the file at `path`: the lines
    /// it added are its, the others keep their origin. A tracked file that
    /// was not what the change found had been changed by the user first.
    /// When the change does not say the text before and after, the file's
    /// lines are not known any more and it is not tracked.
    fn agent_changed(&mut self, path: &str, change: &FileChange, record: u64) {
        let before = match change.kind {
            ChangeKind::Deleted => {
                self.files.remove(path);
                return;
            }
            ChangeKind::Created => Some(""),
            ChangeKind::Modified => change.before_text.as_deref(),
        };
        let (Some(before), Some(after)) = (before, change.after_text()) else {
            self.files.remove(path);
            return;
        };
        self.user_changed(path, Some(before));
        let file = self
            .files
            .entry(path.to_owned())
            .or_insert_with(|| TrackedFile {
                text: before.to_owned(),
                lines: vec![None; before.lines().count()],
            });
        let written = Some(LineOrigin {
            record,
            changed_by_you: false,
        });
        file.lines = through(&file.lines, &change.hunks, written);
        file.text = after;
    }

    /// Drops the records of the session `session` identifies (its id or
    /// the start of it): their lines then report no agent. How many
    /// records went.
    pub fn forget(&mut self, session: &str) -> usize {
        let gone: Vec<u64> = self
            .records
            .iter()
            .filter(|(_, record)| {
                record.session == session
                    || (session.len() >= 4 && record.session.starts_with(session))
            })
            .map(|(key, _)| *key)
            .collect();
        for key in &gone {
            self.records.remove(key);
        }
        for file in self.files.values_mut() {
            for line in &mut file.lines {
                if line.is_some_and(|origin| gone.contains(&origin.record)) {
                    *line = None;
                }
            }
        }
        self.prune();
        gone.len()
    }

    /// The lines `from` to `to` (each clamped; the whole file when `None`)
    /// of the file at `path`, which has `total` lines, in runs of one
    /// origin.
    pub fn spans(&self, path: &str, total: usize, range: Option<(usize, usize)>) -> Vec<Span> {
        let (from, to) = range.unwrap_or((1, total.max(1)));
        let (from, to) = (from.max(1), to.max(from.max(1)));
        let lines = self.files.get(path).map(|file| file.lines.as_slice());
        let origin = |line: usize| {
            lines
                .and_then(|lines| lines.get(line.checked_sub(1)?))
                .copied()
                .flatten()
        };
        let mut spans: Vec<Span> = Vec::new();
        for line in from..=to {
            let origin = origin(line);
            match spans.last_mut() {
                Some(span) if span.origin == origin => span.to = line,
                _ => spans.push(Span {
                    from: line,
                    to: line,
                    origin,
                }),
            }
        }
        spans
    }

    /// Stops tracking files no agent's line is left in, and drops the
    /// records no line is left of.
    fn prune(&mut self) {
        self.files
            .retain(|_, file| file.lines.iter().any(Option::is_some));
        let kept: std::collections::BTreeSet<u64> = self
            .files
            .values()
            .flat_map(|file| file.lines.iter().flatten())
            .map(|origin| origin.record)
            .collect();
        self.records.retain(|key, _| kept.contains(key));
    }
}

/// The origins of the lines after a change, given those before it and
/// its hunks: a kept line keeps its origin, an added one is `written`.
fn through(
    origins: &[Option<LineOrigin>],
    hunks: &[Hunk],
    written: Option<LineOrigin>,
) -> Vec<Option<LineOrigin>> {
    let at = |line: usize| origins.get(line).copied().flatten();
    let mut carried = Vec::with_capacity(origins.len());
    let mut old = 0;
    for hunk in hunks {
        let start = hunk.old.start.saturating_sub(1).max(old);
        carried.extend((old..start).map(at));
        old = start;
        for line in &hunk.lines {
            match line {
                HunkLine::Same(_) => {
                    carried.push(at(old));
                    old += 1;
                }
                HunkLine::Removed(_) => old += 1,
                HunkLine::Added(_) => carried.push(written),
            }
        }
    }
    carried.extend((old..origins.len().max(old)).map(at));
    carried
}

/// The origins of `new`'s lines, given `old`'s: an unchanged line keeps
/// its origin, a changed or added one gets `changed` of the origin of the
/// line it replaced, if any.
fn carry(
    old: &str,
    origins: &[Option<LineOrigin>],
    new: &str,
    changed: &dyn Fn(Option<LineOrigin>) -> Option<LineOrigin>,
) -> Vec<Option<LineOrigin>> {
    let at = |line: usize| origins.get(line).copied().flatten();
    let mut carried = Vec::new();
    for op in TextDiff::from_lines(old, new).ops() {
        let (was, now) = (op.old_range(), op.new_range());
        match op.tag() {
            DiffTag::Equal => carried.extend(was.map(at)),
            DiffTag::Delete => {}
            DiffTag::Insert | DiffTag::Replace => carried.extend(now.enumerate().map(|(k, _)| {
                let replaced = (k < was.len()).then(|| at(was.start + k)).flatten();
                changed(replaced)
            })),
        }
    }
    carried
}
