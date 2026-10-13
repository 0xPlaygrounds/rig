//! The project's provenance records: which tool call wrote each line of
//! each file an agent changed, carried forward through every later change.

use std::collections::BTreeMap;

use rig_harness::prelude::MessageId;
use serde::{Deserialize, Serialize};
use similar::{DiffTag, TextDiff};

/// One tool call that changed files: who made it, where and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The session's id.
    pub session: String,
    /// The session's name, set with `/name`, when it had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    /// The agent's name.
    pub agent: String,
    /// The agent's stable id: the name of its log in the session.
    pub agent_id: String,
    /// The turn, counted from 1 over the session.
    pub turn: u64,
    /// Where the turn begins in the agent's conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_message: Option<MessageId>,
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

impl Record {
    /// The session as it is shown: its name, else the start of its id.
    pub fn session_label(&self) -> &str {
        self.session_name
            .as_deref()
            .unwrap_or_else(|| self.session.get(..8).unwrap_or(&self.session))
    }
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

/// Every provenance record of a project, kept in its `PluginData`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceStore {
    /// The tool calls, by key.
    pub records: BTreeMap<u64, Record>,
    /// The files agents wrote lines of, by path from the project root.
    pub files: BTreeMap<String, TrackedFile>,
}

impl ProvenanceStore {
    /// Keeps `record`, returning its key.
    pub fn add(&mut self, record: Record) -> u64 {
        let key = self
            .records
            .last_key_value()
            .map_or(0, |(key, _)| key.saturating_add(1));
        self.records.insert(key, record);
        key
    }

    /// The file at `path` is now `now` (`None`: gone), changed by the user:
    /// a line of an agent's they changed keeps its origin, marked; a line
    /// they added is no agent's. Whether anything changed.
    pub fn user_changed(&mut self, path: &str, now: Option<&str>) -> bool {
        let Some(file) = self.files.get_mut(path) else {
            return false;
        };
        let Some(now) = now else {
            self.files.remove(path);
            return true;
        };
        if file.text == now {
            return false;
        }
        file.lines = carry(&file.text, &file.lines, now, &|was| {
            was.map(|origin| LineOrigin {
                changed_by_you: true,
                ..origin
            })
        });
        now.clone_into(&mut file.text);
        self.prune();
        true
    }

    /// The call of `record` turned the file at `path` from `before` into
    /// `after` (`None`: deleted): what differs from `before` is its. A
    /// tracked file that is not `before` any more was changed by the user
    /// first.
    pub fn agent_changed(&mut self, path: &str, before: &str, after: Option<&str>, record: u64) {
        self.user_changed(path, Some(before));
        let Some(after) = after else {
            self.files.remove(path);
            return;
        };
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
        file.lines = carry(&file.text, &file.lines, after, &|_| written);
        after.clone_into(&mut file.text);
        self.prune();
    }

    /// Drops the records of the session named or identified by `session`
    /// (its id or the start of it, or its name): their lines then report
    /// no agent. How many records went.
    pub fn forget(&mut self, session: &str) -> usize {
        let gone: Vec<u64> = self
            .records
            .iter()
            .filter(|(_, record)| {
                record.session == session
                    || (session.len() >= 4 && record.session.starts_with(session))
                    || record.session_name.as_deref() == Some(session)
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

    /// Stops tracking files no agent's line is left in.
    fn prune(&mut self) {
        self.files
            .retain(|_, file| file.lines.iter().any(Option::is_some));
    }
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
