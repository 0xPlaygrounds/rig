//! Branches inside a conversation: other continuations of it from one of
//! its entries, kept as asides, so they are saved, forked, rewritten and
//! exported with it and never reach a model.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::pairing::pending_calls;
use super::{
    Aside, AsideId, Author, Change, Conversation, Entry, Halt, MessageId, Meta, Spliced, State,
};

/// The kind of an aside holding an inactive [`Branch`].
pub const BRANCH: &str = "rig.branch";

/// The kind of the aside holding the active continuation's title (a
/// string), which only rig-core writes. It follows no entry, so it stays
/// with the active line.
pub const ACTIVE_TITLE: &str = "rig.branch.active";

/// The active continuation's title until a branch names it.
pub const MAIN: &str = "main";

/// The kind of the aside holding the active line's own values (an object,
/// `{key: value}`), which only rig-core writes: what was set on the line
/// while it had no entry of its own. It follows no entry, so it stays with
/// the active line, and a branch carries it as [`Branch::values`].
pub const LINE_VALUES: &str = "rig.line";

/// An inactive branch, as [`Conversation::branches`] lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct KeptBranch {
    /// Its aside: what [`Conversation::switch_branch`] and
    /// [`Conversation::line`] take.
    pub id: AsideId,
    /// The branch it is kept in; `None` for one kept in the active line.
    pub inside: Option<AsideId>,
    /// How many entries it shares with the line it is kept in: where its
    /// own entries start.
    pub at: usize,
    /// The branch.
    pub branch: Branch,
}

/// A continuation of a conversation that is not the active one: the
/// entries that followed its point, with their ids, authors and made-times,
/// the saved asides that followed them (branches among them), and the
/// halt it ended with. Kept as an aside of kind [`BRANCH`] following its
/// point.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    /// What it is called.
    pub title: String,
    /// Its entries, oldest first.
    pub entries: Vec<Entry>,
    /// The saved asides that followed its entries, with their ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub asides: Vec<(AsideId, Aside)>,
    /// Why its last user message was left unanswered, if it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub halted: Option<Halt>,
    /// Whether its entries are a range moved out of the line it is kept in,
    /// rather than another continuation: they belong back where its aside is.
    #[serde(default, skip_serializing_if = "is_false")]
    pub moved: bool,
    /// The line values of the continuation it keeps
    /// ([`Conversation::line_values`]).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub values: Map<String, Value>,
}

fn is_false(moved: &bool) -> bool {
    !*moved
}

impl Branch {
    /// The branch `aside` holds, when it is one: of kind [`BRANCH`], its
    /// value a branch.
    pub fn of(aside: &Aside) -> Option<Self> {
        (aside.kind == BRANCH)
            .then(|| Self::deserialize(&aside.value).ok())
            .flatten()
    }
}

/// Why a branch could not be made or switched to.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BranchError {
    /// There is no entry there to branch off from.
    #[error("there is no entry {0} to branch off from")]
    NoEntry(usize),
    /// Branching there would part a tool call from its result.
    #[error("a branch there would part a tool call from its result")]
    PartsCall,
    /// The aside is not a branch.
    #[error("aside {} is not a branch", .0.0)]
    NotABranch(AsideId),
    /// A branch's value could not be written.
    #[error("the branch could not be kept: {0}")]
    Value(String),
    /// There is no such aside in the active line.
    #[error("there is no aside {}", .0.0)]
    NoAside(AsideId),
}

/// Where [`Conversation::merge_line`] puts a line's own entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeAt {
    /// At the end of the active line, each with a new id: they are not
    /// this conversation's.
    End,
    /// Back where the aside `pointer` of the active line is, such as the
    /// pointer a range moved into another session left: its first `moved`
    /// entries keep their ids, which this conversation made, the others
    /// get new ones.
    Back {
        /// The aside the entries go back to.
        pointer: AsideId,
        /// How many of the line's own entries this conversation made.
        moved: usize,
    },
}

/// How many first entries two lines share, compared by their
/// [`MessageKey`](super::MessageKey) (id and made-time): where they part.
/// An entry without an id is shared with none.
pub fn shared_prefix(a: &[Entry], b: &[Entry]) -> usize {
    let same = |(a, b): &(&Entry, &Entry)| a.id.is_some() && a.id == b.id && a.made == b.made;
    a.iter().zip(b).take_while(same).count()
}

impl Conversation {
    /// The active continuation's title: [`MAIN`] until
    /// [`branch_off`](Self::branch_off) or
    /// [`switch_branch`](Self::switch_branch) names it.
    pub fn line_title(&self) -> &str {
        let mut asides = self.asides.iter();
        let title = asides.find(|(_, aside)| aside.kind == ACTIVE_TITLE);
        title
            .and_then(|(_, aside)| aside.value.as_str())
            .unwrap_or(MAIN)
    }

    /// Every inactive branch, those kept inside other branches too: each
    /// comes after the one it is kept in, in the order put.
    pub fn branches(&self) -> Vec<KeptBranch> {
        let active: Vec<MessageId> = self.meta.iter().map(|meta| meta.id).collect();
        let mut found = Vec::new();
        for (id, aside) in &self.asides {
            if let Some(branch) = Branch::of(aside) {
                let at = self.aside_position(aside);
                keep(&mut found, (*id, None, at), branch, &active);
            }
        }
        found
    }

    /// The entries of the line branch `id` continues, from the first: the
    /// entries it shares with the line it is kept in, then its own. With
    /// `None`, the active line. `None` when `id` is no branch.
    pub fn line(&self, id: Option<AsideId>) -> Option<Vec<Entry>> {
        let mut line: Vec<Entry> = (0..self.messages.len())
            .filter_map(|at| self.entry(at))
            .collect();
        let Some(id) = id else {
            return Some(line);
        };
        let kept = self.branches();
        for kept in path(&kept, id)? {
            line.truncate(kept.at);
            line.extend(kept.branch.entries.iter().cloned());
        }
        Some(line)
    }

    /// Keeps entries `at..` (with their asides, halt and line values) as a
    /// branch under the current title ([`line_title`](Self::line_title)),
    /// and goes on from entry `at - 1` as a new line titled `title`.
    /// Returns the aside that keeps the old line.
    ///
    /// With `at` equal to the length, the new line starts with a copy of
    /// the line values; otherwise it starts with none. Each change is made
    /// by `by`.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NoEntry`] when `at` is past the
    /// end, [`BranchError::PartsCall`] when an entry before `at` has a call
    /// answered at or after it.
    pub fn branch_off(
        &mut self,
        at: usize,
        title: impl Into<String>,
        by: Author,
    ) -> Result<AsideId, BranchError> {
        let len = self.messages.len();
        if at > len {
            return Err(BranchError::NoEntry(at));
        }
        let before = self.messages.get(..at).unwrap_or_default();
        if at < len && !super::pairing::pending_calls(before).is_empty() {
            return Err(BranchError::PartsCall);
        }
        let point = at.checked_sub(1).and_then(|at| self.id(at));
        let left = self.line_title().to_owned();
        let values = self.line_values().cloned().unwrap_or_default();
        let branch = self.stash(at, left, &by)?;
        let id = self.add_aside(Aside::new(BRANCH, by.clone(), branch).after(point));
        self.title_branch(title.into(), &by);
        if at == len {
            self.put_line_values(values, &by);
        }
        Ok(id)
    }

    /// Makes the kept branch `id` the active continuation, wherever it is
    /// kept (through each branch it is kept in): each continuation it
    /// leaves is kept under its title with its saved asides and halt, even
    /// with no entries of its own, and the branch's entries come back with
    /// their ids and made-times ([`restore`](Self::restore)), its asides
    /// with theirs, its halt, its title and its line values. Each change is
    /// made by `by`. Returns what changed in the active line.
    ///
    /// ```
    /// use rig_core::completion::Message;
    /// use rig_core::conversation::{Author, Conversation};
    ///
    /// let mut conversation = Conversation::from_messages([
    ///     Message::user("Name a colour."),
    ///     Message::assistant("Red."),
    /// ]);
    /// let red = conversation.id(1);
    /// let by = Author::plugin("tree");
    /// let first = conversation.branch_off(1, "blue", by.clone())?;
    /// conversation.push(Message::assistant("Blue."));
    /// assert_eq!(conversation.line_title(), "blue");
    /// conversation.switch_branch(first, by)?;
    /// assert_eq!(conversation.messages().get(1), Some(&Message::assistant("Red.")));
    /// assert_eq!(conversation.id(1), red);
    /// assert_eq!(conversation.line_title(), "main");
    /// let kept = conversation.branches();
    /// assert_eq!(kept.iter().map(|kept| kept.branch.title.as_str()).collect::<Vec<_>>(), ["blue"]);
    /// # Ok::<(), rig_core::conversation::BranchError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NotABranch`] when aside `id` is not
    /// a branch.
    pub fn switch_branch(&mut self, id: AsideId, by: Author) -> Result<Spliced, BranchError> {
        let kept = self.branches();
        let path: Vec<AsideId> = path(&kept, id)
            .ok_or(BranchError::NotABranch(id))?
            .iter()
            .map(|kept| kept.id)
            .collect();
        let before: Vec<MessageId> = self.meta.iter().map(|meta| meta.id).collect();
        let (mut changed, mut restored) = (Vec::new(), Vec::new());
        for id in path {
            let spliced = self.switch_one(id, &by)?;
            changed.extend(spliced.changed);
            restored.extend(spliced.restored);
        }
        let now: HashSet<MessageId> = self.meta.iter().map(|meta| meta.id).collect();
        let still = |ids: Vec<MessageId>| -> Vec<MessageId> {
            let mut seen = HashSet::new();
            let ids = ids.into_iter().filter(|id| now.contains(id));
            ids.filter(|id| seen.insert(*id)).collect()
        };
        Ok(Spliced {
            removed: before.into_iter().filter(|id| !now.contains(id)).collect(),
            changed: still(changed),
            restored: still(restored),
        })
    }

    /// Makes branch `id`, kept in the active line, the active continuation.
    fn switch_one(&mut self, id: AsideId, by: &Author) -> Result<Spliced, BranchError> {
        let aside = self.aside(id).ok_or(BranchError::NotABranch(id))?;
        let branch = Branch::of(aside).ok_or(BranchError::NotABranch(id))?;
        let (point, at) = (aside.after, self.aside_position(aside));
        let left = self.line_title().to_owned();
        self.forget_aside(id, by);
        let kept = self.stash(at, left, by)?;
        let spliced = self.restore(at.., branch.entries, by.clone());
        for (id, aside) in branch.asides {
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, Some(by.clone()));
        }
        if self.halted != branch.halted {
            let halt = branch.halted;
            self.record(Change::Halt { halt }, Some(by.clone()));
        }
        self.add_aside(Aside::new(BRANCH, by.clone(), kept).after(point));
        self.title_branch(branch.title, by);
        self.put_line_values(branch.values, by);
        Ok(spliced)
    }

    /// The values kept with the active line itself rather than after one of
    /// its entries, by key: what was set on it while it had no entry of its
    /// own since it parted. `None` when it has none, as most lines.
    pub fn line_values(&self) -> Option<&Map<String, Value>> {
        let mut asides = self.asides.iter();
        let line = asides.find(|(_, aside)| aside.kind == LINE_VALUES);
        line.and_then(|(_, aside)| aside.value.as_object())
    }

    /// Sets (`Some`) or removes (`None`) the active line's value `key`: one
    /// aside of kind [`LINE_VALUES`] before the first entry, replaced in place, and
    /// none when the line has no values. Made by `by`.
    pub fn set_line_value(&mut self, key: &str, value: Option<Value>, by: Author) {
        let mut values = self.line_values().cloned().unwrap_or_default();
        match value {
            Some(value) => values.insert(key.to_owned(), value),
            None => values.shift_remove(key),
        };
        self.put_line_values(values, &by);
    }

    /// Whether a kept branch parts at the active line's end: an aside after
    /// its last entry would be shared with that branch.
    pub fn parts_at_end(&self) -> bool {
        let end = self.messages.len();
        let mut asides = self.asides.iter();
        asides.any(|(_, aside)| aside.kind == BRANCH && self.aside_position(aside) == end)
    }

    /// Makes `values` the active line's values: one [`LINE_VALUES`] aside, replaced
    /// in place, and none when empty.
    fn put_line_values(&mut self, values: Map<String, Value>, by: &Author) {
        if self
            .line_values()
            .map_or(values.is_empty(), |now| *now == values)
        {
            return;
        }
        let known = self.asides().find(|(_, aside)| aside.kind == LINE_VALUES);
        let known = known.map(|(id, _)| id);
        let empty = values.is_empty();
        let aside = Aside::new(LINE_VALUES, by.clone(), Value::Object(values));
        match known {
            Some(id) if empty => self.forget_aside(id, by),
            Some(id) => drop(self.replace_aside(id, aside)),
            None => drop(self.add_aside(aside)),
        }
    }

    /// Titles the active continuation `title`: one [`ACTIVE_TITLE`] aside,
    /// replaced in place, and none for [`MAIN`].
    fn title_branch(&mut self, title: String, by: &Author) {
        if self.line_title() == title {
            return;
        }
        let known = self.asides().find(|(_, aside)| aside.kind == ACTIVE_TITLE);
        let known = known.map(|(id, _)| id);
        let aside = Aside::new(ACTIVE_TITLE, by.clone(), serde_json::Value::String(title));
        match known {
            Some(id) if aside.value == MAIN => self.forget_aside(id, by),
            Some(id) => drop(self.replace_aside(id, aside)),
            None if aside.value == MAIN => {}
            None => drop(self.add_aside(aside)),
        }
    }

    /// Takes entries `at..` and the saved asides that follow them out, as
    /// a branch titled `title` keeping the line values, which the active
    /// line loses. Returns the branch.
    fn stash(
        &mut self,
        at: usize,
        title: String,
        by: &Author,
    ) -> Result<serde_json::Value, BranchError> {
        let (entries, asides) = self.kept(at..self.messages.len());
        let branch = Branch {
            title,
            entries,
            asides,
            halted: self.halted,
            moved: false,
            values: self.line_values().cloned().unwrap_or_default(),
        };
        let value = serde_json::to_value(&branch)
            .map_err(|failure| BranchError::Value(failure.to_string()))?;
        self.take_kept(at..self.messages.len(), &branch.asides, by);
        self.put_line_values(Map::new(), by);
        Ok(value)
    }

    /// Entries `range` and the saved asides that follow them.
    fn kept(&self, range: Range<usize>) -> (Vec<Entry>, Vec<(AsideId, Aside)>) {
        let entries: Vec<Entry> = range.filter_map(|at| self.entry(at)).collect();
        let ids: HashSet<MessageId> = entries.iter().filter_map(|entry| entry.id).collect();
        let asides = self
            .asides()
            .filter(|(_, aside)| aside.saved && aside.after.is_some_and(|id| ids.contains(&id)))
            .map(|(id, aside)| (id, aside.clone()))
            .collect();
        (entries, asides)
    }

    /// Takes entries `range` and `asides` out.
    fn take_kept(&mut self, range: Range<usize>, asides: &[(AsideId, Aside)], by: &Author) {
        for (id, _) in asides {
            self.forget_aside(*id, by);
        }
        self.splice(range, [], by.clone());
    }

    /// Whether a cut before entry `at` parts a tool call from its result:
    /// an entry before it has a call not answered before it.
    fn parts_at(&self, at: usize) -> bool {
        let before = self.messages.get(..at).unwrap_or(&self.messages);
        !pending_calls(before).is_empty()
    }

    /// Takes entries `range` out as a kept branch titled `title` following
    /// entry `range.start - 1` ([`Branch::moved`]), with the saved asides
    /// that follow them: the active line keeps what is before and after
    /// it, its title, halt and line values. The branch's aside is the
    /// pointer left behind, which [`merge_branch`](Self::merge_branch)
    /// puts the range back at. Each change is made by `by`.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NoEntry`] when the range is empty
    /// or reaches past the end, [`BranchError::PartsCall`] when a call
    /// would be parted from its result at either end.
    pub fn move_out(
        &mut self,
        range: Range<usize>,
        title: impl Into<String>,
        by: Author,
    ) -> Result<AsideId, BranchError> {
        let len = self.messages.len();
        if range.end > len || range.is_empty() {
            return Err(BranchError::NoEntry(range.start.max(len.min(range.end))));
        }
        if self.parts_at(range.start) || self.parts_at(range.end) {
            return Err(BranchError::PartsCall);
        }
        let point = range.start.checked_sub(1).and_then(|at| self.id(at));
        let (entries, asides) = self.kept(range.clone());
        let branch = Branch {
            title: title.into(),
            entries,
            asides,
            halted: None,
            moved: true,
            values: Map::new(),
        };
        let value = serde_json::to_value(&branch)
            .map_err(|failure| BranchError::Value(failure.to_string()))?;
        self.take_kept(range, &branch.asides, &by);
        Ok(self.add_aside(Aside::new(BRANCH, by, value).after(point)))
    }

    /// Brings kept branch `id`, kept in the active line, into it: a range
    /// moved out ([`move_out`](Self::move_out)) back where its aside is,
    /// any other branch's own entries at the end of the active line. Its
    /// entries keep their ids and made-times ([`restore`](Self::restore)),
    /// its saved asides come back with theirs (the branches kept in it
    /// become kept in the active line), and its line values are set over
    /// the active line's. Its aside is removed: its entries are in the
    /// line. Each change is made by `by`. Returns what changed.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NotABranch`] when aside `id` is no
    /// branch kept in the active line, [`BranchError::PartsCall`] when the
    /// entries before where they go hold a call without its result.
    pub fn merge_branch(&mut self, id: AsideId, by: Author) -> Result<Spliced, BranchError> {
        let aside = self.aside(id).ok_or(BranchError::NotABranch(id))?;
        let branch = Branch::of(aside).ok_or(BranchError::NotABranch(id))?;
        let at = if branch.moved {
            self.aside_position(aside)
        } else {
            self.messages.len()
        };
        if self.parts_at(at) {
            return Err(BranchError::PartsCall);
        }
        self.forget_aside(id, &by);
        let spliced = self.restore(at..at, branch.entries, by.clone());
        for (id, aside) in branch.asides {
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, Some(by.clone()));
        }
        let mut values = self.line_values().cloned().unwrap_or_default();
        values.extend(branch.values);
        self.put_line_values(values, &by);
        Ok(spliced)
    }

    /// Brings the own entries of `line`, another conversation's line that
    /// shares a start with the active line (such as a split of it kept in
    /// another session), into the active line: those after the entries
    /// both share ([`shared_prefix`]), where `at` says. The saved asides
    /// that follow them come along, following them under their new ids
    /// (but kept branches, whose entries are not this conversation's), and
    /// `line`'s line values are set over the active line's. The past of
    /// the active line is never rewritten. Each change is made by `by`.
    /// Returns what changed.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NoAside`] when `at`'s pointer is
    /// not in the active line, [`BranchError::PartsCall`] when the
    /// entries before where they go hold a call without its result.
    pub fn merge_line(
        &mut self,
        line: &Conversation,
        at: MergeAt,
        by: Author,
    ) -> Result<Spliced, BranchError> {
        let (position, moved) = match at {
            MergeAt::End => (self.messages.len(), 0),
            MergeAt::Back { pointer, moved } => {
                let aside = self.aside(pointer).ok_or(BranchError::NoAside(pointer))?;
                (self.aside_position(aside), moved)
            }
        };
        if self.parts_at(position) {
            return Err(BranchError::PartsCall);
        }
        let mine: Vec<Entry> = (0..self.messages.len())
            .filter_map(|at| self.entry(at))
            .collect();
        let theirs: Vec<Entry> = (0..line.messages.len())
            .filter_map(|at| line.entry(at))
            .collect();
        let own = theirs
            .get(shared_prefix(&mine, &theirs)..)
            .unwrap_or_default();
        let old: Vec<Option<MessageId>> = own.iter().map(|entry| entry.id).collect();
        let with = own.iter().enumerate().map(|(n, entry)| Entry {
            id: entry.id.filter(|_| n < moved),
            made: entry.made.filter(|_| n < moved),
            ..entry.clone()
        });
        let spliced = self.restore(position..position, with, by.clone());
        let new: HashMap<MessageId, MessageId> = old
            .into_iter()
            .zip(spliced.changed.iter().copied())
            .filter_map(|(old, new)| Some((old?, new)))
            .collect();
        let asides: Vec<Aside> = line
            .asides()
            .filter(|(_, aside)| aside.saved && aside.kind != BRANCH)
            .filter_map(|(_, aside)| {
                let after = new.get(&aside.after?)?;
                Some(aside.clone().after(Some(*after)))
            })
            .collect();
        for aside in asides {
            self.add_aside(aside);
        }
        if let Some(theirs) = line.line_values() {
            let mut values = self.line_values().cloned().unwrap_or_default();
            values.extend(theirs.clone());
            self.put_line_values(values, &by);
        }
        Ok(spliced)
    }

    /// The line `id` (the active one with `None`) through entry `end`
    /// (exclusive) as a conversation of its own: its entries with their
    /// ids, authors and made-times, the saved asides that follow them or
    /// none (the branches kept at them among them, but not those on the
    /// way to the line), its title, and, only when `end` is the line's
    /// length, its line values and halt. It numbers new entries and asides
    /// on from this conversation, so the ids of both stay apart. `None`
    /// when `id` is no branch or `end` is past the line's end.
    pub fn line_conversation(&self, id: Option<AsideId>, end: usize) -> Option<Conversation> {
        let kept = self.branches();
        let path = match id {
            None => Vec::new(),
            Some(id) => path(&kept, id)?,
        };
        let line = self.line(id)?;
        let entries = line.get(..end)?;
        let whole = end == line.len();
        let on_path: HashSet<AsideId> = path.iter().map(|kept| kept.id).collect();
        let shared = path.first().map_or(end, |kept| kept.at.min(end));
        let ids: HashSet<MessageId> = entries.iter().filter_map(|entry| entry.id).collect();
        let copied = |id: &AsideId, aside: &Aside| {
            aside.saved
                && !on_path.contains(id)
                && aside.kind != ACTIVE_TITLE
                && aside.kind != LINE_VALUES
        };
        let mut asides: Vec<(AsideId, Aside)> = self
            .asides()
            .filter(|(id, aside)| copied(id, aside) && self.aside_position(aside) <= shared)
            .map(|(id, aside)| (id, aside.clone()))
            .collect();
        for kept in &path {
            let follows = |aside: &Aside| aside.after.is_some_and(|after| ids.contains(&after));
            let inner = kept.branch.asides.iter();
            let inner = inner.filter(|(id, aside)| copied(id, aside) && follows(aside));
            asides.extend(inner.cloned());
        }
        let (title, values, halted) = match path.last() {
            Some(kept) => (
                kept.branch.title.clone(),
                kept.branch.values.clone(),
                kept.branch.halted,
            ),
            None => (
                self.line_title().to_owned(),
                self.line_values().cloned().unwrap_or_default(),
                self.halted,
            ),
        };
        let mut next_aside = self.next_aside;
        let mut own = |kind: &str, value: Value| {
            let id = AsideId(next_aside);
            next_aside = next_aside.saturating_add(1);
            (id, Aside::new(kind, Author::rig(), value))
        };
        let mut first = Vec::new();
        if title != MAIN {
            first.push(own(ACTIVE_TITLE, Value::String(title)));
        }
        if whole && !values.is_empty() {
            first.push(own(LINE_VALUES, Value::Object(values)));
        }
        first.extend(asides);
        let state = State {
            messages: entries.iter().map(|entry| entry.message.clone()).collect(),
            meta: entries
                .iter()
                .map(|entry| Meta {
                    id: entry.id.unwrap_or_default(),
                    author: entry.author.clone(),
                    made: entry.made.unwrap_or_default(),
                })
                .collect(),
            asides: first,
            halted: halted.filter(|_| whole),
            next_id: self.next_id,
            next_aside,
        };
        Conversation::try_from(state).ok()
    }

    /// Puts `aside` in the line `line`: the active one with `None`, else
    /// the kept branch `line`, at any depth, among the asides it keeps. Its
    /// id, numbered as the active line's are, so it stays its own when the
    /// branch comes back. Made by its author.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NotABranch`] when `line` is no
    /// kept branch, [`BranchError::Value`] when a branch on the way to it
    /// cannot be written.
    pub fn add_aside_in(
        &mut self,
        line: Option<AsideId>,
        aside: Aside,
    ) -> Result<AsideId, BranchError> {
        let Some(line) = line else {
            return Ok(self.add_aside(aside));
        };
        let kept = self.branches();
        let path: Vec<AsideId> = path(&kept, line)
            .ok_or(BranchError::NotABranch(line))?
            .iter()
            .map(|kept| kept.id)
            .collect();
        let (outer, inner) = path.split_first().ok_or(BranchError::NotABranch(line))?;
        let mut holder = self
            .aside(*outer)
            .cloned()
            .ok_or(BranchError::NotABranch(*outer))?;
        let id = AsideId(self.next_aside);
        let by = aside.author.clone();
        holder.value = put_in(&holder.value, inner, id, aside.clone())?;
        // Numbered, put and taken back in the record, so a conversation
        // read back from it numbers on after it too.
        let reserved = self.add_aside(aside);
        self.forget_aside(reserved, &by);
        self.replace_aside(*outer, holder);
        Ok(id)
    }

    /// Removes aside `id`, its removal recorded as done by `by` when it is
    /// saved.
    fn forget_aside(&mut self, id: AsideId, by: &Author) {
        match self.aside(id).map(|aside| aside.saved) {
            Some(true) => self.record(Change::Aside { id, aside: None }, Some(by.clone())),
            Some(false) => drop(self.remove_aside(id)),
            None => {}
        }
    }
}

/// Lists `branch`, kept as `id` in `inside` sharing `at` entries of
/// `line` (the ids of the line it is kept in), then the branches kept in it.
fn keep(
    found: &mut Vec<KeptBranch>,
    (id, inside, at): (AsideId, Option<AsideId>, usize),
    branch: Branch,
    line: &[MessageId],
) {
    let mut own: Vec<MessageId> = line.get(..at).unwrap_or(line).to_vec();
    own.extend(branch.entries.iter().filter_map(|entry| entry.id));
    let inner: Vec<(AsideId, usize, Branch)> = branch
        .asides
        .iter()
        .filter_map(|(inner, aside)| {
            let branch = Branch::of(aside)?;
            let at = aside.after.map_or(0, |after| {
                let at = own.iter().position(|id| *id == after);
                at.map_or(own.len(), |at| at.saturating_add(1))
            });
            Some((*inner, at, branch))
        })
        .collect();
    found.push(KeptBranch {
        id,
        inside,
        at,
        branch,
    });
    for (inner, at, branch) in inner {
        keep(found, (inner, Some(id), at), branch, &own);
    }
}

/// The branch value `value` with `aside` put as `id` among the asides of
/// the branch at `path` inside it (itself with an empty path).
fn put_in(
    value: &Value,
    path: &[AsideId],
    id: AsideId,
    aside: Aside,
) -> Result<Value, BranchError> {
    let invalid = |failure: serde_json::Error| BranchError::Value(failure.to_string());
    let mut branch = Branch::deserialize(value).map_err(invalid)?;
    match path.split_first() {
        None => branch.asides.push((id, aside)),
        Some((next, rest)) => {
            let (_, holder) = branch
                .asides
                .iter_mut()
                .find(|(known, _)| known == next)
                .ok_or(BranchError::NotABranch(*next))?;
            holder.value = put_in(&holder.value, rest, id, aside)?;
        }
    }
    serde_json::to_value(&branch).map_err(invalid)
}

/// The branches from the active line to branch `id` in `kept`, outermost
/// first; `None` when it is not there. Each branch's enclosing one is
/// listed before it, so the walk only goes back.
fn path(kept: &[KeptBranch], id: AsideId) -> Option<Vec<&KeptBranch>> {
    let mut path = Vec::new();
    let mut next = Some(id);
    let mut before = kept.len();
    while let Some(id) = next {
        let at = kept.get(..before)?.iter().position(|kept| kept.id == id)?;
        let found = kept.get(at)?;
        path.push(found);
        (next, before) = (found.inside, at);
    }
    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests;
