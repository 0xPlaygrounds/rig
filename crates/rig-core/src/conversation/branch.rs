//! Branches inside a conversation: other continuations of it from one of
//! its entries, kept as asides, so they are saved, forked, rewritten and
//! exported with it and never reach a model.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Aside, AsideId, Author, Change, Conversation, Entry, Halt, MessageId, Spliced};

/// The kind of an aside holding an inactive [`Branch`].
pub const BRANCH: &str = "rig.branch";

/// The kind of the aside holding the active continuation's title (a
/// string), which only rig-core writes. It follows no entry, so it stays
/// with the active line.
pub const ACTIVE: &str = "rig.branch.active";

/// The active continuation's title until a branch names it.
pub const MAIN: &str = "main";

/// The kind of the aside holding the active line's own values (an object,
/// `{key: value}`), which only rig-core writes: what was set on the line
/// while it had no entry of its own. It follows no entry, so it stays with
/// the active line, and a branch carries it as [`Branch::values`].
pub const LINE: &str = "rig.line";

/// An inactive branch, as [`Conversation::branches`] lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Kept {
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
}

impl Conversation {
    /// The active continuation's title: [`MAIN`] until
    /// [`branch_off`](Self::branch_off) or
    /// [`switch_branch`](Self::switch_branch) names it.
    pub fn branch_title(&self) -> &str {
        let mut asides = self.asides.iter();
        let title = asides.find(|(_, aside)| aside.kind == ACTIVE);
        title
            .and_then(|(_, aside)| aside.value.as_str())
            .unwrap_or(MAIN)
    }

    /// Every inactive branch, those kept inside other branches too: each
    /// comes after the one it is kept in, in the order put.
    pub fn branches(&self) -> Vec<Kept> {
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

    /// Keeps entries `at..`, maybe none, with the saved asides that follow
    /// them, the halt and the line values, as a branch titled as the active
    /// continuation is ([`branch_title`](Self::branch_title)), following
    /// entry `at - 1`, and takes them out: the conversation goes on from
    /// that entry as the branch titled `title`. `at` may be the length: a
    /// new branch at the end, which starts with a copy of the line values
    /// (the line's state there is its state now); one at an earlier entry
    /// starts with none. Each change is made by `by`. The kept branch's
    /// aside.
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
        let left = self.branch_title().to_owned();
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
    /// made by `by`.
    /// What changed from the line left.
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
    /// assert_eq!(conversation.branch_title(), "blue");
    /// conversation.switch_branch(first, by)?;
    /// assert_eq!(conversation.messages().get(1), Some(&Message::assistant("Red.")));
    /// assert_eq!(conversation.id(1), red);
    /// assert_eq!(conversation.branch_title(), "main");
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
        let left = self.branch_title().to_owned();
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
        let line = asides.find(|(_, aside)| aside.kind == LINE);
        line.and_then(|(_, aside)| aside.value.as_object())
    }

    /// Sets (`Some`) or removes (`None`) the active line's value `key`: one
    /// aside of kind [`LINE`] before the first entry, replaced in place, and
    /// none when the line has no values. Made by `by`.
    pub fn set_line_value(&mut self, key: &str, value: Option<Value>, by: Author) {
        let mut values = self.line_values().cloned().unwrap_or_default();
        match value {
            Some(value) => values.insert(key.to_owned(), value),
            None => values.remove(key),
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

    /// Makes `values` the active line's values: one [`LINE`] aside, replaced
    /// in place, and none when empty.
    fn put_line_values(&mut self, values: Map<String, Value>, by: &Author) {
        if self
            .line_values()
            .map_or(values.is_empty(), |now| *now == values)
        {
            return;
        }
        let known = self.asides().find(|(_, aside)| aside.kind == LINE);
        let known = known.map(|(id, _)| id);
        let empty = values.is_empty();
        let aside = Aside::new(LINE, by.clone(), Value::Object(values));
        match known {
            Some(id) if empty => self.forget_aside(id, by),
            Some(id) => drop(self.replace_aside(id, aside)),
            None => drop(self.add_aside(aside)),
        }
    }

    /// Titles the active continuation `title`: one [`ACTIVE`] aside,
    /// replaced in place, and none for [`MAIN`].
    fn title_branch(&mut self, title: String, by: &Author) {
        if self.branch_title() == title {
            return;
        }
        let known = self.asides().find(|(_, aside)| aside.kind == ACTIVE);
        let known = known.map(|(id, _)| id);
        let aside = Aside::new(ACTIVE, by.clone(), serde_json::Value::String(title));
        match known {
            Some(id) if aside.value == MAIN => self.forget_aside(id, by),
            Some(id) => drop(self.replace_aside(id, aside)),
            None if aside.value == MAIN => {}
            None => drop(self.add_aside(aside)),
        }
    }

    /// Takes entries `at..` and the saved asides that follow them out, as
    /// a branch titled `title` keeping the line values, which the active
    /// line loses; its value.
    fn stash(
        &mut self,
        at: usize,
        title: String,
        by: &Author,
    ) -> Result<serde_json::Value, BranchError> {
        let entries: Vec<Entry> = (at..self.messages.len())
            .filter_map(|at| self.entry(at))
            .collect();
        let ids: Vec<MessageId> = entries.iter().filter_map(|entry| entry.id).collect();
        let asides: Vec<(AsideId, Aside)> = self
            .asides()
            .filter(|(_, aside)| aside.saved && aside.after.is_some_and(|id| ids.contains(&id)))
            .map(|(id, aside)| (id, aside.clone()))
            .collect();
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
        for (id, _) in &branch.asides {
            self.forget_aside(*id, by);
        }
        self.put_line_values(Map::new(), by);
        self.splice(at.., [], by.clone());
        Ok(value)
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
    found: &mut Vec<Kept>,
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
    found.push(Kept {
        id,
        inside,
        at,
        branch,
    });
    for (inner, at, branch) in inner {
        keep(found, (inner, Some(id), at), branch, &own);
    }
}

/// The branches from the active line to branch `id` in `kept`, outermost
/// first; `None` when it is not there. Each branch's enclosing one is
/// listed before it, so the walk only goes back.
fn path(kept: &[Kept], id: AsideId) -> Option<Vec<&Kept>> {
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
