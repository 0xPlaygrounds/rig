//! Branches inside a conversation: other continuations of it from one of
//! its entries, kept as asides, so they are saved, forked, rewritten and
//! exported with it and never reach a model.

use serde::{Deserialize, Serialize};

use super::{Aside, AsideId, Author, Change, Conversation, Entry, Halt, MessageId, Spliced};

/// The kind of an aside holding an inactive [`Branch`].
pub const BRANCH: &str = "rig.branch";

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
    /// The inactive branches, in the order put: each aside's id, where its
    /// entries would go ([`aside_position`](Self::aside_position)), and the
    /// branch.
    pub fn branches(&self) -> impl Iterator<Item = (AsideId, usize, Branch)> + '_ {
        self.asides().filter_map(|(id, aside)| {
            let branch = Branch::of(aside)?;
            Some((id, self.aside_position(aside), branch))
        })
    }

    /// Keeps entries `at..` and the saved asides that follow them as a
    /// branch titled `title`, following entry `at - 1`, and takes them out:
    /// the conversation goes on from that entry. Each change is made by
    /// `by`. The branch's aside.
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NoEntry`] when there is no entry
    /// `at`, [`BranchError::PartsCall`] when an entry before `at` has a
    /// call answered at or after it.
    pub fn branch_off(
        &mut self,
        at: usize,
        title: impl Into<String>,
        by: Author,
    ) -> Result<AsideId, BranchError> {
        if at >= self.messages.len() {
            return Err(BranchError::NoEntry(at));
        }
        let before = self.messages.get(..at).unwrap_or_default();
        if !super::pairing::pending_calls(before).is_empty() {
            return Err(BranchError::PartsCall);
        }
        let point = at.checked_sub(1).and_then(|at| self.id(at));
        let branch = self.stash(at, title.into(), &by)?;
        Ok(self.add_aside(Aside::new(BRANCH, by, branch).after(point)))
    }

    /// Makes branch `id` the active continuation: the entries after its
    /// point, with the saved asides that follow them and the halt, are kept
    /// as a branch titled `keep_as` (even with no entries), and the branch's
    /// entries come back with their ids and made-times
    /// ([`restore`](Self::restore)), its asides with theirs and its halt. Each change is made by `by`. What the restore
    /// changed.
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
    /// let first = conversation.branch_off(1, "red", by.clone())?;
    /// conversation.push(Message::assistant("Blue."));
    /// conversation.switch_branch(first, "blue", by)?;
    /// assert_eq!(conversation.messages().get(1), Some(&Message::assistant("Red.")));
    /// assert_eq!(conversation.id(1), red);
    /// assert_eq!(conversation.branches().count(), 1);
    /// # Ok::<(), rig_core::conversation::BranchError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Changing nothing: [`BranchError::NotABranch`] when aside `id` is not
    /// a branch.
    pub fn switch_branch(
        &mut self,
        id: AsideId,
        keep_as: impl Into<String>,
        by: Author,
    ) -> Result<Spliced, BranchError> {
        let aside = self.aside(id).ok_or(BranchError::NotABranch(id))?;
        let branch = Branch::of(aside).ok_or(BranchError::NotABranch(id))?;
        let (point, at) = (aside.after, self.aside_position(aside));
        self.forget_aside(id, &by);
        let kept = self.stash(at, keep_as.into(), &by)?;
        let spliced = self.restore(at.., branch.entries, by.clone());
        for (id, aside) in branch.asides {
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, Some(by.clone()));
        }
        if self.halted != branch.halted {
            let halt = branch.halted;
            self.record(Change::Halt { halt }, Some(by.clone()));
        }
        self.add_aside(Aside::new(BRANCH, by, kept).after(point));
        Ok(spliced)
    }

    /// Takes entries `at..` and the saved asides that follow them out, as
    /// a branch titled `title`; its value.
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
        };
        let value = serde_json::to_value(&branch)
            .map_err(|failure| BranchError::Value(failure.to_string()))?;
        for (id, _) in &branch.asides {
            self.forget_aside(*id, by);
        }
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

#[cfg(test)]
mod tests;
