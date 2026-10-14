//! Splits: one agent's line of a session copied into a new session
//! ([`split`]), with ids and made-times kept, the pointer the source keeps
//! to it ([`SplitPointer`]), and every line of a conversation as a tree
//! ([`lines`]).

use std::io;

use rig_core::conversation::pairing::pending_calls;
use rig_core::conversation::{AsideId, Conversation, Entry, MessageId};
use serde::{Deserialize, Serialize};

use crate::fork::copy_plugin_files;
use crate::{Keep, LoadError, SavedAgent, Session, SessionMeta, Store, now_ms};

/// The kind of the aside a line keeps after the last entry a split of it
/// copied: a [`SplitPointer`].
pub const SPLIT: &str = "rig.split";

/// A pointer in a line to a split of it kept in another session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPointer {
    /// The session the split is kept in.
    pub session: String,
    /// The agent of that session whose line it is.
    pub agent: String,
    /// The split's title.
    pub title: String,
    /// Entries moved out of this line into it; 0 for a copy.
    #[serde(default)]
    pub moved: usize,
    /// Whether it was merged back.
    #[serde(default)]
    pub merged: bool,
}

/// What [`split`] copies: one agent's line through message `through` (the
/// whole line with `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitOf<'a> {
    /// The agent, its log's name.
    pub agent: &'a str,
    /// Its line: a kept branch's aside, the active line with `None`.
    pub line: Option<AsideId>,
    /// The last message copied; the line's last with `None`.
    pub through: Option<MessageId>,
}

/// Writes into `to`, an empty store, a session holding only `of`'s line of
/// the session in `from` (read strictly, older formats through the chain):
/// one log for the agent, the same id, its line as one reset with ids and
/// made-times kept, its saved values as they are now, the session log's
/// values, the blobs its images name, the session's plugin files, and its
/// listing ([`SessionMeta`]) with the copied line's length, so a listing
/// shows the split and its listed values before it ever runs. The
/// asides that follow the copied entries come along, nested branches
/// among them, but not the branches on the path to the line copied. The
/// copy as written.
///
/// # Errors
///
/// Writing nothing: as [`Session::read`](crate::Session::read), when
/// `through` is not on the line or the point parts a call from its
/// result, or when a write fails.
pub fn split(
    from: &dyn Store,
    to: &dyn Store,
    of: &SplitOf<'_>,
) -> Result<Conversation, LoadError> {
    let refuse = |why: String| LoadError::from(io::Error::new(io::ErrorKind::InvalidInput, why));
    let session = Session::read(from)?;
    let agent = session
        .agent(of.agent)
        .ok_or_else(|| refuse(format!("the session has no agent `{}`", of.agent)))?;
    let source = &agent.conversation;
    let no_line = || match of.line {
        Some(line) => format!("the agent `{}` has no line {line}", of.agent),
        None => format!("the agent `{}` has no active line", of.agent),
    };
    let line = source.line(of.line).ok_or_else(|| refuse(no_line()))?;
    let end = match of.through {
        None => line.len(),
        Some(id) => line
            .iter()
            .position(|entry| entry.id == Some(id))
            .map(|at| at.saturating_add(1))
            .ok_or_else(|| refuse(format!("message {id} is not on the line")))?,
    };
    let kept: Vec<_> = line
        .iter()
        .take(end)
        .map(|entry: &Entry| entry.message.clone())
        .collect();
    if end < line.len() && !pending_calls(&kept).is_empty() {
        return Err(refuse(
            "a split there would part a tool call from its result".to_owned(),
        ));
    }
    let conversation = source
        .line_conversation(of.line, end)
        .ok_or_else(|| refuse(no_line()))?;
    let mut copy = Session {
        agents: vec![SavedAgent {
            id: agent.id.clone(),
            parent: None,
            started_ms: agent.started_ms,
            conversation,
            state: agent.state.clone(),
            retired: false,
            rewrites: Vec::new(),
        }],
        state: session.state.clone(),
        repairs: Vec::new(),
    };
    copy.save(to)?;
    copy_plugin_files(from, to, &Keep)?;
    let written = copy.agents.into_iter().next();
    let written = written.map(|agent| agent.conversation).unwrap_or_default();
    if let Some(listing) = SessionMeta::read(from) {
        let listing = SessionMeta {
            messages: written.messages().len(),
            updated: now_ms(),
            ..listing
        };
        listing.write(to)?;
    }
    Ok(written)
}

/// A line of a conversation, as [`lines`] lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    /// Which line it is.
    pub id: LineId,
    /// Its title.
    pub title: String,
    /// The position in the list of the line it is kept in; `None` for the
    /// active line.
    pub inside: Option<usize>,
    /// How many entries it shares with the line it is kept in.
    pub at: usize,
    /// How many entries it has of its own.
    pub entries: usize,
    /// Whether its entries were moved out of the line it is kept in.
    pub moved: bool,
    /// Whether it was merged back.
    pub merged: bool,
}

/// Which line a [`Line`] is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LineId {
    /// The active line.
    Active,
    /// A kept branch, by its aside.
    Branch(AsideId),
    /// A split kept in another session: that session and its agent.
    Session {
        /// The session's id.
        session: String,
        /// The agent's id.
        agent: String,
    },
}

/// The lines of `conversation`: the active one, each kept branch (nested
/// ones inside theirs), and each split kept in another session (its
/// pointer), as a tree in display order, each after the line it is kept
/// in.
pub fn lines(conversation: &Conversation) -> Vec<Line> {
    let kept = conversation.branches();
    let active = Line {
        id: LineId::Active,
        title: conversation.branch_title().to_owned(),
        inside: None,
        at: 0,
        entries: conversation.messages().len(),
        moved: false,
        merged: false,
    };
    // Each line with the node of the line it is kept in: node 0 is the
    // active line, node `n + 1` the `n`th kept branch.
    let mut nodes: Vec<(Option<usize>, Line)> = vec![(None, active)];
    let node_of = |inside: Option<AsideId>| {
        inside.map_or(Some(0), |inside| {
            let at = kept.iter().position(|kept| kept.id == inside)?;
            Some(at.saturating_add(1))
        })
    };
    for kept in &kept {
        let line = Line {
            id: LineId::Branch(kept.id),
            title: kept.branch.title.clone(),
            inside: None,
            at: kept.at,
            entries: kept.branch.entries.len(),
            moved: kept.branch.moved,
            merged: false,
        };
        nodes.push((node_of(kept.inside), line));
    }
    let pointers = conversation
        .asides()
        .map(|(_, aside)| (0, conversation.aside_position(aside), aside));
    let mut found: Vec<(usize, usize, SplitPointer)> = pointers
        .filter_map(|(node, at, aside)| Some((node, at, pointer(aside)?)))
        .collect();
    for (n, kept) in kept.iter().enumerate() {
        let ids: Vec<Option<MessageId>> = conversation
            .line(Some(kept.id))
            .unwrap_or_default()
            .iter()
            .map(|entry| entry.id)
            .collect();
        for (_, aside) in &kept.branch.asides {
            let Some(pointer) = pointer(aside) else {
                continue;
            };
            let at = ids.iter().position(|id| *id == aside.after);
            let at = at.map_or(0, |at| at.saturating_add(1));
            found.push((n.saturating_add(1), at, pointer));
        }
    }
    for (node, at, pointer) in found {
        let line = Line {
            id: LineId::Session {
                session: pointer.session,
                agent: pointer.agent,
            },
            title: pointer.title,
            inside: None,
            at,
            entries: pointer.moved,
            moved: pointer.moved > 0,
            merged: pointer.merged,
        };
        nodes.push((Some(node), line));
    }
    let mut listed = Vec::with_capacity(nodes.len());
    let mut placed = vec![None; nodes.len()];
    list_from(0, None, &nodes, &mut placed, &mut listed);
    listed
}

/// The split pointer `aside` holds, when it is one.
fn pointer(aside: &rig_core::conversation::Aside) -> Option<SplitPointer> {
    (aside.kind == SPLIT)
        .then(|| SplitPointer::deserialize(&aside.value).ok())
        .flatten()
}

/// Lists node `node`, kept in the listed line `inside`, then the lines
/// kept in it by where they part, depth first. Each node is listed once.
fn list_from(
    node: usize,
    inside: Option<usize>,
    nodes: &[(Option<usize>, Line)],
    placed: &mut [Option<usize>],
    listed: &mut Vec<Line>,
) {
    let (Some((_, line)), Some(None)) = (nodes.get(node), placed.get(node)) else {
        return;
    };
    let at = listed.len();
    if let Some(slot) = placed.get_mut(node) {
        *slot = Some(at);
    }
    listed.push(Line {
        inside,
        ..line.clone()
    });
    let mut children: Vec<(usize, usize)> = nodes
        .iter()
        .enumerate()
        .filter(|(_, (parent, _))| *parent == Some(node))
        .map(|(child, (_, line))| (line.at, child))
        .collect();
    children.sort();
    for (_, child) in children {
        list_from(child, Some(at), nodes, placed, listed);
    }
}

#[cfg(test)]
mod tests;
