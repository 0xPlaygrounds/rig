//! OC-10: merge findings from a side branch.
//!
//! The stack has no branches inside one session (a fork is a new
//! session), so this plugin keeps them itself, in a saved [`Branches`]
//! component on each agent: `/branch <name>` starts a branch at the end of
//! the active one, `/switch <name>` splices the agent's conversation from
//! the shared ancestor on to the other branch's messages, keeping the left
//! branch's own messages in the component. The files a branch's tool
//! calls changed (their published `FilesChanged`) are recorded with the
//! content the call left, and the content before the branch first wrote
//! each one (read at Gate from the call's footprint), so a switch puts the
//! working tree's files back as the entered branch left them.
//!
//! `/branches` lists them. `/merge <branch> [into <branch>] [--with-files]`
//! finds the shared ancestor, asks the agent's model to summarize what the
//! other branch did after it (or after the last merge from it), shows the
//! summary for review (Accept / Edit / Cancel) and, on Accept, appends one
//! note "Merged from branch `<name>`" to the target branch: a
//! `DeliveryMode::Note` message for the active branch, so the model reads it
//! from its next request on, else a message at the end of the stored
//! branch. With `--with-files`, the source branch's changed files that
//! differ from the disk are offered, with their diffs, for per-file apply.

use std::collections::BTreeMap;
use std::path::Path;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::conversation::Entry;
use rig_harness::rig_core::message::{Message, UserContent};
use rig_harness::rig_core::tool::{ChangeKind, FilesChanged, Scope, Touches};
use rig_harness::rig_tools::workspace::{Stat, Workspace};
use serde::{Deserialize, Serialize};

/// The author of this plugin's messages and conversation edits.
pub const NAME: &str = "branch-merge";

/// The branch every agent starts on.
pub const ROOT: &str = "main";

/// The text every merge note starts with.
pub const MERGED: &str = "Merged from branch";

/// The summarizer's instructions.
const PREAMBLE: &str = "You summarize a side branch of a coding conversation for the main line, \
    which will read your summary instead of the branch. In a few short bullet lists give: \
    findings (such as a bug's real cause), decisions, commands that worked (verbatim), \
    files it changed, and dead ends to avoid. Leave out everything else.";

/// The most characters of one tool result the summarizer reads.
const RESULT_CHARS: usize = 2_000;

/// A file a branch's tool call changed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileRecord {
    /// The file, as the tool published it.
    pub path: String,
    /// The length of the branch's line when the call settled.
    pub at: usize,
    /// The change as a unified diff, when the tool made one.
    pub diff: Option<String>,
    /// The file as the call left it: `None` when deleted or unread.
    pub after: Option<String>,
    /// Whether the call deleted it.
    pub deleted: bool,
}

/// One branch of an agent's conversation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    /// Its name.
    pub name: String,
    /// The branch it started from; `None` for [`ROOT`].
    pub parent: Option<String>,
    /// How many of the parent's messages it shares.
    pub base: usize,
    /// Its own messages after `base`, while it is not the active branch.
    pub tail: Vec<Entry>,
    /// Each file it changed as it was before its first change (`None`:
    /// absent).
    pub before: BTreeMap<String, Option<String>>,
    /// The files its tool calls changed, oldest first.
    pub files: Vec<FileRecord>,
}

impl Branch {
    /// The newest record of each file it changed.
    pub fn latest_files(&self) -> BTreeMap<&str, &FileRecord> {
        self.files
            .iter()
            .map(|record| (record.path.as_str(), record))
            .collect()
    }
}

/// A merge applied: the source branch's line was summarized through
/// `through` messages into `target`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Merge {
    /// The branch summarized.
    pub source: String,
    /// The branch the note went to.
    pub target: String,
    /// The length of the source's line the note covers.
    pub through: usize,
}

/// An agent's branches, kept with the session. The active branch's
/// messages are the agent's conversation; every other keeps its own.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Branches {
    /// Every branch, [`ROOT`] first.
    pub list: Vec<Branch>,
    /// The active branch's name.
    pub active: String,
    /// The merges applied, oldest first.
    pub merges: Vec<Merge>,
}

impl Default for Branches {
    fn default() -> Self {
        Self {
            list: vec![Branch {
                name: ROOT.to_owned(),
                ..Branch::default()
            }],
            active: ROOT.to_owned(),
            merges: Vec::new(),
        }
    }
}

impl Branches {
    /// The branch `name`.
    pub fn get(&self, name: &str) -> Option<&Branch> {
        self.list.iter().find(|branch| branch.name == name)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Branch> {
        self.list.iter_mut().find(|branch| branch.name == name)
    }

    /// The branches `name` started from, nearest first.
    pub fn ancestors(&self, name: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut at = self.get(name).and_then(|branch| branch.parent.clone());
        while let Some(parent) = at {
            if found.contains(&parent) || found.len() > self.list.len() {
                break;
            }
            at = self.get(&parent).and_then(|branch| branch.parent.clone());
            found.push(parent);
        }
        found
    }

    /// The messages of branch `name`, from the first: `live` (the agent's
    /// conversation) for the active branch.
    pub fn line(&self, name: &str, live: Option<&[Entry]>) -> Vec<Entry> {
        self.line_within(name, live, self.list.len())
    }

    fn line_within(&self, name: &str, live: Option<&[Entry]>, depth: usize) -> Vec<Entry> {
        if let Some(live) = live.filter(|_| name == self.active) {
            return live.to_vec();
        }
        let Some(branch) = self.get(name) else {
            return Vec::new();
        };
        let mut line = match (&branch.parent, depth.checked_sub(1)) {
            (Some(parent), Some(depth)) => self.line_within(parent, live, depth),
            _ => Vec::new(),
        };
        line.truncate(branch.base);
        line.extend(branch.tail.iter().cloned());
        line
    }

    /// The last merge from `source` into `target`.
    pub fn last_merge(&self, source: &str, target: &str) -> Option<&Merge> {
        self.merges
            .iter()
            .rev()
            .find(|merge| merge.source == source && merge.target == target)
    }
}

/// How many leading messages two lines share: the index after their
/// shared ancestor message.
pub fn shared(a: &[Entry], b: &[Entry]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(a, b)| a.message == b.message && a.author == b.author)
        .count()
}

/// The conversation's entries, without their ids.
fn entries(conversation: &Conversation) -> Vec<Entry> {
    conversation
        .entries()
        .map(|entry| Entry {
            message: entry.message.clone(),
            author: entry.author.cloned(),
            id: None,
        })
        .collect()
}

/// A model call summarizing a branch for a merge.
#[derive(Component, Clone, Debug)]
struct Summarizing(Pending);

/// A question about a merge.
#[derive(Component, Clone, Debug)]
struct Reviewing {
    pending: Pending,
    step: Step,
}

/// What a merge question asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// Accept, edit or cancel the summary.
    Review,
    /// The summary as the user edited it.
    Edit,
    /// Which files to apply.
    Files,
}

/// A merge on its way.
#[derive(Clone, Debug)]
struct Pending {
    agent: Entity,
    source: String,
    target: String,
    through: usize,
    changed: Vec<String>,
    with_files: bool,
    summary: String,
}

/// Branches inside a session, `/branches`, `/branch`, `/switch` and
/// `/merge`.
#[derive(Default)]
pub struct BranchMergePlugin;

impl Plugin for BranchMergePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectWorkspace>()
            .register_required_components::<Agent, Branches>()
            .add_systems(AgentPass, note_before.in_set(Stage::Gate))
            .add_observer(record_changes)
            .add_observer(summarized)
            .add_observer(answered)
            .add_command("branches", "List this conversation's branches", list)
            .add_command(
                "branch",
                "Start a branch here and switch to it: /branch <name>",
                branch,
            )
            .add_command("switch", "Switch to another branch: /switch <name>", switch)
            .add_command(
                "merge",
                "Bring back what a branch learned: /merge <branch> [into <branch>] [--with-files]",
                merge,
            );
    }
}

/// The file `path` in `workspace`: `Some(None)` when absent, `None` when
/// it cannot be read as text, or the workspace is another machine's (the
/// main-thread rule).
fn read_file(workspace: &dyn Workspace, path: &Path) -> Option<Option<String>> {
    if workspace.host().is_some() {
        return None;
    }
    match workspace.stat(path).ok()? {
        Stat::Missing => Some(None),
        Stat::File { .. } => workspace.read_text(path).ok().map(Some),
        Stat::Dir | Stat::Other => None,
    }
}

/// Puts `path` as `content` (`None`: removed).
fn put_file(workspace: &dyn Workspace, path: &Path, content: Option<&str>) {
    if workspace.host().is_some() {
        return;
    }
    let _ = match content {
        Some(text) => workspace.write(path, text.as_bytes()),
        None => workspace.remove(path),
    };
}

/// Notes each file a call about to run may write as it is before the
/// active branch's first change to it.
fn note_before(
    calls: Query<(&Footprint, &ForAgent), (Undecided, ToolCalls)>,
    mut agents: Query<&mut Branches>,
    workspace: Res<ProjectWorkspace>,
) {
    for (Footprint(touches), &ForAgent(agent)) in &calls {
        let Touches::Writes(Scope::Paths(paths)) = touches else {
            continue;
        };
        let Ok(mut branches) = agents.get_mut(agent) else {
            continue;
        };
        let active = branches.active.clone();
        let Some(known) = branches.get(&active).map(|branch| &branch.before) else {
            continue;
        };
        let new: Vec<(String, Option<String>)> = paths
            .iter()
            .filter(|path| !known.contains_key(&*path.to_string_lossy()))
            .filter_map(|path| {
                let content = read_file(&*workspace.0, path)?;
                Some((path.to_string_lossy().into_owned(), content))
            })
            .collect();
        if new.is_empty() {
            continue;
        }
        if let Some(branch) = branches.get_mut(&active) {
            branch.before.extend(new);
        }
    }
}

/// Records the files each settled tool call changed on its agent's active
/// branch, with the content it left.
fn record_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ForAgent), With<ToolRequest>>,
    mut agents: Query<(&Conversation, &mut Branches)>,
    workspace: Res<ProjectWorkspace>,
) {
    let Ok((published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok((conversation, mut branches)) = agents.get_mut(agent) else {
        return;
    };
    let at = conversation.messages().len();
    let active = branches.active.clone();
    let Some(branch) = branches.get_mut(&active) else {
        return;
    };
    for change in changes {
        let deleted = change.kind == ChangeKind::Deleted;
        let after = (!deleted)
            .then(|| read_file(&*workspace.0, Path::new(&change.path)).flatten())
            .flatten();
        branch.files.push(FileRecord {
            path: change.path,
            at,
            diff: change.diff,
            after,
            deleted,
        });
    }
}

/// `/branches`: each branch, with its first prompt, its messages after
/// the point it shares with its parent, and the files it changed.
fn list(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, &Branches)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((conversation, branches)) = agents.get(args.agent) else {
        return;
    };
    let live = entries(conversation);
    let mut lines = Vec::new();
    for branch in &branches.list {
        let line = branches.line(&branch.name, Some(&live));
        let own = line.get(branch.base..).unwrap_or_default();
        let prompt = own.iter().find_map(|entry| user_text(&entry.message));
        let files: Vec<&str> = branch.latest_files().into_keys().collect();
        let mark = if branch.name == branches.active {
            "*"
        } else {
            " "
        };
        let from = branch
            .parent
            .as_ref()
            .map(|parent| format!(" from {parent}@{}", branch.base))
            .unwrap_or_default();
        lines.push(format!(
            "{mark} {}{from}: {} messages{}; files: {}",
            branch.name,
            own.len(),
            prompt
                .map(|text| format!(", \"{}\"", first_line(&text)))
                .unwrap_or_default(),
            if files.is_empty() {
                "none".to_owned()
            } else {
                files.join(", ")
            },
        ));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// `/branch <name>`: a branch starting at the end of the active one,
/// switched to.
fn branch(
    In(args): In<CommandArgs>,
    mut agents: Query<(&Conversation, &mut Branches), Without<ActiveTurn>>,
    mut notices: MessageWriter<Notice>,
) {
    let name = args.args.trim();
    let Ok((conversation, mut branches)) = agents.get_mut(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "/branch waits for the turn to end",
        ));
        return;
    };
    if name.is_empty() || name.contains(char::is_whitespace) || branches.get(name).is_some() {
        notices.write(Notice::error(
            args.agent,
            "/branch needs a new one-word name",
        ));
        return;
    }
    let parent = branches.active.clone();
    let live = entries(conversation);
    if let Some(leaving) = branches.get_mut(&parent) {
        leaving.tail = live.get(leaving.base..).unwrap_or_default().to_vec();
    }
    branches.list.push(Branch {
        name: name.to_owned(),
        parent: Some(parent.clone()),
        base: conversation.messages().len(),
        ..Branch::default()
    });
    branches.active = name.to_owned();
    notices.write(Notice::info(
        args.agent,
        format!("On branch {name}, from {parent}."),
    ));
}

/// `/switch <name>`: the conversation spliced from the shared ancestor on
/// to the branch's messages, and its files put back as it left them.
fn switch(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, &mut Branches), Without<ActiveTurn>>,
    workspace: Res<ProjectWorkspace>,
    mut notices: MessageWriter<Notice>,
) {
    let name = args.args.trim();
    let Ok((mut conversation, mut branches)) = agents.get_mut(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "/switch waits for the turn to end",
        ));
        return;
    };
    if branches.get(name).is_none() {
        notices.write(Notice::error(args.agent, format!("No branch {name}.")));
        return;
    }
    if name == branches.active {
        return;
    }
    let live = entries(&conversation);
    let leaving = branches.active.clone();
    if let Some(branch) = branches.get_mut(&leaving) {
        branch.tail = live.get(branch.base..).unwrap_or_default().to_vec();
        for (path, content) in &branch.before {
            put_file(&*workspace.0, Path::new(path), content.as_deref());
        }
    }
    branches.active = name.to_owned();
    let line = branches.line(name, None);
    let from = shared(&live, &line);
    let rest = line.get(from..).unwrap_or_default().to_vec();
    conversation.splice(from.., rest, Author::plugin(NAME));
    if let Some(branch) = branches.get_mut(name) {
        branch.tail.clear();
        for record in branch.latest_files().into_values() {
            if record.deleted || record.after.is_some() {
                put_file(
                    &*workspace.0,
                    Path::new(&record.path),
                    record.after.as_deref(),
                );
            }
        }
    }
    notices.write(Notice::info(args.agent, format!("On branch {name}.")));
}

/// `/merge <branch> [into <branch>] [--with-files]`: asks the agent's
/// model to summarize what the branch did after the shared ancestor, or
/// after its last merge into the target.
fn merge(
    In(args): In<CommandArgs>,
    agents: Query<(&Conversation, &Branches)>,
    pending: Query<&Summarizing>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((conversation, branches)) = agents.get(args.agent) else {
        return;
    };
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let with_files = words.contains(&"--with-files");
    let words: Vec<&str> = words
        .into_iter()
        .filter(|word| *word != "--with-files")
        .collect();
    let (source, target) = match words.as_slice() {
        [source] => (*source, branches.active.as_str()),
        [source, "into", target] => (*source, *target),
        _ => {
            let usage = "/merge <branch> [into <branch>] [--with-files]";
            notices.write(Notice::error(args.agent, usage));
            return;
        }
    };
    let refusal = if branches.get(source).is_none() || branches.get(target).is_none() {
        Some(format!(
            "No branch {}.",
            if branches.get(source).is_none() {
                source
            } else {
                target
            }
        ))
    } else if source == target {
        Some(format!("A branch cannot be merged into itself ({source})."))
    } else if branches
        .ancestors(target)
        .iter()
        .any(|branch| branch == source)
    {
        Some(format!(
            "{target} is a descendant of {source}: it has {source}'s messages already."
        ))
    } else if pending
        .iter()
        .any(|Summarizing(it)| it.agent == args.agent && it.source == source)
    {
        Some(format!("A merge from {source} is under way."))
    } else {
        None
    };
    if let Some(refusal) = refusal {
        notices.write(Notice::error(args.agent, refusal));
        return;
    }
    let live = entries(conversation);
    let (from, to) = (
        branches.line(source, Some(&live)),
        branches.line(target, Some(&live)),
    );
    let ancestor = shared(&from, &to);
    let start = branches
        .last_merge(source, target)
        .map_or(ancestor, |merge| merge.through.max(ancestor));
    let Some(new) = from.get(start..).filter(|new| !new.is_empty()) else {
        notices.write(Notice::info(
            args.agent,
            format!("Nothing new on {source} to merge."),
        ));
        return;
    };
    let mut changed: Vec<String> = branches
        .get(source)
        .map(|branch| {
            let files = branch.files.iter().filter(|record| record.at >= start);
            files.map(|record| record.path.clone()).collect()
        })
        .unwrap_or_default();
    changed.dedup();
    let mut prompt = format!(
        "The branch `{source}` after the message it shares with `{target}`:\n\n{}",
        render(new)
    );
    if !changed.is_empty() {
        prompt.push_str(&format!(
            "\n\nFiles its tool calls changed: {}",
            changed.join(", ")
        ));
    }
    let request = CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE);
    let pending = Pending {
        agent: args.agent,
        source: source.to_owned(),
        target: target.to_owned(),
        through: from.len(),
        changed,
        with_files,
        summary: String::new(),
    };
    commands.spawn((
        Name::new(format!("summarizing branch {source}")),
        Summarizing(pending),
        ModelRequest::new(request),
        Effect::child_of(None).with_purpose(NAME),
        ForAgent(args.agent),
    ));
}

/// Shows each finished summary for review.
fn summarized(
    settled: On<Add<Settled>>,
    calls: Query<(&Summarizing, &Outcome)>,
    answerers: Answerers,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((Summarizing(pending), outcome)) = calls.get(settled.entity) else {
        return;
    };
    let agent = pending.agent;
    let summary = outcome.completion().map(|response| response.text());
    let Some(summary) = summary.filter(|text| !text.trim().is_empty()) else {
        let why = outcome
            .0
            .as_ref()
            .err()
            .map(ToString::to_string)
            .unwrap_or_default();
        notices.write(Notice::error(
            agent,
            format!("The merge summary failed. {why}"),
        ));
        return;
    };
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::warn(
            agent,
            "Nobody can review the merge; nothing was merged.",
        ));
        return;
    }
    let ask = Ask::new(format!(
        "Merge branch `{}` into `{}`?",
        pending.source, pending.target
    ))
    .body(summary.clone())
    .choice("accept", "Accept")
    .choice("edit", "Edit")
    .choice("cancel", "Cancel");
    let pending = Pending {
        summary,
        ..pending.clone()
    };
    commands.spawn((
        ask,
        ForAgent(agent),
        Reviewing {
            pending,
            step: Step::Review,
        },
    ));
}

/// Acts on each answer to a merge question.
fn answered(
    settled: On<Add<Settled>>,
    asks: Query<(&Reviewing, &Outcome), With<Ask>>,
    mut agents: Query<&mut Branches>,
    workspace: Res<ProjectWorkspace>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((Reviewing { pending, step }, outcome)) = asks.get(settled.entity) else {
        return;
    };
    let agent = pending.agent;
    let answer = Answer::of(outcome);
    let note = match (step, &answer) {
        (Step::Review, Some(answer)) if answer.is("accept") => pending.summary.clone(),
        (Step::Review, Some(answer)) if answer.is("edit") => {
            let ask = Ask::new(format!("Edit the note from branch `{}`", pending.source))
                .initial(pending.summary.clone())
                .multiline();
            let step = Step::Edit;
            commands.spawn((
                ask,
                ForAgent(agent),
                Reviewing {
                    pending: pending.clone(),
                    step,
                },
            ));
            return;
        }
        (Step::Edit, Some(Answer::Text(text))) if !text.trim().is_empty() => text.clone(),
        (Step::Files, answer) => {
            let chosen = match answer {
                Some(Answer::Choices(ids)) => ids.clone(),
                Some(Answer::Choice(id)) => vec![id.clone()],
                _ => Vec::new(),
            };
            apply_files(agent, pending, &chosen, &agents, &workspace, &mut notices);
            return;
        }
        _ => {
            notices.write(Notice::info(agent, "Merge cancelled."));
            return;
        }
    };
    let Ok(mut branches) = agents.get_mut(agent) else {
        return;
    };
    let mut text = format!(
        "{MERGED} `{}` (through its message {}):\n\n{}",
        pending.source,
        pending.through,
        note.trim()
    );
    if !pending.changed.is_empty() {
        let files = pending.changed.join(", ");
        text.push_str(&format!("\n\nFiles it changed (not applied here): {files}"));
    }
    let author = Author::plugin(NAME).titled(format!("merged from {}", pending.source));
    if pending.target == branches.active {
        let deliver = Deliver::new(agent, text, DeliveryMode::Note).with_author(author);
        commands.trigger(deliver);
    } else if let Some(branch) = branches.get_mut(&pending.target) {
        branch.tail.push(Entry::new(Message::user(text)).by(author));
    } else {
        notices.write(Notice::error(
            agent,
            format!("No branch {}.", pending.target),
        ));
        return;
    }
    branches.merges.push(Merge {
        source: pending.source.clone(),
        target: pending.target.clone(),
        through: pending.through,
    });
    notices.write(Notice::info(
        agent,
        format!("Merged {} into {}.", pending.source, pending.target),
    ));
    if pending.with_files {
        offer_files(pending, &branches, &workspace, &mut notices, &mut commands);
    }
}

/// The source branch's files that differ from the disk, offered with their
/// diffs, one choice each.
fn offer_files(
    pending: &Pending,
    branches: &Branches,
    workspace: &ProjectWorkspace,
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let Some(branch) = branches.get(&pending.source) else {
        return;
    };
    let differ: Vec<&FileRecord> = branch
        .latest_files()
        .into_values()
        .filter(|record| record.deleted || record.after.is_some())
        .filter(|record| {
            let now = read_file(&*workspace.0, Path::new(&record.path));
            now.as_ref()
                .is_none_or(|now| now.as_deref() != record.after.as_deref())
        })
        .collect();
    if differ.is_empty() {
        notices.write(Notice::info(
            pending.agent,
            "The files are as the branch left them.",
        ));
        return;
    }
    let mut ask = Ask::new(format!("Apply files from branch `{}`?", pending.source)).multiple();
    let mut body = Vec::new();
    for record in differ {
        let shown = record
            .diff
            .clone()
            .unwrap_or_else(|| "(no diff recorded)".to_owned());
        body.push(format!("{}\n{shown}", record.path));
        ask = ask.choice(record.path.clone(), record.path.clone());
    }
    let reviewing = Reviewing {
        pending: pending.clone(),
        step: Step::Files,
    };
    commands.spawn((
        ask.body(body.join("\n\n")),
        ForAgent(pending.agent),
        reviewing,
    ));
}

/// Puts each chosen file as the source branch left it.
fn apply_files(
    agent: Entity,
    pending: &Pending,
    chosen: &[String],
    agents: &Query<&mut Branches>,
    workspace: &ProjectWorkspace,
    notices: &mut MessageWriter<Notice>,
) {
    let Some(branch) = agents.get(agent).ok() else {
        return;
    };
    let Some(branch) = branch.get(&pending.source) else {
        return;
    };
    let latest = branch.latest_files();
    for path in chosen {
        if let Some(record) = latest.get(path.as_str()) {
            put_file(&*workspace.0, Path::new(path), record.after.as_deref());
            notices.write(Notice::info(
                agent,
                format!("Applied {path} from {}.", pending.source),
            ));
        }
    }
}

/// The text the user typed in `message`, if it is a user message with text.
fn user_text(message: &Message) -> Option<String> {
    let Message::User { content } = message else {
        return None;
    };
    content.iter().find_map(|item| match item {
        UserContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

/// `entries` as the summarizer reads them.
fn render(entries: &[Entry]) -> String {
    let mut out = Vec::new();
    for entry in entries {
        match &entry.message {
            Message::System { content } => out.push(format!("[system] {content}")),
            Message::User { content } => {
                for item in content {
                    match item {
                        UserContent::Text(text) => out.push(format!("[user] {}", text.text)),
                        UserContent::ToolResult(result) => {
                            let text: String = result
                                .content
                                .iter()
                                .filter_map(|part| part.as_text())
                                .collect::<Vec<_>>()
                                .join("\n");
                            let cut: String = text.chars().take(RESULT_CHARS).collect();
                            let failed = if result.is_error { " (failed)" } else { "" };
                            out.push(format!(
                                "[result of {}{failed}] {cut}",
                                result.name.as_str()
                            ));
                        }
                        _ => out.push("[user attachment]".to_owned()),
                    }
                }
            }
            Message::Assistant(reply) => {
                let text = reply.text();
                if !text.trim().is_empty() {
                    out.push(format!("[assistant] {text}"));
                }
                for call in reply.tool_calls() {
                    let arguments = call.function.raw_arguments();
                    out.push(format!(
                        "[call {}] {arguments}",
                        call.function.name.as_str()
                    ));
                }
            }
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests;
