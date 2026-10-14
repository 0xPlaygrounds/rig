//! OC-10: merge findings from a side branch.
//!
//! Branches are the conversation's own (rig-core's `branch_off` and
//! `switch_branch`, listed by rig-memory's `lines`, each read as its own
//! conversation with `line_conversation`): every inactive branch is a
//! `rig.branch` aside holding its entries, so it is saved, forked,
//! rewritten and exported with the conversation, and a branch switched
//! back keeps its ids. `/branch <name>` starts a branch at the end of the
//! active one, `/switch <name>` makes another branch the active one.
//!
//! Each tool call's published `FilesChanged` is kept as an aside after the
//! entry that made the call, so it goes with that entry's branch.
//! `/branches` lists every branch with its first prompt, its messages
//! after the point it shares and the files it changed. `/merge <branch>
//! [into <branch>] [--with-files]` finds the shared ancestor
//! (`shared_prefix`), asks the agent's model to summarize what the other
//! branch did after it (or after the last merge from it), shows the
//! summary for review (Accept / Edit / Cancel) and, on Accept, sends one
//! note "Merged from branch `<name>`" to the agent (`DeliveryMode::Note`),
//! followed by an aside recording the source and the last merged message's
//! key. A note into an inactive branch is put there directly, once no turn
//! runs, and left for that branch's next request (`Halt::Kept`). With
//! `--with-files`, the files the source changed are offered with their
//! hunks for per-file apply, each as the source's last change left it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::conversation::{self, Aside, AsideId, Entry, MessageKey, shared_prefix};
use rig_harness::rig_core::message::{Message, UserContent};
use rig_harness::rig_core::tool::{ChangeKind, FileChange, FilesChanged};
use rig_harness::rig_memory::{self, LineId};
use serde::{Deserialize, Serialize};

/// The author of this plugin's messages and conversation edits.
pub const NAME: &str = "branch-merge";

/// The text every merge note starts with.
pub const MERGED: &str = "Merged from branch";

/// The kind of the aside after a merge note: a [`MergeRecord`].
pub const MERGE: &str = "branch-merge.merge";

/// The kind of the aside after an entry whose tool calls changed files:
/// a [`FilesRecord`].
pub const FILES: &str = "branch-merge.files";

/// The summarizer's instructions.
const PREAMBLE: &str = "You summarize a side branch of a coding conversation for the main line, \
    which will read your summary instead of the branch. In a few short bullet lists give: \
    findings (such as a bug's real cause), decisions, commands that worked (verbatim), \
    files it changed, and dead ends to avoid. Leave out everything else.";

/// The most characters of one tool result the summarizer reads.
const RESULT_CHARS: usize = 2_000;

/// What a merge note records.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRecord {
    /// The branch summarized.
    pub source: String,
    /// The last of its messages the note covers.
    pub through: MessageKey,
}

/// The files a tool call changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilesRecord {
    /// The changes, as the tool published them.
    pub files: Vec<FileChange>,
}

/// A branch, as a conversation of its own: its entries from the first,
/// with their ids, and the asides that follow them.
#[derive(Clone, Debug)]
pub struct Line {
    /// Its name.
    pub title: String,
    /// Its entries and asides
    /// ([`line_conversation`](conversation::Conversation::line_conversation)).
    pub conversation: conversation::Conversation,
    /// Where its own messages start: after the point it branched off.
    pub own: usize,
    /// Its branch aside; `None` for the active branch.
    pub id: Option<AsideId>,
}

/// Every branch of `conversation`: the active one first, then each kept
/// one, those kept inside others too.
pub fn lines(conversation: &Conversation) -> Vec<Line> {
    let listed = rig_memory::lines(conversation);
    // The active line's own messages start where the last branch kept in
    // it parts.
    let parts = |n: usize| {
        let inside = listed.iter().filter(|line| line.inside == Some(n));
        let kept = inside.filter(|line| matches!(line.id, LineId::Branch(_)));
        kept.map(|line| line.at).max().unwrap_or_default()
    };
    let found = listed.iter().enumerate().filter_map(|(n, line)| {
        let (id, own) = match line.id {
            LineId::Active => (None, parts(n)),
            LineId::Branch(id) => (Some(id), line.at),
            LineId::Session { .. } => return None,
        };
        let end = line.at.saturating_add(line.entries);
        Some(Line {
            title: line.title.clone(),
            conversation: conversation.line_conversation(id, end)?,
            own,
            id,
        })
    });
    found.collect()
}

/// The kept branch named `title`.
fn kept_named(conversation: &Conversation, title: &str) -> Option<AsideId> {
    let listed = rig_memory::lines(conversation).into_iter();
    listed
        .filter(|line| line.title == title)
        .find_map(|line| match line.id {
            LineId::Branch(id) => Some(id),
            _ => None,
        })
}

/// The files changed on `line` after message `from`, in order: each file
/// with every change made to it.
fn changes_after(line: &Line, from: usize) -> BTreeMap<String, Vec<FileChange>> {
    let conversation = &line.conversation;
    let mut found: Vec<(usize, FileChange)> = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == FILES)
        .map(|(_, aside)| (conversation.aside_position(aside), aside))
        .filter(|(at, _)| *at > from)
        .filter_map(|(at, aside)| Some((at, FilesRecord::deserialize(&aside.value).ok()?)))
        .flat_map(|(at, record)| record.files.into_iter().map(move |file| (at, file)))
        .collect();
    found.sort_by_key(|(at, _)| *at);
    let mut files: BTreeMap<String, Vec<FileChange>> = BTreeMap::new();
    for (_, change) in found {
        files.entry(change.path.clone()).or_default().push(change);
    }
    files
}

/// Makes `title` the active branch.
fn go_to(conversation: &mut Conversation, title: &str) -> Result<(), String> {
    let id = kept_named(conversation, title).ok_or(format!("No branch {title}."))?;
    let spliced = conversation.switch_branch(id, Author::plugin(NAME));
    spliced.map(drop).map_err(|failure| failure.to_string())
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

/// A merge into a branch that is not the active one, accepted while its
/// agent's turn ran: appended there when the turn ends, since switching
/// branches under a running turn would change what it reads.
#[derive(Component, Clone, Debug)]
struct Accepted {
    pending: Pending,
    note: String,
}

/// A merge note on its way through the agent's inbox, and the record its
/// entry gets once it is in the conversation.
#[derive(Component, Clone, Debug)]
struct NoteOf(MergeRecord);

/// On each entity of a merge under way, from its summary to its note:
/// the agent and the branch merged.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
struct Merging {
    agent: Entity,
    source: String,
}

/// Files being applied for an agent.
#[derive(Component, Clone, Copy, Debug)]
struct Applying(Entity);

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
    through: MessageKey,
    changed: Vec<String>,
    with_files: bool,
    summary: String,
    /// The files offered: each as the source's last change left it.
    files: BTreeMap<String, FileChange>,
}

impl Pending {
    fn merging(&self) -> Merging {
        let (agent, source) = (self.agent, self.source.clone());
        Merging { agent, source }
    }
}

/// Branches inside a session, `/branches`, `/branch`, `/switch` and
/// `/merge`.
#[derive(Default)]
pub struct BranchMergePlugin;

impl Plugin for BranchMergePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectWorkspace>()
            .add_observer(record_changes)
            .add_observer(summarized)
            .add_observer(answered)
            .add_observer(delivered)
            .add_observer(turn_ended)
            .add_observer(applied)
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

/// Keeps the files each settled tool call changed after the entry that
/// made the call.
fn record_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ToolRequest, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((published, request, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(files))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let made = conversation
        .entries()
        .rev()
        .find(|entry| match entry.message {
            Message::Assistant(reply) => reply.tool_calls().any(|call| call.id == request.call.id),
            _ => false,
        });
    let Some(after) = made.map(|entry| entry.id) else {
        return;
    };
    if let Ok(value) = serde_json::to_value(FilesRecord { files }) {
        let aside = Aside::new(FILES, Author::plugin(NAME), value).after(Some(after));
        conversation.add_aside(aside);
    }
}

/// `/branches`: each branch, with its first prompt, its messages after
/// the point it shares and the files it changed.
fn list(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(conversation) = agents.get(args.agent) else {
        return;
    };
    let shown: Vec<String> = lines(conversation)
        .iter()
        .map(|line| {
            let messages = line.conversation.messages();
            let own = messages.get(line.own..).unwrap_or_default();
            let prompt = own.iter().find_map(user_text);
            let prompt = prompt.map(|text| format!(", \"{}\"", first_line(&text)));
            let files: Vec<String> = changes_after(line, line.own).into_keys().collect();
            let files = match files.is_empty() {
                true => "none".to_owned(),
                false => files.join(", "),
            };
            let mark = if line.id.is_none() { "*" } else { " " };
            format!(
                "{mark} {}: {} messages{}; files: {files}",
                line.title,
                own.len(),
                prompt.unwrap_or_default(),
            )
        })
        .collect();
    notices.write(Notice::info(args.agent, shown.join("\n")));
}

/// `/branch <name>`: a branch starting at the end of the active one,
/// switched to.
fn branch(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation, Without<ActiveTurn>>,
    mut notices: MessageWriter<Notice>,
) {
    let name = args.args.trim();
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "/branch waits for the turn to end",
        ));
        return;
    };
    let listed = rig_memory::lines(&conversation);
    let taken = listed.iter().any(|line| line.title == name);
    if name.is_empty() || name.contains(char::is_whitespace) || taken {
        notices.write(Notice::error(
            args.agent,
            "/branch needs a new one-word name",
        ));
        return;
    }
    let left = conversation.line_title().to_owned();
    let end = conversation.messages().len();
    if let Err(failure) = conversation.branch_off(end, name, Author::plugin(NAME)) {
        notices.write(Notice::error(args.agent, failure.to_string()));
        return;
    }
    notices.write(Notice::info(
        args.agent,
        format!("On branch {name}, from {left}."),
    ));
}

/// `/switch <name>`: the branch made the active one.
fn switch(
    In(args): In<CommandArgs>,
    mut agents: Query<&mut Conversation, Without<ActiveTurn>>,
    mut notices: MessageWriter<Notice>,
) {
    let name = args.args.trim();
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "/switch waits for the turn to end",
        ));
        return;
    };
    if name == conversation.line_title() {
        return;
    }
    match go_to(&mut conversation, name) {
        Ok(()) => notices.write(Notice::info(args.agent, format!("On branch {name}."))),
        Err(why) => notices.write(Notice::error(args.agent, why)),
    };
}

/// `/merge <branch> [into <branch>] [--with-files]`: asks the agent's
/// model to summarize what the branch did after the shared ancestor, or
/// after its last merge into the target.
fn merge(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    under_way: Query<&Merging, Without<Settled>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(conversation) = agents.get(args.agent) else {
        return;
    };
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let with_files = words.contains(&"--with-files");
    let words: Vec<&str> = words
        .into_iter()
        .filter(|word| *word != "--with-files")
        .collect();
    let active = conversation.line_title().to_owned();
    let (source, target) = match words.as_slice() {
        [source] => (*source, active.as_str()),
        [source, "into", target] => (*source, *target),
        _ => {
            let usage = "/merge <branch> [into <branch>] [--with-files]";
            notices.write(Notice::error(args.agent, usage));
            return;
        }
    };
    let found = lines(conversation);
    let line = |name: &str| found.iter().find(|line| line.title == name);
    let (Some(from), Some(to)) = (line(source), line(target)) else {
        let missing = if line(source).is_none() {
            source
        } else {
            target
        };
        notices.write(Notice::error(args.agent, format!("No branch {missing}.")));
        return;
    };
    let entries = |line: &Line| line.conversation.line(None).unwrap_or_default();
    let ancestor = shared_prefix(&entries(from), &entries(to));
    let said = from.conversation.messages();
    let merging = Merging {
        agent: args.agent,
        source: source.to_owned(),
    };
    let refusal = if source == target {
        Some(format!("A branch cannot be merged into itself ({source})."))
    } else if ancestor == said.len() {
        Some(format!(
            "{target} is a descendant of {source}: it has {source}'s messages already."
        ))
    } else if under_way.iter().any(|it| *it == merging) {
        Some(format!("A merge from {source} is under way."))
    } else {
        None
    };
    if let Some(refusal) = refusal {
        notices.write(Notice::error(args.agent, refusal));
        return;
    }
    // After the last message the last merge from it into the target covered.
    let merged = (to.conversation.asides())
        .filter(|(_, aside)| aside.kind == MERGE)
        .filter_map(|(_, aside)| MergeRecord::deserialize(&aside.value).ok())
        .filter(|record| record.source == source)
        .filter_map(|record| from.conversation.index_of_key(record.through))
        .max();
    let start = merged.map_or(ancestor, |at| at.saturating_add(1).max(ancestor));
    let last = said.len().checked_sub(1);
    let last = last.and_then(|at| from.conversation.key(at));
    let (Some(new), Some(last)) = (said.get(start..), last) else {
        return;
    };
    if new.is_empty() {
        let nothing = format!("Nothing new on {source} to merge.");
        notices.write(Notice::info(args.agent, nothing));
        return;
    }
    let changed: Vec<String> = changes_after(from, start).into_keys().collect();
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
    let files = match with_files {
        true => changes_after(from, ancestor)
            .into_iter()
            .filter_map(|(path, changes)| Some((path, changes.into_iter().last()?)))
            .collect(),
        false => BTreeMap::new(),
    };
    let request = CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE);
    let pending = Pending {
        agent: args.agent,
        source: source.to_owned(),
        target: target.to_owned(),
        through: last,
        changed,
        with_files,
        summary: String::new(),
        files,
    };
    commands.spawn((
        Name::new(format!("summarizing branch {source}")),
        Summarizing(pending),
        merging,
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
        let why = outcome.0.as_ref().err().map(ToString::to_string);
        let why = format!("The merge summary failed. {}", why.unwrap_or_default());
        notices.write(Notice::error(agent, why));
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
    let step = Step::Review;
    let merging = pending.merging();
    commands.spawn((ask, ForAgent(agent), Reviewing { pending, step }, merging));
}

/// Acts on each answer to a merge question.
fn answered(
    settled: On<Add<Settled>>,
    asks: Query<(&Reviewing, &Outcome), With<Ask>>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>)>,
    workspace: Res<ProjectWorkspace>,
    wake: Res<Wake>,
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
            let merging = pending.merging();
            let (pending, step) = (pending.clone(), Step::Edit);
            commands.spawn((ask, ForAgent(agent), Reviewing { pending, step }, merging));
            return;
        }
        (
            Step::Edit,
            Some(Answer {
                text: Some(text), ..
            }),
        ) if !text.trim().is_empty() => text.clone(),
        (Step::Files, answer) => {
            let chosen = answer.as_ref().map(|answer| answer.choices.as_slice());
            let files = chosen.unwrap_or_default().iter();
            let files: Vec<FileChange> = files
                .filter_map(|path| pending.files.get(path))
                .cloned()
                .collect();
            if files.is_empty() {
                notices.write(Notice::info(agent, "No file applied."));
            } else {
                apply_files(agent, files, &workspace, &wake, &mut commands);
            }
            return;
        }
        _ => {
            notices.write(Notice::info(agent, "Merge cancelled."));
            return;
        }
    };
    let Ok((mut conversation, busy)) = agents.get_mut(agent) else {
        return;
    };
    let accepted = Accepted {
        pending: pending.clone(),
        note,
    };
    let merging = pending.merging();
    if conversation.line_title() == pending.target {
        deliver(&accepted, merging, &mut commands);
    } else if busy {
        commands.spawn((accepted, merging));
        return;
    } else {
        append_elsewhere(&mut conversation, &accepted, &mut notices);
    }
    if pending.with_files {
        offer_files(pending, &mut notices, &mut commands);
    }
}

/// The note of an accepted merge, and the record it carries.
fn note_of(Accepted { pending, note }: &Accepted) -> (String, MergeRecord) {
    let mut text = format!("{MERGED} `{}`:\n\n{}", pending.source, note.trim());
    if !pending.changed.is_empty() {
        let files = pending.changed.join(", ");
        text.push_str(&format!("\n\nFiles it changed (not applied here): {files}"));
    }
    let record = MergeRecord {
        source: pending.source.clone(),
        through: pending.through,
    };
    (text, record)
}

/// Who writes the note of a merge recorded as `record`: one author per
/// merge, by which its entry is found once delivered.
fn author_of(record: &MergeRecord) -> Author {
    let author = Author::plugin(NAME).titled(format!("merged from {}", record.source));
    let request = format!("merge {} through {}", record.source, record.through.made);
    Author {
        request: Some(RequestId(request)),
        ..author
    }
}

/// Sends the note of a merge into the active branch as a note to the
/// agent: the running turn's next model call carries it, or the next
/// turn's first.
fn deliver(accepted: &Accepted, merging: Merging, commands: &mut Commands) {
    let (text, record) = note_of(accepted);
    let note = Incoming::new(text, DeliveryMode::Note).with_author(author_of(&record));
    let agent = ForAgent(accepted.pending.agent);
    commands.spawn((note, agent, NoteOf(record), merging));
}

/// Puts the record of each delivered merge note after its entry.
fn delivered(
    settled: On<Add<Settled>>,
    notes: Query<(&NoteOf, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((NoteOf(record), &ForAgent(agent))) = notes.get(settled.entity) else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let author = author_of(record);
    let entry = (conversation.entries().rev())
        .find(|entry| entry.author == Some(&author))
        .map(|entry| entry.id);
    let (Some(entry), Ok(value)) = (entry, serde_json::to_value(record)) else {
        let lost = format!("The note from branch {} was not delivered.", record.source);
        notices.write(Notice::warn(agent, lost));
        return;
    };
    let aside = Aside::new(MERGE, Author::plugin(NAME), value).after(Some(entry));
    conversation.add_aside(aside);
    let merged = format!("Merged {}.", record.source);
    notices.write(Notice::info(agent, merged));
}

/// Appends the merges into other branches accepted while an agent's turn
/// ran, once it ends.
fn turn_ended(
    ended: On<Remove<ActiveTurn>>,
    accepted: Query<(Entity, &Accepted)>,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(mut conversation) = agents.get_mut(ended.entity) else {
        return;
    };
    let waiting = accepted
        .iter()
        .filter(|(_, it)| it.pending.agent == ended.entity);
    for (entity, accepted) in waiting {
        append_elsewhere(&mut conversation, accepted, &mut notices);
        if accepted.pending.with_files {
            offer_files(&accepted.pending, &mut notices, &mut commands);
        }
        commands.entity(entity).despawn();
    }
}

/// Appends the note of an accepted merge, with its record, to its target
/// branch, kept while another is active: switched to, then back.
fn append_elsewhere(
    conversation: &mut Conversation,
    accepted: &Accepted,
    notices: &mut MessageWriter<Notice>,
) {
    let (agent, target) = (accepted.pending.agent, &accepted.pending.target);
    let (text, record) = note_of(accepted);
    let Ok(value) = serde_json::to_value(&record) else {
        return;
    };
    let home = conversation.line_title().to_owned();
    if let Err(why) = go_to(conversation, target) {
        notices.write(Notice::error(agent, why));
        return;
    }
    let entry = Entry::new(Message::user(text)).by(author_of(&record));
    let id = conversation.push(entry);
    conversation.add_aside(Aside::new(MERGE, Author::plugin(NAME), value).after(Some(id)));
    // Read from the branch's next request on, as a note to an idle agent
    // is: not answered by itself when the branch is switched to.
    conversation.halt(Halt::Kept);
    if let Err(why) = go_to(conversation, &home) {
        notices.write(Notice::error(agent, why));
    }
    let merged = format!("Merged {} into {target}.", record.source);
    notices.write(Notice::info(agent, merged));
}

/// The source branch's changed files, offered with their hunks, one
/// choice each.
fn offer_files(pending: &Pending, notices: &mut MessageWriter<Notice>, commands: &mut Commands) {
    let offered = pending.files.iter();
    let offered: Vec<_> = offered
        .filter(|(_, change)| change.kind == ChangeKind::Deleted || change.after_text().is_some())
        .collect();
    if offered.is_empty() {
        let none = format!("Branch `{}` left no file to apply.", pending.source);
        notices.write(Notice::info(pending.agent, none));
        return;
    }
    let mut ask = Ask::new(format!("Apply files from branch `{}`?", pending.source)).multiple();
    let mut body = Vec::new();
    for (path, change) in offered {
        body.push(format!("{path}\n{}", change.unified()));
        ask = ask.choice(path.clone(), path.clone());
    }
    let pending = pending.clone();
    let step = Step::Files;
    commands.spawn((
        ask.body(body.join("\n\n")),
        ForAgent(pending.agent),
        Reviewing { pending, step },
    ));
}

/// Puts each of `files` as its change left it, off the main thread.
fn apply_files(
    agent: Entity,
    files: Vec<FileChange>,
    workspace: &ProjectWorkspace,
    wake: &Wake,
    commands: &mut Commands,
) {
    let workspace = workspace.0.clone();
    let writes: Vec<(PathBuf, Option<String>)> = files
        .into_iter()
        .filter_map(|change| match change.kind {
            ChangeKind::Deleted => Some((PathBuf::from(change.path), None)),
            _ => Some((PathBuf::from(&change.path), Some(change.after_text()?))),
        })
        .collect();
    let work = blocking(move || {
        let mut said = Vec::new();
        for (path, text) in writes {
            let done = match &text {
                Some(text) => workspace.write(&path, text.as_bytes()),
                None => workspace.remove(&path),
            };
            said.push(match done {
                Ok(()) => format!("Applied {}.", path.display()),
                Err(failure) => format!("Could not apply {}: {failure}", path.display()),
            });
        }
        Ok(said.join("\n"))
    });
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let running = Running::spawn(pool, wake, async move {
        work.await.unwrap_or_else(|failure| failure.to_string())
    });
    commands.spawn((Applying(agent), running));
}

/// Says what applying files did.
fn applied(
    done: On<Insert<Done<String>>>,
    applying: Query<(&Applying, &Done<String>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&Applying(agent), Done(said))) = applying.get(done.entity) else {
        return;
    };
    notices.write(Notice::info(agent, said.clone()));
    commands.entity(done.entity).despawn();
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

/// `said` as the summarizer reads it.
fn render(said: &[Message]) -> String {
    let mut out = Vec::new();
    for message in said {
        match message {
            Message::System { content } => out.push(format!("[system] {content}")),
            Message::User { content } => {
                for item in content {
                    match item {
                        UserContent::Text(text) => out.push(format!("[user] {}", text.text)),
                        UserContent::ToolResult(result) => {
                            let text: Vec<&str> = result
                                .content
                                .iter()
                                .filter_map(|part| part.as_text())
                                .collect();
                            let cut: String = text.join("\n").chars().take(RESULT_CHARS).collect();
                            let failed = if result.is_error { " (failed)" } else { "" };
                            let tool = result.name.as_str();
                            out.push(format!("[result of {tool}{failed}] {cut}"));
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
                    let (tool, arguments) =
                        (call.function.name.as_str(), call.function.raw_arguments());
                    out.push(format!("[call {tool}] {arguments}"));
                }
            }
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests;
