//! OC-10: merge findings from a side branch.
//!
//! Branches are the conversation's own (rig-core's `branch_off` /
//! `switch_branch`): every inactive branch is a `rig.branch` aside holding
//! its entries, so it is saved, forked, rewritten and exported with the
//! conversation, and a branch switched back keeps its ids. `/branch <name>`
//! starts a branch at the end of the active one (the one left goes on as
//! an empty branch there), `/switch <name>` makes another branch the
//! active one, through the branches it is kept in. The active branch's
//! name is an aside of this plugin's.
//!
//! Each tool call's published `FilesChanged` is kept as an aside after the
//! entry that made the call, so it goes with that entry's branch.
//! `/branches` lists every branch with its first prompt, its messages
//! after the point it shares and the files it changed. `/merge <branch>
//! [into <branch>] [--with-files]` finds the shared ancestor by message
//! key, asks the agent's model to summarize what the other branch did
//! after it (or after the last merge from it), shows the summary for review
//! (Accept / Edit / Cancel) and, on Accept, appends one note "Merged from
//! branch `<name>`" to the target branch with an aside recording the
//! source and the last merged message's key. With `--with-files`, the
//! files the source changed are offered with their hunks for per-file
//! apply, each as the source's last change left it (`after_text`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::conversation::{
    Aside, AsideId, BRANCH, Branch, Entry, MessageId, MessageKey,
};
use rig_harness::rig_core::message::{Message, UserContent};
use rig_harness::rig_core::tool::{ChangeKind, FileChange, FilesChanged};
use serde::{Deserialize, Serialize};

/// The author of this plugin's messages and conversation edits.
pub const NAME: &str = "branch-merge";

/// The branch every agent starts on.
pub const ROOT: &str = "main";

/// The text every merge note starts with.
pub const MERGED: &str = "Merged from branch";

/// The kind of the aside naming the active branch.
pub const ACTIVE: &str = "branch-merge.active";

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

/// A message of a branch.
#[derive(Clone, Debug)]
pub struct Said {
    /// Its key.
    pub key: MessageKey,
    /// The message.
    pub message: Message,
}

/// A branch with every message on it, from the first.
#[derive(Clone, Debug)]
pub struct Line {
    /// Its name.
    pub title: String,
    /// Its messages.
    pub said: Vec<Said>,
    /// Where its own messages start: after the point it branched off.
    pub own: usize,
    /// The branch asides it is kept in, outermost first; empty for the
    /// active branch.
    pub path: Vec<AsideId>,
}

impl Line {
    /// Where `aside` follows on it, after the entry it follows; `None`
    /// when it follows an entry not on it.
    fn position(&self, aside: &Aside) -> Option<usize> {
        let Some(after) = aside.after else {
            return Some(0);
        };
        let at = self.said.iter().position(|said| said.key.id == after);
        at.map(|at| at.saturating_add(1))
    }

    fn index_of(&self, key: MessageKey) -> Option<usize> {
        self.said.iter().position(|said| said.key == key)
    }
}

/// The active branch's name.
pub fn active_name(conversation: &Conversation) -> String {
    conversation
        .asides()
        .filter(|(_, aside)| aside.kind == ACTIVE)
        .find_map(|(_, aside)| aside.value.get("name")?.as_str().map(str::to_owned))
        .unwrap_or_else(|| ROOT.to_owned())
}

fn set_active_name(conversation: &mut Conversation, name: &str) {
    let value = serde_json::json!({ "name": name });
    let aside = Aside::new(ACTIVE, Author::plugin(NAME), value);
    let known = conversation
        .asides()
        .find(|(_, aside)| aside.kind == ACTIVE);
    match known.map(|(id, _)| id) {
        Some(id) => {
            conversation.replace_aside(id, aside);
        }
        None => {
            conversation.add_aside(aside);
        }
    }
}

/// Every branch of `conversation`: the active one first, then each kept
/// one.
pub fn lines(conversation: &Conversation) -> Vec<Line> {
    let active: Vec<Said> = conversation
        .entries()
        .map(|entry| Said {
            key: entry.key(),
            message: entry.message.clone(),
        })
        .collect();
    let kept: Vec<(AsideId, usize, Branch)> = conversation.branches().collect();
    let own = kept.iter().map(|(_, at, _)| *at).max().unwrap_or_default();
    let mut found = Vec::new();
    for (id, at, branch) in kept {
        let prefix = active.get(..at).unwrap_or_default().to_vec();
        kept_lines(&mut found, branch, prefix, vec![id]);
    }
    let title = active_name(conversation);
    found.insert(
        0,
        Line {
            title,
            said: active,
            own,
            path: Vec::new(),
        },
    );
    found
}

/// The line of the kept `branch` after `prefix`, and those of the branches
/// it keeps.
fn kept_lines(found: &mut Vec<Line>, branch: Branch, prefix: Vec<Said>, path: Vec<AsideId>) {
    let own = prefix.len();
    let mut line = Line {
        title: branch.title,
        said: prefix,
        own,
        path,
    };
    line.said
        .extend(branch.entries.into_iter().filter_map(|entry| {
            let key = MessageKey {
                made: entry.made.unwrap_or_default(),
                id: entry.id?,
            };
            let message = entry.message;
            Some(Said { key, message })
        }));
    for (id, aside) in &branch.asides {
        let Some(inner) = Branch::of(aside) else {
            continue;
        };
        let at = line.position(aside).unwrap_or(line.said.len());
        let prefix = line.said.get(..at).unwrap_or_default();
        let mut path = line.path.clone();
        path.push(*id);
        kept_lines(found, inner, prefix.to_vec(), path);
    }
    found.push(line);
}

/// Every aside of `conversation` that is not a branch, those kept inside
/// branches included.
fn records(conversation: &Conversation) -> Vec<Aside> {
    fn walk(found: &mut Vec<Aside>, aside: &Aside) {
        match Branch::of(aside) {
            Some(branch) => branch
                .asides
                .iter()
                .for_each(|(_, inner)| walk(found, inner)),
            None => found.push(aside.clone()),
        }
    }
    let mut found = Vec::new();
    conversation
        .asides()
        .for_each(|(_, aside)| walk(&mut found, aside));
    found
}

/// The files changed on `line` after message `from`, in order: each file
/// with every change made to it.
fn changes_after(records: &[Aside], line: &Line, from: usize) -> BTreeMap<String, Vec<FileChange>> {
    let mut found: Vec<(usize, FileChange)> = records
        .iter()
        .filter(|aside| aside.kind == FILES && aside.after.is_some())
        .filter_map(|aside| Some((line.position(aside)?, aside)))
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

/// How many leading messages two lines share: the index after their
/// shared ancestor message.
fn shared(a: &Line, b: &Line) -> usize {
    a.said
        .iter()
        .zip(&b.said)
        .take_while(|(a, b)| a.key == b.key)
        .count()
}

/// Makes `title` the active branch, through each branch it is kept in.
fn go_to(conversation: &mut Conversation, title: &str) -> Result<(), String> {
    let found = lines(conversation);
    let line = found.iter().find(|line| line.title == title);
    let path = line
        .map(|line| line.path.clone())
        .ok_or(format!("No branch {title}."))?;
    let by = Author::plugin(NAME);
    let mut current = active_name(conversation);
    for id in path {
        let aside = conversation.aside(id).cloned();
        let aside = aside.ok_or(format!("Branch {title} moved."))?;
        let branch = Branch::of(&aside).ok_or(format!("Branch {title} moved."))?;
        let spliced = conversation.switch_branch(id, current.clone(), by.clone());
        spliced.map_err(|failure| failure.to_string())?;
        current = branch.title;
    }
    set_active_name(conversation, title);
    Ok(())
}

/// Keeps an empty branch titled `title` following entry `after`.
fn keep_empty(conversation: &mut Conversation, title: String, after: Option<MessageId>) {
    let branch = Branch {
        title,
        ..Branch::default()
    };
    if let Ok(value) = serde_json::to_value(branch) {
        let aside = Aside::new(BRANCH, Author::plugin(NAME), value).after(after);
        conversation.add_aside(aside);
    }
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

/// A merge accepted while its agent's turn ran, appended when it ends.
#[derive(Component, Clone, Debug)]
struct Accepted {
    pending: Pending,
    note: String,
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
    let records = records(conversation);
    let shown: Vec<String> = lines(conversation)
        .iter()
        .map(|line| {
            let own = line.said.get(line.own..).unwrap_or_default();
            let prompt = own.iter().find_map(|said| user_text(&said.message));
            let prompt = prompt.map(|text| format!(", \"{}\"", first_line(&text)));
            let files: Vec<String> = changes_after(&records, line, line.own)
                .into_keys()
                .collect();
            let files = match files.is_empty() {
                true => "none".to_owned(),
                false => files.join(", "),
            };
            let mark = if line.path.is_empty() { "*" } else { " " };
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
    let taken = lines(&conversation).iter().any(|line| line.title == name);
    if name.is_empty() || name.contains(char::is_whitespace) || taken {
        notices.write(Notice::error(
            args.agent,
            "/branch needs a new one-word name",
        ));
        return;
    }
    let left = active_name(&conversation);
    let end = conversation.messages().len().checked_sub(1);
    let after = end.and_then(|at| conversation.id(at));
    keep_empty(&mut conversation, left.clone(), after);
    set_active_name(&mut conversation, name);
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
    if name == active_name(&conversation) {
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
    pending: Query<&Summarizing>,
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
    let active = active_name(conversation);
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
    let ancestor = shared(from, to);
    let refusal = if source == target {
        Some(format!("A branch cannot be merged into itself ({source})."))
    } else if ancestor == from.said.len() {
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
    let records = records(conversation);
    // After the last message the last merge from it into the target covered.
    let merged = records
        .iter()
        .filter(|aside| aside.kind == MERGE && to.position(aside).is_some())
        .filter_map(|aside| MergeRecord::deserialize(&aside.value).ok())
        .filter(|record| record.source == source)
        .filter_map(|record| from.index_of(record.through))
        .max();
    let start = merged.map_or(ancestor, |at| at.saturating_add(1).max(ancestor));
    let (Some(new), Some(last)) = (from.said.get(start..), from.said.last()) else {
        return;
    };
    if new.is_empty() {
        let nothing = format!("Nothing new on {source} to merge.");
        notices.write(Notice::info(args.agent, nothing));
        return;
    }
    let changed: Vec<String> = changes_after(&records, from, start).into_keys().collect();
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
        true => changes_after(&records, from, ancestor)
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
        through: last.key,
        changed,
        with_files,
        summary: String::new(),
        files,
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
    commands.spawn((ask, ForAgent(agent), Reviewing { pending, step }));
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
            let (pending, step) = (pending.clone(), Step::Edit);
            commands.spawn((ask, ForAgent(agent), Reviewing { pending, step }));
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
            let files = files.filter_map(|path| pending.files.get(path)).cloned();
            apply_files(agent, files.collect(), &workspace, &wake, &mut commands);
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
    match busy {
        true => {
            commands.spawn(accepted);
        }
        false => append(&mut conversation, &accepted, &mut notices, &mut commands),
    }
}

/// Appends the merges accepted while an agent's turn ran, once it ends.
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
        append(&mut conversation, accepted, &mut notices, &mut commands);
        commands.entity(entity).despawn();
    }
}

/// Appends the note of an accepted merge to its target branch, with its
/// record, and offers its files when asked to.
fn append(
    conversation: &mut Conversation,
    Accepted { pending, note }: &Accepted,
    notices: &mut MessageWriter<Notice>,
    commands: &mut Commands,
) {
    let agent = pending.agent;
    let mut text = format!("{MERGED} `{}`:\n\n{}", pending.source, note.trim());
    if !pending.changed.is_empty() {
        let files = pending.changed.join(", ");
        text.push_str(&format!("\n\nFiles it changed (not applied here): {files}"));
    }
    let record = MergeRecord {
        source: pending.source.clone(),
        through: pending.through,
    };
    let Ok(value) = serde_json::to_value(&record) else {
        return;
    };
    let home = active_name(conversation);
    let elsewhere = home != pending.target;
    if elsewhere && let Err(why) = go_to(conversation, &pending.target) {
        notices.write(Notice::error(agent, why));
        return;
    }
    let by = Author::plugin(NAME);
    let author = by.clone().titled(format!("merged from {}", pending.source));
    let id = conversation.push(Entry::new(Message::user(text)).by(author));
    conversation.add_aside(Aside::new(MERGE, by, value).after(Some(id)));
    if elsewhere && let Err(why) = go_to(conversation, &home) {
        notices.write(Notice::error(agent, why));
    }
    notices.write(Notice::info(
        agent,
        format!("Merged {} into {}.", pending.source, pending.target),
    ));
    if pending.with_files {
        offer_files(pending, notices, commands);
    }
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
fn render(said: &[Said]) -> String {
    let mut out = Vec::new();
    for said in said {
        match &said.message {
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
