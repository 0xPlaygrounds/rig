//! SP-05: Changelog fragments (splits fresh held-out spec).
//!
//! When a turn of the main agent ends having changed files that match
//! `[changelog] include` (default `src/**`) and are not all matched by
//! `exclude` (tests, docs, the fragment directory), a model call of the
//! plugin's own on `[changelog] model` reads the turn's request and diff
//! and answers `category: line`. The line is written to
//! `<dir>/<session>-<turn>.<category>.md` (dir `changes` by default), unless
//! every such file is back as it was before the turn by then. The call is
//! no turn's: the next prompt starts at once, and nothing of it reaches the
//! main agent's model. A failed or timed-out call writes nothing, says so
//! once and is not tried again. `/changelog` lists the session's
//! fragments, `/changelog edit <n>` opens one in `$VISUAL` or `$EDITOR` and
//! `/changelog drop <n>` deletes it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::{MatchOptions, Pattern};
use rig_harness::clock::ClockPlugin;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message};
use rig_harness::rig_core::tool::{
    ChangeKind, ContentHash, FileChange, FilesChanged, content_hash,
};
use rig_harness::rig_tools::workspace::{Listing as FileListing, Stat, Workspace};
use rig_tui::{RunInTerminal, TerminalExit};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The settings' table, the command, and the drafting calls' purpose.
const NAME: &str = "changelog";

/// The categories a fragment may have.
const CATEGORIES: [&str; 4] = ["added", "changed", "fixed", "removed"];

/// The most of a turn's diff the drafting model reads.
const DIFF_CHARS: usize = 12_000;

/// The largest file read to tell whether a turn's change was undone.
const READ_LIMIT: u64 = 16 << 20;

/// What the drafting model is told.
const PREAMBLE: &str = "You write changelog entries for a project's users. Given the \
    request a developer made and the diff of the change it led to, answer with exactly one \
    line: a category (added, changed, fixed or removed), a colon, then one short sentence that \
    says what changed for the project's users, such as \
    \"fixed: Parser no longer panics on empty input\". Nothing else.";

/// Drafts a changelog fragment after each turn of the main agent that
/// changed source files, and `/changelog` to list, edit and drop them.
#[derive(Default)]
pub struct ChangelogFragmentsPlugin;

impl Plugin for ChangelogFragmentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ClockPlugin>() {
            app.add_plugins(ClockPlugin);
        }
        app.add_setting::<ChangelogSettings>(NAME)
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .register_required_components::<TurnOf, TurnChanges>()
            .add_observer(note_changes)
            .add_observer(draft)
            .add_observer(drafted)
            .add_observer(timed_out)
            .add_observer(written)
            .add_observer(answered)
            .add_observer(edited)
            .add_observer(follow_successor::<Drafting>)
            .add_command(
                NAME,
                "This session's changelog fragments: /changelog [edit <n> | drop <n>]",
                command,
            );
    }
}

/// The `[changelog]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChangelogSettings {
    /// The catalog model (`vendor/model`) that drafts fragments.
    pub model: String,
    /// Globs over the changed files' paths, relative to the project: a
    /// turn that changed none of them drafts nothing.
    pub include: Vec<String>,
    /// Globs of files that alone never make a fragment; the fragment
    /// directory is always one.
    pub exclude: Vec<String>,
    /// The fragment directory, relative to the project.
    pub dir: String,
    /// How long a drafting call may take, in seconds.
    pub timeout_secs: u64,
}

impl Default for ChangelogSettings {
    fn default() -> Self {
        let exclude = [
            "tests/**",
            "**/tests/**",
            "**/tests.rs",
            "**/*_test.rs",
            "docs/**",
            "**/*.md",
        ];
        Self {
            model: "deepseek/deepseek-flash".to_owned(),
            include: vec!["src/**".to_owned()],
            exclude: exclude.map(str::to_owned).to_vec(),
            dir: "changes".to_owned(),
            timeout_secs: 60,
        }
    }
}

impl ChangelogSettings {
    /// Whether the changed file at `path` (relative to the project) makes
    /// a turn worth a fragment.
    fn counts(&self, path: &Path) -> bool {
        let options = MatchOptions {
            require_literal_separator: true,
            ..MatchOptions::default()
        };
        let matches = |glob: &String| {
            Pattern::new(glob).is_ok_and(|pattern| pattern.matches_path_with(path, options))
        };
        self.include.iter().any(matches)
            && !self.exclude.iter().any(matches)
            && !path.starts_with(&self.dir)
    }
}

/// On every turn: what its tool calls changed, in order.
#[derive(Component, Default)]
struct TurnChanges(Vec<FileChange>);

/// On the plugin's drafting call, a model call of no turn, and on
/// whichever call takes it over: where its fragment goes and what the
/// turn changed.
#[derive(Component, Clone, Debug)]
pub struct Drafting {
    /// The main agent whose turn it drafts for.
    pub agent: Entity,
    /// The turn's number in the session, from 1.
    pub turn: u64,
    /// The fragment directory, absolute.
    pub dir: PathBuf,
    /// The fragment's name before its category: `<session>-<turn>`.
    pub stem: String,
    /// Each changed file that counts, absolute, as it was before the turn.
    changed: Vec<(PathBuf, Before)>,
}

/// A file as it was before a turn changed it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Before {
    /// It did not exist.
    Absent,
    /// It held content of this hash.
    Held(ContentHash),
    /// Its content is not known.
    Unknown,
}

impl Before {
    /// The file before `change`.
    fn of(change: &FileChange) -> Self {
        if change.kind == ChangeKind::Created {
            return Self::Absent;
        }
        let hash = change.before.as_ref().and_then(|state| state.hash.clone());
        hash.map_or(Self::Unknown, Self::Held)
    }
}

/// On the work writing a fragment: the agent told how it went.
#[derive(Component)]
struct Writing(Entity);

/// What writing a fragment did: the file written, or `None` when the
/// turn's changes were undone.
type Written = Result<Option<PathBuf>, String>;

/// On the work a `/changelog` command does: the agent told how it went.
#[derive(Component)]
struct Job(Entity);

/// What a `/changelog` command's work found.
enum Answer {
    /// Tell the user this.
    Say(String),
    /// Open this fragment in the editor.
    Edit(PathBuf),
}

/// On the editor editing a fragment: the agent told if it failed.
#[derive(Component)]
struct Editing(Entity);

/// Notes the files each tool call of a turn changed, on the turn.
fn note_changes(
    published: On<Add<Published>>,
    calls: Query<(&Published, &CallOf), With<ToolRequest>>,
    mut turns: Query<&mut TurnChanges>,
) {
    let Ok((Published(facts), &CallOf(turn))) = calls.get(published.entity) else {
        return;
    };
    let Ok(Some(changed)) = facts.get::<FilesChanged>() else {
        return;
    };
    if let Ok(mut changes) = turns.get_mut(turn) {
        changes.0.extend(changed.0);
    }
}

/// Starts a drafting call once a turn of the main agent ended having
/// changed files that count.
fn draft(
    settled: On<Add<Settled>>,
    turns: Query<(&Turn, &ForAgent, &TurnChanges)>,
    agents: MainAgentQuery,
    conversations: Query<(&Conversation, Option<&WorkDir>)>,
    (settings, root, workspace, models, paths): (
        Res<ChangelogSettings>,
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Models>,
        Option<Res<SessionPaths>>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((turn, &ForAgent(agent), TurnChanges(changes))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((conversation, work_dir)) = conversations.get(agent) else {
        return;
    };
    if changes.is_empty() || main_agent(&agents) != Some(agent) {
        return;
    }
    let base = workspace
        .0
        .canonical(work_dir.map_or(&root.0, |dir| &dir.0));
    let mut changed: BTreeMap<PathBuf, Before> = BTreeMap::new();
    let mut diff = String::new();
    for change in changes {
        let path = PathBuf::from(&change.path);
        let Ok(relative) = path.strip_prefix(&base) else {
            continue;
        };
        if !settings.counts(relative) {
            continue;
        }
        changed
            .entry(path.clone())
            .or_insert_with(|| Before::of(change));
        let shown = FileChange {
            path: relative.display().to_string(),
            ..change.clone()
        };
        diff.push_str(&shown.unified());
    }
    if changed.is_empty() {
        return;
    }
    let connection = match models.connect(&settings.model) {
        Ok(connection) => connection,
        Err(why) => {
            let text = format!("changelog: cannot draft with {}: {why}", settings.model);
            notices.write(Notice::warn(agent, text));
            return;
        }
    };
    let asked = conversation
        .request(turn.start)
        .and_then(|entry| typed(conversation, entry.index))
        .unwrap_or_default();
    let diff: String = diff.chars().take(DIFF_CHARS).collect();
    let prompt = format!("The request:\n{asked}\n\nThe diff:\n{diff}");
    let request = CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE);
    let number = turn.index + 1;
    let drafting = Drafting {
        agent,
        turn: number,
        dir: base.join(&settings.dir),
        stem: format!("{}-{number}", session_short_id(paths.as_deref())),
        changed: changed.into_iter().collect(),
    };
    commands.spawn((
        Name::new("changelog fragment"),
        drafting,
        ModelRequest::new(request).on(connection),
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
        WakeAt::after(Duration::from_secs(settings.timeout_secs)),
    ));
}

/// The session's short id: the first 8 hex digits of its id's hash, or
/// `session` for a run kept in no session directory.
fn session_short_id(paths: Option<&SessionPaths>) -> String {
    let id = paths.and_then(|paths| paths.id());
    id.map_or_else(
        || "session".to_owned(),
        |id| {
            content_hash(id.as_str().as_bytes())
                .0
                .chars()
                .take(8)
                .collect()
        },
    )
}

/// A drafting call that has not settled when its time is up is dropped:
/// nothing is written, and the user is told once.
fn timed_out(
    due: On<Add<Due>>,
    drafts: Query<&Drafting, Without<Settled>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(drafting) = drafts.get(due.entity) else {
        return;
    };
    let text = format!(
        "changelog: drafting turn {}'s fragment timed out; none written",
        drafting.turn
    );
    notices.write(Notice::warn(drafting.agent, text));
    commands.entity(due.entity).despawn();
}

/// Reads the fragment a drafting call answered and writes it off the main
/// thread; a failure is told once.
fn drafted(
    settled: On<Add<Settled>>,
    drafts: Query<(&Drafting, &Outcome)>,
    (workspace, wake): (Res<ProjectWorkspace>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((drafting, outcome)) = drafts.get(settled.entity) else {
        return;
    };
    let fragment = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model answered no text".to_owned()),
        (Ok(_), Some(response)) => parse(&response.text()),
    };
    let (category, line) = match fragment {
        Ok(fragment) => fragment,
        Err(why) => {
            let text = format!(
                "changelog: drafting turn {}'s fragment failed; none written: {why}",
                drafting.turn
            );
            notices.write(Notice::warn(drafting.agent, text));
            return;
        }
    };
    let (workspace, drafting) = (workspace.0.clone(), drafting.clone());
    let agent = drafting.agent;
    let write = blocking(move || Ok(write_fragment(&*workspace, &drafting, &category, &line)));
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = async move { write.await.map_err(|error| error.to_string())? };
    commands.spawn((Writing(agent), Running::spawn(pool, &wake, work)));
}

/// `category: line` from the first line of the model's answer.
fn parse(answer: &str) -> Result<(String, String), String> {
    const QUOTES: [char; 5] = ['"', '\'', '`', '*', ' '];
    let first = answer.lines().map(str::trim).find(|line| !line.is_empty());
    let Some((category, line)) = first.and_then(|first| first.split_once(':')) else {
        return Err(format!("no `category: line` in {answer:?}"));
    };
    let category = category.trim_matches(QUOTES).to_lowercase();
    let line = line.trim().trim_matches(QUOTES).trim();
    if !CATEGORIES.contains(&category.as_str()) {
        return Err(format!("`{category}` is no category"));
    }
    if line.is_empty() {
        return Err("the line is empty".to_owned());
    }
    Ok((category, line.to_owned()))
}

/// Writes `line` as the fragment of `drafting`, unless every file the
/// turn changed is as it was before the turn.
fn write_fragment(
    workspace: &dyn Workspace,
    drafting: &Drafting,
    category: &str,
    line: &str,
) -> Written {
    if drafting
        .changed
        .iter()
        .all(|(path, before)| is_back(workspace, path, before))
    {
        return Ok(None);
    }
    let path = drafting
        .dir
        .join(format!("{}.{category}.md", drafting.stem));
    workspace
        .write(&path, line.as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(Some(path))
}

/// Whether the file at `path` is as it was `before` a turn.
fn is_back(workspace: &dyn Workspace, path: &Path, before: &Before) -> bool {
    match (workspace.stat(path), before) {
        (Ok(Stat::Missing), Before::Absent) => true,
        (Ok(Stat::File { .. }), Before::Held(hash)) => {
            workspace.read(path, READ_LIMIT).is_ok_and(|read| {
                read.size == read.bytes.len() as u64 && content_hash(&read.bytes) == *hash
            })
        }
        _ => false,
    }
}

/// Tells the user which fragment was written, or that the turn's changes
/// were undone first.
fn written(
    done: On<Insert<Done<Written>>>,
    works: Query<(&Writing, &Done<Written>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Writing(agent), Done(written))) = works.get(done.entity) else {
        return;
    };
    let notice = match written {
        Ok(Some(path)) => Notice::info(agent, format!("changelog: drafted {}", path.display())),
        Ok(None) => Notice::info(
            agent,
            "changelog: the turn's changes were undone; no fragment",
        ),
        Err(why) => Notice::warn(agent, format!("changelog: {why}")),
    };
    notices.write(notice);
    commands.entity(done.entity).despawn();
}

/// `/changelog [edit <n> | drop <n>]`: its work runs off the main thread.
fn command(
    In(args): In<CommandArgs>,
    (settings, root, workspace, wake): (
        Res<ChangelogSettings>,
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Wake>,
    ),
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
) {
    let workspace = workspace.0.clone();
    let dir = workspace.canonical(&root.0).join(&settings.dir);
    let short = session_short_id(paths.as_deref());
    let words: Vec<String> = args.args.split_whitespace().map(str::to_owned).collect();
    let work = blocking(move || Ok(run(&*workspace, &dir, &short, &words)));
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = async move {
        work.await
            .unwrap_or_else(|error| Answer::Say(error.to_string()))
    };
    commands.spawn((Job(args.agent), Running::spawn(pool, &wake, work)));
}

/// What `/changelog` with `words` does with the session `short`'s
/// fragments in `dir`.
fn run(workspace: &dyn Workspace, dir: &Path, short: &str, words: &[String]) -> Answer {
    let fragments = match fragments(workspace, dir, short) {
        Ok(fragments) => fragments,
        Err(error) => return Answer::Say(format!("cannot list {}: {error}", dir.display())),
    };
    let pick = |n: Option<&String>| {
        let n: usize = n.and_then(|n| n.parse().ok()).unwrap_or_default();
        n.checked_sub(1).and_then(|at| fragments.get(at))
    };
    match words.split_first() {
        None => Answer::Say(listing(workspace, &fragments)),
        Some((word, rest)) if word == "edit" => {
            pick(rest.first()).map_or_else(|| no_such(rest), |path| Answer::Edit(path.clone()))
        }
        Some((word, rest)) if word == "drop" => match pick(rest.first()) {
            Some(path) => match workspace.remove(path) {
                Ok(()) => Answer::Say(format!("Dropped {}.", name(path))),
                Err(error) => Answer::Say(format!("cannot drop {}: {error}", name(path))),
            },
            None => no_such(rest),
        },
        Some(_) => Answer::Say("Usage: /changelog [edit <n> | drop <n>]".to_owned()),
    }
}

/// The answer to a fragment number that names none.
fn no_such(rest: &[String]) -> Answer {
    let n = rest.first().map_or("", String::as_str);
    Answer::Say(format!("No fragment {n}; /changelog lists them."))
}

/// The session `short`'s fragments in `dir`, by turn.
fn fragments(workspace: &dyn Workspace, dir: &Path, short: &str) -> std::io::Result<Vec<PathBuf>> {
    if workspace.stat(dir)? != Stat::Dir {
        return Ok(Vec::new());
    }
    let prefix = format!("{short}-");
    let mut found: Vec<(u64, PathBuf)> = workspace
        .files(dir, Some("*.md"), FileListing::All)?
        .files
        .into_iter()
        .filter_map(|path| {
            let turn = name(&path)
                .strip_prefix(&prefix)?
                .split('.')
                .next()?
                .parse()
                .ok()?;
            Some((turn, path))
        })
        .collect();
    found.sort();
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

/// The numbered list of `fragments`, each with its line.
fn listing(workspace: &dyn Workspace, fragments: &[PathBuf]) -> String {
    if fragments.is_empty() {
        return "No changelog fragments in this session.".to_owned();
    }
    let mut text = String::from("Changelog fragments:");
    for (at, path) in fragments.iter().enumerate() {
        let line = workspace.read_text(path).unwrap_or_default();
        text.push_str(&format!("\n{}. {}  {}", at + 1, name(path), line.trim()));
    }
    text
}

/// A fragment's file name.
fn name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
}

/// Tells what a `/changelog` command found, or opens the fragment it
/// names in the editor with the terminal.
fn answered(
    done: On<Insert<Done<Answer>>>,
    jobs: Query<(&Job, &Done<Answer>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Job(agent), Done(answer))) = jobs.get(done.entity) else {
        return;
    };
    match answer {
        Answer::Say(text) => {
            notices.write(Notice::info(agent, text.clone()));
        }
        Answer::Edit(path) => {
            let line = corpus_shared::editor::line();
            let run = corpus_shared::editor::command(line.as_deref(), &[path]);
            commands.spawn((RunInTerminal(run), Editing(agent)));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Says when the editor failed, and lets its run go.
fn edited(
    exit: On<Insert<TerminalExit>>,
    runs: Query<(&Editing, &TerminalExit)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Editing(agent), terminal_exit)) = runs.get(exit.entity) else {
        return;
    };
    if let Some(why) = terminal_exit.failure() {
        notices.write(Notice::warn(
            agent,
            format!("changelog: the editor failed: {why}"),
        ));
    }
    commands.entity(exit.entity).despawn();
}
