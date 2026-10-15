//! Plugin corpus: 30. Auto-commit after each turn (spec `30-auto-commit.md`).
//!
//! A turn that changed files in a git repository ends with a commit of
//! exactly those files: what `git status` shows at its end and did not
//! show at its start, so the user's own uncommitted work stays out. When a
//! message model is set (`[auto-commit] message_model` in the settings),
//! the turn's proposed end is held by a model call that writes the
//! commit's subject from its diff; otherwise, or when that call fails, the
//! subject is "rig: `<first line of the request>`". `/undo` takes the last
//! commit back and tells the agent; `/autocommit on|off` turns it on or
//! off for the session.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message};
use rig_harness::rig_core::tool::FilesChanged;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The plugin's name: its settings' table, its model calls' purpose and
/// the author of what it tells agents.
const NAME: &str = "auto-commit";

/// The most of a commit's diff the message model reads.
const DIFF_CHARS: usize = 12_000;

/// The longest subject kept.
const SUBJECT_CHARS: usize = 72;

/// What the message model is told.
const PREAMBLE: &str = "You write git commit subjects. Given a diff, answer with one \
    imperative line of at most 72 characters that says what the change does, such as \
    \"Add a parser for config files\". No quotes, no prefix, nothing else.";

#[derive(Default)]
pub struct AutoCommitPlugin;

impl Plugin for AutoCommitPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<AutoCommitConfig>(NAME)
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<AutoCommit>()
            .init_resource::<Said>()
            .add_observer(snapshot)
            .add_observer(touch)
            .add_observer(take_subject)
            .add_observer(commit_turn)
            .add_observer(follow_successor::<WritesSubject>)
            .configure_sets(AgentPass, AutoCommitSystems.in_set(Stage::Review))
            .add_systems(AgentPass, ask_subject.in_set(AutoCommitSystems))
            .add_command("undo", "Undo the last auto-commit", undo)
            .add_command(
                "autocommit",
                "Commit each turn's changes: /autocommit on|off",
                toggle,
            );
    }
}

/// The Review system that holds a turn's proposed end while the message
/// model writes its commit's subject, to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AutoCommitSystems;

/// The `[auto-commit]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutoCommitConfig {
    /// The catalog model (`vendor/model`) that writes commit subjects;
    /// none keeps "rig: `<first line of the request>`".
    pub message_model: Option<String>,
    /// How `/undo` takes a commit back.
    pub undo: UndoMode,
}

/// How `/undo` takes the last auto-commit back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UndoMode {
    /// Drops the commit (`git reset --keep`) when it is HEAD and on no
    /// remote branch, else reverts it.
    #[default]
    Reset,
    /// Always `git revert`.
    Revert,
}

/// Whether turns are committed, and the commits made, newest last, which
/// `/undo` takes back. Saved with the session.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Saved)]
pub struct AutoCommit {
    pub enabled: bool,
    pub commits: Vec<String>,
}

impl Default for AutoCommit {
    fn default() -> Self {
        Self {
            enabled: true,
            commits: Vec::new(),
        }
    }
}

/// Why nothing was committed, each said once.
#[derive(Resource, Default)]
struct Said(BTreeSet<String>);

impl Said {
    fn once(&mut self, agent: Entity, why: String, notices: &mut MessageWriter<Notice>) {
        if self.0.insert(why.clone()) {
            notices.write(Notice::info(agent, format!("auto-commit: {why}")));
        }
    }
}

/// On an agent while its turn runs: the repository, what was dirty when
/// the turn started, and the files its tool calls said they changed.
/// Saved, so a turn a restart carries on keeps its start's snapshot.
#[derive(Component, Reflect, Clone, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Serialize, Deserialize)]
struct Before {
    root: PathBuf,
    dirty: BTreeSet<String>,
    touched: BTreeSet<String>,
}

impl Before {
    /// The files the turn changed: dirty now and not when it started.
    fn changed(&self, processes: &Processes) -> Result<Vec<String>, String> {
        let now = status(processes, &self.root)?;
        Ok(now.difference(&self.dirty).cloned().collect())
    }
}

/// On a turn: the subject the message model wrote for its commit.
#[derive(Component)]
struct Subject(String);

/// On the message model's call: the turn whose end it holds.
#[derive(Component)]
struct WritesSubject(Entity);

/// Records on its agent what is dirty as a turn starts, in the repository
/// it works in (its `WorkDir`, else the project); a turn a restart carries
/// on keeps the snapshot saved when it first started.
fn snapshot(
    turn: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, Has<Restored>)>,
    snapshots: Query<(), With<Before>>,
    dirs: Query<&WorkDir>,
    state: Res<AutoCommit>,
    project: Res<ProjectRoot>,
    processes: Res<Processes>,
    mut said: ResMut<Said>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&TurnOf(agent), restored)) = turns.get(turn.entity) else {
        return;
    };
    if !state.enabled || (restored && snapshots.contains(agent)) {
        return;
    }
    let dir = dirs.get(agent).map_or(&project.0, |WorkDir(dir)| dir);
    let before = repository(&processes, dir).and_then(|root| {
        let dirty = status(&processes, &root)?;
        Ok(Before {
            root,
            dirty,
            touched: BTreeSet::new(),
        })
    });
    match before {
        Ok(before) => {
            commands.entity(agent).insert(before);
        }
        Err(why) => said.once(agent, why, &mut notices),
    }
}

/// Notes the files each tool call of a turn says it changed.
fn touch(
    published: On<Add<Published>>,
    calls: Query<(&Published, &ForAgent), With<CallOf>>,
    mut agents: Query<&mut Before>,
) {
    let Ok((Published(facts), &ForAgent(agent))) = calls.get(published.entity) else {
        return;
    };
    let Ok(mut before) = agents.get_mut(agent) else {
        return;
    };
    let Ok(Some(files)) = facts.get::<FilesChanged>() else {
        return;
    };
    for path in files.paths() {
        if let Some(path) = relative(&before.root, Path::new(path)) {
            before.touched.insert(path);
        }
    }
}

/// Holds each proposed end of a turn that changed files with a call to the
/// message model, given the changes' diff.
fn ask_subject(
    proposed: Query<(Entity, &TurnOf), Added<TurnOutcome>>,
    snapshots: Query<&Before>,
    config: Res<AutoCommitConfig>,
    models: Res<Models>,
    processes: Res<Processes>,
    mut said: ResMut<Said>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (turn, &TurnOf(agent)) in &proposed {
        commands.entity(turn).remove::<Subject>();
        let (Some(reference), Ok(before)) = (config.message_model.as_deref(), snapshots.get(agent))
        else {
            continue;
        };
        let changed = match before.changed(&processes) {
            Ok(changed) if !changed.is_empty() => changed,
            _ => continue,
        };
        let connection = match models.connect(reference) {
            Ok(connection) => connection,
            Err(why) => {
                said.once(agent, why, &mut notices);
                continue;
            }
        };
        let paths: Vec<&str> = changed.iter().map(String::as_str).collect();
        let root = &before.root;
        let diff = git(
            &processes,
            root,
            &[&["add", "-A", "--"], &paths[..]].concat(),
        )
        .and_then(|_| {
            let diff = ["diff", "--cached", "--no-color", "--"];
            git(&processes, root, &[&diff[..], &paths[..]].concat())
        })
        .unwrap_or_default();
        let diff: String = diff.chars().take(DIFF_CHARS).collect();
        let request = CompletionRequest::new(Message::user(diff)).preamble(PREAMBLE);
        commands.spawn((
            Name::new("commit message"),
            ModelRequest::new(request).on(connection),
            Effect::default().with_purpose(NAME),
            ForAgent(agent),
            HoldOf(turn),
            WritesSubject(turn),
        ));
    }
}

/// Keeps the subject the message model answered on its turn, and lets the
/// turn end; a failed call leaves the fallback subject.
fn take_subject(
    settled: On<Add<Settled>>,
    calls: Query<(&WritesSubject, &Outcome)>,
    mut commands: Commands,
) {
    let Ok((&WritesSubject(turn), outcome)) = calls.get(settled.entity) else {
        return;
    };
    commands.entity(settled.entity).try_remove::<HoldOf>();
    let subject = outcome.completion().and_then(|response| {
        let text = response.text();
        let line = text.lines().map(str::trim).find(|line| !line.is_empty())?;
        Some(clip(line.trim_matches(['"', '`', '\''])))
    });
    if let Some(subject) = subject {
        commands.entity(turn).try_insert(Subject(subject));
    }
}

/// Commits what each ended turn changed, and warns about files the turn
/// touched that the user had changed before it.
fn commit_turn(
    settled: On<Add<Settled>>,
    turns: Query<(&Turn, &ForAgent, Option<&Subject>)>,
    agents: Query<(&AgentId, &Conversation, &Before)>,
    mut state: ResMut<AutoCommit>,
    processes: Res<Processes>,
    session: Option<Res<SessionPaths>>,
    mut said: ResMut<Said>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((turn, &ForAgent(agent), subject)) = turns.get(settled.entity) else {
        return;
    };
    let Ok((id, conversation, before)) = agents.get(agent) else {
        return;
    };
    commands.entity(agent).remove::<Before>();
    if !state.enabled {
        return;
    }
    let changed = match before.changed(&processes) {
        Ok(changed) => changed,
        Err(why) => return said.once(agent, why, &mut notices),
    };
    let theirs: Vec<&str> = before
        .touched
        .intersection(&before.dirty)
        .map(String::as_str)
        .collect();
    if !theirs.is_empty() {
        let text = format!(
            "auto-commit: left out {}, which had uncommitted changes before the turn.",
            theirs.join(", ")
        );
        notices.write(Notice::info(agent, text));
    }
    if changed.is_empty() {
        return;
    }
    let subject = subject.map_or_else(
        || clip(&format!("rig: {}", request(conversation, turn.start))),
        |subject| subject.0.clone(),
    );
    let session = session
        .and_then(|paths| paths.id())
        .map_or_else(|| "unsaved".to_owned(), |session| session.to_string());
    let body = format!(
        "Session: {session}\nAgent: {}\nTurn: {}",
        id.0,
        turn.index + 1
    );
    let paths: Vec<&str> = changed.iter().map(String::as_str).collect();
    let root = &before.root;
    let committed = git(
        &processes,
        root,
        &[&["add", "-A", "--"], &paths[..]].concat(),
    )
    .and_then(|_| {
        let commit = ["commit", "-q", "-m", &subject, "-m", &body, "--"];
        git(&processes, root, &[&commit[..], &paths[..]].concat())
    })
    .and_then(|_| git(&processes, root, &["rev-parse", "HEAD"]));
    match committed {
        Ok(commit) => {
            let text = format!("auto-commit: {} {subject}", short(&commit));
            notices.write(Notice::info(agent, text));
            state.commits.push(commit);
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("auto-commit failed: {why}")));
        }
    }
}

/// `/undo`: takes the last auto-commit back and tells the agent.
fn undo(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>>,
    mut state: ResMut<AutoCommit>,
    config: Res<AutoCommitConfig>,
    project: Res<ProjectRoot>,
    processes: Res<Processes>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if agents.get(agent).unwrap_or(false) {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let Some(commit) = state.commits.last().cloned() else {
        notices.write(Notice::info(agent, "Nothing auto-committed to undo."));
        return;
    };
    let git = |root: &Path, args: &[&str]| git(&processes, root, args);
    let undone = repository(&processes, &project.0).and_then(|root| {
        let subject = git(&root, &["log", "-1", "--format=%s", &commit])?;
        let head = git(&root, &["rev-parse", "HEAD"])?;
        let pushed = git(&root, &["branch", "-r", "--contains", &commit])?;
        let parent = git(
            &root,
            &["rev-parse", "--verify", "-q", &format!("{commit}^")],
        );
        let resets = config.undo == UndoMode::Reset && head == commit && pushed.is_empty();
        match (resets, parent) {
            (true, Ok(parent)) => git(&root, &["reset", "-q", "--keep", &parent])?,
            _ => git(&root, &["revert", "--no-edit", &commit])?,
        };
        Ok(subject)
    });
    match undone {
        Ok(subject) => {
            state.commits.pop();
            let what = format!("commit {} \"{subject}\"", short(&commit));
            notices.write(Notice::info(agent, format!("Undid {what}.")));
            let text = format!(
                "The user undid your changes of an earlier turn ({what}): those files are \
                 back as they were before it. Read them again before you change them."
            );
            let note = Deliver::new(agent, text, DeliveryMode::Inform);
            commands.trigger(note.with_author(Author::plugin(NAME)));
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("Cannot undo: {why}")));
        }
    }
}

/// `/autocommit on|off`, or without arguments whether it is on.
fn toggle(
    In(args): In<CommandArgs>,
    mut state: ResMut<AutoCommit>,
    mut notices: MessageWriter<Notice>,
) {
    let enabled = match args.args.as_str() {
        "on" => true,
        "off" => false,
        "" => state.enabled,
        _ => {
            notices.write(Notice::error(args.agent, "Use /autocommit on or off."));
            return;
        }
    };
    if state.enabled != enabled {
        state.enabled = enabled;
    }
    let says = if enabled { "on" } else { "off" };
    notices.write(Notice::info(args.agent, format!("Auto-commit is {says}.")));
}

/// The first line of the request that asked for the turn that began at
/// `start`, after the header naming who sent it when that was not the
/// user.
fn request(conversation: &Conversation, start: MessageId) -> String {
    let found = conversation.request(start).and_then(|entry| {
        let Message::User { content } = entry.message else {
            return None;
        };
        let skip = usize::from(entry.author.and_then(Author::header).is_some());
        content.iter().find_map(|content| {
            let UserContent::Text(text) = content else {
                return None;
            };
            text.text
                .lines()
                .skip(skip)
                .map(str::trim)
                .find(|line| !line.is_empty())
        })
    });
    found.unwrap_or("changes").to_owned()
}

/// The top of the repository `dir` is in, or why nothing is committed.
fn repository(processes: &Processes, dir: &Path) -> Result<PathBuf, String> {
    git(processes, dir, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .map_err(|_| {
            format!(
                "{} is not in a git repository; nothing is committed.",
                dir.display()
            )
        })
}

/// Why nothing is committed while a merge, rebase or the like is under way.
fn busy(processes: &Processes, root: &Path) -> Result<(), String> {
    for (file, what) in [
        ("MERGE_HEAD", "a merge"),
        ("rebase-merge", "a rebase"),
        ("rebase-apply", "a rebase"),
        ("CHERRY_PICK_HEAD", "a cherry-pick"),
        ("REVERT_HEAD", "a revert"),
    ] {
        let path = git(processes, root, &["rev-parse", "--git-path", file])?;
        if root.join(path).exists() {
            return Err(format!("{what} is under way; nothing is committed."));
        }
    }
    Ok(())
}

/// The files `git status` shows, relative to the repository's top.
fn status(processes: &Processes, root: &Path) -> Result<BTreeSet<String>, String> {
    busy(processes, root)?;
    let args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"];
    let out = run(processes, root, &args)?;
    let mut paths = BTreeSet::new();
    let mut fields = out.split('\0').filter(|field| !field.is_empty());
    while let Some(entry) = fields.next() {
        let (code, path) = (entry.get(..2).unwrap_or_default(), entry.get(3..));
        paths.extend(path.map(str::to_owned));
        // A rename or copy names the file it came from next.
        if code.contains(['R', 'C']) {
            paths.extend(fields.next().map(str::to_owned));
        }
    }
    Ok(paths)
}

/// The absolute `path` relative to the repository's top `root`, if it is
/// in it.
fn relative(root: &Path, path: &Path) -> Option<String> {
    path.is_absolute().then_some(())?;
    let parent = path.parent()?.canonicalize().ok()?;
    let full = parent.join(path.file_name()?);
    let relative = full.strip_prefix(root).ok()?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

/// Runs git in `dir` as every child process starts: its trimmed output,
/// or its error.
fn git(processes: &Processes, dir: &Path, args: &[&str]) -> Result<String, String> {
    run(processes, dir, args).map(|out| out.trim().to_owned())
}

/// Runs git in `dir`: its output, or its error.
fn run(processes: &Processes, dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = processes
        .launcher()
        .command("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|error| format!("cannot run git: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!("git {}: {}", args.join(" "), error.trim()))
    }
}

/// `text` cut to a subject's length.
fn clip(text: &str) -> String {
    text.chars().take(SUBJECT_CHARS).collect()
}

/// A commit's short id.
fn short(commit: &str) -> &str {
    commit.get(..8).unwrap_or(commit)
}
