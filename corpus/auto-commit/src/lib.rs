//! Plugin corpus: 30. Auto-commit after each turn (spec `30-auto-commit.md`).
//!
//! A turn that changed files in a git repository ends with a commit of
//! exactly those files: what `git status` shows at its end and did not
//! show at its start, so the user's own uncommitted work stays out. The
//! commit is made at once with "rig: <first line of the request>" as its
//! subject, then reworded with the subject a cheap model writes from its
//! diff, when one is configured. `/undo` takes the last one back and tells
//! the agent; `/autocommit on|off` turns it on or off for the session.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, Message};
use rig_harness::rig_core::effect::family::Completion;
use rig_harness::rig_core::effect::{EffectKind, Family};
use rig_harness::rig_ecs::effects::Effects;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The plugin's name, as the origin of what it tells agents.
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
        if !app.world().contains_resource::<AutoCommitConfig>() {
            app.insert_resource(AutoCommitConfig::from_home());
        }
        app.init_resource::<AutoCommit>()
            .init_resource::<Said>()
            .register_required_components::<Agent, Turns>()
            .add_observer(start)
            .add_observer(touch)
            .add_observer(commit_turn)
            .add_observer(reword)
            .add_command("undo", "Undo the last auto-commit", undo)
            .add_command(
                "autocommit",
                "Commit each turn's changes: /autocommit on|off",
                toggle,
            );
    }
}

/// How the plugin works, from `RIG_HOME/auto-commit.json` when it exists.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AutoCommitConfig {
    /// The catalog model (`vendor/model`) that writes commit subjects;
    /// none keeps "rig: <first line of the request>".
    pub message_model: Option<String>,
    /// How `/undo` takes a commit back.
    pub undo: UndoMode,
    /// The directory in the repository, else the working directory.
    pub dir: Option<PathBuf>,
}

impl AutoCommitConfig {
    fn from_home() -> Self {
        let path = Home::from_env().root().join("auto-commit.json");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn dir(&self) -> PathBuf {
        self.dir
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }
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

/// The model that writes commit subjects, in place of the configured one.
#[derive(Resource, Clone)]
pub struct MessageModel(pub Connection);

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

/// The turns the agent ended, for commit bodies. Saved with the session.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Turns(u32);

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
/// the turn started, the files its `edit` and `write` calls named, and the
/// request's first line.
#[derive(Component)]
struct Started {
    root: PathBuf,
    dirty: BTreeSet<String>,
    touched: BTreeSet<String>,
    request: String,
}

/// A commit whose subject the message model writes.
#[derive(Component)]
struct Rewording {
    commit: String,
    root: PathBuf,
    body: String,
}

/// Records what is dirty as a turn starts.
fn start(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: Query<&Conversation>,
    state: Res<AutoCommit>,
    config: Res<AutoCommitConfig>,
    mut said: ResMut<Said>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    if !state.enabled {
        return;
    }
    let root = match repository(&config.dir()) {
        Ok(root) => root,
        Err(why) => return said.once(agent, why, &mut notices),
    };
    let dirty = match status(&root) {
        Ok(dirty) => dirty,
        Err(why) => return said.once(agent, why, &mut notices),
    };
    let request = agents.get(agent).map(request).unwrap_or_default();
    commands.entity(agent).insert(Started {
        root,
        dirty,
        touched: BTreeSet::new(),
        request,
    });
}

/// Notes the file of each `edit` and `write` call.
fn touch(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    mut agents: Query<&mut Started>,
) {
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    let function = &run.call.function;
    if !matches!(function.name.as_str(), "edit" | "write") {
        return;
    }
    let Some(path) = function
        .arguments
        .get("path")
        .and_then(|path| path.as_str())
    else {
        return;
    };
    if let Ok(TurnOf(agent)) = turns.get(*turn)
        && let Ok(mut started) = agents.get_mut(*agent)
        && let Some(path) = relative(&started.root, Path::new(path))
    {
        started.touched.insert(path);
    }
}

/// Commits what the turn changed, then asks the message model to reword it.
fn commit_turn(
    ended: On<TurnEnded>,
    mut agents: Query<(&AgentId, &mut Turns)>,
    started: Query<&Started>,
    mut state: ResMut<AutoCommit>,
    config: Res<AutoCommitConfig>,
    session: Option<Res<SessionPaths>>,
    mut said: ResMut<Said>,
    model: Writer,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = ended.entity;
    let Ok((id, mut turns)) = agents.get_mut(agent) else {
        return;
    };
    turns.0 += 1;
    let Ok(started) = started.get(agent) else {
        return;
    };
    commands.entity(agent).remove::<Started>();
    if !state.enabled {
        return;
    }
    let root = &started.root;
    let changed = match status(root) {
        Ok(now) => &now - &started.dirty,
        Err(why) => return said.once(agent, why, &mut notices),
    };
    let theirs: Vec<&str> = started
        .touched
        .intersection(&started.dirty)
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
    let session = session
        .as_ref()
        .map_or_else(|| id.0.clone(), |paths| paths.path().display().to_string());
    let body = format!("Session: {session}\nAgent: {}\nTurn: {}", id.0, turns.0);
    let subject = clip(&format!("rig: {}", started.request));
    let paths: Vec<&str> = changed.iter().map(String::as_str).collect();
    let committed = git(root, &[&["add", "-A", "--"], paths.as_slice()].concat())
        .and_then(|_| {
            let message = ["commit", "-q", "-m", &subject, "-m", &body, "--"];
            git(root, &[message.as_slice(), paths.as_slice()].concat())
        })
        .and_then(|_| git(root, &["rev-parse", "HEAD"]));
    let commit = match committed {
        Ok(commit) => commit,
        Err(why) => {
            notices.write(Notice::error(agent, format!("auto-commit failed: {why}")));
            return;
        }
    };
    notices.write(Notice::info(
        agent,
        format!("auto-commit: {} {subject}", short(&commit)),
    ));
    state.commits.push(commit.clone());
    let Some(connection) = model.connection(&config, agent, &mut said, &mut notices) else {
        return;
    };
    let diff = git(root, &["show", "--format=", "--patch", &commit]).unwrap_or_default();
    let diff: String = diff.chars().take(DIFF_CHARS).collect();
    let request = CompletionRequest::new(Message::user(diff)).preamble(PREAMBLE);
    let kind = EffectKind::Completion {
        request,
        stream: true,
    };
    let (_, reply) = model
        .effects
        .dispatch(&id.0, None, connection.handler.0, kind);
    let work = async move {
        reply
            .await
            .into_outcome()
            .await
            .and_then(Completion::unwrap)
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new("commit message"),
        Rewording {
            commit,
            root: root.clone(),
            body,
        },
        Running::spawn(pool, &model.wake, work),
    ));
}

/// What sends the message model its request.
#[derive(bevy_ecs::system::SystemParam)]
struct Writer<'w> {
    effects: Res<'w, Effects>,
    wake: Res<'w, Wake>,
    models: Res<'w, Models>,
    own: Option<Res<'w, MessageModel>>,
}

impl Writer<'_> {
    /// The message model, if one is set and can be reached.
    fn connection(
        &self,
        config: &AutoCommitConfig,
        agent: Entity,
        said: &mut Said,
        notices: &mut MessageWriter<Notice>,
    ) -> Option<Connection> {
        if let Some(own) = &self.own {
            return Some(own.0.clone());
        }
        let reference = config.message_model.as_deref()?;
        self.models
            .connect(reference)
            .map_err(|why| said.once(agent, why, notices))
            .ok()
    }
}

/// Rewords the commit with the model's subject while it is still HEAD; a
/// failed reply keeps the first subject.
fn reword(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&Rewording, &Done<ModelReply>)>,
    mut state: ResMut<AutoCommit>,
    mut commands: Commands,
) {
    let Ok((rewording, Done(reply))) = calls.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let subject = reply
        .as_ref()
        .ok()
        .and_then(|response| response.message())
        .and_then(|message| final_answer(&message))
        .and_then(|answer| {
            let line = answer
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty())?;
            Some(clip(line.trim_matches(['"', '`', '\''])))
        });
    let root = &rewording.root;
    let head = git(root, &["rev-parse", "HEAD"]).ok();
    let (Some(subject), true) = (subject, head.as_ref() == Some(&rewording.commit)) else {
        return;
    };
    let amend = ["commit", "--amend", "--only", "-q", "-m", &subject];
    let amended = git(root, &[amend.as_slice(), &["-m", &rewording.body]].concat())
        .and_then(|_| git(root, &["rev-parse", "HEAD"]));
    if let Ok(amended) = amended
        && let Some(last) = state
            .commits
            .iter_mut()
            .rev()
            .find(|commit| **commit == rewording.commit)
    {
        *last = amended;
    }
}

/// `/undo`: takes the last auto-commit back and tells the agent.
fn undo(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>>,
    mut state: ResMut<AutoCommit>,
    config: Res<AutoCommitConfig>,
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
    let undone = repository(&config.dir()).and_then(|root| {
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
            let note = Deliver::new(agent, text, DeliveryMode::Note);
            commands.trigger(note.with_origin(Origin::plugin(NAME)));
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

/// The first line of the request that started the agent's turn: the text
/// of its conversation's last user message, after the line naming who sent
/// it when that was not the user.
fn request(conversation: &Conversation) -> String {
    let messages = conversation.messages();
    let found = messages.iter().enumerate().rev().find_map(|(at, message)| {
        let Message::User { content } = message else {
            return None;
        };
        content.iter().enumerate().find_map(|(item, content)| {
            let UserContent::Text(text) = content else {
                return None;
            };
            let skip = usize::from(conversation.origin(at, item).is_some());
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
fn repository(dir: &Path) -> Result<PathBuf, String> {
    git(dir, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .map_err(|_| {
            format!(
                "{} is not in a git repository; nothing is committed.",
                dir.display()
            )
        })
}

/// Why nothing is committed while a merge, rebase or the like is under way.
fn busy(root: &Path) -> Result<(), String> {
    for (file, what) in [
        ("MERGE_HEAD", "a merge"),
        ("rebase-merge", "a rebase"),
        ("rebase-apply", "a rebase"),
        ("CHERRY_PICK_HEAD", "a cherry-pick"),
        ("REVERT_HEAD", "a revert"),
    ] {
        let path = git(root, &["rev-parse", "--git-path", file])?;
        if root.join(path).exists() {
            return Err(format!("{what} is under way; nothing is committed."));
        }
    }
    Ok(())
}

/// The files `git status` shows, relative to the repository's top.
fn status(root: &Path) -> Result<BTreeSet<String>, String> {
    busy(root)?;
    let out = run(
        root,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
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

/// `path`, as a tool names it, relative to the repository's top.
fn relative(root: &Path, path: &Path) -> Option<String> {
    let absolute = std::path::absolute(path).ok()?;
    let parent = absolute.parent()?.canonicalize().ok()?;
    let full = parent.join(absolute.file_name()?);
    let relative = full.strip_prefix(root).ok()?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

/// Runs git in `dir`: its trimmed output, or its error.
fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    run(dir, args).map(|out| out.trim().to_owned())
}

/// Runs git in `dir`: its output, or its error.
fn run(dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
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
