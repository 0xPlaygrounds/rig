//! Plugin corpus: 31. Checkpoints and rewind (spec `31-checkpoint-rewind.md`).
//!
//! Before each turn of a top-level agent, the working tree, with its
//! untracked files that are not ignored, is written as a commit under a
//! private ref, `refs/rig/checkpoints/<session>/<turn>`, through a
//! temporary index: the user's branch, index and stash stay as they are.
//! The snapshot runs off the main thread while it holds the turn's first
//! model call, after the files the turn before left are noted, so the
//! user's edits between turns are told apart from the agent's. `/rewind`
//! picks one of the user's messages and puts the files back as they were
//! before it, cuts the conversation back to just before it, and puts the
//! message back in the input line. Other plugins rewind the same way by
//! spawning a [`Rewind`] request and waiting for its [`Rewound`].

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tools::process::Launcher;
use serde::{Deserialize, Serialize};

/// Where the checkpoints are kept, below the session's id.
const REFS: &str = "refs/rig/checkpoints";

/// The author of the conversation cuts this plugin makes.
const NAME: &str = "checkpoint-rewind";

/// The files before one of the user's messages.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The turn's index, as in its ref's name.
    pub turn: u64,
    /// The ref holding the files: an earlier turn's when nothing changed.
    pub reference: String,
    /// The files, as a git tree.
    pub tree: String,
    /// The user's message, by id, which splices and a restore keep.
    pub message: MessageId,
    /// The message's text.
    pub text: String,
    /// The files the user changed since the turn before ended.
    pub user_edits: Vec<String>,
}

/// An agent's checkpoints, oldest first, kept with the session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Checkpoints {
    /// The checkpoints.
    pub list: Vec<Checkpoint>,
    /// The files as the last turn left them, as a git tree.
    pub settled: Option<String>,
}

#[derive(Default)]
pub struct CheckpointRewindPlugin;

impl Plugin for CheckpointRewindPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<RigHome>()
            .add_message::<SessionDeleted>()
            .register_required_components::<Agent, Checkpoints>()
            .configure_sets(
                AgentPass,
                (
                    CheckpointSystems::Gate.in_set(Stage::Gate),
                    CheckpointSystems::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    snapshot.in_set(CheckpointSystems::Gate),
                    settle.in_set(CheckpointSystems::Review),
                ),
            )
            .add_systems(Update, ((start, confirmed).after(AgentLoop), prune))
            .add_observer(taken)
            .add_observer(stopped)
            .add_observer(settled)
            .add_observer(requested)
            .add_command(
                "rewind",
                "Rewind files and conversation to before a message [--files-only|--chat-only]",
                rewind,
            );
    }
}

/// The plugin's stage systems, by stage, to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckpointSystems {
    /// Holds a turn's first model call, unless a gate decided it, until
    /// the files are snapshotted.
    Gate,
    /// Holds a turn's proposed end while it notes the files the turn left.
    Review,
}

/// On a top-level agent's turn whose snapshot was taken or asked for: once
/// is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Snapshotted;

/// A snapshot holding a turn's first model call: waiting for the tree
/// the agent's turn before left, then being taken.
#[derive(Component)]
struct Snapshot {
    agent: Entity,
    turn: u64,
    message: MessageId,
    text: String,
}

/// What a snapshot found: `None` outside a git repository.
type Taken = Done<Result<Option<Snap>, String>>;

struct Snap {
    tree: String,
    reference: String,
    user_edits: Vec<String>,
}

/// On a snapshot not started yet.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Waiting;

/// The tree an agent's turn left, being noted.
#[derive(Component)]
struct Settling(Entity);

/// The tree a turn left: `None` outside a git repository.
type Tree = Done<Result<Option<String>, String>>;

/// Where git runs, and the session that names the refs.
#[derive(SystemParam)]
struct Repo<'w> {
    root: Res<'w, ProjectRoot>,
    processes: Res<'w, Processes>,
    home: Res<'w, RigHome>,
    paths: Option<Res<'w, SessionPaths>>,
    wake: Res<'w, Wake>,
}

impl Repo<'_> {
    fn git(&self) -> Git {
        Git {
            root: self.root.0.clone(),
            launcher: self.processes.launcher(),
        }
    }

    fn session(&self) -> String {
        let id = self.paths.as_ref().and_then(|paths| paths.id());
        id.map_or_else(|| "local".to_owned(), |id| id.to_string())
    }

    /// The file naming the repository a session's checkpoints are in, so
    /// they can be pruned with the session from any project.
    fn record(&self, session: &str) -> PathBuf {
        self.home.0.root().join("checkpoints").join(session)
    }

    /// Runs `work` off the main thread, ending as `Done<T>` on its entity.
    fn spawn<T: Send + Sync + 'static>(
        &self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Running {
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        Running::spawn(pool, &self.wake, async move { work() })
    }
}

/// Holds the first model call of each turn of a top-level agent that
/// answers a message of the user's until the files are snapshotted, unless
/// a gate before it decided the call.
fn snapshot(
    calls: Query<
        (Entity, &CallOf, &ForAgent, &Decision),
        (ReadyToStart, ModelCalls, With<TurnCall>),
    >,
    turns: Query<&Turn, Without<Snapshotted>>,
    agents: Query<(&Conversation, &Checkpoints), Without<SpawnedBy>>,
    mut commands: Commands,
) {
    for (call, &CallOf(turn_entity), &ForAgent(agent), decision) in &calls {
        let Ok(turn) = turns.get(turn_entity) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        let Ok((conversation, checkpoints)) = agents.get(agent) else {
            continue;
        };
        commands.entity(turn_entity).insert(Snapshotted);
        let Some((message, text)) = user_text(conversation, turn.start) else {
            continue;
        };
        // A retried or restored turn answers a message that has one.
        if checkpoints
            .list
            .last()
            .is_some_and(|last| last.message == message)
        {
            continue;
        }
        let job = Snapshot {
            agent,
            turn: turn.index,
            message,
            text,
        };
        commands.spawn((job, Waiting, HoldOf(call)));
    }
}

/// Starts each waiting snapshot once its agent's last turn end is noted.
fn start(
    waiting: Query<(Entity, &Snapshot), With<Waiting>>,
    settling: Query<&Settling>,
    agents: Query<&Checkpoints>,
    repo: Repo,
    mut commands: Commands,
) {
    for (entity, job) in &waiting {
        if settling.iter().any(|&Settling(agent)| agent == job.agent) {
            continue;
        }
        let Ok(checkpoints) = agents.get(job.agent) else {
            continue;
        };
        let git = repo.git();
        let session = repo.session();
        let record = repo.record(&session);
        let index = job.turn;
        let previous = checkpoints
            .list
            .last()
            .map(|last| (last.tree.clone(), last.reference.clone()));
        let settled = checkpoints.settled.clone();
        let running =
            repo.spawn(move || take(&git, &session, &record, index, previous, settled.as_deref()));
        commands.entity(entity).remove::<Waiting>().insert(running);
    }
}

/// Keeps the checkpoint a snapshot took, and lets the call go.
fn taken(
    done: On<Add<Taken>>,
    jobs: Query<(&Snapshot, &Taken)>,
    mut agents: Query<&mut Checkpoints>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((job, Done(result))) = jobs.get(done.entity) else {
        return;
    };
    match result {
        Ok(Some(snap)) => {
            if let Ok(mut checkpoints) = agents.get_mut(job.agent) {
                checkpoints.list.push(Checkpoint {
                    turn: job.turn,
                    reference: snap.reference.clone(),
                    tree: snap.tree.clone(),
                    message: job.message,
                    text: job.text.clone(),
                    user_edits: snap.user_edits.clone(),
                });
            }
        }
        Ok(None) => {}
        Err(failure) => {
            notices.write(Notice::error(
                job.agent,
                format!("No checkpoint: {failure}"),
            ));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Holds each proposed end of a snapshotted turn while it notes the files
/// the turn left, so the next snapshot tells the user's edits apart.
fn settle(
    proposed: Query<(Entity, &TurnOf), (Added<TurnOutcome>, With<Snapshotted>)>,
    repo: Repo,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent)) in &proposed {
        let git = repo.git();
        let running = repo.spawn(move || git.working_tree());
        commands.spawn((Settling(agent), HoldOf(turn), running));
    }
}

/// Notes the files a stopped turn left. An interrupt cannot be held, so
/// this runs after the turn ended; the next snapshot waits for it.
fn stopped(
    ended: On<Add<Settled>>,
    turns: Query<(&ForAgent, &TurnOutcome), With<Snapshotted>>,
    repo: Repo,
    mut commands: Commands,
) {
    if let Ok((&ForAgent(agent), TurnOutcome::Stopped(_))) = turns.get(ended.entity) {
        let git = repo.git();
        let running = repo.spawn(move || git.working_tree());
        commands.spawn((Settling(agent), running));
    }
}

fn settled(
    done: On<Add<Tree>>,
    jobs: Query<(&Settling, &Tree)>,
    mut agents: Query<&mut Checkpoints>,
    mut commands: Commands,
) {
    if let Ok((&Settling(agent), Done(Ok(Some(tree))))) = jobs.get(done.entity)
        && let Ok(mut checkpoints) = agents.get_mut(agent)
    {
        checkpoints.settled = Some(tree.clone());
    }
    commands.entity(done.entity).despawn();
}

/// A request to rewind `agent` to before the user's message `message`,
/// which must have a [`Checkpoint`]. Spawn it on an entity of your own: the
/// plugin acts on it at once, asks the user first when the files it puts
/// back would undo edits of theirs, and inserts [`Rewound`] on the entity
/// when it is done. The requester despawns the entity then; despawning it
/// earlier withdraws a request still waiting for the user's answer.
#[derive(Component, Clone, Debug)]
pub struct Rewind {
    /// The agent whose files or conversation go back.
    pub agent: Entity,
    /// The user's message to go back to before.
    pub message: MessageId,
    /// What goes back.
    pub scope: RewindScope,
}

/// What a [`Rewind`] puts back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RewindScope {
    /// The files and the conversation, with the message back in the input.
    #[default]
    All,
    /// Only the files.
    FilesOnly,
    /// Only the conversation, with the message back in the input.
    ChatOnly,
}

impl RewindScope {
    fn files(self) -> bool {
        self != Self::ChatOnly
    }

    fn chat(self) -> bool {
        self != Self::FilesOnly
    }
}

/// How a [`Rewind`] ended, inserted on its entity. The user was told in
/// a notice either way.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub enum Rewound {
    /// What it asked for is back.
    Done,
    /// The user said no to undoing their edits: nothing changed.
    Declined,
    /// It could not be done, and why: nothing changed.
    Refused(String),
}

/// What `/rewind` was asked.
#[derive(Clone, Copy, Debug)]
struct Asked {
    scope: RewindScope,
    turn: Option<u64>,
}

impl Asked {
    fn parse(args: &str) -> Result<Self, String> {
        let (mut files, mut chat, mut turn) = (true, true, None);
        for word in args.split_whitespace() {
            match word {
                "--files-only" => chat = false,
                "--chat-only" => files = false,
                number => {
                    let number = number
                        .parse()
                        .map_err(|_| format!("Not a turn: `{number}`."))?;
                    turn = Some(number);
                }
            }
        }
        let scope = match (files, chat) {
            (true, true) => RewindScope::All,
            (true, false) => RewindScope::FilesOnly,
            (false, true) => RewindScope::ChatOnly,
            (false, false) => {
                return Err("--files-only and --chat-only exclude each other.".to_owned());
            }
        };
        Ok(Self { scope, turn })
    }

    /// The command line that rewinds to before `turn` the same way.
    fn command(self, turn: u64) -> String {
        let only = match self.scope {
            RewindScope::FilesOnly => "--files-only ",
            RewindScope::ChatOnly => "--chat-only ",
            RewindScope::All => "",
        };
        format!("rewind {only}{turn}")
    }
}

/// On a [`Rewind`] `/rewind` made: despawned when it is done.
#[derive(Component)]
struct ByCommand;

/// On a question whether to undo the user's edits too: the [`Rewind`]'s
/// entity.
#[derive(Component)]
struct Confirm(Entity);

/// What a rewind reads and changes.
#[derive(SystemParam)]
struct Rewinder<'w, 's> {
    requests: Query<'w, 's, (&'static Rewind, Has<ByCommand>)>,
    agents: Query<'w, 's, (&'static mut Conversation, &'static mut Checkpoints)>,
    busy: Query<'w, 's, (), With<ActiveTurn>>,
    repo: Repo<'w>,
    answerers: Answerers<'w, 's>,
    notices: MessageWriter<'w, Notice>,
    recalled: MessageWriter<'w, Recalled>,
    commands: Commands<'w, 's>,
}

impl Rewinder<'_, '_> {
    /// A picker of the user's messages with checkpoints.
    fn pick(&mut self, agent: Entity, asked: Asked) {
        let Ok((_, checkpoints)) = self.agents.get(agent) else {
            return;
        };
        if checkpoints.list.is_empty() {
            self.notices
                .write(Notice::error(agent, "No checkpoints yet."));
            return;
        }
        let picker = checkpoints.list.iter().fold(
            Picker::new("Rewind to before").for_agent(agent),
            |picker, checkpoint| {
                let line = checkpoint.text.lines().next().unwrap_or_default();
                picker.item(
                    format!("{}. {line}", checkpoint.turn),
                    asked.command(checkpoint.turn),
                )
            },
        );
        let selected = picker.items.len().saturating_sub(1);
        self.commands.spawn(Picker { selected, ..picker });
    }

    /// Acts on the request `request`, unless it was withdrawn, and ends it
    /// unless the user is asked first. `confirmed` once they said yes.
    fn run(&mut self, request: Entity, confirmed: bool) {
        let Ok((rewind, by_command)) = self.requests.get(request) else {
            return;
        };
        let rewind = rewind.clone();
        let ended = match self.rewind(request, &rewind, confirmed) {
            Ok(None) => return,
            Ok(Some(done)) => {
                self.notices.write(done);
                Rewound::Done
            }
            Err(refused) => {
                let why = refused.text.clone();
                self.notices.write(refused);
                Rewound::Refused(why)
            }
        };
        self.end(request, by_command, ended);
    }

    fn end(&mut self, request: Entity, by_command: bool, ended: Rewound) {
        if by_command {
            self.commands.entity(request).despawn();
        } else {
            self.commands.entity(request).try_insert(ended);
        }
    }

    /// Rewinds as `rewind` asks: what to tell the user once done, or `None`
    /// when the user was asked first.
    fn rewind(
        &mut self,
        request: Entity,
        rewind: &Rewind,
        confirmed: bool,
    ) -> Result<Option<Notice>, Notice> {
        let agent = rewind.agent;
        if self.busy.contains(agent) {
            return Err(Notice::turn_running(agent));
        }
        let refuse = |why: String| Notice::error(agent, why);
        let git = self.repo.git();
        let Ok((mut conversation, mut checkpoints)) = self.agents.get_mut(agent) else {
            return Err(refuse("No such agent.".to_owned()));
        };
        let at = checkpoints
            .list
            .iter()
            .position(|kept| kept.message == rewind.message);
        let Some((at, target)) = at.and_then(|at| Some((at, checkpoints.list.get(at)?.clone())))
        else {
            return Err(refuse("No checkpoint before that message.".to_owned()));
        };
        let Some(message) = conversation.index_of(target.message) else {
            return Err(refuse(
                "That message is no longer in the conversation.".to_owned(),
            ));
        };
        let turn = target.turn;
        if rewind.scope.files() {
            let current = match git.working_tree() {
                Ok(Some(current)) => current,
                Ok(None) => {
                    return Err(refuse("The files are not in a git repository.".to_owned()));
                }
                Err(failure) => return Err(refuse(format!("Could not rewind: {failure}"))),
            };
            let mut edits: BTreeSet<String> = checkpoints
                .list
                .iter()
                .skip(at + 1)
                .flat_map(|later| later.user_edits.iter().cloned())
                .collect();
            if let Some(settled) = &checkpoints.settled {
                edits.extend(git.changed(settled, &current));
            }
            if !edits.is_empty() && !confirmed {
                let edits: Vec<String> = edits.into_iter().collect();
                let why = format!("This also undoes your edits to {}.", edits.join(", "));
                if !self.answerers.can_answer(Some(agent)) {
                    return Err(refuse(format!("{why} Nobody can confirm it here.")));
                }
                let ask = Ask::yes_no(format!("Rewind to before turn {turn}?")).body(why);
                self.commands
                    .spawn((ask, ForAgent(agent), Confirm(request)));
                return Ok(None);
            }
            git.restore(&current, &target.tree)
                .map_err(|failure| refuse(format!("Could not rewind: {failure}")))?;
            checkpoints.settled = Some(target.tree.clone());
        }
        if rewind.scope.chat() {
            // What went before the user's message, such as a stopped
            // turn's tool results, stays, stopped as it was.
            let stopped = message
                .checked_sub(1)
                .and_then(|before| conversation.entries().nth(before))
                .is_some_and(|entry| entry.is_stop_marker());
            let from = if stopped {
                message.saturating_sub(1)
            } else {
                message
            };
            conversation.splice(from.., [], Author::plugin(NAME));
            if stopped {
                conversation.halt(Halt::Stopped);
            }
            let dropped = checkpoints.list.split_off(at);
            let kept: BTreeSet<&str> = checkpoints
                .list
                .iter()
                .map(|kept| kept.reference.as_str())
                .collect();
            let gone: BTreeSet<&str> = dropped.iter().map(|gone| gone.reference.as_str()).collect();
            for reference in gone.difference(&kept) {
                git.run(None, &["update-ref", "-d", reference]).ok();
            }
            self.recalled.write(Recalled {
                agent,
                text: target.text.clone(),
                why: None,
            });
        }
        let what = match rewind.scope {
            RewindScope::FilesOnly => "the files",
            RewindScope::ChatOnly => "the conversation",
            RewindScope::All => "the files and the conversation",
        };
        let done = format!("Rewound {what} to before turn {turn}.");
        Ok(Some(Notice::info(agent, done)))
    }
}

/// `/rewind [--files-only|--chat-only] [turn]`: without a turn, a picker of
/// the user's messages with checkpoints.
fn rewind(In(args): In<CommandArgs>, mut rewinder: Rewinder) {
    let agent = args.agent;
    if rewinder.busy.contains(agent) {
        rewinder.notices.write(Notice::turn_running(agent));
        return;
    }
    let asked = match Asked::parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            rewinder.notices.write(Notice::error(agent, why));
            return;
        }
    };
    let Some(turn) = asked.turn else {
        return rewinder.pick(agent, asked);
    };
    let kept = rewinder
        .agents
        .get(agent)
        .ok()
        .and_then(|(_, checkpoints)| checkpoints.list.iter().find(|kept| kept.turn == turn));
    let Some(message) = kept.map(|kept| kept.message) else {
        let why = format!("No checkpoint {turn}.");
        rewinder.notices.write(Notice::error(agent, why));
        return;
    };
    let scope = asked.scope;
    rewinder.commands.spawn((
        Rewind {
            agent,
            message,
            scope,
        },
        ByCommand,
    ));
}

/// Acts on each [`Rewind`] as it is spawned.
fn requested(added: On<Add<Rewind>>, mut rewinder: Rewinder) {
    rewinder.run(added.entity, false);
}

/// Rewinds once the user said yes to undoing their edits.
fn confirmed(asks: Query<(&Outcome, &ForAgent, &Confirm), Added<Outcome>>, mut rewinder: Rewinder) {
    for (outcome, &ForAgent(agent), &Confirm(request)) in &asks {
        if Answer::of(outcome).is_some_and(|answer| answer.is("yes")) {
            rewinder.run(request, true);
        } else if let Ok((_, by_command)) = rewinder.requests.get(request) {
            rewinder.notices.write(Notice::info(agent, "Not rewound."));
            rewinder.end(request, by_command, Rewound::Declined);
        }
    }
}

/// Deletes a deleted session's checkpoints, in the repository its record
/// names.
fn prune(mut deleted: MessageReader<SessionDeleted>, repo: Repo) {
    for SessionDeleted(id) in deleted.read() {
        let record = repo.record(id.as_str());
        let Ok(root) = fs::read_to_string(&record) else {
            continue;
        };
        let git = Git {
            root: PathBuf::from(root.trim_end()),
            launcher: repo.processes.launcher(),
        };
        let prefix = format!("{REFS}/{id}/");
        IoTaskPool::get_or_init(TaskPool::default)
            .spawn(async move {
                let listed = git.run(None, &["for-each-ref", "--format=%(refname)", &prefix]);
                for reference in listed.unwrap_or_default().lines() {
                    git.run(None, &["update-ref", "-d", reference]).ok();
                }
                fs::remove_file(record).ok();
            })
            .detach();
    }
}

/// The user's own message of the turn that began at `start`, and what
/// they typed in it and any they sent after it: plugins' entries, tool
/// results and the stopped note are left out.
fn user_text(conversation: &Conversation, start: MessageId) -> Option<(MessageId, String)> {
    let own: Vec<EntryRef<'_>> = conversation
        .asked(start)
        .filter(EntryRef::is_users_own)
        .collect();
    let id = own.first()?.id;
    let texts: Vec<&str> = own
        .iter()
        .filter_map(|entry| match entry.message {
            message::Message::User { content } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    (!texts.is_empty()).then(|| (id, texts.join("\n")))
}

/// Takes the snapshot of turn `index`: a new ref, or the last one when the
/// files did not change since.
fn take(
    git: &Git,
    session: &str,
    record: &Path,
    index: u64,
    previous: Option<(String, String)>,
    settled: Option<&str>,
) -> Result<Option<Snap>, String> {
    let Some(tree) = git.working_tree()? else {
        return Ok(None);
    };
    let user_edits = settled
        .map(|settled| git.changed(settled, &tree))
        .unwrap_or_default();
    let reference = match previous {
        Some((last, reference)) if last == tree => reference,
        _ => {
            let message = format!("rig checkpoint: session {session}, turn {index}");
            let commit = git.run(None, &["commit-tree", &tree, "-m", &message])?;
            let reference = format!("{REFS}/{session}/{index}");
            git.run(None, &["update-ref", &reference, &commit])?;
            let top = git.run(None, &["rev-parse", "--show-toplevel"])?;
            if let Some(parent) = record.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(record, top).ok();
            reference
        }
    };
    Ok(Some(Snap {
        tree,
        reference,
        user_edits,
    }))
}

/// Git in the project, started as every child process is.
#[derive(Clone)]
struct Git {
    root: PathBuf,
    launcher: Launcher,
}

impl Git {
    /// Runs git with `index` as its index when given; its output.
    fn run(&self, index: Option<&Path>, args: &[&str]) -> Result<String, String> {
        let mut command: Command = self.launcher.command("git");
        command
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .env("GIT_AUTHOR_NAME", "rig")
            .env("GIT_AUTHOR_EMAIL", "rig@localhost")
            .env("GIT_COMMITTER_NAME", "rig")
            .env("GIT_COMMITTER_EMAIL", "rig@localhost");
        if let Some(index) = index {
            command.env("GIT_INDEX_FILE", index);
        }
        let output = command.output().map_err(|failure| failure.to_string())?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout)
                .trim_end()
                .to_owned())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
        }
    }

    /// Runs `work` with a fresh temporary index in the repository's git
    /// directory, removed afterwards.
    fn with_index<T>(&self, work: impl FnOnce(&Path) -> Result<T, String>) -> Result<T, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let number = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("rig-checkpoint-{}-{number}.index", std::process::id());
        let index = self
            .root
            .join(self.run(None, &["rev-parse", "--git-path", &name])?);
        let done = work(&index);
        fs::remove_file(&index).ok();
        done
    }

    /// The working tree as a git tree: the tracked files and the untracked
    /// ones that are not ignored. `None` outside a git repository.
    fn working_tree(&self) -> Result<Option<String>, String> {
        if self.run(None, &["rev-parse", "--git-dir"]).is_err() {
            return Ok(None);
        }
        let head = self
            .run(None, &["rev-parse", "--verify", "-q", "HEAD"])
            .is_ok();
        self.with_index(|index| {
            if head {
                self.run(Some(index), &["read-tree", "HEAD"])?;
            }
            self.run(Some(index), &["add", "-A"])?;
            self.run(Some(index), &["write-tree"])
        })
        .map(Some)
    }

    /// The paths that differ between the trees `from` and `to`, or that
    /// only one has, with `filter` (`--diff-filter`).
    fn diff(&self, from: &str, to: &str, filter: &str) -> Vec<String> {
        let filter = format!("--diff-filter={filter}");
        let args = [
            "diff-tree",
            "-r",
            "--name-only",
            "--no-renames",
            &filter,
            from,
            to,
        ];
        self.run(None, &args)
            .map(|listed| listed.lines().map(str::to_owned).collect())
            .unwrap_or_default()
    }

    fn changed(&self, from: &str, to: &str) -> Vec<String> {
        self.diff(from, to, "ADMT")
    }

    /// Makes the working tree, now `current`, hold `target`'s files:
    /// removes the files `target` lacks and writes those that differ.
    fn restore(&self, current: &str, target: &str) -> Result<(), String> {
        let top = Self {
            root: PathBuf::from(self.run(None, &["rev-parse", "--show-toplevel"])?),
            launcher: self.launcher.clone(),
        };
        for added in self.diff(target, current, "A") {
            fs::remove_file(top.root.join(added)).map_err(|failure| failure.to_string())?;
        }
        let differing = self.diff(target, current, "DMT");
        if differing.is_empty() {
            return Ok(());
        }
        top.with_index(|index| {
            top.run(Some(index), &["read-tree", target])?;
            let mut args = vec!["checkout-index", "-f", "--"];
            args.extend(differing.iter().map(String::as_str));
            top.run(Some(index), &args).map(drop)
        })
    }
}

#[cfg(test)]
mod tests;
