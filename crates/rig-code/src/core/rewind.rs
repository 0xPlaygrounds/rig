//! Rewind and fork. Every model call of an agent's turn leaves a
//! [`Checkpoint`] in its [`History`]: the call's effect id, how much of the
//! conversation it sent, which compaction was in force, and, when the app
//! has [`Snapshots`], a snapshot of the working tree taken as the call was
//! made. [`Rewind`] puts the agent back at a checkpoint, conversation and
//! files; [`Fork`] clones it there into a new agent and leaves the original
//! as it is. A rewind can be undone with [`UndoRewind`].
//!
//! A checkpoint at the first call of a turn stands for the moment before
//! the user's message: going back there takes the message out and hands its
//! text back to the views as [`Recalled`], to edit and send again. One in
//! the middle of a turn keeps the tool results the call sent; `/retry` sends
//! them again, or a message typed then goes with them.
//!
//! The core only asks for snapshots through [`FileSnapshots`]; the host
//! provides one kept in a git object store outside the project.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::oneshot;
use rig_core::completion::{AssistantContent, Message};
use rig_core::effect::EffectId;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

use super::agent::{
    ActiveTurn, Agent, AgentId, CallOf, Conversation, Effort, Focus, ModelChoice, Notice,
    SystemPrompt, ToolAccess, TurnOf,
};
use super::approval::Policy;
use super::calls::{Done, Running, Wake};
use super::compaction::Compacted;
use super::inbox::Recalled;
use super::save::ReflectSaved;
use super::usage::Spending;

/// Characters of a message shown in a checkpoint's label.
const LABEL_CHARS: usize = 72;

/// Where an agent can go back to: one entry per model call of its turns,
/// oldest first, and the compactions in force at them. Saved with the
/// session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Default,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    Saved
)]
pub struct History {
    /// One per answered model call, in the order they were made.
    pub checkpoints: Vec<Checkpoint>,
    /// Each [`Compacted`] the agent had when a checkpoint was made, the
    /// first one first; a checkpoint names how many were made by then.
    pub compactions: Vec<Compacted>,
}

/// The agent as one model call found it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The model call's effect id, as in `effects.jsonl`.
    pub effect: u64,
    /// How many of the conversation's messages the call sent.
    pub messages: usize,
    /// How many of [`History::compactions`] had been made: the last of
    /// them was in force, or none when 0.
    pub compactions: usize,
    /// The working tree's snapshot taken as the call was made, when the app
    /// keeps them.
    pub files: Option<String>,
    /// When the call was made, in seconds since the Unix epoch.
    pub at: u64,
}

impl History {
    /// A checkpoint for the call `effect` that sends the first `messages`
    /// messages under `compacted`, recording `compacted` when it is new.
    /// It is kept with [`Self::record`] once the call is answered.
    pub(crate) fn checkpoint(
        &mut self,
        effect: EffectId,
        messages: usize,
        compacted: &Compacted,
    ) -> Checkpoint {
        let known = self.compactions.last().map_or_else(
            || compacted.summary.is_empty() && compacted.upto == 0,
            |last| last.upto == compacted.upto && last.summary == compacted.summary,
        );
        if !known {
            self.compactions.push(compacted.clone());
        }
        Checkpoint {
            effect: effect.as_u64(),
            messages,
            compactions: self.compactions.len(),
            files: None,
            at: now(),
        }
    }

    /// Keeps `checkpoint`, with the snapshot taken for it.
    pub(crate) fn record(&mut self, checkpoint: Checkpoint) {
        self.checkpoints.push(checkpoint);
    }

    /// The position of the checkpoint of the call `effect`.
    fn position(&self, effect: u64) -> Option<usize> {
        self.checkpoints
            .iter()
            .position(|checkpoint| checkpoint.effect == effect)
    }

    /// The history as it was when the checkpoint at `position` was made:
    /// the checkpoints before it, the compactions made by then, and the
    /// compaction in force.
    fn before(&self, position: usize) -> (Self, Compacted) {
        let Some(checkpoint) = self.checkpoints.get(position) else {
            return (self.clone(), Compacted::default());
        };
        let compactions: Vec<Compacted> = self
            .compactions
            .iter()
            .take(checkpoint.compactions)
            .cloned()
            .collect();
        let compacted = compactions.last().cloned().unwrap_or_default();
        let history = Self {
            checkpoints: self.checkpoints.iter().take(position).cloned().collect(),
            compactions,
        };
        (history, compacted)
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// What a snapshot store did to put the files back.
#[derive(Clone, Debug)]
pub struct Restored {
    /// A snapshot of the files as they were before, to undo with.
    pub before: String,
    /// Files written back.
    pub written: usize,
    /// Files removed, which the snapshot did not have.
    pub removed: usize,
}

/// Takes and restores snapshots of the working tree. Both block; the core
/// runs them on threads of their own. Ids are opaque to the core.
pub trait FileSnapshots: Send + Sync + 'static {
    /// Records the files as they are now and returns the snapshot's id.
    fn take(&self) -> Result<String, String>;

    /// Puts the files back as the snapshot `id` had them: what changed
    /// since is written back, and what is new is removed.
    fn restore(&self, id: &str) -> Result<Restored, String>;
}

/// The app's snapshot store. Without this resource checkpoints have no
/// files, and rewinds leave the files alone.
#[derive(Resource, Clone)]
pub struct Snapshots {
    store: Arc<dyn FileSnapshots>,
    /// Whether a failed snapshot was reported, so a broken store says so
    /// once rather than at every call.
    reported: Arc<AtomicBool>,
}

impl Snapshots {
    /// The store `snapshots`.
    pub fn new(snapshots: impl FileSnapshots) -> Self {
        Self {
            store: Arc::new(snapshots),
            reported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether a failed snapshot is the first: only that one is reported.
    pub(crate) fn first_failure(&self) -> bool {
        !self.reported.swap(true, Ordering::Relaxed)
    }

    /// Takes a snapshot on a thread of its own.
    pub(crate) fn take(&self) -> impl Future<Output = Result<String, String>> + Send + use<> {
        let store = self.store.clone();
        off_thread(move || store.take())
    }

    /// Restores the snapshot `id` on a thread of its own.
    fn restore(&self, id: String) -> impl Future<Output = Result<Restored, String>> + Send + use<> {
        let store = self.store.clone();
        off_thread(move || store.restore(&id))
    }
}

/// Runs `work` on a new thread and awaits its answer.
async fn off_thread<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let (sender, receiver) = oneshot::channel();
    std::thread::Builder::new()
        .name("rig-code-snapshot".to_owned())
        .spawn(move || {
            sender.send(work()).ok();
        })
        .map_err(|error| format!("could not start a thread: {error}"))?;
    receiver
        .await
        .map_err(|_| "the snapshot thread crashed".to_owned())?
}

/// Put the agent back at the checkpoint of the model call `to`: the
/// conversation as that call found it, and, with `files`, the working tree
/// too. Refused while a turn runs.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Rewind {
    /// The agent.
    pub entity: Entity,
    /// The checkpoint's effect id.
    pub to: u64,
    /// Whether the files go back too.
    pub files: bool,
}

/// Undo the agent's last rewind: its conversation, and its files when the
/// rewind restored them. Undoing again redoes it.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct UndoRewind {
    /// The agent.
    pub entity: Entity,
}

/// Clone the agent into a new one, at the checkpoint of the model call
/// `at`, or as it is with `None`. The new agent has the same model,
/// reasoning setting, prompt and tools, a conversation of its own and the
/// views' focus; the files are shared and stay as they are.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Fork {
    /// The agent.
    pub entity: Entity,
    /// The checkpoint's effect id, or `None` for now.
    pub at: Option<u64>,
}

/// Saved on a forked agent: the stable id of the agent it was cloned from
/// and the checkpoint it was cloned at.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug, Saved)]
pub struct Forked {
    /// The [`AgentId`] of the original.
    pub from: String,
    /// The checkpoint's effect id, or `None` when cloned as it was.
    pub at: Option<u64>,
}

/// What a rewind replaced, to undo it with. Not saved: an undo is for the
/// moments after a rewind.
#[derive(Component)]
pub struct Undo {
    conversation: Conversation,
    compacted: Compacted,
    history: History,
    /// The snapshot of the files before the rewind restored them.
    files: Option<String>,
}

/// What restoring files returns.
pub type FilesRestored = Result<Restored, String>;

/// A call of a rewind's turn restoring the files. The turn keeps the agent
/// busy, so nothing writes files meanwhile.
#[derive(Component, Debug)]
pub struct RestoringFiles {
    /// Whether this restore undoes a rewind.
    undoing: bool,
}

/// One place to go back to, for views and commands.
#[derive(Clone, Debug)]
pub struct Point {
    /// The checkpoint's effect id.
    pub effect: u64,
    /// The message the turn started with, or the tools whose results the
    /// call sent, and how long ago.
    pub label: String,
    /// Whether it is the start of a turn: going back there takes the
    /// user's message out.
    pub turn_start: bool,
    /// Whether it has a snapshot of the files.
    pub files: bool,
}

/// The agent's checkpoints, newest first, labelled.
pub fn points(conversation: &Conversation, history: &History) -> Vec<Point> {
    let now = now();
    history
        .checkpoints
        .iter()
        .rev()
        .map(|checkpoint| {
            let (_, recalled) = cut(&conversation.0, checkpoint.messages);
            let turn_start = recalled.is_some();
            let what = match recalled {
                Some(text) => format!("› {}", one_line(&text)),
                None => format!("  ⋯ {}", step_label(&conversation.0, checkpoint.messages)),
            };
            let files = if checkpoint.files.is_some() {
                " · files"
            } else {
                ""
            };
            Point {
                effect: checkpoint.effect,
                label: format!(
                    "{what}  ({}{files})",
                    age(now.saturating_sub(checkpoint.at))
                ),
                turn_start,
                files: checkpoint.files.is_some(),
            }
        })
        .collect()
}

/// The first line of `text`, cut to [`LABEL_CHARS`].
fn one_line(text: &str) -> String {
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let mut cut: String = line.trim().chars().take(LABEL_CHARS).collect();
    if line.trim().chars().count() > LABEL_CHARS {
        cut.push('…');
    }
    cut
}

/// What a checkpoint inside a turn follows: the tools of the reply whose
/// results its call sent.
fn step_label(conversation: &[Message], messages: usize) -> String {
    let reply = messages
        .checked_sub(2)
        .and_then(|index| conversation.get(index));
    let tools: Vec<&str> = match reply {
        Some(Message::Assistant(reply)) => reply
            .content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    if tools.is_empty() {
        "a step of the turn".to_owned()
    } else {
        format!("after {}", tools.join(", "))
    }
}

/// `seconds` as a short age.
fn age(seconds: u64) -> String {
    match seconds {
        0..60 => "just now".to_owned(),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..86_400 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

/// How many messages to keep to go back to a checkpoint that sent the
/// first `messages`, and the user's text taken out: a checkpoint that sent
/// a message the user typed (not tool results) goes back to before it.
fn cut(conversation: &[Message], messages: usize) -> (usize, Option<String>) {
    let messages = messages.min(conversation.len());
    let Some(index) = messages.checked_sub(1) else {
        return (0, None);
    };
    match conversation.get(index) {
        Some(Message::User { content })
            if !content
                .iter()
                .any(|item| matches!(item, UserContent::ToolResult(_))) =>
        {
            let text: Vec<&str> = content
                .iter()
                .filter_map(|item| match item {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            (index, Some(text.join("\n\n")))
        }
        _ => (messages, None),
    }
}

/// The agent at the checkpoint at `position` of its history: its
/// conversation, compaction and history then, and the text taken out.
fn at_checkpoint(
    conversation: &Conversation,
    history: &History,
    position: usize,
) -> Option<(Conversation, Compacted, History, Option<String>)> {
    let checkpoint = history.checkpoints.get(position)?;
    let (keep, recalled) = cut(&conversation.0, checkpoint.messages);
    let (history, mut compacted) = history.before(position);
    // A compaction never reaches past what it was made over.
    compacted.upto = compacted.upto.min(keep);
    let conversation = Conversation(conversation.0.iter().take(keep).cloned().collect());
    Some((conversation, compacted, history, recalled))
}

/// What the rewind observers read of an agent.
type RewindQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Conversation,
        &'static mut Compacted,
        &'static mut History,
        &'static mut Spending,
        Has<ActiveTurn>,
    ),
    With<Agent>,
>;

/// Puts the agent back at a checkpoint, and starts restoring its files
/// when asked and the checkpoint has them.
pub(crate) fn on_rewind(
    rewind: On<Rewind>,
    mut agents: RewindQuery,
    snapshots: Option<Res<Snapshots>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = rewind.entity;
    let Ok((mut conversation, mut compacted, mut history, mut spent, busy)) = agents.get_mut(agent)
    else {
        return;
    };
    if busy {
        notices.write(Notice::info(
            agent,
            "A turn is running. Press Esc to stop it, then /rewind.",
        ));
        return;
    }
    let Some(position) = history.position(rewind.to) else {
        notices.write(Notice::error(
            agent,
            format!("No checkpoint at effect {}; /rewind lists them.", rewind.to),
        ));
        return;
    };
    let files = history
        .checkpoints
        .get(position)
        .and_then(|checkpoint| checkpoint.files.clone());
    let Some((back, back_compacted, back_history, text)) =
        at_checkpoint(&conversation, &history, position)
    else {
        return;
    };
    let removed = conversation.0.len().saturating_sub(back.0.len());
    let undo = Undo {
        conversation: std::mem::replace(&mut *conversation, back),
        compacted: std::mem::replace(&mut *compacted, back_compacted),
        history: std::mem::replace(&mut *history, back_history),
        files: None,
    };
    spent.context = Some(compacted.estimate(&conversation.0));
    commands.entity(agent).insert(undo);
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        recalled.write(Recalled { agent, text });
    }
    let mut lines = vec![format!(
        "Rewound: {removed} message{} taken out. /rewind undo puts {} back.",
        if removed == 1 { "" } else { "s" },
        if removed == 1 { "it" } else { "them" }
    )];
    match (rewind.files, files, snapshots) {
        (false, ..) => lines.push("The files were left as they are.".to_owned()),
        (true, None, _) => lines.push(
            "The files were left as they are: the checkpoint has no snapshot of them.".to_owned(),
        ),
        (true, Some(_), None) => {
            lines.push("The files were left as they are: this app keeps no snapshots.".to_owned())
        }
        (true, Some(files), Some(snapshots)) => {
            restore_files(agent, &snapshots, files, false, &wake, &mut commands);
            lines.push("Restoring the files…".to_owned());
        }
    }
    notices.write(Notice::info(agent, lines.join(" ")));
}

/// Undoes the agent's last rewind; undoing it again redoes it.
pub(crate) fn on_undo_rewind(
    undo: On<UndoRewind>,
    mut agents: Query<
        (
            &mut Conversation,
            &mut Compacted,
            &mut History,
            &mut Spending,
            Option<&mut Undo>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    snapshots: Option<Res<Snapshots>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = undo.entity;
    let Ok((mut conversation, mut compacted, mut history, mut spent, saved, busy)) =
        agents.get_mut(agent)
    else {
        return;
    };
    if busy {
        notices.write(Notice::info(
            agent,
            "A turn is running. Press Esc to stop it, then /rewind undo.",
        ));
        return;
    }
    let Some(mut saved) = saved else {
        notices.write(Notice::info(agent, "No rewind to undo."));
        return;
    };
    std::mem::swap(&mut *conversation, &mut saved.conversation);
    std::mem::swap(&mut *compacted, &mut saved.compacted);
    std::mem::swap(&mut *history, &mut saved.history);
    spent.context = Some(compacted.estimate(&conversation.0));
    let files = saved.files.take();
    let mut text = "Rewind undone; /rewind undo again redoes it.".to_owned();
    if let (Some(files), Some(snapshots)) = (files, snapshots) {
        restore_files(agent, &snapshots, files, true, &wake, &mut commands);
        text.push_str(" Restoring the files…");
    }
    notices.write(Notice::info(agent, text));
}

/// Starts restoring the snapshot `files` in a turn of the agent's own.
fn restore_files(
    agent: Entity,
    snapshots: &Snapshots,
    files: String,
    undoing: bool,
    wake: &Wake,
    commands: &mut Commands,
) {
    let turn = commands
        .spawn((Name::new("restoring files"), TurnOf(agent)))
        .id();
    commands.spawn((
        Name::new("restoring files"),
        RestoringFiles { undoing },
        Running::spawn(
            IoTaskPool::get_or_init(TaskPool::default),
            wake,
            snapshots.restore(files),
        ),
        CallOf(turn),
    ));
}

/// Reports restored files, keeps the snapshot of the files before so the
/// rewind can be undone, and ends the rewind's turn.
pub(crate) fn on_files_restored(
    done: On<Add<Done<FilesRestored>>>,
    calls: Query<(&CallOf, &RestoringFiles, &Done<FilesRestored>)>,
    turns: Query<&TurnOf>,
    mut undos: Query<&mut Undo>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = done.entity;
    let Ok((&CallOf(turn), restoring, Done(restored))) = calls.get(call) else {
        return;
    };
    let agent = turns.get(turn).ok().map(|of| of.0);
    commands.entity(turn).despawn();
    let Some(agent) = agent else {
        return;
    };
    match restored {
        Ok(restored) => {
            if let Ok(mut undo) = undos.get_mut(agent) {
                undo.files = Some(restored.before.clone());
            }
            let what = if restored.written + restored.removed == 0 {
                "The files were already as the checkpoint had them.".to_owned()
            } else {
                format!(
                    "Files restored{}: {} written back, {} removed.",
                    if restoring.undoing {
                        " as they were before the rewind"
                    } else {
                        ""
                    },
                    restored.written,
                    restored.removed
                )
            };
            notices.write(Notice::info(agent, what));
        }
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Restoring the files failed: {why}. The conversation was rewound."),
            ));
        }
    }
}

/// What a fork copies of the original.
type ForkQuery<'w, 's> = Query<
    'w,
    's,
    (
        (&'static AgentId, &'static Conversation, &'static Compacted),
        &'static History,
        Option<&'static ModelChoice>,
        &'static Effort,
        &'static SystemPrompt,
        &'static ToolAccess,
    ),
    With<Agent>,
>;

/// Clones the agent into a new one, at a checkpoint or as it is, and asks
/// the views to show the new one.
pub(crate) fn on_fork(
    fork: On<Fork>,
    agents: ForkQuery,
    policies: Query<&Policy>,
    mut commands: Commands,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = fork.entity;
    let Ok(((id, conversation, compacted), history, model, effort, prompt, access)) =
        agents.get(agent)
    else {
        return;
    };
    let (conversation, compacted, history, text) = match fork.at {
        None => (
            conversation.clone(),
            compacted.clone(),
            history.clone(),
            None,
        ),
        Some(at) => {
            let Some(found) = history
                .position(at)
                .and_then(|position| at_checkpoint(conversation, history, position))
            else {
                notices.write(Notice::error(
                    agent,
                    format!("No checkpoint at effect {at}; /fork lists them."),
                ));
                return;
            };
            found
        }
    };
    let short = id.0.get(..8).unwrap_or(&id.0).to_owned();
    let messages = conversation.0.len();
    let spent = Spending {
        context: Some(compacted.estimate(&conversation.0)),
        ..Spending::default()
    };
    let mut forked = commands.spawn((
        Name::new(format!("fork of {short}")),
        Agent,
        Forked {
            from: id.0.clone(),
            at: fork.at,
        },
        conversation,
        compacted,
        history,
        spent,
        *effort,
        prompt.clone(),
        access.clone(),
    ));
    if let Some(model) = model {
        forked.insert(model.clone());
    }
    if let Ok(policy) = policies.get(agent) {
        forked.insert(policy.clone());
    }
    let forked = forked.id();
    commands.trigger(Focus { entity: forked });
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        recalled.write(Recalled {
            agent: forked,
            text,
        });
    }
    notices.write(Notice::info(
        forked,
        format!(
            "Forked from agent {short} with {messages} message{}. The files are shared; /agents \
             switches between the two.",
            if messages == 1 { "" } else { "s" }
        ),
    ));
}
