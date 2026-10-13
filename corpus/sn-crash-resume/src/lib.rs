//! Recovers a turn a crash cut short. While a turn runs, its progress is
//! kept in a file of the session's store, `crash-resume/<agent>.json`:
//! the text the model streamed so far (at least every second), each tool
//! call as it is about to start, and each result as it settles. A turn
//! that ends removes the file, so a file found on resume is a turn that
//! never finished.
//!
//! On resume the kernel's restore has already settled what it could (a
//! changing call without a result is answered as interrupted and the turn
//! carries on); this plugin stops that carried-on turn, shows an
//! "Interrupted turn" entry in the transcript and asks what to do:
//!
//! - **Continue**: each call gets its real result, else "may or may not
//!   have completed" (started) or "not run (interrupted)" (never started),
//!   the partial text is kept marked incomplete, and the next request is
//!   sent;
//! - **Discard**: the conversation goes back to before the turn's user
//!   message, which goes back in the input;
//! - **Leave**: nothing changes; the question comes back on the next
//!   message, which waits for the answer.
//!
//! Subagents the crash interrupted are listed in their parent's entry and
//! follow its choice. A run nobody can answer, such as `--resume --print`,
//! continues. A torn last line in an agent's log is reported as a warning
//! by the kernel's restore, which sets it aside.

use std::collections::HashMap;

use rig_harness::note::Note;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::pending_calls;
use rig_harness::rig_core::message::{Message as Chat, ToolResult as Answered};
use rig_harness::rig_ecs::journal::SessionStore;
use rig_tui::TuiInput;
use serde::{Deserialize, Serialize};

/// The author (a plugin name) of the transcript note, of the message that
/// asks the model to carry on and of the plugin's edits.
pub const ORIGIN: &str = "crash-resume";
/// The title of the transcript note and of the question.
pub const TITLE: &str = "Interrupted turn";
/// The result of a call that started and has no result.
pub const UNKNOWN: &str =
    "interrupted by a crash; it may or may not have completed — check before retrying";
/// The result of a call that never started.
pub const NOT_RUN: &str = "not run (interrupted)";
/// What follows the partial text the turn kept.
pub const INCOMPLETE: &str = "[incomplete: interrupted by a crash]";
/// What asks the model to carry on after its kept partial text.
const CARRY_ON: &str =
    "Your last reply was cut off by a crash after the text above; carry on from where it stopped.";
/// How often the streamed text is saved, at least.
const TEXT_EVERY: Duration = Duration::from_secs(1);

/// What a running turn did so far, as saved.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Progress {
    /// The message the turn began at.
    pub first_message: Option<MessageId>,
    /// The text its current model call streamed so far.
    pub text: String,
    /// The calls of its current reply, in call order.
    pub calls: Vec<CallProgress>,
}

/// One tool call of a running turn.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CallProgress {
    /// The call.
    pub call: ToolCall,
    /// Whether it was about to start, or started.
    pub started: bool,
    /// Its result as the model reads it, once it settled.
    pub result: Option<Answered>,
    /// Its output as text, once it settled.
    pub output: Option<String>,
}

/// The name of an agent's progress file in the session's store.
pub fn progress_file(id: &AgentId) -> String {
    format!("{ORIGIN}/{}.json", id.0)
}

/// The saved progress of `id`'s unfinished turn, if any.
pub fn read_progress(store: &dyn Files, id: &AgentId) -> Option<Progress> {
    let bytes = store.read_file(&progress_file(id)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// What the user chose for an interrupted turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recover {
    /// Answer its calls and carry on.
    Continue,
    /// Take it back.
    Discard,
    /// Not now.
    Leave,
}

/// On an agent whose last turn a crash cut short, until it is recovered.
#[derive(Component, Debug)]
pub struct Interrupted {
    /// What the turn did.
    pub progress: Progress,
    /// The transcript note showing it, an unsaved aside of the agent's
    /// conversation.
    pub entry: Option<AsideId>,
    ask: Option<Entity>,
    choice: Option<Recover>,
    left: bool,
}

/// On an agent while a turn runs: what it did so far.
#[derive(Component, Default)]
struct Tracked(Progress);

/// On the question about an agent's interrupted turn.
#[derive(Component)]
struct CrashAsk(Entity);

/// A hold on a user's message to an agent with an interrupted turn.
#[derive(Component)]
struct CrashHold(Entity);

/// On a message this plugin holds or let through.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Seen;

/// Keeps each turn's progress and recovers a turn a crash cut short.
#[derive(Default)]
pub struct CrashResumePlugin;

impl Plugin for CrashResumePlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(turn_started)
            .add_observer(turn_ended)
            .add_observer(model_call_started)
            .add_observer(calls_spawned)
            .add_observer(call_settled)
            .add_systems(PostStartup, recover)
            .add_systems(PostUpdate, save_text.run_if(on_real_timer(TEXT_EVERY)))
            .add_systems(
                AgentPass,
                (
                    mark_started.in_set(GateSystems::After),
                    (take_answers, ask, hold_messages, resolve)
                        .chain()
                        .in_set(Stage::Gate)
                        .after(AskSystems),
                ),
            );
    }
}

/// Writes `progress` as `id`'s, or removes it with `None`. A failed write
/// only loses what a crash would show.
fn save(store: Option<&SessionStore>, id: &AgentId, progress: Option<&Progress>) {
    let Some(SessionStore(store)) = store else {
        return;
    };
    let name = progress_file(id);
    match progress.map(serde_json::to_vec) {
        Some(Ok(bytes)) => store.write_file(&name, &bytes).ok(),
        Some(Err(_)) => None,
        None => store.remove_file(&name).ok(),
    };
}

/// Starts tracking a new turn, and keeps the loop awake while it runs so
/// its text is saved on time. A turn a restore carries on is left to
/// [`recover`].
fn turn_started(
    added: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, &Turn), Without<Restored>>,
    ids: Query<&AgentId>,
    store: Option<Res<SessionStore>>,
    mut commands: Commands,
) {
    let Ok((&TurnOf(agent), turn)) = turns.get(added.entity) else {
        return;
    };
    let progress = Progress {
        first_message: Some(turn.first_message),
        ..Progress::default()
    };
    if let Ok(id) = ids.get(agent) {
        save(store.as_deref(), id, Some(&progress));
    }
    commands.entity(agent).try_insert(Tracked(progress));
    commands
        .entity(added.entity)
        .try_insert(KeepAwake(TEXT_EVERY));
}

/// A turn that ended has nothing to recover.
fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, (With<Turn>, Without<Restored>)>,
    ids: Query<&AgentId, With<Tracked>>,
    store: Option<Res<SessionStore>>,
    mut commands: Commands,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity)
        && let Ok(id) = ids.get(agent)
    {
        save(store.as_deref(), id, None);
        commands.entity(agent).try_remove::<Tracked>();
    }
}

/// A new model call of a turn: the last reply and its results are in the
/// conversation now.
fn model_call_started(
    started: On<Add<Started>>,
    calls: Query<&ForAgent, With<TurnCall>>,
    mut agents: Query<(&AgentId, &mut Tracked)>,
    store: Option<Res<SessionStore>>,
) {
    if let Ok(&ForAgent(agent)) = calls.get(started.entity)
        && let Ok((id, mut tracked)) = agents.get_mut(agent)
        && tracked.0 != Progress::default()
    {
        tracked.0 = Progress {
            first_message: tracked.0.first_message,
            ..Progress::default()
        };
        save(store.as_deref(), id, Some(&tracked.0));
    }
}

/// A reply's calls, as they are spawned; its text is in the conversation
/// now.
fn calls_spawned(
    added: On<Add<ToolRequest>>,
    calls: Query<(&ToolRequest, &ForAgent), (With<CallOf>, Without<Restored>)>,
    mut agents: Query<(&AgentId, &mut Tracked)>,
    store: Option<Res<SessionStore>>,
) {
    let Ok((request, &ForAgent(agent))) = calls.get(added.entity) else {
        return;
    };
    let Ok((id, mut tracked)) = agents.get_mut(agent) else {
        return;
    };
    if tracked
        .0
        .calls
        .iter()
        .any(|known| known.call.id == request.call.id)
    {
        return;
    }
    tracked.0.text.clear();
    tracked.0.calls.push(CallProgress {
        call: request.call.clone(),
        started: false,
        result: None,
        output: None,
    });
    save(store.as_deref(), id, Some(&tracked.0));
}

/// Marks each call about to start, before it starts, once every gate had
/// its say.
fn mark_started(
    calls: Query<(&ToolRequest, &ForAgent, &Decision), (Undecided, ToolCalls, Without<Holds>)>,
    mut agents: Query<(&AgentId, &mut Tracked)>,
    store: Option<Res<SessionStore>>,
) {
    for (request, &ForAgent(agent), decision) in &calls {
        let Ok((id, mut tracked)) = agents.get_mut(agent) else {
            continue;
        };
        let known = tracked.0.calls.iter_mut();
        let Some(call) = known
            .filter(|known| known.call.id == request.call.id)
            .find(|known| !known.started)
        else {
            continue;
        };
        if !decision.is_decided() {
            call.started = true;
            save(store.as_deref(), id, Some(&tracked.0));
        }
    }
}

/// Keeps each call's result as it settles.
fn call_settled(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome, &ForAgent), Without<SupersededBy>>,
    mut agents: Query<(&AgentId, &mut Tracked)>,
    store: Option<Res<SessionStore>>,
) {
    let Ok((request, outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok((id, mut tracked)) = agents.get_mut(agent) else {
        return;
    };
    let known = tracked.0.calls.iter_mut();
    if let Some(call) = known
        .into_iter()
        .find(|known| known.call.id == request.call.id)
    {
        let result = outcome.tool_result();
        call.output = Some(result.output().render());
        call.result = Some(request.call.answer(&result));
        call.started = true;
        save(store.as_deref(), id, Some(&tracked.0));
    }
}

/// Saves the text each turn's model call streamed so far.
fn save_text(
    calls: Query<(&Partial, &ForAgent), (With<TurnCall>, Without<Settled>)>,
    mut agents: Query<(&AgentId, &mut Tracked)>,
    store: Option<Res<SessionStore>>,
) {
    for (partial, &ForAgent(agent)) in &calls {
        if let Ok((id, mut tracked)) = agents.get_mut(agent)
            && tracked.0.text != partial.text
        {
            tracked.0.text.clone_from(&partial.text);
            save(store.as_deref(), id, Some(&tracked.0));
        }
    }
}

/// On resume: each agent with a saved unfinished turn is [`Interrupted`],
/// the turn the restore carries on for it is stopped, and its note,
/// listing its interrupted subagents too, goes in the transcript.
fn recover(
    mut agents: Query<(
        Entity,
        &AgentId,
        &mut Conversation,
        Option<&SpawnedBy>,
        Option<&Name>,
    )>,
    restored: Query<(Entity, &TurnOf), With<Restored>>,
    store: Option<Res<SessionStore>>,
    mut commands: Commands,
) {
    let Some(SessionStore(store)) = store.as_deref() else {
        return;
    };
    let found: HashMap<Entity, Progress> = agents
        .iter()
        .filter_map(|(agent, id, ..)| Some((agent, read_progress(&**store, id)?)))
        .collect();
    for (turn, &TurnOf(agent)) in &restored {
        if found.contains_key(&agent) {
            let stopped = TurnOutcome::Stopped(Some("recovering from a crash".to_owned()));
            commands.entity(turn).insert(stopped);
        }
    }
    let mut notes = Vec::new();
    for (agent, _, _, parent, _) in &agents {
        let Some(progress) = found.get(&agent) else {
            continue;
        };
        let mut text = describe(progress);
        for (child, _, _, of, name) in &agents {
            if of.is_some_and(|of| of.0 == agent)
                && let Some(progress) = found.get(&child)
            {
                let name = name.map_or("subagent", Name::as_str);
                let lines = describe(progress).replace('\n', "\n    ");
                text.push_str(&format!(
                    "\n  {name}, interrupted inside its call:\n    {lines}"
                ));
            }
        }
        let nested = parent.is_some_and(|parent| found.contains_key(&parent.0));
        notes.push((agent, progress, (!nested).then_some(text)));
    }
    for (agent, progress, text) in notes {
        let Ok((_, _, mut conversation, ..)) = agents.get_mut(agent) else {
            continue;
        };
        let entry = text.map(|text| {
            let after = conversation.messages().len().checked_sub(1);
            let after = after.and_then(|at| conversation.id(at));
            let note = Note::new(TITLE, text).aside(Author::plugin(ORIGIN), after);
            conversation.add_aside(note.unsaved())
        });
        commands.entity(agent).insert(Interrupted {
            progress: progress.clone(),
            entry,
            ask: None,
            choice: None,
            left: false,
        });
    }
}

/// What an interrupted turn did, for the transcript and the question.
fn describe(progress: &Progress) -> String {
    let mut lines = Vec::new();
    if !progress.text.trim().is_empty() {
        lines.push(format!(
            "Partial reply (incomplete): {}",
            progress.text.trim_end()
        ));
    }
    for known in &progress.calls {
        let call = &known.call.function;
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        let what = format!("{} {args}", call.name.as_str());
        lines.push(match (&known.output, known.started) {
            (Some(output), _) => format!("finished: {what}\n  result: {output}"),
            (None, true) => format!("unknown outcome: may have run: {what}"),
            (None, false) => format!("never started: {what}"),
        });
    }
    if lines.is_empty() {
        lines.push("The model had not answered yet.".to_owned());
    }
    lines.join("\n")
}

/// Reads each answer to a question about an interrupted turn.
fn take_answers(
    asks: Query<(&CrashAsk, &Outcome), Added<Outcome>>,
    mut agents: Query<&mut Interrupted>,
) {
    for (&CrashAsk(agent), outcome) in &asks {
        let Ok(mut interrupted) = agents.get_mut(agent) else {
            continue;
        };
        interrupted.ask = None;
        let answer = Answer::of(outcome);
        let is = |id| answer.as_ref().is_some_and(|answer| answer.is(id));
        interrupted.choice = Some(if is("continue") {
            Recover::Continue
        } else if is("discard") {
            Recover::Discard
        } else {
            Recover::Leave
        });
    }
}

/// The question about an interrupted turn: asked on resume and again with
/// the next message after Leave; a run nobody answers continues.
fn new_ask(agent: Entity, interrupted: &Interrupted) -> impl Bundle {
    let body = describe(&interrupted.progress);
    let ask = Ask::new(TITLE)
        .body(body)
        .choice("continue", "Continue")
        .choice("discard", "Discard")
        .choice("leave", "Leave");
    (ask, ForAgent(agent), CrashAsk(agent))
}

/// Asks about each interrupted turn that is not a subagent's inside its
/// parent's, unless it was left for now.
fn ask(
    mut agents: Query<(Entity, &mut Interrupted, Option<&SpawnedBy>)>,
    parents: Query<(), With<Interrupted>>,
    answerers: Answerers,
    mut commands: Commands,
) {
    for (agent, mut interrupted, parent) in &mut agents {
        let nested = parent.is_some_and(|parent| parents.contains(parent.0));
        if nested || interrupted.left || interrupted.ask.is_some() || interrupted.choice.is_some() {
            continue;
        }
        if answerers.can_answer(Some(agent)) {
            interrupted.ask = Some(commands.spawn(new_ask(agent, &interrupted)).id());
        } else {
            interrupted.choice = Some(Recover::Continue);
        }
    }
}

/// Holds each message the user sends an agent with an interrupted turn
/// until it is recovered, asking again after Leave.
fn hold_messages(
    messages: Query<
        (Entity, &Incoming, &ForAgent, &Decision),
        (Undecided, IncomingMessages, Without<Seen>),
    >,
    mut agents: Query<&mut Interrupted>,
    mut commands: Commands,
) {
    for (message, incoming, &ForAgent(agent), decision) in &messages {
        if incoming.author.kind != AuthorKind::User || decision.is_decided() {
            continue;
        }
        let Ok(mut interrupted) = agents.get_mut(agent) else {
            continue;
        };
        commands.entity(message).insert(Seen);
        let name = Name::new("waiting for the interrupted turn's recovery");
        commands.spawn((name, CrashHold(agent), HoldOf(message)));
        if interrupted.left {
            interrupted.left = false;
            interrupted.ask = Some(commands.spawn(new_ask(agent, &interrupted)).id());
        }
    }
}

/// Carries out each choice once its agent is idle, its subagents
/// following it, and lets the held messages go.
#[allow(clippy::too_many_arguments)]
fn resolve(
    mut agents: Query<(
        Entity,
        &AgentId,
        &mut Interrupted,
        &mut Conversation,
        Option<&SpawnedBy>,
        Has<ActiveTurn>,
    )>,
    holds: Query<(Entity, &CrashHold)>,
    mut input: Option<ResMut<TuiInput>>,
    store: Option<Res<SessionStore>>,
    mut commands: Commands,
) {
    let chosen: HashMap<Entity, Recover> = agents
        .iter()
        .filter_map(|(agent, _, interrupted, ..)| Some((agent, interrupted.choice?)))
        .collect();
    for (_, _, mut interrupted, _, parent, _) in &mut agents {
        if interrupted.choice.is_none()
            && let Some(&choice) = parent.and_then(|parent| chosen.get(&parent.0))
        {
            interrupted.choice = Some(choice);
        }
    }
    for (agent, id, mut interrupted, mut conversation, parent, busy) in &mut agents {
        let Some(choice) = interrupted.choice.filter(|_| !busy) else {
            continue;
        };
        let held: Vec<Entity> = holds
            .iter()
            .filter(|(_, hold)| hold.0 == agent)
            .map(|(hold, _)| hold)
            .collect();
        let title = match choice {
            Recover::Leave if held.is_empty() => {
                interrupted.choice = None;
                interrupted.left = true;
                continue;
            }
            Recover::Leave => "Interrupted turn (left as it was)",
            Recover::Continue => {
                let sends = held.is_empty();
                carry_on(&interrupted.progress, &mut conversation);
                if sends && conversation.awaits_model() {
                    commands.trigger(Retry { entity: agent });
                } else if sends {
                    let note = Deliver::new(agent, CARRY_ON, DeliveryMode::Queue);
                    commands.trigger(note.with_author(Author::plugin(ORIGIN)));
                }
                "Interrupted turn (continued)"
            }
            Recover::Discard => {
                let first = interrupted.progress.first_message;
                let at = first.map_or(conversation.messages().len(), |id| {
                    conversation.start_of(id)
                });
                // The user's own message is among the entries the turn
                // answers: results and notes come before it.
                let typed = first
                    .into_iter()
                    .flat_map(|id| conversation.asked(id))
                    .find_map(|entry| typed(&conversation, entry.index));
                conversation.splice(at.., [], Author::plugin(ORIGIN));
                if let (None, Some(text), Some(input)) = (parent, typed, input.as_mut()) {
                    input.set(text);
                }
                "Interrupted turn (discarded)"
            }
        };
        if let Some(id) = interrupted.entry
            && let Some(aside) = conversation.aside(id)
            && let Some(note) = Note::of(aside)
        {
            let retitled = Note::new(title, note.text).aside(aside.author.clone(), aside.after);
            conversation.replace_aside(id, retitled.unsaved());
        }
        save(store.as_deref(), id, None);
        for hold in held {
            commands.entity(hold).despawn();
        }
        commands.entity(agent).remove::<Interrupted>();
    }
}

/// Answers the interrupted reply's calls, each with its real result, else
/// as started or never started, and keeps the partial text of the model
/// call the crash cut short, marked incomplete.
fn carry_on(progress: &Progress, conversation: &mut Conversation) {
    let (from, calls) = interrupted_calls(conversation.messages(), progress);
    if !calls.is_empty() {
        let results = calls
            .iter()
            .map(|call| {
                let known = progress.calls.iter().find(|known| known.call.id == call.id);
                match known {
                    Some(CallProgress {
                        result: Some(result),
                        ..
                    }) => result.clone(),
                    Some(CallProgress { started: true, .. }) => failed(call, UNKNOWN),
                    _ => failed(call, NOT_RUN),
                }
            })
            .collect();
        conversation.splice(from.., [], Author::plugin(ORIGIN));
        conversation.push(Chat::tool_results(results));
        return;
    }
    // The text is the current model call's: a reply's calls clear it, and
    // the next model call starts it afresh. The conversation is halted, as
    // the restored turn was stopped, so its last message is what counts.
    let owed = matches!(conversation.messages().last(), Some(Chat::User { .. }));
    if !progress.text.trim().is_empty() && owed {
        let text = format!("{}\n\n{INCOMPLETE}", progress.text.trim_end());
        conversation.push(Chat::assistant(text));
    }
}

/// `call` answered with the error `why`.
fn failed(call: &ToolCall, why: &str) -> Answered {
    call.answer(&ToolResult::failed(ToolExecutionError::other(why)))
}

/// The calls of the conversation's last reply that the crash left
/// unanswered or the restore answered, and where their results begin:
/// its end when none are answered yet. Answered calls the turn's
/// `progress` does not track are an earlier reply's, whose real results
/// were committed before the next model call started: not interrupted.
fn interrupted_calls(messages: &[Chat], progress: &Progress) -> (usize, Vec<ToolCall>) {
    let pending = pending_calls(messages);
    if !pending.is_empty() {
        return (messages.len(), pending);
    }
    let [.., Chat::Assistant(reply), Chat::User { content }] = messages else {
        return (messages.len(), Vec::new());
    };
    let results_only = content
        .iter()
        .all(|item| matches!(item, UserContent::ToolResult(_)));
    let calls: Vec<ToolCall> = reply.tool_calls().cloned().collect();
    let tracked = calls.iter().any(|call| {
        let known = progress.calls.iter();
        known.into_iter().any(|known| known.call.id == call.id)
    });
    if results_only && tracked {
        (messages.len().saturating_sub(1), calls)
    } else {
        (messages.len(), Vec::new())
    }
}

#[cfg(test)]
mod tests;
