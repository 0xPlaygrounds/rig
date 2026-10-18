//! Recovers a turn a crash cut short.
//!
//! The kernel keeps what a running turn's calls did: a call that may
//! change something is marked on the conversation before it starts, any
//! call that settles while its reply runs is marked with its result, and a
//! restart carries every call left without a result on as [`Restored`]
//! work, the marked ones [`Interrupted`], in the same turn. What the
//! kernel does not keep is the text a model call streamed; this plugin
//! keeps it in its session data, `crash-resume/<agent>.json`, at least
//! every second. A turn that ends removes the file.
//!
//! On resume, each agent with restored work is [`Recovering`]: its
//! restored calls and model call are held, the "Interrupted turn" note goes
//! in its transcript, and the user is asked:
//!
//! - **Continue**: each call gets its result, else "may or may not have
//!   completed" (started) or "not run (interrupted)" (not run: it never
//!   started, or it only reads and changed nothing); the
//!   partial text of a model call the crash cut is kept, marked incomplete;
//!   the turn carries on;
//! - **Discard**: the turn is stopped and the conversation goes back to
//!   before its user message, which goes back in the input;
//! - **Leave**: nothing changes; the question comes back with the next
//!   message, which waits for the answer.
//!
//! Subagents the crash interrupted are listed in their parent's note and
//! follow its choice. A run nobody can answer, or an unattended one such
//! as `rig -p -r`'s, continues. A torn last line in an agent's log is
//! reported by the kernel's restore, which sets it aside.

use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Entry;
use rig_harness::rig_core::conversation::Note;
use rig_harness::rig_core::message::Message as Chat;
use rig_tui::MainInput;
use serde::{Deserialize, Serialize};

/// The author (a plugin name) of the transcript note, of the message that
/// asks the model to carry on, of the plugin's edits and decisions.
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

/// What the kernel does not keep of a running turn, as saved.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Progress {
    /// The text its running model call streamed so far.
    pub text: String,
    /// The conversation's last message when that text was streamed: the
    /// text is the reply to it.
    pub after: Option<MessageId>,
}

/// The name of an agent's progress file in the session's plugin data.
pub fn progress_file(id: &AgentId) -> String {
    format!("{ORIGIN}/{}.json", id.0)
}

/// The saved progress of `id`'s unfinished turn, if any.
pub fn read_progress(data: &dyn Files, id: &AgentId) -> Option<Progress> {
    let bytes = data.read_file(&progress_file(id)).ok()?;
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
pub struct Recovering {
    /// The message the turn began at.
    pub first_message: MessageId,
    /// The partial text of the model call the crash cut short.
    pub text: Option<String>,
    /// What the turn did, as shown.
    pub body: String,
    /// The restored calls, held.
    pub calls: Vec<Entity>,
    /// The transcript note, an unsaved aside of the agent's conversation.
    pub entry: Option<AsideId>,
    turn: Entity,
    model_call: Option<Entity>,
    holds: Vec<Entity>,
    ask: Option<Entity>,
    choice: Option<Recover>,
    left: bool,
    stopping: bool,
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

/// Keeps each turn's streamed text and recovers a turn a crash cut short.
#[derive(Default)]
pub struct CrashResumePlugin;

impl Plugin for CrashResumePlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(turn_started)
            .add_observer(turn_ended)
            .add_systems(PostStartup, recover)
            .add_systems(PostUpdate, save_text.run_if(on_real_timer(TEXT_EVERY)))
            .add_systems(
                AgentPass,
                (take_answers, ask, hold_messages, resolve)
                    .chain()
                    .in_set(Stage::Gate)
                    .after(AskSystems),
            );
    }
}

/// Writes `progress` as `id`'s, or removes it with `None`. A failed write
/// only loses what a crash would show.
fn save(data: Option<&PluginData>, id: &AgentId, progress: Option<&Progress>) {
    let Some(data) = data else {
        return;
    };
    let name = progress_file(id);
    match progress.map(serde_json::to_vec) {
        Some(Ok(bytes)) => data.session.write_file(&name, &bytes).ok(),
        Some(Err(_)) => None,
        None => data.session.remove_file(&name).ok(),
    };
}

/// Starts tracking a new turn, and keeps the loop awake while it runs so
/// its text is saved on time. A turn a restore carries on is left to
/// [`recover`].
fn turn_started(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf, Without<Restored>>,
    mut commands: Commands,
) {
    let Ok(&TurnOf(agent)) = turns.get(added.entity) else {
        return;
    };
    commands.entity(agent).try_insert(Tracked::default());
    commands
        .entity(added.entity)
        .try_insert(KeepAwake(TEXT_EVERY));
}

/// A turn that ended has nothing to recover.
fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    ids: Query<&AgentId, With<Tracked>>,
    data: Option<Res<PluginData>>,
    mut commands: Commands,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity)
        && let Ok(id) = ids.get(agent)
    {
        save(data.as_deref(), id, None);
        commands.entity(agent).try_remove::<Tracked>();
    }
}

/// Saves the text each running model call streamed so far, with the
/// message it answers.
fn save_text(
    calls: Query<(&Partial, &ForAgent), (With<TurnCall>, With<Started>, Without<Settled>)>,
    mut agents: Query<(&AgentId, &Conversation, &mut Tracked)>,
    data: Option<Res<PluginData>>,
) {
    for (partial, &ForAgent(agent)) in &calls {
        let Ok((id, conversation, mut tracked)) = agents.get_mut(agent) else {
            continue;
        };
        let after = conversation.entries().next_back().map(|entry| entry.id);
        if tracked.0.text != partial.text || tracked.0.after != after {
            tracked.0.text.clone_from(&partial.text);
            tracked.0.after = after;
            save(data.as_deref(), id, Some(&tracked.0));
        }
    }
}

/// On resume: each agent with restored work is [`Recovering`], that work
/// held, and its note, listing its interrupted subagents too, goes in the
/// transcript.
#[allow(clippy::type_complexity)]
fn recover(
    mut agents: Query<(
        &AgentId,
        &mut Conversation,
        Option<&SpawnedBy>,
        Option<&Name>,
    )>,
    turns: Query<(Entity, &TurnOf, &Turn, &Calls), With<Restored>>,
    calls: Query<(Option<&ToolRequest>, Option<&Interrupted>, Has<TurnCall>)>,
    data: Option<Res<PluginData>>,
    mut commands: Commands,
) {
    let mut found: HashMap<Entity, Recovering> = HashMap::new();
    for (
        turn,
        &TurnOf(agent),
        &Turn {
            start: first_message,
            ..
        },
        of,
    ) in &turns
    {
        let Ok((id, conversation, ..)) = agents.get(agent) else {
            continue;
        };
        let progress = data
            .as_deref()
            .and_then(|data| read_progress(&*data.session, id))
            .unwrap_or_default();
        let mut recovering = Recovering {
            first_message,
            text: None,
            body: String::new(),
            calls: Vec::new(),
            entry: None,
            turn,
            model_call: None,
            holds: Vec::new(),
            ask: None,
            choice: None,
            left: false,
            stopping: false,
        };
        let mut lines = Vec::new();
        for effect in of.iter() {
            let Ok((request, interrupted, model_call)) = calls.get(effect) else {
                continue;
            };
            if let Some(request) = request {
                recovering.calls.push(effect);
                lines.push(describe(&request.call, interrupted));
            } else if model_call {
                recovering.model_call = Some(effect);
                let last = conversation.entries().next_back().map(|entry| entry.id);
                let cut = !progress.text.trim().is_empty() && progress.after == last;
                recovering.text = cut.then(|| progress.text.trim_end().to_owned());
            }
            // Held, Interrupted or not, until the user chose.
            let name = Name::new("waiting for the interrupted turn's recovery");
            let hold = commands.spawn((name, HoldOf(effect))).id();
            recovering.holds.push(hold);
        }
        if let Some(text) = &recovering.text {
            lines.insert(0, format!("Partial reply (incomplete): {text}"));
        }
        if lines.is_empty() {
            lines.push("The model had not answered yet.".to_owned());
        }
        recovering.body = lines.join("\n");
        commands.entity(agent).try_insert(Tracked::default());
        found.insert(agent, recovering);
    }
    let mut notes = Vec::new();
    for (&agent, recovering) in &found {
        let Ok((_, _, parent, _)) = agents.get(agent) else {
            continue;
        };
        if parent.is_some_and(|parent| found.contains_key(&parent.0)) {
            continue;
        }
        let mut text = recovering.body.clone();
        for (&child, nested) in &found {
            let Ok((_, _, Some(of), name)) = agents.get(child) else {
                continue;
            };
            if of.0 == agent {
                let name = name.map_or("subagent", Name::as_str);
                let lines = nested.body.replace('\n', "\n    ");
                text.push_str(&format!(
                    "\n  {name}, interrupted inside its call:\n    {lines}"
                ));
            }
        }
        notes.push((agent, text));
    }
    for (agent, text) in notes {
        if let Ok((_, mut conversation, ..)) = agents.get_mut(agent)
            && let Some(recovering) = found.get_mut(&agent)
        {
            let after = conversation.entries().next_back().map(|entry| entry.id);
            let note = Note::new(TITLE, text).aside(Author::plugin(ORIGIN), after);
            recovering.entry = Some(conversation.add_aside(note.unsaved()));
        }
    }
    for (agent, recovering) in found {
        commands.entity(agent).try_insert(recovering);
    }
}

/// How far a restored call got, for the transcript and the question. An
/// unmarked call never started, or only reads and had not finished: either
/// way it changed nothing and has no result.
fn describe(call: &ToolCall, interrupted: Option<&Interrupted>) -> String {
    let args = serde_json::to_string(&call.function.arguments).unwrap_or_default();
    let what = format!("{} {args}", call.function.name.as_str());
    match interrupted {
        Some(Interrupted::Settled(result)) => {
            format!("finished: {what}\n  result: {}", result.output().render())
        }
        Some(Interrupted::Started) => format!("unknown outcome: may have run: {what}"),
        None => format!("not run: {what}"),
    }
}

/// Reads each answer to a question about an interrupted turn.
fn take_answers(
    asks: Query<(&CrashAsk, &Outcome), Added<Outcome>>,
    mut agents: Query<&mut Recovering>,
) {
    for (&CrashAsk(agent), outcome) in &asks {
        let Ok(mut recovering) = agents.get_mut(agent) else {
            continue;
        };
        recovering.ask = None;
        let answer = Answer::of(outcome);
        let is = |id| answer.as_ref().is_some_and(|answer| answer.is(id));
        recovering.choice = Some(if is("continue") {
            Recover::Continue
        } else if is("discard") {
            Recover::Discard
        } else {
            Recover::Leave
        });
    }
}

/// The question about an interrupted turn.
fn new_ask(agent: Entity, recovering: &Recovering) -> impl Bundle {
    let ask = Ask::new(TITLE)
        .body(recovering.body.clone())
        .choice("continue", "Continue")
        .choice("discard", "Discard")
        .choice("leave", "Leave");
    (ask, ForAgent(agent), CrashAsk(agent))
}

/// Asks about each interrupted turn that is not a subagent's inside its
/// parent's, unless it was left for now; a run nobody answers, or an
/// unattended one, continues.
fn ask(
    mut agents: Query<(Entity, &mut Recovering, Option<&SpawnedBy>)>,
    parents: Query<(), With<Recovering>>,
    answerers: Answerers,
    unattended: Option<Res<rig_harness::Unattended>>,
    mut commands: Commands,
) {
    let unattended = unattended.is_some_and(|unattended| unattended.0);
    for (agent, mut recovering, parent) in &mut agents {
        let nested = parent.is_some_and(|parent| parents.contains(parent.0));
        if nested || recovering.left || recovering.ask.is_some() || recovering.choice.is_some() {
            continue;
        }
        if !unattended && answerers.can_answer(Some(agent)) {
            recovering.ask = Some(commands.spawn(new_ask(agent, &recovering)).id());
        } else {
            recovering.choice = Some(Recover::Continue);
        }
    }
}

/// Holds each message the user sends an agent with an interrupted turn
/// until it is recovered, asking again after Leave.
fn hold_messages(
    messages: Query<
        (Entity, &Incoming, &ForAgent, &Decision),
        (ReadyToStart, IncomingMessages, Without<Seen>),
    >,
    mut agents: Query<&mut Recovering>,
    mut commands: Commands,
) {
    for (message, incoming, &ForAgent(agent), decision) in &messages {
        if incoming.author.kind != AuthorKind::User || decision.is_decided() {
            continue;
        }
        let Ok(mut recovering) = agents.get_mut(agent) else {
            continue;
        };
        commands.entity(message).insert(Seen);
        let name = Name::new("waiting for the interrupted turn's recovery");
        commands.spawn((name, CrashHold(agent), HoldOf(message)));
        if recovering.left {
            recovering.left = false;
            recovering.ask = Some(commands.spawn(new_ask(agent, &recovering)).id());
        }
    }
}

/// Carries out each choice, subagents following their parent's: Continue
/// at once; Discard, and Leave with a message waiting, by stopping the
/// turn and finishing once the agent is idle. Lets the held messages go.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn resolve(
    mut agents: Query<(
        Entity,
        &AgentId,
        &mut Recovering,
        &mut Conversation,
        Option<&SpawnedBy>,
        Has<ActiveTurn>,
    )>,
    holds: Query<(Entity, &CrashHold)>,
    mut decisions: Query<(&mut Decision, Option<&Interrupted>), (With<Restored>, With<Holds>)>,
    mut input: Option<MainInput>,
    data: Option<Res<PluginData>>,
    mut commands: Commands,
) {
    let chosen: HashMap<Entity, Recover> = agents
        .iter()
        .filter_map(|(agent, _, recovering, ..)| Some((agent, recovering.choice?)))
        .collect();
    for (_, _, mut recovering, _, parent, _) in &mut agents {
        if recovering.choice.is_none()
            && let Some(&choice) = parent.and_then(|parent| chosen.get(&parent.0))
        {
            recovering.choice = Some(choice);
        }
    }
    for (agent, id, mut recovering, mut conversation, parent, busy) in &mut agents {
        let Some(choice) = recovering.choice else {
            continue;
        };
        let held: Vec<Entity> = holds
            .iter()
            .filter(|(_, hold)| hold.0 == agent)
            .map(|(hold, _)| hold)
            .collect();
        if choice == Recover::Leave && held.is_empty() {
            recovering.choice = None;
            recovering.left = true;
            continue;
        }
        if choice != Recover::Continue && busy {
            // The kernel answers the held calls as stopped and ends the
            // turn; what it commits is cut or left as it was.
            if !recovering.stopping {
                recovering.stopping = true;
                let why = "an interrupted turn the user did not continue".to_owned();
                commands
                    .entity(recovering.turn)
                    .try_insert(TurnOutcome::Stopped(Some(why)));
            }
            continue;
        }
        let title = match choice {
            Recover::Continue => {
                carry_on(
                    agent,
                    &recovering,
                    &mut conversation,
                    &mut decisions,
                    &mut commands,
                );
                // The turn carries on: keep its streamed text from here.
                save(data.as_deref(), id, None);
                commands.entity(agent).try_insert(Tracked::default());
                commands
                    .entity(recovering.turn)
                    .try_insert(KeepAwake(TEXT_EVERY));
                "Interrupted turn (continued)"
            }
            Recover::Discard => {
                let first = recovering.first_message;
                let at = conversation.start_of(first);
                // The user's own message is among the entries the turn
                // answers: results and notes come before it.
                let typed = conversation
                    .asked(first)
                    .find_map(|entry| typed(&conversation, entry.index));
                conversation.splice(at.., [], Author::plugin(ORIGIN));
                if let (None, Some(text), Some(input)) = (parent, typed, input.as_mut()) {
                    input.set(text);
                }
                "Interrupted turn (discarded)"
            }
            Recover::Leave => "Interrupted turn (left as it was)",
        };
        if let Some(entry) = recovering.entry
            && let Some(aside) = conversation.aside(entry)
            && let Some(note) = Note::of(aside)
        {
            let retitled = Note::new(title, note.text).aside(aside.author.clone(), aside.after);
            conversation.replace_aside(entry, retitled.unsaved());
        }
        for hold in held.into_iter().chain(recovering.holds.iter().copied()) {
            commands.entity(hold).try_despawn();
        }
        commands.entity(agent).remove::<Recovering>();
    }
}

/// Answers the held calls, each with its result, else as started or never
/// started; or keeps the partial text of the model call the crash cut
/// short, marked incomplete, and sends a fresh call in place of the held
/// one, so its request has the text.
fn carry_on(
    agent: Entity,
    recovering: &Recovering,
    conversation: &mut Conversation,
    decisions: &mut Query<(&mut Decision, Option<&Interrupted>), (With<Restored>, With<Holds>)>,
    commands: &mut Commands,
) {
    for &call in &recovering.calls {
        let Ok((mut decision, interrupted)) = decisions.get_mut(call) else {
            continue;
        };
        let answer = match interrupted {
            Some(Interrupted::Settled(result)) => result.clone(),
            Some(Interrupted::Started) => ToolResult::failed(ToolExecutionError::other(UNKNOWN)),
            None => ToolResult::failed(ToolExecutionError::other(NOT_RUN)),
        };
        decision.decide(Decider::plugin(ORIGIN), answer);
    }
    if let (Some(text), Some(held)) = (&recovering.text, recovering.model_call) {
        let author = Author::plugin(ORIGIN);
        conversation.push(Chat::assistant(format!("{text}\n\n{INCOMPLETE}")));
        conversation.push(Entry::new(Chat::user(CARRY_ON)).by(author));
        let fresh = (TurnCall, CallOf(recovering.turn), ForAgent(agent));
        let fresh = commands.spawn(fresh).id();
        commands.entity(held).try_insert(SupersededBy(fresh));
    }
}

#[cfg(test)]
mod tests;
