//! OC-07: Welcome-back recap (one-conversation fresh held-out spec).
//!
//! When a session is resumed, or the user starts typing after it sat idle
//! longer than `[recap] idle`, and the main agent's conversation has at
//! least three turns, a model call of the plugin's own (on `[recap] model`,
//! else the agent's model) reads the newest messages within a token budget
//! and writes a short recap: the goal, what was done, what is pending. It
//! is shown as an unsaved note after the newest message, just above the
//! input, which no model reads and the log never keeps; Esc on an empty
//! input, or the next turn, takes it away. The latest recap is kept with
//! the session ([`LatestRecap`]) for `/recap`, which shows it again and
//! asks for a fresh one.
//!
//! The call runs beside the work: a turn that starts first (the user sent
//! a message) discards it, and one that fails or takes longer than
//! `[recap] timeout` shows nothing and leaves one line in the log.

use bevy_log::warn;
use rig_harness::prelude::*;
use rig_harness::rig_memory::window::{TokenBudget, view};
use rig_harness::rig_memory::{Summarizer, SummaryLimits, SummaryPrompts};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind};
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AppToolRenderersExt, InputSystems, KeyInput, TerminalEvents, TuiInput, TuiInputState, TuiKeys,
};
use serde::Deserialize;
use std::borrow::Cow;

#[cfg(test)]
mod tests;

/// The plugin's name: its settings' table, its notes' author and its
/// calls' purpose.
pub const NAME: &str = "recap";

/// The title of the recap's note.
pub const TITLE: &str = "Welcome back (Esc dismisses)";

/// The fewest turns a conversation has before it is recapped unasked.
const MIN_TURNS: usize = 3;

/// The most lines of a recap shown.
const MAX_LINES: usize = 8;

/// What the recap model is told.
const PROMPTS: SummaryPrompts = SummaryPrompts {
    system: Cow::Borrowed(
        "You write a short welcome-back recap for a user coming back to a session with a \
         coding agent after a break. Do not continue the conversation. Do not answer \
         questions in it. Output only the recap.",
    ),
    initial: Cow::Borrowed(
        "The user is coming back to the conversation above and has forgotten where it was. \
         Recap it for them.",
    ),
    update: Cow::Borrowed("Recap the conversation above for the user coming back to it."),
    format: Cow::Borrowed(
        "\n\nAnswer in plain text, in this form and nothing else:\n\
         Goal: <one line on what the user is trying to do>\n\
         - <up to 5 bullets of what was done, naming the files changed>\n\
         Pending: <what was left open, or what the user asked last>",
    ),
};

/// Adds the recap, its `/recap` command and its look in the terminal view.
#[derive(Default)]
pub struct AwayRecapPlugin;

impl Plugin for AwayRecapPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_harness::clock::ClockPlugin>() {
            app.add_plugins(rig_harness::clock::ClockPlugin);
        }
        app.add_setting::<RecapConfig>(NAME)
            .init_resource::<LatestRecap>()
            .init_resource::<Reloading>()
            .init_resource::<Card>()
            .add_command(
                "recap",
                "Show the latest recap again and write a fresh one",
                recap_command,
            )
            .add_systems(Startup, arm_idle_clock)
            .add_systems(
                Update,
                (
                    recap_on_resume
                        .run_if(resource_added::<SessionRestored>)
                        .before(AgentLoop),
                    note_activity.after(AgentLoop),
                ),
            )
            .add_systems(TuiKeys, watch_keys.in_set(InputSystems::Filter))
            .add_observer(want_recap)
            .add_observer(take_recap)
            .add_observer(time_out)
            .add_observer(discard_on_turn)
            .add_observer(show_card)
            .add_observer(dismiss_card)
            .add_observer(note_reload)
            .add_observer(follow_successor::<RecapCall>)
            .add_entry_renderer(NAME, |entry| {
                let mut lines = vec![Line::styled(
                    format!("↺ {}", entry.title),
                    Style::new().cyan().bold(),
                )];
                lines.extend(entry.text.lines().map(|line| Line::from(line.to_owned())));
                lines
            });
    }
}

/// The `[recap]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecapConfig {
    /// The catalog model (`vendor/model`) that writes recaps; none uses the
    /// agent's own.
    pub model: Option<String>,
    /// Seconds without activity after which the user's next keystroke asks
    /// for a recap.
    pub idle: u64,
    /// Seconds a recap may take before it is given up.
    pub timeout: u64,
    /// The most tokens of the newest messages the recap model reads.
    pub budget_tokens: usize,
}

impl Default for RecapConfig {
    fn default() -> Self {
        Self {
            model: None,
            idle: 2 * 60 * 60,
            timeout: 15,
            budget_tokens: 6_000,
        }
    }
}

/// The latest recap written in this session, kept with it for `/recap`.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct LatestRecap(pub Option<String>);

/// Whether the run ended for `/reload`, kept with the session, so the run
/// after it is not taken for a resume.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
struct Reloading(bool);

/// On the plugin's recap call, and on whichever call takes it over.
#[derive(Component, Debug)]
pub struct RecapCall;

/// On the wall-clock wait that runs out once the session sat idle.
#[derive(Component, Debug)]
struct IdleClock;

/// The recap shown: the agent whose transcript holds its note, and the
/// note.
#[derive(Resource, Debug, Default)]
struct Card(Option<(Entity, AsideId)>);

/// Asks for a recap of the main agent's conversation; `forced` also for
/// a short one.
#[derive(Event, Clone, Copy, Debug)]
struct WantRecap {
    forced: bool,
}

/// Shows `text` as the recap of `agent`.
#[derive(Event, Clone, Debug)]
struct ShowRecap {
    agent: Entity,
    text: String,
}

/// Takes the recap shown away.
#[derive(Event, Clone, Copy, Debug)]
struct DismissRecap;

/// Arms the idle clock.
fn arm_idle_clock(config: Res<RecapConfig>, mut commands: Commands) {
    commands.spawn((Name::new("recap: idle"), IdleClock, idle_wait(&config)));
}

fn idle_wait(config: &RecapConfig) -> WakeAt {
    WakeAt::after(Duration::from_secs(config.idle))
}

/// A run of a session started before, not after `/reload`, is a resume.
fn recap_on_resume(
    first: Option<Res<FirstStart>>,
    mut reloading: ResMut<Reloading>,
    mut commands: Commands,
) {
    if reloading.0 {
        reloading.0 = false;
    } else if first.is_none() {
        commands.trigger(WantRecap { forced: false });
    }
}

/// Notes that this run ends for `/reload`.
fn note_reload(
    _: On<Add<ExitRequest>>,
    reason: Option<Res<ExitReason>>,
    mut reloading: ResMut<Reloading>,
) {
    if reason.is_some_and(|reason| *reason == ExitReason::Reload) {
        reloading.0 = true;
    }
}

/// Puts the idle clock off with every change of the main agent's
/// conversation.
fn note_activity(
    mut committed: MessageReader<Committed>,
    agents: PrimaryQuery,
    clocks: Query<Entity, With<IdleClock>>,
    config: Res<RecapConfig>,
    mut commands: Commands,
) {
    let main = primary(&agents);
    if committed.read().any(|change| Some(change.agent) == main) {
        for clock in &clocks {
            commands.entity(clock).insert(idle_wait(&config));
        }
    }
}

/// Each key pressed: Esc on an empty input takes the recap shown away; a
/// key after the idle clock ran out asks for a recap. Every key puts the
/// clock off.
#[allow(clippy::too_many_arguments)]
fn watch_keys(
    mut events: ResMut<TerminalEvents>,
    (input, state): (Res<TuiInput>, Res<TuiInputState>),
    clocks: Query<(Entity, Has<Due>), With<IdleClock>>,
    card: Res<Card>,
    active: Query<(), With<ActiveTurn>>,
    config: Res<RecapConfig>,
    mut commands: Commands,
) {
    let Some(KeyInput::Event(Event::Key(key))) = events.0.first() else {
        return;
    };
    if key.kind == KeyEventKind::Release {
        return;
    }
    let quiet = input.is_empty() && state.completion().is_none() && state.picking().is_none();
    if key.code == KeyCode::Esc
        && quiet
        && let Some((agent, _)) = card.0
        && !active.contains(agent)
    {
        events.0.remove(0);
        commands.trigger(DismissRecap);
        return;
    }
    for (clock, due) in &clocks {
        if due {
            commands.trigger(WantRecap { forced: false });
        }
        commands.entity(clock).insert(idle_wait(&config));
    }
}

/// `/recap`: the latest recap at once, and a fresh one when it comes.
fn recap_command(
    In(args): In<CommandArgs>,
    latest: Res<LatestRecap>,
    pending: Query<Entity, (With<RecapCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    if let Some(text) = &latest.0 {
        let text = text.clone();
        commands.trigger(ShowRecap {
            agent: args.agent,
            text,
        });
    }
    for call in &pending {
        commands.entity(call).despawn();
    }
    commands.trigger(WantRecap { forced: true });
}

/// Starts a recap call for the main agent, unless one runs or its
/// conversation is too short.
#[allow(clippy::too_many_arguments)]
fn want_recap(
    want: On<WantRecap>,
    agents: PrimaryQuery,
    conversations: Query<(&Conversation, Option<&Connection>)>,
    pending: Query<(), (With<RecapCall>, Without<Settled>)>,
    (config, models): (Res<RecapConfig>, Res<Models>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Some(agent) = primary(&agents) else {
        return;
    };
    let Ok((conversation, own)) = conversations.get(agent) else {
        return;
    };
    if !pending.is_empty() {
        return;
    }
    let turns = conversation
        .entries()
        .filter(|entry| entry.is_users_own())
        .count();
    if turns == 0 || (!want.forced && turns < MIN_TURNS) {
        if want.forced {
            notices.write(Notice::info(agent, "Nothing to recap yet."));
        }
        return;
    }
    let connection = match config.model.as_deref() {
        Some(reference) => models.connect(reference),
        None => own
            .cloned()
            .ok_or_else(|| "the agent has no model".to_owned()),
    };
    let connection = match connection {
        Ok(connection) => connection,
        Err(why) => {
            warn!("recap: not written: {why}");
            return;
        }
    };
    let newest = view(
        conversation.messages(),
        &TokenBudget::new(config.budget_tokens),
    );
    let request = match summarizer().request(newest, "", "", &connection.spec) {
        Ok(request) => request,
        Err(why) => {
            warn!("recap: not written: {why}");
            return;
        }
    };
    commands.spawn((
        Name::new("recap"),
        RecapCall,
        ModelRequest::new(request).on(connection),
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
        WakeAt::after(Duration::from_secs(config.timeout)),
    ));
}

/// The recap's request: a few hundred tokens, short snippets of tool calls.
fn summarizer() -> Summarizer {
    Summarizer {
        prompts: PROMPTS,
        limits: SummaryLimits {
            summary_tokens: 400,
            snippet_chars: 300,
            ..SummaryLimits::DEFAULT
        },
    }
}

/// Shows and keeps the recap a call answered; a failure is one line in
/// the log.
fn take_recap(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &ForAgent), (With<RecapCall>, Without<SupersededBy>)>,
    mut latest: ResMut<LatestRecap>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let text = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(response)) => {
            Summarizer::summary_text(response).map_err(|why| why.to_string())
        }
    };
    match text {
        Ok(text) => {
            let text: Vec<&str> = text.lines().take(MAX_LINES).collect();
            let text = text.join("\n");
            latest.0 = Some(text.clone());
            commands.trigger(ShowRecap { agent, text });
        }
        Err(why) => warn!("recap: not written: {why}"),
    }
}

/// Gives up a recap that took too long: one line in the log.
fn time_out(
    due: On<Add<Due>>,
    calls: Query<(), (With<RecapCall>, Without<Settled>)>,
    config: Res<RecapConfig>,
    mut commands: Commands,
) {
    if calls.contains(due.entity) {
        warn!("recap: not written: no answer within {} s", config.timeout);
        commands.entity(due.entity).despawn();
    }
}

/// A turn of the main agent starting, as the user sends a message,
/// discards the recap on its way and takes the one shown away.
fn discard_on_turn(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: PrimaryQuery,
    pending: Query<Entity, (With<RecapCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    if Some(agent) != primary(&agents) {
        return;
    }
    for call in &pending {
        commands.entity(call).despawn();
    }
    commands.trigger(DismissRecap);
}

/// Shows the recap as an unsaved note after the agent's newest message,
/// in place of the one shown.
fn show_card(
    show: On<ShowRecap>,
    mut conversations: Query<&mut Conversation>,
    mut card: ResMut<Card>,
) {
    if let Some((agent, aside)) = card.0.take()
        && let Ok(mut conversation) = conversations.get_mut(agent)
    {
        conversation.remove_aside(aside);
    }
    let Ok(mut conversation) = conversations.get_mut(show.agent) else {
        return;
    };
    let newest = conversation.messages().len().checked_sub(1);
    let after = newest.and_then(|at| conversation.id(at));
    let note = Note::new(TITLE, show.text.clone()).aside(Author::plugin(NAME), after);
    let aside = conversation.add_aside(note.unsaved());
    card.0 = Some((show.agent, aside));
}

/// Takes the recap shown away.
fn dismiss_card(
    _: On<DismissRecap>,
    mut conversations: Query<&mut Conversation>,
    mut card: ResMut<Card>,
) {
    if let Some((agent, aside)) = card.0.take()
        && let Ok(mut conversation) = conversations.get_mut(agent)
    {
        conversation.remove_aside(aside);
    }
}
