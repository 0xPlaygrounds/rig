//! OC-07: Welcome-back recap (one-conversation fresh held-out spec).
//!
//! When a session is resumed ([`AppStart`] says [`StartReason::Resumed`]),
//! or the user starts typing after it sat idle longer than `[recap] idle`,
//! and the main agent's conversation has at least three turns, a model
//! call of the plugin's own (on `[recap] model`, else the agent's model)
//! reads the newest messages within a token budget and writes a short
//! recap: the goal, what was done, what is pending. It is shown in a panel
//! of the terminal view just above the input, never in the conversation,
//! so no model reads it and the log never keeps it; Esc on an empty input,
//! or the user's next message, takes it away. The latest recap is kept
//! with the session ([`LatestRecap`]) for `/recap`, which shows it again
//! and asks for a fresh one.
//!
//! The call runs beside the work: a message the user sends first discards
//! it, and one that fails or takes longer than `[recap] timeout` shows
//! nothing and leaves one line in the log.

use bevy_log::warn;
use rig_harness::prelude::*;
use rig_harness::rig_memory::recent::{TokenBudget, view};
use rig_harness::rig_memory::{CompactReason, Summarizer, SummaryLimits, SummaryPrompts};
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    Editor, InputContext, InputSystems, KeyInput, MainEditor, PanelCanvas, PendingInput, Placement,
    Theme, TuiKeys, TuiPanel, TuiSystems,
};
use serde::Deserialize;
use std::borrow::Cow;

#[cfg(test)]
mod tests;

/// The plugin's name: its settings' table and its calls' purpose.
pub const NAME: &str = "recap";

/// The title of the recap's card.
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
        "The user is coming back to the conversation and has forgotten where it was. \
         Recap it for them.",
    ),
    update: Cow::Borrowed("Recap the conversation for the user coming back to it."),
    format: Cow::Borrowed(
        "\n\nAnswer in plain text, in this form and nothing else:\n\
         Goal: <one line on what the user is trying to do>\n\
         - <up to 5 bullets of what was done, naming the files changed>\n\
         Pending: <what was left open, or what the user asked last>",
    ),
    ..SummaryPrompts::DEFAULT
};

/// Adds the recap, its `/recap` command and its card in the terminal view.
#[derive(Default)]
pub struct AwayRecapPlugin;

impl Plugin for AwayRecapPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_harness::clock::ClockPlugin>() {
            app.add_plugins(rig_harness::clock::ClockPlugin);
        }
        app.add_setting::<RecapConfig>(NAME)
            .init_resource::<LatestRecap>()
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
            .add_systems(PostUpdate, draw_card.in_set(TuiSystems::Draw))
            .add_observer(want_recap)
            .add_observer(take_recap)
            .add_observer(time_out)
            .add_observer(discard_on_send)
            .add_observer(show_card)
            .add_observer(dismiss_card)
            .add_observer(follow_successor::<RecapCall>);
    }
}

/// The `[recap]` table of the settings.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecapConfig {
    /// The catalog model (`vendor/model`) that writes recaps; none uses the
    /// agent's own.
    pub model: Option<String>,
    /// Seconds without activity after which the user's next typed key asks
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

/// On the plugin's recap call, and on whichever call takes it over.
#[derive(Component, Debug)]
pub struct RecapCall;

/// The recap shown: a panel of the terminal view above the input.
#[derive(Component, Debug)]
pub struct RecapCard(pub String);

/// On the wall-clock wait that runs out once the session sat idle.
#[derive(Component, Debug)]
struct IdleClock;

/// Asks for a recap of the main agent's conversation; `forced` also for
/// a short one.
#[derive(Event, Clone, Copy, Debug)]
struct WantRecap {
    forced: bool,
}

/// Shows `0` as the recap.
#[derive(Event, Clone, Debug)]
struct ShowRecap(String);

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

/// Recaps a resumed session once its conversation is restored.
fn recap_on_resume(started: Option<Res<AppStart>>, mut commands: Commands) {
    if started.is_some_and(|started| started.0 == StartReason::Resumed) {
        commands.trigger(WantRecap { forced: false });
    }
}

/// Puts the idle clock off with every change of the main agent's
/// conversation.
fn note_activity(
    mut committed: MessageReader<ConversationChanged>,
    agents: MainAgentQuery,
    clocks: Query<Entity, With<IdleClock>>,
    config: Res<RecapConfig>,
    mut commands: Commands,
) {
    let main = main_agent(&agents);
    if committed.read().any(|change| Some(change.agent) == main) {
        for clock in &clocks {
            commands.entity(clock).insert(idle_wait(&config));
        }
    }
}

/// Each key pressed: Esc on an empty input takes the recap shown away; a
/// typed character after the idle clock ran out asks for a recap. Any key
/// before then puts the clock off.
fn watch_keys(
    mut events: ResMut<PendingInput>,
    (input, state): (Option<Single<&Editor, With<MainEditor>>>, Res<InputContext>),
    clocks: Query<(Entity, Has<Due>), With<IdleClock>>,
    cards: Query<Entity, With<RecapCard>>,
    config: Res<RecapConfig>,
    mut commands: Commands,
) {
    let Some(KeyInput::Event(Event::Key(key))) = events.0.first() else {
        return;
    };
    if key.kind == KeyEventKind::Release {
        return;
    }
    let quiet = input.is_none_or(|input| input.is_empty())
        && state.completion().is_none()
        && state.picking().is_none();
    if key.code == KeyCode::Esc && quiet && !cards.is_empty() {
        events.0.remove(0);
        commands.trigger(DismissRecap);
        return;
    }
    let typing = matches!(key.code, KeyCode::Char(_))
        && !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT);
    for (clock, due) in &clocks {
        if due && typing {
            commands.trigger(WantRecap { forced: false });
        }
        if !due || typing {
            commands.entity(clock).insert(idle_wait(&config));
        }
    }
}

/// `/recap`: the latest recap at once, and a fresh one when it comes.
fn recap_command(
    In(_): In<CommandArgs>,
    latest: Res<LatestRecap>,
    pending: Query<Entity, (With<RecapCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    if let Some(text) = &latest.0 {
        commands.trigger(ShowRecap(text.clone()));
    }
    for call in &pending {
        commands.entity(call).despawn();
    }
    commands.trigger(WantRecap { forced: true });
}

/// Starts a recap call for the main agent, unless one runs or its
/// conversation is too short.
fn want_recap(
    want: On<WantRecap>,
    agents: MainAgentQuery,
    conversations: Query<(&Conversation, Option<&Connection>)>,
    pending: Query<(), (With<RecapCall>, Without<Settled>)>,
    (config, models): (Res<RecapConfig>, Res<Models>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(&agents) else {
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
    let newest = view(
        conversation.messages(),
        &TokenBudget::new(config.budget_tokens),
    );
    let request = connection.and_then(|connection| {
        let request = summarizer().request(newest, "", &CompactReason::Threshold, &connection.spec);
        Ok((request.map_err(|why| why.to_string())?, connection))
    });
    let (request, connection) = match request {
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
    calls: Query<&Outcome, (With<RecapCall>, Without<SupersededBy>)>,
    mut latest: ResMut<LatestRecap>,
    mut commands: Commands,
) {
    let Ok(outcome) = calls.get(settled.entity) else {
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
            let text = text.lines().take(MAX_LINES).collect::<Vec<_>>().join("\n");
            latest.0 = Some(text.clone());
            commands.trigger(ShowRecap(text));
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

/// A message the user sends the main agent discards the recap on its way
/// and takes the one shown away.
fn discard_on_send(
    deliver: On<Deliver>,
    agents: MainAgentQuery,
    pending: Query<Entity, (With<RecapCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    let message = &deliver.message;
    let users = message.author.kind == AuthorKind::User && message.command().is_none();
    if !users || Some(deliver.entity) != main_agent(&agents) {
        return;
    }
    for call in &pending {
        commands.entity(call).despawn();
    }
    commands.trigger(DismissRecap);
}

/// Shows the recap in a panel above the input, in place of the one shown.
fn show_card(show: On<ShowRecap>, cards: Query<Entity, With<RecapCard>>, mut commands: Commands) {
    for card in &cards {
        commands.entity(card).despawn();
    }
    let rows = u16::try_from(show.0.lines().count().saturating_add(2)).unwrap_or(u16::MAX);
    commands.spawn((
        Name::new("recap: card"),
        RecapCard(show.0.clone()),
        TuiPanel::new(Placement::Bottom(Constraint::Length(rows))),
    ));
}

/// Takes the recap shown away.
fn dismiss_card(
    _: On<DismissRecap>,
    cards: Query<Entity, With<RecapCard>>,
    mut commands: Commands,
) {
    for card in &cards {
        commands.entity(card).despawn();
    }
}

/// Draws the recap's card.
fn draw_card(mut cards: Query<(&RecapCard, &mut PanelCanvas)>, theme: Res<Theme>) {
    for (card, mut canvas) in &mut cards {
        canvas.render(Paragraph::new(card.0.as_str()).block(theme.raised(TITLE)));
    }
}
