//! Plugin corpus (fresh held-out): FH-13: Next-step suggestions (spec
//! `13-next-step-suggestions.md`).
//!
//! When a turn of the agent the user looks at ends with an answer, a model
//! call of the plugin's own (on `[suggest] model`, else the agent's model)
//! reads the last message the user typed and the final reply, and answers
//! up to three next prompts. The terminal view shows them as numbered chips
//! in a one-row panel under the input; `Alt+1..3` sends one to the
//! agent as the user's message and `Alt+0` hides them. Those keys are bound
//! only while chips show and no question waits beside the work, so they
//! never take the question's own `Alt+N` keys. Any message the user
//! sends, and any new turn, takes them away and cancels a call still
//! running. A failed or stopped turn, a subagent, and a run without the
//! terminal view get none. The suggestions are no part of the conversation
//! and are not restored; the call's usage counts under the purpose
//! `suggest`. `/suggest off|on` switches them for the session.

use bevy_ecs::entity_disabling::Disabled;
use bevy_ecs::query::Allow;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::message::Message;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, PanelCanvas, Placement, RequestRedraw,
    TuiPanel, TuiPlugin, TuiSystems,
};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The settings' table, the command and the calls' purpose.
const NAME: &str = "suggest";

/// The most suggestions kept.
const MOST: usize = 3;

/// The longest suggestion kept, in characters.
const LONGEST: usize = 60;

/// What the suggesting model is told.
const PREAMBLE: &str = "You suggest what a user asks a coding agent next. Given the user's \
    last message and the agent's final reply, answer with up to 3 short next requests the user \
    would likely send, such as \"run the full test suite\". One per line, each at most 60 \
    characters, imperative, no numbering, nothing else.";

/// Suggests the user's next prompts after each answer.
#[derive(Default)]
pub struct NextStepSuggestionsPlugin;

impl Plugin for NextStepSuggestionsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.add_setting::<SuggestConfig>(NAME)
            .init_resource::<SuggestOff>()
            .add_command(
                NAME,
                "Suggest next prompts after each answer: /suggest off|on",
                toggle,
            )
            .add_observer(ask)
            .add_observer(take)
            .add_observer(on_message)
            .add_observer(on_turn)
            .add_observer(follow_successor::<SuggestCall>)
            .add_systems(
                PostUpdate,
                (place_panel, arm_keys).in_set(TuiSystems::Prepare),
            )
            .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
        let mut bound = Vec::new();
        for (index, key) in ['1', '2', '3'].into_iter().enumerate() {
            let label = format!("Send suggestion {}", index + 1);
            let send = move |pressed: On<KeyPressed>,
                             agents: Query<&Suggestions>,
                             mut commands: Commands| {
                send(&pressed, index, &agents, &mut commands);
            };
            bound.extend(app.add_key_binding(KeyBinding::new(KeyPress::alt(key), label), send));
        }
        let label = "Hide the suggestions";
        bound.extend(app.add_key_binding(KeyBinding::new(KeyPress::alt('0'), label), hide));
        for key in bound {
            app.world_mut().entity_mut(key).insert((ChipKey, Disabled));
        }
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(rig_tui::FRONT) {
            app.init_resource::<ChipsShown>();
        }
    }
}

/// The `[suggest]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SuggestConfig {
    /// The catalog model (`vendor/model`) that suggests, a cheap one; none
    /// uses the agent's own.
    pub model: Option<String>,
}

/// There while a front shows the suggestions: the terminal view's run
/// inserts it, and another front that shows them may too. Without it no
/// suggestion is asked for, as in a `--print` run.
#[derive(Resource, Default)]
pub struct ChipsShown;

/// Whether `/suggest off` switched the suggestions off, for the session.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SuggestOff(pub bool);

/// On an agent: the prompts suggested after its last answer.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Suggestions(pub Vec<String>);

/// On the plugin's suggesting call, and on whichever call takes it over.
#[derive(Component, Debug)]
pub struct SuggestCall;

/// The panel the chips are drawn in.
#[derive(Component)]
struct ChipsPanel;

/// On the plugin's key bindings, which are `Disabled` while no chips show.
#[derive(Component)]
struct ChipKey;

/// Asks for suggestions when a turn of the agent shown ended with an
/// answer, unless they are off, no front shows them, or it is a subagent.
fn ask(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    agents: Query<&Conversation, Without<SpawnedBy>>,
    (shown, chips, off, config, models): (
        Option<Res<ShownAgent>>,
        Option<Res<ChipsShown>>,
        Res<SuggestOff>,
        Res<SuggestConfig>,
        Res<Models>,
    ),
    mut commands: Commands,
) {
    let Ok((TurnOutcome::Answered(message), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let shown = shown.and_then(|shown| shown.0) == Some(agent);
    if chips.is_none() || off.0 || !shown {
        return;
    }
    let Ok(conversation) = agents.get(agent) else {
        return;
    };
    let len = conversation.messages().len();
    let Some(asked) = (0..len).rev().find_map(|at| typed(conversation, at)) else {
        return;
    };
    let answer = message.final_answer().unwrap_or_default();
    let prompt = format!("The user's last message:\n{asked}\n\nThe agent's final reply:\n{answer}");
    let mut request =
        ModelRequest::new(CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE));
    if let Some(connection) = config
        .model
        .as_deref()
        .and_then(|to| models.connect(to).ok())
    {
        request = request.on(connection);
    }
    // No `Name`: the agent's activity shows it idle while this runs.
    commands.spawn((
        SuggestCall,
        request,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
    ));
}

/// Keeps the suggestions the call answered on its agent.
fn take(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &ForAgent), With<SuggestCall>>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let suggested = outcome
        .completion()
        .map(|reply| suggestions(&reply.text()))
        .unwrap_or_default();
    if !suggested.is_empty() {
        commands.entity(agent).try_insert(Suggestions(suggested));
    }
}

/// A message the user sends takes the agent's suggestions away.
fn on_message(
    added: On<Add<Incoming>>,
    messages: Query<(&Incoming, &ForAgent)>,
    calls: Query<(Entity, &ForAgent), (With<SuggestCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    if let Ok((message, &ForAgent(agent))) = messages.get(added.entity)
        && message.author.kind == AuthorKind::User
    {
        clear(agent, &calls, &mut commands);
    }
}

/// A new turn takes the agent's suggestions away, whoever started it.
fn on_turn(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    calls: Query<(Entity, &ForAgent), (With<SuggestCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    if let Ok(&TurnOf(agent)) = turns.get(added.entity) {
        clear(agent, &calls, &mut commands);
    }
}

/// Takes `agent`'s suggestions away and cancels its suggesting call: a
/// call despawned before it settles is recorded as cancelled.
fn clear(
    agent: Entity,
    calls: &Query<(Entity, &ForAgent), (With<SuggestCall>, Without<Settled>)>,
    commands: &mut Commands,
) {
    for (call, _) in calls.iter().filter(|(_, of)| of.0 == agent) {
        commands.entity(call).despawn();
    }
    commands.entity(agent).try_remove::<Suggestions>();
}

/// `/suggest off|on`.
fn toggle(
    In(args): In<CommandArgs>,
    mut off: ResMut<SuggestOff>,
    calls: Query<(Entity, &ForAgent), (With<SuggestCall>, Without<Settled>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    match args.args.as_str() {
        "off" => {
            off.0 = true;
            clear(args.agent, &calls, &mut commands);
            notices.write(Notice::info(
                args.agent,
                "No more suggestions this session.",
            ));
        }
        "on" => {
            off.0 = false;
            notices.write(Notice::info(args.agent, "Suggestions after each answer."));
        }
        _ => {
            notices.write(Notice::error(args.agent, "/suggest needs off or on"));
        }
    }
}

/// `Alt+N`: sends suggestion `index` of the agent shown as the user's
/// message, as if typed.
fn send(pressed: &KeyPressed, index: usize, agents: &Query<&Suggestions>, commands: &mut Commands) {
    let Some(agent) = pressed.agent.filter(|_| !pressed.released) else {
        return;
    };
    let Some(text) = agents.get(agent).ok().and_then(|shown| shown.0.get(index)) else {
        return;
    };
    commands.trigger(Deliver::new(agent, text.clone(), DeliveryMode::Steer));
}

/// `Alt+0`: hides the suggestions of the agent shown.
fn hide(pressed: On<KeyPressed>, mut commands: Commands) {
    if let Some(agent) = pressed.agent.filter(|_| !pressed.released) {
        commands.entity(agent).try_remove::<Suggestions>();
    }
}

/// Keeps one chips panel while the agent shown has suggestions.
fn place_panel(
    shown: Option<Res<ShownAgent>>,
    agents: Query<Ref<Suggestions>>,
    panels: Query<Entity, With<ChipsPanel>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let suggested = shown
        .and_then(|shown| shown.0)
        .and_then(|agent| agents.get(agent).ok());
    match (suggested, panels.iter().next()) {
        (Some(_), None) => {
            let panel = TuiPanel::new(Placement::BelowInput(Constraint::Length(1)));
            commands.spawn((ChipsPanel, panel));
        }
        (Some(suggested), Some(_)) if suggested.is_changed() => {
            redraw.write(RequestRedraw);
        }
        (None, Some(panel)) => commands.entity(panel).despawn(),
        _ => {}
    }
}

/// Binds the chips' keys while the agent shown has suggestions and no
/// question waits beside the work, whose `Alt+N` keys a global binding
/// would shadow; frees them otherwise.
fn arm_keys(
    shown: Option<Res<ShownAgent>>,
    agents: Query<(), With<Suggestions>>,
    asks: Query<&Ask, (Without<Outcome>, Without<Settled>)>,
    keys: Query<(Entity, Has<Disabled>), (With<ChipKey>, Allow<Disabled>)>,
    mut commands: Commands,
) {
    let suggested = shown
        .and_then(|shown| shown.0)
        .is_some_and(|agent| agents.contains(agent));
    let armed = suggested && !asks.iter().any(|ask| ask.inline);
    for (key, disabled) in &keys {
        match (armed, disabled) {
            (true, true) => {
                commands.entity(key).remove::<Disabled>();
            }
            (false, false) => {
                commands.entity(key).insert(Disabled);
            }
            _ => {}
        }
    }
}

/// Draws the chips of the agent shown.
fn draw(
    mut panels: Query<&mut PanelCanvas, With<ChipsPanel>>,
    shown: Option<Res<ShownAgent>>,
    agents: Query<&Suggestions>,
) {
    let Some(Suggestions(suggested)) = shown
        .and_then(|shown| shown.0)
        .and_then(|agent| agents.get(agent).ok())
    else {
        return;
    };
    let line = chips(suggested);
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(line.clone()));
    }
}

/// The chips' row: `[1] run the full test suite  [2] update CHANGELOG`.
pub fn chips(suggested: &[String]) -> Line<'static> {
    let mut spans = Vec::new();
    for (index, text) in suggested.iter().enumerate() {
        let gap = if index == 0 { "" } else { "  " };
        spans.push(Span::from(format!("{gap}[{}] ", index + 1)).bold());
        spans.push(Span::from(text.clone()).dim());
    }
    spans.push(Span::from("   Alt+0 hides").dim().italic());
    Line::from(spans)
}

/// The suggestions in a reply: up to three non-empty lines, without
/// bullets, numbering or quotes, each cut to 60 characters.
pub fn suggestions(reply: &str) -> Vec<String> {
    reply
        .lines()
        .map(|line| unnumbered(line.trim().trim_start_matches(['-', '*', '•', ' '])))
        .map(|line| line.trim_matches(['"', '`', '\'', ' ']))
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.chars()
                .take(LONGEST)
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .take(MOST)
        .collect()
}

/// `line` without a leading `1.`, `1)` or `[1]`.
fn unnumbered(line: &str) -> &str {
    let rest = line.trim_start_matches('[');
    let digits = rest.trim_start_matches(|c: char| c.is_ascii_digit());
    if digits.len() < rest.len()
        && let Some(after) = digits.strip_prefix(['.', ')', ']'])
    {
        return after.trim_start();
    }
    line
}
