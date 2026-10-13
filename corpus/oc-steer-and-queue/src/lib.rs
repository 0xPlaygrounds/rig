//! OC-19: steer or queue messages while the agent works.
//!
//! **Any front.** What the user sends a busy agent is a pending message of
//! one of two kinds, kept per agent in its saved [`PendingMessages`]:
//!
//! - **Steer** (`DeliveryMode::Steer`, Enter in the terminal view): handed
//!   to the kernel at once, which gives it to the running turn's next model
//!   call, after the reply's running tool calls settle, without stopping
//!   them; a turn that would end with it waiting carries on with it.
//! - **Queued** (`DeliveryMode::Queue`, Alt+Enter): taken over from the
//!   kernel, whose own queue gives everything queued in one step of the
//!   same turn. Each queued message instead starts a new turn of its own
//!   once a turn ends answered, one per turn, in order. An interrupted or
//!   failed turn leaves them queued.
//!
//! [`EditPending`] edits, deletes, toggles or moves a pending message from
//! any front; a steering message the kernel holds is taken back first (by
//! deciding its `Incoming`) and handed again as it is now. A steering
//! message the kernel does not hold, as after a restart, is handed to the
//! next running turn, or joins the start of the user's next message to an
//! idle agent. After a crash or a restart pending messages show again,
//! unsent: a queued one waits for a turn to end answered.
//!
//! An interrupted turn hands the steering messages the kernel held back to
//! the views, joined in order (the kernel's `Recalled`): the terminal view
//! puts them in the input, so the next Enter starts the next turn with
//! them. The plugin cannot keep them pending instead (see the report).
//!
//! **The terminal view.** Alt+Enter queues while the shown agent is busy
//! and sends while it is idle. A strip above the status line lists the
//! pending messages with their kind; Up on an empty input selects them:
//! Up and Down move, `e` edits in the input (Enter saves, Esc cancels),
//! `d` deletes, `t` toggles the kind, `k` and `j` move it up and down, Esc
//! or Enter leaves.

use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Modifier, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    Focused, InputSystems, KeyInput, PanelCanvas, Placement, RequestRedraw, Sent, TerminalEvents,
    TuiInput, TuiInputState, TuiKeys, TuiPanel, TuiPlugin, TuiSystems,
};
use serde::{Deserialize, Serialize};

/// Who decides a steering message the plugin takes back, or a queued one
/// it takes over.
const NAME: &str = "steer-and-queue";

/// When a pending message goes to the agent.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PendingKind {
    /// At the running turn's next model call.
    Steer,
    /// As a new turn of its own, after a turn ends answered.
    Queued,
}

/// A message the user sent a busy agent that has not reached its model.
#[derive(Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pending {
    /// Its id among the agent's pending messages.
    pub id: u64,
    /// What it says.
    pub text: String,
    /// When it goes.
    pub kind: PendingKind,
}

/// An agent's pending messages, in the order they go; kept with the
/// session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct PendingMessages {
    /// The messages, in order.
    pub items: Vec<Pending>,
    /// The id the next one gets.
    pub next: u64,
}

impl PendingMessages {
    /// Adds `text` of `kind` at the end; its id.
    fn push(&mut self, text: String, kind: PendingKind) -> u64 {
        let id = self.next;
        self.next += 1;
        self.items.push(Pending { id, text, kind });
        id
    }

    /// The position of the message `id`.
    fn position(&self, id: u64) -> Option<usize> {
        self.items.iter().position(|item| item.id == id)
    }
}

/// A change to one of an agent's pending messages, from any front.
#[derive(EntityEvent, Clone, Debug)]
pub struct EditPending {
    /// The agent.
    pub entity: Entity,
    /// The message's [`Pending::id`].
    pub id: u64,
    /// What changes.
    pub change: PendingChange,
}

/// What [`EditPending`] changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PendingChange {
    /// Its text becomes this.
    Text(String),
    /// It is deleted.
    Delete,
    /// Steer becomes queued, queued becomes steer.
    Toggle,
    /// It moves one place earlier (`true`) or later.
    Move(bool),
}

/// On a message the plugin keeps as pending: the agent and the
/// [`Pending::id`] it stands for.
#[derive(Component, Clone, Copy, Debug)]
struct Adopted {
    agent: Entity,
    id: u64,
}

/// On an adopted message the plugin took back from the kernel.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct TakenBack;

/// Steers and queues messages to busy agents, in any front, and in the
/// terminal view.
#[derive(Default)]
pub struct SteerQueuePlugin;

impl Plugin for SteerQueuePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.register_required_components::<Agent, PendingMessages>()
            .init_resource::<Strip>()
            .add_systems(
                AgentPass,
                (
                    adopt.in_set(Stage::Gate).after(AskSystems),
                    join_held.in_set(PrepareSystems::Input),
                ),
            )
            .add_systems(Update, hand_steering.before(AgentLoop))
            .add_systems(TuiKeys, strip_keys.in_set(InputSystems::Filter))
            .add_systems(Startup, spawn_strip)
            .add_systems(
                PostUpdate,
                (
                    size_strip.in_set(TuiSystems::Prepare),
                    draw_strip.in_set(TuiSystems::Draw),
                ),
            )
            .add_observer(settled)
            .add_observer(edit);
    }
}

/// Keeps each message the user sends a busy agent as pending: a steering
/// one goes on to the kernel, a queued one is taken over.
fn adopt(
    mut messages: Query<
        (Entity, &Incoming, &ForAgent, &mut Decision),
        (Undecided, IncomingMessages, Without<Adopted>),
    >,
    mut agents: Query<&mut PendingMessages, With<ActiveTurn>>,
    mut commands: Commands,
) {
    for (message, incoming, &ForAgent(agent), mut decision) in &mut messages {
        let typed = incoming.author.kind == AuthorKind::User && incoming.command().is_none();
        if !typed || decision.is_decided() {
            continue;
        }
        let Ok(mut pending) = agents.get_mut(agent) else {
            continue;
        };
        let text = incoming.text().trim().to_owned();
        match incoming.mode {
            DeliveryMode::Steer => {
                let id = pending.push(text, PendingKind::Steer);
                commands.entity(message).insert(Adopted { agent, id });
            }
            DeliveryMode::Queue => {
                pending.push(text, PendingKind::Queued);
                decision.decide(Decider::plugin(NAME), Outcome::custom("queued"));
            }
            DeliveryMode::Note => {}
        }
    }
}

/// The steering messages of `pending` the kernel does not hold, given the
/// ids it holds.
fn held<'a>(pending: &'a PendingMessages, live: &[u64]) -> impl Iterator<Item = &'a Pending> {
    pending
        .items
        .iter()
        .filter(move |item| item.kind == PendingKind::Steer && !live.contains(&item.id))
}

/// The ids of `agent`'s pending messages the kernel holds.
fn live_ids(live: &Query<&Adopted, LiveFilter>, agent: Entity) -> Vec<u64> {
    live.iter()
        .filter(|adopted| adopted.agent == agent)
        .map(|adopted| adopted.id)
        .collect()
}

/// Adopted messages the kernel holds: not settled, not taken back.
type LiveFilter = (Without<TakenBack>, Without<Settled>, Without<Outcome>);

/// Hands a busy agent each steering message the kernel does not hold, as
/// a turn starts or the pending messages change.
fn hand_steering(
    agents: Query<
        (Entity, &PendingMessages),
        (
            With<ActiveTurn>,
            Or<(Changed<PendingMessages>, Added<ActiveTurn>)>,
        ),
    >,
    live: Query<&Adopted, LiveFilter>,
    mut commands: Commands,
) {
    for (agent, pending) in &agents {
        let live = live_ids(&live, agent);
        for item in held(pending, &live) {
            commands.spawn((
                Incoming::new(item.text.clone(), DeliveryMode::Steer),
                ForAgent(agent),
                Adopted { agent, id: item.id },
            ));
        }
    }
}

/// The user's message to an idle agent starts with the steering messages
/// the kernel does not hold, joined in order.
fn join_held(
    mut messages: Query<
        (&mut Incoming, &ForAgent),
        (Added<Admitted>, Without<Adopted>, Without<Outcome>),
    >,
    mut agents: Query<&mut PendingMessages, Without<ActiveTurn>>,
    live: Query<&Adopted, LiveFilter>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        let typed = message.author.kind == AuthorKind::User && message.command().is_none();
        let Ok(mut pending) = agents.get_mut(agent) else {
            continue;
        };
        let live = live_ids(&live, agent);
        let joined: Vec<(u64, String)> = held(&pending, &live)
            .map(|item| (item.id, item.text.clone()))
            .collect();
        if !typed || joined.is_empty() {
            continue;
        }
        let mut texts: Vec<&str> = joined.iter().map(|(_, text)| text.as_str()).collect();
        let typed_text = message.text().trim().to_owned();
        texts.push(&typed_text);
        *message.text_mut() = texts.join("\n\n");
        pending
            .items
            .retain(|item| !joined.iter().any(|(id, _)| *id == item.id));
    }
}

/// An adopted message that settled is no longer pending, unless the plugin
/// took it back: delivered, or handed back to the views. A turn that ended
/// answered starts the agent's first queued message.
fn settled(
    settled: On<Add<Settled>>,
    adopted: Query<&Adopted, Without<TakenBack>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut agents: Query<(&mut PendingMessages, Has<ActiveTurn>)>,
    live: Query<&Adopted, LiveFilter>,
    mut commands: Commands,
) {
    if let Ok(&Adopted { agent, id }) = adopted.get(settled.entity) {
        if let Ok((mut pending, _)) = agents.get_mut(agent)
            && let Some(at) = pending.position(id)
        {
            pending.items.remove(at);
        }
        return;
    }
    let Ok((TurnOutcome::Answered(_), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((pending, busy)) = agents.get(agent) else {
        return;
    };
    let live = live_ids(&live, agent);
    let next = pending
        .items
        .iter()
        .find(|item| item.kind == PendingKind::Queued);
    let sending = pending
        .items
        .iter()
        .any(|item| item.kind == PendingKind::Queued && live.contains(&item.id));
    if let Some(next) = next
        && !busy
        && !sending
    {
        commands.spawn((
            Incoming::new(next.text.clone(), DeliveryMode::Steer),
            ForAgent(agent),
            Adopted { agent, id: next.id },
        ));
    }
}

/// Applies an [`EditPending`], taking the agent's steering messages the
/// kernel holds back where they change; [`hand_steering`] hands them again.
fn edit(
    edit: On<EditPending>,
    mut agents: Query<&mut PendingMessages>,
    mut live: Query<(Entity, &Adopted, &mut Decision), (IncomingMessages, LiveFilter)>,
    mut commands: Commands,
) {
    let (agent, id) = (edit.entity, edit.id);
    let Ok(mut pending) = agents.get_mut(agent) else {
        return;
    };
    let Some(at) = pending.position(id) else {
        return;
    };
    // The messages whose kernel copy goes: the edited one, or every one
    // when the order changes.
    let all = matches!(edit.change, PendingChange::Move(_));
    match &edit.change {
        PendingChange::Text(text) if text.trim().is_empty() => {
            pending.items.remove(at);
        }
        PendingChange::Text(text) => {
            if let Some(item) = pending.items.get_mut(at) {
                item.text = text.trim().to_owned();
            }
        }
        PendingChange::Delete => {
            pending.items.remove(at);
        }
        PendingChange::Toggle => {
            if let Some(item) = pending.items.get_mut(at) {
                item.kind = match item.kind {
                    PendingKind::Steer => PendingKind::Queued,
                    PendingKind::Queued => PendingKind::Steer,
                };
            }
        }
        PendingChange::Move(earlier) => {
            let to = match earlier {
                true => at.checked_sub(1),
                false => Some(at + 1).filter(|to| *to < pending.items.len()),
            };
            if let Some(to) = to {
                pending.items.swap(at, to);
            }
        }
    }
    for (message, adopted, mut decision) in &mut live {
        if adopted.agent == agent && (all || adopted.id == id) {
            commands.entity(message).insert(TakenBack);
            decision.decide(Decider::Person, Outcome::custom("taken back"));
        }
    }
}

/// What the terminal view's strip shows selected, and the message being
/// edited in the input.
#[derive(Resource, Default)]
struct Strip {
    selected: Option<usize>,
    editing: Option<u64>,
}

/// The strip of pending messages above the status line.
#[derive(Component)]
struct StripPanel;

fn spawn_strip(mut commands: Commands) {
    commands.spawn((
        StripPanel,
        TuiPanel::new(Placement::Bottom(Constraint::Length(0))),
    ));
}

/// The pending messages of the agent shown, and whether it is busy.
type ShownAgent<'a> = (Entity, &'a PendingMessages, Has<ActiveTurn>);

/// The strip's keys, Alt+Enter, and editing a message in the input.
#[expect(
    clippy::too_many_arguments,
    reason = "a key filter reads the view's input and state"
)]
fn strip_keys(
    mut events: ResMut<TerminalEvents>,
    mut input: ResMut<TuiInput>,
    state: Res<TuiInputState>,
    mut strip: ResMut<Strip>,
    shown: Query<ShownAgent, With<Focused>>,
    mut sent: MessageWriter<Sent>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Some(KeyInput::Event(Event::Key(key))) = events.0.first() else {
        return;
    };
    if key.kind == KeyEventKind::Release {
        return;
    }
    let (code, modifiers) = (key.code, key.modifiers);
    let Some((agent, pending, busy)) = shown.iter().next() else {
        return;
    };
    let plain = modifiers.difference(KeyModifiers::SHIFT).is_empty();
    let quiet =
        state.completion().is_none() && state.picking().is_none() && state.answering().is_none();
    let change = |id, change| EditPending {
        entity: agent,
        id,
        change,
    };
    if let Some(id) = strip.editing {
        match code {
            KeyCode::Enter if modifiers.is_empty() => {
                commands.trigger(change(id, PendingChange::Text(input.take())));
            }
            KeyCode::Esc => {
                input.take();
            }
            _ => return,
        }
        strip.editing = None;
        events.0.remove(0);
        redraw.write(RequestRedraw);
        return;
    }
    if let Some(selected) = strip.selected {
        events.0.remove(0);
        redraw.write(RequestRedraw);
        let count = pending.items.len();
        let Some(item) = pending.items.get(selected.min(count.saturating_sub(1))) else {
            strip.selected = None;
            return;
        };
        let id = item.id;
        strip.selected = match code {
            KeyCode::Up => Some(selected.saturating_sub(1)),
            KeyCode::Down => Some(selected + 1).filter(|next| *next < count),
            KeyCode::Char('e') => {
                input.set(item.text.clone());
                strip.editing = Some(id);
                None
            }
            KeyCode::Char('d') => {
                commands.trigger(change(id, PendingChange::Delete));
                (count > 1).then_some(selected)
            }
            KeyCode::Char('t') => {
                commands.trigger(change(id, PendingChange::Toggle));
                Some(selected)
            }
            KeyCode::Char('k') => {
                commands.trigger(change(id, PendingChange::Move(true)));
                Some(selected.saturating_sub(1))
            }
            KeyCode::Char('j') => {
                commands.trigger(change(id, PendingChange::Move(false)));
                Some((selected + 1).min(count.saturating_sub(1)))
            }
            KeyCode::Esc | KeyCode::Enter => None,
            _ => Some(selected),
        };
        return;
    }
    let alt_enter = code == KeyCode::Enter && modifiers == KeyModifiers::ALT;
    if alt_enter && quiet && !input.text().trim().is_empty() {
        events.0.remove(0);
        let text = input.take();
        sent.write(Sent(text.clone()));
        let mode = match busy {
            true => DeliveryMode::Queue,
            false => DeliveryMode::Steer,
        };
        commands.trigger(Deliver::new(agent, text, mode));
        return;
    }
    if code == KeyCode::Up && plain && quiet && input.is_empty() && !pending.items.is_empty() {
        events.0.remove(0);
        strip.selected = Some(pending.items.len().saturating_sub(1));
        redraw.write(RequestRedraw);
    }
}

/// The strip takes a row per pending message of the agent shown.
fn size_strip(
    mut panels: Query<&mut TuiPanel, With<StripPanel>>,
    shown: Query<&PendingMessages, With<Focused>>,
) {
    let rows = shown.iter().next().map_or(0, |pending| pending.items.len());
    let placement = Placement::Bottom(Constraint::Length(u16::try_from(rows).unwrap_or(u16::MAX)));
    for mut panel in &mut panels {
        if panel.placement != placement {
            panel.placement = placement;
        }
    }
}

/// Each pending message of the agent shown, with its kind; the selected
/// one highlighted, the one being edited marked.
fn draw_strip(
    mut panels: Query<&mut PanelCanvas, With<StripPanel>>,
    shown: Query<&PendingMessages, With<Focused>>,
    strip: Res<Strip>,
) {
    let Some(pending) = shown.iter().next() else {
        return;
    };
    let lines: Vec<Line> = pending
        .items
        .iter()
        .enumerate()
        .map(|(at, item)| {
            let kind = match item.kind {
                PendingKind::Steer => "steer ",
                PendingKind::Queued => "queued",
            };
            let mark = match strip.editing == Some(item.id) {
                true => "✎",
                false => "⏵",
            };
            let first = item.text.lines().next().unwrap_or_default();
            let line = Line::from(vec![
                Span::from(format!("  {mark} {kind} ")).cyan(),
                Span::from(first.to_owned()),
            ]);
            match strip.selected == Some(at) {
                true => line.style(Style::new().add_modifier(Modifier::REVERSED)),
                false => line,
            }
        })
        .collect();
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.clone()));
    }
}

#[cfg(test)]
mod tests;
