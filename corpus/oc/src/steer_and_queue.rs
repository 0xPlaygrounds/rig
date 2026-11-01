//! OC-19: steer or queue messages while the agent works.
//!
//! **Any front.** What the user sends a busy agent waits as a pending
//! message of one of two kinds:
//!
//! - **Steer** (`DeliveryMode::Steer`, Enter in the terminal view): the
//!   kernel's own. It waits in the agent's inbox for the running turn's
//!   next model call, after the reply's running tool calls settle, without
//!   stopping them; a turn that would end with it waiting carries on with
//!   it, and the inbox is saved.
//! - **Queued** (`DeliveryMode::Queue`, Alt+Enter): taken over in Gate,
//!   because the kernel's queue gives everything queued in one step of the
//!   same turn. The plugin keeps it in the agent's saved [`Kept`], and
//!   each one starts a new turn of its own once a turn ends answered, one
//!   per turn, in order. A stopped or failed turn leaves them queued.
//!
//! When the user stops a turn, and when a restart brings the inbox back to
//! an idle agent, the user's unsent steering messages ([`Unsent`]) are
//! taken over into [`Kept`] too: they join the start of the user's next
//! message, in order, as the guide's "With the user's next message" says.
//! A steering message kept while the agent is busy goes to the kernel.
//!
//! [`PendingMessages`] lists an agent's pending messages and
//! [`EditPending`] edits, deletes, toggles or moves one, from any front; a
//! changed steering message the kernel holds is taken back (decided) with
//! those after it and sent again as it is now.
//!
//! **The terminal view.** Alt+Enter queues while the shown agent is busy
//! (the view's Tab) and sends while it is idle. A strip above the status
//! line lists the pending messages with their kind; Up on an empty input
//! selects them: Up and Down move, `e` edits in the input (Enter saves, Esc
//! cancels), `d` deletes, `t` toggles the kind, `k` and `j` move it up and
//! down, Esc or Enter leaves.

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::{Modifier, Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    InputAction, InputContext, InputSystems, KeyInput, MainInput, PanelCanvas, PendingInput,
    Placement, RequestRedraw, TuiKeys, TuiPanel, TuiPlugin, TuiSystems,
};

/// Who decides the messages the plugin takes over.
const NAME: &str = "steer-and-queue";

/// When a pending message goes to the agent.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PendingKind {
    /// At the running turn's next model call, or with the user's next
    /// message to an idle agent.
    Steer,
    /// As a new turn of its own, after a turn ends answered.
    Queued,
}

/// A message the user sent that has not reached the model.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    /// What it says.
    pub text: String,
    /// When it goes.
    pub kind: PendingKind,
}

/// The pending messages the plugin keeps for an agent, in order, saved
/// with the session: the queued ones, and steering ones for the user's
/// next message. The steering messages for a running turn are the
/// kernel's inbox.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct Kept(pub Vec<Pending>);

/// A change to an agent's pending message, from any front.
#[derive(EntityEvent, Clone, Debug)]
pub struct EditPending {
    /// The agent.
    pub entity: Entity,
    /// The message's place in [`PendingMessages::of`].
    pub at: usize,
    /// What changes.
    pub change: PendingChange,
}

/// What [`EditPending`] changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PendingChange {
    /// Its text becomes this; an empty text deletes it.
    Text(String),
    /// It is deleted.
    Delete,
    /// Steer becomes queued, queued becomes steer.
    Toggle,
    /// It moves one place earlier (`true`) or later.
    Move(bool),
}

/// A message the agent accepted and has not sent yet, as the guide reads
/// the inbox.
type Inboxed<'a> = (Entity, &'a Effect, &'a Incoming, &'a ForAgent, &'a Decision);

/// The inbox: accepted, not answered.
type InboxFilter = (With<Started>, Without<Outcome>);

/// The user's steering messages `agent` holds, oldest first, from the
/// `inbox`.
fn steering<'a>(agent: Entity, inbox: impl Iterator<Item = Inboxed<'a>>) -> Vec<(Entity, Pending)> {
    let mut held: Vec<_> = inbox
        .filter(|(_, _, message, to, decision)| {
            to.0 == agent && typed(message) && !decision.is_decided()
        })
        .filter(|(_, _, message, ..)| message.mode == DeliveryMode::Steer)
        .map(|(entity, effect, message, ..)| (effect.id(), entity, steer(message.text())))
        .collect();
    held.sort_unstable_by_key(|(id, ..)| *id);
    held.into_iter()
        .map(|(_, entity, pending)| (entity, pending))
        .collect()
}

/// Whether `message` is the user's own text, not a slash command.
fn typed(message: &Incoming) -> bool {
    message.author.kind == AuthorKind::User && message.command().is_none()
}

fn steer(text: &str) -> Pending {
    Pending {
        text: text.trim().to_owned(),
        kind: PendingKind::Steer,
    }
}

/// An agent's pending messages, for any front.
#[derive(SystemParam)]
pub struct PendingMessages<'w, 's> {
    inbox: Query<'w, 's, Inboxed<'static>, InboxFilter>,
    kept: Query<'w, 's, &'static Kept>,
}

impl PendingMessages<'_, '_> {
    /// `agent`'s pending messages in the order they go: the steering
    /// messages the kernel holds, then those the plugin keeps.
    pub fn of(&self, agent: Entity) -> Vec<Pending> {
        let held = steering(agent, self.inbox.iter()).into_iter();
        let kept = self.kept.get(agent).into_iter().flat_map(|kept| &kept.0);
        held.map(|(_, pending)| pending)
            .chain(kept.cloned())
            .collect()
    }
}

/// Steers and queues messages to busy agents, in any front, and in the
/// terminal view.
#[derive(Default)]
pub struct SteerQueuePlugin;

impl Plugin for SteerQueuePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        let gate = (take_queued, keep_unsent).chain();
        app.register_required_components::<Agent, Kept>()
            .init_resource::<Strip>()
            .add_systems(
                AgentPass,
                (
                    gate.in_set(Stage::Gate).after(AskSystems),
                    join_kept.in_set(PrepareSystems::Input),
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
            .add_observer(next_queued)
            .add_observer(edit);
    }
}

/// Takes over each message the user queues for a busy agent, in the order
/// they were sent.
fn take_queued(
    mut messages: Query<
        (&Effect, &Incoming, &ForAgent, &mut Decision),
        (ReadyToStart, IncomingMessages),
    >,
    mut agents: Query<&mut Kept, With<ActiveTurn>>,
) {
    let mut queued: Vec<_> = messages
        .iter_mut()
        .filter(|(_, message, _, decision)| {
            typed(message) && message.mode == DeliveryMode::Queue && !decision.is_decided()
        })
        .collect();
    queued.sort_unstable_by_key(|(effect, ..)| effect.id());
    for (_, message, &ForAgent(agent), mut decision) in queued {
        if let Ok(mut kept) = agents.get_mut(agent) {
            kept.0.push(Pending {
                text: message.text().trim().to_owned(),
                kind: PendingKind::Queued,
            });
            decision.decide(Decider::plugin(NAME), Outcome::custom("queued"));
        }
    }
}

/// Takes over the user's steering messages a turn the user stopped did not
/// send, or a restart brought back to an idle agent: kept, in order, for
/// the user's next message.
fn keep_unsent(
    mut unsent: Query<(&Unsent, &Effect, &Incoming, &ForAgent, &mut Decision), Added<Unsent>>,
    mut agents: Query<&mut Kept>,
) {
    let mut unsent: Vec<_> = unsent
        .iter_mut()
        .filter(|(unsent, _, message, _, decision)| {
            let by_user = match &unsent.ended {
                None => true,
                Some(TurnOutcome::Stopped(stop)) => *stop.cause() == StopCause::User,
                Some(_) => false,
            };
            by_user
                && typed(message)
                && message.mode == DeliveryMode::Steer
                && !decision.is_decided()
        })
        .collect();
    unsent.sort_unstable_by_key(|(_, effect, ..)| effect.id());
    for (_, _, message, &ForAgent(agent), mut decision) in unsent {
        if let Ok(mut kept) = agents.get_mut(agent) {
            // After the steering messages kept before, ahead of the queue.
            let at = kept
                .0
                .iter()
                .take_while(|item| item.kind == PendingKind::Steer)
                .count();
            kept.0.insert(at, steer(message.text()));
            decision.decide(Decider::plugin(NAME), Outcome::custom("kept"));
        }
    }
}

/// The user's next message to an idle agent starts with the steering
/// messages kept for it, joined in order.
fn join_kept(
    mut messages: Query<
        (&mut Incoming, &ForAgent),
        (Added<Admitted>, Without<Started>, Without<Outcome>),
    >,
    mut agents: Query<&mut Kept, Without<ActiveTurn>>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        let Ok(mut kept) = agents.get_mut(agent) else {
            continue;
        };
        if !typed(&message) || !kept.0.iter().any(|item| item.kind == PendingKind::Steer) {
            continue;
        }
        let (steering, queued): (Vec<_>, Vec<_>) = std::mem::take(&mut kept.0)
            .into_iter()
            .partition(|item| item.kind == PendingKind::Steer);
        kept.0 = queued;
        let mut texts: Vec<String> = steering.into_iter().map(|item| item.text).collect();
        texts.push(message.text().trim().to_owned());
        *message.text_mut() = texts.join("\n\n");
    }
}

/// Gives a busy agent's kernel the steering messages the plugin keeps for
/// it, in order.
fn hand_steering(
    mut agents: Query<
        (Entity, &mut Kept),
        (With<ActiveTurn>, Or<(Changed<Kept>, Added<ActiveTurn>)>),
    >,
    mut commands: Commands,
) {
    for (agent, mut kept) in &mut agents {
        if !kept.0.iter().any(|item| item.kind == PendingKind::Steer) {
            continue;
        }
        kept.0.retain(|item| {
            let queued = item.kind == PendingKind::Queued;
            if !queued {
                let message = Incoming::new(item.text.clone(), DeliveryMode::Steer);
                commands.spawn((message, ForAgent(agent)));
            }
            queued
        });
    }
}

/// A turn that ended answered starts its idle agent's first queued
/// message.
fn next_queued(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    mut agents: Query<&mut Kept, Without<ActiveTurn>>,
    mut commands: Commands,
) {
    let Ok((TurnOutcome::Answered(_), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(mut kept) = agents.get_mut(agent) else {
        return;
    };
    let Some(at) = kept
        .0
        .iter()
        .position(|item| item.kind == PendingKind::Queued)
    else {
        return;
    };
    let next = kept.0.remove(at);
    commands.spawn((
        Incoming::new(next.text, DeliveryMode::Steer),
        ForAgent(agent),
    ));
}

/// Applies an [`EditPending`] to the agent's list. When the steering
/// messages the kernel holds are no longer the list's first, unchanged,
/// they are all taken back and kept, and [`hand_steering`] sends them
/// again.
fn edit(
    edit: On<EditPending>,
    mut inbox: Query<(Entity, &Effect, &Incoming, &ForAgent, &mut Decision), InboxFilter>,
    mut agents: Query<&mut Kept>,
) {
    let agent = edit.entity;
    let Ok(mut kept) = agents.get_mut(agent) else {
        return;
    };
    let held = steering(agent, inbox.iter());
    let mut list: Vec<Pending> = held.iter().map(|(_, pending)| pending.clone()).collect();
    list.extend(kept.0.iter().cloned());
    if !change(&mut list, edit.at, &edit.change) {
        return;
    }
    let unchanged =
        list.len() >= held.len() && held.iter().zip(&list).all(|((_, was), now)| was == now);
    if unchanged {
        kept.0 = list.split_off(held.len());
        return;
    }
    for (message, _) in held {
        if let Ok((.., mut decision)) = inbox.get_mut(message) {
            decision.decide(Decider::plugin(NAME), Outcome::custom("taken back"));
        }
    }
    kept.0 = list;
}

/// Applies `change` to the message `at` of `list`; whether there was one.
fn change(list: &mut Vec<Pending>, at: usize, change: &PendingChange) -> bool {
    if at >= list.len() {
        return false;
    }
    match change {
        PendingChange::Text(text) if !text.trim().is_empty() => {
            if let Some(item) = list.get_mut(at) {
                item.text = text.trim().to_owned();
            }
        }
        PendingChange::Text(_) | PendingChange::Delete => {
            list.remove(at);
        }
        PendingChange::Toggle => {
            if let Some(item) = list.get_mut(at) {
                item.kind = match item.kind {
                    PendingKind::Steer => PendingKind::Queued,
                    PendingKind::Queued => PendingKind::Steer,
                };
            }
        }
        PendingChange::Move(earlier) => {
            let to = match earlier {
                true => at.checked_sub(1),
                false => Some(at + 1).filter(|to| *to < list.len()),
            };
            if let Some(to) = to {
                list.swap(at, to);
            }
        }
    }
    true
}

/// What the terminal view's strip shows selected, and the message being
/// edited in the input, as it was.
#[derive(Resource, Default)]
struct Strip {
    selected: Option<usize>,
    editing: Option<Pending>,
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

/// The strip's keys, Alt+Enter, and editing a message in the input.
fn strip_keys(
    mut events: ResMut<PendingInput>,
    mut input: MainInput,
    state: Res<InputContext>,
    mut strip: ResMut<Strip>,
    (shown, turns): (Res<ShownAgent>, Query<Has<ActiveTurn>>),
    pending: PendingMessages,
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
    let Some((agent, Ok(busy))) = shown.0.map(|agent| (agent, turns.get(agent))) else {
        return;
    };
    let list = pending.of(agent);
    let plain = modifiers.difference(KeyModifiers::SHIFT).is_empty();
    let quiet =
        state.completion().is_none() && state.picking().is_none() && state.answering().is_none();
    let mut change = |at, change| {
        commands.trigger(EditPending {
            entity: agent,
            at,
            change,
        });
    };
    if let Some(was) = &strip.editing {
        match code {
            // The message, wherever it is now; gone if it was sent.
            KeyCode::Enter if modifiers.is_empty() => {
                if let Some(at) = list.iter().position(|item| item == was) {
                    change(at, PendingChange::Text(input.take()));
                }
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
        let last = list.len().saturating_sub(1);
        let selected = selected.min(last);
        let Some(item) = list.get(selected) else {
            strip.selected = None;
            return;
        };
        strip.selected = match code {
            KeyCode::Up => Some(selected.saturating_sub(1)),
            KeyCode::Down => Some((selected + 1).min(last)),
            KeyCode::Char('e') => {
                input.set(item.text.clone());
                strip.editing = Some(item.clone());
                None
            }
            KeyCode::Char('d') => {
                change(selected, PendingChange::Delete);
                (last > 0).then_some(selected)
            }
            KeyCode::Char('t') => {
                change(selected, PendingChange::Toggle);
                Some(selected)
            }
            KeyCode::Char('k') => {
                change(selected, PendingChange::Move(true));
                Some(selected.saturating_sub(1))
            }
            KeyCode::Char('j') => {
                change(selected, PendingChange::Move(false));
                Some((selected + 1).min(last))
            }
            KeyCode::Esc | KeyCode::Enter => None,
            _ => Some(selected),
        };
        return;
    }
    if code == KeyCode::Enter && modifiers == KeyModifiers::ALT && quiet {
        // The view's own actions: Tab queues while busy, Send sends.
        let action = match busy {
            true => InputAction::Tab,
            false => InputAction::Send,
        };
        if let Some(first) = events.0.first_mut() {
            *first = KeyInput::Action(action);
        }
        return;
    }
    if code == KeyCode::Up && plain && quiet && input.is_empty() && !list.is_empty() {
        events.0.remove(0);
        strip.selected = Some(list.len().saturating_sub(1));
        redraw.write(RequestRedraw);
    }
}

/// The strip takes a row per pending message of the agent shown.
fn size_strip(
    mut panels: Query<&mut TuiPanel, With<StripPanel>>,
    shown: Res<ShownAgent>,
    pending: PendingMessages,
) {
    let rows = shown.0.map_or(0, |agent| pending.of(agent).len());
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
    shown: Res<ShownAgent>,
    pending: PendingMessages,
    strip: Res<Strip>,
) {
    let Some(agent) = shown.0 else {
        return;
    };
    let lines: Vec<Line> = pending
        .of(agent)
        .iter()
        .enumerate()
        .map(|(at, item)| {
            let kind = match item.kind {
                PendingKind::Steer => "steer ",
                PendingKind::Queued => "queued",
            };
            let mark = match strip.editing.as_ref() == Some(item) {
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
