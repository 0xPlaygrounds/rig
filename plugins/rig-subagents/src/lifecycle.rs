//! Who waits on whom, the reports, and the steps of a [`Lifecycle`].

use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;

use super::{AwaitedBy, Awaits, Lifecycle, OpenRequests, Peers, Request, Subtask, answered};
use rig_harness::prelude::*;

/// The report on a request a restart cut short.
const INTERRUPTED: &str = "Interrupted: the session restarted before the subagent answered.";

/// What every report that holds no answer ends with.
const NO_ANSWER: &str = "No answer will come for this request; send a `message` to carry it on.";

pub(super) fn add(app: &mut App) {
    app.add_observer(name_subagent)
        .add_observer(work_on_turn_start)
        .add_observer(step_on_turn_end)
        .add_observer(report_when_finished)
        .add_observer(end_waits)
        .add_observer(interrupt_restored);
}

/// An agent as the subagent tools see it.
#[derive(QueryData)]
pub(super) struct AgentData {
    pub(super) entity: Entity,
    pub(super) id: &'static AgentId,
    pub(super) subtask: Option<&'static Subtask>,
    pub(super) peers: Has<Peers>,
    pub(super) busy: Has<ActiveTurn>,
    pub(super) lifecycle: Option<&'static Lifecycle>,
    pub(super) family: (Option<&'static Spawned>, Option<&'static SpawnedBy>),
}

/// The agents, their open requests and the open `wait` calls: who waits on
/// whom, and the reports.
#[derive(SystemParam)]
pub(super) struct Agents<'w, 's> {
    pub(super) agents: Query<'w, 's, AgentData>,
    pub(super) open: Query<'w, 's, &'static mut OpenRequests>,
    pub(super) inboxes: Query<'w, 's, &'static mut Inbox>,
    waits: Query<'w, 's, (Entity, &'static Awaits, &'static CallOf)>,
    turns: Query<'w, 's, &'static TurnOf>,
    pub(super) commands: Commands<'w, 's>,
}

impl Agents<'_, '_> {
    /// The agent with the id `id`.
    pub(super) fn find(&self, id: &AgentId) -> Option<Entity> {
        id.find_in(self.agents.iter().map(|agent| (agent.entity, agent.id)))
    }

    /// Output of `agent`, titled with its task when it has one.
    pub(super) fn origin(&self, agent: Entity) -> Option<Origin> {
        let agent = self.agents.get(agent).ok()?;
        let origin = Origin::agent(agent.id.clone(), None);
        Some(match agent.subtask {
            Some(subtask) => origin.titled(subtask.0.clone()),
            None => origin,
        })
    }

    /// The agent the open `wait` call `call` belongs to.
    fn waiter(&self, call: Entity) -> Option<Entity> {
        let (_, _, of) = self.waits.get(call).ok()?;
        self.turns.get(of.0).ok().map(|turn| turn.0)
    }

    /// The open `wait` calls of `waiter` for a message from `awaited`.
    pub(super) fn wait_calls(&self, waiter: Entity, awaited: Entity) -> Vec<Entity> {
        let calls = self
            .waits
            .iter()
            .filter(|(_, awaits, _)| awaits.0 == awaited);
        let calls = calls.filter(|&(call, ..)| self.waiter(call) == Some(waiter));
        calls.map(|(call, ..)| call).collect()
    }

    /// The agents `agent` waits on: those that owe it a report, then those
    /// its open `wait` calls wait for.
    fn awaited(&self, agent: Entity) -> Vec<Entity> {
        let Ok(id) = self.agents.get(agent).map(|agent| agent.id) else {
            return Vec::new();
        };
        let asked = |open: &OpenRequests| open.0.iter().any(|request| request.asker == *id);
        let owing = self.agents.iter().map(|agent| agent.entity);
        let mut awaited: Vec<Entity> = owing
            .filter(|&owing| self.open.get(owing).is_ok_and(asked))
            .collect();
        let calls = self
            .waits
            .iter()
            .filter(|&(call, ..)| self.waiter(call) == Some(agent));
        awaited.extend(calls.map(|(_, awaits, _)| awaits.0));
        awaited
    }

    /// Whether `from` waits on `to`, directly or through other agents.
    pub(super) fn reaches(&self, from: Entity, to: Entity) -> bool {
        let (mut seen, mut stack) = (vec![from], vec![from]);
        while let Some(agent) = stack.pop() {
            for next in self.awaited(agent) {
                if next == to {
                    return true;
                }
                if !seen.contains(&next) {
                    seen.push(next);
                    stack.push(next);
                }
            }
        }
        false
    }

    /// Reports `text` from `agent` on `requests`: one message per asker, in
    /// the order of each asker's first request, naming its last one. It
    /// goes with the asker's next model call when the asker waits for
    /// `agent`; it is a note while another task of its batch is open, and
    /// otherwise carries the asker on. Returns the askers reported to.
    pub(super) fn report(
        &mut self,
        agent: Entity,
        mut requests: Vec<Request>,
        text: &str,
    ) -> Vec<Entity> {
        let from = self.origin(agent).unwrap_or_default();
        let mut reported = Vec::new();
        while let Some(asker) = requests.first().map(|first| first.asker.clone()) {
            let asked: Vec<Request> = requests.extract_if(.., |r| r.asker == asker).collect();
            let (Some(last), Some(asker)) = (asked.last(), self.find(&asker)) else {
                continue;
            };
            let mut open = self.open.iter().flat_map(|open| &open.0);
            let batched = asked.iter().filter_map(|r| r.batch);
            let batched = batched.collect::<Vec<_>>();
            let mode = if !self.wait_calls(asker, agent).is_empty() {
                DeliveryMode::Steer
            } else if open.any(|r| r.batch.is_some_and(|batch| batched.contains(&batch))) {
                DeliveryMode::Note
            } else {
                DeliveryMode::Queue
            };
            let mut origin = from.clone();
            origin.request = Some(last.id.clone());
            self.commands
                .trigger(Deliver::new(asker, text, mode).with_origin(origin));
            reported.push(asker);
        }
        reported
    }
}

/// Names a subagent by its task's title, when it is spawned or restored.
fn name_subagent(inserted: On<Insert<Subtask>>, subtasks: Query<&Subtask>, mut commands: Commands) {
    if let Ok(subtask) = subtasks.get(inserted.entity) {
        let name = Name::new(subtask.0.clone());
        commands.entity(inserted.entity).insert(name);
    }
}

/// A spawned agent whose turn starts is [`Lifecycle::Working`].
fn work_on_turn_start(
    started: On<Add<ActiveTurn>>,
    spawned: Query<(), With<SpawnedBy>>,
    mut commands: Commands,
) {
    if spawned.contains(started.entity) {
        commands.entity(started.entity).insert(Lifecycle::Working);
    }
}

/// A spawned agent whose turn ended, and that has not started another, is
/// [`Lifecycle::WaitingOn`] an agent that owes it a report, else
/// [`Lifecycle::Finished`] with its answer, or why there is none.
fn step_on_turn_end(
    end: On<TurnEnded>,
    idle: Query<(), (With<SpawnedBy>, Without<ActiveTurn>)>,
    mut agents: Agents,
) {
    let agent = end.entity;
    if !idle.contains(agent) {
        return;
    }
    let failed = |why: &str| format!("{why} {NO_ANSWER}");
    let step = match agents.awaited(agent).first() {
        Some(&owing) => Lifecycle::WaitingOn(owing),
        None => Lifecycle::Finished(match &end.outcome {
            TurnOutcome::Answered(message) => final_answer(message)
                .unwrap_or_else(|| failed("Failed: the subagent ended without a final message.")),
            TurnOutcome::Failed(why) => failed(&format!("Failed: {why}")),
            TurnOutcome::Stopped => failed("Interrupted: the subagent was stopped first."),
        }),
    };
    agents.commands.entity(agent).insert(step);
}

/// Reports a [`Lifecycle::Finished`] agent's answer on every request it
/// owes, and ends the `wait` calls for it of agents it did not report to.
fn report_when_finished(
    inserted: On<Insert<Lifecycle>>,
    awaited: Query<&AwaitedBy>,
    mut agents: Agents,
) {
    let agent = inserted.entity;
    let Ok(Some(Lifecycle::Finished(text))) = agents.agents.get(agent).map(|a| a.lifecycle) else {
        return;
    };
    let (text, id) = (text.clone(), agents.agents.get(agent).map(|a| a.id.clone()));
    let requests = match agents.open.get_mut(agent) {
        Ok(mut open) => std::mem::take(&mut open.0),
        Err(_) => Vec::new(),
    };
    let reported = agents.report(agent, requests, &text);
    let (Ok(id), Ok(awaited)) = (id, awaited.get(agent)) else {
        return;
    };
    for call in awaited.iter() {
        if agents
            .waiter(call)
            .is_some_and(|waiter| !reported.contains(&waiter))
        {
            let output = answered(format!(
                "`{}` finished without a message for you.",
                id.short()
            ));
            agents.commands.entity(call).insert_if_new(output);
        }
    }
}

/// Ends the `wait` calls of the agent a message is delivered to that wait
/// for its sender. The message goes with the model call after them.
fn end_waits(sent: On<Deliver>, mut agents: Agents) {
    let Some(from) = sent.origin.from.as_ref() else {
        return;
    };
    let Some(sender) = agents.find(from) else {
        return;
    };
    for call in agents.wait_calls(sent.entity, sender) {
        let output = answered(format!(
            "`{}` sent you a message; it follows.",
            from.short()
        ));
        agents
            .commands
            .entity(call)
            .remove::<Awaits>()
            .insert_if_new(output);
    }
}

/// After a restart, a restored agent that owed reports answers each as
/// interrupted and stays idle: nothing it was doing runs again by itself.
fn interrupt_restored(
    mut restored: On<Restored>,
    open: Query<&OpenRequests>,
    mut commands: Commands,
) {
    let agent = restored.entity;
    if open.get(agent).is_ok_and(|open| !open.0.is_empty()) {
        restored.event_mut().resume = false;
        let interrupted = Lifecycle::Finished(format!("{INTERRUPTED} {NO_ANSWER}"));
        commands.entity(agent).insert(interrupted);
    }
}
