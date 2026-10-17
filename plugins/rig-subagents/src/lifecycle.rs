//! Who waits on whom, the reports, and the steps of a [`Lifecycle`].

use bevy_ecs::query::QueryData;
use bevy_ecs::system::SystemParam;
use rig_core::completion::Message;
use rig_core::tool::feedback::not_finished;

use super::tools::{PEER_ROLE, SUBAGENT_ROLE};
use super::{
    AwaitedBy, Awaits, Lifecycle, OpenRequests, Peers, Report, Request, SubagentSystems, Subtask,
    answered,
};
use rig_harness::prelude::*;

/// Why a restored subagent's turn is stopped.
const RESTARTED: &str = "restart";

/// What every report that holds no answer ends with.
const NO_ANSWER: &str = "No answer will come for this request; send a `message` to carry it on.";

pub(super) fn add(app: &mut App) {
    app.add_observer(name_subagent)
        .add_observer(brief_subagent)
        .add_observer(brief_peer)
        .add_observer(work_on_turn_start)
        .add_observer(step_on_turn_end)
        .add_observer(report_when_finished)
        .add_observer(end_waits)
        .add_observer(note_stops)
        .init_resource::<Stopping>()
        .add_systems(Last, forget_stops)
        .configure_sets(
            AgentPass,
            SubagentSystems::StopRestored
                .after(GateSystems::After)
                .before(ApprovalSystems)
                .before(GateSystems::Fallback),
        )
        .add_systems(
            AgentPass,
            stop_restored.in_set(SubagentSystems::StopRestored),
        );
}

/// An agent as the subagent tools see it.
#[derive(QueryData)]
pub(super) struct AgentData {
    pub(super) entity: Entity,
    pub(super) id: &'static AgentId,
    pub(super) subtask: Has<Subtask>,
    pub(super) title: Option<&'static AgentTitle>,
    pub(super) peers: Has<Peers>,
    pub(super) busy: Has<ActiveTurn>,
    pub(super) lifecycle: Option<&'static Lifecycle>,
    pub(super) children: Option<&'static Spawned>,
    pub(super) parent: Option<&'static SpawnedBy>,
    effect_parent: Option<&'static EffectParent>,
}

/// The agents, their open requests and the open `wait` calls: who waits on
/// whom, and the reports.
#[derive(SystemParam)]
pub(super) struct Agents<'w, 's> {
    pub(super) agents: Query<'w, 's, AgentData>,
    pub(super) open: Query<'w, 's, &'static mut OpenRequests>,
    /// The messages to each agent its model has not read yet: accepted
    /// (`true`), or on their way.
    pub(super) inboxes:
        Query<'w, 's, (&'static mut Incoming, &'static ForAgent, Has<Started>), Without<Outcome>>,
    waits: Query<'w, 's, (Entity, &'static Awaits, &'static CallOf)>,
    awaited_by: Query<'w, 's, &'static AwaitedBy>,
    turns: Query<'w, 's, &'static TurnOf>,
    /// The open tool calls' answers.
    calls: Query<'w, 's, &'static mut Decision, ToolCalls>,
    wake: Res<'w, Wake>,
    pub(super) commands: Commands<'w, 's>,
}

impl Agents<'_, '_> {
    /// Decides the open tool call `call` with `outcome`.
    pub(super) fn answer(&mut self, call: Entity, outcome: impl Into<Outcome>) {
        if let Ok(mut decision) = self.calls.get_mut(call) {
            decision.decide(Decider::plugin("rig-subagents"), outcome);
        }
    }

    /// The agent with the id `id`.
    pub(super) fn find(&self, id: &AgentId) -> Option<Entity> {
        id.find_in(self.agents.iter().map(|agent| (agent.entity, agent.id)))
    }

    /// Output of `agent`, titled with its task when it has one.
    pub(super) fn author(&self, agent: Entity) -> Option<Author> {
        let agent = self.agents.get(agent).ok()?;
        let author = Author::agent(agent.id.clone(), None);
        Some(match agent.title.filter(|_| agent.subtask) {
            Some(title) => author.titled(title.0.clone()),
            None => author,
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
            .awaited_by
            .get(awaited)
            .into_iter()
            .flat_map(|by| by.iter());
        calls
            .filter(|&call| self.waiter(call) == Some(waiter))
            .collect()
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
    /// the order of each asker's first request, naming its last one, and
    /// marked [`Report`]. A `note`, the report of a stopped agent, is a
    /// note from this plugin that starts no turn. Otherwise it goes with
    /// the asker's next model call when the asker waits for `agent`; it is
    /// a note while another task of its batch is open, and otherwise
    /// carries the asker on. Returns the askers reported to.
    pub(super) fn report(
        &mut self,
        agent: Entity,
        mut requests: Vec<Request>,
        text: &str,
        note: bool,
    ) -> Vec<Entity> {
        let from = self.author(agent).unwrap_or_default();
        let mut reported = Vec::new();
        while let Some(asker) = requests.first().map(|first| first.asker.clone()) {
            let asked: Vec<Request> = requests.extract_if(.., |r| r.asker == asker).collect();
            let (Some(last), Some(asker)) = (asked.last(), self.find(&asker)) else {
                continue;
            };
            let mut open = self.open.iter().flat_map(|open| &open.0);
            let batched = asked.iter().filter_map(|r| r.batch);
            let batched = batched.collect::<Vec<_>>();
            let mode = if note {
                DeliveryMode::Inform
            } else if !self.wait_calls(asker, agent).is_empty() {
                DeliveryMode::Steer
            } else if open.any(|r| r.batch.is_some_and(|batch| batched.contains(&batch))) {
                DeliveryMode::Inform
            } else {
                DeliveryMode::Queue
            };
            let mut author = from.clone();
            author.request = Some(last.id.clone());
            if note {
                author.kind = AuthorKind::Plugin("rig-subagents".to_owned());
            }
            let parent = self.agents.get(asker).ok().and_then(|a| a.effect_parent);
            self.commands.spawn((
                Incoming::new(text, mode).with_author(author),
                ForAgent(asker),
                Effect::with_parent_id(parent.map(|parent| parent.0)),
                Report,
            ));
            self.wake.wake();
            reported.push(asker);
        }
        reported
    }
}

/// Names an agent by its title, such as a subagent or a helper when it is
/// spawned or restored.
fn name_subagent(
    inserted: On<Insert<AgentTitle>>,
    titles: Query<&AgentTitle>,
    mut commands: Commands,
) {
    if let Ok(AgentTitle(title)) = titles.get(inserted.entity) {
        let name = Name::new(title.clone());
        commands.entity(inserted.entity).insert(name);
    }
}

/// Gives a subagent its role as a prompt section of its own when it is
/// spawned or restored, so the text follows the build: it is not saved.
fn brief_subagent(added: On<Add<Subtask>>, mut commands: Commands) {
    commands.spawn(role(added.entity, 0, "subagent_role", SUBAGENT_ROLE));
}

/// Tells a subagent with [`Peers`] about them, as [`brief_subagent`] does.
fn brief_peer(added: On<Add<Peers>>, mut commands: Commands) {
    commands.spawn(role(added.entity, 1, "peer_role", PEER_ROLE));
}

/// The prompt section of `agent`'s role `tag`, `after` other roles.
fn role(agent: Entity, after: i32, tag: &'static str, text: &str) -> impl Bundle {
    let section = PromptSection::new(PromptSection::ORDER_ROLE + after, tag, text);
    (Name::new(tag), section, SectionOf(agent))
}

/// A subagent whose turn starts is [`Lifecycle::Working`].
fn work_on_turn_start(
    started: On<Add<ActiveTurn>>,
    spawned: Query<(), With<Subtask>>,
    mut commands: Commands,
) {
    if spawned.contains(started.entity) {
        commands.entity(started.entity).insert(Lifecycle::Working);
    }
}

/// The agents a [`StopTurn`] stopped this frame: the agent it names and
/// every agent below it, as the kernel stops them.
#[derive(Resource, Default)]
pub(super) struct Stopping(Vec<Entity>);

/// Notes the agents a [`StopTurn`] stops, before their turns end.
fn note_stops(stop: On<StopTurn>, spawned: Query<&Spawned>, mut stopping: ResMut<Stopping>) {
    let below = spawned.iter_descendants::<Spawned>(stop.entity);
    stopping.0.extend(std::iter::once(stop.entity).chain(below));
}

/// Forgets the agents stopped this frame, whose turns ended by now.
fn forget_stops(mut stopping: ResMut<Stopping>) {
    stopping.0.clear();
}

/// On a [`Lifecycle::Finished`] subagent whose turn a [`StopTurn`] stopped,
/// of it or of an agent above it: its report is a note that starts no turn
/// of the agent it goes to.
#[derive(Component)]
pub(super) struct StoppedEarly;

/// A subagent whose turn ended, and that has not started another, is
/// [`Lifecycle::WaitingOn`] an agent that owes it a report, unless a
/// restart or a [`StopTurn`] stopped it, else [`Lifecycle::Finished`] with
/// its answer, or why there is none and what it said last. A stopped turn
/// reports as rig-core's [`not_finished`] says a stopped call that may
/// have partly run does: a turn a plugin stops, such as for a budget,
/// reports `Stopped: <reason>. It may have partly run; …`, and one a
/// `StopTurn` stopped is [`StoppedEarly`].
fn step_on_turn_end(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    idle: Query<&Conversation, (With<Subtask>, Without<ActiveTurn>)>,
    stopping: Res<Stopping>,
    mut agents: Agents,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(conversation) = idle.get(agent) else {
        return;
    };
    let said = last_words(conversation).map(|said| format!("\n\nIt had said: {said}\n\n"));
    let said = said.unwrap_or_else(|| " ".to_owned());
    let failed = |why: &str| format!("{why}{said}{NO_ANSWER}");
    let stopped = |why: &str| failed(&not_finished(why, true, &Touches::Independent));
    // A restart or a stop ends it for good: it waits on no one.
    let early = matches!(outcome, TurnOutcome::Stopped(_)) && stopping.0.contains(&agent);
    let restarted = matches!(outcome, TurnOutcome::Stopped(Some(why)) if why == RESTARTED);
    let step = match agents
        .awaited(agent)
        .first()
        .filter(|_| !restarted && !early)
    {
        Some(&owing) => Lifecycle::WaitingOn(owing),
        None => Lifecycle::Finished(match outcome {
            TurnOutcome::Answered(message) => message
                .final_answer()
                .unwrap_or_else(|| failed("Failed: the subagent ended without a final message.")),
            TurnOutcome::Failed(why) => failed(&format!("Failed: {why}")),
            TurnOutcome::Stopped(None) => {
                stopped("the user stopped the subagent before it answered")
            }
            TurnOutcome::Stopped(Some(why)) if why == RESTARTED => {
                stopped("the session restarted before the subagent answered")
            }
            TurnOutcome::Stopped(Some(why)) => stopped(why),
        }),
    };
    let mut agent = agents.commands.entity(agent);
    match early {
        true => agent.insert((step, StoppedEarly)),
        false => agent.remove::<StoppedEarly>().insert(step),
    };
}

/// Reports a [`Lifecycle::Finished`] agent's answer on every request it
/// owes, and ends the `wait` calls for it of agents it did not report to.
fn report_when_finished(
    inserted: On<Insert<Lifecycle>>,
    awaited: Query<&AwaitedBy>,
    early: Query<(), With<StoppedEarly>>,
    mut agents: Agents,
) {
    let agent = inserted.entity;
    let note = early.contains(agent);
    let Ok(Some(Lifecycle::Finished(text))) = agents.agents.get(agent).map(|a| a.lifecycle) else {
        return;
    };
    let (text, id) = (text.clone(), agents.agents.get(agent).map(|a| a.id.clone()));
    let requests = match agents.open.get_mut(agent) {
        Ok(mut open) => std::mem::take(&mut open.0),
        Err(_) => Vec::new(),
    };
    let reported = agents.report(agent, requests, &text, note);
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
            agents.answer(call, output);
        }
    }
}

/// Ends the `wait` calls of the agent that accepted a message that wait
/// for its sender. The message goes with the model call after them, even
/// one sent to read after the agent's turn.
fn end_waits(accepted: On<Add<Started>>, mut agents: Agents) {
    let Ok((message, &ForAgent(to), _)) = agents.inboxes.get(accepted.entity) else {
        return;
    };
    let Some(from) = message.author.from.clone().map(AgentId) else {
        return;
    };
    let from = &from;
    let Some(sender) = agents.find(from) else {
        return;
    };
    let waits = agents.wait_calls(to, sender);
    if !waits.is_empty()
        && let Ok((mut message, ..)) = agents.inboxes.get_mut(accepted.entity)
        && message.mode == DeliveryMode::Queue
    {
        message.mode = DeliveryMode::Steer;
    }
    for call in waits {
        let output = answered(format!(
            "`{}` sent you a message; it follows.",
            from.short()
        ));
        agents.commands.entity(call).remove::<Awaits>();
        agents.answer(call, output);
    }
}

/// On a restored turn [`stop_restored`] looked at.
#[derive(Component)]
pub(super) struct Considered;

/// After a restart, a restored agent that owed reports stays idle: its
/// restored turn is stopped before anything of it starts, and its end
/// reports each request as stopped, unless a plugin claimed any of the
/// turn's restored work in Gate: held, decided or superseded one of its
/// calls, or ended the turn. It looks at each restored turn once, after
/// every gate of the pass its calls are admitted in, before the user is
/// asked to approve any of them and the kernel's own answers.
pub(super) fn stop_restored(
    turns: Query<
        (Entity, &TurnOf, &Calls),
        (With<Restored>, Without<TurnOutcome>, Without<Considered>),
    >,
    calls: Query<(Has<Admitted>, Has<Holds>, Has<SupersededBy>, &Decision)>,
    open: Query<&OpenRequests>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), of) in &turns {
        let calls: Vec<_> = calls.iter_many(of.iter()).flatten().collect();
        if !calls.iter().any(|&(admitted, ..)| admitted) {
            continue;
        }
        commands.entity(turn).insert(Considered);
        let claimed = calls
            .iter()
            .any(|&(_, held, superseded, decision)| held || superseded || decision.is_decided());
        if !claimed && open.get(agent).is_ok_and(|open| !open.0.is_empty()) {
            let stopped = TurnOutcome::Stopped(Some(RESTARTED.to_owned()));
            commands.entity(turn).insert(stopped);
        }
    }
}

/// What the model said last in the conversation's last request: the text
/// of its last message with text since the last message that was not a
/// tool's result.
fn last_words(conversation: &Conversation) -> Option<String> {
    for message in conversation.messages().iter().rev() {
        match &**message {
            Message::Assistant(reply) => {
                let parts = reply.content.iter().filter_map(|part| match part {
                    AssistantContent::Text(text) => Some(text.text.trim()),
                    _ => None,
                });
                let text: Vec<&str> = parts.filter(|text| !text.is_empty()).collect();
                if !text.is_empty() {
                    return Some(text.join("\n\n"));
                }
            }
            Message::User { content } => {
                let results = |part: &UserContent| matches!(part, UserContent::ToolResult(_));
                if !content.iter().all(results) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    None
}
