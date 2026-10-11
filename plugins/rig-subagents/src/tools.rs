//! The `task`, `message` and `wait` tools.

use schemars::JsonSchema;
use serde::Deserialize;

use super::lifecycle::{AgentDataItem, Agents};
use super::spawn::{Brief, Role, SpawnSubagent, TaskCall};
use super::{Awaits, Lifecycle, OpenRequests, Peers, Request, Subtask, answered, refused};
use rig_harness::prelude::*;

/// The tool that starts a subagent.
pub const TASK: &str = "task";

/// The tool that sends a subagent or a peer a request, or an asker a
/// report.
pub const MESSAGE: &str = "message";

/// The tool that waits for an agent's message without ending the turn.
pub const WAIT: &str = "wait";

const TASK_DESCRIPTION: &str = "Start a new subagent on a self-contained task: a new agent with \
    a conversation of its own, which sees only `prompt` and works with its tools until it can \
    answer. The call returns at once with its id; exactly one report arrives later as a \
    message headed with that id and the task title. The reports of the `task` calls of one \
    reply reach you together, once all are done; `wait` for a subagent to read its report as \
    soon as it is ready. With `peers`, the subagents you start are told each other's ids and \
    can message and wait for each other. Continue a subagent with `message`, not another \
    `task`.";

const MESSAGE_DESCRIPTION: &str = "Send an agent you can reach a request, by its id: one of \
    your subagents, or, when you were started with `peers`, a sibling started with `peers`. It \
    reads `text` at once when idle, after its current work when busy, and exactly one report \
    arrives later as a message headed with its id. A message to an agent that asked you \
    something, such as the agent that started you, asks nothing: it answers that agent's \
    oldest request you have read, and your final answer goes only to the requests still \
    open. A refusal lists the agents you can reach.";

const WAIT_DESCRIPTION: &str = "Wait, without ending your turn, until the agent `agent` sends \
    you a message: its report, or a request of its own. Returns once it is here, at once when \
    one came already, or when that agent finishes without one; the message comes right after. \
    Waiting for an agent that is idle, or that waits for you, is refused.";

const RULES: &[&str] = &[
    "Use `task` for work that would fill your context or can run on its own: a broad search \
     or investigation across many files, an independent change, or a second opinion from \
     another model. Not for one quick read or edit.",
    "Write a `task` prompt as a full brief: the goal, what is known, where to look, and what \
     to return. Do not give two subagents changes to the same files.",
    "Each `task` or `message` call is a request, answered by exactly one report that arrives \
     as a message. Meanwhile keep working on what does not need them, or end your turn; call \
     `wait` only when you cannot go on without a report, and never poll.",
    "Never invent, simulate or paraphrase as fact another agent's reply. If the requested \
     interaction is not supported, say so before offering an alternative.",
];

/// What every subagent is told about its role, in a prompt section of its
/// own after its parent's own system prompt and the tools' rules.
pub(super) const SUBAGENT_ROLE: &str = "You are a subagent. Another agent gave you the task in the \
    first message and expects your answer; nobody answers questions while you work, so \
    decide for yourself and say what you assumed. Your last message is sent to that agent \
    as the task's result: make it complete on its own, with file paths, findings and \
    what you changed, and keep it short. That agent may send you follow-ups later; answer \
    each the same way. If you start subagents of your own, end your turn while they work: \
    their reports come back to you, and you answer once they have. Never invent, simulate \
    or paraphrase as fact another agent's reply.";

/// What a subagent started with `peers` is told, after [`SUBAGENT_ROLE`].
pub(super) const PEER_ROLE: &str = "Other subagents of that agent work beside you; your task names the \
    ones started before or with you, and a note names each one started later. `message` sends \
    one of them a request by its id, or answers one it sent you; your last message answers the \
    requests still open. To wait for a peer's message, call `wait`: never end your turn to wait, \
    since your last message is your answer. Ask only for what you need.";

pub(super) fn add(app: &mut App) {
    let independent = |rules| ToolOptions {
        rules,
        footprint: Some(Footprint::Independent),
    };
    app.add_open_tool(TASK, TASK_DESCRIPTION, independent(RULES), on_task)
        .add_open_tool(MESSAGE, MESSAGE_DESCRIPTION, independent(&[]), on_message)
        .add_open_tool(WAIT, WAIT_DESCRIPTION, independent(&[]), on_wait)
        .add_systems(Update, send_briefs.before(AgentLoop));
}

/// The arguments of a `task` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct TaskArgs {
    /// A short title for the task, 3 to 6 words, shown to the user.
    description: String,
    /// The task in full: the goal, what is known, where to look and what to
    /// return. The subagent sees nothing else.
    prompt: String,
    /// A catalog model as vendor/model for the subagent. Yours when absent.
    model: Option<String>,
    /// The subagent's reasoning setting, such as low or high. Yours when
    /// absent and the model is yours, else the model's default.
    effort: Option<String>,
    /// The tools the subagent may use, from yours. All of yours when absent;
    /// an empty list gives it none.
    tools: Option<Vec<String>>,
    /// When true, the subagent is told the ids of the other subagents you
    /// start with `peers`, and they can `message` and `wait` for each other;
    /// each request is still answered by exactly one report, to the one
    /// that asked. Off when absent.
    #[serde(default)]
    peers: bool,
}

/// The arguments of a `message` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MessageArgs {
    /// The id of one of your subagents, as `task` returned it.
    agent: String,
    /// The follow-up, in full: the subagent reads it as a new request.
    text: String,
}

/// The arguments of a `wait` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WaitArgs {
    /// The id of the agent whose message you wait for.
    agent: String,
}

/// Asks for the subagent of a `task` call: a [`SpawnSubagent`], a child of
/// the call, which answers the call once it settled.
fn on_task(called: On<ToolCalled<TaskArgs>>, mut commands: Commands) {
    let args = &called.args;
    let request = SpawnSubagent {
        title: args.description.clone(),
        brief: args.prompt.clone(),
        role: Role::Task,
        model: args.model.clone(),
        effort: args.effort.clone(),
        tools: args.tools.clone(),
        system_prompt: None,
        peers: args.peers,
    };
    commands.spawn((
        request,
        ForAgent(called.agent),
        ChildOf(called.call),
        TaskCall {
            request: RequestId(called.tool_call.id.to_string()),
            // A call a restart runs again has no reply to batch with.
            batch: called.parent,
        },
    ));
}

/// Sends each subagent started since it last ran its task, naming its
/// peers when it has [`Peers`], before the agent loop, which starts the
/// subagent's turn in the same frame. Peers briefed in an earlier frame,
/// such as those of an earlier reply, learn of it by a note.
fn send_briefs(
    briefs: Query<(Entity, &Brief, &SpawnedBy)>,
    families: Query<&Spawned>,
    peers: Query<(&AgentId, &Subtask, Has<Brief>), With<Peers>>,
    mut commands: Commands,
) {
    let named = |(id, subtask, _): (&AgentId, &Subtask, bool)| {
        format!("`{}` (\"{}\")", id.short(), subtask.0)
    };
    for (child, Brief(brief), parent) in &briefs {
        let mut brief = brief.clone();
        if let Ok(me) = peers.get(child) {
            let family = families
                .get(parent.0)
                .into_iter()
                .flat_map(|family| family.iter());
            let siblings = family.filter(|&sibling| sibling != child);
            let siblings: Vec<_> = siblings
                .filter_map(|sibling| Some((sibling, peers.get(sibling).ok()?)))
                .collect();
            let names: Vec<String> = siblings.iter().map(|(_, peer)| named(*peer)).collect();
            if !names.is_empty() {
                let peers = format!("\n\nYour peers, by id: {}.", names.join(", "));
                brief.message.text.push_str(&peers);
            }
            for (sibling, _) in siblings.iter().filter(|(_, (.., briefing))| !briefing) {
                let text = format!("A new peer works beside you: {}.", named(me));
                let note = Deliver::new(*sibling, text, DeliveryMode::Note);
                commands.trigger(note.with_origin(Origin::plugin("subagents")));
            }
        }
        commands.entity(child).remove::<Brief>();
        commands.trigger(brief);
    }
}

/// An agent a `message` or `wait` call can reach.
struct Reachable {
    entity: Entity,
    id: AgentId,
    title: Option<String>,
    /// For an agent the caller owes a report: the oldest request of it
    /// the caller has read, which a message answers.
    asked: Option<RequestId>,
}

impl Reachable {
    /// Its short id and task title, such as `1a2b3c4d` ("Fix the parser").
    fn named(&self) -> String {
        let title = self.title.as_ref().map(|title| format!(" (\"{title}\")"));
        let asked = self.asked.as_ref().map(|_| " (waiting for your report)");
        let (title, asked) = (title.unwrap_or_default(), asked.unwrap_or_default());
        format!("`{}`{title}{asked}", self.id.short())
    }
}

impl Agents<'_, '_> {
    /// The agent `wanted` names, by id or short id, among those `caller`
    /// can reach: the agents it owes a report, its subagents, and its
    /// siblings when both have [`Peers`]. Else a refusal that lists them.
    fn reachable(&self, caller: Entity, wanted: &str) -> Result<Reachable, String> {
        let Ok(me) = self.agents.get(caller) else {
            return Err("You are not an agent that can reach others.".to_owned());
        };
        // A message to an agent the caller owes answers its oldest request
        // the caller has read, one no longer in its inbox.
        let unread: Vec<&Incoming> = self
            .inboxes
            .iter()
            .filter(|(sent, to, _)| to.0 == caller && sent.mode != DeliveryMode::Note)
            .map(|(sent, ..)| sent)
            .collect();
        let read = |request: &&Request| {
            let named = |sent: &&Incoming| sent.origin.request.as_ref() == Some(&request.id);
            !unread.iter().any(named)
        };
        let owed = self.open.get(caller).into_iter();
        let askers = owed
            .flat_map(|open| open.0.iter().filter(read))
            .filter_map(|request| Some((self.find(&request.asker)?, Some(request.id.clone()))));
        let (spawned, parent) = me.family;
        // Its subagents, not the helpers plugins spawned for it.
        let children = spawned.into_iter().flat_map(|spawned| spawned.iter());
        let children =
            children.filter(|&child| self.agents.get(child).is_ok_and(|a| a.subtask.is_some()));
        let peer = |agent: &AgentDataItem| {
            agent.peers && agent.family.1.map(|of| of.0) == parent.map(|of| of.0)
        };
        let siblings = self
            .agents
            .iter()
            .filter(|agent| me.peers && agent.entity != caller && peer(agent));
        let siblings = siblings.map(|sibling| sibling.entity);
        let mut reach: Vec<Reachable> = Vec::new();
        for (entity, asked) in askers.chain(children.chain(siblings).map(|agent| (agent, None))) {
            if let Ok(agent) = self.agents.get(entity)
                && !reach.iter().any(|reachable| reachable.entity == entity)
            {
                reach.push(Reachable {
                    entity,
                    id: agent.id.clone(),
                    title: agent.subtask.map(|subtask| subtask.0.clone()),
                    asked,
                });
            }
        }
        let wanted = wanted.trim();
        if let Some(at) = reach
            .iter()
            .position(|r| r.id.0 == wanted || r.id.short() == wanted)
        {
            return Ok(reach.swap_remove(at));
        }
        let named: Vec<String> = reach.iter().map(Reachable::named).collect();
        let named = if named.is_empty() {
            "none".to_owned()
        } else {
            named.join(", ")
        };
        Err(format!(
            "`{wanted}` is not an agent you can reach. You can reach: {named}."
        ))
    }

    /// Sends `called`'s text to `target`: to an agent the caller owes, as
    /// the report on its request; to any other, as a new request, unless
    /// that agent waits on the caller's own report and not for this
    /// message.
    fn send(&mut self, called: &ToolCalled<MessageArgs>, target: Reachable) -> Outcome {
        let (caller, text) = (called.agent, called.args.text.trim());
        let waiting = !self.wait_calls(target.entity, caller).is_empty();
        let said = if let Some(request) = &target.asked {
            let closed = self.open.get_mut(caller).ok().and_then(|mut open| {
                let at = open.0.iter().position(|owed| owed.id == *request)?;
                Some(open.0.remove(at))
            });
            self.report(caller, closed.into_iter().collect(), text);
            "as your report on its request. Your final answer goes only to the requests still \
             open."
        } else if !waiting && self.reaches(target.entity, caller) {
            return refused(format!(
                "`{}` is waiting for your report, so it cannot be asked, and nothing was sent. \
                 Put what you would ask or tell it in your answer.",
                target.id.short()
            ));
        } else {
            let request = RequestId(called.tool_call.id.to_string());
            let asked = Request {
                id: request.clone(),
                asker: called.caller.clone(),
                batch: None,
            };
            let busy = self.agents.get(target.entity).is_ok_and(|agent| agent.busy);
            let mut entity = self.commands.entity(target.entity);
            let mut open = entity.entry::<OpenRequests>();
            open.or_default().and_modify(|mut open| open.0.push(asked));
            if !busy {
                entity.insert(EffectParent(called.effect));
            }
            let origin = Origin::agent(called.caller.clone(), Some(request));
            let mode = match waiting {
                true => DeliveryMode::Steer,
                false => DeliveryMode::Queue,
            };
            let request = Deliver::new(target.entity, text, mode);
            self.commands.trigger(request.with_origin(origin));
            match (waiting, busy) {
                (true, _) => "now; it was waiting for you.",
                (false, true) => "to read after its current work.",
                (false, false) => "to work on in the background.",
            }
        };
        answered(format!("Sent to {} {said}", target.named()))
    }

    /// Whether a message from `from` came to `agent` that its model has
    /// not read yet: `Some(true)` once `agent` accepted one, which then
    /// goes with its next model call even if it was to wait for the end
    /// of its turn; `Some(false)` while one is on its way.
    fn came(&mut self, agent: Entity, from: &AgentId) -> Option<bool> {
        let mut came = None;
        for (mut sent, to, accepted) in &mut self.inboxes {
            if to.0 != agent || sent.origin.from.as_ref() != Some(from) {
                continue;
            }
            came = Some(came == Some(true) || accepted);
            if accepted && sent.mode == DeliveryMode::Queue {
                sent.mode = DeliveryMode::Steer;
            }
        }
        came
    }
}

/// Sends a `message` call's text to an agent the caller can reach, as
/// [`Agents::send`] says, and answers the call.
fn on_message(called: On<ToolCalled<MessageArgs>>, mut agents: Agents) {
    let output = match agents.reachable(called.agent, &called.args.agent) {
        Err(why) => refused(format!("{why} Nothing was sent.")),
        Ok(_) if called.args.text.trim().is_empty() => {
            refused("`text` must not be empty. Nothing was sent.".to_owned())
        }
        Ok(target) => agents.send(&called, target),
    };
    agents.commands.entity(called.call).insert_if_new(output);
}

/// Waits for a message from an agent the caller can reach: ends at once
/// when the caller accepted one already, else stays open until it accepts
/// one or that agent finishes. Refused when the agent is idle, or waits on
/// the caller, and no message from it is on its way.
fn on_wait(called: On<ToolCalled<WaitArgs>>, mut agents: Agents) {
    let (call, caller) = (called.call, called.agent);
    let target = match agents.reachable(caller, &called.args.agent) {
        Ok(target) => target,
        Err(why) => {
            agents.commands.entity(call).insert_if_new(refused(why));
            return;
        }
    };
    let short = target.id.short();
    let working = agents.agents.get(target.entity).is_ok_and(|agent| {
        let waits = matches!(
            agent.lifecycle,
            Some(Lifecycle::Working | Lifecycle::WaitingOn(_))
        );
        agent.busy || waits
    });
    let output = match agents.came(caller, &target.id) {
        Some(true) => answered(format!("`{short}` sent you a message; it follows.")),
        None if !working => refused(format!(
            "`{short}` is idle, so nothing will come from it; `message` it."
        )),
        None if agents.reaches(target.entity, caller) => refused(format!(
            "`{short}` is waiting for you, so waiting for it would never end."
        )),
        // A message on its way ends the wait once the caller accepts it.
        Some(false) | None => {
            agents.commands.entity(call).insert(Awaits(target.entity));
            return;
        }
    };
    agents.commands.entity(call).insert_if_new(output);
}
