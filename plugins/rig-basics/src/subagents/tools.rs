//! The `task`, `message` and `wait` tools.

use schemars::JsonSchema;
use serde::Deserialize;

use super::lifecycle::{AgentDataItem, Agents};
use super::spawn::{Brief, Role, SpawnSubagent, TaskCall};
use super::{
    Awaits, Lifecycle, ORIGIN, OpenRequests, Peers, Request, SubagentsPlugin, Subtask, answered,
    refused,
};
use rig_harness::prelude::*;

/// The tool that starts a subagent.
pub const TASK: &str = "task";

/// The tool that sends a subagent or a peer a request, or an asker a
/// report.
pub const MESSAGE: &str = "message";

/// The tool that waits for an agent's message without ending the turn.
pub const WAIT: &str = "wait";

const TASK_DESCRIPTION: &str = "Start a new subagent on a self-contained task: a new agent with \
    a conversation of its own, which sees only `prompt` and `environment` and works with its \
    tools until it can answer. It starts in your working directory. The call returns at once \
    with its id; its report, its final answer, arrives as a message headed with that id and \
    the task title as soon as it is done. It may `message` you a question meanwhile. With \
    `peers`, the subagents you start are told each other's ids and can message and wait for \
    each other. Continue a subagent with `message`, not another `task`. Use `task` for work \
    that would fill your context or can run on its own: a broad search or investigation \
    across many files, an independent change, or a second opinion from another model; not for \
    one quick read or edit. Write `prompt` as a full brief: the goal, what is known, where to \
    look, and what to return. Put what it must know of the environment and cannot see in \
    `environment`: where the work is, how to reach other machines (the exact command), what \
    is missing locally. Do not give two subagents changes to the same files. Each `task` or \
    `message` call is a request, answered by exactly one report: meanwhile keep working on \
    what does not need it, or end your turn; call `wait` only when you cannot go on without a \
    report, and never poll. Never invent, simulate or paraphrase as fact another agent's \
    reply; if the requested interaction is not supported, say so before offering an \
    alternative.";

const MESSAGE_DESCRIPTION: &str = "Send `text` to an agent you can reach, by its id: one of your \
    subagents, an agent that asked you something (such as the agent that started you), or, \
    when you were started with `peers`, a sibling started with `peers`. It reads it with its \
    next model call, even while it works, or starts on it when idle. To one of your subagents \
    or a peer it is a request, which its next final answer reports on. To an agent that asked \
    you something it is only a message, such as a question: your report to it is your final \
    answer, when your turn ends. A refusal lists the agents you can reach.";

const WAIT_DESCRIPTION: &str = "Wait, without ending your turn, until the agent `agent` sends \
    you a message: its report, an answer, or a request of its own. Returns once it is here, at \
    once when one came already, or when that agent ends its turn without one; the message \
    comes right after. Waiting for an agent that is idle with nothing from you to read, or \
    that waits for you, is refused.";

/// What every subagent is told about its role, in a prompt section of its
/// own after its parent's own system prompt.
pub(super) const SUBAGENT_ROLE: &str = "You are a subagent. Another agent gave you the task in the \
    first message and expects your answer. Decide what you can yourself and say what you \
    assumed; when you cannot go on without that agent, `message` it your question and `wait` \
    for its answer. A message is not your report: your final answer, when your turn ends, is \
    sent to that agent as the task's result, so make it complete on its own, with file paths, \
    findings and what you changed, and keep it short. That agent may send you corrections \
    while you work and follow-ups later; answer each in your final answer. If you start \
    subagents of your own, end your turn while they work: their reports come back to you, \
    and you answer once they have. Never invent, simulate or paraphrase as fact another \
    agent's reply.";

/// What a subagent started with `peers` is told, after [`SUBAGENT_ROLE`].
pub(super) const PEER_ROLE: &str = "Other subagents of that agent work beside you; your task names the \
    ones started before or with you. Peers may also start later, in a later reply of that \
    agent: a message names each one when it starts. `message` sends one of them a request by its \
    id, or an answer to one it sent you; your final answer also reports on every request you \
    were sent. To wait for a peer's message, call `wait`, which returns once that peer messages \
    you: never end your turn to wait, since your final answer is your report. Ask only for \
    what you need.";

pub(super) fn add(app: &mut App) {
    let independent = || ToolOptions {
        footprint: Some(Touches::Independent),
    };
    app.add_open_tool(TASK, TASK_DESCRIPTION, independent(), on_task)
        .add_open_tool(MESSAGE, MESSAGE_DESCRIPTION, independent(), on_message)
        .add_open_tool(WAIT, WAIT_DESCRIPTION, independent(), on_wait)
        .add_systems(
            Update,
            send_briefs
                .run_if(any_with_component::<Brief>)
                .before(AgentLoop),
        );
}

/// The arguments of a `task` call. Any argument besides these is one
/// another plugin added to the tool's parameters (the kernel refuses one
/// they do not declare), and goes to the request's
/// [`extra`](SpawnSubagent::extra).
#[derive(Deserialize, JsonSchema)]
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
    /// The tools the subagent may use, from yours, besides `message` and
    /// `wait`, which it always has. All of yours when absent; an empty list
    /// gives it no others.
    tools: Option<Vec<String>>,
    /// What the subagent must know of its environment that it cannot see:
    /// where the work is, how to reach other machines (the exact command),
    /// what is missing locally, such as a toolchain. Its own subagents get
    /// it too. What you were given when absent.
    environment: Option<String>,
    /// When true, the subagent is told the ids of the other subagents you
    /// start with `peers`, and they can `message` and `wait` for each other;
    /// each request is still answered by exactly one report, to the one
    /// that asked. Off when absent.
    #[serde(default)]
    peers: bool,
    /// The arguments another plugin added.
    #[serde(flatten)]
    #[schemars(skip)]
    extra: serde_json::Map<String, serde_json::Value>,
}

/// The arguments of a `message` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MessageArgs {
    /// The id of an agent you can reach, such as one of your subagents, as
    /// `task` returned it.
    agent: String,
    /// The message, in full: a follow-up or correction to a subagent, or a
    /// question to the agent that asked you.
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
fn on_task(
    called: On<ToolCalled<TaskArgs>>,
    me: Option<Res<PluginEntity<SubagentsPlugin>>>,
    mut commands: Commands,
) {
    let args = &called.args;
    let request = SpawnSubagent {
        title: args.description.clone(),
        brief: args.prompt.clone(),
        role: Role::Task,
        model: args.model.clone(),
        effort: args.effort.clone(),
        tools: args.tools.clone(),
        system_prompt: None,
        environment: args.environment.clone(),
        peers: args.peers,
        once: false,
        id: None,
        extra: args.extra.clone(),
    };
    let mut spawned = commands.spawn((
        request,
        ForAgent(called.agent),
        ChildOf(called.call),
        TaskCall(RequestId(called.tool_call.id.to_string())),
    ));
    if let Some(me) = me {
        spawned.insert(ProvidedBy(me.entity()));
    }
}

/// Sends each subagent started since it last ran its task, naming its
/// peers when it has [`Peers`], before the agent loop, which starts the
/// subagent's turn in the same frame. Peers briefed in an earlier frame,
/// such as those of an earlier reply, learn of it by an Inform message.
fn send_briefs(
    briefs: Query<(Entity, &Brief, &SpawnedBy)>,
    families: Query<&Spawned>,
    peers: Query<(&AgentId, &AgentTitle, Has<Brief>), (With<Peers>, With<Subtask>)>,
    mut commands: Commands,
) {
    let named = |(id, title, _): (&AgentId, &AgentTitle, bool)| {
        format!("`{}` (\"{}\")", id.short(), title.0)
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
                brief.message.text_mut().push_str(&peers);
            }
            for (sibling, _) in siblings.iter().filter(|(_, (.., briefing))| !briefing) {
                let text = format!("A new peer works beside you: {}.", named(me));
                let inform = Deliver::new(*sibling, text, DeliveryMode::Inform);
                commands.trigger(inform.with_author(Author::plugin(ORIGIN)));
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
    /// Whether the caller owes it a report: a message to it is no request.
    owed: bool,
}

impl Reachable {
    /// Its short id and task title, such as `1a2b3c4d` ("Fix the parser").
    fn named(&self) -> String {
        let title = self.title.as_ref().map(|title| format!(" (\"{title}\")"));
        let owed = if self.owed {
            " (waiting for your report)"
        } else {
            ""
        };
        format!("`{}`{}{owed}", self.id.short(), title.unwrap_or_default())
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
        let owed = self.open.get(caller).into_iter();
        let askers = owed
            .flat_map(|open| open.0.iter())
            .filter_map(|request| Some((self.find(&request.asker)?, true)));
        let (spawned, parent) = (me.children, me.parent);
        // Its subagents, not the helpers plugins spawned for it.
        let children = spawned.into_iter().flat_map(|spawned| spawned.iter());
        let children = children.filter(|&child| self.agents.get(child).is_ok_and(|a| a.subtask));
        let peer = |agent: &AgentDataItem| {
            agent.peers && agent.parent.map(|of| of.0) == parent.map(|of| of.0)
        };
        let siblings = self
            .agents
            .iter()
            .filter(|agent| me.peers && agent.entity != caller && peer(agent));
        let siblings = siblings.map(|sibling| sibling.entity);
        let mut reach: Vec<Reachable> = Vec::new();
        for (entity, owed) in askers.chain(children.chain(siblings).map(|agent| (agent, false))) {
            if let Ok(agent) = self.agents.get(entity)
                && !reach.iter().any(|reachable| reachable.entity == entity)
            {
                reach.push(Reachable {
                    entity,
                    id: agent.id.clone(),
                    title: agent
                        .title
                        .filter(|_| agent.subtask)
                        .map(|title| title.0.clone()),
                    owed,
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
        let named = match (named.is_empty(), me.peers) {
            (false, _) => named.join(", "),
            (true, false) => "none".to_owned(),
            (true, true) => "none yet. Peers may start later, in a later reply of the agent \
                             that started you: a message names each one when it starts, and `wait` \
                             returns once a peer messages you"
                .to_owned(),
        };
        Err(format!(
            "`{wanted}` is not an agent you can reach. You can reach: {named}."
        ))
    }

    /// Sends `called`'s text to `target`, with its next model call: to an
    /// agent the caller owes, as a message; to any other, as a new request,
    /// unless that agent waits on the caller's own report and not for this
    /// message.
    fn send(&mut self, called: &ToolCalled<MessageArgs>, target: Reachable) -> Outcome {
        let (caller, text) = (called.agent, called.args.text.trim());
        let waiting = !self.wait_calls(target.entity, caller).is_empty();
        let busy = self.agents.get(target.entity).is_ok_and(|agent| agent.busy);
        let said = match (waiting, busy) {
            (true, _) => "now; it was waiting for you.",
            (false, true) => "to read with its next step.",
            (false, false) => "to work on in the background.",
        };
        let request = if target.owed {
            None
        } else if !waiting && self.reaches(target.entity, caller, true) {
            return refused(format!(
                "`{}` is waiting for your report, so it cannot be asked, and nothing was sent. \
                 Put what you would ask or tell it in your answer.",
                target.id.short()
            ));
        } else {
            let request = RequestId(called.tool_call.id.to_string());
            let asked = Request {
                id: request.clone(),
                asker: called.agent_id.clone(),
            };
            let mut entity = self.commands.entity(target.entity);
            let mut open = entity.entry::<OpenRequests>();
            open.or_default().and_modify(|mut open| open.0.push(asked));
            if !busy {
                entity.insert(EffectParent(called.effect));
            }
            Some(request)
        };
        let report = match request {
            Some(_) => "",
            None => " It is a message, not your report: your final answer is, when your turn ends.",
        };
        let author = Author::agent(called.agent_id.clone(), request);
        let message = Deliver::new(target.entity, text, DeliveryMode::Steer);
        self.commands.trigger(message.with_author(author));
        answered(format!("Sent to {} {said}{report}", target.named()))
    }

    /// Whether a message from `from` came to `agent` that its model has
    /// not read yet. An accepted one goes with its next model call even if
    /// it was to wait for the end of its turn.
    fn came(&mut self, agent: Entity, from: &AgentId) -> Arrival {
        let mut came = Arrival::Nothing;
        for (mut sent, to, accepted) in &mut self.inboxes {
            if to.0 != agent || sent.author.from.as_deref() != Some(from.0.as_str()) {
                continue;
            }
            let this = if accepted {
                Arrival::Accepted
            } else {
                Arrival::OnItsWay
            };
            came = came.max(this);
            if accepted && sent.mode == DeliveryMode::Queue {
                sent.mode = DeliveryMode::Steer;
            }
        }
        came
    }
}

/// How far a message from one agent to another got, as [`Agents::came`]
/// says.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Arrival {
    /// None came.
    Nothing,
    /// One is on its way.
    OnItsWay,
    /// One was accepted.
    Accepted,
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
    agents.answer(called.call, output);
}

/// Waits for a message from an agent the caller can reach: ends at once
/// when the caller accepted one already, else stays open until it accepts
/// one or that agent's turn ends without one. Refused when that would
/// never end: the agent is idle with nothing of the caller's to read, or
/// it waits on the caller.
fn on_wait(called: On<ToolCalled<WaitArgs>>, mut agents: Agents) {
    let (call, caller) = (called.call, called.agent);
    let target = match agents.reachable(caller, &called.args.agent) {
        Ok(target) => target,
        Err(why) => {
            agents.answer(call, refused(why));
            return;
        }
    };
    let short = target.id.short();
    let Ok(agent) = agents.agents.get(target.entity) else {
        return;
    };
    let busy = agent.busy;
    let parked = matches!(
        agent.lifecycle,
        Some(Lifecycle::Working | Lifecycle::WaitingOn(_))
    );
    // A message of the caller's it has yet to read starts or steers its
    // turn, and ends a wait of its for the caller.
    let asked = agents.on_its_way(target.entity, &called.agent_id);
    let output = match agents.came(caller, &target.id) {
        Arrival::Accepted => answered(format!("`{short}` sent you a message; it follows.")),
        Arrival::Nothing if !busy && !parked && !asked => refused(format!(
            "`{short}` is idle, so nothing will come from it; `message` it."
        )),
        // A turn that runs ends, which ends the wait; an agent whose turn
        // ended waits for the reports it is owed.
        Arrival::Nothing if !asked && agents.reaches(target.entity, caller, !busy) => refused(
            format!("`{short}` is waiting for you, so waiting for it would never end."),
        ),
        Arrival::OnItsWay | Arrival::Nothing => {
            agents.commands.entity(call).insert(Awaits(target.entity));
            return;
        }
    };
    agents.answer(call, output);
}
