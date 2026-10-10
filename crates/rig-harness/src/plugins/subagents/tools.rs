//! The `task`, `message` and `wait` tools.

use rig_ecs::inbox::Pending;
use rig_ecs::tools::ToolDef;
use schemars::JsonSchema;
use serde::Deserialize;

use super::lifecycle::{AgentDataItem, Agents};
use super::{Awaits, Lifecycle, OpenRequests, Peers, Request, Subtask, answered, refused};
use crate::prelude::*;

/// The tool that starts a subagent.
pub const TASK: &str = "task";

/// The tool that sends a subagent or a peer a request, or an asker a
/// report.
pub const MESSAGE: &str = "message";

/// The tool that waits for an agent's message without ending the turn.
pub const WAIT: &str = "wait";

/// How deep subagents nest: an agent nothing spawned is at depth 0, and an
/// agent at this depth cannot start subagents of its own.
pub const MAX_DEPTH: usize = 2;

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

/// What every subagent is told about its role, after its parent's own
/// system prompt.
const SUBAGENT_ROLE: &str = "\n\nYou are a subagent. Another agent gave you the task in the \
    first message and expects your answer; nobody answers questions while you work, so \
    decide for yourself and say what you assumed. Your last message is sent to that agent \
    as the task's result: make it complete on its own, with file paths, findings and \
    what you changed, and keep it short. That agent may send you follow-ups later; answer \
    each the same way. If you start subagents of your own, end your turn while they work: \
    their reports come back to you, and you answer once they have. Never invent, simulate \
    or paraphrase as fact another agent's reply.";

/// What a subagent started with `peers` is told, after [`SUBAGENT_ROLE`].
const PEER_ROLE: &str = "\n\nOther subagents of that agent work beside you; your task names the \
    ones started with you. `message` sends one of them a request by its id, or answers one it \
    sent you; your last message answers the requests still open. To wait for a peer's message, \
    call `wait`: never end your turn to wait, since your last message is your answer. Ask only \
    for what you need.";

pub(super) fn add(app: &mut App) {
    let independent = |rules| ToolOptions {
        rules,
        footprint: Footprint::Independent,
    };
    app.add_open_tool(TASK, TASK_DESCRIPTION, independent(RULES), on_task)
        .add_open_tool(MESSAGE, MESSAGE_DESCRIPTION, independent(&[]), on_message)
        .add_open_tool(WAIT, WAIT_DESCRIPTION, independent(&[]), on_wait)
        .add_systems(PostUpdate, send_briefs);
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

/// On a subagent `task` just started: its task, sent once the reply's other
/// tasks started too, so that it names its peers.
#[derive(Component)]
struct Brief(Deliver);

/// The subagent's model, reasoning setting and tools of a `task` call,
/// checked against its parent's model and reasoning setting, its depth
/// (how many agents above it spawned it) and its tools by name.
fn settle(
    args: &TaskArgs,
    (parent, depth): ((Option<&ModelChoice>, Effort), usize),
    tools: &[&str],
    models: &Models,
) -> Result<(ModelChoice, Effort, Vec<String>), String> {
    if args.description.trim().is_empty() || args.prompt.trim().is_empty() {
        return Err("`description` and `prompt` must not be empty".to_owned());
    }
    let (model, effort) = ModelChoice::inherit(
        &models.0,
        parent,
        args.model.as_deref(),
        args.effort.as_deref(),
    )?;
    let model = model.ok_or("You have no model to give the subagent; name one in `model`")?;
    let may_delegate = depth + 1 < MAX_DEPTH;
    // A peer keeps `message` and `wait` for its siblings even where it
    // cannot delegate.
    let delegates = |name: &str| name == TASK || (!args.peers && (name == MESSAGE || name == WAIT));
    let mut tools: Vec<String> = match &args.tools {
        None => tools
            .iter()
            .filter(|name| may_delegate || !delegates(name))
            .map(|name| (*name).to_owned())
            .collect(),
        Some(asked) => {
            for name in asked {
                if !tools.contains(&name.as_str()) {
                    return Err(format!(
                        "`{name}` is not one of your tools ({})",
                        tools.join(", ")
                    ));
                }
                if !may_delegate && delegates(name) {
                    return Err(format!(
                        "A subagent at depth {} cannot start subagents of its own; leave \
                         `{name}` out",
                        depth + 1
                    ));
                }
            }
            asked.clone()
        }
    };
    for peer_tool in [MESSAGE, WAIT].into_iter().filter(|_| args.peers) {
        if !tools.iter().any(|name| name == peer_tool) {
            tools.push(peer_tool.to_owned());
        }
    }
    Ok((model, effort, tools))
}

/// Starts the subagent of a `task` call and answers the call with its id,
/// or with why none started. A subagent on its parent's model shares the
/// parent's connection.
fn on_task(
    called: On<ToolCalled<TaskArgs>>,
    agents: Query<(
        &ToolAccess,
        Option<&ModelChoice>,
        &Effort,
        &SystemPrompt,
        Option<&Connection>,
    )>,
    lineage: Query<&SpawnedBy>,
    tools: Query<&ToolDef>,
    models: Res<Models>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (call, caller, args) = (called.call, called.agent, &called.args);
    let Ok((access, model, &effort, prompt, connection)) = agents.get(caller) else {
        return;
    };
    let depth = lineage.iter_ancestors::<SpawnedBy>(caller).count();
    let names = tools.iter().map(|def| def.0.name.as_str());
    let mine: Vec<&str> = names.filter(|name| access.allows(name)).collect();
    let (choice, effort, allowed) = match settle(args, ((model, effort), depth), &mine, &models) {
        Ok(settled) => settled,
        Err(why) => {
            let why = format!("{why}. No subagent was started; fix the call and send it again.");
            commands.entity(call).insert_if_new(refused(why));
            return;
        }
    };
    let (task, id) = (args.description.trim().to_owned(), AgentId::default());
    let request = RequestId(called.run.call.id.to_string());
    // A call a restart runs again has no reply to batch with.
    let batch = called.run.parent;
    let role = if args.peers { PEER_ROLE } else { "" };
    let mut child = commands.spawn_empty();
    // Before the model choice, so connecting it keeps this connection.
    if let Some(connection) = connection.filter(|_| model == Some(&choice)) {
        child.insert(connection.clone());
    }
    let brief = Deliver {
        entity: child.id(),
        text: args.prompt.trim().to_owned(),
        origin: Origin::agent(called.caller.clone(), Some(request.clone())).titled(task.clone()),
        mode: DeliveryMode::Queue,
        attachments: Vec::new(),
    };
    child.insert((
        Agent,
        id.clone(),
        SpawnedBy(caller),
        Subtask(task.clone()),
        OpenRequests(vec![Request {
            id: request.clone(),
            asker: called.caller.clone(),
            batch,
        }]),
        Lifecycle::Working,
        choice,
        effort,
        ToolAccess::Only(allowed),
        SystemPrompt(format!("{}{SUBAGENT_ROLE}{role}", prompt.0)),
        EffectParent(called.effect),
        Brief(brief),
    ));
    child.insert_if(Peers, || args.peers);
    wake.wake();
    let together = match batch {
        Some(_) => " together with the reports of the other tasks of this reply, once all are done",
        None => "",
    };
    let said = format!(
        "Started subagent `{}` on \"{}\". It works in the background; its report will arrive as \
         a message{together}. Continue it with `message`; /agents shows it.",
        id.short(),
        task,
    );
    commands.entity(call).insert_if_new(answered(said));
}

/// Sends each subagent started this frame its task, naming its peers when
/// it has [`Peers`].
fn send_briefs(
    briefs: Query<(Entity, &Brief, &SpawnedBy, Has<Peers>)>,
    families: Query<&Spawned>,
    peers: Query<(&AgentId, &Subtask), With<Peers>>,
    mut commands: Commands,
) {
    for (child, Brief(brief), parent, is_peer) in &briefs {
        let siblings = families
            .get(parent.0)
            .into_iter()
            .flat_map(|family| family.iter());
        let siblings = siblings.filter(|&sibling| is_peer && sibling != child);
        let named: Vec<String> = siblings
            .filter_map(|sibling| peers.get(sibling).ok())
            .map(|(id, subtask)| format!("`{}` (\"{}\")", id.short(), subtask.0))
            .collect();
        let mut brief = brief.clone();
        if !named.is_empty() {
            let peers = format!("\n\nYour peers, by id: {}.", named.join(", "));
            brief.text.push_str(&peers);
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
        let inbox = self.inboxes.get(caller).ok();
        let unread: Vec<&Pending> = inbox
            .into_iter()
            .flat_map(|inbox| inbox.steering.iter().chain(&inbox.queued))
            .collect();
        let read = |request: &&Request| {
            let named = |sent: &&Pending| sent.origin.request.as_ref() == Some(&request.id);
            !unread.iter().any(named)
        };
        let owed = self.open.get(caller).into_iter();
        let askers = owed
            .flat_map(|open| open.0.iter().filter(read))
            .filter_map(|request| Some((self.find(&request.asker)?, Some(request.id.clone()))));
        let (spawned, parent) = me.family;
        let children = spawned.into_iter().flat_map(|spawned| spawned.iter());
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
    fn send(&mut self, called: &ToolCalled<MessageArgs>, target: Reachable) -> ToolOutput {
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
            let request = RequestId(called.run.call.id.to_string());
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
            self.commands.trigger(Deliver {
                entity: target.entity,
                text: text.to_owned(),
                origin,
                mode: if waiting {
                    DeliveryMode::Steer
                } else {
                    DeliveryMode::Queue
                },
                attachments: Vec::new(),
            });
            match (waiting, busy) {
                (true, _) => "now; it was waiting for you.",
                (false, true) => "to read after its current work.",
                (false, false) => "to work on in the background.",
            }
        };
        answered(format!("Sent to {} {said}", target.named()))
    }

    /// Whether a message from `from` came to `agent` and is not read yet;
    /// one waiting for the end of its turn goes with its next model call.
    fn came(&mut self, agent: Entity, from: &AgentId) -> bool {
        let Ok(mut inbox) = self.inboxes.get_mut(agent) else {
            return false;
        };
        let sent_by = |pending: &Pending| pending.origin.from.as_ref() == Some(from);
        let (sent, rest): (Vec<_>, Vec<_>) = inbox.queued.drain(..).partition(sent_by);
        inbox.queued.extend(rest);
        let came = !sent.is_empty();
        inbox.steering.extend(sent);
        came || inbox.steering.iter().chain(&inbox.notes).any(sent_by)
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
/// when one came already, else stays open until one comes or that agent
/// finishes. Refused when the agent is idle, or waits on the caller.
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
    let output = if agents.came(caller, &target.id) {
        answered(format!("`{short}` sent you a message; it follows."))
    } else if !working {
        refused(format!(
            "`{short}` is idle, so nothing will come from it; `message` it."
        ))
    } else if agents.reaches(target.entity, caller) {
        refused(format!(
            "`{short}` is waiting for you, so waiting for it would never end."
        ))
    } else {
        agents.commands.entity(call).insert(Awaits(target.entity));
        return;
    };
    agents.commands.entity(call).insert_if_new(output);
}
