//! The built-in subagents, written only against the core's public
//! primitives: open tools ([`AppToolsExt::add_open_tool`]), spawned agents
//! ([`SpawnedBy`]), [`Deliver`] with an [`Origin`], [`TurnEnded`] and saved
//! components. Leave [`SubagentsPlugin`] out of the app, or replace
//! it, and the core has no subagents.
//!
//! `task` spawns a child agent, in process, with its own model, reasoning
//! setting and tools, and answers at once with the child's id. `message`
//! sends one of the caller's own children a follow-up; the child keeps its
//! conversation. Each of the two delivers a request whose id is the call's
//! id, so it doubles as an idempotency key, and each request ends in
//! exactly one report back to the caller: a [`Deliver`] whose origin names
//! the child and the request, with the status done, failed or interrupted.
//! A busy child queues a request. Nothing waits for a child: its report
//! starts a turn of an idle caller, or is queued for a busy one, and the
//! reports that reach the caller together go to its model in one step. A
//! report that only says its request was answered with another one needs
//! no answer: it is a [`DeliveryMode::Note`], read with that other report
//! and starting no turn of its own.
//!
//! Subagents can also work together. A `task` with `peers` set gives the
//! child [`Peers`]: it may send a `message` to its siblings that have
//! [`Peers`] too, and its own report goes to the sibling that asked, as
//! [`PeerRequests`] record. A request to an agent that waits on the
//! sender's own report is refused, and an agent holds its reports only
//! for agents that do not wait on it, so two peers never wait on each
//! other.
//!
//! A child's model calls are recorded under the call that gave it its work
//! ([`EffectParent`]), so the effect log nests a subagent's work under the
//! request. Each child has its own log, whose header names its parent.
//! Despawning an agent despawns its children; a finished child stays, so
//! a view can show it and the user can talk to it. The open requests
//! are saved with the child; after a restart each one is answered as
//! interrupted, and the child is not carried on.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::Message;
use rig_core::message::{ToolCall, ToolResult, ToolResultContent};
use serde::{Deserialize, Serialize};

use crate::agent::{
    ActiveTurn, Agent, AgentId, EffectParent, Effort, ModelChoice, Spawned, SpawnedBy,
    SystemPrompt, ToolAccess, ToolCallRun, TurnEnded, TurnOutcome, answer_text,
};
use crate::inbox::{Deliver, DeliveryMode, Origin, RequestId};
use crate::journal::AppSaveExt;
use crate::models::{self, ModelConnector};
use crate::restore::Restored;
use crate::tools::{
    AppToolsExt, Footprint, OpenCall, ToolCalled, ToolDef, ToolOptions, ToolOutput, failed,
};

/// The tool that starts a subagent.
pub const TASK: &str = "task";

/// The tool that sends one of the caller's subagents a follow-up.
pub const MESSAGE: &str = "message";

/// How deep subagents nest: an agent nothing spawned is at depth 0, and an
/// agent at this depth cannot start subagents of its own.
pub const MAX_DEPTH: usize = 2;

/// The most bytes of a subagent's answer that go back to its parent.
const MAX_ANSWER_BYTES: usize = 50 * 1024;

/// The rule against made-up replies, for every agent that can reach
/// another.
const HONESTY: &str = "Never invent, simulate or paraphrase as fact another agent's reply. If \
    the requested interaction is not supported, say so before offering an alternative.";

const TASK_DESCRIPTION: &str = "Start a new subagent on a self-contained task: a new agent with \
    a conversation of its own, which works with its tools until it can answer. It sees nothing \
    of this conversation, only `prompt`. The call returns at once with the subagent's id and \
    the request's id, as {\"agent\": …, \"request\": …}; the subagent works in the background, \
    and exactly one report for the request arrives later as a message headed as that \
    subagent's output, naming the request. Several `task` calls run side by side. To continue \
    a subagent you already started, use `message` with its id, not another `task`.";

const MESSAGE_DESCRIPTION: &str = "Send a follow-up to one of your own subagents, by the id \
    `task` returned. The subagent continues with its conversation, model and tools, and reads \
    `text` as a new request: at once when idle, after its current work when busy. The call \
    returns at once with the request's id; exactly one report for it arrives later as a \
    message headed as the subagent's output, naming the request. Only subagents you started \
    can be reached, and, when you were started with `peers`, your sibling subagents started \
    with `peers`; any other agent is refused, and the refusal lists the ones you can reach.";

const RULES: &[&str] = &[
    "Use `task` for work that would fill your context or can run on its own: a broad search \
     or investigation across many files, an independent change, or a second opinion from \
     another model. Not for one quick read or edit.",
    "Write a `task` prompt as a full brief: the goal, what is known, where to look, and what \
     to return. Do not give two subagents changes to the same files.",
    "`task` starts a new subagent; `message` continues one you started, by its id, with its \
     conversation kept. Each call is a request, answered by exactly one report that arrives \
     as its own message. Meanwhile keep working on what does not need it, or end your turn; \
     do not poll or wait for it.",
    HONESTY,
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
const PEER_ROLE: &str = "\n\nOther subagents of that agent may work beside you. With \
    `message` you can send one of them that was started with `peers` a request by its id; \
    its report comes back to you as a message. A request one of them sends you is answered \
    by your last message, like your task. A `message` to an id you cannot reach lists the \
    ones you can. Ask only for what you need, and never ask an agent that is waiting for \
    your own report: answer it instead.";

/// Adds the `task` and `message` tools and the reports that answer their
/// requests.
#[derive(Default)]
pub struct SubagentsPlugin;

impl Plugin for SubagentsPlugin {
    fn build(&self, app: &mut App) {
        app.add_open_tool(
            TASK,
            TASK_DESCRIPTION,
            task_parameters(),
            ToolOptions {
                rules: RULES,
                footprint: Footprint::Independent,
            },
            on_task,
        )
        .add_open_tool(
            MESSAGE,
            MESSAGE_DESCRIPTION,
            message_parameters(),
            ToolOptions {
                rules: &[],
                footprint: Footprint::Independent,
            },
            on_message,
        )
        .save_component::<Subtask>()
        .save_component::<Requests>()
        .save_component::<Peers>()
        .save_component::<PeerRequests>()
        .add_observer(name_subagent)
        .add_observer(report_on_turn_end)
        .add_observer(report_restored);
    }
}

/// On a subagent: its task's short title, which names it.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Subtask {
    /// The title.
    pub title: String,
}

/// On a subagent: the requests it was sent that no report answered yet,
/// oldest first.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Requests(pub Vec<RequestId>);

/// On a subagent: it may send a `message` to its siblings that have
/// [`Peers`] too, and they to it. A `task` with `peers` set inserts it;
/// any plugin may too.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Peers;

/// On a subagent: the requests its peers sent it that no report answered
/// yet, oldest first. [`Requests`] holds its parent's.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct PeerRequests(pub Vec<PeerRequest>);

/// A request one peer sent another.
#[derive(Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerRequest {
    /// The request.
    pub request: RequestId,
    /// The peer that sent it, which its report goes to.
    pub from: AgentId,
}

fn task_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A short title for the task, 3 to 6 words, shown to the user."
            },
            "prompt": {
                "type": "string",
                "description": "The task in full: the goal, what is known, where to look and \
                    what to return. The subagent sees nothing else."
            },
            "model": {
                "type": "string",
                "description": "A catalog model as vendor/model for the subagent. Yours when \
                    absent."
            },
            "effort": {
                "type": "string",
                "description": "The subagent's reasoning setting, such as low or high. Yours \
                    when absent and the model is yours, else the model's default."
            },
            "tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "The tools the subagent may use, from yours. All of yours when \
                    absent; an empty list gives it none."
            },
            "peers": {
                "type": "boolean",
                "description": "When true, the subagent can send a `message` to the other \
                    subagents you start with `peers`, and they to it; each request is still \
                    answered by exactly one report, to the one that asked. Off when absent."
            }
        },
        "required": ["description", "prompt"]
    })
}

fn message_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "The id of one of your subagents, as `task` returned it."
            },
            "text": {
                "type": "string",
                "description": "The follow-up, in full: the subagent reads it as a new request."
            }
        },
        "required": ["agent", "text"]
    })
}

/// The arguments of a `task` call.
#[derive(Deserialize)]
struct TaskArgs {
    description: String,
    prompt: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    effort: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    peers: bool,
}

/// The arguments of a `message` call.
#[derive(Deserialize)]
struct MessageArgs {
    agent: String,
    text: String,
}

/// What a subagent starts from: the agent that calls `task`.
struct Parent<'a> {
    model: Option<&'a ModelChoice>,
    effort: Effort,
    /// How many agents above it spawned it.
    depth: usize,
}

/// A `task` call's arguments checked against the parent.
struct Settled {
    task: String,
    instructions: String,
    model: ModelChoice,
    effort: Effort,
    tools: Vec<String>,
    peers: bool,
}

/// The arguments of `call`, or why they do not fit.
fn arguments<T: for<'de> Deserialize<'de>>(call: &ToolCall) -> Result<T, String> {
    serde_json::from_value(serde_json::Value::Object(call.function.arguments.clone()))
        .map_err(|error| format!("The arguments do not fit: {error}"))
}

/// A result for `call`: `data` for programs, then `text` for people.
fn answer(call: &ToolCall, data: serde_json::Value, text: String) -> ToolResult {
    call.result(vec![
        ToolResultContent::json(data),
        ToolResultContent::text(text),
    ])
}

/// Checks a `task` call against its parent. `tools` are the parent's
/// tools by name.
fn settle(
    call: &ToolCall,
    parent: &Parent<'_>,
    tools: &[&str],
    connector: &ModelConnector,
) -> Result<Settled, String> {
    let args: TaskArgs = arguments(call)?;
    let task = args.description.trim().to_owned();
    let instructions = args.prompt.trim().to_owned();
    if task.is_empty() || instructions.is_empty() {
        return Err("`description` and `prompt` must not be empty".to_owned());
    }
    let (model, effort) = models::child_model(
        connector,
        parent.model,
        parent.effort,
        args.model.as_deref(),
    )?;
    let model = model.ok_or("You have no model to give the subagent; name one in `model`")?;
    let effort = match args.effort.as_deref().map(str::trim) {
        Some(name) if !name.is_empty() => {
            let spec = connector
                .resolve(&model.0)
                .ok_or_else(|| format!("The catalog has no model `{}`", model.0))?;
            Effort(models::effort_named(&spec, name)?)
        }
        _ => effort,
    };
    let may_delegate = parent.depth + 1 < MAX_DEPTH;
    // A peer keeps `message` for its siblings even where it cannot delegate.
    let delegates = |name: &str| name == TASK || (name == MESSAGE && !args.peers);
    let mut tools: Vec<String> = match args.tools {
        None => tools
            .iter()
            .filter(|name| may_delegate || !delegates(name))
            .map(|name| (*name).to_owned())
            .collect(),
        Some(asked) => {
            for name in &asked {
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
                        parent.depth + 1
                    ));
                }
            }
            asked
        }
    };
    if args.peers && !tools.iter().any(|name| name == MESSAGE) {
        tools.push(MESSAGE.to_owned());
    }
    Ok(Settled {
        task,
        instructions,
        model,
        effort,
        tools,
        peers: args.peers,
    })
}

/// Starts the subagent of a `task` call and answers the call with its id,
/// or with why none started.
fn on_task(
    called: On<ToolCalled>,
    calls: Query<(&ToolCallRun, Option<&OpenCall>)>,
    agents: Query<(
        &AgentId,
        &ToolAccess,
        Option<&ModelChoice>,
        &Effort,
        &SystemPrompt,
    )>,
    lineage: Query<&SpawnedBy>,
    tools: Query<&ToolDef>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
) {
    let (call, caller) = (called.call, called.agent);
    let Ok((run, open)) = calls.get(call) else {
        return;
    };
    let call_id = &run.call;
    let output = match agents.get(caller) {
        Err(_) => failed(call_id, "The calling agent is gone.".to_owned()),
        Ok((id, access, model, &effort, prompt)) => {
            let parent = Parent {
                model,
                effort,
                depth: lineage.iter_ancestors::<SpawnedBy>(caller).count(),
            };
            let mine: Vec<&str> = tools
                .iter()
                .map(|def| def.0.name.as_str())
                .filter(|name| access.allows(name))
                .collect();
            match settle(call_id, &parent, &mine, &connector) {
                Err(why) => failed(
                    call_id,
                    format!("{why}. No subagent was started; fix the call and send it again."),
                ),
                Ok(settled) => {
                    let child_id = AgentId::default();
                    let request = RequestId(call_id.id.to_string());
                    let role = if settled.peers {
                        format!("{SUBAGENT_ROLE}{PEER_ROLE}")
                    } else {
                        SUBAGENT_ROLE.to_owned()
                    };
                    let mut child = commands.spawn((
                        Agent,
                        child_id.clone(),
                        SpawnedBy(caller),
                        Subtask {
                            title: settled.task.clone(),
                        },
                        Requests(vec![request.clone()]),
                        settled.model,
                        settled.effort,
                        ToolAccess::Only(settled.tools),
                        SystemPrompt(format!("{}{role}", prompt.0)),
                    ));
                    if settled.peers {
                        child.insert(Peers);
                    }
                    if let Some(open) = open {
                        child.insert(EffectParent(open.0.id()));
                    }
                    let child = child.id();
                    commands.trigger(Deliver {
                        entity: child,
                        text: settled.instructions,
                        origin: Origin::agent(id.clone(), Some(request.clone())),
                        mode: DeliveryMode::Queue,
                        attachments: Vec::new(),
                    });
                    answer(
                        call_id,
                        serde_json::json!({
                            "agent": child_id.short(),
                            "request": request.0,
                            "status": "started",
                        }),
                        format!(
                            "Started subagent `{}` on \"{}\". It works in the background; its \
                             report for request {} will arrive as a message. Continue it with \
                             `message`; /agents shows it.",
                            child_id.short(),
                            settled.task,
                            request.0
                        ),
                    )
                }
            }
        }
    };
    commands.entity(call).insert_if_new(ToolOutput(output));
}

/// The open requests of every agent, to tell who waits on whom.
type Debts<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AgentId,
        Option<&'static SpawnedBy>,
        Option<&'static Requests>,
        Option<&'static PeerRequests>,
    ),
>;

/// Who waits on whom, as (waiting, owing) pairs: an agent waits on each
/// subagent that holds an open request of its, and on each peer that
/// holds an open request it sent.
fn waits(debts: &Debts<'_, '_>) -> Vec<(Entity, Entity)> {
    let mut edges = Vec::new();
    for (owing, _, parent, requests, peer_requests) in debts {
        if let (Some(parent), Some(requests)) = (parent, requests)
            && !requests.0.is_empty()
        {
            edges.push((parent.0, owing));
        }
        for peer in peer_requests.into_iter().flat_map(|open| open.0.iter()) {
            if let Some(waiting) = find(debts, &peer.from) {
                edges.push((waiting, owing));
            }
        }
    }
    edges
}

/// The agent with this id.
fn find(debts: &Debts<'_, '_>, id: &AgentId) -> Option<Entity> {
    debts
        .iter()
        .find(|(_, other, ..)| *other == id)
        .map(|(entity, ..)| entity)
}

/// Whether `to` can be reached from `from` along `edges`, in one step or
/// more.
fn reaches<T: Copy + PartialEq>(edges: &[(T, T)], from: T, to: T) -> bool {
    let mut seen = vec![from];
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        for &(start, end) in edges {
            if start != node || seen.contains(&end) {
                continue;
            }
            if end == to {
                return true;
            }
            seen.push(end);
            stack.push(end);
        }
    }
    false
}

/// An agent a `message` call can reach.
struct Reachable<'a> {
    entity: Entity,
    id: &'a AgentId,
    subtask: Option<&'a Subtask>,
    requests: Option<&'a Requests>,
    peer_requests: Option<&'a PeerRequests>,
    busy: bool,
    /// A sibling rather than the caller's own subagent.
    peer: bool,
}

/// Sends a `message` call's text as a request to one of the caller's own
/// subagents, or to a sibling when both have [`Peers`], and answers the
/// call with the request's id. Refuses any other target, and a peer that
/// waits on the caller's own report.
fn on_message(
    called: On<ToolCalled>,
    calls: Query<(&ToolCallRun, Option<&OpenCall>)>,
    callers: Query<(&AgentId, Option<&Spawned>, Option<&SpawnedBy>, Has<Peers>)>,
    families: Query<&Spawned>,
    targets: Query<(
        &AgentId,
        Option<&Subtask>,
        Option<&Requests>,
        Option<&PeerRequests>,
        Has<ActiveTurn>,
        Has<Peers>,
    )>,
    debts: Debts,
    mut commands: Commands,
) {
    let (call, caller) = (called.call, called.agent);
    let Ok((run, open)) = calls.get(call) else {
        return;
    };
    let call_id = &run.call;
    let output = match (callers.get(caller), arguments::<MessageArgs>(call_id)) {
        (Err(_), _) => failed(call_id, "The calling agent is gone.".to_owned()),
        (_, Err(why)) => failed(call_id, format!("{why}. Nothing was sent.")),
        (Ok((id, spawned, parent, is_peer)), Ok(args)) => {
            let wanted = args.agent.trim();
            let text = args.text.trim();
            let reachable = |entity: Entity, peer: bool| {
                let (id, subtask, requests, peer_requests, busy, has_peers) =
                    targets.get(entity).ok()?;
                (!peer || has_peers).then_some(Reachable {
                    entity,
                    id,
                    subtask,
                    requests,
                    peer_requests,
                    busy,
                    peer,
                })
            };
            let siblings = parent
                .filter(|_| is_peer)
                .and_then(|parent| families.get(parent.0).ok());
            let reach: Vec<Reachable<'_>> = spawned
                .into_iter()
                .flat_map(|spawned| spawned.iter())
                .filter_map(|child| reachable(child, false))
                .chain(
                    siblings
                        .into_iter()
                        .flat_map(|siblings| siblings.iter())
                        .filter(|&sibling| sibling != caller)
                        .filter_map(|sibling| reachable(sibling, true)),
                )
                .collect();
            let target = reach
                .iter()
                .find(|target| target.id.0 == wanted || target.id.short() == wanted);
            match target {
                None => failed(
                    call_id,
                    format!(
                        "`{wanted}` is not an agent you can reach, so nothing was sent. {}",
                        listing(&reach)
                    ),
                ),
                Some(_) if text.is_empty() => failed(
                    call_id,
                    "`text` must not be empty. Nothing was sent.".to_owned(),
                ),
                Some(target) if target.peer && reaches(&waits(&debts), target.entity, caller) => {
                    failed(
                        call_id,
                        format!(
                            "`{}` is waiting for your report, so it cannot be asked, and \
                             nothing was sent. Put what you would ask or tell it in your answer.",
                            target.id.short()
                        ),
                    )
                }
                Some(target) => {
                    let request = RequestId(call_id.id.to_string());
                    let mut entity = commands.entity(target.entity);
                    if target.peer {
                        let mut open_requests = target
                            .peer_requests
                            .map(|open| open.0.clone())
                            .unwrap_or_default();
                        open_requests.push(PeerRequest {
                            request: request.clone(),
                            from: id.clone(),
                        });
                        entity.insert(PeerRequests(open_requests));
                    } else {
                        let mut open_requests = target
                            .requests
                            .map(|open| open.0.clone())
                            .unwrap_or_default();
                        open_requests.push(request.clone());
                        entity.insert(Requests(open_requests));
                    }
                    if !target.busy
                        && let Some(open) = open
                    {
                        entity.insert(EffectParent(open.0.id()));
                    }
                    commands.trigger(Deliver {
                        entity: target.entity,
                        text: text.to_owned(),
                        origin: Origin::agent(id.clone(), Some(request.clone())),
                        mode: DeliveryMode::Queue,
                        attachments: Vec::new(),
                    });
                    let (status, when) = if target.busy {
                        (
                            "queued",
                            "It is busy, so it reads this after its current work.",
                        )
                    } else {
                        ("started", "It works on it in the background.")
                    };
                    let kind = if target.peer { "peer" } else { "subagent" };
                    answer(
                        call_id,
                        serde_json::json!({
                            "agent": target.id.short(),
                            "request": request.0,
                            "status": status,
                        }),
                        format!(
                            "Sent to {kind} `{}`. {when} Its report for request {} will \
                             arrive as a message.",
                            target.id.short(),
                            request.0
                        ),
                    )
                }
            }
        }
    };
    commands.entity(call).insert_if_new(ToolOutput(output));
}

/// The agents a `message` call can reach, for a refusal.
fn listing(reach: &[Reachable<'_>]) -> String {
    let named = |target: &Reachable<'_>| match target.subtask {
        Some(subtask) => format!("`{}` (\"{}\")", target.id.short(), subtask.title),
        None => format!("`{}`", target.id.short()),
    };
    let mine: Vec<String> = reach
        .iter()
        .filter(|target| !target.peer)
        .map(named)
        .collect();
    let peers: Vec<String> = reach
        .iter()
        .filter(|target| target.peer)
        .map(named)
        .collect();
    let mine = if mine.is_empty() {
        "You have no subagents; start one with `task`.".to_owned()
    } else {
        format!("Your subagents are: {}.", mine.join(", "))
    };
    if peers.is_empty() {
        mine
    } else {
        format!("{mine} Your peers are: {}.", peers.join(", "))
    }
}

/// Names a subagent by its task's title, when it is spawned or restored.
fn name_subagent(inserted: On<Insert<Subtask>>, subtasks: Query<&Subtask>, mut commands: Commands) {
    if let Ok(subtask) = subtasks.get(inserted.entity) {
        commands
            .entity(inserted.entity)
            .insert(Name::new(subtask.title.clone()));
    }
}

/// How a request ended, for its report.
enum Status {
    /// Answered, with the answer.
    Done(String),
    /// Failed, for the reason given.
    Failed(String),
    /// Stopped before an answer.
    Interrupted,
}

/// The open requests of a subagent by the agent each one's report goes
/// to, oldest first within each: its parent's, then each peer's in the
/// order of its first request. A peer that is gone gets none.
fn by_asker(
    parent: Entity,
    requests: Option<&Requests>,
    peer_requests: Option<&PeerRequests>,
    debts: &Debts<'_, '_>,
) -> Vec<(Entity, Vec<RequestId>)> {
    let mut askers: Vec<(Entity, Vec<RequestId>)> = Vec::new();
    if let Some(requests) = requests
        && !requests.0.is_empty()
    {
        askers.push((parent, requests.0.clone()));
    }
    for peer in peer_requests.into_iter().flat_map(|open| open.0.iter()) {
        let Some(asker) = find(debts, &peer.from) else {
            continue;
        };
        match askers.iter_mut().find(|(entity, _)| *entity == asker) {
            Some((_, ids)) => ids.push(peer.request.clone()),
            None => askers.push((asker, vec![peer.request.clone()])),
        }
    }
    askers
}

/// Reports on every open request of a subagent whose turn ended, to the
/// agent that sent it: its parent, or a peer. Each request gets exactly
/// one report. Per asker, the last request's report holds the answer and
/// the earlier ones are notes that point to it. A subagent that ends its
/// turn while an agent it asked still owes it a report reports after
/// that report carried it on, unless that agent waits on it in turn.
fn report_on_turn_end(
    end: On<TurnEnded>,
    agents: Query<
        (
            &AgentId,
            &SpawnedBy,
            Option<&Requests>,
            Option<&PeerRequests>,
            Option<&Subtask>,
        ),
        Without<ActiveTurn>,
    >,
    debts: Debts,
    mut commands: Commands,
) {
    let agent = end.entity;
    // Only the agent whose turn ended, not the agents it propagates to.
    if agent != end.original_event_target() {
        return;
    }
    let Ok((id, parent, requests, peer_requests, subtask)) = agents.get(agent) else {
        return;
    };
    let askers = by_asker(parent.0, requests, peer_requests, &debts);
    if askers.is_empty() {
        return;
    }
    let edges = waits(&debts);
    let owed = edges
        .iter()
        .any(|&(waiting, owing)| waiting == agent && !reaches(&edges, owing, agent));
    if owed {
        return;
    }
    let title = subtask.map_or("the task", |subtask| subtask.title.as_str());
    let status = match &end.outcome {
        TurnOutcome::Answered(message) => match clipped_answer(message) {
            Some(text) => Status::Done(text),
            None => Status::Failed("The subagent ended without a final message.".to_owned()),
        },
        TurnOutcome::Failed(why) => Status::Failed(why.clone()),
        TurnOutcome::Stopped => Status::Interrupted,
    };
    let report = match status {
        Status::Done(text) => format!("Done: \"{title}\".\n{text}"),
        Status::Failed(why) => format!(
            "Failed: \"{title}\". {why} No answer will come for this request; /agents shows \
             the subagent's transcript."
        ),
        Status::Interrupted => format!(
            "Interrupted: \"{title}\". The subagent was stopped before it answered. No answer \
             will come for this request; send a `message` to carry it on."
        ),
    };
    commands.entity(agent).insert(Requests::default());
    if peer_requests.is_some() {
        commands.entity(agent).insert(PeerRequests::default());
    }
    for (asker, ids) in askers {
        let Some((last, earlier)) = ids.split_last() else {
            continue;
        };
        // The requests answered together with the last one: notes, read
        // with its report, which comes last and asks for the turn.
        for request in earlier {
            commands.trigger(Deliver {
                entity: asker,
                text: format!(
                    "Done: \"{title}\". This request was answered together with request {}; \
                     that report holds the answer.",
                    last.0
                ),
                origin: Origin::agent(id.clone(), Some(request.clone())),
                mode: DeliveryMode::Note,
                attachments: Vec::new(),
            });
        }
        commands.trigger(Deliver {
            entity: asker,
            text: report.clone(),
            origin: Origin::agent(id.clone(), Some(last.clone())),
            mode: DeliveryMode::Queue,
            attachments: Vec::new(),
        });
    }
}

/// After a restart, answers every request a restored subagent had not
/// answered with an interrupted report to the agent that sent it, and
/// keeps the subagent idle: nothing it was doing runs again by itself. A
/// `message` carries it on.
fn report_restored(
    mut restored: On<Restored>,
    agents: Query<(
        &AgentId,
        &SpawnedBy,
        Option<&Requests>,
        Option<&PeerRequests>,
        Option<&Subtask>,
    )>,
    debts: Debts,
    mut commands: Commands,
) {
    let agent = restored.entity;
    let Ok((id, parent, requests, peer_requests, subtask)) = agents.get(agent) else {
        return;
    };
    let askers = by_asker(parent.0, requests, peer_requests, &debts);
    let open = requests.is_some_and(|requests| !requests.0.is_empty())
        || peer_requests.is_some_and(|open| !open.0.is_empty());
    if !open {
        return;
    }
    restored.event_mut().resume = false;
    let title = subtask.map_or("the task", |subtask| subtask.title.as_str());
    for (asker, ids) in askers {
        for request in ids {
            commands.trigger(Deliver {
                entity: asker,
                text: format!(
                    "Interrupted: \"{title}\". The session restarted before the subagent \
                     answered, and it was not carried on. No answer will come for this \
                     request; send a `message` to carry it on."
                ),
                origin: Origin::agent(id.clone(), Some(request)),
                mode: DeliveryMode::Queue,
                attachments: Vec::new(),
            });
        }
    }
    commands.entity(agent).insert(Requests::default());
    if peer_requests.is_some() {
        commands.entity(agent).insert(PeerRequests::default());
    }
}

/// The text of the model's final message ([`answer_text`]), cut to
/// [`MAX_ANSWER_BYTES`].
fn clipped_answer(message: &Message) -> Option<String> {
    let text = answer_text(message)?;
    if text.len() <= MAX_ANSWER_BYTES {
        return Some(text);
    }
    let cut = text.floor_char_boundary(MAX_ANSWER_BYTES);
    Some(format!(
        "{}\n\n[The answer was cut at {MAX_ANSWER_BYTES} bytes.]",
        text.get(..cut).unwrap_or_default()
    ))
}

#[cfg(test)]
mod tests;
