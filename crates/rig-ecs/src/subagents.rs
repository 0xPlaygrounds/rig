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
//! and starting no turn of its own. A report is headed by the child's short
//! id and task title ([`Origin::titled`]); the request it answers stays in
//! its [`Origin`].
//!
//! The `task` calls of one model reply form a batch unless a call asks to
//! report alone ([`Owed::batch`]): a child of a batch that is done holds its
//! reports ([`Owed::held`]) until every task of the batch has reported, and then
//! they all reach the caller in the same frame, so as one step. The caller
//! is not blocked meanwhile: the user can still talk to it and stop it.
//!
//! Subagents can also work together. A `task` with `peers` set gives the
//! child [`Peers`]: it may send a `message` to its siblings that have
//! [`Peers`] too, and its own report goes to the sibling that asked. Every
//! agent's open requests, whoever sent them, are its [`Owes`]. A `message`
//! to an agent it owes, its parent or a peer that asked, is the report on
//! that agent's oldest request it has read, and the turn's end reports only
//! on the requests still open. A new request to an agent that waits on the
//! sender's own report is refused, so no two agents wait on each other.
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
use bevy_ecs::system::SystemParam;
use bevy_reflect::prelude::*;
use rig_core::completion::Message;
use rig_core::effect::EffectId;
use rig_core::message::{ToolCall, ToolResult, ToolResultContent};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::agent::{
    ActiveTurn, Agent, AgentId, EffectParent, Effort, ModelChoice, Spawned, SpawnedBy,
    SystemPrompt, ToolAccess, TurnEnded, TurnOutcome, answer_text,
};
use crate::inbox::{Deliver, DeliveryMode, Inbox, Origin, RequestId};
use crate::journal::AppSaveExt;
use crate::models::{self, ModelConnector};
use crate::restore::Restored;
use crate::tools::{AppToolsExt, Footprint, ToolCalled, ToolDef, ToolOptions, ToolOutput, failed};

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
    and exactly one report for the request arrives later as a message headed with that \
    subagent's id and task title. Several `task` calls run side by side. By default the \
    reports of the `task` calls of one reply arrive together, in one message, once every one \
    of them is done; set `report` to \"alone\" for a task whose report should arrive as soon \
    as it is ready. To continue a subagent you already started, use `message` with its id, \
    not another `task`.";

const MESSAGE_DESCRIPTION: &str = "Send a follow-up to one of your own subagents, by the id \
    `task` returned. The subagent continues with its conversation, model and tools, and reads \
    `text` as a new request: at once when idle, after its current work when busy. The call \
    returns at once with the request's id; exactly one report for it arrives later as a \
    message headed with the subagent's id and task title. Only subagents you started \
    can be reached, and, when you were started with `peers`, your sibling subagents started \
    with `peers`; any other agent is refused, and the refusal lists the ones you can reach. \
    A message to an agent that asked you something, such as the agent that started you, asks \
    nothing: it answers that agent's oldest request you have read, and your final answer goes only to \
    the requests still open.";

const RULES: &[&str] = &[
    "Use `task` for work that would fill your context or can run on its own: a broad search \
     or investigation across many files, an independent change, or a second opinion from \
     another model. Not for one quick read or edit.",
    "Write a `task` prompt as a full brief: the goal, what is known, where to look, and what \
     to return. Do not give two subagents changes to the same files.",
    "`task` starts a new subagent; `message` continues one you started, by its id, with its \
     conversation kept. Each call is a request, answered by exactly one report that arrives \
     as a message; the reports of the tasks you start in one reply arrive together. \
     Meanwhile keep working on what does not need them, or end your turn; do not poll or \
     wait for them.",
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
    by your last message, like your task, or sooner by a `message` back to it: messaging an \
    agent that asked you answers its request, and your final answer goes to the requests \
    still open. A `message` to an id you cannot reach lists the ones you can. Ask only for \
    what you need.";

/// Adds the `task` and `message` tools and the reports that answer their
/// requests.
#[derive(Default)]
pub struct SubagentsPlugin;

impl Plugin for SubagentsPlugin {
    fn build(&self, app: &mut App) {
        app.add_open_tool(
            TASK,
            TASK_DESCRIPTION,
            ToolOptions {
                rules: RULES,
                footprint: Footprint::Independent,
            },
            on_task,
        )
        .add_open_tool(
            MESSAGE,
            MESSAGE_DESCRIPTION,
            ToolOptions {
                rules: &[],
                footprint: Footprint::Independent,
            },
            on_message,
        )
        .save_component::<Subtask>()
        .save_component::<Owes>()
        .save_component::<Peers>()
        .add_observer(name_subagent)
        .add_observer(report_on_turn_end)
        .add_observer(release_on_leave)
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

/// On an agent: the requests other agents sent it, oldest first, until
/// their reports are delivered. Each one gets exactly one report, to its
/// asker. A report to the parent that waits for its batch is
/// [`Owed::held`] here until every task of the batch has reported, then
/// delivered in the order the reports were made.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Owes(Vec<Owed>);

/// A request an agent owes a report on.
#[derive(Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Owed {
    /// The request.
    pub request: RequestId,
    /// The agent that sent it, which its report goes to.
    pub asker: AgentId,
    /// For a `task` started with the other `task` calls of one reply: the
    /// parent's model call of that reply. Not saved: a restart answers the
    /// open requests as interrupted and hands over what was held.
    #[serde(skip)]
    #[reflect(ignore)]
    pub batch: Option<EffectId>,
    /// Its report, made and waiting for the batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held: Option<HeldReport>,
}

impl Owes {
    /// The requests no report answered yet, oldest first.
    pub fn open(&self) -> impl Iterator<Item = &Owed> {
        self.0.iter().filter(|owed| owed.held.is_none())
    }

    /// Removes the open request `request`.
    pub fn close(&mut self, request: &RequestId) -> Option<Owed> {
        let open = |owed: &Owed| owed.held.is_none() && owed.request == *request;
        Some(self.0.remove(self.0.iter().position(open)?))
    }

    /// Removes every open request, grouped by asker in the order of each
    /// asker's first request, oldest first within each.
    pub fn drain_by_asker(&mut self) -> Vec<Vec<Owed>> {
        let mut open: Vec<Owed> = self.0.extract_if(.., |owed| owed.held.is_none()).collect();
        let mut askers = Vec::new();
        while let Some(asker) = open.first().map(|first| first.asker.clone()) {
            askers.push(open.extract_if(.., |owed| owed.asker == asker).collect());
        }
        askers
    }

    /// The batch its held reports wait for, if any holds one.
    fn held_batch(&self) -> Option<EffectId> {
        self.0
            .iter()
            .find_map(|owed| owed.held.as_ref().and(owed.batch))
    }

    /// Removes the held reports, in the order they were made.
    fn take_held(&mut self) -> Vec<HeldReport> {
        let held = self.0.extract_if(.., |owed| owed.held.is_some());
        held.filter_map(|owed| owed.held).collect()
    }
}

/// On a subagent: it may send a `message` to its siblings that have
/// [`Peers`] too, and they to it. A `task` with `peers` set inserts it;
/// any plugin may too.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(Component, Default, Clone, Debug)]
pub struct Peers;

/// A report held back for its batch.
#[derive(Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeldReport {
    /// The report's text.
    pub text: String,
    /// Where it comes from, with the request it answers.
    pub origin: Origin,
    /// [`DeliveryMode::Note`] when it needs no answer, else
    /// [`DeliveryMode::Queue`].
    pub mode: DeliveryMode,
}

impl HeldReport {
    /// The report, delivered to `asker`.
    fn deliver(self, asker: Entity) -> Deliver {
        Deliver {
            entity: asker,
            text: self.text,
            origin: self.origin,
            mode: self.mode,
            attachments: Vec::new(),
        }
    }
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
    /// When the report arrives. `together` (the default): with the reports of
    /// the other `task` calls of this reply, in one message once all of them
    /// are done. `alone`: as soon as this task is done.
    #[serde(default)]
    report: Report,
    /// When true, the subagent can send a `message` to the other subagents
    /// you start with `peers`, and they to it; each request is still
    /// answered by exactly one report, to the one that asked. Off when
    /// absent.
    #[serde(default)]
    peers: bool,
}

/// When a task's report reaches its parent: with its reply's other tasks,
/// once all are done, or alone, as soon as it is done.
#[derive(Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Report {
    #[default]
    Together,
    Alone,
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
}

/// A result for `call`: `data` for programs, then `text` for people.
fn answer(call: &ToolCall, data: serde_json::Value, text: String) -> ToolResult {
    call.result(vec![
        ToolResultContent::json(data),
        ToolResultContent::text(text),
    ])
}

/// Checks a `task` call's arguments against its parent. `tools` are the
/// parent's tools by name.
fn settle(
    args: &TaskArgs,
    parent: &Parent<'_>,
    tools: &[&str],
    connector: &ModelConnector,
) -> Result<Settled, String> {
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
                        parent.depth + 1
                    ));
                }
            }
            asked.clone()
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
    })
}

/// Starts the subagent of a `task` call and answers the call with its id,
/// or with why none started.
fn on_task(
    called: On<ToolCalled<TaskArgs>>,
    agents: Query<(&ToolAccess, Option<&ModelChoice>, &Effort, &SystemPrompt)>,
    lineage: Query<&SpawnedBy>,
    tools: Query<&ToolDef>,
    connector: Res<ModelConnector>,
    mut commands: Commands,
) {
    let (call, caller, run) = (called.call, called.agent, &called.run);
    let Ok((access, model, &effort, prompt)) = agents.get(caller) else {
        return;
    };
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
    let settled = match settle(&called.args, &parent, &mine, &connector) {
        Ok(settled) => settled,
        Err(why) => {
            let why = format!("{why}. No subagent was started; fix the call and send it again.");
            commands
                .entity(call)
                .insert_if_new(ToolOutput(failed(&run.call, why)));
            return;
        }
    };
    let child_id = AgentId::default();
    let request = RequestId(run.call.id.to_string());
    // A call a restart runs again has no reply to batch with.
    let batch = match (called.args.report, run.parent) {
        (Report::Together, Some(reply)) => Some(reply),
        _ => None,
    };
    let role = if called.args.peers {
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
        Owes(vec![Owed {
            request: request.clone(),
            asker: called.caller.clone(),
            batch,
            held: None,
        }]),
        settled.model,
        settled.effort,
        ToolAccess::Only(settled.tools),
        SystemPrompt(format!("{}{role}", prompt.0)),
        EffectParent(called.effect),
    ));
    if called.args.peers {
        child.insert(Peers);
    }
    let child = child.id();
    let together = if batch.is_some() {
        " together with the reports of the other tasks of this reply, once all \
         are done"
    } else {
        ""
    };
    commands.trigger(Deliver {
        entity: child,
        text: settled.instructions,
        origin: Origin::agent(called.caller.clone(), Some(request.clone()))
            .titled(settled.task.clone()),
        mode: DeliveryMode::Queue,
        attachments: Vec::new(),
    });
    let output = answer(
        &run.call,
        serde_json::json!({
            "agent": child_id.short(),
            "request": request.0,
            "status": "started",
        }),
        format!(
            "Started subagent `{}` on \"{}\". It works in the background; its \
             report will arrive as a message{together}. Continue it with \
             `message`; /agents shows it.",
            child_id.short(),
            settled.task,
        ),
    );
    commands.entity(call).insert_if_new(ToolOutput(output));
}

/// Every agent's [`Owes`], with what reporting on them needs.
#[derive(SystemParam)]
struct Ledgers<'w, 's> {
    agents: Query<'w, 's, (Entity, &'static AgentId, Option<&'static mut Owes>)>,
    families: Query<'w, 's, &'static Spawned>,
    commands: Commands<'w, 's>,
}

impl Ledgers<'_, '_> {
    /// The [`Owes`] of `agent`.
    fn owes(&mut self, agent: Entity) -> Option<Mut<'_, Owes>> {
        self.agents.get_mut(agent).ok().and_then(|(.., owes)| owes)
    }

    /// The agent with the id `id`.
    fn find(&self, id: &AgentId) -> Option<Entity> {
        let mut agents = self.agents.iter();
        agents
            .find(|(_, other, _)| *other == id)
            .map(|(entity, ..)| entity)
    }

    /// Who waits on whom, as (waiting, owing) pairs: an agent waits on
    /// each agent that owes it a report. Each asker is looked up once per
    /// agent that owes it, however many requests it sent.
    fn waits(&self) -> Vec<(Entity, Entity)> {
        let mut edges = Vec::new();
        for (owing, _, owes) in &self.agents {
            let mut askers: Vec<&AgentId> = Vec::new();
            for owed in owes.into_iter().flat_map(Owes::open) {
                if !askers.contains(&&owed.asker) {
                    askers.push(&owed.asker);
                }
            }
            let found = askers.into_iter().filter_map(|asker| self.find(asker));
            edges.extend(found.map(|waiting| (waiting, owing)));
        }
        edges
    }

    /// Reports `text` from `agent` on its requests `owed`, all from one
    /// asker, headed by `from`: the last request's report holds the text,
    /// and the earlier ones are notes that point to it. A report to the
    /// agent's `parent` that answers a task of a batch, or follows reports
    /// held already, is held in its [`Owes`] until the batch is done. An
    /// asker that is gone gets nothing.
    fn report(
        &mut self,
        (agent, parent): (Entity, Option<Entity>),
        from: &Origin,
        owed: Vec<Owed>,
        text: &str,
    ) {
        let last = owed
            .last()
            .and_then(|last| Some((self.find(&last.asker)?, last.request.0.clone())));
        let Some((asker, last)) = last else {
            return;
        };
        let count = owed.len();
        let batch = owed.iter().find_map(|owed| owed.batch);
        // The requests answered together with the last one: notes, read
        // with its report, which comes last and asks for the turn.
        let reports = owed.into_iter().enumerate().map(|(at, owed)| {
            let (text, mode) = if at + 1 < count {
                let text =
                    format!("Answered together with request {last}; that report holds the answer.");
                (text, DeliveryMode::Note)
            } else {
                (text.to_owned(), DeliveryMode::Queue)
            };
            let mut origin = from.clone();
            origin.request = Some(owed.request.clone());
            Owed {
                held: Some(HeldReport { text, origin, mode }),
                ..owed
            }
        });
        let owes = self.owes(agent).filter(|_| parent == Some(asker));
        match owes.and_then(|owes| Some((batch.or(owes.held_batch())?, owes))) {
            Some((batch, mut owes)) => {
                owes.0.extend(reports);
                self.release(asker, batch, None);
            }
            None => {
                for report in reports.filter_map(|owed| owed.held) {
                    self.commands.trigger(report.deliver(asker));
                }
            }
        }
    }

    /// Delivers the held reports of `batch` to `parent`, in the order the
    /// parent started the tasks, unless a subagent other than `gone` still
    /// owes the report on its task of the batch.
    fn release(&mut self, parent: Entity, batch: EffectId, gone: Option<Entity>) {
        let Ok(family) = self.families.get(parent) else {
            return;
        };
        let owing = family
            .iter()
            .filter(|&child| Some(child) != gone)
            .filter_map(|child| self.agents.get(child).ok()?.2)
            .any(|owes| owes.open().any(|owed| owed.batch == Some(batch)));
        if owing {
            return;
        }
        for child in family.iter() {
            if let Ok((_, _, Some(mut owes))) = self.agents.get_mut(child)
                && owes.held_batch() == Some(batch)
            {
                for report in owes.take_held() {
                    self.commands.trigger(report.deliver(parent));
                }
            }
        }
    }
}

/// Whether `to` can be reached from `from` along `edges`, in one step or
/// more.
fn reaches<T: Copy + PartialEq>(edges: &[(T, T)], from: T, to: T) -> bool {
    let (mut seen, mut stack) = (vec![from], vec![from]);
    while let Some(node) = stack.pop() {
        for &(_, end) in edges.iter().filter(|(start, _)| *start == node) {
            if end == to {
                return true;
            }
            if !seen.contains(&end) {
                seen.push(end);
                stack.push(end);
            }
        }
    }
    false
}

/// How a `message` call reaches an agent.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// One of the caller's own subagents, sent a new request.
    Child,
    /// A sibling, both having [`Peers`], sent a new request.
    Peer,
    /// An agent the caller owes a report: the text is the report on its
    /// oldest request the caller has read.
    Asker(RequestId),
}

/// An agent a `message` call can reach.
struct Reachable<'a> {
    entity: Entity,
    id: &'a AgentId,
    subtask: Option<&'a Subtask>,
    busy: bool,
    target: Target,
}

impl Reachable<'_> {
    /// How it is reached, its short id and its task title, such as
    /// peer `1a2b3c4d` ("Fix the parser").
    fn named(&self) -> String {
        let kind = match self.target {
            Target::Child => "subagent",
            Target::Peer => "peer",
            Target::Asker(_) => "the agent waiting for your report",
        };
        match self.subtask {
            Some(subtask) => format!("{kind} `{}` (\"{}\")", self.id.short(), subtask.title),
            None => format!("{kind} `{}`", self.id.short()),
        }
    }
}

/// Sends a `message` call's text to an agent the caller can reach, and
/// answers the call. To an agent the caller owes a report, the text is
/// that report, on the agent's oldest request it has read. To one of the caller's
/// own subagents, or to a sibling when both have [`Peers`], it is a new
/// request, refused when that agent waits on the caller's own report.
fn on_message(
    called: On<ToolCalled<MessageArgs>>,
    callers: Query<(Option<&Spawned>, Option<&SpawnedBy>, Has<Peers>, &Inbox)>,
    targets: Query<(&AgentId, Option<&Subtask>, Has<ActiveTurn>, Has<Peers>)>,
    mut ledgers: Ledgers,
) {
    let (call, caller, call_id) = (called.call, called.agent, &called.run.call);
    let (Ok((spawned, parent, is_peer, inbox)), Ok((id, subtask, ..))) =
        (callers.get(caller), targets.get(caller))
    else {
        return;
    };
    let args = &called.args;
    let wanted = args.agent.trim();
    let text = args.text.trim();
    // The agents the caller owes first: a message to one answers its
    // oldest request the caller has read, one no longer in its inbox.
    let unread = inbox.steering.iter().chain(&inbox.queued);
    let unread: Vec<_> = unread.flat_map(|sent| &sent.origin.request).collect();
    let read = |owed: &&Owed| !unread.contains(&&owed.request);
    let owes = ledgers.agents.get(caller).ok().and_then(|(.., owes)| owes);
    let requests = owes.into_iter().flat_map(Owes::open);
    let askers = requests.filter(read).filter_map(|owed| {
        let asker = ledgers.find(&owed.asker)?;
        Some((asker, Target::Asker(owed.request.clone())))
    });
    let children = spawned.into_iter().flat_map(|spawned| spawned.iter());
    let siblings = parent
        .filter(|_| is_peer)
        .and_then(|parent| ledgers.families.get(parent.0).ok())
        .into_iter()
        .flat_map(|siblings| siblings.iter())
        .filter(|&sibling| sibling != caller);
    let mut reach: Vec<Reachable<'_>> = Vec::new();
    for (entity, target) in askers
        .chain(children.map(|child| (child, Target::Child)))
        .chain(siblings.map(|sibling| (sibling, Target::Peer)))
    {
        if let Ok((id, subtask, busy, has_peers)) = targets.get(entity)
            && (target != Target::Peer || has_peers)
            && !reach.iter().any(|reachable| reachable.entity == entity)
        {
            reach.push(Reachable {
                entity,
                id,
                subtask,
                busy,
                target,
            });
        }
    }
    let target = reach
        .iter()
        .find(|target| target.id.0 == wanted || target.id.short() == wanted);
    let sent = match target {
        None => Err(format!(
            "`{wanted}` is not an agent you can reach, so nothing was sent. {}",
            listing(&reach)
        )),
        Some(_) if text.is_empty() => Err("`text` must not be empty. Nothing was sent.".to_owned()),
        Some(
            asker @ Reachable {
                target: Target::Asker(request),
                ..
            },
        ) => {
            let owed = ledgers
                .owes(caller)
                .and_then(|mut owes| owes.close(request));
            let from = titled(Origin::agent(id.clone(), None), subtask);
            let agents = (caller, parent.map(|parent| parent.0));
            ledgers.report(agents, &from, owed.into_iter().collect(), text);
            let said = format!(
                "Sent to {} as your report on its request. Your final answer goes only \
                 to the requests still open.",
                asker.named()
            );
            Ok((asker, request.clone(), "answered", said))
        }
        Some(target) if reaches(&ledgers.waits(), target.entity, caller) => Err(format!(
            "`{}` is waiting for your report, so it cannot be asked, and nothing was \
             sent. Put what you would ask or tell it in your answer.",
            target.id.short()
        )),
        Some(target) => {
            let request = RequestId(call_id.id.to_string());
            let owed = Owed {
                request: request.clone(),
                asker: id.clone(),
                batch: None,
                held: None,
            };
            let mut entity = ledgers.commands.entity(target.entity);
            let mut owes = entity.entry::<Owes>();
            owes.or_default().and_modify(|mut owes| owes.0.push(owed));
            if !target.busy {
                entity.insert(EffectParent(called.effect));
            }
            ledgers.commands.trigger(Deliver {
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
            let said = format!(
                "Sent to {}. {when} Its report will arrive as a message headed with its \
                 id.",
                target.named()
            );
            Ok((target, request, status, said))
        }
    };
    let output = match sent {
        Err(why) => failed(call_id, why),
        Ok((target, request, status, said)) => answer(
            call_id,
            serde_json::json!({
                "agent": target.id.short(),
                "request": request.0,
                "status": status,
            }),
            said,
        ),
    };
    ledgers
        .commands
        .entity(call)
        .insert_if_new(ToolOutput(output));
}

/// The agents a `message` call can reach, for a refusal.
fn listing(reach: &[Reachable<'_>]) -> String {
    if reach.is_empty() {
        return "You have no subagents; start one with `task`.".to_owned();
    }
    let named = reach.iter().map(|reachable| reachable.named());
    format!("You can reach: {}.", named.collect::<Vec<_>>().join(", "))
}

/// Names a subagent by its task's title, when it is spawned or restored.
fn name_subagent(inserted: On<Insert<Subtask>>, subtasks: Query<&Subtask>, mut commands: Commands) {
    if let Ok(subtask) = subtasks.get(inserted.entity) {
        commands
            .entity(inserted.entity)
            .insert(Name::new(subtask.title.clone()));
    }
}

/// Reports on every open request of a subagent whose turn ended, to the
/// agent that sent it ([`Ledgers::report`]): its answer, or why there is
/// none. A subagent that ends its turn while an agent it asked still owes
/// it a report reports after that report carried it on. No agent waits on
/// one that waits on it, so that report comes.
fn report_on_turn_end(
    end: On<TurnEnded>,
    agents: Query<(&SpawnedBy, Option<&Subtask>), Without<ActiveTurn>>,
    mut ledgers: Ledgers,
) {
    let agent = end.entity;
    // Only the agent whose turn ended, not the agents it propagates to.
    if agent != end.original_event_target() {
        return;
    }
    let Ok((parent, subtask)) = agents.get(agent) else {
        return;
    };
    let owed = ledgers.waits().iter().any(|&(waiting, _)| waiting == agent);
    let Ok((_, id, Some(mut owes))) = ledgers.agents.get_mut(agent) else {
        return;
    };
    if owed || owes.open().next().is_none() {
        return;
    }
    let from = titled(Origin::agent(id.clone(), None), subtask);
    let askers = owes.drain_by_asker();
    let text = match &end.outcome {
        TurnOutcome::Answered(message) => match clipped_answer(message) {
            Some(text) => text,
            None => "Failed: the subagent ended without a final message. No answer will come \
                     for this request; /agents shows the subagent's transcript."
                .to_owned(),
        },
        TurnOutcome::Failed(why) => format!(
            "Failed: {why} No answer will come for this request; /agents shows the subagent's \
             transcript."
        ),
        TurnOutcome::Stopped => "Interrupted: the subagent was stopped before it answered. No \
             answer will come for this request; send a `message` to carry it on."
            .to_owned(),
    };
    for owed in askers {
        ledgers.report((agent, Some(parent.0)), &from, owed, &text);
    }
}

/// `origin` titled with the task, when there is one.
fn titled(origin: Origin, subtask: Option<&Subtask>) -> Origin {
    match subtask {
        Some(subtask) => origin.titled(subtask.title.clone()),
        None => origin,
    }
}

/// Releases a batch's held reports when one of its subagents goes away
/// before it reported, so the batch does not wait for it forever.
fn release_on_leave(removed: On<Remove<Owes>>, parents: Query<&SpawnedBy>, mut ledgers: Ledgers) {
    let gone = removed.entity;
    let (Ok(parent), Ok((.., Some(owes)))) = (parents.get(gone), ledgers.agents.get(gone)) else {
        return;
    };
    // A subagent is in one batch at most: its task's.
    let batch = owes.open().find_map(|owed| owed.batch);
    if let Some(batch) = batch {
        ledgers.release(parent.0, batch, Some(gone));
    }
}

/// After a restart, answers every request a restored subagent had not
/// answered with an interrupted report to the agent that sent it, hands
/// over the reports it held for its batch, and keeps the subagent idle:
/// nothing it was doing runs again by itself. A `message` carries it on.
fn report_restored(
    mut restored: On<Restored>,
    agents: Query<(&SpawnedBy, Option<&Subtask>)>,
    mut ledgers: Ledgers,
) {
    let agent = restored.entity;
    let Ok((parent, subtask)) = agents.get(agent) else {
        return;
    };
    let Ok((_, id, Some(mut owes))) = ledgers.agents.get_mut(agent) else {
        return;
    };
    if owes.0.is_empty() {
        return;
    }
    let held = owes.take_held();
    let open: Vec<Owed> = owes.drain_by_asker().into_iter().flatten().collect();
    restored.event_mut().resume &= open.is_empty();
    let from = titled(Origin::agent(id.clone(), None), subtask);
    for report in held {
        ledgers.commands.trigger(report.deliver(parent.0));
    }
    let text = "Interrupted: the session restarted before the subagent answered, and it was \
                not carried on. No answer will come for this request; send a `message` to \
                carry it on.";
    for owed in open {
        ledgers.report((agent, Some(parent.0)), &from, vec![owed], text);
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
