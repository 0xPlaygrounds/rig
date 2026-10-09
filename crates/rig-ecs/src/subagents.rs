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
//! starts a turn of an idle caller, or is queued for a busy one.
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
    can be reached; any other agent is refused.";

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
    let delegates = |name: &str| name == TASK || name == MESSAGE;
    let tools = match args.tools {
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
                        SystemPrompt(format!("{}{SUBAGENT_ROLE}", prompt.0)),
                    ));
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

/// Sends a `message` call's text to one of the caller's own subagents as
/// a request, and answers the call with the request's id, or refuses any
/// other target.
fn on_message(
    called: On<ToolCalled>,
    calls: Query<(&ToolCallRun, Option<&OpenCall>)>,
    callers: Query<(&AgentId, Option<&Spawned>)>,
    children: Query<(
        &AgentId,
        Option<&Subtask>,
        Option<&Requests>,
        Has<ActiveTurn>,
    )>,
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
        (Ok((id, spawned)), Ok(args)) => {
            let wanted = args.agent.trim();
            let text = args.text.trim();
            let mine: Vec<(Entity, &AgentId, Option<&Subtask>, Option<&Requests>, bool)> = spawned
                .into_iter()
                .flat_map(|spawned| spawned.iter())
                .filter_map(|child| {
                    let (child_id, subtask, requests, busy) = children.get(child).ok()?;
                    Some((child, child_id, subtask, requests, busy))
                })
                .collect();
            let target = mine
                .iter()
                .find(|(_, child_id, ..)| child_id.0 == wanted || child_id.short() == wanted);
            match target {
                None => {
                    let listed: Vec<String> = mine
                        .iter()
                        .map(|(_, child_id, subtask, ..)| match subtask {
                            Some(subtask) => {
                                format!("`{}` (\"{}\")", child_id.short(), subtask.title)
                            }
                            None => format!("`{}`", child_id.short()),
                        })
                        .collect();
                    let yours = if listed.is_empty() {
                        "You have none; start one with `task`.".to_owned()
                    } else {
                        format!("Yours are: {}.", listed.join(", "))
                    };
                    failed(
                        call_id,
                        format!(
                            "`{wanted}` is not one of your subagents, so `message` cannot reach \
                             it, and nothing was sent. {yours}"
                        ),
                    )
                }
                Some(_) if text.is_empty() => failed(
                    call_id,
                    "`text` must not be empty. Nothing was sent.".to_owned(),
                ),
                Some(&(child, child_id, _, requests, busy)) => {
                    let request = RequestId(call_id.id.to_string());
                    let mut open_requests = requests
                        .map(|requests| requests.0.clone())
                        .unwrap_or_default();
                    open_requests.push(request.clone());
                    let mut entity = commands.entity(child);
                    entity.insert(Requests(open_requests));
                    if !busy && let Some(open) = open {
                        entity.insert(EffectParent(open.0.id()));
                    }
                    commands.trigger(Deliver {
                        entity: child,
                        text: text.to_owned(),
                        origin: Origin::agent(id.clone(), Some(request.clone())),
                        mode: DeliveryMode::Queue,
                        attachments: Vec::new(),
                    });
                    let (status, when) = if busy {
                        (
                            "queued",
                            "It is busy, so it reads this after its current work.",
                        )
                    } else {
                        ("started", "It works on it in the background.")
                    };
                    answer(
                        call_id,
                        serde_json::json!({
                            "agent": child_id.short(),
                            "request": request.0,
                            "status": status,
                        }),
                        format!(
                            "Sent to subagent `{}`. {when} Its report for request {} will \
                             arrive as a message.",
                            child_id.short(),
                            request.0
                        ),
                    )
                }
            }
        }
    };
    commands.entity(call).insert_if_new(ToolOutput(output));
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

/// Reports to its parent on every open request of a subagent whose turn
/// ended with none of its own subagents still at a request: one report per
/// request, so each gets exactly one. A subagent that ends its turn while
/// its own subagents work reports after their reports carried it on.
fn report_on_turn_end(
    end: On<TurnEnded>,
    agents: Query<
        (
            &AgentId,
            &Requests,
            &SpawnedBy,
            Option<&Spawned>,
            Option<&Subtask>,
        ),
        Without<ActiveTurn>,
    >,
    open: Query<&Requests>,
    mut commands: Commands,
) {
    let agent = end.entity;
    // Only the agent whose turn ended, not the agents it propagates to.
    if agent != end.original_event_target() {
        return;
    }
    let Ok((id, requests, parent, spawned, subtask)) = agents.get(agent) else {
        return;
    };
    let Some((last, earlier)) = requests.0.split_last() else {
        return;
    };
    let children_work = spawned.is_some_and(|spawned| {
        spawned
            .iter()
            .any(|child| open.get(child).is_ok_and(|requests| !requests.0.is_empty()))
    });
    if children_work {
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
    let mut reports: Vec<(RequestId, String)> = earlier
        .iter()
        .map(|request| {
            (
                request.clone(),
                format!(
                    "Done: \"{title}\". This request was answered together with request {}; \
                     that report holds the answer.",
                    last.0
                ),
            )
        })
        .collect();
    reports.push((
        last.clone(),
        match status {
            Status::Done(text) => format!("Done: \"{title}\".\n{text}"),
            Status::Failed(why) => format!(
                "Failed: \"{title}\". {why} No answer will come for this request; /agents \
                 shows the subagent's transcript."
            ),
            Status::Interrupted => format!(
                "Interrupted: \"{title}\". The subagent was stopped before it answered. No \
                 answer will come for this request; send a `message` to carry it on."
            ),
        },
    ));
    commands.entity(agent).insert(Requests::default());
    for (request, text) in reports {
        commands.trigger(Deliver {
            entity: parent.0,
            text,
            origin: Origin::agent(id.clone(), Some(request)),
            mode: DeliveryMode::Queue,
            attachments: Vec::new(),
        });
    }
}

/// After a restart, answers every request a restored subagent had not
/// answered with an interrupted report, and keeps the subagent idle:
/// nothing it was doing runs again by itself. A `message` carries it on.
fn report_restored(
    mut restored: On<Restored>,
    agents: Query<(&AgentId, &Requests, &SpawnedBy, Option<&Subtask>)>,
    mut commands: Commands,
) {
    let agent = restored.entity;
    let Ok((id, requests, parent, subtask)) = agents.get(agent) else {
        return;
    };
    if requests.0.is_empty() {
        return;
    }
    restored.event_mut().resume = false;
    let title = subtask.map_or("the task", |subtask| subtask.title.as_str());
    for request in &requests.0 {
        commands.trigger(Deliver {
            entity: parent.0,
            text: format!(
                "Interrupted: \"{title}\". The session restarted before the subagent \
                 answered, and it was not carried on. No answer will come for this request; \
                 send a `message` to carry it on."
            ),
            origin: Origin::agent(id.clone(), Some(request.clone())),
            mode: DeliveryMode::Queue,
            attachments: Vec::new(),
        });
    }
    commands.entity(agent).insert(Requests::default());
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
