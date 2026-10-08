//! Subagents. A `task` tool call spawns a child agent entity, in process,
//! with its own model, reasoning setting and tools; the child is
//! [`SpawnedBy`] the agent that called and works through turns like any
//! agent, in the background: the call answers at once that it started. The
//! call is dispatched and recorded like every tool call, and the child's
//! model calls are recorded with that call's effect as their parent, so the
//! effect log nests a subagent's work under the call that asked for it.
//!
//! The task goes to the child, and its answer back to the parent, with
//! [`Deliver`]: each carries an [`Origin`] naming the agent it came from
//! and the `task` call as its request. When the child's
//! [`TurnEnded`], with none of its own subagents still at work, its final
//! message, or why it has none, is queued for the parent: it starts a turn
//! of an idle parent, and waits for the end of a busy one's. So nothing
//! waits for a subagent, and the user can talk to any agent meanwhile.
//!
//! Stopping an agent's turn stops only that agent. Despawning an agent
//! despawns its subagents (`linked_spawn`). A finished subagent stays, so
//! a view can show its transcript and the user can talk to it. Each
//! subagent has its own log, whose header names the call that started it;
//! a restored session links them again by it.

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::completion::{AssistantContent, Message};
use rig_core::effect::{
    EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, Outcome, family, tool_key,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{CallId, ToolCall};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::tool::{ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult};
use serde::Deserialize;

use super::agent::{
    ActiveTurn, Agent, AgentId, Conversation, Effort, ModelChoice, Spawned, SpawnedBy,
    SystemPrompt, ToolAccess, TurnEnded,
};
use super::inbox::{Deliver, DeliveryMode, Origin, RequestId};
use super::journal::SessionLog;
use super::models;
use super::tools::{Footprint, ToolOptions, register_tool};
use super::usage::Spending;

/// The name of the tool that starts a subagent.
pub const TASK: &str = "task";

/// How deep subagents nest: an agent the user talks to is at depth 0, and
/// an agent at this depth cannot start subagents of its own.
pub const MAX_DEPTH: usize = 2;

/// The most bytes of a subagent's answer that go back to its parent.
const MAX_ANSWER_BYTES: usize = 50 * 1024;

const DESCRIPTION: &str = "Hand a self-contained task to a subagent: a new agent with a \
    conversation of its own, which works with its tools until it can answer. The call \
    returns at once and the subagent works in the background; its final message arrives \
    later as a message headed as that subagent's output. It sees nothing of this \
    conversation, only `prompt`. Several `task` calls run side by side.";

const RULES: &[&str] = &[
    "Use `task` for work that would fill your context or can run on its own: a broad search \
     or investigation across many files, an independent change, or a second opinion from \
     another model. Not for one quick read or edit.",
    "Write a `task` prompt as a full brief: the goal, what is known, where to look, and what \
     to return. Do not give two subagents changes to the same files.",
    "A subagent's answer arrives as its own message when it finishes. Meanwhile keep working \
     on what does not need it, or end your turn; do not poll or wait for it.",
];

/// What every subagent is told about its role, after its parent's own
/// system prompt.
const SUBAGENT_ROLE: &str = "\n\nYou are a subagent. Another agent gave you the task in the \
    first message and expects your answer; nobody answers questions while you work, so \
    decide for yourself and say what you assumed. Your last message is sent to that agent \
    as the task's result: make it complete on its own, with file paths, findings and \
    what you changed, and keep it short. If you start subagents of your own, end your \
    turn while they work: their answers come back to you, and you answer once they have.";

/// Marks a tool whose calls a subagent answers: starting one spawns a
/// child agent instead of running a handler.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Delegates;

/// On a subagent: the task's title, as its log's header holds it.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct Delegated {
    /// The task's short title.
    pub task: String,
}

/// On a subagent while it works on its task: the `task` call's effect,
/// which the subagent's model calls are recorded under, and the request
/// its answer names. The turn end that sends its answer to the parent
/// takes it. A restored subagent whose answer did not arrive gets one
/// again, with no effect.
#[derive(Component, Debug)]
pub struct Assignment {
    /// The `task` call's effect, unknown after a restart.
    pub effect: Option<EffectId>,
    /// The `task` call's id.
    pub request: RequestId,
}

/// Registers the `task` tool, which runs beside the reply's other calls
/// unless they touch anything.
pub fn add_task_tool(app: &mut App) {
    let options = ToolOptions {
        rules: RULES,
        footprint: Footprint::Independent,
    };
    let handler = ErasedHandler::new(TaskHandler::unbound());
    if let Some(tool) = register_tool(
        app.world_mut(),
        TASK,
        DESCRIPTION.to_owned(),
        parameters(),
        Some(handler),
        options,
    ) {
        app.world_mut().entity_mut(tool).insert(Delegates);
    }
}

fn parameters() -> serde_json::Value {
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

/// What a subagent starts from: the agent that calls `task`.
pub(crate) struct Parent<'a> {
    /// Its model, if it has one.
    pub(crate) model: Option<&'a ModelChoice>,
    /// Its reasoning setting.
    pub(crate) effort: Effort,
    /// Its system prompt, which the subagent's starts with.
    pub(crate) prompt: &'a SystemPrompt,
    /// How many agents above it started it.
    pub(crate) depth: usize,
}

/// A subagent ready to spawn.
pub(crate) struct Plan {
    id: AgentId,
    task: String,
    instructions: String,
    model: ModelChoice,
    effort: Effort,
    tools: Vec<String>,
    prompt: String,
}

/// Plans the subagent of a `task` call: the per-call handler that answers
/// at once that it started, and the plan to spawn it with once the call
/// has its effect id. A call that cannot start one gets a handler that
/// answers with why, and no plan. `tools` are the parent's tools: each
/// name, and whether it [`Delegates`].
pub(crate) fn plan(
    call: &ToolCall,
    parent: &Parent<'_>,
    tools: &[(&str, bool)],
) -> (ErasedHandler, Option<Plan>) {
    match settle(call, parent, tools) {
        Ok(settled) => {
            let id = AgentId::default();
            let started = started(&id, &settled.task);
            let plan = Plan {
                id,
                task: settled.task,
                instructions: settled.instructions,
                model: settled.model,
                effort: settled.effort,
                tools: settled.tools,
                prompt: format!("{}{SUBAGENT_ROLE}", parent.prompt.0),
            };
            (
                ErasedHandler::new(TaskHandler::new(Ok(started))),
                Some(plan),
            )
        }
        Err(why) => (
            ErasedHandler::new(TaskHandler::new(Err(format!(
                "{why}. No subagent was started; fix the call and send it again."
            )))),
            None,
        ),
    }
}

/// A `task` call's arguments checked against the parent.
struct Settled {
    task: String,
    instructions: String,
    model: ModelChoice,
    effort: Effort,
    tools: Vec<String>,
}

fn settle(call: &ToolCall, parent: &Parent<'_>, tools: &[(&str, bool)]) -> Result<Settled, String> {
    if call.function.invalid_arguments.is_some() {
        return Err("The arguments are not a JSON object".to_owned());
    }
    let args: TaskArgs =
        serde_json::from_value(serde_json::Value::Object(call.function.arguments.clone()))
            .map_err(|error| format!("The arguments do not fit: {error}"))?;
    let task = args.description.trim().to_owned();
    let instructions = args.prompt.trim().to_owned();
    if task.is_empty() || instructions.is_empty() {
        return Err("`description` and `prompt` must not be empty".to_owned());
    }
    let model = match args.model.as_deref().map(str::trim) {
        Some(reference) if !reference.is_empty() => {
            let spec = models::resolve(reference).ok_or_else(|| {
                format!("The catalog has no model `{reference}`; use vendor/model")
            })?;
            if !spec.tools {
                return Err(format!("{reference} cannot call tools"));
            }
            ModelChoice(models::reference(spec))
        }
        _ => parent
            .model
            .cloned()
            .ok_or("You have no model to give the subagent; name one in `model`")?,
    };
    let spec = models::resolve(&model.0)
        .ok_or_else(|| format!("The catalog has no model `{}`", model.0))?;
    let effort = match args.effort.as_deref().map(str::trim) {
        Some(name) if !name.is_empty() => {
            let options = models::effort_options(spec);
            let option = options
                .iter()
                .find(|option| option.0 == name)
                .ok_or_else(|| {
                    let names: Vec<&str> = options.iter().map(|option| option.0).collect();
                    format!(
                        "{} takes the reasoning settings {}, not `{name}`",
                        spec.display_name,
                        names.join(", ")
                    )
                })?;
            Effort(option.1)
        }
        _ if parent.model == Some(&model) => parent.effort,
        _ => Effort(None),
    };
    let may_delegate = parent.depth + 1 < MAX_DEPTH;
    let tools = match args.tools {
        None => tools
            .iter()
            .filter(|(_, delegates)| may_delegate || !delegates)
            .map(|(name, _)| (*name).to_owned())
            .collect(),
        Some(asked) => {
            for name in &asked {
                match tools.iter().find(|(tool, _)| tool == name) {
                    None => {
                        let mine: Vec<&str> = tools.iter().map(|(tool, _)| *tool).collect();
                        return Err(format!(
                            "`{name}` is not one of your tools ({})",
                            mine.join(", ")
                        ));
                    }
                    Some((_, true)) if !may_delegate => {
                        return Err(format!(
                            "A subagent at depth {} cannot start subagents of its own; leave \
                             `{name}` out",
                            parent.depth + 1
                        ));
                    }
                    Some(_) => {}
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

/// What a `task` call that started the subagent `id` on `task` answers.
pub(crate) fn started(id: &AgentId, task: &str) -> String {
    format!(
        "Started subagent `{}` on \"{task}\". It works in the background; its answer \
         will arrive as a message when it finishes. Use /agents to watch it.",
        id.short()
    )
}

/// Spawns the subagent of `plan` for the `task` call `call` of `parent`
/// whose effect is `effect`, starts its log and its turn.
pub(crate) fn spawn(
    commands: &mut Commands,
    log: &SessionLog,
    plan: Plan,
    parent: (Entity, &AgentId),
    call: &CallId,
    effect: EffectId,
) {
    let Plan {
        id,
        task,
        instructions,
        model,
        effort,
        tools,
        prompt,
    } = plan;
    log.open_subagent(&id, parent.1, call, &task);
    let request = RequestId(call.to_string());
    let child = commands
        .spawn((
            Name::new(format!("subagent: {task}")),
            Agent,
            id,
            SpawnedBy(parent.0),
            Delegated { task },
            Assignment {
                effect: Some(effect),
                request: request.clone(),
            },
            model,
            effort,
            ToolAccess::Only(tools),
            SystemPrompt(prompt),
        ))
        .id();
    commands.trigger(Deliver {
        entity: child,
        text: instructions,
        origin: Origin::agent(parent.1.clone(), Some(request)),
        mode: DeliveryMode::Queue,
    });
}

/// Sends a subagent's answer to the agent that started it once a turn of
/// the subagent ends with none of its own subagents still at their tasks:
/// its final message, or why it has none. A subagent that ends its turn
/// while its own subagents work answers after their answers carried it on.
/// Nothing is sent while the app exits: the restart carries the turn on.
pub(crate) fn answer_on_turn_end(
    end: On<TurnEnded>,
    agents: Query<
        (
            &AgentId,
            &Conversation,
            &Delegated,
            &SpawnedBy,
            &Assignment,
            Option<&Spawned>,
        ),
        Without<ActiveTurn>,
    >,
    assigned: Query<(), With<Assignment>>,
    log: Res<SessionLog>,
    mut commands: Commands,
) {
    let agent = end.entity;
    // Only the agent whose turn ended, not the agents it propagates to.
    if agent != end.original_event_target() {
        return;
    }
    let Ok((id, conversation, delegated, parent, assignment, children)) = agents.get(agent) else {
        return;
    };
    if log.is_exiting()
        || children.is_some_and(|children| children.iter().any(|child| assigned.contains(child)))
    {
        return;
    }
    commands.entity(agent).try_remove::<Assignment>();
    commands.trigger(Deliver {
        entity: parent.0,
        text: answer(&delegated.task, final_answer(conversation.messages())),
        origin: Origin::agent(id.clone(), Some(assignment.request.clone())),
        mode: DeliveryMode::Queue,
    });
}

/// The text a subagent sends the agent that started it on `task`: its
/// final message, or why it has none.
pub(crate) fn answer(task: &str, answer: Result<String, String>) -> String {
    match answer {
        Ok(answer) => format!("Task \"{task}\" finished:\n{answer}"),
        Err(why) => format!("Task \"{task}\" stopped without an answer:\n{why}"),
    }
}

/// The text of the subagent's last message when it is the model's, cut to
/// [`MAX_ANSWER_BYTES`]; otherwise why the subagent has no answer.
pub(crate) fn final_answer(conversation: &[Message]) -> Result<String, String> {
    let Some(Message::Assistant(reply)) = conversation.last() else {
        return Err(
            "The subagent stopped before it answered: it was interrupted, or its model \
             call failed. /agents shows its transcript."
                .to_owned(),
        );
    };
    let text: Vec<&str> = reply
        .content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let text = text.join("\n\n");
    let text = text.trim();
    if text.is_empty() {
        return Err("The subagent ended without a final message.".to_owned());
    }
    if text.len() <= MAX_ANSWER_BYTES {
        return Ok(text.to_owned());
    }
    let cut = text.floor_char_boundary(MAX_ANSWER_BYTES);
    Ok(format!(
        "{}\n\n[The answer was cut at {MAX_ANSWER_BYTES} bytes.]",
        text.get(..cut).unwrap_or_default()
    ))
}

/// The handler of `task` calls. The registered one only describes the
/// tool; each call gets one of its own that answers at once that its
/// subagent started, or why none did.
struct TaskHandler {
    answer: Option<Result<String, String>>,
}

impl TaskHandler {
    fn unbound() -> Self {
        Self { answer: None }
    }

    fn new(answer: Result<String, String>) -> Self {
        Self {
            answer: Some(answer),
        }
    }
}

impl Serve for TaskHandler {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: tool_key(TASK),
            family: FamilyDescriptor::Tool {
                name: TASK.to_owned(),
                description: DESCRIPTION.to_owned(),
                parameters: parameters(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        if !matches!(kind, EffectKind::ToolCall { .. }) {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                "the task tool answers tool calls only",
            )));
        }
        let failure = |kind, why: String| ToolResult::failed(ToolExecutionError::new(kind, why));
        let result = match &self.answer {
            None => failure(
                ToolErrorKind::Other,
                "`task` runs only inside an agent's turn".to_owned(),
            ),
            Some(Err(why)) => failure(ToolErrorKind::InvalidArgs, why.clone()),
            Some(Ok(started)) => ToolResult::success(ToolOutput::text(started.clone())),
        };
        Reply::Outcome(Ok(Outcome::ToolResult { result }))
    }
}

/// One agent in [`roster`]: how deep it is and a line describing it.
#[derive(Clone, Debug)]
pub struct RosterEntry {
    /// The agent.
    pub agent: Entity,
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    pub depth: usize,
    /// Its title, model, state and cost, indented by depth.
    pub label: String,
}

/// What [`roster`] reads of each agent.
pub type RosterQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AgentId,
        Option<&'static Delegated>,
        Option<&'static SpawnedBy>,
        Option<&'static Spawned>,
        Option<&'static ModelChoice>,
        Has<ActiveTurn>,
        &'static Spending,
    ),
    With<Agent>,
>;

/// Every agent as a tree: the agents the user started, by id, each
/// followed by its subagents in the order they were started.
pub fn roster(agents: &RosterQuery) -> Vec<RosterEntry> {
    let mut roots: Vec<(Entity, &AgentId)> = agents
        .iter()
        .filter(|(_, _, _, of, ..)| of.is_none_or(|of| !agents.contains(of.0)))
        .map(|(entity, id, ..)| (entity, id))
        .collect();
    roots.sort_by(|a, b| a.1.0.cmp(&b.1.0));
    let several = roots.len() > 1;
    let total = agents.iter().count();
    let mut stack: Vec<(Entity, usize)> = roots.iter().rev().map(|(root, _)| (*root, 0)).collect();
    let mut entries = Vec::new();
    while let Some((agent, depth)) = stack.pop() {
        // A relationship loop cannot happen, but a bound costs nothing.
        if entries.len() >= total {
            break;
        }
        let Ok((_, id, delegated, _, subagents, model, busy, spent)) = agents.get(agent) else {
            continue;
        };
        let title = match delegated {
            Some(delegated) => delegated.task.clone(),
            None if several => format!("agent {}", id.short()),
            None => "main agent".to_owned(),
        };
        let mut label = format!(
            "{}{title} · {} · {}",
            "  ".repeat(depth),
            model.map_or("no model", |model| model.0.as_str()),
            if busy { "working" } else { "idle" }
        );
        if let Some(cost) = spent.cost_label() {
            label.push_str(&format!(" · {cost}"));
        }
        entries.push(RosterEntry {
            agent,
            depth,
            label,
        });
        for child in subagents
            .into_iter()
            .flat_map(|subagents| subagents.iter().rev())
        {
            stack.push((child, depth + 1));
        }
    }
    entries
}
