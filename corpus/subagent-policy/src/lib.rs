//! Plugin corpus: 39. Subagent policy (spec `39-subagent-policy.md`).
//!
//! One policy for every subagent, read from `RIG_HOME/subagents.toml` into
//! [`SubagentPolicy`]: how deep subagents nest, how many work at once (per
//! parent and in total), the model and tools a new one gets, and how many
//! model calls a turn of one and how much one may spend before it is
//! stopped.
//!
//! rig-subagents decides depth (a constant) and tools inside its private
//! `task` observer, so this plugin replaces that tool with its own `task`,
//! which spawns the subagent with rig-subagents' public components
//! ([`Subtask`], [`OpenRequests`], [`Lifecycle`]) so its lifecycle, reports,
//! `message` and `wait` keep working. A subagent with no free slot is
//! spawned [`Held`]: every message to it, its task first, waits until a
//! slot frees. `/agents` is replaced by one that also shows each agent's
//! depth, model, spend and whether it is queued.

use std::path::Path;

use bevy_ecs::system::SystemParam;
use rig_ecs::turn::ModelCall;
use rig_harness::harness_protocol::Home;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_subagents::{Awaits, Lifecycle, OpenRequests, Request, SubagentsPlugin, Subtask, TASK};
use rig_telemetry::{Spending, UsagePlugin};
use rig_tui::{Focus, PickItem, PickRequest};
use schemars::JsonSchema;
use serde::Deserialize;

/// The policy, `RIG_HOME/subagents.toml`. Every field may be left out.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SubagentPolicy {
    /// How deep subagents nest: an agent nothing spawned is at depth 0, and
    /// a subagent at this depth cannot start subagents.
    pub max_depth: usize,
    /// How many subagents of one parent work at once; more wait.
    pub max_concurrent: Option<usize>,
    /// How many subagents work at once in the whole session.
    pub max_concurrent_total: Option<usize>,
    /// The model of a new subagent, as vendor/model; the parent's if none.
    pub default_model: Option<String>,
    /// The tools a new subagent gets, those of them its parent has; all of
    /// the parent's if none.
    pub default_tools: Option<Vec<String>>,
    /// Whether a `task` call may name its subagent's `model` and `tools`.
    pub allow_override: bool,
    /// How many model calls one turn of a subagent may make.
    pub max_turn_calls: Option<u32>,
    /// How much one subagent may spend, in USD.
    pub max_cost: Option<f64>,
}

impl Default for SubagentPolicy {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_concurrent: None,
            max_concurrent_total: None,
            default_model: None,
            default_tools: None,
            allow_override: true,
            max_turn_calls: None,
            max_cost: None,
        }
    }
}

impl SubagentPolicy {
    /// The policy in `path`: the default when there is no such file, else
    /// the file's, or why it cannot be read.
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|why| format!("{}: {why}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(format!("{}: {error}", path.display())),
        }
    }
}

/// On a subagent waiting for a slot: what was sent to it meanwhile, its
/// task first, delivered once it may work. `order` is its place in line.
#[derive(Component, Debug)]
pub struct Held {
    order: u64,
    messages: Vec<Deliver>,
}

/// The model calls of a turn so far.
#[derive(Component, Default)]
struct ModelCalls(u32);

/// The next place in line for a [`Held`] subagent.
#[derive(Resource, Default)]
struct Line(u64);

/// Why the policy could not be read, shown once to the first agent.
#[derive(Resource)]
struct PolicyError(String);

/// Replaces rig-subagents' `task` and rig-basics' `/agents`, adding those
/// plugins and rig-telemetry's usage unless they are there.
#[derive(Default)]
pub struct SubagentPolicyPlugin;

impl Plugin for SubagentPolicyPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        if !app.is_plugin_added::<UsagePlugin>() {
            app.add_plugins(UsagePlugin);
        }
        if !app.world().contains_resource::<SubagentPolicy>() {
            match SubagentPolicy::load(&Home::from_env().root().join("subagents.toml")) {
                Ok(policy) => app.insert_resource(policy),
                Err(why) => app
                    .insert_resource(SubagentPolicy::default())
                    .insert_resource(PolicyError(why)),
            };
        }
        app.init_resource::<Line>()
            .add_message::<PickRequest>()
            .register_required_components::<TurnOf, ModelCalls>()
            .add_observer(hold)
            .add_observer(count_model_calls)
            .add_systems(
                Update,
                (
                    cap_turn_calls.before(PollCalls),
                    cap_spending,
                    report_policy_error,
                ),
            )
            .add_systems(PostUpdate, release.run_if(any_with_component::<Held>));
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<(Entity, &ToolDef)>();
        let task = tools
            .iter(world)
            .find(|(_, def)| def.0.name.as_str() == TASK);
        if let Some((task, _)) = task {
            world.entity_mut(task).insert(Disabled);
        }
        let mut commands = world.query_filtered::<(Entity, &Name), With<SlashCommand>>();
        let listed = commands
            .iter(world)
            .find(|(_, name)| name.as_str() == "/agents");
        if let Some((agents, _)) = listed {
            world.entity_mut(agents).insert(Disabled);
        }
        let options = ToolOptions {
            rules: RULES,
            footprint: Some(Footprint::Independent),
        };
        app.add_open_tool(TASK, TASK_DESCRIPTION, options, on_task)
            .add_command(
                "agents",
                "Pick an agent to show, with its depth, model, spend and state; /agents <id> shows it",
                agents,
            );
    }
}

const TASK_DESCRIPTION: &str = "Start a new subagent on a self-contained task: a new agent with \
    a conversation of its own, which sees only `prompt` and works with its tools until it can \
    answer. The call returns at once with its id, and says whether it started or is queued for \
    a free slot; exactly one report arrives later as a message headed with that id. Continue a \
    subagent with `message`, not another `task`.";

const RULES: &[&str] = &[
    "Use `task` for work that would fill your context or can run on its own; not for one quick \
     read or edit. Write its `prompt` as a full brief: the goal, what is known, where to look, \
     and what to return.",
    "Each `task` is answered by exactly one report that arrives as a message. Meanwhile keep \
     working on what does not need it, or end your turn.",
];

/// The arguments of a `task` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct TaskArgs {
    /// The task in full: the goal, what is known, where to look and what to
    /// return. The subagent sees nothing else.
    prompt: String,
    /// A short title for the task, 3 to 6 words, shown to the user. The
    /// prompt's first words when absent.
    description: Option<String>,
    /// A catalog model as vendor/model for the subagent, when the setup
    /// lets calls choose. The setup's default when absent.
    model: Option<String>,
    /// The subagent's reasoning setting, such as low or high.
    effort: Option<String>,
    /// The tools the subagent may use, from yours, when the setup lets
    /// calls choose. The setup's default when absent.
    tools: Option<Vec<String>>,
}

/// The subagent's model, reasoning setting and tools for `args`, from the
/// policy, its parent's model and reasoning setting and its parent's
/// `tools`; or why there are none.
fn settle(
    args: &TaskArgs,
    policy: &SubagentPolicy,
    parent: (Option<&ModelChoice>, Effort),
    tools: &[&str],
    models: &Models,
) -> Result<(ModelChoice, Effort, Vec<String>), String> {
    if args.prompt.trim().is_empty() {
        return Err("`prompt` must not be empty".to_owned());
    }
    if !policy.allow_override && (args.model.is_some() || args.tools.is_some()) {
        return Err(
            "this setup gives subagents their model and tools: leave out `model` and \
                    `tools`"
                .to_owned(),
        );
    }
    let model = args.model.as_deref().or(policy.default_model.as_deref());
    let (model, effort) = ModelChoice::inherit(&models.0, parent, model, args.effort.as_deref())?;
    let model = model.ok_or("You have no model to give the subagent; name one in `model`")?;
    let tools = match (&args.tools, &policy.default_tools) {
        (Some(asked), _) => {
            if let Some(name) = asked.iter().find(|name| !tools.contains(&name.as_str())) {
                return Err(format!(
                    "`{name}` is not one of your tools ({})",
                    tools.join(", ")
                ));
            }
            asked.clone()
        }
        (None, Some(default)) => default
            .iter()
            .filter(|name| tools.contains(&name.as_str()))
            .cloned()
            .collect(),
        (None, None) => tools.iter().map(|name| (*name).to_owned()).collect(),
    };
    Ok((model, effort, tools))
}

/// The subagents holding a slot or waiting for one: each one's parent, and
/// whether it waits.
#[derive(SystemParam)]
struct Slots<'w, 's> {
    subagents: Query<'w, 's, (Entity, &'static SpawnedBy, &'static Lifecycle, Has<Held>)>,
    waits: Query<'w, 's, &'static CallOf, With<Awaits>>,
    turns: Query<'w, 's, &'static TurnOf>,
    policy: Res<'w, SubagentPolicy>,
}

impl Slots<'_, '_> {
    /// The parent of each subagent at work, and of each held one when
    /// `held`. A subagent at work is one whose task runs, unless a `wait`
    /// call of it waits for another agent: it then holds no slot, so the
    /// agent it waits for can take one.
    fn taken(&self, held: bool) -> Vec<Entity> {
        let waiting: Vec<Entity> = self
            .waits
            .iter()
            .filter_map(|of| self.turns.get(of.0).ok())
            .map(|turn| turn.0)
            .collect();
        let at_work = |(agent, _, lifecycle, is_held): &(Entity, &SpawnedBy, &Lifecycle, bool)| {
            if *is_held {
                return held;
            }
            matches!(lifecycle, Lifecycle::Working) && !waiting.contains(agent)
        };
        self.subagents
            .iter()
            .filter(at_work)
            .map(|(_, parent, ..)| parent.0)
            .collect()
    }

    /// Why a new subagent of `parent` must wait while `taken` slots are
    /// held, or `None` when one is free.
    fn full(&self, taken: &[Entity], parent: Entity) -> Option<String> {
        let mine = taken.iter().filter(|&&of| of == parent).count();
        match (self.policy.max_concurrent, self.policy.max_concurrent_total) {
            (Some(max), _) if mine >= max => Some(format!(
                "{mine} of your subagents already work or wait (max_concurrent {max})"
            )),
            (_, Some(max)) if taken.len() >= max => Some(format!(
                "{} subagents already work or wait (max_concurrent_total {max})",
                taken.len()
            )),
            _ => None,
        }
    }
}

/// Starts the subagent of a `task` call, or holds it for a free slot, and
/// answers the call; refuses one beyond the policy's depth.
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
    slots: Slots,
    mut line: ResMut<Line>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (call, caller, args) = (called.call, called.agent, &called.args);
    let Ok((access, model, &effort, prompt, connection)) = agents.get(caller) else {
        return;
    };
    let policy = &*slots.policy;
    let depth = lineage.iter_ancestors::<SpawnedBy>(caller).count();
    if depth >= policy.max_depth {
        let why = format!(
            "subagent depth limit {} reached; do it yourself. No subagent was started.",
            policy.max_depth
        );
        commands.entity(call).insert_if_new(refused(why));
        return;
    }
    let names = tools.iter().map(|def| def.0.name.as_str());
    let mine: Vec<&str> = names.filter(|name| access.allows(name)).collect();
    let (choice, effort, allowed) = match settle(args, policy, (model, effort), &mine, &models) {
        Ok(settled) => settled,
        Err(why) => {
            let why = format!("{why}. No subagent was started; fix the call and send it again.");
            commands.entity(call).insert_if_new(refused(why));
            return;
        }
    };
    let task = match args.description.as_deref().map(str::trim) {
        Some(title) if !title.is_empty() => title.to_owned(),
        _ => args
            .prompt
            .split_whitespace()
            .take(6)
            .collect::<Vec<_>>()
            .join(" "),
    };
    let (id, request) = (
        AgentId::default(),
        RequestId(called.run.call.id.to_string()),
    );
    let queued = slots.full(&slots.taken(true), caller);
    let mut child = commands.spawn_empty();
    // Before the model choice, so connecting it keeps this connection.
    if let Some(connection) = connection.filter(|_| model == Some(&choice)) {
        child.insert(connection.clone());
    }
    let origin = Origin::agent(called.caller.clone(), Some(request.clone())).titled(task.clone());
    let brief = Deliver::new(child.id(), args.prompt.trim(), DeliveryMode::Queue);
    line.0 += 1;
    let said = format!(
        "subagent `{}` on \"{task}\" ({}, tools: {}).",
        id.short(),
        choice.0,
        if allowed.is_empty() {
            "none".to_owned()
        } else {
            allowed.join(", ")
        },
    );
    child.insert((
        Agent,
        id,
        SpawnedBy(caller),
        Subtask(task),
        OpenRequests(vec![Request {
            id: request,
            asker: called.caller.clone(),
            batch: called.run.parent,
        }]),
        Lifecycle::Working,
        choice,
        effort,
        ToolAccess::Only(allowed),
        prompt.clone(),
        EffectParent(called.effect),
        Held {
            order: line.0,
            messages: vec![brief.with_origin(origin)],
        },
    ));
    wake.wake();
    let said = match queued {
        Some(why) => format!(
            "Queued {said} It waits for a free slot: {why}. It starts once one finishes; its \
             report will arrive as a message, and /agents shows it as queued."
        ),
        None => format!(
            "Started {said} It works in the background; its report will arrive as a message. \
             Continue it with `message`; /agents shows it."
        ),
    };
    commands.entity(call).insert_if_new(answered(said));
}

/// Keeps what is sent to a [`Held`] subagent until it may work.
fn hold(mut sent: On<Deliver>, mut held: Query<&mut Held>) {
    if let Ok(mut held) = held.get_mut(sent.entity)
        && !sent.text.is_empty()
    {
        held.messages.push(sent.clone());
        sent.text.clear();
        sent.attachments.clear();
    }
}

/// Lets the [`Held`] subagents work, first held first, while slots are
/// free: what was sent to each is delivered now, its task first.
fn release(held: Query<(Entity, &Held, &SpawnedBy)>, slots: Slots, mut commands: Commands) {
    let mut taken = slots.taken(false);
    let mut line: Vec<_> = held.iter().collect();
    line.sort_by_key(|(_, held, _)| held.order);
    for (agent, held, parent) in line {
        if slots.full(&taken, parent.0).is_some() {
            continue;
        }
        taken.push(parent.0);
        commands.entity(agent).remove::<Held>();
        for message in &held.messages {
            commands.trigger(message.clone());
        }
    }
}

/// Counts a turn's model calls.
fn count_model_calls(
    started: On<Add<ModelCall>>,
    calls: Query<&CallOf>,
    mut turns: Query<&mut ModelCalls>,
) {
    if let Ok(&CallOf(turn)) = calls.get(started.entity)
        && let Ok(mut count) = turns.get_mut(turn)
    {
        count.0 += 1;
    }
}

/// Stops a subagent's turn once it started the model call beyond the
/// policy's `max_turn_calls`, before that call's reply is taken. Not from
/// [`count_model_calls`]: the kernel is still setting the call up then.
fn cap_turn_calls(
    turns: Query<(&TurnOf, &ModelCalls), Changed<ModelCalls>>,
    subagents: Query<&Conversation, With<SpawnedBy>>,
    policy: Res<SubagentPolicy>,
    mut commands: Commands,
) {
    let Some(max) = policy.max_turn_calls else {
        return;
    };
    for (&TurnOf(agent), count) in &turns {
        if count.0 > max
            && let Ok(conversation) = subagents.get(agent)
        {
            let why =
                format!("it reached the limit of {max} model calls per turn (max_turn_calls)");
            stop(&mut commands, agent, conversation, &why);
        }
    }
}

/// Stops a working subagent that spent the policy's `max_cost`.
fn cap_spending(
    spent: Query<
        (Entity, &Spending, &Conversation),
        (Changed<Spending>, With<SpawnedBy>, With<ActiveTurn>),
    >,
    policy: Res<SubagentPolicy>,
    mut commands: Commands,
) {
    let Some(max) = policy.max_cost else {
        return;
    };
    for (agent, Spending(totals), conversation) in &spent {
        if totals.cost >= max {
            let why = format!(
                "it spent ${:.4}, over the limit of ${max} (max_cost)",
                totals.cost
            );
            stop(&mut commands, agent, conversation, &why);
        }
    }
}

/// Reports a subagent as stopped for `why`, with its latest text as the
/// partial answer, then stops its turn.
fn stop(commands: &mut Commands, agent: Entity, conversation: &Conversation, why: &str) {
    let partial = conversation.messages().iter().rev().find_map(|message| {
        let Message::Assistant(reply) = message else {
            return None;
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
        (!text.trim().is_empty()).then_some(text)
    });
    let partial = match partial {
        Some(text) => format!("Its partial answer:\n\n{}", text.trim()),
        None => "It had written no answer yet.".to_owned(),
    };
    let report = format!(
        "Stopped: {why}. {partial}\n\nSend it a `message` to carry it on, or do the rest yourself."
    );
    // Reported first, so the stop's own report finds nothing open.
    commands.entity(agent).insert(Lifecycle::Finished(report));
    commands.trigger(Interrupt { entity: agent });
}

/// Tells the first agent once that the policy file could not be read.
fn report_policy_error(
    error: Option<Res<PolicyError>>,
    agents: Query<Entity, (With<Agent>, Without<SpawnedBy>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let (Some(error), Some(agent)) = (error, agents.iter().next()) else {
        return;
    };
    let why = format!("subagents.toml not used, the defaults apply: {}", error.0);
    notices.write(Notice::error(agent, why));
    commands.remove_resource::<PolicyError>();
}

/// `/agents`: picks an agent to show, each under the agent that spawned
/// it with its depth, model, spend and state, or shows the one whose id
/// starts with the argument.
fn agents(
    In(args): In<CommandArgs>,
    agents: Query<
        (
            Entity,
            &AgentId,
            &Name,
            Option<&ModelChoice>,
            Option<&Spending>,
            Has<ActiveTurn>,
            Has<Held>,
        ),
        With<Agent>,
    >,
    spawned: Query<&Spawned>,
    parents: Query<&SpawnedBy>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    if !args.args.is_empty() {
        match agents
            .iter()
            .find(|agent| agent.1.0.starts_with(&args.args))
        {
            Some((entity, ..)) => commands.trigger(Focus { entity }),
            None => {
                let why = format!("No agent's id starts with `{}`.", args.args);
                notices.write(Notice::error(args.agent, why));
            }
        }
        return;
    }
    let mut roots: Vec<_> = agents
        .iter()
        .filter(|agent| !parents.contains(agent.0))
        .collect();
    roots.sort_by(|a, b| a.1.0.cmp(&b.1.0));
    let listed = roots.iter().flat_map(|root| {
        std::iter::once(root.0).chain(spawned.iter_descendants_depth_first(root.0))
    });
    let (mut items, mut selected) = (Vec::new(), 0);
    for (entity, id, name, model, spending, busy, held) in
        listed.filter_map(|agent| agents.get(agent).ok())
    {
        if entity == args.agent {
            selected = items.len();
        }
        let depth = parents.iter_ancestors(entity).count();
        let model = model.map_or("no model", |model| model.0.as_str());
        let spend = spending.map_or("nothing spent".to_owned(), |spent| spent.0.cost_or_tokens());
        let state = match (held, busy) {
            (true, _) => "queued",
            (_, true) => "working",
            _ => "idle",
        };
        items.push(PickItem {
            label: format!(
                "{}{name} · depth {depth} · {model} · {spend} · {state}",
                "  ".repeat(depth)
            ),
            command: format!("agents {}", id.0),
        });
    }
    picks.write(PickRequest {
        agent: args.agent,
        title: "Show an agent".to_owned(),
        items,
        selected,
    });
}

/// A call's answer.
fn answered(text: String) -> ToolOutput {
    ToolOutput(ToolResult::success(text.into()))
}

/// A call's refusal.
fn refused(why: String) -> ToolOutput {
    ToolOutput(ToolResult::failed(ToolExecutionError::other(why)))
}

#[cfg(test)]
mod tests;
