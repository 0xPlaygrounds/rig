//! How a subagent is spawned: a [`SpawnSubagent`] request moves through the
//! stages like any effect, so plugins shape it in Prepare and refuse or
//! hold it in Gate, and this plugin spawns the agent when it starts.

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::world::DeferredWorld;
use rig_core::effect::{EffectId, EffectKind};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::tools::{MESSAGE, TASK, WAIT};
use super::{
    Helper, Lifecycle, OpenRequests, Peers, Request, SubagentSystems, SubagentsPlugin, Subtask,
    answered, refused,
};
use rig_harness::prelude::*;

/// The kind of a [`SpawnSubagent`] effect in the record.
pub const SUBAGENT: &str = "subagent";

/// A request for an agent spawned by the agent it is [`ForAgent`], its
/// parent. It is an open effect of the kind [`SUBAGENT`], recorded with the
/// request as it started and the new agent's id: Prepare systems edit it
/// (`Query<&mut SpawnSubagent, Added<Admitted>>`), Gate systems refuse it
/// (`Decision::decide(by, Outcome::refused(..))`) or hold it (`HoldOf`), and
/// when it starts,
/// [`SubagentsPlugin`](super::SubagentsPlugin) spawns the agent, answers it
/// with the agent's id and puts the agent on it as [`SpawnedAgent`].
/// A `task` call makes one, a child of the call (`ChildOf`), and answers
/// with how it went once it settled; a plugin spawns one itself for agents
/// of its own, such as the sides of a debate.
///
/// Each setting left out is the parent's: its model and, with it, its
/// connection and reasoning setting, its system prompt, and every tool it
/// may use but those that start subagents, when the new agent is as deep as
/// [`SubagentPolicy::max_depth`] allows.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[component(on_add = record_request)]
#[require(Name = Name::new(SUBAGENT))]
pub struct SpawnSubagent {
    /// A short title that names the agent, such as a task's.
    pub title: String,
    /// Its first message, sent from its parent once it is spawned; it
    /// stays idle without one. A task's brief.
    pub brief: String,
    /// What it is to its parent.
    pub role: Role,
    /// A catalog model as `vendor/model`.
    pub model: Option<String>,
    /// A reasoning setting, such as `low`.
    pub effort: Option<String>,
    /// The tools it may use, from its parent's.
    pub tools: Option<Vec<String>>,
    /// Its own system prompt.
    pub system_prompt: Option<String>,
    /// For a task: it may `message` and `wait` for its siblings started with
    /// `peers`, and they for it.
    pub peers: bool,
    /// For a helper: it does one job ([`once`](Self::once)).
    #[serde(default)]
    pub once: bool,
    /// What other plugins carry on the request, such as the arguments they
    /// added to `task`'s parameters (`ToolDef`), which a `task` call puts
    /// here by name; read it in Prepare or Review. Recorded with the rest.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

/// What a spawned agent is to its parent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// It works on a task for its parent and reports on it, as `task`
    /// starts one: it has a [`Subtask`] and a [`Lifecycle`], is told its
    /// role, and its parent can `message` it.
    #[default]
    Task,
    /// It works for the plugin that spawned it, which talks to it: it has a
    /// [`Helper`], reports to no one, and `message` cannot reach it.
    Helper,
}

impl SpawnSubagent {
    /// A subagent working on the task `title`, briefed with `brief`.
    pub fn task(title: impl Into<String>, brief: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            brief: brief.into(),
            role: Role::Task,
            model: None,
            effort: None,
            tools: None,
            system_prompt: None,
            peers: false,
            once: false,
            extra: Map::new(),
        }
    }

    /// A helper named `title`, idle until it is sent a message.
    pub fn helper(title: impl Into<String>) -> Self {
        Self {
            role: Role::Helper,
            ..Self::task(title, "")
        }
    }

    /// The same request, briefed with `brief`.
    #[must_use]
    pub fn brief(self, brief: impl Into<String>) -> Self {
        Self {
            brief: brief.into(),
            ..self
        }
    }

    /// The same request for a helper that does one job: it retires when
    /// its first turn ends, so it is despawned and a restart does not
    /// bring it back. Read its answer in an `On<Add<Settled>>` observer of
    /// that turn. A [`Role::Task`] subagent ignores it: it lives on for
    /// its parent's follow-ups.
    #[must_use]
    pub fn once(self) -> Self {
        Self { once: true, ..self }
    }

    /// The same request, with the system prompt `prompt`.
    #[must_use]
    pub fn system_prompt(self, prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            ..self
        }
    }
}

/// On a helper that does one job ([`SpawnSubagent::once`]).
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub(super) struct Once;

/// On an answered [`SpawnSubagent`]: the agent it spawned.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnedAgent(pub Entity);

/// How deep agents nest, as the `[subagents]` table of the settings says:
///
/// ```toml
/// [subagents]
/// max_depth = 2
/// ```
#[derive(Resource, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubagentPolicy {
    /// An agent nothing spawned is at depth 0. A request for an agent
    /// deeper than this is refused, and an agent at this depth is not given
    /// the tools that start subagents. 2 when not set.
    pub max_depth: usize,
}

impl Default for SubagentPolicy {
    fn default() -> Self {
        Self { max_depth: 2 }
    }
}

/// On a [`SpawnSubagent`] a `task` call made: the request it opens.
#[derive(Component)]
pub(super) struct TaskCall {
    /// The call's id, which the report names.
    pub(super) request: RequestId,
    /// The model call whose reply made the call, whose `task` calls report
    /// together.
    pub(super) batch: Option<EffectId>,
}

/// On a subagent just spawned: its first message, sent once the other
/// subagents spawned with it are there too, so that it names its peers.
#[derive(Component)]
pub(super) struct Brief(pub(super) Deliver);

pub(super) fn add(app: &mut App) {
    app.add_setting::<SubagentPolicy>("subagents")
        .add_observer(spawn_requested)
        .add_observer(follow_successor::<SpawnSubagent>)
        .add_observer(follow_successor::<TaskCall>)
        .add_observer(answer_task)
        .add_observer(end_with_call)
        .add_observer(retire_once)
        .add_systems(AgentPass, refuse_too_deep.in_set(SubagentSystems::Gate));
}

/// Makes the request an open effect the kernel records.
fn record_request(mut world: DeferredWorld, context: HookContext) {
    if let Some(request) = world.get::<SpawnSubagent>(context.entity) {
        let effect = open_effect(request);
        world
            .commands()
            .entity(context.entity)
            .insert_if_new(effect);
    }
}

/// The open effect of `request`.
fn open_effect(request: &SpawnSubagent) -> EffectRequest {
    let kind = EffectKind::Custom {
        kind: SUBAGENT.into(),
        payload: serde_json::to_value(request).unwrap_or_default(),
    };
    EffectRequest {
        kind,
        handler: None,
    }
}

/// Refuses a request for an agent deeper than [`SubagentPolicy`] allows.
fn refuse_too_deep(
    mut requests: Query<
        (&ForAgent, &mut Decision),
        (Undecided, PluginEffects, With<SpawnSubagent>, Without<Ask>),
    >,
    lineage: Query<&SpawnedBy>,
    policy: Res<SubagentPolicy>,
) {
    for (&ForAgent(parent), mut decision) in &mut requests {
        let depth = lineage.iter_ancestors::<SpawnedBy>(parent).count() + 1;
        if depth > policy.max_depth {
            let why = format!(
                "Agents nest at most {} deep, and this one would be at depth {depth}; do the work \
                 yourself",
                policy.max_depth
            );
            decision.decide(Decider::plugin("rig-subagents"), Outcome::refused(why));
        }
    }
}

/// The parent of a request, as its settings are inherited.
type Parent<'a> = (
    &'a AgentId,
    &'a ToolAccess,
    Option<&'a ModelChoice>,
    &'a Effort,
    &'a SystemPrompt,
    Option<&'a Connection>,
);

/// The new agent's model, reasoning setting and tools for `request`,
/// checked against its parent's model and reasoning setting, the new
/// agent's depth and the parent's tools by name.
fn settle(
    request: &SpawnSubagent,
    (parent, depth): ((Option<&ModelChoice>, Effort), usize),
    tools: &[&str],
    (models, policy): (&Models, &SubagentPolicy),
) -> Result<(ModelChoice, Effort, Vec<String>), String> {
    let briefless = request.role == Role::Task && request.brief.trim().is_empty();
    if request.title.trim().is_empty() || briefless {
        return Err("`description` and `prompt` must not be empty".to_owned());
    }
    let (model, effort) = ModelChoice::inherit(
        &models.0,
        parent,
        request.model.as_deref(),
        request.effort.as_deref(),
    )?;
    let model = model.ok_or("You have no model to give the subagent; name one in `model`")?;
    let may_delegate = depth < policy.max_depth;
    // A peer keeps `message` and `wait` for its siblings even where it
    // cannot delegate.
    let delegates =
        |name: &str| name == TASK || (!request.peers && (name == MESSAGE || name == WAIT));
    let mut tools: Vec<String> = match &request.tools {
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
                        "A subagent at depth {depth} cannot start subagents of its own; leave \
                         `{name}` out"
                    ));
                }
            }
            asked.clone()
        }
    };
    for peer_tool in [MESSAGE, WAIT].into_iter().filter(|_| request.peers) {
        if !tools.iter().any(|name| name == peer_tool) {
            tools.push(peer_tool.to_owned());
        }
    }
    Ok((model, effort, tools))
}

/// Spawns the agent of a request that started, and answers the request
/// with its id, or with why none was spawned. An agent on its parent's
/// model shares the parent's connection.
fn spawn_requested(
    started: On<Add<Started>>,
    mut requests: Query<(
        &SpawnSubagent,
        &ForAgent,
        &Effect,
        Option<&TaskCall>,
        &mut Decision,
    )>,
    parents: Query<Parent>,
    lineage: Query<&SpawnedBy>,
    tools: Query<&ToolDef>,
    (models, policy, wake): (Res<Models>, Res<SubagentPolicy>, Res<Wake>),
    me: Option<Res<PluginEntity<SubagentsPlugin>>>,
    mut commands: Commands,
) {
    let entity = started.entity;
    let Ok((request, &ForAgent(parent), effect, task, mut decision)) = requests.get_mut(entity)
    else {
        return;
    };
    commands.entity(entity).insert(open_effect(request));
    let Ok((caller, access, model, &effort, prompt, connection)) = parents.get(parent) else {
        decision.decide(
            Decider::plugin("rig-subagents"),
            Outcome::refused("its parent is not an agent"),
        );
        return;
    };
    let depth = lineage.iter_ancestors::<SpawnedBy>(parent).count() + 1;
    let names = tools.iter().map(|def| def.0.name.as_str());
    let mine: Vec<&str> = names.filter(|name| access.allows(name)).collect();
    let settings = ((model, effort), depth);
    let (choice, effort, allowed) = match settle(request, settings, &mine, (&models, &policy)) {
        Ok(settled) => settled,
        Err(why) => {
            decision.decide(Decider::plugin("rig-subagents"), Outcome::refused(why));
            return;
        }
    };
    let (title, id) = (request.title.trim().to_owned(), AgentId::default());
    let mut child = commands.spawn_empty();
    // Before the model choice, so connecting it keeps this connection.
    if let Some(connection) = connection.filter(|_| model == Some(&choice)) {
        child.insert(connection.clone());
    }
    let prompt = request
        .system_prompt
        .clone()
        .map_or(prompt.clone(), SystemPrompt);
    child.insert((
        Agent,
        id.clone(),
        SpawnedBy(parent),
        choice,
        effort,
        ToolAccess::Only(allowed),
        prompt,
        EffectParent(effect.id()),
    ));
    let asked = match task {
        Some(task) => task.request.clone(),
        None => RequestId(format!("{SUBAGENT}-{}", effect.id())),
    };
    let author = match request.role {
        Role::Task => Author::agent(caller.clone(), Some(asked.clone())),
        Role::Helper => Author::agent(caller.clone(), None),
    };
    let brief = request.brief.trim();
    if !brief.is_empty() {
        let brief = Deliver::new(child.id(), brief, DeliveryMode::Queue);
        child.insert(Brief(brief.with_author(author.titled(title.clone()))));
    }
    match request.role {
        Role::Task => {
            child.insert((
                Subtask,
                AgentTitle(title),
                OpenRequests(vec![Request {
                    id: asked,
                    asker: caller.clone(),
                    batch: task.and_then(|task| task.batch),
                }]),
                Lifecycle::Working,
            ));
            child.insert_if(Peers, || request.peers);
        }
        Role::Helper => {
            child.insert((Helper, AgentTitle(title)));
            child.insert_if(Once, || request.once);
        }
    }
    if let Some(me) = me {
        child.insert(ProvidedBy(me.entity()));
    }
    let child = child.id();
    wake.wake();
    commands.entity(entity).insert(SpawnedAgent(child));
    decision.decide(
        Decider::plugin("rig-subagents"),
        Outcome::custom(serde_json::json!({ "agent": id.0 })),
    );
}

/// Answers the `task` call whose request settled: with the subagent it
/// started, or with why none did.
fn answer_task(
    settled: On<Add<Settled>>,
    requests: Query<(&SpawnSubagent, &Outcome, &ChildOf, &TaskCall)>,
    agents: Query<&AgentId>,
    spawned: Query<&SpawnedAgent>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Ok((request, outcome, of, task)) = requests.get(settled.entity) else {
        return;
    };
    let agent = spawned.get(settled.entity).ok();
    let id = agent.and_then(|agent| agents.get(agent.0).ok());
    let output = match (&outcome.0, id) {
        (Ok(_), Some(id)) => {
            let together = match task.batch {
                Some(_) => {
                    " together with the reports of the other tasks of this reply, once all are \
                     done"
                }
                None => "",
            };
            answered(format!(
                "Started subagent `{}` on \"{}\". It works in the background; its report will \
                 arrive as a message{together}. Continue it with `message`; /agents shows it.",
                id.short(),
                request.title.trim(),
            ))
        }
        (Err(report), _) => refused(format!(
            "{}. No subagent was started; fix the call and send it again.",
            report.message.trim_end_matches('.')
        )),
        (Ok(_), None) => refused("No subagent was started.".to_owned()),
    };
    if let Ok(mut call) = calls.get_mut(of.parent()) {
        call.decide(Decider::plugin("rig-subagents"), output);
    }
}

/// Ends the request of a `task` call that settles first, as when an
/// interrupt ends the call while a gate holds the request: it is despawned,
/// which records it as cancelled and ends whatever held it.
fn end_with_call(
    settled: On<Add<Settled>>,
    children: Query<&Children>,
    open: Query<(), (With<TaskCall>, Without<Settled>)>,
    mut commands: Commands,
) {
    let requests = children.get(settled.entity).into_iter().flatten();
    for &request in requests.filter(|&&request| open.contains(request)) {
        commands.entity(request).despawn();
    }
}

/// Retires a helper that does one job when its turn ends.
fn retire_once(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    once: Query<(), With<Once>>,
    mut commands: Commands,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity)
        && once.contains(agent)
    {
        commands.trigger(Retire { entity: agent });
    }
}
