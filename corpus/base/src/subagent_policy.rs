//! Plugin corpus: 39. Subagent policy (spec `39-subagent-policy.md`).
//!
//! One policy for every subagent, read from `RIG_HOME/subagents.toml` into
//! [`SubagentsConfig`]: how deep subagents nest, how many work at once (per
//! parent and in total), the model and tools a new one gets, and how many
//! model calls a turn of one makes and how much one spends before it is
//! stopped. It shapes `rig_basics::subagents`' [`SpawnSubagent`] requests in the
//! stages: Prepare gives them the defaults, Gate holds those without a free
//! slot and releases them as slots free, and a `task` call deeper than the
//! policy allows is refused for the model to read. A subagent over a cap
//! has its turn stopped with the reason, which `rig_basics::subagents` reports with
//! what it had said. `/agents` shows each subagent's depth and spend, and
//! the parent's status line how many of its tasks are queued.

use std::path::Path;

use bevy_log::warn;
use rig_basics::sessions::agents::{AgentColumn, ColumnOf};
use rig_basics::subagents::{
    Lifecycle, Role, SpawnSubagent, SubagentPolicy, SubagentSystems, SubagentsPlugin, TASK,
};
use rig_basics::telemetry::usage::{AgentUsage, UsagePlugin};
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::Deserialize;

/// `RIG_HOME/subagents.toml`. Every field may be left out.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SubagentsConfig {
    /// How deep subagents nest; an agent nothing spawned is at depth 0.
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
    pub max_turn_calls: Option<usize>,
    /// How much one subagent may spend, in USD.
    pub max_cost: Option<f64>,
}

impl Default for SubagentsConfig {
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

impl SubagentsConfig {
    /// The policy in `path`: the default when there is no such file, or
    /// when it cannot be read, which is logged.
    pub fn load(path: &Path) -> Self {
        let read = std::fs::read_to_string(path).map_err(|error| error.to_string());
        let parsed = match read {
            Err(_) if !path.exists() => return Self::default(),
            Ok(text) => toml::from_str(&text).map_err(|error| error.to_string()),
            Err(error) => Err(error),
        };
        parsed.unwrap_or_else(|why| {
            warn!("{} not used, the defaults apply: {why}", path.display());
            Self::default()
        })
    }
}

/// A hold of a `task`'s subagent until a slot frees; the number is its
/// place in line.
#[derive(Component)]
struct Queued(u64);

/// On an agent: how many of its `task`s wait for a slot.
#[derive(Component, Default, PartialEq)]
struct QueuedTasks(usize);

/// On a subagent: its `/agents` column.
#[derive(Component)]
struct PolicyColumn(Entity);

/// After the agent's subagents at work in the status line.
const QUEUED: StatusItem = StatusItem::new("subagent-policy", Side::Left, 65, 15);

/// The plugin's stage systems, to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubagentPolicySystems {
    /// In [`Stage::Prepare`]: gives `task` requests the default model and
    /// tools.
    Prepare,
    /// In [`Stage::Gate`], after [`SubagentSystems::Gate`]: refuses a
    /// `task` call too deep, holds requests without a free slot, and stops
    /// a subagent's turn over a cap.
    Gate,
}

/// Applies [`SubagentsConfig`] to `rig_basics::subagents`' requests, adding that
/// plugin and `rig_basics::telemetry`'s usage unless they are there.
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
        if !app.world().contains_resource::<SubagentsConfig>() {
            let path = RigHome::of(app.world()).root().join("subagents.toml");
            app.insert_resource(SubagentsConfig::load(&path));
        }
        app.register_required_components::<Agent, QueuedTasks>()
            .add_observer(add_column)
            .configure_sets(
                AgentPass,
                (
                    SubagentPolicySystems::Prepare.in_set(Stage::Prepare),
                    SubagentPolicySystems::Gate
                        .in_set(Stage::Gate)
                        .after(SubagentSystems::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    with_defaults.in_set(SubagentPolicySystems::Prepare),
                    (refuse_too_deep, queue, cap).in_set(SubagentPolicySystems::Gate),
                ),
            )
            .add_systems(Update, show_column)
            .add_systems(PostUpdate, show_queued.in_set(StatusSystems::Show));
    }

    /// Once `rig_basics::subagents` read its own settings: its depth is this policy's.
    fn finish(&self, app: &mut App) {
        if let Some(config) = app.world().get_resource::<SubagentsConfig>() {
            let max_depth = config.max_depth;
            app.insert_resource(SubagentPolicy { max_depth });
        }
    }
}

/// Gives each new subagent the default model and tools, unless its `task`
/// call named its own and the policy allows that. Default tools its parent
/// does not keep are left out.
fn with_defaults(
    mut requests: Query<(&mut SpawnSubagent, &ForAgent), (Added<Admitted>, PluginEffects)>,
    access: Query<&ToolAccess>,
    config: Res<SubagentsConfig>,
) {
    for (mut request, &ForAgent(parent)) in &mut requests {
        if request.role != Role::Task {
            continue;
        }
        if !config.allow_override {
            request.model = None;
            request.tools = None;
        }
        if request.model.is_none() {
            request.model.clone_from(&config.default_model);
        }
        if request.tools.is_none()
            && let Some(tools) = &config.default_tools
        {
            let keeps = |name: &&String| access.get(parent).is_ok_and(|a| a.keeps(name));
            request.tools = Some(tools.iter().filter(keeps).cloned().collect());
        }
    }
}

/// Refuses a `task` call whose subagent would be deeper than the policy
/// allows, for its model to read. An agent at the limit is not offered
/// `task`, but its model may call it anyway.
fn refuse_too_deep(
    mut calls: Query<(&ToolRequest, &ForAgent, &mut Decision), (ReadyToStart, ToolCalls)>,
    lineage: Query<&SpawnedBy>,
    config: Res<SubagentsConfig>,
) {
    for (request, &ForAgent(agent), mut decision) in &mut calls {
        let depth = lineage.iter_ancestors(agent).count() + 1;
        if request.call.function.name.as_str() == TASK && depth > config.max_depth {
            let why = format!(
                "subagent depth limit {} reached; do it yourself",
                config.max_depth
            );
            decision.decide(Decider::plugin("subagent-policy"), Outcome::refused(why));
        }
    }
}

/// Holds each `task`'s request while its parent, or the session, has no
/// slot free, and releases the held ones, first held first, as slots free.
/// A subagent holds a slot while it works or waits on its own subagents.
/// The session's limit leaves out the parent and the agents above it, which
/// wait on the request.
fn queue(
    new: Query<
        (Entity, &SpawnSubagent, &ForAgent, &Decision),
        (ReadyToStart, PluginEffects, Without<Ask>),
    >,
    held: Query<(Entity, &Queued, &HoldOf)>,
    parents: Query<&ForAgent, With<SpawnSubagent>>,
    (subagents, lineage): (Query<(Entity, &SpawnedBy, &Lifecycle)>, Query<&SpawnedBy>),
    mut counts: Query<(Entity, &mut QueuedTasks)>,
    (config, mut line): (Res<SubagentsConfig>, Local<u64>),
    mut commands: Commands,
) {
    let at_work = subagents
        .iter()
        .filter(|(.., step)| matches!(step, Lifecycle::Working | Lifecycle::WaitingOn(_)));
    let mut at_work: Vec<(Entity, Entity)> = at_work.map(|(agent, by, _)| (agent, by.0)).collect();
    let free = |at_work: &[(Entity, Entity)], parent: Entity| {
        let mine = at_work.iter().filter(|(_, by)| *by == parent).count();
        let above: Vec<Entity> = lineage.iter_ancestors(parent).collect();
        let others = at_work.iter().filter(|(agent, _)| *agent != parent);
        let total = others.filter(|(agent, _)| !above.contains(agent)).count();
        config.max_concurrent.is_none_or(|max| mine < max)
            && config.max_concurrent_total.is_none_or(|max| total < max)
    };
    let mut waiting: Vec<_> = held.iter().collect();
    waiting.sort_by_key(|(_, Queued(place), _)| *place);
    let mut still = Vec::new();
    for (hold, _, &HoldOf(request)) in waiting {
        let Ok(&ForAgent(parent)) = parents.get(request) else {
            continue;
        };
        if free(&at_work, parent) {
            at_work.push((Entity::PLACEHOLDER, parent));
            commands.entity(hold).despawn();
        } else {
            still.push(parent);
        }
    }
    for (request, spawn, &ForAgent(parent), decision) in &new {
        if spawn.role != Role::Task || decision.is_decided() {
            continue;
        }
        if free(&at_work, parent) {
            at_work.push((Entity::PLACEHOLDER, parent));
        } else {
            *line += 1;
            commands.spawn((Queued(*line), HoldOf(request)));
            still.push(parent);
        }
    }
    for (agent, mut count) in &mut counts {
        let queued = still.iter().filter(|&&parent| parent == agent).count();
        count.set_if_neq(QueuedTasks(queued));
    }
}

/// Stops a subagent's turn before a model call beyond `max_turn_calls`,
/// or once the subagent spent `max_cost`. `rig_basics::subagents` reports the
/// reason with what it had said.
fn cap(
    calls: Query<(&CallOf, &ForAgent), (ReadyToStart, ModelCalls, With<TurnCall>)>,
    turns: Query<&Calls, Without<TurnOutcome>>,
    model_calls: Query<(), With<TurnCall>>,
    subagents: Query<&AgentUsage, With<SpawnedBy>>,
    config: Res<SubagentsConfig>,
    mut commands: Commands,
) {
    for (&CallOf(turn), &ForAgent(agent)) in &calls {
        let (Ok(spent), Ok(of_turn)) = (subagents.get(agent), turns.get(turn)) else {
            continue;
        };
        let made = of_turn.iter().filter(|&c| model_calls.contains(c)).count();
        let why = match (config.max_turn_calls, config.max_cost) {
            (Some(max), _) if made > max => {
                format!("it reached the limit of {max} model calls per turn (max_turn_calls)")
            }
            (_, Some(max)) if spent.total.cost >= max => format!(
                "it spent ${:.4}, the limit is ${max} (max_cost)",
                spent.total.cost
            ),
            _ => continue,
        };
        commands.trigger(StopTurn::plugin_alone(agent, "subagent-policy", &why));
    }
}

/// Gives each subagent, spawned or restored, its `/agents` column.
fn add_column(added: On<Add<SpawnedBy>>, mut commands: Commands) {
    let agent = added.entity;
    let column = commands
        .spawn((AgentColumn::new("policy", ""), ColumnOf(agent)))
        .id();
    commands.entity(agent).insert(PolicyColumn(column));
}

/// Keeps each subagent's column current: its depth and spend.
fn show_column(
    agents: Query<
        (Entity, &PolicyColumn, &AgentUsage),
        Or<(Added<PolicyColumn>, Changed<AgentUsage>)>,
    >,
    lineage: Query<&SpawnedBy>,
    mut columns: Query<&mut AgentColumn>,
) {
    for (agent, &PolicyColumn(column), spent) in &agents {
        let depth = lineage.iter_ancestors(agent).count();
        let text = format!("depth {depth} · spent {}", spent.total.cost_or_tokens());
        if let Ok(mut column) = columns.get_mut(column) {
            column.text = text;
        }
    }
}

/// How many of an agent's `task`s are queued, in its status line.
fn show_queued(mut agents: Query<(&QueuedTasks, &mut StatusItems), Changed<QueuedTasks>>) {
    for (&QueuedTasks(queued), mut items) in &mut agents {
        let text = match queued {
            0 => String::new(),
            1 => "1 task queued".to_owned(),
            n => format!("{n} tasks queued"),
        };
        items.show(QUEUED.says(text, Tone::Yellow));
    }
}

#[cfg(test)]
mod tests;
