//! Permission modes: how much each agent may do without asking. `ask`
//! confirms every call that may change something, `edits` lets file edits
//! run and asks for the rest, `auto` asks for nothing, and `read-only`
//! neither offers nor runs a tool that changes anything. `/mode <name>`
//! sets the agent's mode and `/mode` alone cycles it; the status line shows
//! it. A subagent starts in its parent's mode, or in the configured one
//! when that is stricter. A call that needs approval waits for a picker
//! (Allow, Always for this tool, Deny); "Always" is kept per agent. A run
//! nobody sits at denies such calls. Modes and "Always" answers are saved
//! with the session.
//!
//! Each tool that may change something is gated: in `finish`, its entity
//! is disabled and an open tool of the same name and definition takes its
//! place. A call of the gate is decided by the calling agent's mode; one
//! that may run starts the original tool as a call of its own, under the
//! gate's call, whose output becomes the gate call's.

use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_ecs::turn::ToolStarter;
use rig_tui::{PickItem, PickRequest, Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(test)]
mod tests;

/// After the model's reasoning setting; gone early when the line is short.
const MODE_ITEM: StatusItem = StatusItem::at(Side::Left, 45, 9);

/// How much an agent may do without asking. Saved with the session.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[reflect(Component, Saved, Default, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// Nothing that changes anything runs, or is offered.
    ReadOnly,
    /// Every call that may change something asks first.
    #[default]
    Ask,
    /// File edits run; every other call that may change something asks.
    Edits,
    /// Everything runs.
    Auto,
}

impl PermissionMode {
    /// Every mode, strictest first; `/mode` cycles through them.
    pub const ALL: [Self; 4] = [Self::ReadOnly, Self::Ask, Self::Edits, Self::Auto];

    /// Its name, as `/mode` takes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Ask => "ask",
            Self::Edits => "edits",
            Self::Auto => "auto",
        }
    }

    /// The mode named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.name() == name)
    }

    /// The stricter of the two.
    pub fn stricter(self, other: Self) -> Self {
        let rank = |mode: Self| Self::ALL.iter().position(|m| *m == mode);
        if rank(self) <= rank(other) {
            self
        } else {
            other
        }
    }

    /// What a call of a tool of `class` does in this mode.
    fn decide(self, class: ToolClass) -> Verdict {
        match (self, class) {
            (_, ToolClass::Read) | (Self::Auto, _) | (Self::Edits, ToolClass::Edit) => Verdict::Run,
            (Self::ReadOnly, _) => Verdict::Refuse,
            (Self::Ask | Self::Edits, _) => Verdict::Ask,
        }
    }
}

/// What a tool's calls may do, from its footprint or the configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolClass {
    /// Changes nothing.
    Read,
    /// Edits files.
    Edit,
    /// Runs commands.
    Exec,
    /// Reaches the network.
    Network,
    /// Anything else that may change something.
    Other,
}

/// What happens to a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Run,
    Ask,
    Refuse,
}

/// The configuration, `RIG_HOME/permissions.json` when it exists:
/// `{"mode": "ask", "subagents": "edits", "tools": {"shell": "exec"}}`.
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PermissionConfig {
    /// The mode of an agent nobody set one for.
    pub mode: PermissionMode,
    /// A mode subagents start in when it is stricter than their parent's.
    pub subagents: Option<PermissionMode>,
    /// The class of a tool by name, over what its footprint says.
    pub tools: HashMap<String, ToolClass>,
}

impl Default for PermissionConfig {
    fn default() -> Self {
        let tools = [
            ("shell", ToolClass::Exec),
            ("edit", ToolClass::Edit),
            ("write", ToolClass::Edit),
            ("fetch", ToolClass::Network),
        ];
        Self {
            mode: PermissionMode::Ask,
            subagents: None,
            tools: tools
                .into_iter()
                .map(|(name, class)| (name.to_owned(), class))
                .collect(),
        }
    }
}

impl PermissionConfig {
    /// The class of the tool `name` whose calls have `footprint`.
    fn class(&self, name: &str, footprint: Footprint) -> ToolClass {
        self.tools.get(name).copied().unwrap_or(match footprint {
            Footprint::Exclusive => ToolClass::Other,
            Footprint::ReadOnly | Footprint::Independent => ToolClass::Read,
        })
    }
}

/// The tools the agent may call without asking, answered "Always". Saved
/// with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone)]
pub struct AlwaysAllowed(pub Vec<String>);

/// On a gate: the tool entity it stands for, now disabled, and its class.
#[derive(Component, Clone, Copy, Debug)]
pub struct Gate {
    /// The original tool.
    pub original: Entity,
    /// What its calls may do.
    pub class: ToolClass,
}

/// On a gate call waiting for the user's answer.
#[derive(Component, Clone)]
pub struct Awaiting {
    /// What `/permit` names it by.
    pub id: u64,
    /// The calling agent.
    pub agent: Entity,
    /// The gate.
    pub gate: Entity,
    /// The tool's name.
    pub tool: String,
    /// The call, and the effect it is recorded under.
    run: (ToolCallRun, EffectId),
}

/// On the original tool's call, which a gate call started: the gate call.
#[derive(Component, Clone, Copy, Debug)]
struct Inner(Entity);

/// The ids of approvals, and the one the picker shows.
#[derive(Resource, Default)]
struct Approvals {
    last: u64,
    shown: Option<u64>,
}

/// Permission modes for every agent.
#[derive(Default)]
pub struct PermissionModesPlugin;

impl Plugin for PermissionModesPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<PermissionConfig>() {
            let path = rig_harness::harness_protocol::Home::from_env()
                .root()
                .join("permissions.json");
            let config: Option<PermissionConfig> = rig_tools::fs::read_json(&path);
            app.insert_resource(config.unwrap_or_default());
        }
        app.init_resource::<Approvals>()
            .add_message::<PickRequest>()
            .register_type::<PermissionMode>()
            .register_type::<AlwaysAllowed>()
            .add_command(
                "mode",
                "Set how much the agent may do without asking: read-only, ask, edits or auto",
                set_mode,
            )
            .add_command(
                "permit",
                "Answer a call waiting for approval: <id> allow|always|deny",
                permit,
            )
            .add_observer(inherit)
            .add_observer(offer)
            .add_observer(inner_done)
            .add_observer(forget)
            .add_systems(PostUpdate, show_mode.in_set(StatusSystems));
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let Some(config) = world.get_resource::<PermissionConfig>().cloned() else {
            return;
        };
        let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
        let gated: Vec<_> = tools
            .iter(world)
            .map(|(entity, def, rules, &footprint)| {
                let class = config.class(def.0.name.as_str(), footprint);
                (entity, def.clone(), rules.clone(), footprint, class)
            })
            .filter(|(.., class)| *class != ToolClass::Read)
            .collect();
        for (original, def, rules, footprint, class) in gated {
            app.world_mut().entity_mut(original).insert(Disabled);
            let name = def.0.name.as_str().to_owned();
            let options = ToolOptions {
                rules: &[],
                footprint,
            };
            app.add_open_tool::<Value, _>(&name, &def.0.description, options, gate);
            let world = app.world_mut();
            let mut defs = world.query::<(Entity, &ToolDef)>();
            let gate = defs
                .iter(world)
                .find(|(_, other)| other.0.name.as_str() == name)
                .map(|(gate, _)| gate);
            if let Some(gate) = gate {
                world
                    .entity_mut(gate)
                    .insert((def, rules, Gate { original, class }));
            }
        }
    }
}

/// The mode `agent` is in.
fn mode_of(mode: Option<&PermissionMode>, config: &PermissionConfig) -> PermissionMode {
    mode.copied().unwrap_or(config.mode)
}

/// A subagent starts in its parent's mode, or the configured one when that
/// is stricter. A restored subagent already has its saved mode.
fn inherit(
    spawned: On<Add<SpawnedBy>>,
    agents: Query<(&SpawnedBy, Option<&PermissionMode>)>,
    modes: Query<Option<&PermissionMode>>,
    config: Res<PermissionConfig>,
    mut commands: Commands,
) {
    let Ok((SpawnedBy(parent), None)) = agents.get(spawned.entity) else {
        return;
    };
    let parent = mode_of(modes.get(*parent).ok().flatten(), &config);
    let mode = config
        .subagents
        .map_or(parent, |configured| configured.stricter(parent));
    commands.entity(spawned.entity).insert(mode);
}

/// An agent in `read-only` is offered no tool that changes anything.
fn offer(
    mut prepare: On<PrepareRequest>,
    agents: Query<Option<&PermissionMode>>,
    gates: Query<&ToolDef, With<Gate>>,
    config: Res<PermissionConfig>,
) {
    let mode = mode_of(agents.get(prepare.agent).ok().flatten(), &config);
    if mode != PermissionMode::ReadOnly {
        return;
    }
    let gated: Vec<String> = gates
        .iter()
        .map(|def| def.0.name.as_str().to_owned())
        .collect();
    prepare
        .tools
        .retain(|tool| !gated.iter().any(|name| *name == tool.name.as_str()));
}

/// Decides a gated call by its agent's mode: runs it, refuses it, or asks.
fn gate(
    gated: On<ToolCalled<Value>>,
    gates: Query<&Gate>,
    agents: Query<(Option<&PermissionMode>, Option<&AlwaysAllowed>)>,
    config: Res<PermissionConfig>,
    invoked: Option<Res<Invoked>>,
    mut approvals: ResMut<Approvals>,
    mut commands: Commands,
) {
    let Ok(&Gate { class, .. }) = gates.get(gated.entity) else {
        return;
    };
    let Ok((mode, always)) = agents.get(gated.agent) else {
        return;
    };
    let mode = mode_of(mode, &config);
    let tool = gated.run.call.function.name.as_str().to_owned();
    let verdict = match mode.decide(class) {
        Verdict::Ask if always.is_some_and(|always| always.0.contains(&tool)) => Verdict::Run,
        verdict => verdict,
    };
    let refuse = |commands: &mut Commands, why: String| {
        let output = ToolOutput(ToolResult::failed(ToolExecutionError::other(why)));
        commands.entity(gated.call).insert_if_new(output);
    };
    match verdict {
        Verdict::Run => {
            let (gate, call, agent) = (gated.entity, gated.call, gated.agent);
            let run = (gated.run.clone(), gated.effect);
            commands.queue(move |world: &mut World| run_original(world, gate, call, agent, run));
        }
        Verdict::Refuse => refuse(
            &mut commands,
            format!(
                "Refused: this agent is in read-only mode, so `{tool}` may not run. Nothing \
                 was changed; use the tools that only read."
            ),
        ),
        Verdict::Ask if !invoked.is_some_and(|invoked| invoked.interactive()) => refuse(
            &mut commands,
            format!(
                "Denied: `{tool}` needs the user's approval in {} mode, and nobody can approve \
                 it in a non-interactive run. Nothing was changed.",
                mode.name()
            ),
        ),
        Verdict::Ask => {
            approvals.last += 1;
            let awaiting = Awaiting {
                id: approvals.last,
                agent: gated.agent,
                gate: gated.entity,
                tool,
                run: (gated.run.clone(), gated.effect),
            };
            commands.entity(gated.call).insert(awaiting);
            commands.queue(ask_next);
        }
    }
}

/// Starts the original tool of `gate` for the gate's `call` of `agent`, as
/// a call of its own under it: the original is enabled and the gate
/// disabled for the start only, so the tool starter finds the original by
/// the call's name.
fn run_original(
    world: &mut World,
    gate: Entity,
    call: Entity,
    agent: Entity,
    run: (ToolCallRun, EffectId),
) {
    let Some(&Gate { original, .. }) = world.get::<Gate>(gate) else {
        return;
    };
    if world.get_entity(call).is_err() {
        return;
    }
    let swap = |world: &mut World, on: Entity, off: Entity| {
        if let Ok(mut on) = world.get_entity_mut(on) {
            on.remove::<Disabled>();
        }
        if let Ok(mut off) = world.get_entity_mut(off) {
            off.insert(Disabled);
        }
    };
    swap(world, original, gate);
    if let Err(error) = world.run_system_cached_with(start_inner, (call, agent, run)) {
        let why = format!("could not start the tool: {error}");
        let output = ToolOutput(ToolResult::failed(ToolExecutionError::other(why)));
        world.entity_mut(call).insert(output);
    }
    world.flush();
    swap(world, gate, original);
}

/// Spawns and starts the original tool's call under the gate's `call`.
fn start_inner(
    In((call, agent, (run, effect))): In<(Entity, Entity, (ToolCallRun, EffectId))>,
    starter: ToolStarter,
    mut commands: Commands,
) {
    let inner = starter.run(run.call, Some(effect));
    let entity = commands
        .spawn((
            Name::new("gated tool call"),
            Inner(call),
            ChildOf(call),
            inner.clone(),
        ))
        .id();
    starter.start(&mut commands, entity, agent, &inner);
}

/// The original tool's output is the gate call's.
fn inner_done(
    done: On<Add<ToolOutput>>,
    inner: Query<(&Inner, &ToolOutput)>,
    mut commands: Commands,
) {
    if let Ok((&Inner(call), output)) = inner.get(done.entity) {
        commands.entity(call).try_insert_if_new(output.clone());
        commands.entity(done.entity).try_despawn();
    }
}

/// A waiting call that went away, answered or interrupted, frees the
/// picker for the next.
fn forget(
    gone: On<Remove<Awaiting>>,
    awaiting: Query<&Awaiting>,
    mut approvals: ResMut<Approvals>,
    mut commands: Commands,
) {
    if let Ok(awaiting) = awaiting.get(gone.entity)
        && approvals.shown == Some(awaiting.id)
    {
        approvals.shown = None;
        commands.queue(ask_next);
    }
}

/// Opens the picker for the oldest waiting call unless one is shown.
fn ask_next(world: &mut World) {
    if world.resource::<Approvals>().shown.is_some() {
        return;
    }
    let mut waiting = world.query::<&Awaiting>();
    let Some(oldest) = waiting
        .iter(world)
        .min_by_key(|awaiting| awaiting.id)
        .cloned()
    else {
        return;
    };
    let who = world.get::<AgentId>(oldest.agent).map(AgentId::short);
    let who = match (who, world.get::<SpawnedBy>(oldest.agent)) {
        (Some(id), Some(_)) => format!("Subagent {id}"),
        _ => "The agent".to_owned(),
    };
    let args = oldest.run.0.call.function.raw_arguments();
    let args: String = args.chars().take(120).collect();
    let title = format!("{who} wants to run `{}` {args}", oldest.tool);
    let choice = |label: String, answer: &str| PickItem {
        label,
        command: format!("permit {} {answer}", oldest.id),
    };
    let items = vec![
        choice("Allow".to_owned(), "allow"),
        choice(
            format!("Always allow `{}` for this agent", oldest.tool),
            "always",
        ),
        choice("Deny".to_owned(), "deny"),
    ];
    world.resource_mut::<Approvals>().shown = Some(oldest.id);
    // The picker runs the answer for the asking agent, so answering a
    // subagent's question does not show it.
    world.write_message(PickRequest {
        agent: oldest.agent,
        title,
        items,
        selected: 0,
    });
}

/// `/mode [name]`: sets the agent's mode, or cycles it to the next.
fn set_mode(
    In(args): In<CommandArgs>,
    agents: Query<Option<&PermissionMode>>,
    config: Res<PermissionConfig>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let current = mode_of(agents.get(args.agent).ok().flatten(), &config);
    let mode = match args.args.as_str() {
        "" => {
            let at = PermissionMode::ALL.iter().position(|mode| *mode == current);
            let next = at.map_or(0, |at| (at + 1) % PermissionMode::ALL.len());
            PermissionMode::ALL.get(next).copied().unwrap_or_default()
        }
        name => {
            let Some(mode) = PermissionMode::parse(name) else {
                let why = format!("No mode `{name}`: read-only, ask, edits or auto.");
                notices.write(Notice::error(args.agent, why));
                return;
            };
            mode
        }
    };
    commands.entity(args.agent).insert(mode);
    notices.write(Notice::info(args.agent, format!("Mode: {}.", mode.name())));
}

/// `/permit <id> allow|always|deny`: answers a waiting call, or opens the
/// picker again without arguments.
fn permit(
    In(args): In<CommandArgs>,
    waiting: Query<(Entity, &Awaiting)>,
    mut always: Query<&mut AlwaysAllowed>,
    mut approvals: ResMut<Approvals>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut words = args.args.split_whitespace();
    let (id, answer) = (words.next(), words.next());
    let Some(id) = id.and_then(|id| id.parse::<u64>().ok()) else {
        approvals.shown = None;
        commands.queue(ask_next);
        return;
    };
    let Some((call, awaiting)) = waiting.iter().find(|(_, awaiting)| awaiting.id == id) else {
        notices.write(Notice::error(
            args.agent,
            format!("No call {id} waits for approval."),
        ));
        return;
    };
    let release = |commands: &mut Commands, call: Entity, awaiting: &Awaiting| {
        commands.entity(call).remove::<Awaiting>();
        let (gate, agent, run) = (awaiting.gate, awaiting.agent, awaiting.run.clone());
        commands.queue(move |world: &mut World| run_original(world, gate, call, agent, run));
    };
    match answer {
        Some("allow") => release(&mut commands, call, awaiting),
        Some("always") => {
            let tool = awaiting.tool.clone();
            match always.get_mut(awaiting.agent) {
                Ok(mut always) => always.0.push(tool.clone()),
                Err(_) => {
                    let remembered = AlwaysAllowed(vec![tool.clone()]);
                    commands.entity(awaiting.agent).insert(remembered);
                }
            }
            // The agent's other calls of the tool waiting now run too.
            let same = waiting
                .iter()
                .filter(|(_, other)| other.agent == awaiting.agent && other.tool == tool);
            for (call, awaiting) in same {
                release(&mut commands, call, awaiting);
            }
        }
        Some("deny") => {
            let why = format!(
                "Denied: the user did not allow this `{}` call. Nothing was changed; ask the \
                 user or do something else.",
                awaiting.tool
            );
            let output = ToolOutput(ToolResult::failed(ToolExecutionError::other(why)));
            commands
                .entity(call)
                .remove::<Awaiting>()
                .insert_if_new(output);
        }
        _ => {
            notices.write(Notice::error(args.agent, "Answer allow, always or deny."));
        }
    }
}

/// Shows each agent's mode on the status line.
fn show_mode(
    mut agents: Query<
        (Option<&PermissionMode>, &mut StatusItems),
        (
            With<Agent>,
            Or<(Changed<PermissionMode>, Added<StatusItems>)>,
        ),
    >,
    config: Res<PermissionConfig>,
) {
    for (mode, mut items) in &mut agents {
        let mode = mode_of(mode, &config);
        let tone = match mode {
            PermissionMode::ReadOnly => Tone::Cyan,
            PermissionMode::Ask => Tone::Dim,
            PermissionMode::Edits => Tone::Yellow,
            PermissionMode::Auto => Tone::Red,
        };
        items.show(MODE_ITEM.says(mode.name(), tone));
    }
}
