//! The tool registry and tool calls. A tool is an entity holding its
//! definition and its default [`Footprint`], and, for an
//! ordinary tool, its effect handler; plugins add tools with
//! [`AppToolsExt::add_tool`]. A tool call is an effect entity with a
//! [`ToolRequest`]. An ordinary tool's handler answers it; an open tool's
//! call triggers [`ToolCalled`] and stays open until any system decides it
//! ([`Decision`]).

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::{IntoObserverSystem, SystemParam};
use bevy_log::warn;
use bevy_reflect::prelude::*;
use rig_core::completion::ToolDefinition;
use rig_core::effect::{Answered, EffectId, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, ToolName};
use rig_core::redact;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::feedback::{arguments_refusal, unknown_tool_feedback};
use rig_core::tool::{Scope, Tool, ToolErrorKind, Touches, args_schema, tool_definition};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use super::agent::{AgentId, CallOf, ToolAccess, TurnOutcome};
use super::effects::Handler;
use super::stage::{
    Admitted, Decision, Effect, ForAgent, Outcome, Settled, Started, SupersededBy, ToolCalls,
};
use super::turn::Ending;

/// What the model is told about a tool. A call naming an argument its
/// parameters do not declare is refused before the tool runs.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Serialize, Deserialize)]
#[require(Footprint)]
#[serde(transparent)]
pub struct ToolDef(pub Arc<ToolDefinition>);

/// How a tool's calls run, on the tool's entity.
#[derive(Component, Clone)]
pub(crate) enum Serves {
    /// An ordinary tool: its effect handler runs each call.
    Handler(Handler),
    /// An open tool: starts its calls ([`Open`]).
    Open(Open),
}

/// Starts the call `call` of the open tool `tool`, read from the call's
/// own components: its arguments parsed into the tool's arguments type, it
/// is [`Started`] and its [`ToolCalled`] triggered. Why it cannot start,
/// such as arguments that do not fit, and it is not started.
pub(crate) type Open = fn(&mut World, Entity, Entity, Started) -> Result<(), ErrorReport>;

/// [`Open`] for the arguments type `A`. A call gone, or no tool call any
/// more, is left alone.
fn open<A: DeserializeOwned + Send + Sync + 'static>(
    world: &mut World,
    tool: Entity,
    call: Entity,
    started: Started,
) -> Result<(), ErrorReport> {
    let refused = |kind, why| ErrorReport::new(ErrorKind::Tool(kind), why);
    let Ok(entity) = world.get_entity(call) else {
        return Ok(());
    };
    let (Some(ToolRequest { call: tool_call }), Some(effect)) =
        (entity.get::<ToolRequest>(), entity.get::<Effect>())
    else {
        return Ok(());
    };
    let agent = entity.get::<ForAgent>().map(|agent| agent.0);
    let caller = agent.and_then(|agent| world.get::<AgentId>(agent));
    let (Some(agent), Some(caller)) = (agent, caller) else {
        let why = "a tool call needs an agent".to_owned();
        return Err(refused(ToolErrorKind::NotFound, why));
    };
    let args = A::deserialize(&tool_call.function.arguments).map_err(|error| {
        let why = format!(
            "The arguments do not fit: {error}. Nothing ran; fix the call and send it again."
        );
        refused(ToolErrorKind::InvalidArgs, why)
    })?;
    let called = ToolCalled {
        entity: tool,
        call,
        agent,
        agent_id: caller.clone(),
        effect: effect.id(),
        parent: effect.parent,
        tool_call: tool_call.clone(),
        args,
    };
    world.entity_mut(call).insert(started);
    world.trigger(called);
    Ok(())
}

/// A tool call: an effect entity, spawned by the model's reply or a plugin as
/// `(ToolRequest { call }, ForAgent(agent))`, often [`CallOf`] a turn. Its
/// [`Outcome`] is its result.
///
/// When admitted it gets its tool's [`CallHandler`] unless it has one;
/// Prepare may change the call or the handler, once. Each time Gate opens the
/// kernel gives it the [`Footprint`] of the call as it stands, so a held call
/// edited meanwhile is gated again. After the answers from a record and
/// before every gate, the kernel refuses a call no record decided whose tool
/// is unregistered or not the agent's to use, or whose arguments its tool
/// does not declare; no gate sees it and no later answer replaces it. Start
/// runs the handler, or opens the call for an open tool.
///
/// An unavailable name gets feedback listing the agent's allowed tools.
/// Calling the same unavailable name three times in one turn fails that
/// turn, even when successful calls occur between them.
///
/// [`CallOf`]: super::agent::CallOf
#[derive(Component, Clone, Debug)]
#[require(Effect)]
pub struct ToolRequest {
    /// The call as the model sent it.
    pub call: ToolCall,
}

/// The handler that answers a tool call, on the call: its tool's, which a
/// Prepare system may wrap (rig-core's `ErasedHandler::layered`) or
/// replace. A call without one is opened for its open tool.
#[derive(Component, Clone)]
pub struct CallHandler(pub Handler);

/// A call of an open tool started, its arguments parsed into `A`: triggered
/// on the tool's entity, so the observer [`AppToolsExt::add_open_tool`]
/// registered handles it. The call stays open until its
/// [`Decision`] is decided, such as with
/// `Outcome::from(ToolResult::success("Done.".into()))`; despawning it, as
/// a [`StopTurn`](super::agent::StopTurn) does, cancels it.
#[derive(EntityEvent, Clone)]
pub struct ToolCalled<A: Send + Sync + 'static> {
    /// The tool.
    pub entity: Entity,
    /// The call.
    pub call: Entity,
    /// The calling agent.
    pub agent: Entity,
    /// The calling agent's id.
    pub agent_id: AgentId,
    /// The call's effect, which work done for it is recorded under.
    pub effect: EffectId,
    /// The effect that asked for the call, such as the model call whose
    /// reply it is.
    pub parent: Option<EffectId>,
    /// The call as the model sent it.
    pub tool_call: ToolCall,
    /// The call's arguments.
    pub args: A,
}

/// What a tool call may read and change ([`Touches`]), so the calls of one
/// reply that touch different things run side by side. On a tool's entity,
/// what its calls touch when they cannot say more; on a call, what it
/// touches as Gate sees it, computed by the kernel when Gate opens: what
/// its handler says of its final arguments, else its tool's, its paths
/// [`normalized`](Touches::normalized), unless a plugin [`DeclaredFootprint`] it on
/// the call. A call waits
/// while an earlier call of its reply that conflicts with it is not
/// settled ([`Waiting`](super::stage::Waiting)).
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Debug, Default, PartialEq)]
#[serde(transparent)]
pub struct Footprint(pub Touches);

impl Default for Footprint {
    /// May change anything, right for any tool that does not say.
    fn default() -> Self {
        Self(Touches::Writes(Scope::Anything))
    }
}

/// A [`Footprint`] that stands. On a tool registered with a
/// [`ToolOptions::footprint`]: that footprint is each of its calls',
/// whatever its handler says. On a call, inserted with its `Footprint` in
/// Prepare: the kernel keeps that footprint (made
/// [`normalized`](Touches::normalized)) instead of asking the call's
/// handler, so a plugin that knows what a call touches, such as a shell
/// command that only reads, lets the reply's other calls run beside it.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct DeclaredFootprint;

/// How a tool is registered with [`AppToolsExt::add_tool_with`].
#[derive(Clone, Debug, Default)]
pub struct ToolOptions {
    /// What every call of it touches, whatever its handler says: a local
    /// declaration, which stands, such as one that trusts a remote tool to
    /// only read. `None` asks each call's handler
    /// (rig-core's `Tool::footprint`), and for a call it cannot say, such
    /// as an open tool's, takes it from the tool's annotations: those of a
    /// typed tool ([`AppToolsExt::add_tool_with`]) as its code declares
    /// them (`Touches::from_annotations`), any other's as hints that never
    /// make a call read only (`Touches::from_hints`).
    pub footprint: Option<Touches>,
}

/// Registers tools on an [`App`].
pub trait AppToolsExt {
    /// Makes `tool` available to every agent whose [`ToolAccess`] allows its
    /// name. A name already registered is refused with a warning; to replace
    /// another plugin's tool, insert Bevy's `Disabled` on its entity first, in
    /// `Plugin::finish`.
    ///
    /// Tool futures run on Bevy's async compute pool, a few threads every
    /// agent's calls share, so a tool that blocks (`std::fs`,
    /// `std::process::Command`, a long computation) runs that work on a thread
    /// of its own, such as with rig-tools' `blocking`.
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        self.add_tool_with(tool, ToolOptions::default())
    }

    /// [`add_tool`](Self::add_tool), with what its calls touch.
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions) -> &mut Self;

    /// Make the open tool `name` available, its parameters derived from
    /// `A` ([`args_schema`]): each call whose arguments parse into `A`
    /// triggers [`ToolCalled<A>`] on the tool's entity, where `observer`
    /// watches, and ends when the call's
    /// [`Decision`] is decided, at once or much
    /// later; any other is refused.
    /// Its calls are recorded like any other.
    fn add_open_tool<A, M>(
        &mut self,
        name: &str,
        description: &str,
        options: ToolOptions,
        observer: impl IntoObserverSystem<ToolCalled<A>, M>,
    ) -> &mut Self
    where
        A: DeserializeOwned + JsonSchema + Send + Sync + 'static;
}

impl AppToolsExt for App {
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions) -> &mut Self {
        let definition = tool_definition(&tool);
        let declared = Touches::from_annotations(&definition.annotations);
        let handler = Handler(ErasedHandler::new(ToolAdapter::new(tool)));
        let serves = Serves::Handler(handler);
        register_tool(self.world_mut(), definition, serves, options, declared);
        self
    }

    fn add_open_tool<A, M>(
        &mut self,
        name: &str,
        description: &str,
        options: ToolOptions,
        observer: impl IntoObserverSystem<ToolCalled<A>, M>,
    ) -> &mut Self
    where
        A: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    {
        let world = self.world_mut();
        let Ok(tool_name) = ToolName::new(name) else {
            warn!("tool not registered: `{name}` is not a valid tool name");
            return self;
        };
        let definition = ToolDefinition::new(tool_name, description, args_schema::<A>());
        let serves = Serves::Open(open::<A>);
        let changes = Touches::Writes(Scope::Anything);
        if let Some(tool) = register_tool(world, definition, serves, options, changes) {
            world.entity_mut(tool).observe(observer);
        }
        self
    }
}

/// Registers tools while the app runs, such as those an MCP server lists
/// once it answers, from a system:
/// `commands.queue(move |world: &mut World| { world.spawn_tool(..); })`.
/// Each request reads the registered tools again, so the next one offers
/// it; despawning its entity removes it. A tool list that changes misses
/// the provider's prompt cache once.
pub trait WorldToolsExt {
    /// Make the tool `definition` describes available as
    /// [`AppToolsExt::add_tool_with`] does, its calls answered by `handler`
    /// on the one dispatch path (a rig-core `Serve` of the tool family,
    /// such as a `ToolAdapter`), its footprint the one `options` names, else
    /// its handler's for each call, else what its annotations hint
    /// (`Touches::from_hints`): they may be a remote server's, so a tool
    /// that says it only reads is never run again after a restart. Its
    /// entity, or `None`, with a warning, when its name is taken.
    fn spawn_tool(
        &mut self,
        definition: ToolDefinition,
        handler: ErasedHandler,
        options: ToolOptions,
    ) -> Option<Entity>;
}

impl WorldToolsExt for World {
    fn spawn_tool(
        &mut self,
        definition: ToolDefinition,
        handler: ErasedHandler,
        options: ToolOptions,
    ) -> Option<Entity> {
        let serves = Serves::Handler(Handler(handler));
        let hinted = Touches::from_hints(&definition.annotations);
        register_tool(self, definition, serves, options, hinted)
    }
}

/// Spawns the entity of the tool `definition` names, whose calls run as
/// `serves` says, touching what `options` declares, else `default`, and
/// returns it; `None`, with a warning, when the name is taken.
fn register_tool(
    world: &mut World,
    definition: ToolDefinition,
    serves: Serves,
    options: ToolOptions,
    default: Touches,
) -> Option<Entity> {
    let name = definition.name.as_str().to_owned();
    if world
        .query::<&ToolDef>()
        .iter(world)
        .any(|def| def.0.name == definition.name)
    {
        warn!("tool not registered: `{name}` exists; insert `Disabled` on it to replace it");
        return None;
    }
    let declared = options.footprint.is_some();
    let footprint = options.footprint.unwrap_or(default);
    let mut entity = world.spawn((
        Name::new(format!("tool:{name}")),
        ToolDef(definition.into()),
        Footprint(footprint),
        serves,
    ));
    if declared {
        entity.insert(DeclaredFootprint);
    }
    Some(entity.id())
}

/// What `call` touches, run by `handler`, of the tool `tool` (its default
/// footprint, and whether that was declared): the declared one, else what
/// the handler says of the call, else the tool's default, made
/// [`normalized`](Touches::normalized) so no `..` climbs past a guard. A
/// call of a tool that is not registered may change anything.
pub(crate) fn footprint_of(
    call: &ToolCall,
    handler: Option<&Handler>,
    tool: Option<(&Footprint, bool)>,
) -> Footprint {
    let Footprint(touches) = match tool {
        Some((footprint, true)) => footprint.clone(),
        _ => handler
            .and_then(|handler| handler.0.footprint(&EffectKind::from(call)))
            .map(Footprint)
            .or_else(|| tool.map(|(footprint, _)| footprint.clone()))
            .unwrap_or_default(),
    };
    Footprint(touches.normalized())
}

/// Completes each tool call about to be admitted with its tool's
/// [`CallHandler`], and a [`Name`], where it has none of its own.
pub(crate) fn complete_calls(
    calls: Query<(Entity, &ToolRequest), (Without<Admitted>, Without<Settled>)>,
    tools: Query<(&ToolDef, &Serves)>,
    mut commands: Commands,
) {
    for (entity, ToolRequest { call }) in &calls {
        let name = call.function.name.as_str();
        let tool = tools.iter().find(|(def, _)| def.0.name.as_str() == name);
        let mut completed = commands.entity(entity);
        completed.try_insert_if_new(Name::new(format!("tool call {name}")));
        if let Some((_, Serves::Handler(handler))) = tool {
            completed.try_insert_if_new(CallHandler(handler.clone()));
        }
    }
}

/// Gives each admitted tool call that has not started the [`Footprint`]
/// of the call as it stands, when Gate opens: after every Prepare edit of
/// its arguments or handler; one a plugin [`DeclaredFootprint`] on the call stands.
pub(crate) fn compute_footprints(
    calls: Query<
        (
            Entity,
            &ToolRequest,
            Option<&CallHandler>,
            Option<&Footprint>,
            Has<DeclaredFootprint>,
        ),
        (With<Admitted>, Without<Started>, Without<Settled>),
    >,
    tools: Query<(&ToolDef, &Footprint, Has<DeclaredFootprint>)>,
    mut commands: Commands,
) {
    for (entity, ToolRequest { call }, handler, had, declared) in &calls {
        let footprint = match had {
            Some(Footprint(touches)) if declared => Footprint(touches.clone().normalized()),
            _ => {
                let name = call.function.name.as_str();
                let tool = tools
                    .iter()
                    .find(|(def, ..)| def.0.name.as_str() == name)
                    .map(|(_, footprint, declared)| (footprint, declared));
                footprint_of(call, handler.map(|handler| &handler.0), tool)
            }
        };
        if had != Some(&footprint) {
            commands.entity(entity).try_insert(footprint);
        }
    }
}

/// How Start starts a tool call.
pub(crate) enum ToolStart {
    /// The kernel runs `kind` through the handler.
    Run(Handler, EffectKind),
    /// The open call is started by its tool, the entity.
    Open(Open, Entity),
    /// Refused before it started.
    Refused(ErrorReport),
}

/// The tools a model working for an agent with `access` is offered: every
/// registered one it allows, sorted by name so the prompt cache keeps them.
pub(crate) fn offered<'a>(
    tools: impl IntoIterator<Item = &'a ToolDef>,
    access: &ToolAccess,
) -> Vec<&'a ToolDef> {
    let mut offered: Vec<&ToolDef> = (tools.into_iter())
        .filter(|def| access.allows(def.0.name.as_str()))
        .collect();
    offered.sort_by(|a, b| a.0.name.as_str().cmp(b.0.name.as_str()));
    offered
}

/// What starting a tool call reads: the registered tools and the agents'
/// access to them.
#[derive(SystemParam)]
pub(crate) struct ToolStarts<'w, 's> {
    tools: Query<'w, 's, (Entity, &'static ToolDef, &'static Serves)>,
    defs: Query<'w, 's, &'static ToolDef>,
    agents: Query<'w, 's, Option<&'static ToolAccess>, With<AgentId>>,
}

impl ToolStarts<'_, '_> {
    /// The tool that runs `call` for `agent`, or why the kernel refuses
    /// it: no agent, no tool of its name is registered, the agent may not
    /// use it (a refusal, `Denied`), or its arguments do not fit the tool's
    /// parameters.
    fn tool(
        &self,
        call: &ToolCall,
        agent: Option<Entity>,
    ) -> Result<(Entity, &Serves), ErrorReport> {
        let name = call.function.name.as_str();
        let refused = |kind, why| ErrorReport::new(ErrorKind::Tool(kind), why);
        let Some(access) = agent.and_then(|agent| self.agents.get(agent).ok()) else {
            let why = "a tool call needs an agent".to_owned();
            return Err(refused(ToolErrorKind::NotFound, why));
        };
        let missing = || {
            let offered = access.map(|access| offered(self.defs.iter(), access));
            let names = offered.iter().flatten().map(|def| def.0.name.as_str());
            unknown_tool_feedback(name, names)
        };
        let Some((tool, def, serves)) = self
            .tools
            .iter()
            .find(|(_, def, _)| def.0.name.as_str() == name)
        else {
            return Err(refused(ToolErrorKind::NotFound, missing()));
        };
        if !access.is_some_and(|access| access.allows(name)) {
            return Err(ErrorReport::new(ErrorKind::Denied, missing()).refused());
        }
        if let Some(why) = arguments_refusal(&def.0.parameters, call) {
            return Err(refused(ToolErrorKind::InvalidArgs, why));
        }
        Ok((tool, serves))
    }

    /// How `call`, with `handler`, starts for `agent`: run through
    /// `handler`, or opened for an open tool; refused as
    /// [`refuse_calls`] refuses it, should it have changed since.
    pub(crate) fn start(
        &self,
        call: &ToolCall,
        handler: Option<&CallHandler>,
        agent: Option<Entity>,
    ) -> ToolStart {
        match (self.tool(call, agent), handler) {
            (Err(report), _) => ToolStart::Refused(report),
            (Ok(_), Some(CallHandler(handler))) | (Ok((_, Serves::Handler(handler))), None) => {
                ToolStart::Run(handler.clone(), EffectKind::from(call))
            }
            (Ok((tool, Serves::Open(open))), None) => ToolStart::Open(*open, tool),
        }
    }
}

/// Unavailable names called during this turn, including calls between successes.
#[derive(Component, Default)]
pub(crate) struct MissingTools(HashMap<String, u32>);

/// Answers each admitted tool call that has not started, held or not,
/// that the kernel refuses ([`ToolStarts`]), unless it is decided: after
/// the answers from a record and before every gate, so no gate sees it,
/// and one after the gates reads its [`Outcome`]. Run on every pass, so
/// what changed since, such as the agent's [`ToolAccess`], counts.
pub(crate) fn refuse_calls(
    calls: Query<
        (
            Entity,
            &ToolRequest,
            &Decision,
            Option<&ForAgent>,
            Option<&CallOf>,
        ),
        (
            With<Admitted>,
            Without<Started>,
            Without<Outcome>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
    tools: ToolStarts,
    mut turns: Query<&mut MissingTools>,
    ending: Query<(), With<TurnOutcome>>,
    mut commands: Commands,
) {
    for (entity, ToolRequest { call }, decision, agent, turn) in &calls {
        if decision.is_decided() {
            continue;
        }
        if let Err(report) = tools.tool(call, agent.map(|agent| agent.0)) {
            if matches!(
                report.kind,
                ErrorKind::Tool(ToolErrorKind::NotFound) | ErrorKind::Denied
            ) && let Some(&CallOf(turn)) = turn
                && let Ok(mut missing) = turns.get_mut(turn)
            {
                let name = call.function.name.as_str();
                let count = missing.0.entry(name.to_owned()).or_default();
                *count += 1;
                // It fails only this turn, so the agent's subagents go on,
                // unless the turn already ends some other way.
                if *count == 3 && !ending.contains(turn) {
                    let why = format!(
                        "The turn failed: the model kept calling `{name}`, which is not one \
                         of its tools"
                    );
                    commands
                        .entity(turn)
                        .try_insert((TurnOutcome::Failed(why), Ending));
                }
            }
            commands
                .entity(entity)
                .try_insert_if_new(Outcome(Err(report)));
        }
    }
}

/// Redacts every secret in each tool call's answer, such as a key `env`
/// printed ([`redact::tool_result`]), as Commit opens, after every
/// reviewer and before the answer is committed: no key a tool returns
/// reaches the model, the conversation, the transcript or a log. Only an answer changed since the
/// last pass is searched, and only one holding a secret is copied.
pub(crate) fn redact_results(
    mut calls: Query<&mut Outcome, (ToolCalls, Changed<Outcome>, Without<Settled>)>,
) {
    for mut outcome in &mut calls {
        let redacted = match &outcome.0 {
            Ok(Answered::ToolResult { result }) => {
                redact::tool_result(result).map(|result| Ok(Answered::ToolResult { result }))
            }
            Err(report) => match redact::text(&report.message) {
                Cow::Owned(message) => Some(Err(ErrorReport {
                    message,
                    ..report.clone()
                })),
                Cow::Borrowed(_) => None,
            },
            Ok(_) => None,
        };
        if let Some(redacted) = redacted {
            outcome.0 = redacted;
        }
    }
}
