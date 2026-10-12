//! The tool registry and tool calls. A tool is an entity holding its
//! definition, its [`ToolRules`] and its default [`Footprint`], and, for
//! an ordinary tool, its effect handler; plugins add tools with
//! [`AppToolsExt::add_tool`]. A tool call is an effect entity with a
//! [`ToolRequest`]: the kernel completes it with its tool's [`CallHandler`]
//! when it is admitted, so Prepare may change the call or its handler,
//! gives it the [`Footprint`] of the call as it stands when Gate opens,
//! and starts it in [`Stage::Start`](super::stage::Stage::Start). An ordinary
//! tool's handler answers it; an open tool's call triggers [`ToolCalled`]
//! and stays open until any system decides it
//! ([`Decision`](super::stage::Decision)).
//!
//! ```ignore
//! // `Wait { signal: String }` derives `Deserialize` and `JsonSchema`.
//! app.add_open_tool("wait", "Waits for a signal.", ToolOptions::default(),
//!     |called: On<ToolCalled<Wait>>, mut commands: Commands| {
//!         // Keep `called.call`; decide its `Decision` once `called.args.signal` comes.
//!     });
//! ```

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::{IntoObserverSystem, SystemParam};
use bevy_log::warn;
use bevy_reflect::prelude::*;
use rig_core::completion::ToolDefinition;
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, ToolName};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::{Scope, Tool, ToolErrorKind, Touches, args_schema, tool_definition};
use rig_core::transcript::arguments_refusal;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::agent::{AgentId, ToolAccess};
use super::effects::Handler;
use super::prompt::ToolRules;
use super::stage::{Admitted, Effect, ForAgent, Settled, Started};

/// What the model is told about a tool. A call naming an argument its
/// parameters do not declare is refused before the tool runs.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Serialize, Deserialize)]
#[require(ToolRules, Footprint)]
#[serde(transparent)]
pub struct ToolDef(pub ToolDefinition);

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
        caller: caller.clone(),
        effect: effect.id(),
        parent: effect.parent,
        tool_call: tool_call.clone(),
        args,
    };
    world.entity_mut(call).insert(started);
    world.trigger(called);
    Ok(())
}

/// A tool call: an effect entity, as the model's reply or a plugin spawns
/// it, `(ToolRequest { call }, ForAgent(agent))`, often [`CallOf`] a turn.
/// Admitted, it gets its tool's [`CallHandler`] unless it has its own;
/// Prepare may change the call or the handler, and Gate sees the
/// [`Footprint`] of the call as Prepare left it.
/// Start refuses a call of a tool that is not registered or that the agent
/// may not use, or whose arguments its tool does not declare, then runs
/// its handler, or opens it for an open tool. Its [`Outcome`] is its
/// result.
///
/// [`CallOf`]: super::agent::CallOf
/// [`Outcome`]: super::stage::Outcome
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
/// [`Decision`](super::stage::Decision) is decided, such as with
/// `Outcome::from(ToolResult::success("Done.".into()))`; despawning it, as
/// an [`Interrupt`](super::agent::Interrupt) does, cancels it.
#[derive(EntityEvent, Clone)]
pub struct ToolCalled<A: Send + Sync + 'static> {
    /// The tool.
    pub entity: Entity,
    /// The call.
    pub call: Entity,
    /// The calling agent.
    pub agent: Entity,
    /// The calling agent's id.
    pub caller: AgentId,
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
/// its handler says of its final arguments, else its tool's. A call waits
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

/// On a tool registered with a [`ToolOptions::footprint`]: that footprint
/// is each of its calls', whatever its handler says.
#[derive(Component)]
pub(crate) struct Declared;

/// How a tool is registered with [`AppToolsExt::add_tool_with`].
#[derive(Clone, Debug, Default)]
pub struct ToolOptions<'a> {
    /// Lines of the system prompt of every agent the tool is offered to,
    /// on how to use it, such as "Use `read` to look at files, not `cat` in
    /// `shell`". The tool's description says what it does; its rules say
    /// when to pick it.
    pub rules: &'a [&'a str],
    /// What every call of it touches, whatever its handler says. `None`
    /// asks each call's handler (rig-core's `Tool::footprint`), and for a
    /// call it cannot say, such as an open tool's, takes it from the tool's
    /// annotations: `Reads(Anything)` for a tool that says it only reads,
    /// else `Writes(Anything)`.
    pub footprint: Option<Touches>,
}

/// Registers tools on an [`App`].
pub trait AppToolsExt {
    /// Make `tool` available to every agent whose
    /// [`ToolAccess`] allows its name. A
    /// name already registered is refused with a warning: to replace
    /// another plugin's tool, insert Bevy's `Disabled` on its entity first,
    /// which frees its name (in `Plugin::finish`, once every plugin's
    /// `build` ran).
    ///
    /// Tool futures run on Bevy's async compute pool, a few threads that
    /// every agent's tool calls share. A tool that blocks, such as one
    /// using `std::fs`, `std::process::Command` or a long computation,
    /// wraps that work in `rig_tools::blocking`, which is in the prelude,
    /// so it runs on a thread of its own:
    ///
    /// ```ignore
    /// async fn call(&self, args: Args) -> Result<String, ToolExecutionError> {
    ///     blocking(move || {
    ///         std::fs::read_to_string(&args.path).map_err(ToolExecutionError::from_error)
    ///     })
    ///     .await
    /// }
    /// ```
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        self.add_tool_with(tool, ToolOptions::default())
    }

    /// [`add_tool`](Self::add_tool), with the rules on how to use it and
    /// what its calls touch:
    ///
    /// ```ignore
    /// app.add_tool_with(Outline, ToolOptions {
    ///     rules: &["Use `outline` before reading a large file."],
    ///     footprint: Some(Touches::Reads(Scope::Anything)),
    /// });
    /// ```
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self;

    /// Make the open tool `name` available, its parameters derived from
    /// `A` ([`args_schema`]): each call whose arguments parse into `A`
    /// triggers [`ToolCalled<A>`] on the tool's entity, where `observer`
    /// watches, and ends when the call's
    /// [`Decision`](super::stage::Decision) is decided, at once or much
    /// later; any other is refused.
    /// Its calls are recorded like any other.
    fn add_open_tool<A, M>(
        &mut self,
        name: &str,
        description: &str,
        options: ToolOptions<'_>,
        observer: impl IntoObserverSystem<ToolCalled<A>, M>,
    ) -> &mut Self
    where
        A: DeserializeOwned + JsonSchema + Send + Sync + 'static;
}

impl AppToolsExt for App {
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self {
        let definition = tool_definition(&tool);
        let handler = Handler(ErasedHandler::new(ToolAdapter::new(tool)));
        register_tool(
            self.world_mut(),
            definition,
            Serves::Handler(handler),
            options,
        );
        self
    }

    fn add_open_tool<A, M>(
        &mut self,
        name: &str,
        description: &str,
        options: ToolOptions<'_>,
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
        if let Some(tool) = register_tool(world, definition, Serves::Open(open::<A>), options) {
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
    /// such as a `ToolAdapter`), its footprint from its annotations unless
    /// `options` names one. Its entity, or `None`, with a warning, when its
    /// name is taken.
    fn spawn_tool(
        &mut self,
        definition: ToolDefinition,
        handler: ErasedHandler,
        options: ToolOptions<'_>,
    ) -> Option<Entity>;
}

impl WorldToolsExt for World {
    fn spawn_tool(
        &mut self,
        definition: ToolDefinition,
        handler: ErasedHandler,
        options: ToolOptions<'_>,
    ) -> Option<Entity> {
        let serves = Serves::Handler(Handler(handler));
        register_tool(self, definition, serves, options)
    }
}

/// Spawns the entity of the tool `definition` names, whose calls run as
/// `serves` says, and returns it; `None`, with a warning, when the name is
/// taken.
fn register_tool(
    world: &mut World,
    definition: ToolDefinition,
    serves: Serves,
    options: ToolOptions<'_>,
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
    let footprint = options
        .footprint
        .unwrap_or_else(|| Touches::from_annotations(&definition.annotations));
    let mut entity = world.spawn((
        Name::new(format!("tool:{name}")),
        ToolDef(definition),
        ToolRules(
            options
                .rules
                .iter()
                .map(|rule| (*rule).to_owned())
                .collect(),
        ),
        Footprint(footprint),
        serves,
    ));
    if declared {
        entity.insert(Declared);
    }
    Some(entity.id())
}

/// What `call` touches, run by `handler`, of the tool `tool` (its default
/// footprint, and whether that was declared): the declared one, else what
/// the handler says of the call, else the tool's default. A call of a tool
/// that is not registered may change anything.
pub(crate) fn footprint_of(
    call: &ToolCall,
    handler: Option<&Handler>,
    tool: Option<(&Footprint, bool)>,
) -> Footprint {
    match tool {
        Some((footprint, true)) => footprint.clone(),
        _ => handler
            .and_then(|handler| handler.0.footprint(&EffectKind::from(call)))
            .map(Footprint)
            .or_else(|| tool.map(|(footprint, _)| footprint.clone()))
            .unwrap_or_default(),
    }
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
/// its arguments or handler.
pub(crate) fn compute_footprints(
    calls: Query<
        (
            Entity,
            &ToolRequest,
            Option<&CallHandler>,
            Option<&Footprint>,
        ),
        (With<Admitted>, Without<Started>, Without<Settled>),
    >,
    tools: Query<(&ToolDef, &Footprint, Has<Declared>)>,
    mut commands: Commands,
) {
    for (entity, ToolRequest { call }, handler, had) in &calls {
        let name = call.function.name.as_str();
        let tool = tools
            .iter()
            .find(|(def, ..)| def.0.name.as_str() == name)
            .map(|(_, footprint, declared)| (footprint, declared));
        let footprint = footprint_of(call, handler.map(|handler| &handler.0), tool);
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

/// What starting a tool call reads: the registered tools and the agents'
/// access to them.
#[derive(SystemParam)]
pub(crate) struct ToolStarts<'w, 's> {
    tools: Query<'w, 's, (Entity, &'static ToolDef, &'static Serves)>,
    access: Query<'w, 's, &'static ToolAccess>,
}

impl ToolStarts<'_, '_> {
    /// How `call`, with `handler`, starts for `agent`:
    /// refused when no tool of its name is registered, the agent may not
    /// use it (a refusal, `Denied`), or its arguments do not fit the tool's
    /// parameters; else run through `handler`, or opened for an open tool.
    pub(crate) fn start(
        &self,
        call: &ToolCall,
        handler: Option<&CallHandler>,
        agent: Option<Entity>,
    ) -> ToolStart {
        let name = call.function.name.as_str();
        let refused = |kind, why| ToolStart::Refused(ErrorReport::new(ErrorKind::Tool(kind), why));
        let Some(agent) = agent else {
            return refused(
                ToolErrorKind::NotFound,
                "a tool call needs an agent".to_owned(),
            );
        };
        let allowed = self
            .access
            .get(agent)
            .is_ok_and(|access| access.allows(name));
        let tool = self
            .tools
            .iter()
            .find(|(_, def, _)| def.0.name.as_str() == name);
        let missing = format!("no tool named `{name}` is available");
        let Some((tool, def, serves)) = tool else {
            return refused(ToolErrorKind::NotFound, missing);
        };
        if !allowed {
            // A refusal, recorded as one; the model reads what it reads for
            // a tool that does not exist.
            return ToolStart::Refused(ErrorReport::new(ErrorKind::Denied, missing).refused());
        }
        if let Some(why) = arguments_refusal(&def.0.parameters, call) {
            return refused(ToolErrorKind::InvalidArgs, why);
        }
        match (handler, serves) {
            (Some(CallHandler(handler)), _) | (None, Serves::Handler(handler)) => {
                ToolStart::Run(handler.clone(), EffectKind::from(call))
            }
            (None, Serves::Open(open)) => ToolStart::Open(*open, tool),
        }
    }
}
