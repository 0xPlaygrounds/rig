//! The tool registry and tool calls. A tool is an entity holding its
//! definition, its [`ToolRules`] and its [`Footprint`], and, for an
//! ordinary tool, its effect handler; plugins add tools with
//! [`AppToolsExt::add_tool`]. A tool call is an effect entity with a
//! [`ToolRequest`]: the kernel completes it with its tool's [`Footprint`]
//! and [`CallHandler`] when it is admitted, so Prepare may change them, and
//! starts it in [`Stage::Start`](super::stage::Stage::Start). An ordinary
//! tool's handler answers it; an open tool's call triggers [`ToolCalled`]
//! and stays open until any system inserts its
//! [`Outcome`](super::stage::Outcome).
//!
//! ```ignore
//! // `Wait { signal: String }` derives `Deserialize` and `JsonSchema`.
//! app.add_open_tool("wait", "Waits for a signal.", ToolOptions::default(),
//!     |called: On<ToolCalled<Wait>>, mut commands: Commands| {
//!         // Keep `called.call`; insert its `Outcome` once `called.args.signal` comes.
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
use rig_core::tool::{Tool, ToolErrorKind, args_schema, tool_definition};
use rig_core::transcript::arguments_refusal;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::agent::{AgentId, ToolAccess};
use super::effects::Handler;
use super::prompt::ToolRules;
use super::stage::{Admitted, Effect, Settled};

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
    /// An open tool: parses a call's arguments into its arguments type.
    Open(fn(&ToolCall) -> Result<Opened, String>),
}

/// Triggers [`ToolCalled`] for an open call, its arguments parsed.
pub(crate) type Opened = Box<dyn FnOnce(&mut World, Opening) + Send>;

/// An open call as its [`ToolCalled`] names it, but its arguments.
pub(crate) struct Opening {
    tool: Entity,
    call: Entity,
    agent: Entity,
    caller: AgentId,
    effect: EffectId,
    parent: Option<EffectId>,
    tool_call: ToolCall,
}

/// Parses `call`'s arguments into `A`, or says why they do not fit.
fn open<A: DeserializeOwned + Send + Sync + 'static>(call: &ToolCall) -> Result<Opened, String> {
    let args = A::deserialize(&call.function.arguments).map_err(|error| {
        format!("The arguments do not fit: {error}. Nothing ran; fix the call and send it again.")
    })?;
    Ok(Box::new(move |world, opening: Opening| {
        let Opening {
            tool,
            call,
            agent,
            caller,
            effect,
            parent,
            tool_call,
        } = opening;
        world.trigger(ToolCalled {
            entity: tool,
            call,
            agent,
            caller,
            effect,
            parent,
            tool_call,
            args,
        });
    }))
}

/// A tool call: an effect entity, as the model's reply or a plugin spawns
/// it, `(ToolRequest { call }, ForAgent(agent))`, often [`CallOf`] a turn.
/// Admitted, it gets its tool's [`Footprint`] and [`CallHandler`] unless it
/// has its own; Prepare may change the call, the footprint or the handler.
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
/// registered handles it. The call stays open until an
/// [`Outcome`](super::stage::Outcome) is inserted on it, such as
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

/// Whether a tool's calls may run beside the other calls of one reply, on
/// the tool's entity.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Footprint {
    /// May change anything: a call waits for every earlier call of its
    /// reply and holds back every later one. The default, right for
    /// `write`, `edit`, `shell` and any tool that does not say.
    #[default]
    Exclusive,
    /// Changes nothing: calls run side by side with every call that is
    /// not exclusive, and a call a restart cut short runs again.
    ReadOnly,
    /// Changes nothing the reply's other calls see, such as handing work
    /// to another agent: runs side by side like [`ReadOnly`](Self::ReadOnly),
    /// but is never run again after a restart.
    Independent,
}

impl Footprint {
    /// Whether a call of this footprint must wait for an earlier call of
    /// its reply with footprint `earlier`.
    pub fn waits_for(self, earlier: Self) -> bool {
        self == Self::Exclusive || earlier == Self::Exclusive
    }
}

/// How a tool is registered with [`AppToolsExt::add_tool_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolOptions<'a> {
    /// Lines of the system prompt of every agent the tool is offered to,
    /// on how to use it, such as "Use `read` to look at files, not `cat` in
    /// `shell`". The tool's description says what it does; its rules say
    /// when to pick it.
    pub rules: &'a [&'a str],
    /// Whether its calls run beside others. `None` takes it from the tool's
    /// annotations: [`Footprint::ReadOnly`] for a tool that says it only
    /// reads, else [`Footprint::Exclusive`].
    pub footprint: Option<Footprint>,
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
    /// whether its calls run beside others:
    ///
    /// ```ignore
    /// app.add_tool_with(Outline, ToolOptions {
    ///     rules: &["Use `outline` before reading a large file."],
    ///     footprint: Some(Footprint::ReadOnly),
    /// });
    /// ```
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self;

    /// Make the open tool `name` available, its parameters derived from
    /// `A` ([`args_schema`]): each call whose arguments parse into `A`
    /// triggers [`ToolCalled<A>`] on the tool's entity, where `observer`
    /// watches, and ends when an [`Outcome`](super::stage::Outcome) is
    /// inserted on the call, at once or much later; any other is refused.
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
    let footprint = options
        .footprint
        .unwrap_or(match definition.annotations.read_only() {
            true => Footprint::ReadOnly,
            false => Footprint::Exclusive,
        });
    let entity = world.spawn((
        Name::new(format!("tool:{name}")),
        ToolDef(definition),
        ToolRules(
            options
                .rules
                .iter()
                .map(|rule| (*rule).to_owned())
                .collect(),
        ),
        footprint,
        serves,
    ));
    Some(entity.id())
}

/// The kind of effect a tool call is.
pub(crate) fn tool_kind(call: &ToolCall) -> EffectKind {
    EffectKind::ToolCall {
        name: call.function.name.as_str().to_owned(),
        args: call.function.raw_arguments(),
    }
}

/// Completes each tool call about to be admitted with its tool's
/// [`Footprint`] and [`CallHandler`], and a [`Name`], where it has none of
/// its own. A call of a tool that is not registered may change anything.
pub(crate) fn complete_calls(
    calls: Query<(Entity, &ToolRequest), (Without<Admitted>, Without<Settled>)>,
    tools: Query<(&ToolDef, &Serves, &Footprint)>,
    mut commands: Commands,
) {
    for (entity, ToolRequest { call }) in &calls {
        let name = call.function.name.as_str();
        let tool = tools.iter().find(|(def, ..)| def.0.name.as_str() == name);
        let mut completed = commands.entity(entity);
        completed.try_insert_if_new((
            Name::new(format!("tool call {name}")),
            tool.map_or_else(Footprint::default, |(.., &footprint)| footprint),
        ));
        if let Some((_, Serves::Handler(handler), _)) = tool {
            completed.try_insert_if_new(CallHandler(handler.clone()));
        }
    }
}

/// How Start starts a tool call.
pub(crate) enum ToolStart {
    /// The kernel runs `kind` through the handler.
    Run(Handler, EffectKind),
    /// The open call is started: this triggers its [`ToolCalled`] with the
    /// call's entity, effect, parent and call.
    Open(Box<dyn FnOnce(&mut World, Entity, EffectId, Option<EffectId>, ToolCall) + Send>),
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
    /// How `call`, with `handler`, starts for `agent` (its entity and id):
    /// refused when no tool of its name is registered, the agent may not
    /// use it (a refusal, `Denied`), or its arguments do not fit the tool's
    /// parameters; else run through `handler`, or opened for an open tool.
    pub(crate) fn start(
        &self,
        call: &ToolCall,
        handler: Option<&CallHandler>,
        agent: Option<(Entity, &AgentId)>,
    ) -> ToolStart {
        let name = call.function.name.as_str();
        let refused = |kind, why| ToolStart::Refused(ErrorReport::new(ErrorKind::Tool(kind), why));
        let Some((agent, caller)) = agent else {
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
                ToolStart::Run(handler.clone(), tool_kind(call))
            }
            (None, Serves::Open(open)) => match open(call) {
                Ok(opened) => {
                    let caller = caller.clone();
                    ToolStart::Open(Box::new(move |world, call, effect, parent, tool_call| {
                        let opening = Opening {
                            tool,
                            call,
                            agent,
                            caller,
                            effect,
                            parent,
                            tool_call,
                        };
                        opened(world, opening);
                    }))
                }
                Err(why) => refused(ToolErrorKind::InvalidArgs, why),
            },
        }
    }
}
