//! The tool registry and how tool calls end. A tool is an entity holding
//! its definition, its [`ToolRules`] and its [`Footprint`], and, for an
//! ordinary tool, its effect handler; plugins add tools with
//! [`AppToolsExt::add_tool`]. A tool call is an entity too, and inserting a
//! [`ToolOutput`] on it is the one way it ends: an ordinary tool's output
//! is inserted when its future resolves, and an open tool's call stays
//! open until any system or observer inserts one.
//!
//! ```ignore
//! // `Wait { signal: String }` derives `Deserialize` and `JsonSchema`.
//! app.add_open_tool("wait", "Waits for a signal.", ToolOptions::default(),
//!     |called: On<ToolCalled<Wait>>, mut commands: Commands| {
//!         // Keep `called.call`; insert its `ToolOutput` once `called.args.signal` comes.
//!     });
//! ```

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::IntoObserverSystem;
use bevy_log::tracing::Instrument;
use bevy_log::{info_span, warn};
use bevy_reflect::prelude::*;
use bevy_tasks::ConditionalSendFuture;
use rig_core::completion::ToolDefinition;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::message::{ToolCall, ToolName};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{ErasedHandler, OpenRecord};
use rig_core::tool::{Tool, ToolExecutionError, ToolResult, args_schema};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::agent::{AgentId, ToolCallRun};
use super::effects::{Effects, Handler};
use super::prompt::ToolRules;

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

/// Triggers [`ToolCalled`] with an open call's parsed arguments.
pub(crate) type Opened =
    Box<dyn FnOnce(&mut World, [Entity; 3], AgentId, EffectId, ToolCallRun) + Send>;

/// Parses `call`'s arguments into `A`, or says why they do not fit.
fn open<A: DeserializeOwned + Send + Sync + 'static>(call: &ToolCall) -> Result<Opened, String> {
    let args = A::deserialize(&call.function.arguments).map_err(|error| {
        format!("The arguments do not fit: {error}. Nothing ran; fix the call and send it again.")
    })?;
    Ok(Box::new(
        move |world, [entity, call, agent], caller, effect, run| {
            world.trigger(ToolCalled {
                entity,
                call,
                agent,
                caller,
                effect,
                run,
                args,
            });
        },
    ))
}

/// What a tool call did, inserted on the call entity: the one way a call
/// ends, such as `ToolOutput(ToolResult::success("Done.".into()))`. Insert
/// it once, with [`EntityCommands::insert_if_new`] when another system may
/// answer the same call; the first one counts. Work off the main thread
/// ends with one as a `Running::spawn_into::<ToolOutput, _>` task
/// ([`Running`](super::calls::Running)). The turn then starts the
/// calls that waited for this one and, once every call of the reply has
/// one, sends their results to the model.
#[derive(Component, Reflect, Clone, Debug, Serialize, Deserialize)]
#[reflect(opaque, Component, Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolOutput(pub ToolResult);

impl From<ToolResult> for ToolOutput {
    fn from(result: ToolResult) -> Self {
        Self(result)
    }
}

/// A call of an open tool started, its arguments parsed into `A`: triggered
/// on the tool's entity, while the call and the calling agent exist, so the
/// observer [`AppToolsExt::add_open_tool`] registered handles it. The call
/// entity is usually a [`CallOf`](super::agent::CallOf) the turn of
/// `agent`; it stays open until a [`ToolOutput`] is inserted on it, and
/// despawning it, as an [`Interrupt`](super::agent::Interrupt) does,
/// cancels it.
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
    /// The call as the model sent it, and the model call that asked for it.
    pub run: ToolCallRun,
    /// The call's arguments.
    pub args: A,
}

/// The effect record of an open tool call, settled with the call's
/// [`ToolOutput`].
#[derive(Component)]
pub(crate) struct OpenCall(pub(crate) OpenRecord);

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
    /// the reply with footprint `earlier`.
    pub(crate) fn waits_for(self, earlier: Self) -> bool {
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
    /// Whether its calls run beside others; the default runs each call on
    /// its own.
    pub footprint: Footprint,
}

/// Registers tools on an [`App`].
pub trait AppToolsExt {
    /// Make `tool` available to every agent whose
    /// [`ToolAccess`](crate::agent::ToolAccess) allows its name. A
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
    ///     footprint: Footprint::ReadOnly,
    /// });
    /// ```
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self;

    /// Make the open tool `name` available, its parameters derived from
    /// `A` ([`args_schema`]): each call whose arguments parse into `A`
    /// triggers [`ToolCalled<A>`] on the tool's entity, where `observer`
    /// watches, and ends when a [`ToolOutput`] is inserted on the call, at
    /// once or much later; any other is refused. Its calls are recorded in
    /// the effect log like any other, with that output as the outcome.
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
        let (description, parameters) = (tool.description(), tool.parameters());
        let handler = Handler(ErasedHandler::new(ToolAdapter::new(tool)));
        register_tool(
            self.world_mut(),
            T::NAME,
            description,
            parameters,
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
        if let Some(tool) = register_tool(
            world,
            name,
            description.to_owned(),
            args_schema::<A>(),
            Serves::Open(open::<A>),
            options,
        ) {
            world.entity_mut(tool).observe(observer);
        }
        self
    }
}

/// Spawns the entity of the tool `name`, whose calls run as `serves` says,
/// and returns it; `None`, with a warning, when the name is invalid or
/// taken.
fn register_tool(
    world: &mut World,
    name: &str,
    description: String,
    parameters: serde_json::Value,
    serves: Serves,
    options: ToolOptions<'_>,
) -> Option<Entity> {
    let tool_name = match ToolName::new(name) {
        Ok(tool_name) => tool_name,
        Err(error) => {
            warn!("tool not registered: {error}");
            return None;
        }
    };
    if world
        .query::<&ToolDef>()
        .iter(world)
        .any(|def| def.0.name == tool_name)
    {
        warn!("tool not registered: `{name}` exists; insert `Disabled` on it to replace it");
        return None;
    }
    let definition = ToolDefinition::new(tool_name, description, parameters);
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
        options.footprint,
        serves,
    ));
    Some(entity.id())
}

/// Runs `run` through `handler` on the one dispatch path, recorded for the
/// agent `scope` with the model call that asked for it (unknown for a call
/// a restart runs again) as its parent. A failure or a panic becomes an
/// error result for the model.
pub(crate) fn run_tool_call(
    effects: &Effects,
    scope: &str,
    run: &ToolCallRun,
    handler: ErasedHandler,
) -> impl ConditionalSendFuture<Output = ToolResult> + 'static {
    let (name, parent) = (run.call.function.name.as_str(), run.parent);
    let span = info_span!("tool_call", agent = scope, tool = name, parent = ?parent);
    let (name, args) = (name.to_owned(), run.call.function.raw_arguments());
    let (id, reply) = effects.dispatch(scope, parent, handler, EffectKind::ToolCall { name, args });
    let outcome = effects.caught(id, async { reply.await.into_outcome().await });
    async move {
        match outcome.await {
            Ok(Outcome::ToolResult { result }) => result,
            Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
                "the tool answered with a {} outcome",
                other.family()
            ))),
            Err(report) => ToolResult::failed(report.into()),
        }
    }
    .instrument(span)
}
