//! The tool registry and how tool calls end. A tool is an entity holding
//! its definition, its [`ToolRules`] and its [`Footprint`], and, for an
//! ordinary tool, its effect handler; plugins add tools with
//! [`AppToolsExt::add_tool`]. A tool call is an entity too, and inserting a
//! [`ToolOutput`] on it is the one way it ends: an ordinary tool's output
//! is inserted when its future resolves, and an open tool's call stays
//! open until any system or observer inserts one.
//!
//! ```ignore
//! app.add_open_tool("wait", "Waits for a signal.", schema, ToolOptions::default(),
//!     |called: On<ToolCalled>, runs: Query<&ToolCallRun>, mut commands: Commands| {
//!         // Keep `called.call`; insert its `ToolOutput` once the signal comes.
//!     });
//! ```

use bevy_app::App;
use bevy_ecs::observer::IntoEntityObserver;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_core::completion::ToolDefinition;
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor, Outcome, family};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, ToolName, ToolResult, ToolResultContent};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, OpenRecord, Reply, Serve};
use rig_core::tool::{Tool, ToolErrorKind, ToolExecutionError};

use super::effects::Effects;
use super::prompt::ToolRules;

/// What the model is told about a tool. Its parameters are strict: the
/// schema's top level gets `additionalProperties: false`, and a call naming
/// an argument it does not declare is refused before the tool runs.
#[derive(Component, Clone)]
#[require(ToolRules, Footprint)]
pub struct ToolDef(pub ToolDefinition);

impl ToolDef {
    /// How the effect log describes the tool.
    pub fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor::tool(
            self.0.name.as_str(),
            &self.0.description,
            self.0.parameters.clone(),
        )
    }
}

/// The effect handler that runs an ordinary tool. A tool without one is
/// open: each call triggers [`ToolCalled`] on the tool's entity and ends
/// when something inserts its [`ToolOutput`].
#[derive(Component, Clone)]
pub struct ToolHandler(pub ErasedHandler);

/// A tool call's result, inserted on the call entity: the one way a call
/// ends. Insert it once, with [`EntityCommands::insert_if_new`] when
/// another system may answer the same call; the first one counts. The turn
/// then starts the calls that waited for this one and, once every call of
/// the reply has one, sends the results to the model.
#[derive(Component, Clone, Debug)]
pub struct ToolOutput(pub ToolResult);

impl From<ToolResult> for ToolOutput {
    fn from(result: ToolResult) -> Self {
        Self(result)
    }
}

/// A call of an open tool started: triggered on the tool's entity, so the
/// observer [`AppToolsExt::add_open_tool`] registered handles it. The call
/// entity holds the [`ToolCallRun`](super::agent::ToolCallRun) and is a
/// [`CallOf`](super::agent::CallOf) the turn of `agent`; it stays open
/// until a [`ToolOutput`] is inserted on it, and despawning it, as an
/// [`Interrupt`](super::agent::Interrupt) does, cancels it.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct ToolCalled {
    /// The tool.
    pub entity: Entity,
    /// The call.
    pub call: Entity,
    /// The calling agent.
    pub agent: Entity,
}

/// The effect record of an open tool call, settled with the call's
/// [`ToolOutput`].
#[derive(Component)]
pub struct OpenCall(pub OpenRecord);

/// Whether a tool's calls may run beside the other calls of one reply, on
/// the tool's entity.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// [`ToolAccess`](crate::core::agent::ToolAccess) allows its name. A
    /// name already registered is refused with a warning.
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

    /// Make the open tool `name` available: each call whose arguments fit
    /// the JSON schema `parameters` triggers [`ToolCalled`] on the tool's
    /// entity, where `observer` watches, and ends when a [`ToolOutput`] is
    /// inserted on the call, at once or much later. Its calls are recorded
    /// in the effect log like any other, with that output as the outcome.
    fn add_open_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: serde_json::Value,
        options: ToolOptions<'_>,
        observer: impl IntoEntityObserver<M>,
    ) -> &mut Self;
}

impl AppToolsExt for App {
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self {
        let (description, parameters) = (tool.description(), tool.parameters());
        let handler = ErasedHandler::new(ToolAdapter::new(tool));
        register_tool(
            self.world_mut(),
            T::NAME,
            description,
            parameters,
            Some(handler),
            options,
        );
        self
    }

    fn add_open_tool<M>(
        &mut self,
        name: &str,
        description: &str,
        parameters: serde_json::Value,
        options: ToolOptions<'_>,
        observer: impl IntoEntityObserver<M>,
    ) -> &mut Self {
        let world = self.world_mut();
        if let Some(tool) = register_tool(
            world,
            name,
            description.to_owned(),
            parameters,
            None,
            options,
        ) {
            world.entity_mut(tool).observe(observer);
        }
        self
    }
}

/// Spawns the entity of the tool `name`, served by `handler` or open
/// without one, and returns it; `None`, with a warning, when the name is
/// invalid or taken. The parameters are made strict.
pub(crate) fn register_tool(
    world: &mut World,
    name: &str,
    description: String,
    mut parameters: serde_json::Value,
    handler: Option<ErasedHandler>,
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
        warn!("tool not registered: a tool named `{name}` already exists");
        return None;
    }
    if let Some(schema) = parameters.as_object_mut() {
        schema.insert(
            "additionalProperties".to_owned(),
            serde_json::Value::Bool(false),
        );
    }
    let definition = ToolDefinition::new(tool_name, description, parameters);
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
        options.footprint,
    ));
    if let Some(handler) = handler {
        entity.insert(ToolHandler(handler));
    }
    Some(entity.id())
}

/// The arguments of `call` as the effect log records them.
pub(crate) fn recorded_args(call: &ToolCall) -> String {
    call.function
        .invalid_arguments
        .clone()
        .unwrap_or_else(|| serde_json::Value::Object(call.function.arguments.clone()).to_string())
}

/// `result` as an effect outcome, for the record of an open call.
pub(crate) fn outcome_of(result: &ToolResult) -> Outcome {
    let output = rig_core::tool::ToolOutput::content(result.content.clone())
        .unwrap_or_else(|_| rig_core::tool::ToolOutput::text(""));
    let result = if result.is_error {
        rig_core::tool::ToolResult::failed(
            ToolExecutionError::other(output.render()).with_model_output(output),
        )
    } else {
        rig_core::tool::ToolResult::success(output)
    };
    Outcome::ToolResult { result }
}

/// Answers a call that cannot run, to a tool that is not available or
/// with arguments that do not fit, with an error saying why, so that call
/// is recorded like any other.
pub(crate) struct Refused {
    /// The tool called.
    pub(crate) name: String,
    /// What kind of error it is.
    pub(crate) kind: ToolErrorKind,
    /// Why the call cannot run.
    pub(crate) why: String,
}

impl Serve for Refused {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor::tool(
            &self.name,
            "A tool call that could not run.",
            serde_json::json!({"type": "object"}),
        )
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::Tool(self.kind),
            self.why.clone(),
        )))
    }
}

/// Run `call` through `handler` on the one dispatch path, recorded with
/// `parent` (the model call that asked for it, unknown for a call a restart
/// runs again) as its parent. Bad arguments, a failure or a panic all
/// become an error result for the model. Returns the call's effect id and
/// its work.
pub(crate) fn run_tool_call(
    effects: &Effects,
    scope: &str,
    parent: Option<EffectId>,
    handler: ErasedHandler,
    call: ToolCall,
) -> (EffectId, impl Future<Output = ToolResult> + Send + 'static) {
    let name = call.function.name.as_str().to_owned();
    let args = recorded_args(&call);
    let (id, reply) = effects.dispatch(scope, parent, handler, EffectKind::ToolCall { name, args });
    let outcome = effects.caught(id, async { reply.await.into_outcome().await });
    let work = async move {
        match outcome.await {
            Ok(Outcome::ToolResult { result }) => {
                let is_error = !result.is_success();
                let content = result.output().clone().into_content();
                if is_error {
                    call.error_result(content)
                } else {
                    call.result(content)
                }
            }
            Ok(other) => failed(
                &call,
                format!("the tool answered with a {} outcome", other.family()),
            ),
            Err(report) => failed(&call, report.to_string()),
        }
    };
    (id, work)
}

/// An error result for `call` saying `why`.
pub fn failed(call: &ToolCall, why: impl Into<String>) -> ToolResult {
    call.error_result(vec![ToolResultContent::text(why.into())])
}
