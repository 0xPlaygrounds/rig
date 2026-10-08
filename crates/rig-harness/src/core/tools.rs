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

use std::path::{Component as PathPart, Path, PathBuf};

use bevy_app::App;
use bevy_ecs::observer::IntoEntityObserver;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_core::completion::ToolDefinition;
use rig_core::effect::{
    EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, Outcome, family, tool_key,
};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{ToolCall, ToolName, ToolResult, ToolResultContent};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::tool::{Tool, ToolErrorKind, ToolExecutionError};

use super::effects::{Effects, OpenEffect};
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
        let name = self.0.name.as_str();
        HandlerDescriptor {
            key: tool_key(name),
            family: FamilyDescriptor::Tool {
                name: name.to_owned(),
                description: self.0.description.clone(),
                parameters: self.0.parameters.clone(),
                embedding: None,
            },
            layers: Vec::new(),
        }
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
/// until a [`ToolOutput`] is inserted on it, and despawning it, as Esc
/// does, cancels it.
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
pub struct OpenCall(pub OpenEffect);

/// What a tool's calls touch, on the tool's entity. It decides which calls
/// of one reply run at once: calls that only read run side by side, and a
/// call waits for every earlier call of the reply that touches what it
/// touches, so two edits of one file still apply in order.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Footprint {
    /// Anything: a call waits for every earlier call of its reply and holds
    /// back every later one. The default, right for `shell` and for any
    /// tool that does not say.
    #[default]
    Exclusive,
    /// Reads the file or directory named by the string argument `arg`, the
    /// working directory when the argument is absent, and changes nothing.
    Reads {
        /// The argument holding the path.
        arg: &'static str,
    },
    /// Changes the file named by the string argument `arg`, and nothing
    /// else.
    Writes {
        /// The argument holding the path.
        arg: &'static str,
    },
    /// Nothing the reply's other calls touch, as far as ordering goes: a
    /// call waits only for earlier [`Exclusive`](Self::Exclusive) calls.
    /// Right for a tool that hands work to another agent.
    Independent,
}

impl Footprint {
    /// What a call with `args` touches.
    pub(crate) fn of(self, args: &serde_json::Map<String, serde_json::Value>) -> Touch {
        let path = |arg: &str| args.get(arg).and_then(serde_json::Value::as_str);
        match self {
            Self::Exclusive => Touch::All,
            Self::Independent => Touch::Nothing,
            Self::Reads { arg } => Touch::Read(lexical(path(arg).unwrap_or("."))),
            Self::Writes { arg } => match path(arg) {
                Some(path) => Touch::Write(lexical(path)),
                // The call fails on its arguments; it still waits its turn.
                None => Touch::All,
            },
        }
    }
}

/// What one tool call touches, from its tool's [`Footprint`] and its
/// arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Touch {
    /// Anything.
    All,
    /// Reads this path, or anything under it.
    Read(PathBuf),
    /// Changes this path.
    Write(PathBuf),
    /// Nothing another call waits for, unless that call touches anything.
    Nothing,
}

impl Touch {
    /// Whether a call touching `self` must wait for an earlier call
    /// touching `earlier`: either may touch anything, or one changes a
    /// path the other reads or changes. Paths are compared as written,
    /// made absolute without following links, so two names for one file
    /// through a symlink are not caught.
    pub(crate) fn waits_for(&self, earlier: &Touch) -> bool {
        let overlap = |a: &Path, b: &Path| a.starts_with(b) || b.starts_with(a);
        match (self, earlier) {
            (Touch::All, _) | (_, Touch::All) => true,
            (Touch::Nothing, _) | (_, Touch::Nothing) | (Touch::Read(_), Touch::Read(_)) => false,
            (Touch::Read(a) | Touch::Write(a), Touch::Read(b) | Touch::Write(b)) => overlap(a, b),
        }
    }
}

/// `path` made absolute against the working directory, with `.` and `..`
/// folded away without touching the file system.
fn lexical(path: &str) -> PathBuf {
    let path = std::path::absolute(path).unwrap_or_else(|_| PathBuf::from(path));
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            PathPart::CurDir => {}
            PathPart::ParentDir => {
                out.pop();
            }
            part => out.push(part),
        }
    }
    out
}

/// How a tool is registered with [`AppToolsExt::add_tool_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ToolOptions<'a> {
    /// Lines of the system prompt of every agent the tool is offered to,
    /// on how to use it, such as "Use `read` to look at files, not `cat` in
    /// `shell`". The tool's description says what it does; its rules say
    /// when to pick it.
    pub rules: &'a [&'a str],
    /// What its calls touch; the default runs each call on its own.
    pub footprint: Footprint,
    /// For an open tool: a call left without an output by a restart is
    /// started again, so the tool's observer gets [`ToolCalled`] for it once
    /// more and can carry it on, instead of the core answering it as
    /// interrupted. The observer sees the same call id and decides from its
    /// own saved state what is left to do.
    pub resumable: bool,
}

/// On an open tool's entity: its calls are started again after a restart
/// (see [`ToolOptions::resumable`]).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Resumable;

/// Registers tools on an [`App`].
pub trait AppToolsExt {
    /// Make `tool` available to every agent whose
    /// [`ToolAccess`](crate::core::agent::ToolAccess) allows its name. A
    /// name already registered is refused with a warning.
    ///
    /// Tool futures run on Bevy's async compute pool, a few threads that
    /// every agent's tool calls share. A tool that blocks, such as one
    /// using `std::fs`, `std::process::Command` or a long computation,
    /// wraps that work in [`blocking`](crate::core::blocking::blocking),
    /// which is in the prelude, so it runs on a thread of its own:
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
    ///     footprint: Footprint::Reads { arg: "path" },
    ///     ..ToolOptions::default()
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
    match handler {
        Some(handler) => {
            entity.insert(ToolHandler(handler));
        }
        None if options.resumable => {
            entity.insert(Resumable);
        }
        None => {}
    }
    Some(entity.id())
}

/// Why `call`'s arguments do not fit the tool's strict `parameters`: they
/// are not a JSON object, or name an argument the schema does not declare.
/// The model gets this as the call's error, with what to do instead.
pub(crate) fn refusal(parameters: &serde_json::Value, call: &ToolCall) -> Option<String> {
    let name = call.function.name.as_str();
    if let Some(invalid) = &call.function.invalid_arguments {
        return Some(format!(
            "the arguments of `{name}` are not a JSON object: {invalid}. Call it again with a \
             JSON object."
        ));
    }
    let declared = parameters
        .get("properties")
        .and_then(serde_json::Value::as_object);
    let unknown: Vec<&str> = call
        .function
        .arguments
        .keys()
        .filter(|arg| !declared.is_some_and(|declared| declared.contains_key(arg.as_str())))
        .map(String::as_str)
        .collect();
    if unknown.is_empty() {
        return None;
    }
    let known: Vec<String> = declared
        .map(|declared| declared.keys().map(|arg| format!("`{arg}`")).collect())
        .unwrap_or_default();
    let known = if known.is_empty() {
        "none".to_owned()
    } else {
        known.join(", ")
    };
    Some(format!(
        "`{name}` has no argument {}. Its arguments are: {known}. Call it again with only those.",
        unknown
            .iter()
            .map(|arg| format!("`{arg}`"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
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
        HandlerDescriptor {
            key: tool_key(&self.name),
            family: FamilyDescriptor::Tool {
                name: self.name.clone(),
                description: "A tool call that could not run.".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
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
pub(crate) fn failed(call: &ToolCall, why: String) -> ToolResult {
    call.error_result(vec![ToolResultContent::text(why)])
}
