//! The tool registry. A tool is an entity holding its definition, its
//! effect handler, its [`ToolRules`] and its [`Footprint`]; plugins add
//! tools with [`AppToolsExt::add_tool`].

use std::path::{Component as PathPart, Path, PathBuf};

use bevy_app::App;
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
use rig_core::tool::{Tool, ToolErrorKind};

use super::effects::Effects;
use super::prompt::ToolRules;

/// What the model is told about a tool.
#[derive(Component, Clone)]
#[require(ToolRules, Footprint)]
pub struct ToolDef(pub ToolDefinition);

/// The effect handler that runs a tool.
#[derive(Component, Clone)]
pub struct ToolHandler(pub ErasedHandler);

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
    /// Right for `task`, whose subagents run side by side.
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
    /// });
    /// ```
    fn add_tool_with<T: Tool + 'static>(&mut self, tool: T, options: ToolOptions<'_>) -> &mut Self;
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
            handler,
            options,
        );
        self
    }
}

/// Spawns the entity of the tool `name`, served by `handler`, and returns
/// it; `None`, with a warning, when the name is invalid or taken.
pub(crate) fn register_tool(
    world: &mut World,
    name: &str,
    description: String,
    parameters: serde_json::Value,
    handler: ErasedHandler,
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
    let definition = ToolDefinition::new(tool_name, description, parameters);
    let entity = world.spawn((
        Name::new(format!("tool:{name}")),
        ToolDef(definition),
        ToolHandler(handler),
        ToolRules(
            options
                .rules
                .iter()
                .map(|rule| (*rule).to_owned())
                .collect(),
        ),
        options.footprint,
    ));
    Some(entity.id())
}

/// Answers a call to a tool that is not registered, or that the agent may
/// not use, with an error, so that call is recorded like any other.
struct Unavailable(String);

impl Serve for Unavailable {
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: tool_key(&self.0),
            family: FamilyDescriptor::Tool {
                name: self.0.clone(),
                description: "A tool the model called that is not available.".to_owned(),
                parameters: serde_json::json!({"type": "object"}),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        Reply::Outcome(Err(ErrorReport::new(
            ErrorKind::Tool(ToolErrorKind::NotFound),
            format!("no tool named `{}` is available", self.0),
        )))
    }
}

/// Run `call` through `handler` on the one dispatch path, recorded with
/// `parent` (the model call that asked for it) as its parent; with no
/// handler, the call is answered as unavailable on that same path. A
/// missing tool, bad arguments, a failure or a panic all become an error
/// result for the model. Returns the call's effect id and its work.
pub(crate) fn run_tool_call(
    effects: &Effects,
    scope: &str,
    parent: EffectId,
    handler: Option<ErasedHandler>,
    call: ToolCall,
) -> (EffectId, impl Future<Output = ToolResult> + Send + 'static) {
    let name = call.function.name.as_str().to_owned();
    let handler = handler.unwrap_or_else(|| ErasedHandler::new(Unavailable(name.clone())));
    let args =
        call.function.invalid_arguments.clone().unwrap_or_else(|| {
            serde_json::Value::Object(call.function.arguments.clone()).to_string()
        });
    let (id, reply) = effects.dispatch(
        scope,
        Some(parent),
        handler,
        EffectKind::ToolCall { name, args },
    );
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
