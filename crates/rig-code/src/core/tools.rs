//! The tool registry. A tool is an entity holding its definition, its
//! effect handler and its [`ToolRules`]; plugins add tools with
//! [`AppToolsExt::add_tool`].

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
pub struct ToolDef(pub ToolDefinition);

/// The effect handler that runs a tool.
#[derive(Component, Clone)]
pub struct ToolHandler(pub ErasedHandler);

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
        self.add_tool_with_rules(tool, &[])
    }

    /// [`add_tool`](Self::add_tool), with `rules` on how to use it: lines
    /// of the system prompt of every agent the tool is offered to, such as
    /// "Use `read` to look at files, not `cat` in `shell`". The tool's
    /// description says what it does; its rules say when to pick it.
    fn add_tool_with_rules<T: Tool + 'static>(&mut self, tool: T, rules: &[&str]) -> &mut Self;
}

impl AppToolsExt for App {
    fn add_tool_with_rules<T: Tool + 'static>(&mut self, tool: T, rules: &[&str]) -> &mut Self {
        let name = match ToolName::new(T::NAME) {
            Ok(name) => name,
            Err(error) => {
                warn!("tool not registered: {error}");
                return self;
            }
        };
        let world = self.world_mut();
        if world
            .query::<&ToolDef>()
            .iter(world)
            .any(|def| def.0.name == name)
        {
            warn!("tool not registered: a tool named `{name}` already exists");
            return self;
        }
        let definition = ToolDefinition::new(name, tool.description(), tool.parameters());
        world.spawn((
            Name::new(format!("tool:{}", T::NAME)),
            ToolDef(definition),
            ToolHandler(ErasedHandler::new(ToolAdapter::new(tool))),
            ToolRules(rules.iter().map(|rule| (*rule).to_owned()).collect()),
        ));
        self
    }
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
/// result for the model.
pub(crate) fn run_tool_call(
    effects: &Effects,
    scope: &str,
    parent: EffectId,
    handler: Option<ErasedHandler>,
    call: ToolCall,
) -> impl Future<Output = ToolResult> + Send + 'static {
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
    async move {
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
    }
}

/// An error result for `call` saying `why`.
pub(crate) fn failed(call: &ToolCall, why: String) -> ToolResult {
    call.error_result(vec![ToolResultContent::text(why)])
}
