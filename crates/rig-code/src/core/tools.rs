//! The tool registry. A tool is an entity holding its definition and its
//! effect handler; plugins add tools with [`AppToolsExt::add_tool`].

use std::panic::AssertUnwindSafe;

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use futures::FutureExt;
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
    /// Tool futures run on Bevy's async compute pool, which model calls
    /// share, so a tool that blocks (file or process work, long
    /// computation) wraps that work in [`blocking`](crate::core::blocking::blocking).
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self;
}

impl AppToolsExt for App {
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
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

/// Run `call` through `handler` on the one dispatch path; with no handler,
/// the call is answered as unavailable on that same path. A missing tool,
/// bad arguments, a failure or a panic all become an error result for the
/// model.
pub fn run_tool_call(
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
    let (_, reply) = effects.dispatch(
        scope,
        Some(parent),
        handler,
        EffectKind::ToolCall {
            name: name.clone(),
            args,
        },
    );
    async move {
        let outcome = AssertUnwindSafe(async { reply.await.into_outcome().await })
            .catch_unwind()
            .await;
        match outcome {
            Ok(Ok(Outcome::ToolResult { result })) => {
                let is_error = !result.is_success();
                let content = result.output().clone().into_content();
                if is_error {
                    call.error_result(content)
                } else {
                    call.result(content)
                }
            }
            Ok(Ok(other)) => failed(
                &call,
                format!("the tool answered with a {} outcome", other.family()),
            ),
            Ok(Err(report)) => failed(&call, report.to_string()),
            Err(_) => failed(&call, format!("the tool `{name}` panicked")),
        }
    }
}

/// An error result for `call` saying `why`.
pub fn failed(call: &ToolCall, why: String) -> ToolResult {
    call.error_result(vec![ToolResultContent::text(why)])
}
