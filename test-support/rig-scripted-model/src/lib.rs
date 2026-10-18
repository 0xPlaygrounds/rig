#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]
//! A model that answers from a script, for end-to-end tests of real rig
//! processes: `rigd`, its workers, the CLI, the TUI, any client.
//!
//! With [`ScriptedModelPlugin`] in the agent's plugins and
//! `RIG_SCRIPTED_MODEL` naming a JSON file, every agent without a model
//! gets [`MODEL`] (`scripted/model`), whose turns come from the file:
//!
//! ```json
//! [
//!   {"reply": "the first answer"},
//!   {"when": "start a helper", "calls": [{"tool": "start_agent", "args": {"task": "wait"}}]},
//!   {"when": "start a helper", "reply": "started it"},
//!   {"when": "wait", "calls": [{"tool": "shell", "args": {"command": "sleep 600"}}]}
//! ]
//! ```
//!
//! A turn is `reply` (text), `calls` (tool calls, each `{tool, args}`), or
//! both, with an optional `reasoning`. Which turn answers a model call:
//! - The conversation is the first one whose `when` occurs in the user
//!   messages before the first answer (the task or prompt), else the turns
//!   with no `when`. So a parent and its children, which share the script,
//!   each follow their own turns.
//! - Within it, model call *n* (counted by the assistant messages already in
//!   the history) gets its *n*-th turn. That is fixed by the history alone,
//!   so a worker restarted from its saved session answers the same way.
//! - Past the last turn, the reply is [`EXHAUSTED`].
//!
//! With a script it also registers the tool [`PANIC_TOOL`], `scripted_panic`,
//! whose call panics inside a system of the worker's app, as a plugin's bug
//! would: a chaos test calls it to crash one session's worker. It asks for
//! approval like any tool that may change things; allow it up front with the
//! grant rule `{"rule": "tool", "name": "scripted_panic"}`.
//!
//! Without the variable the plugin does nothing.

use std::path::Path;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::catalog::{Catalog, ModelSpec, Transport as CatalogTransport};
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::Message;
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockFrame, MockScript, MockStreamEvent};
use rig_core::wire::Mode;
use rig_ecs::agent::Agent;
use rig_ecs::model::{ModelChoice, Models};
use rig_ecs::tools::{AppToolsExt, ToolCalled, ToolOptions};
use serde::Deserialize;

/// The variable naming the script file.
pub const ENV: &str = "RIG_SCRIPTED_MODEL";
/// The model it serves.
pub const MODEL: &str = "scripted/model";
/// The tool whose call panics.
pub const PANIC_TOOL: &str = "scripted_panic";
/// The reply past a conversation's last turn.
pub const EXHAUSTED: &str = "(the scripted model has no more turns for this conversation)";

/// One scripted model turn.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    /// Which conversation it belongs to: a text its first user message holds.
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub reply: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub calls: Vec<Call>,
}

/// A tool call a turn makes.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// Serves [`MODEL`] from the script `RIG_SCRIPTED_MODEL` names (or one
/// given in code), and gives it to every agent without a model.
#[derive(Default)]
pub struct ScriptedModelPlugin {
    script: Option<Arc<Vec<Turn>>>,
}

impl ScriptedModelPlugin {
    /// The plugin answering from `script` instead of the file.
    pub fn with_script(script: Vec<Turn>) -> Self {
        Self {
            script: Some(Arc::new(script)),
        }
    }
}

impl Plugin for ScriptedModelPlugin {
    fn build(&self, app: &mut App) {
        let script = match (&self.script, std::env::var_os(ENV)) {
            (Some(script), _) => Arc::clone(script),
            (None, Some(path)) => Arc::new(read(Path::new(&path)).unwrap_or_else(|why| {
                bevy_log::error!("{why}");
                Vec::new()
            })),
            (None, None) => return,
        };
        app.add_systems(Startup, move |mut models: ResMut<Models>| {
            serve(&mut models, Arc::clone(&script));
        })
        .add_observer(choose)
        .add_open_tool(
            PANIC_TOOL,
            "Crashes this agent's process, for tests of crash recovery. Never call it unless told to.",
            ToolOptions::default(),
            crash,
        );
    }
}

/// The arguments of [`PANIC_TOOL`]: none.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NoArgs {}

#[expect(
    clippy::panic,
    reason = "the tool's whole point is a plugin bug that panics"
)]
fn crash(called: On<ToolCalled<NoArgs>>) {
    panic!("{PANIC_TOOL} was called (call {:?})", called.call);
}

/// The script at `path`.
pub fn read(path: &Path) -> Result<Vec<Turn>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("{ENV}: cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("{ENV}: {} is not a script: {e}", path.display()))
}

/// Adds [`MODEL`] to `models`, answered by `script`.
pub fn serve(models: &mut Models, script: Arc<Vec<Turn>>) {
    let Some((vendor, name)) = MODEL.split_once('/') else {
        return;
    };
    let Ok(id) = ProviderId::declare(vendor) else {
        return;
    };
    models
        .0
        .catalog_mut()
        .insert(ModelSpec::new(id, name).with_tools(true));
    models.0.provide(
        id,
        move |spec: &ModelSpec, _: &Catalog, _: &CatalogTransport| {
            if spec.id != name {
                return Err(ConnectError::Provider(format!(
                    "{} is not scripted",
                    spec.id
                )));
            }
            let model = Model::new(
                MockScript::new(vendor).with_id(name),
                Scripted(Arc::clone(&script)),
            );
            Ok(ErasedHandler::new(ModelAdapter::<Completion>::new(
                spec.reference(),
                model,
            )))
        },
    );
}

/// Gives a new agent the scripted model unless it has one.
fn choose(add: On<Add<Agent>>, without: Query<(), Without<ModelChoice>>, mut commands: Commands) {
    if without.contains(add.entity) {
        commands
            .entity(add.entity)
            .insert(ModelChoice(MODEL.to_owned()));
    }
}

/// The scripted transport: each send answers with the turn the request's
/// history selects.
#[derive(Clone)]
struct Scripted(Arc<Vec<Turn>>);

impl Transport<MockScript> for Scripted {
    fn send(&self, request: CompletionRequest, exchange: Exchange) -> Opening<MockFrame> {
        if exchange.mode != Mode::Streaming {
            return Opening::failed(ProviderError::Provider(
                "the scripted model only streams".into(),
            ));
        }
        let events = events(&pick(&self.0, &request), assistant_turns(&request));
        Opening::ready(Opened::new(futures::stream::iter(
            events.into_iter().map(|event| Ok(MockFrame::Event(event))),
        )))
    }
}

fn assistant_turns(request: &CompletionRequest) -> usize {
    request
        .chat_history
        .iter()
        .filter(|message| matches!(message.as_ref(), Message::Assistant(_)))
        .count()
}

/// The turn that answers `request`.
pub fn pick(script: &[Turn], request: &CompletionRequest) -> Turn {
    // The opening: every user message before the first answer.
    let first: String = request
        .chat_history
        .iter()
        .take_while(|message| !matches!(message.as_ref(), Message::Assistant(_)))
        .filter(|message| matches!(message.as_ref(), Message::User { .. }))
        .filter_map(|message| serde_json::to_string(message.as_ref()).ok())
        .collect();
    let conversation = script
        .iter()
        .filter_map(|turn| turn.when.as_deref())
        .find(|when| first.contains(when));
    script
        .iter()
        .filter(|turn| turn.when.as_deref() == conversation)
        .nth(assistant_turns(request))
        .cloned()
        .unwrap_or_else(|| Turn {
            reply: Some(EXHAUSTED.to_owned()),
            ..Turn::default()
        })
}

/// The stream of `turn`, model call `n` of its conversation.
fn events(turn: &Turn, n: usize) -> Vec<MockStreamEvent> {
    let mut events = Vec::new();
    if let Some(reasoning) = &turn.reasoning {
        events.push(MockStreamEvent::Reasoning {
            id: format!("r-{n}"),
            text: reasoning.clone(),
        });
    }
    if let Some(reply) = &turn.reply {
        events.push(MockStreamEvent::text(reply));
    }
    for (i, call) in turn.calls.iter().enumerate() {
        events.push(MockStreamEvent::tool_call(
            format!("call-{n}-{i}"),
            &call.tool,
            call.args.clone(),
        ));
    }
    events.push(MockStreamEvent::final_response_with_default_usage());
    events
}

#[cfg(test)]
mod tests;
