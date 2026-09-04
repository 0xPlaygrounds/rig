//! The mocks the examples run against, so no provider feature and no key:
//! a scripted model (each request pops the next answer), a streaming
//! scripted model (each request streams the next answer, a word at a
//! time), a few tools, and an app with both plugins that exits when a run
//! ends. A real `CompletionAdapter` and real tools register under the same
//! keys with the same `Serve` trait.

#![allow(
    dead_code,
    reason = "each example uses the part of the support it needs"
)]

use std::sync::Mutex;

use bevy_app::{App, AppExit, ScheduleRunnerPlugin};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::{CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{OutcomeSink, Serve},
    streaming::StreamFinal,
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::{
    agent::{
        AdditionalParams, DefaultMaxTurns, Failed, InvalidCalls, MaxTokens, MaxTurns, Output,
        Owner, Preamble, RunResult, Settled, Temperature, ToolChoiceSpec, UsesModel,
    },
    bus::{BusPlugin, Handlers},
    systems::AgentPlugin,
};

pub const MODEL: &str = "demo/model:default";

/// A model answering each request with the next scripted turn.
pub struct Scripted {
    turns: Mutex<std::collections::VecDeque<Vec<AssistantContent>>>,
}

impl Scripted {
    pub fn new(turns: Vec<Vec<AssistantContent>>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
        }
    }

    fn next(&self) -> Vec<AssistantContent> {
        self.turns
            .lock()
            .expect("turns")
            .pop_front()
            .unwrap_or_else(|| vec![AssistantContent::text("(the script is over)")])
    }
}

impl Serve for Scripted {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        model_descriptor()
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                let response = CompletionResponse::new(self.next(), Usage::new(), "scripted");
                sink.resolve(Ok(Outcome::Completion(response))).await;
            }
            EffectKind::Completion { stream: true, .. } => {
                // The next answer, streamed a word at a time.
                let mut writer = sink.writer();
                for part in self.next() {
                    match part {
                        AssistantContent::Text(text) => {
                            for word in text.text.split_inclusive(' ') {
                                if writer.text(word).await.is_err() {
                                    return;
                                }
                            }
                        }
                        AssistantContent::ToolCall(call) => {
                            if writer
                                .tool_call(call.function.name, call.function.arguments)
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                        AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
                    }
                }
                let _ = writer
                    .finish(StreamFinal {
                        usage: Usage::new(),
                        finish_reason: None,
                        message_id: None,
                        response_id: None,
                        provider_request_id: None,
                        provider: "scripted".to_owned(),
                        model: None,
                        raw: serde_json::Value::Null,
                    })
                    .await;
            }
            other => {
                sink.resolve(Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!("a model cannot serve {}", other.name()),
                )))
                .await;
            }
        }
    }
}

fn model_descriptor() -> HandlerDescriptor {
    HandlerDescriptor {
        key: HandlerKey::from(MODEL),
        family: FamilyDescriptor::Completion {
            model: ModelRef::new("scripted"),
            capabilities: ProviderCapabilities::default(),
        },
        layers: Vec::new(),
    }
}

/// A tool call the script makes.
pub fn call(name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::tool_call(format!("call-{name}"), name, arguments)
}

/// A tool: a name, a description, a parameter schema, and a pure function
/// of its JSON arguments.
pub struct Tool {
    pub key: String,
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: serde_json::Value,
    pub run: fn(serde_json::Value) -> serde_json::Value,
}

impl Tool {
    pub fn key(name: &str, index: usize) -> String {
        format!("demo/tool:{name}#{index}")
    }
}

impl Serve for Tool {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(self.key.as_str()),
            family: FamilyDescriptor::Tool {
                name: self.name.to_owned(),
                description: self.description.to_owned(),
                parameters: self.parameters.clone(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let EffectKind::ToolCall { args, .. } = kind else {
            sink.resolve(Err(ErrorReport::new(ErrorKind::Internal, "not a call")))
                .await;
            return;
        };
        let args: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
        let value = (self.run)(args);
        sink.resolve(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::json(value)),
        }))
        .await;
    }
}

fn xy() -> serde_json::Value {
    serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
}

pub fn add() -> Tool {
    Tool {
        key: Tool::key("add", 0),
        name: "add",
        description: "Add x and y",
        parameters: xy(),
        run: |args| {
            serde_json::json!(args["x"].as_i64().unwrap_or(0) + args["y"].as_i64().unwrap_or(0))
        },
    }
}

pub fn subtract() -> Tool {
    Tool {
        key: Tool::key("subtract", 1),
        name: "subtract",
        description: "Subtract y from x",
        parameters: xy(),
        run: |args| {
            serde_json::json!(args["x"].as_i64().unwrap_or(0) - args["y"].as_i64().unwrap_or(0))
        },
    }
}

pub fn send_email() -> Tool {
    Tool {
        key: Tool::key("send_email", 0),
        name: "send_email",
        description: "Send an email to a recipient.",
        parameters: serde_json::json!({"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}),
        run: |args| serde_json::json!(format!("sent to {}", args["to"].as_str().unwrap_or("?"))),
    }
}

/// An app with both plugins; an example adds its own exit.
pub fn app() -> App {
    let mut app = App::new();
    app.add_plugins((
        ScheduleRunnerPlugin::default(),
        BusPlugin::default(),
        AgentPlugin::default(),
    ));
    app.add_observer(exit_when_failed);
    app
}

/// Register the model and the tools from a startup system: the model's
/// handler entity and the tools', in order.
pub fn register(
    handlers: &mut Handlers,
    model: Scripted,
    tools: Vec<Tool>,
) -> (Entity, Vec<Entity>) {
    let model = handlers.register(MODEL, model).expect("a fresh key");
    let tools = tools
        .into_iter()
        .map(|tool| {
            let key = tool.key.clone();
            handlers.register(key.as_str(), tool).expect("a fresh key")
        })
        .collect();
    (model, tools)
}

/// An agent over `model`: the preamble, the defaults, `max_turns` turns.
pub fn agent(commands: &mut Commands, model: Entity, preamble: &str, max_turns: usize) -> Entity {
    commands
        .spawn((
            Owner("demo".to_owned()),
            Preamble(Some(preamble.to_owned())),
            Temperature(None),
            MaxTokens(Some(1024)),
            AdditionalParams(None),
            ToolChoiceSpec(None),
            Output::default(),
            DefaultMaxTurns(Some(max_turns)),
            MaxTurns(max_turns),
            InvalidCalls::default(),
            UsesModel(model),
        ))
        .id()
}

/// An observer: the first run to settle prints its answer and the app exits.
pub fn print_the_answer_and_exit(
    settled: On<Add, Settled>,
    results: Query<&RunResult>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Ok(result) = results.get(settled.event().entity) {
        println!("{}", result.0);
    }
    exit.write(AppExit::Success);
}

/// An observer: a failed run is reported and the app exits with an error.
pub fn exit_when_failed(
    failed: On<Add, Failed>,
    failures: Query<&Failed>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Ok(Failed(failure)) = failures.get(failed.event().entity) {
        eprintln!("the run failed: {failure:?}");
    }
    exit.write(AppExit::error());
}
