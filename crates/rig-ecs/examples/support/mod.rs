//! The mocks the examples run against, so no provider feature and no key:
//! a scripted model (each request pops the next answer), a streaming
//! scripted model (each request streams the next answer, a word at a
//! time), a few tools, and result-printing observers. Setup stays visible
//! in each example. A real `CompletionAdapter` and real tools register under the same
//! keys with the same `Serve` trait.

#![allow(
    dead_code,
    reason = "each example uses the part of the support it needs"
)]

use rig_core::serve::Dispatch;
use std::sync::Mutex;

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use rig_core::{
    completion::{CompletionResponse, ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    serve::{Reply, Serve},
    streaming::StreamFinal,
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::agent::{Failed, RunResult, Settled};

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
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| vec![AssistantContent::text("(the script is over)")])
    }
}

impl Serve for Scripted {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        model_descriptor()
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                let response = CompletionResponse::new(self.next(), Usage::new(), "scripted");
                Reply::Outcome(Ok(Outcome::Completion(response)))
            }
            EffectKind::Completion { stream: true, .. } => {
                // The next answer, streamed a word at a time.
                let parts = self.next();
                Reply::written(move |mut writer| async move {
                    for part in parts {
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
                })
            }
            other => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!("a model cannot serve {}", other.name()),
            ))),
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

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> rig_core::serve::Reply {
        let EffectKind::ToolCall { args, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                "not a call",
            )));
        };
        let args: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();
        let value = (self.run)(args);
        rig_core::serve::Reply::Outcome(Ok(Outcome::ToolResult {
            result: ToolResult::success(ToolOutput::json(value)),
        }))
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
            serde_json::json!(
                args.get("x")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
                    + args
                        .get("y")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0)
            )
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
            serde_json::json!(
                args.get("x")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
                    - args
                        .get("y")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0)
            )
        },
    }
}

pub fn send_email() -> Tool {
    Tool {
        key: Tool::key("send_email", 0),
        name: "send_email",
        description: "Send an email to a recipient.",
        parameters: serde_json::json!({"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}),
        run: |args| {
            serde_json::json!(format!(
                "sent to {}",
                args.get("to")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?")
            ))
        },
    }
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
