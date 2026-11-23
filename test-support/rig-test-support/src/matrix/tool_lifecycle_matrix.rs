//! Shared tool lifecycle matrix: the cells, tools, agent run and recorded-wire
//! assertions every Chat Completions wire's matrix uses.
//!
//! The invoking module defines `run_cell`, which runs one [`Cell`] on its
//! cassette client into an [`Observation`], and `assert_cell`, which checks the
//! recorded scenario, usually through [`LifecycleWire::assert_cell`].

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rig_agent::AgentBuilder;
use rig_core::completion::{FinishReason, ToolDefinition};
use rig_core::message::Message;
use rig_core::tool::{Tool, ToolContext};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The system preamble every cell sends.
pub const PREAMBLE: &str =
    "Follow the user's tool-call instruction exactly. Do not answer in prose.";

/// How a cell reaches the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// One request, one reply.
    Blocking,
    /// One request, a server-sent event stream.
    Streaming,
}

/// The tool calls a cell asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One call with no arguments.
    Zero,
    /// One call with a nested object holding an array and Unicode.
    Nested,
    /// Two calls in one turn.
    Parallel,
}

/// The public surface a cell drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// A completion model call, observed on its normalized reply.
    Model,
    /// A one-turn agent, observed on its tool invocations.
    Agent,
}

/// One matrix cell.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// How the cell reaches the provider.
    pub transport: Transport,
    /// The provider's model name.
    pub model: &'static str,
    /// The tool calls the cell asks for.
    pub shape: Shape,
    /// The surface the cell drives.
    pub surface: Surface,
}

/// A [`Cell`] from its four coordinates.
pub fn cell(transport: Transport, model: &'static str, shape: Shape, surface: Surface) -> Cell {
    Cell {
        transport,
        model,
        shape,
        surface,
    }
}

/// What one cell's run produced.
#[derive(Debug, Default)]
pub struct Observation {
    /// The normalized finish reason of a model cell.
    pub finish_reason: Option<FinishReason>,
    /// The normalized call names of a model cell, in order.
    pub names: Vec<String>,
    /// The normalized call ids of a model cell, in order.
    pub ids: Vec<String>,
    /// The normalized call arguments of a model cell, in order.
    pub arguments: Vec<Value>,
    /// Every error the run surfaced.
    pub errors: Vec<String>,
    /// The tools an agent cell invoked, in order.
    pub invocations: Vec<String>,
}

/// The user prompt for `shape`.
pub fn prompt(shape: Shape) -> &'static str {
    match shape {
        Shape::Zero => "Call ping exactly once with no arguments.",
        Shape::Nested => {
            "Call record_payload exactly once with label café 東京, values [3, 5, 8], and meta.active true."
        }
        Shape::Parallel => {
            "Call alpha with value red and beta with value blue in the same turn, in that order."
        }
    }
}

/// The tool names `shape` calls, in order.
pub fn expected_names(shape: Shape) -> &'static [&'static str] {
    match shape {
        Shape::Zero => &["ping"],
        Shape::Nested => &["record_payload"],
        Shape::Parallel => &["alpha", "beta"],
    }
}

/// The definition of the matrix tool `name`.
///
/// # Panics
///
/// When `name` is not one of [`expected_names`].
pub fn tool_definition(name: &str) -> ToolDefinition {
    let parameters = match name {
        "ping" => json!({ "type": "object", "properties": {}, "additionalProperties": false }),
        "record_payload" => json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" },
                "values": { "type": "array", "items": { "type": "integer" } },
                "meta": { "type": "object", "properties": { "active": { "type": "boolean" } }, "required": ["active"] }
            },
            "required": ["label", "values", "meta"]
        }),
        "alpha" | "beta" => {
            json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] })
        }
        other => panic!("unknown matrix tool {other}"),
    };
    ToolDefinition {
        name: rig_core::message::ToolName::new(name).expect("tool name"),
        description: format!("Matrix tool {name}"),
        parameters,
        annotations: Default::default(),
    }
}

/// The arguments `shape` must send, in call order.
fn expected_arguments(shape: Shape) -> Vec<Value> {
    match shape {
        Shape::Zero => vec![json!({})],
        Shape::Nested => {
            vec![json!({ "label": "café 東京", "values": [3, 5, 8], "meta": { "active": true } })]
        }
        Shape::Parallel => vec![json!({ "value": "red" }), json!({ "value": "blue" })],
    }
}

type InvocationLog = Arc<Mutex<Vec<String>>>;

#[derive(Debug, thiserror::Error)]
#[error("matrix tool failed")]
struct MatrixToolError;

#[derive(Debug, Deserialize, Serialize)]
struct EmptyArgs {}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadArgs {
    label: String,
    values: Vec<i64>,
    meta: PayloadMeta,
}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadMeta {
    active: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ValueArgs {
    value: String,
}

macro_rules! impl_matrix_tool {
    ($ty:ident, $name:literal, $args:ty) => {
        #[derive(Clone)]
        struct $ty(InvocationLog);

        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = MatrixToolError;
            type Args = $args;
            type Output = String;

            fn description(&self) -> String {
                format!("Matrix tool {}", Self::NAME)
            }

            fn parameters(&self) -> Value {
                tool_definition(Self::NAME).parameters
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                self.0
                    .lock()
                    .expect("invocation log poisoned")
                    .push(Self::NAME.to_owned());
                Ok(Self::NAME.to_owned())
            }
        }
    };
}

impl_matrix_tool!(Ping, "ping", EmptyArgs);
impl_matrix_tool!(RecordPayload, "record_payload", PayloadArgs);
impl_matrix_tool!(Alpha, "alpha", ValueArgs);
impl_matrix_tool!(Beta, "beta", ValueArgs);

/// One wire's lifecycle matrix: where its fixtures live and how it forces a tool call.
pub struct LifecycleWire {
    /// The provider directory under the cassette corpus.
    pub provider: &'static str,
    /// The `tool_choice` value that forces a call on this wire.
    pub tool_choice: &'static str,
}

impl LifecycleWire {
    /// Run `cell` as a one-turn agent on `builder`, which holds the cell's
    /// model and any wire-specific options.
    pub async fn run_agent(&self, builder: AgentBuilder, cell: Cell) -> Observation {
        let invocations = InvocationLog::default();
        let builder = builder
            .preamble(PREAMBLE)
            .parallel_tool_calls(cell.shape == Shape::Parallel)
            .additional_params(json!({ "tool_choice": self.tool_choice }))
            .max_tokens(128)
            .default_max_turns(1);
        let agent = match cell.shape {
            Shape::Zero => builder.tool(Ping(Arc::clone(&invocations))).build(),
            Shape::Nested => builder
                .tool(RecordPayload(Arc::clone(&invocations)))
                .build(),
            Shape::Parallel => builder
                .tool(Alpha(Arc::clone(&invocations)))
                .tool(Beta(Arc::clone(&invocations)))
                .build(),
        };
        let mut errors = Vec::new();
        match cell.transport {
            Transport::Blocking => {
                if let Err(error) = agent.prompt(prompt(cell.shape)).await {
                    errors.push(error.to_string());
                }
            }
            Transport::Streaming => {
                let mut stream = agent
                    .prompt(prompt(cell.shape))
                    .history(Vec::<Message>::new())
                    .max_turns(1)
                    .stream();
                errors = crate::support::collect_stream_observation(&mut stream)
                    .await
                    .errors;
            }
        }
        Observation {
            errors,
            invocations: invocations.lock().expect("invocation log poisoned").clone(),
            ..Default::default()
        }
    }

    /// The recorded request of `scenario`.
    pub fn recorded_request(&self, scenario: &str) -> Value {
        crate::cassettes::recorded_json_request(self.provider, scenario)
    }

    /// The recorded blocking response of `scenario`.
    pub fn recorded_response(&self, scenario: &str) -> Value {
        crate::cassettes::recorded_json_response(self.provider, scenario)
    }

    /// The recorded stream frames of `scenario`.
    pub fn recorded_chunks(&self, scenario: &str) -> Vec<Value> {
        crate::cassettes::recorded_sse_json_frames(self.provider, scenario)
    }

    fn recorded_finish_and_calls(
        &self,
        scenario: &str,
        transport: Transport,
    ) -> (String, Vec<RecordedCall>) {
        match transport {
            Transport::Blocking => {
                let response = self.recorded_response(scenario);
                let finish = response["choices"][0]["finish_reason"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned();
                let calls = response["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .map(|calls| {
                        calls
                            .iter()
                            .map(|call| RecordedCall {
                                id: call["id"].as_str().unwrap_or_default().to_owned(),
                                name: call["function"]["name"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                                arguments: call["function"]["arguments"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (finish, calls)
            }
            Transport::Streaming => {
                let mut finish = String::new();
                let mut calls = Vec::<RecordedCall>::new();
                for chunk in self.recorded_chunks(scenario) {
                    for choice in chunk["choices"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|choice| choice["index"].as_u64() == Some(0))
                    {
                        if let Some(reason) = choice["finish_reason"].as_str() {
                            finish = reason.to_owned();
                        }
                        for call in choice["delta"]["tool_calls"]
                            .as_array()
                            .into_iter()
                            .flatten()
                        {
                            let index = call["index"].as_u64().unwrap_or(0) as usize;
                            while calls.len() <= index {
                                calls.push(RecordedCall::default());
                            }
                            if let Some(id) = call["id"].as_str().filter(|id| !id.is_empty()) {
                                calls[index].id = id.to_owned();
                            }
                            if let Some(name) = call["function"]["name"]
                                .as_str()
                                .filter(|name| !name.is_empty())
                            {
                                calls[index].name = name.to_owned();
                            }
                            if let Some(fragment) = call["function"]["arguments"].as_str() {
                                calls[index].arguments.push_str(fragment);
                            }
                        }
                    }
                }
                (finish, calls)
            }
        }
    }

    /// Assert the recorded request and reply of `scenario` match `cell`, and
    /// that `observation` saw exactly the calls `cell` asked for.
    pub fn assert_cell(&self, scenario: &str, cell: Cell, observation: Observation) {
        let request = self.recorded_request(scenario);
        assert_eq!(request["model"], cell.model, "{scenario}: model");
        assert_eq!(request["max_tokens"], 128, "{scenario}: cap");
        assert!(
            request.get("max_completion_tokens").is_none(),
            "{scenario}: these legacy Chat Completions models use max_tokens"
        );
        assert_eq!(
            request["tool_choice"], self.tool_choice,
            "{scenario}: forced tool"
        );
        assert_eq!(
            request["tools"].as_array().map(Vec::len),
            Some(expected_names(cell.shape).len()),
            "{scenario}: tool schema"
        );
        assert_eq!(
            request["parallel_tool_calls"],
            cell.shape == Shape::Parallel,
            "{scenario}: parallel policy"
        );
        assert_eq!(
            request["stream"].as_bool().unwrap_or(false),
            cell.transport == Transport::Streaming,
            "{scenario}: transport"
        );

        let (finish, wire_calls) = self.recorded_finish_and_calls(scenario, cell.transport);
        assert_eq!(finish, "tool_calls", "{scenario}: tool terminal");
        assert_eq!(
            wire_calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>(),
            expected_names(cell.shape),
            "{scenario}: wire call order"
        );
        assert_nonempty_distinct_ids(
            scenario,
            &wire_calls
                .iter()
                .map(|call| call.id.clone())
                .collect::<Vec<_>>(),
        );
        let wire_arguments = wire_calls
            .iter()
            .map(|call| {
                serde_json::from_str::<Value>(&call.arguments)
                    .with_context(|| format!("{scenario}: parse {} arguments", call.name))
                    .expect("recorded tool JSON")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            wire_arguments,
            expected_arguments(cell.shape),
            "{scenario}: exact wire arguments"
        );

        match cell.surface {
            Surface::Model => {
                assert!(
                    observation.errors.is_empty(),
                    "{scenario}: {:?}",
                    observation.errors
                );
                assert_eq!(
                    observation.finish_reason,
                    Some(FinishReason::ToolCalls),
                    "{scenario}"
                );
                assert_eq!(
                    observation
                        .names
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    expected_names(cell.shape),
                    "{scenario}: normalized call order"
                );
                assert_nonempty_distinct_ids(scenario, &observation.ids);
                assert_eq!(
                    observation.arguments,
                    expected_arguments(cell.shape),
                    "{scenario}: normalized arguments"
                );
            }
            Surface::Agent => {
                assert_eq!(
                    observation
                        .invocations
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    expected_names(cell.shape),
                    "{scenario}: exact-once invocation order"
                );
                assert!(
                    observation
                        .errors
                        .iter()
                        .all(|error| !error.contains("ProviderResponseError")),
                    "{scenario}: lifecycle reaches agent loop: {:?}",
                    observation.errors
                );
            }
        }
    }
}

#[derive(Debug, Default)]
struct RecordedCall {
    id: String,
    name: String,
    arguments: String,
}

fn assert_nonempty_distinct_ids(scenario: &str, ids: &[String]) {
    assert!(
        ids.iter().all(|id| !id.is_empty()),
        "{scenario}: every call has an id: {ids:?}"
    );
    assert_eq!(
        ids.iter().collect::<BTreeSet<_>>().len(),
        ids.len(),
        "{scenario}: call ids are distinct"
    );
}

/// Run the declared cell through the wire, then assert its recorded scenario.
#[macro_export]
macro_rules! tool_lifecycle_matrix_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            let cell = $cell;
            let observed = ::std::sync::Arc::new(::std::sync::Mutex::new(None));
            $wrapper($scenario, {
                let observed = ::std::sync::Arc::clone(&observed);
                move |client| async move {
                    let observation = run_cell(client, cell).await;
                    *observed.lock().expect("observation mutex poisoned") = Some(observation);
                    Ok::<(), anyhow::Error>(())
                }
            })
            .await?;
            let observation = observed
                .lock()
                .expect("observation mutex poisoned")
                .take()
                .expect("cell observation");
            assert_cell($scenario, cell, observation);
            Ok(())
        }
    };
}

pub use tool_lifecycle_matrix_case;
