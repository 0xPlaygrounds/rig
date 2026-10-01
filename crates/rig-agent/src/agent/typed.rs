//! Typed run outputs with optional retries for execution or deserialization failures.
//! Native prompting parses final text; extractors require an output-tool call.
//!
//! ```no_run
//! # async fn example(agent: rig_agent::Agent) -> Result<(), rig_agent::completion::StructuredOutputError> {
//! let response = agent.prompt_typed::<Vec<String>>("List three colors.").retries(1).await?;
//! assert_eq!(response.output.len(), 3);
//! # Ok(())
//! # }
//! ```

use std::{future::IntoFuture, marker::PhantomData};

use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use tracing_futures::Instrument;

use super::{
    Agent,
    run::{OutputMode, spec::UnhandledInvalidToolCall},
    runner::AgentRunner,
};
use crate::{
    completion::{Message, StructuredOutputError, Usage},
    run::response::{CompletionCall, PromptResponse},
};

/// A typed run's response: the deserialized value plus the accepted attempt's
/// transcript, the run's usage, and completion calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedPromptResponse<T> {
    /// The parsed structured output.
    pub output: T,
    /// Usage accumulated across every attempt, including attempts that
    /// received a billed response but failed to produce a parseable value.
    pub usage: Usage,
    /// Successfully completed completion requests made by the accepted attempt.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. An entry whose counters are all `None` means the provider
    /// reported no usage metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
    /// The accepted attempt's transcript; see
    /// [`PromptResponse::messages`](crate::agent::PromptResponse::messages).
    /// Append it to caller-owned history to continue the conversation.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// How the accepted attempt's conversation-memory append settled; see
    /// [`PromptResponse::memory_append`](crate::agent::PromptResponse::memory_append).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_append: Option<crate::run::MemoryAppend>,
}

impl<T> TypedPromptResponse<T> {
    /// A response of `output` with the run's `usage`.
    pub fn new(output: T, usage: Usage) -> Self {
        Self {
            output,
            usage,
            completion_calls: Vec::new(),
            messages: Vec::new(),
            memory_append: None,
        }
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Returns successfully completed completion requests made by this agent run.
    ///
    /// An entry whose counters are all `None` means the provider reported no
    /// usage metrics for that request.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Number of completion requests this agent run made.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }
}

/// How a [`TypedRun`] recovers `T` from the accepted [`PromptResponse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputSource {
    /// Parse the model's final text; tolerate prose or fences around the JSON.
    Native,
    /// The value is the arguments of the run's output tool call; the model not
    /// calling it is an empty response.
    OutputTool,
}

/// The typed-output state of a [`TypedRun`]: how `T` is recovered from the
/// accepted response and how many failed attempts are retried.
#[derive(Debug)]
pub struct TypedOutput<T> {
    source: OutputSource,
    retries: u64,
    _t: PhantomData<fn() -> T>,
}

/// A run that deserializes its accepted output as `T`.
///
/// Configure it with the same setters as [`AgentRunner`], then `.await` it for
/// a [`TypedPromptResponse<T>`]. With a [`retries`](AgentRunner::retries) budget, a
/// failed attempt (run error, empty output, unparseable output) is retried
/// from scratch; usage accumulates across attempts.
pub type TypedRun<T> = AgentRunner<TypedOutput<T>>;

impl<T> TypedRun<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// A native-mode typed run: the schema for `T` is the run's structured
    /// output schema and the model's final text is parsed as `T`.
    pub(crate) fn native(agent: &Agent, prompt: impl Into<Message>) -> Self {
        let mut runner = AgentRunner::from_agent(agent, prompt);
        runner.config.output_schema = Some(schema_for!(T));
        // Native typed prompting parses final text rather than requiring an output call.
        runner.config.output_mode = OutputMode::Native;
        Self::from_runner(runner, OutputSource::Native)
    }

    /// An output-tool typed run over an already configured runner: the value
    /// is the arguments of the run's output tool call. Unhandled invalid tool
    /// calls are ignored rather than failing the run.
    pub(crate) fn output_tool(runner: AgentRunner) -> Self {
        Self::from_runner(
            runner.unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore),
            OutputSource::OutputTool,
        )
    }

    fn from_runner(runner: AgentRunner, source: OutputSource) -> Self {
        let output = TypedOutput {
            source,
            retries: 0,
            _t: PhantomData,
        };
        runner.replace_output(output).0
    }

    /// Retry a failed attempt up to `retries` more times. An attempt fails when
    /// the run errors, produces no output, or produces output that does not
    /// parse as `T`. Usage accumulates across attempts.
    pub fn retries(mut self, retries: u64) -> Self {
        self.output.retries = retries;
        self
    }

    async fn send(
        self,
        ambient: tracing::Span,
    ) -> Result<TypedPromptResponse<T>, StructuredOutputError> {
        let (
            runner,
            TypedOutput {
                source, retries, ..
            },
        ) = self.replace_output(());
        let mut usage = Usage::default();
        let mut last_error = None;

        for attempt in 0..=retries {
            if retries > 0 {
                tracing::debug!(
                    "Attempting to extract structured output. Retries left: {}",
                    retries - attempt
                );
            }
            let (result, error_usage) = runner.clone().run_with_error_usage(ambient.clone()).await;
            let outcome = match result {
                Ok(response) => {
                    usage += response.usage;
                    recover_output(&response, source).map(|output| TypedPromptResponse {
                        output,
                        usage,
                        completion_calls: response.completion_calls,
                        messages: response.messages,
                        memory_append: response.memory_append,
                    })
                }
                Err(err) => {
                    usage += error_usage;
                    Err(StructuredOutputError::Prompt(err))
                }
            };
            match outcome {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if attempt < retries {
                        tracing::warn!(
                            "Attempt {attempt} to extract structured output failed: {err:?}. Retrying..."
                        );
                    }
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or(StructuredOutputError::EmptyResponse))
    }
}

/// Recover `T` from an accepted response according to the run's output mode.
fn recover_output<T: DeserializeOwned>(
    response: &PromptResponse,
    source: OutputSource,
) -> Result<T, StructuredOutputError> {
    match source {
        OutputSource::Native => {
            if response.output.is_empty() {
                return Err(StructuredOutputError::EmptyResponse);
            }
            deserialize_structured_output(&response.output)
                .map_err(|error| deserialization_error(&response.output, error))
        }
        OutputSource::OutputTool => {
            let submissions = response.output_tool_calls();
            // A whole JSON answer is the run's output too: the protocol accepts
            // schema-valid text, and a model that rejects forced tool choice
            // answers in native structured output instead of calling the tool.
            if submissions == 0
                && let Ok(value) = serde_json::from_str::<T>(response.output.trim())
            {
                return Ok(value);
            }
            if submissions == 0 {
                tracing::warn!(
                    "The submit tool was not called. If this happens more than once, please ensure the model you are using is powerful enough to reliably call tools."
                );
                return Err(StructuredOutputError::EmptyResponse);
            }
            if submissions > 1 {
                tracing::warn!(
                    "Multiple submit calls detected, using the first one. Providers / agents should only ensure one submit call."
                );
            }
            serde_json::from_str(&response.output)
                .map_err(|error| deserialization_error(&response.output, error))
        }
    }
}

fn deserialization_error(output: &str, error: serde_json::Error) -> StructuredOutputError {
    StructuredOutputError::Deserialization {
        output: output.to_string(),
        error,
    }
}

/// Deserialize final text directly, then try one value starting at the first
/// object or array delimiter. Returns a parse error if neither attempt yields `T`.
pub(crate) fn deserialize_structured_output<T: DeserializeOwned>(
    text: &str,
) -> Result<T, serde_json::Error> {
    let trimmed = text.trim();
    match serde_json::from_str::<T>(trimmed) {
        Ok(value) => Ok(value),
        Err(direct_err) => {
            let Some(start) = trimmed.find(['{', '[']) else {
                return Err(direct_err);
            };
            serde_json::Deserializer::from_str(&trimmed[start..])
                .into_iter::<T>()
                .next()
                .unwrap_or(Err(direct_err))
        }
    }
}

impl<T> IntoFuture for TypedRun<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        // Captured in the synchronous part of the call, like `run()`: a
        // typed run belongs to the span it was started in, not to the task
        // that first polls it.
        let ambient = tracing::Span::current();
        let run_under = ambient.clone();
        Box::pin(self.send(run_under).instrument(ambient))
    }
}
