//! Typed runs: an [`AgentRunner`] whose accepted output is deserialized into
//! `T`, with an optional retry budget.
//!
//! [`Agent::prompt_typed`] builds one in *native* mode: the schema for `T` is
//! sent to the provider as the structured-output schema and the model's final
//! text is parsed as `T`. [`Extractor`](crate::extractor::Extractor) builds
//! one in *output-tool* mode: the model must call a synthetic `submit` tool
//! whose arguments are the value. Both are the same run type; only how the
//! value is recovered from the [`PromptResponse`] differs.

use std::{future::IntoFuture, marker::PhantomData};

use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};

use super::{Agent, hook::AgentHook, run::OutputMode, runner::AgentRunner};
use crate::{
    completion::{Message, StructuredOutputError, Usage},
    run::response::{CompletionCall, PromptResponse},
    tool::ToolContext,
};

/// A typed run's response: the deserialized value plus the run's usage and
/// completion calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedPromptResponse<T> {
    pub output: T,
    /// Usage accumulated across every attempt, including attempts that
    /// received a billed response but failed to produce a parseable value.
    pub usage: Usage,
    /// Successfully completed completion requests made by the accepted attempt.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
}

impl<T> TypedPromptResponse<T> {
    pub fn new(output: T, usage: Usage) -> Self {
        Self {
            output,
            usage,
            completion_calls: Vec::new(),
        }
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Returns successfully completed completion requests made by this agent run.
    ///
    /// Zero-valued entry usage means the provider reported no usage metrics
    /// for that request.
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
pub(crate) enum TypedOutput {
    /// Parse the model's final text; tolerate prose or fences around the JSON.
    Native,
    /// The value is the arguments of the run's output tool call; the model not
    /// calling it is an empty response.
    OutputTool,
}

/// A run that deserializes its accepted output as `T`.
///
/// Configure it with the same setters as [`AgentRunner`], then `.await` it for
/// a [`TypedPromptResponse<T>`]. With a [`retries`](Self::retries) budget, a
/// failed attempt (run error, empty output, unparseable output) is retried
/// from scratch; usage accumulates across attempts.
#[must_use = "a typed run does nothing until awaited"]
pub struct TypedRun<T> {
    runner: AgentRunner,
    retries: u64,
    output: TypedOutput,
    _t: PhantomData<T>,
}

/// The setters that forward verbatim to the inner [`AgentRunner`].
macro_rules! forward_runner_setters {
    () => {
        /// Attach a per-call [`ToolContext`] for this run.
        ///
        /// Every tool the agent executes during this run can read the
        /// caller-provided values (auth tokens, session IDs, conversation state, …)
        /// through the tool's [`ToolContext`](crate::tool::ToolContext),
        /// without the model ever seeing them.
        pub fn tool_context(mut self, context: ToolContext) -> Self {
            self.runner = self.runner.tool_context(context);
            self
        }

        /// Add chat history to the run.
        pub fn history<H, Item>(mut self, history: H) -> Self
        where
            H: IntoIterator<Item = Item>,
            Item: Into<Message>,
        {
            self.runner = self.runner.history(history);
            self
        }

        /// Override the agent preamble for this run.
        pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
            self.runner = self.runner.preamble(preamble);
            self
        }

        /// Remove the agent's configured preamble for this run.
        pub fn without_preamble(mut self) -> Self {
            self.runner = self.runner.without_preamble();
            self
        }

        /// Append one static context document for this run.
        pub fn document(mut self, document: crate::completion::Document) -> Self {
            self.runner = self.runner.document(document);
            self
        }

        /// Append static context documents for this run.
        pub fn documents(
            mut self,
            documents: impl IntoIterator<Item = crate::completion::Document>,
        ) -> Self {
            self.runner = self.runner.documents(documents);
            self
        }

        /// Override the model temperature for this run.
        pub fn temperature(mut self, temperature: f64) -> Self {
            self.runner = self.runner.temperature(temperature);
            self
        }

        /// Remove the agent's configured temperature for this run.
        pub fn without_temperature(mut self) -> Self {
            self.runner = self.runner.without_temperature();
            self
        }

        /// Override the maximum completion token count for this run.
        pub fn max_tokens(mut self, max_tokens: u64) -> Self {
            self.runner = self.runner.max_tokens(max_tokens);
            self
        }

        /// Remove the agent's configured maximum token count for this run.
        pub fn without_max_tokens(mut self) -> Self {
            self.runner = self.runner.without_max_tokens();
            self
        }

        /// Shallow-merge object fields into the provider-specific parameters
        /// for this run. Later fields win.
        pub fn merge_additional_params(
            mut self,
            params: serde_json::Map<String, serde_json::Value>,
        ) -> Self {
            self.runner = self.runner.merge_additional_params(params);
            self
        }

        /// Replace all provider-specific parameters for this run.
        pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
            self.runner = self.runner.replace_additional_params(params);
            self
        }

        /// Remove the agent's configured provider-specific parameters for this run.
        pub fn without_additional_params(mut self) -> Self {
            self.runner = self.runner.without_additional_params();
            self
        }

        /// Override the tool-choice policy for this run.
        pub fn tool_choice(mut self, tool_choice: rig_core::message::ToolChoice) -> Self {
            self.runner = self.runner.tool_choice(tool_choice);
            self
        }

        /// Remove the agent's configured tool-choice policy for this run.
        pub fn without_tool_choice(mut self) -> Self {
            self.runner = self.runner.without_tool_choice();
            self
        }

        /// Opt in or out of recording sensitive request, response, and tool
        /// content on GenAI telemetry spans for this run.
        ///
        /// Defaults to the agent's setting, which defaults to `false`. Enabling
        /// this can expose prompts, retrieved context, tool results, model
        /// responses, and other sensitive or high-cardinality data through
        /// OpenTelemetry span attributes. Structural metadata and token usage
        /// remain available when disabled.
        pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
            self.runner = self.runner.record_content_telemetry(enabled);
            self
        }

        /// Set the conversation id used to load and persist memory for this run.
        ///
        /// Overrides any default conversation id set on the agent. If memory is not
        /// configured on the agent, this has no effect.
        pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
            self.runner = self.runner.conversation(id);
            self
        }

        /// Disable conversation memory for this run.
        ///
        /// History will neither be loaded from nor saved to the agent's memory backend.
        pub fn without_memory(mut self) -> Self {
            self.runner = self.runner.without_memory();
            self
        }

        /// Set the retry budget for invalid tool-call recovery.
        ///
        /// Invalid tool-call retries also consume the total model-call budget.
        pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
            self.runner = self.runner.max_invalid_tool_call_retries(retries);
            self
        }

        /// Set the default model candidate for this run.
        ///
        /// This does not suppress registered model-selection hooks, which may
        /// replace this candidate before each model call (including retries).
        pub fn using_model(mut self, model: $crate::agent::ModelHandle) -> Self {
            self.runner = self.runner.using_model(model);
            self
        }

        /// Erase and set a typed default model for this run.
        pub fn using_model_value<M>(mut self, model: M) -> Self
        where
            M: $crate::completion::CompletionModel + 'static,
        {
            self.runner = self.runner.using_model_value(model);
            self
        }

        /// Set the total model-call budget, including the initial call and every
        /// retry or continuation. Zero emits no model calls; one permits only the
        /// initial call. Exceeding the budget returns a
        /// [`StructuredOutputError::PromptError`] wrapping a `MaxTurnsError`.
        pub fn max_turns(mut self, max_turns: usize) -> Self {
            self.runner = self.runner.max_turns(max_turns);
            self
        }

        /// Append a hook to this run's hook stack (on top of any the agent
        /// already carries). See the [`hook`](crate::agent::hook) module docs.
        pub fn add_hook<H>(mut self, hook: H) -> Self
        where
            H: AgentHook + 'static,
        {
            self.runner = self.runner.add_hook(hook);
            self
        }

        /// Execute up to `concurrency` of a turn's tool calls at once. See
        /// [`AgentRunner::tool_concurrency`] for ordering guarantees.
        pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
            self.runner = self.runner.tool_concurrency(concurrency);
            self
        }
    };
}

impl<T> TypedRun<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// A native-mode typed run: the schema for `T` is the run's structured
    /// output schema and the model's final text is parsed as `T`.
    pub(crate) fn native(agent: &Agent, prompt: impl Into<Message>) -> Self {
        let mut runner = AgentRunner::from_agent(agent, prompt);
        runner.config.output_schema = Some(schema_for!(T));
        // Typed prompts deserialize the model's final string, so they pin
        // `Native` structured output to keep the typed API's behavior unchanged
        // across all providers. Routing the typed path through `Tool` output
        // mode for tool-using agents on non-composing providers is a follow-up;
        // use the untyped `output_schema`/`output_mode` API for tool-composing
        // structured output today.
        runner.config.output_mode = OutputMode::Native;
        Self::from_runner(runner, TypedOutput::Native)
    }

    /// An output-tool typed run over an already configured runner: the value
    /// is the arguments of the run's output tool call. An invalid tool call
    /// the hooks leave unhandled is ignored rather than failing the run, so a
    /// model that fumbles the call once can still succeed within the budget.
    pub(crate) fn output_tool(runner: AgentRunner) -> Self {
        Self::from_runner(
            runner.ignore_unhandled_invalid_tool_calls(),
            TypedOutput::OutputTool,
        )
    }

    pub(crate) fn from_runner(runner: AgentRunner, output: TypedOutput) -> Self {
        Self {
            runner,
            retries: 0,
            output,
            _t: PhantomData,
        }
    }

    /// Retry a failed attempt up to `retries` more times. An attempt fails when
    /// the run errors, produces no output, or produces output that does not
    /// parse as `T`. Usage accumulates across attempts.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = retries;
        self
    }

    forward_runner_setters!();

    async fn send(self) -> Result<TypedPromptResponse<T>, StructuredOutputError> {
        let mut usage = Usage::new();
        let mut last_error = None;

        for attempt in 0..=self.retries {
            if self.retries > 0 {
                tracing::debug!(
                    "Attempting to extract structured output. Retries left: {}",
                    self.retries - attempt
                );
            }
            let (result, error_usage) = self.runner.clone().run_with_error_usage().await;
            let outcome = match result {
                Ok(response) => {
                    usage += response.usage;
                    recover_output(&response, self.output).map(|output| TypedPromptResponse {
                        output,
                        usage,
                        completion_calls: response.completion_calls,
                    })
                }
                Err(err) => {
                    usage += error_usage;
                    Err(StructuredOutputError::PromptError(Box::new(err)))
                }
            };
            match outcome {
                Ok(response) => return Ok(response),
                Err(err) => {
                    if attempt < self.retries {
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
    output: TypedOutput,
) -> Result<T, StructuredOutputError> {
    match output {
        TypedOutput::Native => {
            if response.output.is_empty() {
                return Err(StructuredOutputError::EmptyResponse);
            }
            Ok(deserialize_structured_output(&response.output)?)
        }
        TypedOutput::OutputTool => {
            let submissions = response.output_tool_calls();
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
            Ok(serde_json::from_str(&response.output)?)
        }
    }
}

/// Deserialize a typed structured response from the model's final text.
///
/// Tries a direct parse first (the common path — native and tool-call output is
/// already clean JSON), then falls back to the first balanced JSON value in the
/// text so prose or markdown code fences around the JSON don't break weaker
/// `Prompted`/best-effort output.
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
        Box::pin(self.send())
    }
}
