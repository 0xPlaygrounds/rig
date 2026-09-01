pub mod streaming;

use super::{Agent, hook::AgentHook, run::OutputMode, runner::AgentRunner};
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};

use crate::{
    completion::{Message, PromptError, Usage},
    tool::ToolContext,
};
use serde::{Deserialize, Serialize};
use std::{future::IntoFuture, marker::PhantomData};

/// The provider-neutral identity carrier, re-exported from rig-core so agent
/// callers name one type across core responses, stream terminals, completion
/// calls, and hook events.
pub use rig_core::completion::ResponseIdentity;

/// Generate the request-builder setters that forward verbatim to an inner
/// receiver — `AgentRunner` for the blocking builder, the wrapped
/// `PromptRequest` for the typed builder, and the `AgentRunner` for the
/// streaming builder. Only the setters whose signature *and* documentation are
/// identical across all three builders live here; `max_turns`, `add_hook`, and
/// `tool_concurrency`, whose docs are builder-specific, stay hand-written (the
/// blocking builders share `tool_concurrency` via [`forward_tool_concurrency`]).
/// `$recv` is the field name to delegate through (`runner` or `inner`).
macro_rules! forward_prompt_setters {
    ($recv:ident) => {
        /// Attach a per-call [`ToolContext`] for this request.
        ///
        /// Every tool the agent executes during this request can read the
        /// caller-provided values (auth tokens, session IDs, conversation state, …)
        /// through the tool's [`ToolContext`](crate::tool::ToolContext),
        /// without the model ever seeing them.
        pub fn tool_context(mut self, context: ToolContext) -> Self {
            self.$recv = self.$recv.tool_context(context);
            self
        }

        /// Add chat history to the prompt request.
        pub fn history<H, Item>(mut self, history: H) -> Self
        where
            H: IntoIterator<Item = Item>,
            Item: Into<Message>,
        {
            self.$recv = self.$recv.history(history);
            self
        }

        /// Override the agent preamble for this request.
        pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
            self.$recv = self.$recv.preamble(preamble);
            self
        }

        /// Remove the agent's configured preamble for this request.
        pub fn without_preamble(mut self) -> Self {
            self.$recv = self.$recv.without_preamble();
            self
        }

        /// Append one static context document for this request.
        pub fn document(mut self, document: crate::completion::Document) -> Self {
            self.$recv = self.$recv.document(document);
            self
        }

        /// Append static context documents for this request.
        pub fn documents(
            mut self,
            documents: impl IntoIterator<Item = crate::completion::Document>,
        ) -> Self {
            self.$recv = self.$recv.documents(documents);
            self
        }

        /// Override the model temperature for this request.
        pub fn temperature(mut self, temperature: f64) -> Self {
            self.$recv = self.$recv.temperature(temperature);
            self
        }

        /// Remove the agent's configured temperature for this request.
        pub fn without_temperature(mut self) -> Self {
            self.$recv = self.$recv.without_temperature();
            self
        }

        /// Override the maximum completion token count for this request.
        pub fn max_tokens(mut self, max_tokens: u64) -> Self {
            self.$recv = self.$recv.max_tokens(max_tokens);
            self
        }

        /// Remove the agent's configured maximum token count for this request.
        pub fn without_max_tokens(mut self) -> Self {
            self.$recv = self.$recv.without_max_tokens();
            self
        }

        /// Shallow-merge object fields into the provider-specific parameters
        /// for this request. Later fields win.
        pub fn merge_additional_params(
            mut self,
            params: serde_json::Map<String, serde_json::Value>,
        ) -> Self {
            self.$recv = self.$recv.merge_additional_params(params);
            self
        }

        /// Replace all provider-specific parameters for this request.
        pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
            self.$recv = self.$recv.replace_additional_params(params);
            self
        }

        /// Remove the agent's configured provider-specific parameters for this request.
        pub fn without_additional_params(mut self) -> Self {
            self.$recv = self.$recv.without_additional_params();
            self
        }

        /// Override the tool-choice policy for this request.
        pub fn tool_choice(mut self, tool_choice: rig_core::message::ToolChoice) -> Self {
            self.$recv = self.$recv.tool_choice(tool_choice);
            self
        }

        /// Remove the agent's configured tool-choice policy for this request.
        pub fn without_tool_choice(mut self) -> Self {
            self.$recv = self.$recv.without_tool_choice();
            self
        }

        /// Opt in or out of recording sensitive request, response, and tool
        /// content on GenAI telemetry spans for this request.
        ///
        /// Defaults to the agent's setting, which defaults to `false`. Enabling
        /// this can expose prompts, retrieved context, tool results, model
        /// responses, and other sensitive or high-cardinality data through
        /// OpenTelemetry span attributes, which can increase observability
        /// backend storage and query costs. Only enable it when content
        /// telemetry is acceptable for this request. Structural metadata and
        /// token usage remain available when disabled.
        pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
            self.$recv = self.$recv.record_content_telemetry(enabled);
            self
        }

        /// Set the conversation id used to load and persist memory for this request.
        ///
        /// Overrides any default conversation id set on the agent. If memory is not
        /// configured on the agent, this has no effect.
        pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
            self.$recv = self.$recv.conversation(id);
            self
        }

        /// Disable conversation memory for this request.
        ///
        /// History will neither be loaded from nor saved to the agent's memory backend.
        pub fn without_memory(mut self) -> Self {
            self.$recv = self.$recv.without_memory();
            self
        }

        /// Set the retry budget for invalid tool-call recovery.
        ///
        /// Invalid tool-call retries also consume the total model-call budget.
        pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
            self.$recv = self.$recv.max_invalid_tool_call_retries(retries);
            self
        }

        /// Set the default model candidate for this run.
        ///
        /// This does not suppress registered model-selection hooks, which may
        /// replace this candidate before each model call (including retries).
        pub fn using_model(mut self, model: $crate::agent::ModelHandle) -> Self {
            self.$recv = self.$recv.using_model(model);
            self
        }

        /// Erase and set a typed default model for this run.
        pub fn using_model_value<M>(mut self, model: M) -> Self
        where
            M: $crate::completion::CompletionModel + 'static,
        {
            self.$recv = self.$recv.using_model_value(model);
            self
        }
    };
}
pub(crate) use forward_prompt_setters;

/// Generate the `tool_concurrency` setter for the blocking builders, whose doc
/// is identical to each other but differs from the streaming builder's (the
/// streaming version documents how tool items are ordered in the emitted
/// stream). `$recv` is the field name to delegate through (`runner` or `inner`).
macro_rules! forward_tool_concurrency {
    ($recv:ident) => {
        /// Execute up to `concurrency` of a turn's tool calls at once.
        ///
        /// See [`AgentRunner::tool_concurrency`] for ordering guarantees: the tool
        /// batch commits and surfaces atomically, so persisted history and streamed
        /// tool results are both in tool-call order (results are surfaced only after
        /// the whole batch settles successfully).
        pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
            self.$recv = self.$recv.tool_concurrency(concurrency);
            self
        }
    };
}

pub trait PromptType {}
pub struct Standard;
pub struct Extended;

impl PromptType for Standard {}
impl PromptType for Extended {}

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// When the agent has no configured `default_max_turns`, the implicit budget is
/// one model call. Use [`.max_turns()`](Self::max_turns) to override the agent's
/// configured or implicit budget; a tool call followed by a model-authored final
/// answer generally requires at least two model calls.
pub struct PromptRequest<S>
where
    S: PromptType,
{
    /// The hook-aware driver this request configures and runs.
    pub(crate) runner: AgentRunner,
    /// Phantom data to track the type of the request (Standard vs Extended).
    state: PhantomData<S>,
}

impl PromptRequest<Standard> {
    /// Create a new PromptRequest from an agent, cloning the agent's data and
    /// default hook stack.
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        PromptRequest {
            runner: AgentRunner::from_agent(agent, prompt),
            state: PhantomData,
        }
    }
}

impl<S> PromptRequest<S>
where
    S: PromptType,
{
    /// Enable returning extended details for responses (includes aggregated token usage
    /// and the full message history accumulated during the agent loop).
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation and inspecting the full message exchange.
    pub fn extended_details(self) -> PromptRequest<Extended> {
        PromptRequest {
            runner: self.runner,
            state: PhantomData,
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns
    /// [`crate::completion::PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.runner = self.runner.max_turns(max_turns);
        self
    }

    /// Append a hook for this request (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while
    /// model-turn steering and observe-only/recovery events use
    /// first-non-`Continue`-wins). See the [`hook`](crate::agent::hook) module
    /// docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.runner = self.runner.add_hook(hook);
        self
    }

    forward_prompt_setters!(runner);
    forward_tool_concurrency!(runner);
}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl IntoFuture for PromptRequest<Standard> {
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl IntoFuture for PromptRequest<Extended> {
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl PromptRequest<Standard> {
    async fn send(self) -> Result<String, PromptError> {
        self.extended_details().send().await.map(|resp| resp.output)
    }
}

/// Details for one successfully completed completion request made by an agent run.
pub use crate::run::response::{CompletionCall, PromptResponse};
pub(crate) use crate::run::transcript::{
    assistant_text_from_choice, is_empty_assistant_turn, tool_result_output,
};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedPromptResponse<T> {
    pub output: T,
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
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

impl PromptRequest<Extended> {
    async fn send(self) -> Result<PromptResponse, PromptError> {
        self.runner.run().await
    }
}

// ================================================================
// TypedPromptRequest - for structured output with automatic deserialization
// ================================================================

use crate::completion::StructuredOutputError;
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;

/// A builder for creating typed prompt requests that return deserialized structured output.
///
/// This struct wraps a standard `PromptRequest` and adds:
/// - Automatic JSON schema generation from the target type `T`
/// - Automatic deserialization of the response into `T`
///
/// The type parameter `S` represents the state of the request (Standard or Extended).
/// Use `.extended_details()` to transition to Extended state for usage tracking.
///
/// # Example
/// ```rust,ignore
/// let forecast: WeatherForecast = agent
///     .prompt_typed("What's the weather in NYC?")
///     .max_turns(3)
///     .await?;
/// ```
pub struct TypedPromptRequest<T, S>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
{
    inner: PromptRequest<S>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Create a new TypedPromptRequest from an agent.
    ///
    /// This automatically sets the output schema based on the type parameter `T`.
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        let mut inner = PromptRequest::from_agent(agent, prompt);
        // Override the output schema with the schema for T
        inner.runner.config.output_schema = Some(schema_for!(T));
        // Typed prompts deserialize the model's final string, so they pin
        // `Native` structured output to keep the typed API's behavior unchanged
        // across all providers (#1928). Routing the typed path through `Tool`
        // output mode for tool-using agents on non-composing providers is a
        // follow-up; use the untyped `output_schema`/`output_mode` API for
        // tool-composing structured output today.
        inner.runner.config.output_mode = OutputMode::Native;
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S> TypedPromptRequest<T, S>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
{
    /// Enable returning extended details for responses (includes aggregated token usage).
    ///
    /// Note: This changes the type of the response from `.send()` to return a `TypedPromptResponse<T>` struct
    /// instead of just `T`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> TypedPromptRequest<T, Extended> {
        TypedPromptRequest {
            inner: self.inner.extended_details(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns a
    /// [`StructuredOutputError::PromptError`] wrapping a `MaxTurnsError`.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.inner = self.inner.max_turns(max_turns);
        self
    }

    /// Append a hook to this request's hook stack (on top of any the agent
    /// already carries).
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.inner = self.inner.add_hook(hook);
        self
    }

    forward_prompt_setters!(inner);
    forward_tool_concurrency!(inner);
}

/// Deserialize a typed structured response from the model's final text.
///
/// Tries a direct parse first (the common path — native and tool-call output is
/// already clean JSON), then falls back to the first balanced JSON value in the
/// text so prose or markdown code fences around the JSON don't break weaker
/// `Prompted`/best-effort output (#1928).
fn deserialize_structured_output<T: DeserializeOwned>(text: &str) -> Result<T, serde_json::Error> {
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

impl<T> TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Send the typed prompt request and deserialize the response.
    async fn send(self) -> Result<T, StructuredOutputError> {
        let response = self.inner.send().await.map_err(Box::new)?;

        if response.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = deserialize_structured_output(&response)?;
        Ok(parsed)
    }
}

impl<T> TypedPromptRequest<T, Extended>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
{
    /// Send the typed prompt request with extended details and deserialize the response.
    async fn send(self) -> Result<TypedPromptResponse<T>, StructuredOutputError> {
        let response = self.inner.send().await.map_err(Box::new)?;

        if response.output.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = deserialize_structured_output(&response.output)?;
        Ok(TypedPromptResponse::new(parsed, response.usage)
            .with_completion_calls(response.completion_calls))
    }
}

impl<T> IntoFuture for TypedPromptRequest<T, Standard>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
{
    type Output = Result<T, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<T> IntoFuture for TypedPromptRequest<T, Extended>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}
#[cfg(test)]
mod tests;
