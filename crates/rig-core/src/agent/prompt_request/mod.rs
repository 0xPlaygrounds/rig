pub mod streaming;

use super::{Agent, hook::AgentHook, run::OutputMode, runner::AgentRunner};
use crate::{
    OneOrMany,
    completion::{CompletionModel, Message, PromptError, Usage},
    message::{AssistantContent, ToolResultContent, UserContent},
    tool::ToolCallExtensions,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use serde::{Deserialize, Serialize};
use std::{future::IntoFuture, marker::PhantomData};

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
        /// Attach a per-call [`ToolCallExtensions`] for this request.
        ///
        /// Every tool the agent executes during this request can read the
        /// caller-provided values (auth tokens, session IDs, conversation state, …)
        /// via [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions),
        /// without the model ever seeing them.
        pub fn tool_extensions(mut self, extensions: ToolCallExtensions) -> Self {
            self.$recv = self.$recv.tool_extensions(extensions);
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

        /// Set the conversation id used to load and persist memory for this request.
        ///
        /// Overrides any default conversation id set on the agent. If memory is not
        /// configured on the agent, this has no effect.
        pub fn conversation(mut self, id: impl Into<String>) -> Self {
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
pub struct PromptRequest<S, M>
where
    S: PromptType,
    M: CompletionModel,
{
    /// The hook-aware driver this request configures and runs.
    pub(crate) runner: AgentRunner<M>,
    /// Phantom data to track the type of the request (Standard vs Extended).
    state: PhantomData<S>,
}

impl<M> PromptRequest<Standard, M>
where
    M: CompletionModel,
{
    /// Create a new PromptRequest from an agent, cloning the agent's data and
    /// default hook stack.
    pub fn from_agent(agent: &Agent<M>, prompt: impl Into<Message>) -> Self {
        PromptRequest {
            runner: AgentRunner::from_agent(agent, prompt),
            state: PhantomData,
        }
    }
}

impl<S, M> PromptRequest<S, M>
where
    S: PromptType,
    M: CompletionModel,
{
    /// Enable returning extended details for responses (includes aggregated token usage
    /// and the full message history accumulated during the agent loop).
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation and inspecting the full message exchange.
    pub fn extended_details(self) -> PromptRequest<Extended, M> {
        PromptRequest {
            runner: self.runner,
            state: PhantomData,
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns
    /// [`crate::completion::request::PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.runner = self.runner.max_turns(max_turns);
        self
    }

    /// Append a hook for this request (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (`CompletionCall` request patches accumulate and merge,
    /// `ToolCall`/`ToolResult` rewrites chain, and only observe-only/recovery
    /// events use first-non-`Continue`-wins). See the
    /// [`hook`](crate::agent::hook) module docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook<M> + 'static,
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
impl<M> IntoFuture for PromptRequest<Standard, M>
where
    M: CompletionModel + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M> IntoFuture for PromptRequest<Extended, M>
where
    M: CompletionModel + 'static,
{
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M> PromptRequest<Standard, M>
where
    M: CompletionModel,
{
    async fn send(self) -> Result<String, PromptError> {
        self.extended_details().send().await.map(|resp| resp.output)
    }
}

/// Details for one successfully completed completion request made by an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionCall {
    /// Zero-based index of the completion request within this agent run.
    pub call_index: usize,
    /// Token usage reported for this completion request.
    ///
    /// Zero-valued usage is [`Usage`]'s documented sentinel for missing
    /// provider usage metrics; rig does not distinguish "reported all zeros"
    /// from "unreported".
    #[serde(default, deserialize_with = "usage_null_as_default")]
    pub usage: Usage,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run.
    pub fn new(call_index: usize, usage: Usage) -> Self {
        Self { call_index, usage }
    }
}

/// Tolerate `null` usage from data serialized before rig dropped the
/// `Option<Usage>` encoding of missing provider usage metrics.
///
/// This tolerance requires a self-describing format such as JSON; data
/// serialized with non-self-describing formats (e.g. bincode) from before the
/// change cannot round-trip.
fn usage_null_as_default<'de, D>(deserializer: D) -> Result<Usage, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Usage>::deserialize(deserializer)?.unwrap_or_default())
}

/// The result of an agent run, returned by **both** the blocking
/// ([`PromptRequest`]) and streaming ([`StreamingPromptRequest`]) surfaces so a
/// call site reads identically whether it used `.prompt()` or `.stream_prompt()`.
///
/// On the streaming surface this is the payload of the terminal
/// [`MultiTurnStreamItem::FinalResponse`] item.
///
/// [`StreamingPromptRequest`]: crate::agent::StreamingPromptRequest
/// [`MultiTurnStreamItem::FinalResponse`]: crate::agent::MultiTurnStreamItem::FinalResponse
#[derive(Debug, Clone, Serialize, Deserialize)]
// Serialize *and* deserialize both go through `PromptResponseRepr` so the two
// directions agree on `content`'s wire shape (an `Option`). Routing only
// deserialize through the shadow would make serialize write a bare `OneOrMany`
// while deserialize expects an `Option`, breaking round-trips for positional /
// non-self-describing formats (e.g. bincode). The repr carries the field serde
// attributes, so the JSON shape is unchanged.
#[serde(from = "PromptResponseRepr", into = "PromptResponseRepr")]
#[non_exhaustive]
pub struct PromptResponse {
    /// Concatenated assistant text for the final turn.
    pub output: String,
    /// Aggregated token usage across the whole run.
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    pub completion_calls: Vec<CompletionCall>,
    /// Accumulated message history for the run (the run's persisted transcript),
    /// unless memory/history bookkeeping was disabled for the request.
    pub messages: Option<Vec<Message>>,
    /// Structured assistant content for the final turn.
    ///
    /// Where [`output`](Self::output) is the concatenated text, this preserves
    /// the individual content parts (text, reasoning, images, …).
    pub content: OneOrMany<AssistantContent>,
}

/// Serde shadow for [`PromptResponse`]. `content` is an `Option` here so runs
/// serialized before the field existed still deserialize: a missing `content`
/// reconstructs the structured final turn from `output` (a single text part),
/// keeping [`PromptResponse::output`] and [`PromptResponse::content`] consistent
/// for legacy data rather than defaulting to empty text. It carries the field
/// serde attributes for both directions, keeping the serialized shape identical
/// (`completion_calls` omitted when empty; `messages`/`content` always present).
#[derive(Serialize, Deserialize)]
struct PromptResponseRepr {
    output: String,
    usage: Usage,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    completion_calls: Vec<CompletionCall>,
    messages: Option<Vec<Message>>,
    #[serde(default)]
    content: Option<OneOrMany<AssistantContent>>,
}

impl From<PromptResponseRepr> for PromptResponse {
    fn from(repr: PromptResponseRepr) -> Self {
        let content = repr
            .content
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text(repr.output.clone())));
        Self {
            output: repr.output,
            usage: repr.usage,
            completion_calls: repr.completion_calls,
            messages: repr.messages,
            content,
        }
    }
}

impl From<PromptResponse> for PromptResponseRepr {
    fn from(response: PromptResponse) -> Self {
        Self {
            output: response.output,
            usage: response.usage,
            completion_calls: response.completion_calls,
            messages: response.messages,
            content: Some(response.content),
        }
    }
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(f)
    }
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        let output = output.into();
        Self {
            content: OneOrMany::one(AssistantContent::text(output.clone())),
            output,
            usage,
            completion_calls: Vec::new(),
            messages: None,
        }
    }

    /// An empty run result (empty output, zero usage, no history).
    pub fn empty() -> Self {
        Self::new(String::new(), Usage::new())
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Attach completion call details to this response.
    pub fn with_completion_calls(mut self, completion_calls: Vec<CompletionCall>) -> Self {
        self.completion_calls = completion_calls;
        self
    }

    /// Set the structured assistant content for the final turn.
    pub fn with_content(mut self, content: OneOrMany<AssistantContent>) -> Self {
        self.content = content;
        self
    }

    /// The concatenated assistant text for the final turn.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Aggregated token usage across the whole run.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The run's accumulated message history, if tracked.
    pub fn messages(&self) -> Option<&[Message]> {
        self.messages.as_deref()
    }

    /// The structured assistant content for the final turn.
    pub fn content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
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

pub(crate) const TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER: &str =
    "Tool not executed because another tool call in the same assistant turn was invalid.";

/// Combine input history with new messages for building completion requests.
pub(crate) fn build_history_for_request(
    chat_history: Option<&[Message]>,
    new_messages: &[Message],
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().chain(new_messages.iter()).cloned().collect()
}

/// Build the full history for error reporting (input + new messages).
pub(crate) fn build_full_history(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().cloned().chain(new_messages).collect()
}

/// Wrap already-shaped tool-result content for the model (see
/// [`tool_result_output`] / [`tool_result_message`]).
fn tool_result_with(
    id: String,
    call_id: Option<String>,
    content: OneOrMany<ToolResultContent>,
) -> UserContent {
    match call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(id, call_id, content),
        None => UserContent::tool_result(id, content),
    }
}

/// Shape a **real tool output** as a tool result. Routes through
/// [`ToolResultContent::from_tool_output`], which parses structured/multimodal
/// payloads (text, images, …). Use this only for actual tool-server output.
pub(crate) fn tool_result_output(
    id: String,
    call_id: Option<String>,
    output: String,
) -> UserContent {
    tool_result_with(id, call_id, ToolResultContent::from_tool_output(output))
}

/// Shape a **synthetic message** (a hook skip reason, recovery feedback, or a
/// "not executed" notice) as a tool result. Emitted **verbatim as text** and
/// never re-parsed as structured tool output, so a JSON-shaped message is not
/// silently reinterpreted as an image/multimodal result. Used identically by the
/// blocking and streaming drivers so synthetic results match across both.
pub(crate) fn tool_result_message(
    id: String,
    call_id: Option<String>,
    message: String,
) -> UserContent {
    tool_result_with(
        id,
        call_id,
        OneOrMany::one(ToolResultContent::text(message)),
    )
}

pub(crate) fn invalid_tool_retry_user_message(
    assistant_content: &OneOrMany<AssistantContent>,
    invalid_tool_call_id: &str,
    feedback: String,
) -> Option<Message> {
    let retry_results = assistant_content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) if tool_call.id == invalid_tool_call_id => {
                Some(tool_result_message(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    feedback.clone(),
                ))
            }
            AssistantContent::ToolCall(tool_call) => Some(tool_result_message(
                tool_call.id.clone(),
                tool_call.call_id.clone(),
                TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    Some(Message::User {
        content: OneOrMany::from_iter_optional(retry_results)?,
    })
}

pub(crate) fn is_empty_assistant_turn(choice: &OneOrMany<AssistantContent>) -> bool {
    choice.len() == 1
        && matches!(
            choice.first(),
            AssistantContent::Text(text) if text.text.is_empty() && text.additional_params.is_none()
        )
}

pub(crate) fn assistant_text_from_choice(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

impl<M> PromptRequest<Extended, M>
where
    M: CompletionModel,
{
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
pub struct TypedPromptRequest<T, S, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
    M: CompletionModel,
{
    inner: PromptRequest<S, M>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, M> TypedPromptRequest<T, Standard, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
{
    /// Create a new TypedPromptRequest from an agent.
    ///
    /// This automatically sets the output schema based on the type parameter `T`.
    pub fn from_agent(agent: &Agent<M>, prompt: impl Into<Message>) -> Self {
        let mut inner = PromptRequest::from_agent(agent, prompt);
        // Override the output schema with the schema for T
        inner.runner.output_schema = Some(schema_for!(T));
        // Typed prompts deserialize the model's final string, so they pin
        // `Native` structured output to keep the typed API's behavior unchanged
        // across all providers (#1928). Routing the typed path through `Tool`
        // output mode for tool-using agents on non-composing providers is a
        // follow-up; use the untyped `output_schema`/`output_mode` API for
        // tool-composing structured output today.
        inner.runner.output_mode = OutputMode::Native;
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S, M> TypedPromptRequest<T, S, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
    M: CompletionModel,
{
    /// Enable returning extended details for responses (includes aggregated token usage).
    ///
    /// Note: This changes the type of the response from `.send()` to return a `TypedPromptResponse<T>` struct
    /// instead of just `T`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> TypedPromptRequest<T, Extended, M> {
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
        H: AgentHook<M> + 'static,
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

impl<T, M> TypedPromptRequest<T, Standard, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
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

impl<T, M> TypedPromptRequest<T, Extended, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
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

impl<T, M> IntoFuture for TypedPromptRequest<T, Standard, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
    M: CompletionModel + 'static,
{
    type Output = Result<T, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<T, M> IntoFuture for TypedPromptRequest<T, Extended, M>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
    M: CompletionModel + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionCall, PromptResponse, PromptResponseRepr, TypedPromptResponse};
    use crate::{
        agent::{
            AgentBuilder,
            hook::{AgentHook, Flow, HookContext, InvalidToolCallContext, StepEvent},
        },
        completion::{
            AssistantContent, CompletionError, CompletionRequest, Message, Prompt, PromptError,
            StructuredOutputError, TypedPrompt, Usage,
        },
        message::{Text, ToolCall, ToolChoice, ToolFunction, UserContent},
        test_utils::{
            AppendFailingMemory, CountingMemory, FailingMemory, MockAddTool, MockCompletionModel,
            MockExtensionsProbeTool, MockOperationArgs, MockSubtractTool, MockToolError, MockTurn,
            SessionId,
        },
        tool::{Tool, ToolCallExtensions},
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    };

    #[derive(Serialize)]
    struct SerializeOnly {
        value: &'static str,
    }

    #[derive(Deserialize)]
    struct DeserializeOnly {
        value: String,
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct TypedAnswer {
        value: String,
    }

    #[test]
    fn deserialize_structured_output_tolerates_fences_and_prose() {
        // Clean JSON (native / output-tool path).
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>(r#"{"value":"x"}"#).unwrap(),
            TypedAnswer { value: "x".into() }
        );
        // Markdown-fenced JSON (weak Prompted-mode models).
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>("```json\n{\"value\":\"y\"}\n```")
                .unwrap(),
            TypedAnswer { value: "y".into() }
        );
        // Prose around the JSON object.
        assert_eq!(
            super::deserialize_structured_output::<TypedAnswer>(
                "Here you go: {\"value\":\"z\"} — hope that helps!"
            )
            .unwrap(),
            TypedAnswer { value: "z".into() }
        );
        // No JSON at all still errors.
        assert!(super::deserialize_structured_output::<TypedAnswer>("no json here").is_err());
    }

    #[derive(Clone)]
    struct PanicOnUnknownToolHook;

    impl AgentHook<MockCompletionModel> for PanicOnUnknownToolHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::CompletionResponse { .. } => {
                    panic!("unknown tool response should fail before response hooks run")
                }
                StepEvent::ToolCall { .. } => {
                    panic!("unknown tool call should fail before tool hooks run")
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct PanicOnToolCallHook;

    impl AgentHook<MockCompletionModel> for PanicOnToolCallHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::ToolCall { .. } => {
                    panic!("recovered invalid turn should not invoke normal tool hooks")
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiAndPanicOnToolCallHook;

    impl AgentHook<MockCompletionModel> for SkipDefaultApiAndPanicOnToolCallHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(context) => {
                    SkipDefaultApiHook
                        .on_event(_ctx, StepEvent::InvalidToolCall(context))
                        .await
                }
                event @ StepEvent::ToolCall { .. } => {
                    PanicOnToolCallHook.on_event(_ctx, event).await
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct RepairDefaultApiHook;

    impl AgentHook<MockCompletionModel> for RepairDefaultApiHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(context) => {
                    assert_eq!(context.tool_name, "default_api");
                    Flow::repair("add")
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct RepairToSubtractHook;

    impl AgentHook<MockCompletionModel> for RepairToSubtractHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(_) => Flow::repair("subtract"),
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct RetryDefaultApiHook;

    impl AgentHook<MockCompletionModel> for RetryDefaultApiHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(context) => {
                    let allowed_tools = &context.allowed_tools;
                    Flow::retry(format!("Use one of these tools instead: {allowed_tools:?}"))
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiHook;

    impl AgentHook<MockCompletionModel> for SkipDefaultApiHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(_) => Flow::skip("default_api is not available"),
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone, Default)]
    struct RecordingInvalidToolCallHook {
        contexts: Arc<Mutex<Vec<InvalidToolCallContext>>>,
    }

    impl RecordingInvalidToolCallHook {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .clone()
        }
    }

    impl AgentHook<MockCompletionModel> for RecordingInvalidToolCallHook {
        async fn on_event(
            &self,
            _ctx: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            match event {
                StepEvent::InvalidToolCall(context) => {
                    self.contexts
                        .lock()
                        .expect("invalid tool context records mutex was poisoned")
                        .push(context.clone());
                    Flow::fail()
                }
                _ => Flow::cont(),
            }
        }
    }

    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    impl Tool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = MockOperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(0)
        }
    }

    fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn typed_prompt_response_serializes_with_serialize_only_output() {
        let response = TypedPromptResponse::new(
            SerializeOnly { value: "ok" },
            Usage {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            },
        );

        let json = serde_json::to_string(&response).expect("serialize typed prompt response");
        assert!(json.contains("\"value\":\"ok\""));
    }

    #[test]
    fn typed_prompt_response_deserializes_with_deserialize_only_output() {
        let response: TypedPromptResponse<DeserializeOnly> = serde_json::from_str(
            r#"{"output":{"value":"ok"},"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3,"cached_input_tokens":0,"cache_creation_input_tokens":0,"reasoning_tokens":0}}"#,
        )
        .expect("deserialize typed prompt response");

        assert_eq!(response.requests(), 0);
        assert_eq!(response.output.value, "ok");
        assert_eq!(response.usage.input_tokens, 1);
        assert_eq!(response.usage.output_tokens, 2);
        assert_eq!(response.usage.total_tokens, 3);
    }

    #[test]
    fn prompt_response_serializes_completion_calls_with_missing_usage() {
        let reported_usage = usage(3, 4);
        let response = PromptResponse::new("ok", reported_usage).with_completion_calls(vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, reported_usage),
        ]);

        let value = serde_json::to_value(&response).expect("serialize prompt response");

        // Unreported usage serializes as a plain zero-valued object: zero is
        // Usage's documented sentinel for missing provider metrics, so there
        // is no null encoding to keep in sync.
        assert_eq!(
            value.get("completion_calls"),
            Some(&json!([
                {
                    "call_index": 0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                },
                {
                    "call_index": 1,
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                }
            ]))
        );

        let response: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, reported_usage)
            ]
        );
        assert_eq!(response.requests(), 2);
    }

    #[test]
    fn prompt_response_deserializes_pre_monoid_null_usage_format() {
        // Fixture captured from rig before CompletionCall.usage dropped its
        // Option encoding; `"usage": null` must map to zero-valued usage.
        let fixture = r#"{"output":"ok","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":null},{"call_index":1,"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0}}],"messages":[{"role":"user","content":[{"type":"text","text":"add things"}]}]}"#;

        let response: PromptResponse =
            serde_json::from_str(fixture).expect("old-format response should deserialize");
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, usage(3, 4))
            ]
        );
    }

    #[test]
    fn prompt_response_missing_content_reconstructs_from_output() {
        // Runs serialized before `content` existed must not deserialize to empty
        // text: the structured final turn is reconstructed from `output`, so
        // `output()` and `content()` stay consistent for legacy data.
        let mut value = serde_json::to_value(PromptResponse::new("hello", Usage::new()))
            .expect("serialize prompt response");
        value
            .as_object_mut()
            .expect("prompt response serializes to a JSON object")
            .remove("content");
        assert!(
            value.get("content").is_none(),
            "fixture must omit the content field to model legacy data"
        );

        let response: PromptResponse = serde_json::from_value(value)
            .expect("legacy response without content should deserialize");

        assert_eq!(response.output(), "hello");
        assert_eq!(response.content().iter().count(), 1);
        assert_eq!(response.content().first(), AssistantContent::text("hello"));
    }

    #[test]
    fn prompt_response_missing_content_empty_output_stays_empty_text() {
        let mut value =
            serde_json::to_value(PromptResponse::empty()).expect("serialize prompt response");
        value
            .as_object_mut()
            .expect("prompt response serializes to a JSON object")
            .remove("content");

        let response: PromptResponse = serde_json::from_value(value)
            .expect("legacy empty response without content should deserialize");

        assert_eq!(response.output(), "");
        assert_eq!(response.content().first(), AssistantContent::text(""));
    }

    #[test]
    fn prompt_response_roundtrip_preserves_explicit_content() {
        // An explicitly-set `content` (e.g. the streaming surface's structured
        // final turn) must survive a serialize/deserialize round-trip and is not
        // clobbered by the output-derived fallback.
        let response = PromptResponse::new("visible text", Usage::new())
            .with_content(crate::OneOrMany::one(AssistantContent::text("structured")));

        let value = serde_json::to_value(&response).expect("serialize prompt response");
        assert!(
            value.get("content").is_some(),
            "content is part of the serialized shape"
        );

        let round: PromptResponse =
            serde_json::from_value(value).expect("deserialize prompt response");
        assert_eq!(round.output(), "visible text");
        // The stored content is "structured" — distinct from `output` — proving the
        // output-derived fallback only fills a genuinely absent `content`. (Compare
        // the text directly to sidestep the unrelated `Text::additional_params`
        // serde round-trip asymmetry.)
        let AssistantContent::Text(text) = round.content().first() else {
            panic!("expected text content, got {:?}", round.content().first());
        };
        assert_eq!(text.text, "structured");
    }

    #[test]
    fn prompt_response_serialize_and_deserialize_agree_on_wire_shape() {
        // Serialize *and* deserialize both route through `PromptResponseRepr`, so
        // the two directions agree on `content`'s wire shape (an `Option`).
        // Routing only deserialize through the shadow would make serialize write a
        // bare `OneOrMany` while deserialize expects an `Option`, breaking
        // round-trips for positional / non-self-describing formats. Assert this
        // structurally: the message content types use `#[serde(flatten)]`, which no
        // length-prefixed binary format can encode, and self-describing formats
        // (JSON) collapse `Some(x)` and `x` to identical bytes, hiding the mismatch.
        let response = PromptResponse::new("hi", usage(1, 2))
            .with_completion_calls(vec![CompletionCall::new(0, usage(1, 2))]);

        let from_response = serde_json::to_value(&response).expect("serialize response");
        let from_shadow = serde_json::to_value(PromptResponseRepr::from(response.clone()))
            .expect("serialize shadow");
        assert_eq!(
            from_response, from_shadow,
            "serialize must route through the same shadow as deserialize"
        );

        // ...and the value still round-trips back to an equivalent response.
        let round: PromptResponse =
            serde_json::from_value(from_response).expect("deserialize response");
        assert_eq!(round.output(), "hi");
        assert_eq!(round.usage(), usage(1, 2));
        assert_eq!(
            round.completion_calls(),
            &[CompletionCall::new(0, usage(1, 2))]
        );
    }

    #[tokio::test]
    async fn prompt_response_records_completion_call_without_reported_usage() {
        let model = MockCompletionModel::new([MockTurn::text("ok")]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("say ok")
            .extended_details()
            .await
            .expect("prompt should succeed");

        assert_eq!(response.output, "ok");
        assert_eq!(response.usage, Usage::new());
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, Usage::new())]
        );
    }

    #[tokio::test]
    async fn typed_prompt_response_preserves_completion_calls() {
        let call_usage = Usage {
            input_tokens: 4,
            output_tokens: 6,
            total_tokens: 10,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let model =
            MockCompletionModel::new([MockTurn::text(r#"{"value":"ok"}"#).with_usage(call_usage)]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .extended_details()
            .await
            .expect("typed prompt should succeed");

        assert_eq!(
            response.output,
            TypedAnswer {
                value: "ok".to_string()
            }
        );
        assert_eq!(response.usage, call_usage);
        assert_eq!(
            response.completion_calls(),
            &[CompletionCall::new(0, call_usage)]
        );
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) {
        let history = request.chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            history.len(),
            3,
            "follow-up request should contain the prompt, assistant tool call, and user tool result: {history:?}"
        );

        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ));

        assert!(matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ));

        assert!(matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        ));
    }

    fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.function.name == tool_name
                    ))
            )
        })
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("use the tool")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("unknown model-emitted tool should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "default_api");
                assert_eq!(available_tools, vec!["add".to_string()]);
                assert_eq!(allowed_tools, vec!["add".to_string()]);
                assert!(history_contains_tool_call(&chat_history, "default_api"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    /// The motivating use-case: a `ToolCallExtensions` set on the prompt request is
    /// threaded all the way to the tool the agent loop executes.
    #[tokio::test]
    async fn tool_extensions_reach_tool_through_agent_loop() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockExtensionsProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let mut extensions = ToolCallExtensions::new();
        extensions.insert(SessionId("abc-123".to_string()));

        let out = agent
            .prompt("use the tool")
            .tool_extensions(extensions)
            .max_turns(3)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        assert_eq!(probe.observed().as_deref(), Some("session:abc-123"));
    }

    /// Extensions persist for the whole run, across *multiple* tool-call rounds
    /// (the headline value prop). The model calls the probe in two consecutive
    /// rounds; both must observe the same injected value, not just the first.
    #[tokio::test]
    async fn tool_extensions_persist_across_multiple_rounds() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("c1", "context_probe", json!({})),
            MockTurn::tool_call("c2", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockExtensionsProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let mut extensions = ToolCallExtensions::new();
        extensions.insert(SessionId("abc-123".to_string()));

        let out = agent
            .prompt("use the tool twice")
            .tool_extensions(extensions)
            .max_turns(5)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        assert_eq!(
            probe.observations(),
            vec!["session:abc-123".to_string(), "session:abc-123".to_string()],
        );
    }

    /// Without a context, the same tool runs with an empty one (no panic, no
    /// stale value) — the backward-compatible default path.
    #[tokio::test]
    async fn tool_runs_with_empty_context_when_none_supplied() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "context_probe", json!({})),
            MockTurn::text("done"),
        ]);
        let probe = MockExtensionsProbeTool::default();
        let agent = AgentBuilder::new(model).tool(probe.clone()).build();

        let out = agent
            .prompt("use the tool")
            .max_turns(3)
            .await
            .expect("run succeeds");

        assert_eq!(out, "done");
        // Reaches `call_with_extensions` with an empty context (the override is the
        // single entry point), so it observes "no-session" rather than the
        // plain-`call` "call-no-context".
        assert_eq!(probe.observed().as_deref(), Some("no-session"));
    }

    /// Pins the probe's sentinel: its plain `call` body records
    /// `"call-no-context"`. The dispatched-run tests above assert `"no-session"`
    /// instead, which is what proves dispatch routes through `call_with_extensions`
    /// rather than `call`.
    #[tokio::test]
    async fn probe_plain_call_records_sentinel() {
        let probe = MockExtensionsProbeTool::default();
        let out = probe.call(json!({})).await.expect("call succeeds");
        assert_eq!(out, "call-no-context");
        assert_eq!(probe.observed().as_deref(), Some("call-no-context"));
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_tool_call_provider_id() {
        let invalid_hook = RecordingInvalidToolCallHook::default();
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2}))
                .with_call_id("provider_call_1"),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("use the tool")
            .add_hook(invalid_hook.clone())
            .max_turns(3)
            .await
            .expect_err("invalid tool should fail");

        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert_eq!(context.internal_call_id, None);
        assert!(!context.is_streaming);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "subtract", json!({"x": 3, "y": 1})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .prompt("use the allowed tool")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("disallowed model-emitted tool should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "subtract");
                assert_eq!(
                    available_tools,
                    vec!["add".to_string(), "subtract".to_string()]
                );
                assert_eq!(allowed_tools, vec!["add".to_string()]);
                assert!(history_contains_tool_call(&chat_history, "subtract"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_non_streaming_tool_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject returned tool calls");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => {
                assert_eq!(tool_name, "add");
                assert_eq!(available_tools, vec!["add".to_string()]);
                assert!(allowed_tools.is_empty());
                assert!(history_contains_tool_call(&chat_history, "add"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_non_streaming_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("done"),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("repaired tool call should execute");

        assert_eq!(response.output, "done");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(!history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        crate::message::ToolResultContent::Text(text)
                                            if text.text == "5"
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_adds_feedback_and_retries_non_streaming() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .extended_details()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(recorded.request_count(), 2);
        let messages = response.messages.expect("messages should be present");
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        crate::message::ToolResultContent::Text(text)
                                            if text.text.contains("Use one of these tools instead")
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retries_mixed_non_streaming_turn_without_executing_valid_call()
    {
        let add_calls = Arc::new(AtomicU32::new(0));
        let mut valid_tool_call = ToolCall::new(
            "tool_call_1".to_string(),
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        );
        valid_tool_call.call_id = Some("call_1".to_string());
        let mut invalid_tool_call = ToolCall::new(
            "tool_call_2".to_string(),
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        );
        invalid_tool_call.call_id = Some("call_2".to_string());
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ])
            .expect("tool-call response should be non-empty"),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .extended_details()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(retry_history.len(), 3);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "add"
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.call_id.as_deref() == Some("call_1")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    crate::message::ToolResultContent::Text(text)
                                        if text.text == super::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.call_id.as_deref() == Some("call_2")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    crate::message::ToolResultContent::Text(text)
                                        if text.text.contains("Use one of these tools instead")
                                ))
            ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skips_mixed_non_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let mut valid_tool_call = ToolCall::new(
            "tool_call_1".to_string(),
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        );
        valid_tool_call.call_id = Some("call_1".to_string());
        let mut invalid_tool_call = ToolCall::new(
            "tool_call_2".to_string(),
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        );
        invalid_tool_call.call_id = Some("call_2".to_string());
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ])
            .expect("tool-call response should be non-empty"),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiAndPanicOnToolCallHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should recover without executing peer tools");

        assert_eq!(response.output, "skipped");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(matches!(
            messages.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.call_id.as_deref() == Some("call_1")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    crate::message::ToolResultContent::Text(text)
                                        if text.text == super::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.call_id.as_deref() == Some("call_2")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    crate::message::ToolResultContent::Text(text)
                                        if text.text == "default_api is not available"
                                ))
                    ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt("add")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .await
            .expect_err("retry without budget should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                chat_history,
                ..
            } => {
                assert_eq!(tool_name, "default_api");
                assert!(history_contains_tool_call(&chat_history, "default_api"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_skip_structured_non_streaming_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should continue with synthetic tool result");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        crate::message::ToolResultContent::Text(text)
                                            if text.text == "default_api is not available"
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn skip_under_specific_tool_choice_returns_synthetic_feedback() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .prompt("add")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .extended_details()
            .await
            .expect("skip should produce synthetic feedback under Specific");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.id == "tool_call_1"
                                    && result.content.iter().any(|content| {
                                        matches!(
                                            content,
                                            crate::message::ToolResultContent::Text(text)
                                                if text.text == "default_api is not available"
                                        )
                                    })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn repair_to_disallowed_specific_tool_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .prompt("add")
            .add_hook(RepairToSubtractHook)
            .max_turns(3)
            .await
            .expect_err("repair to a disallowed tool should fail");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "subtract");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn repair_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject repaired tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "add");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn skip_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .prompt("do not use tools")
            .add_hook(SkipDefaultApiHook)
            .max_turns(3)
            .await
            .expect_err("ToolChoice::None should reject skipped tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "default_api");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_default_invalid_tool_call_fails_fast() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .await
            .expect_err("typed prompt should preserve fail-fast default");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_repair_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"repaired"}"#),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RepairDefaultApiHook)
            .max_turns(3)
            .await
            .expect("typed prompt should repair invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "repaired".to_string()
            }
        );
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_retry_and_parse_response() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"retried"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .await
            .expect("typed prompt should retry invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "retried".to_string()
            }
        );
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let err = agent
            .prompt_typed::<TypedAnswer>("return typed json")
            .add_hook(RetryDefaultApiHook)
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .await
            .expect_err("typed prompt should fail when retry budget is exhausted");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_specific_tool_choice_fails_before_non_streaming_provider_request() {
        let model = MockCompletionModel::text("should not be requested");
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .build();

        let err = agent
            .prompt("use the missing tool")
            .await
            .expect_err("invalid ToolChoice::Specific should fail before provider request");

        match err {
            PromptError::CompletionError(CompletionError::RequestError(err)) => {
                let msg = err.to_string();
                assert!(msg.contains("missing"), "got: {msg}");
                assert!(msg.contains("add"), "got: {msg}");
            }
            other => panic!("expected CompletionError::RequestError, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 0);
    }

    #[tokio::test]
    async fn allowed_specific_tool_call_executes_normally() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .prompt("use the allowed tool")
            .max_turns(3)
            .await
            .expect("allowed specific tool should execute");

        assert_eq!(response, "done");
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn prompt_request_stops_cleanly_on_empty_terminal_turn() {
        let first_call_usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let second_call_usage = Usage {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        };
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2}))
                .with_call_id("call_1")
                .with_usage(first_call_usage),
            MockTurn::text("").with_usage(second_call_usage),
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let response = agent
            .prompt("do tool work")
            .max_turns(3)
            .extended_details()
            .await
            .expect("empty terminal turn should not error");

        assert!(response.output.is_empty());
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 2,
                output_tokens: 2,
                total_tokens: 4,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            response.completion_calls(),
            &[
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );

        let history = response
            .messages
            .expect("extended response should include history");
        assert_eq!(history.len(), 3);
        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        )));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::User { content }
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        )));
        assert!(!history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text.is_empty()
                ))
        )));
        let requests = agent.model.requests();
        assert_eq!(requests.len(), 2);
        validate_follow_up_tool_history(&requests[1]);
    }

    #[tokio::test]
    async fn prompt_request_concatenates_text_blocks_without_inserted_newlines() {
        let model = MockCompletionModel::new([MockTurn::from_contents([
            AssistantContent::Text(Text::new("According to the document, ")),
            AssistantContent::Text(Text::new("the grass is green")),
            AssistantContent::Text(Text::new(" and the sky is blue.")),
        ])
        .expect("mock response should contain text blocks")]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("answer with cited spans")
            .await
            .expect("prompt should succeed");

        assert_eq!(
            response,
            "According to the document, the grass is green and the sky is blue."
        );
    }

    #[tokio::test]
    async fn prompt_request_preserves_metadata_only_text_turn_in_history() {
        let metadata = json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": null,
                "encrypted_index": "encrypted-reference"
            }]
        });
        let model =
            MockCompletionModel::new([MockTurn::from_content(AssistantContent::Text(Text {
                text: String::new(),
                additional_params: Some(metadata.clone()),
            }))]);
        let agent = AgentBuilder::new(model).build();

        let response = agent
            .prompt("answer with cited metadata")
            .extended_details()
            .await
            .expect("metadata-only text turn should succeed");

        assert!(response.output.is_empty());
        let history = response
            .messages
            .expect("extended response should include history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    AssistantContent::Text(text)
                        if text.text.is_empty()
                            && text.additional_params.as_ref() == Some(&metadata)
                )
        )));
    }

    // ----- Conversation memory integration tests -----

    use crate::memory::{ConversationMemory, InMemoryConversationMemory};

    #[tokio::test]
    async fn memory_loads_into_request_history() {
        let memory = InMemoryConversationMemory::new();
        memory
            .append(
                "thread-1",
                vec![Message::user("hello"), Message::assistant("hi there")],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();

        let agent = AgentBuilder::new(model).memory(memory).build();
        let _ = agent
            .prompt("ping")
            .conversation("thread-1")
            .await
            .expect("prompt should succeed");

        let received = recorded.requests()[0]
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            received.len(),
            3,
            "loaded memory (2) + current prompt should appear in request: {received:?}"
        );
    }

    #[tokio::test]
    async fn memory_appends_full_turn_after_success() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model).memory(memory.clone()).build();

        let _ = agent
            .prompt("hello")
            .conversation("t1")
            .await
            .expect("prompt should succeed");

        let stored = memory.load("t1").await.unwrap();
        assert_eq!(stored.len(), 2, "user prompt + assistant response saved");
    }

    #[tokio::test]
    async fn explicit_with_history_overrides_memory() {
        let memory = CountingMemory::default();
        memory
            .inner()
            .append("t1", vec![Message::user("from-memory")])
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();

        let agent = AgentBuilder::new(model).memory(memory.clone()).build();
        let _ = agent
            .prompt("hello")
            .conversation("t1")
            .history(vec![Message::user("from-caller")])
            .await
            .expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0, "load skipped");
        let appends = memory.append_count();
        assert_eq!(appends, 0, "append skipped");

        let received = recorded.requests()[0]
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(received.len(), 2, "caller history (1) + current prompt");
        assert!(matches!(
            received.first(),
            Some(Message::User { content })
                if matches!(content.first(), UserContent::Text(t) if t.text == "from-caller")
        ));
    }

    #[tokio::test]
    async fn memory_unchanged_on_provider_error() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::new([MockTurn::error("boom")]);

        let agent = AgentBuilder::new(model).memory(memory.clone()).build();
        let result = agent.prompt("hello").conversation("t1").await;
        assert!(result.is_err());

        let stored = memory.load("t1").await.unwrap();
        assert!(stored.is_empty(), "no append on error");
    }

    #[tokio::test]
    async fn missing_conversation_id_behaves_as_no_memory() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model).memory(memory.clone()).build();

        let _ = agent.prompt("hello").await.expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0);
        assert_eq!(memory.append_count(), 0);
    }

    #[tokio::test]
    async fn default_conversation_id_is_used_when_none_per_request() {
        let memory = InMemoryConversationMemory::new();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .conversation("default-thread")
            .build();

        let _ = agent.prompt("hello").await.expect("prompt should succeed");
        let stored = memory.load("default-thread").await.unwrap();
        assert_eq!(stored.len(), 2);
    }

    #[tokio::test]
    async fn with_filter_truncates_loaded_history() {
        let memory = InMemoryConversationMemory::new()
            .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).rev().collect());
        memory
            .append(
                "t1",
                vec![
                    Message::user("1"),
                    Message::assistant("2"),
                    Message::user("3"),
                    Message::assistant("4"),
                ],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::text("ack");
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).memory(memory).build();

        let _ = agent
            .prompt("ping")
            .conversation("t1")
            .await
            .expect("prompt should succeed");

        let received = recorded.requests()[0]
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            received.len(),
            3,
            "window-truncated history (2) + current prompt"
        );
    }

    #[tokio::test]
    async fn without_memory_disables_for_request() {
        let memory = CountingMemory::default();
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(memory.clone())
            .conversation("t1")
            .build();

        let _ = agent
            .prompt("hello")
            .without_memory()
            .await
            .expect("prompt should succeed");

        assert_eq!(memory.load_count(), 0);
        assert_eq!(memory.append_count(), 0);
    }

    #[tokio::test]
    async fn memory_load_error_surfaces_as_prompt_error() {
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(FailingMemory::default())
            .build();
        let result = agent.prompt("hello").conversation("t1").await;

        match result {
            Err(PromptError::MemoryError(err)) => {
                let msg = err.to_string();
                assert!(msg.contains("load boom"), "got: {msg}");
            }
            other => panic!("expected PromptError::MemoryError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn memory_append_error_does_not_drop_response() {
        let model = MockCompletionModel::text("ack");
        let agent = AgentBuilder::new(model)
            .memory(AppendFailingMemory::default())
            .build();
        let response: String = agent
            .prompt("hello")
            .conversation("t1")
            .await
            .expect("append failure must not block successful completion");

        assert!(!response.is_empty());
    }
}
