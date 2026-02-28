pub mod hooks;
pub mod streaming;

use hooks::{HookAction, PromptHook, ToolCallHookAction};
use std::{
    future::IntoFuture,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tracing::{Instrument, span::Id};

use futures::{StreamExt, stream};
use tracing::info_span;

use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Message, PromptError, Usage},
    json_utils,
    message::{AssistantContent, ToolChoice, ToolResultContent, UserContent},
    tool::server::ToolServerHandle,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};

use super::{
    Agent,
    completion::{DynamicContextStore, build_completion_request},
};

pub trait PromptType {}
pub struct Standard;
pub struct Extended;

impl PromptType for Standard {}
impl PromptType for Extended {}

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// If you expect to continuously call tools, you will want to ensure you use the `.multi_turn()`
/// argument to add more turns as by default, it is 0 (meaning only 1 tool round-trip). Otherwise,
/// attempting to await (which will send the prompt request) can potentially return
/// [`crate::completion::request::PromptError::MaxTurnsError`] if the agent decides to call tools
/// back to back.
pub struct PromptRequest<'a, S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history to include with the prompt
    /// Note: chat history needs to outlive the agent as it might be used with other agents
    chat_history: Option<&'a mut Vec<Message>>,
    /// Maximum depth for multi-turn conversations (0 means no multi-turn)
    max_turns: usize,

    // Agent data (cloned from agent to allow hook type transitions):
    /// The completion model
    model: Arc<M>,
    /// Agent name for logging
    agent_name: Option<String>,
    /// System prompt
    preamble: Option<String>,
    /// Static context documents
    static_context: Vec<Document>,
    /// Temperature setting
    temperature: Option<f64>,
    /// Max tokens setting
    max_tokens: Option<u64>,
    /// Additional model parameters
    additional_params: Option<serde_json::Value>,
    /// Tool server handle for tool execution
    tool_server_handle: ToolServerHandle,
    /// Dynamic context store
    dynamic_context: DynamicContextStore,
    /// Tool choice setting
    tool_choice: Option<ToolChoice>,

    /// Phantom data to track the type of the request
    state: PhantomData<S>,
    /// Optional per-request hook for events
    hook: Option<P>,
    /// How many tools should be executed at the same time (1 by default).
    concurrency: usize,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
}

impl<'a, M, P> PromptRequest<'a, Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Create a new PromptRequest from an agent, cloning the agent's data and default hook.
    pub fn from_agent(agent: &Agent<M, P>, prompt: impl Into<Message>) -> Self {
        PromptRequest {
            prompt: prompt.into(),
            chat_history: None,
            max_turns: agent.default_max_turns.unwrap_or_default(),
            model: agent.model.clone(),
            agent_name: agent.name.clone(),
            preamble: agent.preamble.clone(),
            static_context: agent.static_context.clone(),
            temperature: agent.temperature,
            max_tokens: agent.max_tokens,
            additional_params: agent.additional_params.clone(),
            tool_server_handle: agent.tool_server_handle.clone(),
            dynamic_context: agent.dynamic_context.clone(),
            tool_choice: agent.tool_choice.clone(),
            state: PhantomData,
            hook: agent.hook.clone(),
            concurrency: 1,
            output_schema: agent.output_schema.clone(),
        }
    }
}

impl<'a, S, M, P> PromptRequest<'a, S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Enable returning extended details for responses (includes aggregated token usage)
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> PromptRequest<'a, Extended, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_turns: self.max_turns,
            model: self.model,
            agent_name: self.agent_name,
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_server_handle: self.tool_server_handle,
            dynamic_context: self.dynamic_context,
            tool_choice: self.tool_choice,
            state: PhantomData,
            hook: self.hook,
            concurrency: self.concurrency,
            output_schema: self.output_schema,
        }
    }

    /// Set the maximum number of turns for multi-turn conversations. A given agent may require multiple turns for tool-calling before giving an answer.
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, depth: usize) -> Self {
        self.max_turns = depth;
        self
    }

    /// Add concurrency to the prompt request.
    /// This will cause the agent to execute tools concurrently.
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Add chat history to the prompt request
    pub fn with_history(mut self, history: &'a mut Vec<Message>) -> Self {
        self.chat_history = Some(history);
        self
    }

    /// Attach a per-request hook for tool call events.
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> PromptRequest<'a, S, M, P2>
    where
        P2: PromptHook<M>,
    {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_turns: self.max_turns,
            model: self.model,
            agent_name: self.agent_name,
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_server_handle: self.tool_server_handle,
            dynamic_context: self.dynamic_context,
            tool_choice: self.tool_choice,
            state: PhantomData,
            hook: Some(hook),
            concurrency: self.concurrency,
            output_schema: self.output_schema,
        }
    }
}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl<'a, M, P> IntoFuture for PromptRequest<'a, Standard, M, P>
where
    M: CompletionModel + 'a,
    P: PromptHook<M> + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<'a, M, P> IntoFuture for PromptRequest<'a, Extended, M, P>
where
    M: CompletionModel + 'a,
    P: PromptHook<M> + 'static,
{
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M, P> PromptRequest<'_, Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    async fn send(self) -> Result<String, PromptError> {
        self.extended_details().send().await.map(|resp| resp.output)
    }
}

#[derive(Debug, Clone)]
pub struct PromptResponse {
    pub output: String,
    pub usage: Usage,
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        Self {
            output: output.into(),
            usage,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypedPromptResponse<T> {
    pub output: T,
    pub total_usage: Usage,
}

impl<T> TypedPromptResponse<T> {
    pub fn new(output: T, total_usage: Usage) -> Self {
        Self {
            output,
            total_usage,
        }
    }
}

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

impl<M, P> PromptRequest<'_, Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn agent_name(&self) -> &str {
        self.agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    async fn send(mut self) -> Result<PromptResponse, PromptError> {
        let agent_span = if tracing::Span::current().is_disabled() {
            info_span!(
                "invoke_agent",
                gen_ai.operation.name = "invoke_agent",
                gen_ai.agent.name = self.agent_name(),
                gen_ai.system_instructions = self.preamble,
                gen_ai.prompt = tracing::field::Empty,
                gen_ai.completion = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        if let Some(text) = self.prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        // Capture agent_name before borrowing chat_history
        let agent_name_for_span = self.agent_name.clone();

        let chat_history = if let Some(history) = self.chat_history.as_mut() {
            history.push(self.prompt.to_owned());
            history
        } else {
            &mut vec![self.prompt.to_owned()]
        };

        let mut current_max_turns = 0;
        let mut usage = Usage::new();
        let current_span_id: AtomicU64 = AtomicU64::new(0);

        // We need to do at least 2 loops for 1 roundtrip (user expects normal message)
        let last_prompt = loop {
            let prompt = chat_history
                .last()
                .cloned()
                .expect("there should always be at least one message in the chat history");

            if current_max_turns > self.max_turns + 1 {
                break prompt;
            }

            current_max_turns += 1;

            if self.max_turns > 1 {
                tracing::info!(
                    "Current conversation depth: {}/{}",
                    current_max_turns,
                    self.max_turns
                );
            }

            if let Some(ref hook) = self.hook
                && let HookAction::Terminate { reason } = hook
                    .on_completion_call(&prompt, &chat_history[..chat_history.len() - 1])
                    .await
            {
                return Err(PromptError::prompt_cancelled(chat_history.to_vec(), reason));
            }

            let span = tracing::Span::current();
            let chat_span = info_span!(
                target: "rig::agent_chat",
                parent: &span,
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.agent.name = agent_name_for_span.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                gen_ai.system_instructions = self.preamble,
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            );

            let chat_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                chat_span.follows_from(id).to_owned()
            } else {
                chat_span
            };

            if let Some(id) = chat_span.id() {
                current_span_id.store(id.into_u64(), Ordering::SeqCst);
            };

            let resp = build_completion_request(
                &self.model,
                prompt.clone(),
                chat_history[..chat_history.len() - 1].to_vec(),
                self.preamble.as_deref(),
                &self.static_context,
                self.temperature,
                self.max_tokens,
                self.additional_params.as_ref(),
                self.tool_choice.as_ref(),
                &self.tool_server_handle,
                &self.dynamic_context,
                self.output_schema.as_ref(),
            )
            .await?
            .send()
            .instrument(chat_span.clone())
            .await?;

            usage += resp.usage;

            if let Some(ref hook) = self.hook
                && let HookAction::Terminate { reason } =
                    hook.on_completion_response(&prompt, &resp).await
            {
                return Err(PromptError::prompt_cancelled(chat_history.to_vec(), reason));
            }

            let (tool_calls, texts): (Vec<_>, Vec<_>) = resp
                .choice
                .iter()
                .partition(|choice| matches!(choice, AssistantContent::ToolCall(_)));

            chat_history.push(Message::Assistant {
                id: resp.message_id.clone(),
                content: resp.choice.clone(),
            });

            if tool_calls.is_empty() {
                let merged_texts = texts
                    .into_iter()
                    .filter_map(|content| {
                        if let AssistantContent::Text(text) = content {
                            Some(text.text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if self.max_turns > 1 {
                    tracing::info!("Depth reached: {}/{}", current_max_turns, self.max_turns);
                }

                agent_span.record("gen_ai.completion", &merged_texts);
                agent_span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                agent_span.record("gen_ai.usage.output_tokens", usage.output_tokens);

                // If there are no tool calls, depth is not relevant, we can just return the merged text response.
                return Ok(PromptResponse::new(merged_texts, usage));
            }

            let hook = self.hook.clone();
            let tool_server_handle = self.tool_server_handle.clone();

            let tool_calls: Vec<AssistantContent> = tool_calls.into_iter().cloned().collect();
            let tool_content = stream::iter(tool_calls)
                .map(|choice| {
                    let hook1 = hook.clone();
                    let hook2 = hook.clone();
                    let tool_server_handle = tool_server_handle.clone();

                    let tool_span = info_span!(
                        "execute_tool",
                        gen_ai.operation.name = "execute_tool",
                        gen_ai.tool.type = "function",
                        gen_ai.tool.name = tracing::field::Empty,
                        gen_ai.tool.call.id = tracing::field::Empty,
                        gen_ai.tool.call.arguments = tracing::field::Empty,
                        gen_ai.tool.call.result = tracing::field::Empty
                    );

                    let tool_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                        let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                        tool_span.follows_from(id).to_owned()
                    } else {
                        tool_span
                    };

                    if let Some(id) = tool_span.id() {
                        current_span_id.store(id.into_u64(), Ordering::SeqCst);
                    };

                    let cloned_chat_history = chat_history.clone().to_vec();

                    async move {
                        if let AssistantContent::ToolCall(tool_call) = choice {
                            let tool_name = &tool_call.function.name;
                            let args =
                                json_utils::value_to_json_string(&tool_call.function.arguments);
                            let internal_call_id = nanoid::nanoid!();
                            let tool_span = tracing::Span::current();
                            tool_span.record("gen_ai.tool.name", tool_name);
                            tool_span.record("gen_ai.tool.call.id", &tool_call.id);
                            tool_span.record("gen_ai.tool.call.arguments", &args);
                            if let Some(hook) = hook1 {
                                let action = hook
                                    .on_tool_call(
                                        tool_name,
                                        tool_call.call_id.clone(),
                                        &internal_call_id,
                                        &args,
                                    )
                                    .await;

                                if let ToolCallHookAction::Terminate { reason } = action {
                                    return Err(PromptError::prompt_cancelled(
                                        cloned_chat_history,
                                        reason,
                                    ));
                                }

                                if let ToolCallHookAction::Skip { reason } = action {
                                    // Tool execution rejected, return rejection message as tool result
                                    tracing::info!(
                                        tool_name = tool_name,
                                        reason = reason,
                                        "Tool call rejected"
                                    );
                                    if let Some(call_id) = tool_call.call_id.clone() {
                                        return Ok(UserContent::tool_result_with_call_id(
                                            tool_call.id.clone(),
                                            call_id,
                                            OneOrMany::one(reason.into()),
                                        ));
                                    } else {
                                        return Ok(UserContent::tool_result(
                                            tool_call.id.clone(),
                                            OneOrMany::one(reason.into()),
                                        ));
                                    }
                                }
                            }
                            let output = match tool_server_handle.call_tool(tool_name, &args).await
                            {
                                Ok(res) => res,
                                Err(e) => {
                                    tracing::warn!("Error while executing tool: {e}");
                                    e.to_string()
                                }
                            };
                            if let Some(hook) = hook2
                                && let HookAction::Terminate { reason } = hook
                                    .on_tool_result(
                                        tool_name,
                                        tool_call.call_id.clone(),
                                        &internal_call_id,
                                        &args,
                                        &output.to_string(),
                                    )
                                    .await
                            {
                                return Err(PromptError::prompt_cancelled(
                                    cloned_chat_history,
                                    reason,
                                ));
                            }

                            tool_span.record("gen_ai.tool.call.result", &output);
                            tracing::info!(
                                "executed tool {tool_name} with args {args}. result: {output}"
                            );
                            if let Some(call_id) = tool_call.call_id.clone() {
                                Ok(UserContent::tool_result_with_call_id(
                                    tool_call.id.clone(),
                                    call_id,
                                    ToolResultContent::from_tool_output(output),
                                ))
                            } else {
                                Ok(UserContent::tool_result(
                                    tool_call.id.clone(),
                                    ToolResultContent::from_tool_output(output),
                                ))
                            }
                        } else {
                            unreachable!(
                                "This should never happen as we already filtered for `ToolCall`"
                            )
                        }
                    }
                    .instrument(tool_span)
                })
                .buffer_unordered(self.concurrency)
                .collect::<Vec<Result<UserContent, PromptError>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

            chat_history.push(Message::User {
                content: OneOrMany::many(tool_content).expect("There is atleast one tool call"),
            });
        };

        // If we reach here, we never resolved the final tool call. We need to do ... something.
        Err(PromptError::MaxTurnsError {
            max_turns: self.max_turns,
            chat_history: Box::new(chat_history.clone()),
            prompt: Box::new(last_prompt),
        })
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
pub struct TypedPromptRequest<'a, T, S, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    inner: PromptRequest<'a, S, M, P>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, M, P> TypedPromptRequest<'a, T, Standard, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Create a new TypedPromptRequest from an agent.
    ///
    /// This automatically sets the output schema based on the type parameter `T`.
    pub fn from_agent(agent: &Agent<M, P>, prompt: impl Into<Message>) -> Self {
        let mut inner = PromptRequest::from_agent(agent, prompt);
        // Override the output schema with the schema for T
        inner.output_schema = Some(schema_for!(T));
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, S, M, P> TypedPromptRequest<'a, T, S, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Enable returning extended details for responses (includes aggregated token usage).
    ///
    /// Note: This changes the type of the response from `.send()` to return a `TypedPromptResponse<T>` struct
    /// instead of just `T`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> TypedPromptRequest<'a, T, Extended, M, P> {
        TypedPromptRequest {
            inner: self.inner.extended_details(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the maximum number of turns for multi-turn conversations.
    ///
    /// A given agent may require multiple turns for tool-calling before giving an answer.
    /// If the maximum turn number is exceeded, it will return a
    /// [`StructuredOutputError::PromptError`] wrapping a `MaxTurnsError`.
    pub fn max_turns(mut self, depth: usize) -> Self {
        self.inner = self.inner.max_turns(depth);
        self
    }

    /// Add concurrency to the prompt request.
    ///
    /// This will cause the agent to execute tools concurrently.
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.inner = self.inner.with_tool_concurrency(concurrency);
        self
    }

    /// Add chat history to the prompt request.
    pub fn with_history(mut self, history: &'a mut Vec<Message>) -> Self {
        self.inner = self.inner.with_history(history);
        self
    }

    /// Attach a per-request hook for tool call events.
    ///
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> TypedPromptRequest<'a, T, S, M, P2>
    where
        P2: PromptHook<M>,
    {
        TypedPromptRequest {
            inner: self.inner.with_hook(hook),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, M, P> TypedPromptRequest<'a, T, Standard, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Send the typed prompt request and deserialize the response.
    async fn send(self) -> Result<T, StructuredOutputError> {
        let response = self.inner.send().await?;

        if response.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = serde_json::from_str(&response)?;
        Ok(parsed)
    }
}

impl<'a, T, M, P> TypedPromptRequest<'a, T, Extended, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Send the typed prompt request with extended details and deserialize the response.
    async fn send(self) -> Result<TypedPromptResponse<T>, StructuredOutputError> {
        let response = self.inner.send().await?;

        if response.output.is_empty() {
            return Err(StructuredOutputError::EmptyResponse);
        }

        let parsed: T = serde_json::from_str(&response.output)?;
        Ok(TypedPromptResponse::new(parsed, response.total_usage))
    }
}

impl<'a, T, M, P> IntoFuture for TypedPromptRequest<'a, T, Standard, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'a,
    M: CompletionModel + 'a,
    P: PromptHook<M> + 'static,
{
    type Output = Result<T, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<'a, T, M, P> IntoFuture for TypedPromptRequest<'a, T, Extended, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'a,
    M: CompletionModel + 'a,
    P: PromptHook<M> + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}
