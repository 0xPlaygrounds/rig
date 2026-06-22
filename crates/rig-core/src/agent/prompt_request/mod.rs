pub mod hooks;
pub mod streaming;

use super::{
    Agent,
    completion::{DynamicContextStore, build_prepared_completion_request},
    run::{
        AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, OutputMode,
        PendingToolCall,
    },
};
use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Message, PromptError, Usage},
    json_utils,
    memory::ConversationMemory,
    message::{AssistantContent, ToolChoice, ToolResultContent, UserContent},
    tool::server::ToolServerHandle,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{StreamExt, stream};
use hooks::{HookAction, InvalidToolCallHookAction, PromptHook, ToolCallHookAction};
use serde::{Deserialize, Serialize};
use std::{
    future::IntoFuture,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tracing::info_span;
use tracing::{Instrument, span::Id};

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
pub struct PromptRequest<S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history provided by the caller.
    chat_history: Option<Vec<Message>>,
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
    /// Maximum number of invalid tool-call retries for this request.
    max_invalid_tool_call_retries: usize,
    /// How many tools should be executed at the same time (1 by default).
    concurrency: usize,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
    output_mode: OutputMode,
    /// Optional conversation memory backend cloned from the agent.
    memory: Option<Arc<dyn ConversationMemory>>,
    /// Optional conversation id used for loading and saving memory.
    conversation_id: Option<String>,
}

impl<M, P> PromptRequest<Standard, M, P>
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
            max_invalid_tool_call_retries: 0,
            concurrency: 1,
            output_schema: agent.output_schema.clone(),
            output_mode: agent.output_mode.clone(),
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
        }
    }
}

impl<S, M, P> PromptRequest<S, M, P>
where
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Enable returning extended details for responses (includes aggregated token usage
    /// and the full message history accumulated during the agent loop).
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation and inspecting the full message exchange.
    pub fn extended_details(self) -> PromptRequest<Extended, M, P> {
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
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            concurrency: self.concurrency,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            conversation_id: self.conversation_id,
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

    /// Add chat history to the prompt request.
    pub fn with_history<H, T>(mut self, history: H) -> Self
    where
        H: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.chat_history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Set the conversation id used to load and persist memory for this request.
    ///
    /// Overrides any default conversation id set on the agent. If memory is not
    /// configured on the agent, this has no effect.
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Disable conversation memory for this request.
    ///
    /// History will neither be loaded from nor saved to the agent's memory backend.
    pub fn without_memory(mut self) -> Self {
        self.memory = None;
        self.conversation_id = None;
        self
    }

    /// Attach a per-request hook for tool call events.
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> PromptRequest<S, M, P2>
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
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            concurrency: self.concurrency,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            memory: self.memory,
            conversation_id: self.conversation_id,
        }
    }

    /// Set the retry budget for [`InvalidToolCallHookAction::Retry`].
    ///
    /// Invalid tool-call retries also consume normal multi-turn depth.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }
}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl<M, P> IntoFuture for PromptRequest<Standard, M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M, P> IntoFuture for PromptRequest<Extended, M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<M, P> PromptRequest<Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptResponse {
    pub output: String,
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last
    /// entry's usage to inspect the final completion request's prompt/context
    /// length. Zero-valued entry usage means the provider reported no usage
    /// metrics for that request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completion_calls: Vec<CompletionCall>,
    pub messages: Option<Vec<Message>>,
}

impl std::fmt::Display for PromptResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output.fmt(f)
    }
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, usage: Usage) -> Self {
        Self {
            output: output.into(),
            usage,
            completion_calls: Vec::new(),
            messages: None,
        }
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

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

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

pub(crate) fn tool_result_user_content(
    id: String,
    call_id: Option<String>,
    tool_result: String,
) -> UserContent {
    let content = ToolResultContent::from_tool_output(tool_result);
    match call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(id, call_id, content),
        None => UserContent::tool_result(id, content),
    }
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
                Some(tool_result_user_content(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    feedback.clone(),
                ))
            }
            AssistantContent::ToolCall(tool_call) => Some(tool_result_user_content(
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

impl<M, P> PromptRequest<Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    fn agent_name(&self) -> &str {
        self.agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    async fn send(self) -> Result<PromptResponse, PromptError> {
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
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                gen_ai.usage.reasoning_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        if let Some(text) = self.prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        let agent_name_for_span = self.agent_name.clone();
        // When the caller passes explicit history, memory is fully bypassed for this
        // request (no load AND no save). Otherwise, if a memory backend and
        // conversation id are both configured, load prior history; if either is
        // missing, behave as if no memory is configured.
        let (chat_history, memory_handle) = match self.chat_history {
            Some(history) => (Some(history), None),
            None => match (self.memory, self.conversation_id) {
                (Some(memory), Some(id)) => {
                    let loaded = memory.load(&id).await?;
                    (Some(loaded), Some((memory, id)))
                }
                _ => (None, None),
            },
        };

        let mut run = AgentRun::new(self.prompt.clone())
            .max_turns(self.max_turns)
            .max_invalid_tool_call_retries(self.max_invalid_tool_call_retries)
            .with_output_validation(
                self.output_schema
                    .as_ref()
                    .map(|schema| schema.as_value().clone()),
                DEFAULT_OUTPUT_RETRIES,
            );
        if let Some(history) = chat_history {
            run = run.with_history(history);
        }
        if let Some(tool_choice) = self.tool_choice.clone() {
            run = run.with_tool_choice(tool_choice);
        }

        let current_span_id: AtomicU64 = AtomicU64::new(0);

        loop {
            match run.next_step()? {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    if self.max_turns > 1 {
                        tracing::info!("Current conversation depth: {}/{}", turn, self.max_turns);
                    }

                    if let Some(ref hook) = self.hook
                        && let HookAction::Terminate { reason } =
                            hook.on_completion_call(&prompt, &history).await
                    {
                        return Err(run.cancel_error(reason));
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
                        gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                        gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                        gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                        gen_ai.usage.reasoning_tokens = tracing::field::Empty,
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

                    // Pin Tool output mode once committed so later turns stay
                    // consistent even if the per-turn tool set changes (#1928).
                    // The pin rides on `output_tool_name`, which is persisted on
                    // the run, so it also survives a serialize/resume.
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let prepared_request = build_prepared_completion_request(
                        &self.model,
                        prompt.clone(),
                        &history,
                        self.preamble.as_deref(),
                        &self.static_context,
                        self.temperature,
                        self.max_tokens,
                        self.additional_params.as_ref(),
                        self.tool_choice.as_ref(),
                        &self.tool_server_handle,
                        &self.dynamic_context,
                        self.output_schema.as_ref(),
                        &self.output_mode,
                        committed_output_tool.as_deref(),
                    )
                    .await?;

                    let resp = prepared_request
                        .builder
                        .send()
                        .instrument(chat_span.clone())
                        .await?;

                    run.set_output_tool_name(prepared_request.output_tool_name.clone());

                    let mut outcome = run.model_response(ModelTurn::new(
                        resp.message_id.clone(),
                        resp.choice.clone(),
                        resp.usage,
                        prepared_request.executable_tool_names,
                        prepared_request.allowed_tool_names,
                    ))?;

                    loop {
                        match outcome {
                            ModelTurnOutcome::NeedsResolution(context) => {
                                let action = match self.hook.as_ref() {
                                    Some(hook) => hook.on_invalid_tool_call(&context).await,
                                    None => InvalidToolCallHookAction::fail(),
                                };
                                outcome = run.resolve_invalid_tool_call(action)?;
                            }
                            ModelTurnOutcome::TurnRetried => break,
                            ModelTurnOutcome::Continue {
                                response_hook_suppressed,
                            } => {
                                if !response_hook_suppressed
                                    && let Some(ref hook) = self.hook
                                    && let HookAction::Terminate { reason } =
                                        hook.on_completion_response(&prompt, &resp).await
                                {
                                    return Err(run.cancel_error(reason));
                                }
                                break;
                            }
                        }
                    }
                }
                AgentRunStep::CallTools { calls } => {
                    let hook = self.hook.clone();
                    let tool_server_handle = self.tool_server_handle.clone();

                    // For error handling in concurrent tool execution, we need to build full history
                    let full_history_for_errors = run.full_history();

                    let tool_content = stream::iter(calls)
                        .map(|pending| {
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

                            // Clone full history for error reporting in concurrent tool execution
                            let cloned_history_for_error = full_history_for_errors.clone();

                            async move {
                                let PendingToolCall {
                                    tool_call,
                                    preresolved_result,
                                    ..
                                } = pending;
                                let tool_name = &tool_call.function.name;
                                let args =
                                    json_utils::value_to_json_string(&tool_call.function.arguments);
                                let internal_call_id = crate::id::generate();
                                if let Some(result) = preresolved_result {
                                    return Ok(result);
                                }
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
                                            cloned_history_for_error,
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
                                let output =
                                    match tool_server_handle.call_tool(tool_name, &args).await {
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
                                        cloned_history_for_error,
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
                            }
                            .instrument(tool_span)
                        })
                        .buffer_unordered(self.concurrency)
                        .collect::<Vec<Result<UserContent, PromptError>>>()
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;

                    run.tool_results(tool_content)?;
                }
                AgentRunStep::Done(response) => {
                    if self.max_turns > 1 {
                        tracing::info!("Depth reached: {}/{}", run.turn(), self.max_turns);
                    }

                    let usage = response.usage;
                    agent_span.record("gen_ai.completion", &response.output);
                    agent_span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                    agent_span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                    agent_span.record(
                        "gen_ai.usage.cache_read.input_tokens",
                        usage.cached_input_tokens,
                    );
                    agent_span.record(
                        "gen_ai.usage.cache_creation.input_tokens",
                        usage.cache_creation_input_tokens,
                    );
                    agent_span.record(
                        "gen_ai.usage.tool_use_prompt_tokens",
                        usage.tool_use_prompt_tokens,
                    );
                    agent_span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);

                    if let Some((memory, id)) = memory_handle.as_ref()
                        && let Err(err) = memory
                            .append(id, response.messages.clone().unwrap_or_default())
                            .await
                    {
                        tracing::warn!(
                            error = %err,
                            conversation_id = %id,
                            "conversation memory append failed; returning model response anyway"
                        );
                    }

                    return Ok(response);
                }
            }
        }
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
pub struct TypedPromptRequest<T, S, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    S: PromptType,
    M: CompletionModel,
    P: PromptHook<M>,
{
    inner: PromptRequest<S, M, P>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, M, P> TypedPromptRequest<T, Standard, M, P>
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
        // Typed prompts deserialize the model's final string, so they pin
        // `Native` structured output to keep the typed API's behavior unchanged
        // across all providers (#1928). Routing the typed path through `Tool`
        // output mode for tool-using agents on non-composing providers is a
        // follow-up; use the untyped `output_schema`/`output_mode` API for
        // tool-composing structured output today.
        inner.output_mode = OutputMode::Native;
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S, M, P> TypedPromptRequest<T, S, M, P>
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
    pub fn extended_details(self) -> TypedPromptRequest<T, Extended, M, P> {
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

    /// Set the retry budget for invalid tool-call recovery.
    ///
    /// Invalid tool-call retries also consume normal multi-turn depth.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.inner = self.inner.max_invalid_tool_call_retries(retries);
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
    pub fn with_history<H, U>(mut self, history: H) -> Self
    where
        H: IntoIterator<Item = U>,
        U: Into<Message>,
    {
        self.inner = self.inner.with_history(history);
        self
    }

    /// Set the conversation id used to load and persist memory for this request.
    ///
    /// Overrides any default conversation id set on the agent. If memory is not
    /// configured on the agent, this has no effect.
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.inner = self.inner.conversation(id);
        self
    }

    /// Disable conversation memory for this request.
    ///
    /// History will neither be loaded from nor saved to the agent's memory backend.
    pub fn without_memory(mut self) -> Self {
        self.inner = self.inner.without_memory();
        self
    }

    /// Attach a per-request hook for tool call events.
    ///
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> TypedPromptRequest<T, S, M, P2>
    where
        P2: PromptHook<M>,
    {
        TypedPromptRequest {
            inner: self.inner.with_hook(hook),
            _phantom: std::marker::PhantomData,
        }
    }
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

impl<T, M, P> TypedPromptRequest<T, Standard, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
    P: PromptHook<M>,
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

impl<T, M, P> TypedPromptRequest<T, Extended, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend,
    M: CompletionModel,
    P: PromptHook<M>,
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

impl<T, M, P> IntoFuture for TypedPromptRequest<T, Standard, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type Output = Result<T, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

impl<T, M, P> IntoFuture for TypedPromptRequest<T, Extended, M, P>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static,
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type Output = Result<TypedPromptResponse<T>, StructuredOutputError>;
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionCall, PromptResponse, TypedPromptResponse};
    use crate::{
        agent::{
            AgentBuilder,
            prompt_request::hooks::{
                HookAction, InvalidToolCallContext, InvalidToolCallHookAction, PromptHook,
                ToolCallHookAction,
            },
        },
        completion::{
            AssistantContent, CompletionError, CompletionModel, CompletionRequest, Message, Prompt,
            PromptError, StructuredOutputError, ToolDefinition, TypedPrompt, Usage,
        },
        message::{Text, ToolCall, ToolChoice, ToolFunction, UserContent},
        test_utils::{
            AppendFailingMemory, CountingMemory, FailingMemory, MockAddTool, MockCompletionModel,
            MockOperationArgs, MockSubtractTool, MockToolError, MockTurn,
        },
        tool::Tool,
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

    impl PromptHook<MockCompletionModel> for PanicOnUnknownToolHook {
        async fn on_completion_response(
            &self,
            _prompt: &Message,
            _response: &crate::completion::CompletionResponse<
                <MockCompletionModel as CompletionModel>::Response,
            >,
        ) -> HookAction {
            panic!("unknown tool response should fail before response hooks run")
        }

        async fn on_tool_call(
            &self,
            _tool_name: &str,
            _tool_call_id: Option<String>,
            _internal_call_id: &str,
            _args: &str,
        ) -> ToolCallHookAction {
            panic!("unknown tool call should fail before tool hooks run")
        }
    }

    #[derive(Clone)]
    struct PanicOnToolCallHook;

    impl PromptHook<MockCompletionModel> for PanicOnToolCallHook {
        async fn on_tool_call(
            &self,
            _tool_name: &str,
            _tool_call_id: Option<String>,
            _internal_call_id: &str,
            _args: &str,
        ) -> ToolCallHookAction {
            panic!("recovered invalid turn should not invoke normal tool hooks")
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiAndPanicOnToolCallHook;

    impl PromptHook<MockCompletionModel> for SkipDefaultApiAndPanicOnToolCallHook {
        async fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> InvalidToolCallHookAction {
            SkipDefaultApiHook.on_invalid_tool_call(context).await
        }

        async fn on_tool_call(
            &self,
            tool_name: &str,
            tool_call_id: Option<String>,
            internal_call_id: &str,
            args: &str,
        ) -> ToolCallHookAction {
            PanicOnToolCallHook
                .on_tool_call(tool_name, tool_call_id, internal_call_id, args)
                .await
        }
    }

    #[derive(Clone)]
    struct RepairDefaultApiHook;

    impl PromptHook<MockCompletionModel> for RepairDefaultApiHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl std::future::Future<Output = InvalidToolCallHookAction> + Send {
            let tool_name = context.tool_name.clone();
            async move {
                assert_eq!(tool_name, "default_api");
                InvalidToolCallHookAction::repair("add")
            }
        }
    }

    #[derive(Clone)]
    struct RepairToSubtractHook;

    impl PromptHook<MockCompletionModel> for RepairToSubtractHook {
        async fn on_invalid_tool_call(
            &self,
            _context: &InvalidToolCallContext,
        ) -> InvalidToolCallHookAction {
            InvalidToolCallHookAction::repair("subtract")
        }
    }

    #[derive(Clone)]
    struct RetryDefaultApiHook;

    impl PromptHook<MockCompletionModel> for RetryDefaultApiHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl std::future::Future<Output = InvalidToolCallHookAction> + Send {
            let allowed_tools = context.allowed_tools.clone();
            async move {
                InvalidToolCallHookAction::retry(format!(
                    "Use one of these tools instead: {allowed_tools:?}"
                ))
            }
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiHook;

    impl PromptHook<MockCompletionModel> for SkipDefaultApiHook {
        async fn on_invalid_tool_call(
            &self,
            _context: &InvalidToolCallContext,
        ) -> InvalidToolCallHookAction {
            InvalidToolCallHookAction::skip("default_api is not available")
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

    impl PromptHook<MockCompletionModel> for RecordingInvalidToolCallHook {
        async fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> InvalidToolCallHookAction {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .push(context.clone());
            InvalidToolCallHookAction::fail()
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

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
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
            .with_hook(PanicOnUnknownToolHook)
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
            .with_hook(invalid_hook.clone())
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
            .with_hook(PanicOnUnknownToolHook)
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
            .with_hook(PanicOnUnknownToolHook)
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
            .with_hook(RepairDefaultApiHook)
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
            .with_hook(RetryDefaultApiHook)
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
            .with_hook(RetryDefaultApiHook)
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
            .with_hook(SkipDefaultApiAndPanicOnToolCallHook)
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
            .with_hook(RetryDefaultApiHook)
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
            .with_hook(SkipDefaultApiHook)
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
            .with_hook(SkipDefaultApiHook)
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
            .with_hook(RepairToSubtractHook)
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
            .with_hook(RepairDefaultApiHook)
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
            .with_hook(SkipDefaultApiHook)
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
            .with_hook(PanicOnUnknownToolHook)
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
            .with_hook(RepairDefaultApiHook)
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
            .with_hook(RetryDefaultApiHook)
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
            .with_hook(RetryDefaultApiHook)
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
            .with_history(vec![Message::user("from-caller")])
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
            .conversation_id("default-thread")
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
            .conversation_id("t1")
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
            Err(PromptError::CompletionError(CompletionError::RequestError(err))) => {
                let msg = format!("{err}");
                assert!(msg.contains("load boom"), "got: {msg}");
            }
            other => panic!("expected PromptError::CompletionError(RequestError), got {other:?}"),
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
