pub mod hooks;
pub mod streaming;

use super::{
    Agent,
    completion::{DynamicContextStore, build_completion_request},
};
use crate::{
    OneOrMany,
    completion::{
        CompletionModel, Document, Message, PromptError, ToolCallNameValidator,
        UnknownToolCallError, Usage,
    },
    json_utils,
    memory::ConversationMemory,
    message::{AssistantContent, ToolChoice, ToolResultContent, UserContent},
    tool::{
        ToolSetError,
        server::{ToolServerError, ToolServerHandle},
    },
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{StreamExt, stream};
use hooks::{HookAction, PromptHook, ToolCallHookAction};
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
    /// How many tools should be executed at the same time (1 by default).
    concurrency: usize,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
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
            concurrency: 1,
            output_schema: agent.output_schema.clone(),
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
            concurrency: self.concurrency,
            output_schema: self.output_schema,
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
    pub fn with_history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
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
            concurrency: self.concurrency,
            output_schema: self.output_schema,
            memory: self.memory,
            conversation_id: self.conversation_id,
        }
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
    /// Token usage reported for this completion request, when the provider supplied it.
    ///
    /// Rig normalizes zero-valued [`Usage`] to `None` because zero-valued usage
    /// is the existing sentinel for missing provider usage metrics.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl CompletionCall {
    /// Create details for one completion request in an agent run.
    pub fn new(call_index: usize, usage: Option<Usage>) -> Self {
        Self { call_index, usage }
    }

    pub(crate) fn from_reported_usage(call_index: usize, usage: Usage) -> Self {
        Self::new(call_index, reported_usage(usage))
    }
}

pub(crate) fn reported_usage(usage: Usage) -> Option<Usage> {
    (usage != Usage::new()).then_some(usage)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptResponse {
    pub output: String,
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run, with token usage when available.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last entry's
    /// usage, when present, to inspect the final completion request's
    /// prompt/context length.
    /// If a provider does not report token usage, its entry contains `None`.
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

    /// Returns successfully completed completion requests made by this agent run, with token usage when available.
    ///
    /// If a provider does not report token usage, its entry contains `None`.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TypedPromptResponse<T> {
    pub output: T,
    pub usage: Usage,
    /// Successfully completed completion requests made by this agent run, with token usage when available.
    ///
    /// `usage` remains the aggregate across the whole run. Use the last entry's
    /// usage, when present, to inspect the final completion request's
    /// prompt/context length.
    /// If a provider does not report token usage, its entry contains `None`.
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

    /// Returns successfully completed completion requests made by this agent run, with token usage when available.
    ///
    /// If a provider does not report token usage, its entry contains `None`.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }
}

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// Combine input history with new messages for building completion requests.
fn build_history_for_request(
    chat_history: Option<&[Message]>,
    new_messages: &[Message],
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().chain(new_messages.iter()).cloned().collect()
}

/// Build the full history for error reporting (input + new messages).
fn build_full_history(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().cloned().chain(new_messages).collect()
}

fn is_empty_assistant_turn(choice: &OneOrMany<AssistantContent>) -> bool {
    choice.len() == 1
        && matches!(
            choice.first(),
            AssistantContent::Text(text) if text.text.is_empty() && text.additional_params.is_none()
        )
}

fn assistant_text_from_choice(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn log_unknown_tool_call(error: &UnknownToolCallError) {
    tracing::warn!(
        tool_name = error.tool_name.as_str(),
        tool_call_id = error.tool_call_id.as_str(),
        call_id = ?error.call_id,
        available_tool_names = ?error.available_tool_names,
        "Model requested an unknown tool"
    );
}

pub(super) async fn validate_registered_tool_call(
    tool_server_handle: &ToolServerHandle,
    tool_call: &crate::message::ToolCall,
) -> Result<(), UnknownToolCallError> {
    tool_server_handle
        .validate_tool_call_name(
            &tool_call.function.name,
            &tool_call.id,
            tool_call.call_id.clone(),
            tool_call.function.arguments.clone(),
        )
        .await
        .inspect_err(log_unknown_tool_call)
}

pub(super) async fn unknown_tool_call_error(
    tool_server_handle: &ToolServerHandle,
    tool_name: String,
    tool_call: &crate::message::ToolCall,
) -> UnknownToolCallError {
    let error = tool_server_handle
        .unknown_tool_call_error(
            tool_name,
            &tool_call.id,
            tool_call.call_id.clone(),
            tool_call.function.arguments.clone(),
        )
        .await;
    log_unknown_tool_call(&error);
    error
}

fn validate_response_tool_calls(
    tool_call_validator: &ToolCallNameValidator,
    choice: &OneOrMany<AssistantContent>,
) -> Result<(), crate::completion::CompletionError> {
    for content in choice.iter() {
        if let AssistantContent::ToolCall(tool_call) = content {
            tool_call_validator.validate_tool_call(tool_call)?;
        }
    }

    Ok(())
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
        let mut new_messages: Vec<Message> = vec![self.prompt.clone()];

        let mut current_max_turns = 0;
        let mut usage = Usage::new();
        let mut completion_calls = Vec::new();
        let mut completion_call_index = 0;
        let current_span_id: AtomicU64 = AtomicU64::new(0);

        // We need to do at least 2 loops for 1 roundtrip (user expects normal message)
        let last_prompt = loop {
            // Get the last message (the current prompt)
            let Some((prompt_ref, history_for_current_turn)) = new_messages.split_last() else {
                return Err(PromptError::prompt_cancelled(
                    build_full_history(chat_history.as_deref(), new_messages),
                    "prompt loop lost its pending prompt",
                ));
            };
            let prompt = prompt_ref.clone();

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

            // Build history for hook callback (input + new messages except last)
            let history_for_hook =
                build_history_for_request(chat_history.as_deref(), history_for_current_turn);

            if let Some(ref hook) = self.hook
                && let HookAction::Terminate { reason } =
                    hook.on_completion_call(&prompt, &history_for_hook).await
            {
                return Err(PromptError::prompt_cancelled(
                    build_full_history(chat_history.as_deref(), new_messages),
                    reason,
                ));
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

            // Build history for completion request (input + new messages except last)
            let history_for_request =
                build_history_for_request(chat_history.as_deref(), history_for_current_turn);

            let completion_request = build_completion_request(
                &self.model,
                prompt.clone(),
                &history_for_request,
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
            .build();
            let tool_call_validator =
                ToolCallNameValidator::from_completion_request("rig", &completion_request);
            let resp = self
                .model
                .completion(completion_request)
                .instrument(chat_span.clone())
                .await?;

            completion_calls.push(CompletionCall::from_reported_usage(
                completion_call_index,
                resp.usage,
            ));
            completion_call_index += 1;
            usage += resp.usage;

            validate_response_tool_calls(&tool_call_validator, &resp.choice)?;

            if let Some(ref hook) = self.hook
                && let HookAction::Terminate { reason } =
                    hook.on_completion_response(&prompt, &resp).await
            {
                return Err(PromptError::prompt_cancelled(
                    build_full_history(chat_history.as_deref(), new_messages),
                    reason,
                ));
            }

            let tool_calls = resp
                .choice
                .iter()
                .filter(|choice| matches!(choice, AssistantContent::ToolCall(_)))
                .collect::<Vec<_>>();

            // Some providers normalize textless terminal turns into a single empty text item
            // because the generic completion response cannot represent an empty choice. Treat
            // that sentinel as "no assistant output" so it does not pollute returned history.
            if !is_empty_assistant_turn(&resp.choice) {
                new_messages.push(Message::Assistant {
                    id: resp.message_id.clone(),
                    content: resp.choice.clone(),
                });
            }

            if tool_calls.is_empty() {
                let merged_texts = assistant_text_from_choice(&resp.choice);

                if self.max_turns > 1 {
                    tracing::info!("Depth reached: {}/{}", current_max_turns, self.max_turns);
                }

                agent_span.record("gen_ai.completion", &merged_texts);
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
                agent_span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);

                if let Some((memory, id)) = memory_handle.as_ref()
                    && let Err(err) = memory.append(id, new_messages.clone()).await
                {
                    tracing::warn!(
                        error = %err,
                        conversation_id = %id,
                        "conversation memory append failed; returning model response anyway"
                    );
                }

                return Ok(PromptResponse::new(merged_texts, usage)
                    .with_messages(new_messages)
                    .with_completion_calls(completion_calls));
            }

            let hook = self.hook.clone();
            let tool_server_handle = self.tool_server_handle.clone();

            // For error handling in concurrent tool execution, we need to build full history
            let full_history_for_errors =
                build_full_history(chat_history.as_deref(), new_messages.clone());

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

                    // Clone full history for error reporting in concurrent tool execution
                    let cloned_history_for_error = full_history_for_errors.clone();

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
                            validate_registered_tool_call(&tool_server_handle, &tool_call)
                                .await
                                .map_err(PromptError::UnknownToolCall)?;
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
                            let output = match tool_server_handle.call_tool(tool_name, &args).await
                            {
                                Ok(res) => res,
                                Err(ToolServerError::ToolsetError(
                                    ToolSetError::ToolNotFoundError(name),
                                )) => {
                                    let error = unknown_tool_call_error(
                                        &tool_server_handle,
                                        name,
                                        &tool_call,
                                    )
                                    .await;
                                    return Err(PromptError::UnknownToolCall(error));
                                }
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
                        } else {
                            Err(PromptError::prompt_cancelled(
                                Vec::new(),
                                "tool execution received non-tool assistant content",
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

            let Some(content) = OneOrMany::from_iter_optional(tool_content) else {
                return Err(PromptError::prompt_cancelled(
                    build_full_history(chat_history.as_deref(), new_messages),
                    "tool execution produced no tool results",
                ));
            };

            new_messages.push(Message::User { content });
        };

        // If we reach here, we exceeded max turns without a final response
        Err(PromptError::MaxTurnsError {
            max_turns: self.max_turns,
            chat_history: build_full_history(chat_history.as_deref(), new_messages).into(),
            prompt: last_prompt.into(),
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

    /// Add concurrency to the prompt request.
    ///
    /// This will cause the agent to execute tools concurrently.
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.inner = self.inner.with_tool_concurrency(concurrency);
        self
    }

    /// Add chat history to the prompt request.
    pub fn with_history<I, H>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = H>,
        H: Into<Message>,
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

        let parsed: T = serde_json::from_str(&response)?;
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

        let parsed: T = serde_json::from_str(&response.output)?;
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
        agent::AgentBuilder,
        completion::{
            AssistantContent, CompletionError, CompletionRequest, Message, Prompt, PromptError,
            ToolCallValidationReason, TypedPrompt, Usage,
        },
        message::{Text, ToolChoice, UserContent},
        test_utils::{
            AppendFailingMemory, CountingMemory, FailingMemory, MockAddTool, MockCompletionModel,
            MockSubtractTool, MockTurn,
        },
        tool::{ToolSet, server::ToolServer},
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

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

        assert_eq!(response.output.value, "ok");
        assert_eq!(response.usage.input_tokens, 1);
        assert_eq!(response.usage.output_tokens, 2);
        assert_eq!(response.usage.total_tokens, 3);
    }

    #[test]
    fn prompt_response_serializes_completion_calls_with_missing_usage() {
        let reported_usage = usage(3, 4);
        let response = PromptResponse::new("ok", reported_usage).with_completion_calls(vec![
            CompletionCall::new(0, None),
            CompletionCall::new(1, Some(reported_usage)),
        ]);

        let value = serde_json::to_value(&response).expect("serialize prompt response");

        assert_eq!(
            value.get("completion_calls"),
            Some(&json!([
                {
                    "call_index": 0,
                    "usage": null,
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
                CompletionCall::new(0, None),
                CompletionCall::new(1, Some(reported_usage))
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
        assert_eq!(response.completion_calls(), &[CompletionCall::new(0, None)]);
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
            &[CompletionCall::new(0, Some(call_usage))]
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

    #[tokio::test]
    async fn prompt_request_stops_cleanly_on_empty_terminal_turn_after_known_tool() {
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
                CompletionCall::new(0, Some(first_call_usage)),
                CompletionCall::new(1, Some(second_call_usage))
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
    async fn prompt_request_rejects_undeclared_tool_call_without_follow_up_turn() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "missing_tool", json!({"input": "value"}))
                .with_call_id("call_1"),
            MockTurn::text("should not be called"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).build();

        let err = agent
            .prompt("do tool work")
            .max_turns(3)
            .await
            .expect_err("undeclared tool call should stop the prompt");

        assert!(
            matches!(
                err,
                PromptError::CompletionError(CompletionError::InvalidToolCall(ref error))
                    if error.provider == "rig"
                        && error.tool_name == "missing_tool"
                        && error.declared_tool_names.is_empty()
                        && error.allowed_tool_names.is_none()
                        && error.reason == ToolCallValidationReason::Undeclared
            ),
            "expected invalid tool call error, got {err:?}"
        );
        assert_eq!(recorded.requests().len(), 1);
    }

    #[tokio::test]
    async fn prompt_request_rejects_registered_but_disallowed_tool_without_dispatch() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "subtract", json!({"x": 4, "y": 2}))
                .with_call_id("call_1"),
            MockTurn::text("should not be called"),
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
            .prompt("do tool work")
            .max_turns(3)
            .await
            .expect_err("disallowed tool should stop the prompt");

        assert!(
            matches!(
                err,
                PromptError::CompletionError(CompletionError::InvalidToolCall(ref invalid))
                    if invalid.provider == "rig"
                        && invalid.tool_name == "subtract"
                        && invalid.declared_tool_names
                            == vec!["add".to_string(), "subtract".to_string()]
                        && invalid.allowed_tool_names == Some(vec!["add".to_string()])
                        && invalid.reason == ToolCallValidationReason::Disallowed
            ),
            "expected invalid tool-call error, got {err:?}"
        );
        assert_eq!(recorded.requests().len(), 1);
    }

    #[tokio::test]
    async fn prompt_request_accepts_provider_extra_function_declared_in_params() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 2, "y": 3}))
                .with_call_id("call_1"),
            MockTurn::text("done"),
        ]);
        let recorded = model.clone();
        let tool_server_handle = ToolServer::new()
            .add_tools(ToolSet::from_tools(vec![MockAddTool]))
            .run();
        let agent = AgentBuilder::new(model)
            .additional_params(json!({
                "tools": [
                    {
                        "type": "function",
                        "name": "add",
                        "description": "Add numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": { "type": "number" },
                                "y": { "type": "number" }
                            },
                            "required": ["x", "y"]
                        }
                    }
                ]
            }))
            .tool_server_handle(tool_server_handle)
            .build();

        let response = agent
            .prompt("do tool work")
            .max_turns(3)
            .await
            .expect("provider-extra function should be accepted and dispatched");

        assert_eq!(response, "done");
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].tools.is_empty());
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
