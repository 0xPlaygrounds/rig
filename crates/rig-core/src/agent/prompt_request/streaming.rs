use crate::{
    OneOrMany,
    agent::completion::{DynamicContextStore, build_completion_request},
    agent::prompt_request::{HookAction, hooks::PromptHook},
    completion::{Document, GetTokenUsage},
    json_utils,
    memory::ConversationMemory,
    message::{AssistantContent, ToolChoice, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent},
    tool::server::{ToolServerError, ToolServerHandle},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tracing::info_span;
use tracing_futures::Instrument;

use super::{
    CompletionCall, ToolCallHookAction, reported_usage, unknown_tool_call_error,
    validate_registered_tool_call,
};
use crate::{
    agent::Agent,
    completion::{
        CompletionError, CompletionModel, PromptError, ToolCallNameValidator, UnknownToolCallError,
    },
    message::{Message, Text},
    tool::ToolSetError,
};

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[non_exhaustive]
pub enum MultiTurnStreamItem<R> {
    /// A streamed assistant content item.
    StreamAssistantItem(StreamedAssistantContent<R>),
    /// A streamed user content item (mostly for tool results).
    StreamUserItem(StreamedUserContent),
    /// Details for one successfully completed completion request made by this agent stream.
    ///
    /// This is emitted when a provider call finishes. Usage is the provider's
    /// final usage for that completion request when available; it is not
    /// incremental per streamed token.
    ///
    /// ```rust,ignore
    /// match item {
    ///     MultiTurnStreamItem::CompletionCall(completion_call) => {
    ///         let context_tokens = completion_call.usage.map(|usage| usage.input_tokens);
    ///     }
    ///     _ => {}
    /// }
    /// ```
    CompletionCall(CompletionCall),
    /// The final result from the stream.
    FinalResponse(FinalResponse),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinalResponse {
    /// Structured assistant content for the final turn.
    content: OneOrMany<AssistantContent>,
    /// Concatenated assistant text for the final turn.
    /// This is empty only when the turn completed without emitting any text.
    response: String,
    aggregated_usage: crate::completion::Usage,
    /// Successfully completed completion requests made by this agent stream.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    completion_calls: Vec<CompletionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<Message>>,
}

impl FinalResponse {
    pub fn empty() -> Self {
        Self::new(
            OneOrMany::one(AssistantContent::text("")),
            crate::completion::Usage::new(),
            None,
        )
    }

    pub fn new(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        let response = assistant_text_from_choice(&content);
        Self {
            content,
            response,
            aggregated_usage,
            completion_calls: Vec::new(),
            history,
        }
    }

    /// Returns the concatenated assistant text for the final turn.
    pub fn response(&self) -> &str {
        &self.response
    }

    /// Returns the structured assistant content for the final turn.
    pub fn content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    /// Returns the structured assistant content for the final turn.
    pub fn assistant_content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    pub fn usage(&self) -> crate::completion::Usage {
        self.aggregated_usage
    }

    /// Returns successfully completed completion requests made by this agent stream, with usage when available.
    ///
    /// Each entry represents one provider completion request. When present,
    /// usage is a whole-request provider snapshot, not incremental usage per
    /// streamed token. Streaming providers may omit usage for some calls; those
    /// calls have an entry with `None` usage.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    pub fn history(&self) -> Option<&[Message]> {
        self.history.as_deref()
    }
}

impl<R> MultiTurnStreamItem<R> {
    pub(crate) fn stream_item(item: StreamedAssistantContent<R>) -> Self {
        Self::StreamAssistantItem(item)
    }

    pub fn final_response(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(FinalResponse::new(content, aggregated_usage, None))
    }

    pub fn final_response_with_history(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(FinalResponse::new(content, aggregated_usage, history))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        completion_calls: Vec<CompletionCall>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let mut response = FinalResponse::new(content, aggregated_usage, history);
        response.completion_calls = completion_calls;
        Self::FinalResponse(response)
    }
}

fn merge_reasoning_blocks(
    accumulated_reasoning: &mut Vec<crate::message::Reasoning>,
    incoming: &crate::message::Reasoning,
) {
    let ids_match = |existing: &crate::message::Reasoning| {
        matches!(
            (&existing.id, &incoming.id),
            (Some(existing_id), Some(incoming_id)) if existing_id == incoming_id
        )
    };

    if let Some(existing) = accumulated_reasoning
        .iter_mut()
        .rev()
        .find(|existing| ids_match(existing))
    {
        existing.content.extend(incoming.content.clone());
    } else {
        accumulated_reasoning.push(incoming.clone());
    }
}

/// Build full history for error reporting (input + new messages).
fn build_full_history(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().cloned().chain(new_messages).collect()
}

/// Combine input history with new messages for building completion requests.
fn build_history_for_request(
    chat_history: Option<&[Message]>,
    new_messages: &[Message],
) -> Vec<Message> {
    let input = chat_history.unwrap_or(&[]);
    input.iter().chain(new_messages.iter()).cloned().collect()
}

async fn cancelled_prompt_error(
    chat_history: Option<&[Message]>,
    new_messages: Vec<Message>,
    reason: String,
) -> StreamingError {
    StreamingError::Prompt(
        PromptError::prompt_cancelled(build_full_history(chat_history, new_messages), reason)
            .into(),
    )
}

fn tool_result_to_user_message(
    id: String,
    call_id: Option<String>,
    tool_result: String,
) -> Message {
    let content = ToolResultContent::from_tool_output(tool_result);
    let user_content = match call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(id, call_id, content),
        None => UserContent::tool_result(id, content),
    };

    Message::User {
        content: OneOrMany::one(user_content),
    }
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

fn assistant_text_items_from_choice(choice: &OneOrMany<AssistantContent>) -> Vec<AssistantContent> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => (!text.text.is_empty()
                || text.additional_params.is_some())
            .then(|| AssistantContent::Text(text.clone())),
            _ => None,
        })
        .collect()
}

fn is_empty_assistant_choice(choice: &OneOrMany<AssistantContent>) -> bool {
    choice.len() == 1
        && matches!(
            choice.first(),
            AssistantContent::Text(text)
                if text.text.is_empty() && text.additional_params.is_none()
        )
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
    #[error("UnknownToolCall: {0}")]
    UnknownToolCall(UnknownToolCallError),
}

/// Surface [`crate::memory::ConversationMemory`] failures through the existing
/// [`CompletionError::RequestError`] variant so adding memory support does not
/// require a new top-level [`StreamingError`] arm.
impl From<crate::memory::MemoryError> for StreamingError {
    fn from(err: crate::memory::MemoryError) -> Self {
        Self::Completion(CompletionError::RequestError(Box::new(err)))
    }
}

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// If you expect to continuously call tools, you will want to ensure you use the `.multi_turn()`
/// argument to add more turns as by default, it is 0 (meaning only 1 tool round-trip). Otherwise,
/// attempting to await (which will send the prompt request) can potentially return
/// [`crate::completion::request::PromptError::MaxTurnsError`] if the agent decides to call tools
/// back to back.
pub struct StreamingPromptRequest<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history provided by the caller.
    chat_history: Option<Vec<Message>>,
    /// Maximum Turns for multi-turn conversations (0 means no multi-turn)
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
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
    /// Optional per-request hook for events
    hook: Option<P>,
    /// Optional conversation memory backend cloned from the agent.
    memory: Option<Arc<dyn ConversationMemory>>,
    /// Optional conversation id used for loading and saving memory.
    conversation_id: Option<String>,
}

impl<M, P> StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend + GetTokenUsage,
    P: PromptHook<M>,
{
    /// Create a new StreamingPromptRequest with the given prompt and model.
    /// Note: This creates a request without an agent hook. Use `from_agent` to include the agent's hook.
    pub fn new(agent: Arc<Agent<M>>, prompt: impl Into<Message>) -> StreamingPromptRequest<M, ()> {
        StreamingPromptRequest {
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
            output_schema: agent.output_schema.clone(),
            hook: None,
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
        }
    }

    /// Create a new StreamingPromptRequest from an agent, cloning the agent's data and default hook.
    pub fn from_agent<P2>(
        agent: &Agent<M, P2>,
        prompt: impl Into<Message>,
    ) -> StreamingPromptRequest<M, P2>
    where
        P2: PromptHook<M>,
    {
        StreamingPromptRequest {
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
            output_schema: agent.output_schema.clone(),
            hook: agent.hook.clone(),
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
        }
    }

    fn agent_name(&self) -> &str {
        self.agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Set the maximum Turns for multi-turn conversations (ie, the maximum number of turns an LLM can have calling tools before writing a text response).
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxTurnsError`].
    pub fn multi_turn(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// Add chat history to the prompt request.
    ///
    /// When history is provided, the final [`FinalResponse`] will include the
    /// updated chat history (original messages + new user prompt + assistant response).
    /// ```ignore
    /// let mut stream = agent
    ///     .stream_prompt("Hello")
    ///     .with_history(vec![])
    ///     .await;
    /// // ... consume stream ...
    /// // Access updated history from FinalResponse::history()
    /// ```
    pub fn with_history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.chat_history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Attach a per-request hook for tool call events.
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> StreamingPromptRequest<M, P2>
    where
        P2: PromptHook<M>,
    {
        StreamingPromptRequest {
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
            output_schema: self.output_schema,
            hook: Some(hook),
            memory: self.memory,
            conversation_id: self.conversation_id,
        }
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

    async fn send(self) -> StreamingResult<M::StreamingResponse> {
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

        let prompt = self.prompt;
        if let Some(text) = prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        // Clone fields needed inside the stream
        let model = self.model.clone();
        let preamble = self.preamble.clone();
        let static_context = self.static_context.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let additional_params = self.additional_params.clone();
        let tool_server_handle = self.tool_server_handle.clone();
        let dynamic_context = self.dynamic_context.clone();
        let tool_choice = self.tool_choice.clone();
        let agent_name = self.agent_name.clone();
        // When the caller passes explicit history, memory is fully bypassed for
        // this request (no load AND no save). Otherwise, if a memory backend and
        // conversation id are both configured, load prior history; if either is
        // missing, behave as if no memory is configured.
        let (chat_history, memory_handle) = match self.chat_history {
            Some(history) => (Some(history), None),
            None => match (self.memory, self.conversation_id) {
                (Some(memory), Some(id)) => match memory.load(&id).await {
                    Ok(loaded) => (Some(loaded), Some((memory, id))),
                    Err(err) => {
                        let stream = async_stream::stream! {
                            yield Err(StreamingError::from(err));
                        };
                        return Box::pin(stream);
                    }
                },
                _ => (None, None),
            },
        };
        let has_history = chat_history.is_some();
        let mut new_messages: Vec<Message> = vec![prompt.clone()];

        let mut current_max_turns = 0;
        let mut last_prompt_error = String::new();

        let mut text_delta_response = String::new();
        let mut saw_text_this_turn = false;
        let mut max_turns_reached = false;
        let output_schema = self.output_schema;

        let mut aggregated_usage = crate::completion::Usage::new();
        let mut completion_calls = Vec::new();
        let mut completion_call_index = 0;

        // NOTE: We use .instrument(agent_span) instead of span.enter() to avoid
        // span context leaking to other concurrent tasks. Using span.enter() inside
        // async_stream::stream! holds the guard across yield points, which causes
        // thread-local span context to leak when other tasks run on the same thread.
        // See: https://docs.rs/tracing/latest/tracing/span/struct.Span.html#in-asynchronous-code
        // See also: https://github.com/rust-lang/rust-clippy/issues/8722
        let stream = async_stream::stream! {
            'outer: loop {
                let Some((current_prompt_ref, previous_messages)) = new_messages.split_last() else {
                    yield Err(cancelled_prompt_error(
                        chat_history.as_deref(),
                        new_messages.clone(),
                        "streaming loop lost its pending prompt".to_string(),
                    ).await);
                    break 'outer;
                };
                let current_prompt = current_prompt_ref.clone();

                if current_max_turns > self.max_turns + 1 {
                    last_prompt_error = current_prompt.rag_text().unwrap_or_default();
                    max_turns_reached = true;
                    break;
                }

                current_max_turns += 1;

                if self.max_turns > 1 {
                    tracing::info!(
                        "Current conversation Turns: {}/{}",
                        current_max_turns,
                        self.max_turns
                    );
                }

                let history_snapshot: Vec<Message> = build_history_for_request(
                    chat_history.as_deref(),
                    previous_messages,
                );

                if let Some(ref hook) = self.hook
                    && let HookAction::Terminate { reason } =
                        hook.on_completion_call(&current_prompt, &history_snapshot).await
                {
                    yield Err(
                        cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason)
                            .await,
                    );
                    break 'outer;
                }

                let chat_stream_span = info_span!(
                    target: "rig::agent_chat",
                    parent: tracing::Span::current(),
                    "chat_streaming",
                    gen_ai.operation.name = "chat",
                    gen_ai.agent.name = agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                    gen_ai.system_instructions = preamble,
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

                let completion_request = build_completion_request(
                    &model,
                    current_prompt.clone(),
                    &history_snapshot,
                    preamble.as_deref(),
                    &static_context,
                    temperature,
                    max_tokens,
                    additional_params.as_ref(),
                    tool_choice.as_ref(),
                    &tool_server_handle,
                    &dynamic_context,
                    output_schema.as_ref(),
                )
                .await?
                .build();
                let tool_call_validator =
                    ToolCallNameValidator::from_completion_request("rig", &completion_request);
                let mut stream = tracing::Instrument::instrument(
                    model.stream(completion_request),
                    chat_stream_span
                )
                .await?;

                let call_index = completion_call_index;
                completion_call_index += 1;
                let mut current_call_usage = None;
                let mut completion_call_emitted = false;
                let mut tool_calls = vec![];
                let mut tool_results = vec![];
                let mut accumulated_reasoning: Vec<rig::message::Reasoning> = vec![];
                // Kept separate from accumulated_reasoning so providers requiring
                // signatures (e.g. Anthropic) never see unsigned blocks.
                let mut pending_reasoning_delta_text = String::new();
                let mut pending_reasoning_delta_id: Option<String> = None;
                let mut saw_tool_call_this_turn = false;

                while let Some(content) = stream.next().await {
                    match content {
                        Ok(StreamedAssistantContent::Text(text)) => {
                            if !saw_text_this_turn {
                                text_delta_response.clear();
                                saw_text_this_turn = true;
                            }
                            text_delta_response.push_str(&text.text);
                            if let Some(ref hook) = self.hook &&
                                let HookAction::Terminate { reason } = hook.on_text_delta(&text.text, &text_delta_response).await {
                                    yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    break 'outer;
                            }

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Text(text)));
                        },
                        Ok(StreamedAssistantContent::ToolCall { tool_call, internal_call_id }) => {
                            let tool_args = json_utils::value_to_json_string(&tool_call.function.arguments);
                            if let Err(error) = tool_call_validator.validate_tool_call(&tool_call) {
                                yield Err(StreamingError::Completion(error));
                                break 'outer;
                            }
                            if let Err(error) = validate_registered_tool_call(&tool_server_handle, &tool_call).await {
                                yield Err(StreamingError::UnknownToolCall(error));
                                break 'outer;
                            }

                            let tool_span = info_span!(
                                parent: tracing::Span::current(),
                                "execute_tool",
                                gen_ai.operation.name = "execute_tool",
                                gen_ai.tool.type = "function",
                                gen_ai.tool.name = tracing::field::Empty,
                                gen_ai.tool.call.id = tracing::field::Empty,
                                gen_ai.tool.call.arguments = tracing::field::Empty,
                                gen_ai.tool.call.result = tracing::field::Empty
                            );

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ToolCall { tool_call: tool_call.clone(), internal_call_id: internal_call_id.clone() }));

                            let tc_result = async {
                                let tool_span = tracing::Span::current();
                                if let Some(ref hook) = self.hook {
                                    let action = hook
                                        .on_tool_call(&tool_call.function.name, tool_call.call_id.clone(), &internal_call_id, &tool_args)
                                        .await;

                                    if let ToolCallHookAction::Terminate { reason } = action {
                                        return Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    }

                                    if let ToolCallHookAction::Skip { reason } = action {
                                        // Tool execution rejected, return rejection message as tool result
                                        tracing::info!(
                                            tool_name = tool_call.function.name.as_str(),
                                            reason = reason,
                                            "Tool call rejected"
                                        );
                                        let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());
                                        tool_calls.push(tool_call_msg);
                                        tool_results.push((tool_call.id.clone(), tool_call.call_id.clone(), reason.clone()));
                                        saw_tool_call_this_turn = true;
                                        return Ok(reason);
                                    }
                                }

                                tool_span.record("gen_ai.tool.name", &tool_call.function.name);
                                tool_span.record("gen_ai.tool.call.arguments", &tool_args);

                                let tool_result = match
                                tool_server_handle.call_tool(&tool_call.function.name, &tool_args).await {
                                    Ok(thing) => thing,
                                    Err(ToolServerError::ToolsetError(
                                        ToolSetError::ToolNotFoundError(name),
                                    )) => {
                                        let error = unknown_tool_call_error(
                                            &tool_server_handle,
                                            name,
                                            &tool_call,
                                        )
                                        .await;
                                        return Err(StreamingError::UnknownToolCall(error));
                                    }
                                    Err(e) => {
                                        tracing::warn!("Error while calling tool: {e}");
                                        e.to_string()
                                    }
                                };

                                tool_span.record("gen_ai.tool.call.result", &tool_result);

                                if let Some(ref hook) = self.hook &&
                                    let HookAction::Terminate { reason } =
                                    hook.on_tool_result(
                                        &tool_call.function.name,
                                        tool_call.call_id.clone(),
                                        &internal_call_id,
                                        &tool_args,
                                        &tool_result.to_string()
                                    )
                                    .await {
                                        return Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    }

                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id.clone(), tool_call.call_id.clone(), tool_result.clone()));

                                saw_tool_call_this_turn = true;
                                Ok(tool_result)
                            }.instrument(tool_span).await;

                            match tc_result {
                                Ok(text) => {
                                    let tr = ToolResult { id: tool_call.id, call_id: tool_call.call_id, content: ToolResultContent::from_tool_output(text) };
                                    yield Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult{ tool_result: tr, internal_call_id }));
                                }
                                Err(e) => {
                                    yield Err(e);
                                    break 'outer;
                                }
                            }
                        },
                        Ok(StreamedAssistantContent::ToolCallDelta { id, internal_call_id, content }) => {
                            if let rig::streaming::ToolCallDeltaContent::Name(name) = &content
                                && let Err(error) = tool_call_validator.validate(name)
                            {
                                yield Err(StreamingError::Completion(error));
                                break 'outer;
                            }

                            if let Some(ref hook) = self.hook {
                                let (name, delta) = match &content {
                                    rig::streaming::ToolCallDeltaContent::Name(n) => {
                                        (Some(n.as_str()), "")
                                    }
                                    rig::streaming::ToolCallDeltaContent::Delta(d) => {
                                        (None, d.as_str())
                                    }
                                };

                                if let HookAction::Terminate { reason } = hook.on_tool_call_delta(&id, &internal_call_id, name, delta)
                                .await {
                                    yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    break 'outer;
                                }
                            }

                            yield Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCallDelta { id, internal_call_id, content }));
                        }
                        Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                            // Accumulate reasoning for inclusion in chat history with tool calls.
                            // OpenAI Responses API requires reasoning items to be sent back
                            // alongside function_call items in multi-turn conversations.
                            merge_reasoning_blocks(&mut accumulated_reasoning, &reasoning);
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Reasoning(reasoning)));
                        },
                        Ok(StreamedAssistantContent::ReasoningDelta { reasoning, id }) => {
                            // Deltas lack signatures/encrypted content that full
                            // blocks carry; mixing them into accumulated_reasoning
                            // causes Anthropic to reject with "signature required".
                            pending_reasoning_delta_text.push_str(&reasoning);
                            if pending_reasoning_delta_id.is_none() {
                                pending_reasoning_delta_id = id.clone();
                            }
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ReasoningDelta { reasoning, id }));
                        },
                        Ok(StreamedAssistantContent::Final(final_resp)) => {
                            if let Some(usage) = final_resp.token_usage() {
                                current_call_usage = reported_usage(usage);
                            }
                            if let Some(usage) = current_call_usage {
                                aggregated_usage += usage;
                            }
                            let completion_call = CompletionCall::new(call_index, current_call_usage);
                            completion_calls.push(completion_call);
                            completion_call_emitted = true;
                            yield Ok(MultiTurnStreamItem::CompletionCall(completion_call));

                            if saw_text_this_turn {
                                if let Some(ref hook) = self.hook &&
                                     let HookAction::Terminate { reason } = hook.on_stream_completion_response_finish(&current_prompt, &final_resp).await {
                                        yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                        break 'outer;
                                    }

                                yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Final(final_resp)));
                                saw_text_this_turn = false;
                            }
                        }
                        Err(e) => {
                            yield Err(e.into());
                            break 'outer;
                        }
                    }
                }

                if !completion_call_emitted {
                    let completion_call = CompletionCall::new(call_index, current_call_usage);
                    completion_calls.push(completion_call);
                    yield Ok(MultiTurnStreamItem::CompletionCall(completion_call));
                }

                // Providers like Gemini emit thinking as incremental deltas
                // without signatures; assemble into a single block so
                // reasoning survives into the next turn's chat history.
                if accumulated_reasoning.is_empty() && !pending_reasoning_delta_text.is_empty() {
                    let mut assembled = crate::message::Reasoning::new(&pending_reasoning_delta_text);
                    if let Some(id) = pending_reasoning_delta_id.take() {
                        assembled = assembled.with_id(id);
                    }
                    accumulated_reasoning.push(assembled);
                }

                let final_turn_content = stream.choice.clone();
                let turn_text_response = assistant_text_from_choice(&final_turn_content);
                tracing::Span::current().record("gen_ai.completion", &turn_text_response);

                // Add text, reasoning, and tool calls to chat history.
                // OpenAI Responses API requires reasoning items to precede function_call items.
                if !tool_calls.is_empty() || !accumulated_reasoning.is_empty() {
                    // Text before tool calls so the model sees its own prior output.
                    let mut content_items = assistant_text_items_from_choice(&final_turn_content);

                    // Reasoning must come before tool calls (OpenAI requirement)
                    for reasoning in accumulated_reasoning.drain(..) {
                        content_items.push(rig::message::AssistantContent::Reasoning(reasoning));
                    }

                    content_items.extend(tool_calls.clone());

                    if let Some(content) = OneOrMany::from_iter_optional(content_items) {
                        new_messages.push(Message::Assistant {
                            id: stream.message_id.clone(),
                            content,
                        });
                    }
                }

                for (id, call_id, tool_result) in tool_results {
                    new_messages.push(tool_result_to_user_message(id, call_id, tool_result));
                }

                if !saw_tool_call_this_turn {
                    // Add user message and assistant response to history before finishing
                    if !is_empty_assistant_choice(&final_turn_content) {
                        new_messages.push(Message::Assistant {
                            id: stream.message_id.clone(),
                            content: final_turn_content.clone(),
                        });
                    } else {
                        tracing::warn!(
                            agent_name = agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                            message_id = ?stream.message_id,
                            "Streaming turn completed without assistant text; final response will be empty"
                        );
                    }

                    let current_span = tracing::Span::current();
                    current_span.record("gen_ai.usage.input_tokens", aggregated_usage.input_tokens);
                    current_span.record("gen_ai.usage.output_tokens", aggregated_usage.output_tokens);
                    current_span.record("gen_ai.usage.cache_read.input_tokens", aggregated_usage.cached_input_tokens);
                    current_span.record("gen_ai.usage.cache_creation.input_tokens", aggregated_usage.cache_creation_input_tokens);
                    current_span.record("gen_ai.usage.reasoning_tokens", aggregated_usage.reasoning_tokens);
                    tracing::info!("Agent multi-turn stream finished");
                    if let Some((memory, id)) = memory_handle.as_ref()
                        && let Err(err) = memory.append(id, new_messages.clone()).await
                    {
                        tracing::warn!(
                            error = %err,
                            conversation_id = %id,
                            "conversation memory append failed; yielding final response anyway"
                        );
                    }
                    let final_messages: Option<Vec<Message>> = if has_history {
                        Some(new_messages.clone())
                    } else {
                        None
                    };
                    yield Ok(MultiTurnStreamItem::final_response_with_completion_calls(
                        final_turn_content,
                        aggregated_usage,
                        completion_calls,
                        final_messages,
                    ));
                    break;
                }
            }

            if max_turns_reached {
                yield Err(Box::new(PromptError::MaxTurnsError {
                    max_turns: self.max_turns,
                    chat_history: build_full_history(chat_history.as_deref(), new_messages.clone()).into(),
                    prompt: Box::new(last_prompt_error.clone().into()),
                }).into());
            }
        };

        Box::pin(stream.instrument(agent_span))
    }
}

impl<M, P> IntoFuture for StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    P: PromptHook<M> + 'static,
{
    type Output = StreamingResult<M::StreamingResponse>; // what `.await` returns
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        // Wrap send() in a future, because send() returns a stream immediately
        Box::pin(async move { self.send().await })
    }
}

/// Helper function to stream assistant-visible completion output to stdout.
///
/// This helper prints streamed assistant text and reasoning only. Streaming
/// metadata events, such as `MultiTurnStreamItem::CompletionCall`, are not
/// printed; metadata is returned on the `FinalResponse` via accessors such as
/// `FinalResponse::completion_calls`.
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text, .. },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
            _ => {}
        }
    }

    Ok(final_res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::client::ProviderClient;
    use crate::client::completion::CompletionClient;
    use crate::completion::{CompletionRequest, ToolCallValidationReason, Usage};
    use crate::message::{
        AssistantContent, DocumentSourceKind, ImageMediaType, Message, ReasoningContent,
        ToolChoice, ToolResultContent, UserContent,
    };
    use crate::providers::anthropic;
    use crate::streaming::{StreamingPrompt, ToolCallDeltaContent};
    use crate::test_utils::{
        AppendFailingMemory, FailingMemory, MockAddTool, MockCompletionModel, MockResponse,
        MockStreamEvent, MockSubtractTool,
    };
    use crate::tool::{ToolSet, server::ToolServer};
    use futures::StreamExt;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn merge_reasoning_blocks_preserves_order_and_signatures() {
        let mut accumulated = Vec::new();
        let first = crate::message::Reasoning {
            id: Some("rs_1".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: Some("sig-1".to_string()),
            }],
        };
        let second = crate::message::Reasoning {
            id: Some("rs_1".to_string()),
            content: vec![
                ReasoningContent::Text {
                    text: "step-2".to_string(),
                    signature: Some("sig-2".to_string()),
                },
                ReasoningContent::Summary("summary".to_string()),
            ],
        };

        merge_reasoning_blocks(&mut accumulated, &first);
        merge_reasoning_blocks(&mut accumulated, &second);

        assert_eq!(accumulated.len(), 1);
        let merged = accumulated.first().expect("expected accumulated reasoning");
        assert_eq!(merged.id.as_deref(), Some("rs_1"));
        assert_eq!(merged.content.len(), 3);
        assert!(matches!(
            merged.content.first(),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-1" && sig == "sig-1"
        ));
        assert!(matches!(
            merged.content.get(1),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-2" && sig == "sig-2"
        ));
    }

    #[test]
    fn merge_reasoning_blocks_keeps_distinct_ids_as_separate_items() {
        let mut accumulated = vec![crate::message::Reasoning {
            id: Some("rs_a".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: None,
            }],
        }];
        let incoming = crate::message::Reasoning {
            id: Some("rs_b".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-2".to_string(),
                signature: None,
            }],
        };

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert_eq!(
            accumulated.first().and_then(|r| r.id.as_deref()),
            Some("rs_a")
        );
        assert_eq!(
            accumulated.get(1).and_then(|r| r.id.as_deref()),
            Some("rs_b")
        );
    }

    #[test]
    fn merge_reasoning_blocks_keeps_none_ids_separate_items() {
        let mut accumulated = vec![crate::message::Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: "first".to_string(),
                signature: None,
            }],
        }];
        let incoming = crate::message::Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: "second".to_string(),
                signature: None,
            }],
        };

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert!(matches!(
            accumulated.first(),
            Some(crate::message::Reasoning {
                id: None,
                content
            }) if matches!(
                content.first(),
                Some(ReasoningContent::Text { text, .. }) if text == "first"
            )
        ));
        assert!(matches!(
            accumulated.get(1),
            Some(crate::message::Reasoning {
                id: None,
                content
            }) if matches!(
                content.first(),
                Some(ReasoningContent::Text { text, .. }) if text == "second"
            )
        ));
    }

    #[test]
    fn tool_result_to_user_message_preserves_multimodal_tool_output() {
        let message = tool_result_to_user_message(
            "tool_call_1".to_string(),
            Some("call_1".to_string()),
            serde_json::json!({
                "response": {
                    "instruction": "Use the image part to answer."
                },
                "parts": [
                    {
                        "type": "image",
                        "data": "base64data==",
                        "mimeType": "image/png"
                    }
                ]
            })
            .to_string(),
        );

        let tool_result = match message {
            Message::User { content } => match content.first() {
                UserContent::ToolResult(tool_result) => tool_result,
                other => panic!("expected tool result content, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        };

        assert_eq!(tool_result.id, "tool_call_1");
        assert_eq!(tool_result.call_id.as_deref(), Some("call_1"));
        assert_eq!(tool_result.content.len(), 2);

        let mut items = tool_result.content.iter();
        match items.next() {
            Some(ToolResultContent::Text(text)) => {
                assert!(text.text.contains("Use the image part to answer."));
            }
            other => panic!("expected structured text payload first, got {other:?}"),
        }

        match items.next() {
            Some(ToolResultContent::Image(image)) => {
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert!(matches!(
                    image.data,
                    DocumentSourceKind::Base64(ref data) if data == "base64data=="
                ));
            }
            other => panic!("expected image payload second, got {other:?}"),
        }
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) -> Result<(), String> {
        let history = request.chat_history.iter().cloned().collect::<Vec<_>>();
        if history.len() != 3 {
            return Err(format!(
                "follow-up request should contain [original user prompt, assistant tool call, user tool result]: {history:?}"
            ));
        }

        if !matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ) {
            return Err(format!(
                "follow-up request should begin with the original user prompt: {history:?}"
            ));
        }

        if !matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request is missing the assistant tool call in position 2: {history:?}"
            ));
        }

        if !matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request should end with the user tool result: {history:?}"
            ));
        }

        Ok(())
    }

    fn streaming_tool_then_text_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
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
    fn completion_calls_stream_item_serializes_and_deserializes_expected_shape() {
        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(2, Some(usage(3, 4))));

        let value = serde_json::to_value(&item).expect("serialize completion call event");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 2,
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "tool_use_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                }
            })
        );

        let item: MultiTurnStreamItem<MockResponse> =
            serde_json::from_value(value).expect("deserialize completion call event");
        match item {
            MultiTurnStreamItem::CompletionCall(call_usage) => {
                assert_eq!(call_usage, CompletionCall::new(2, Some(usage(3, 4))));
            }
            other => panic!("expected completion call event, got {other:?}"),
        }

        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(3, None));
        let value = serde_json::to_value(&item).expect("serialize missing usage event");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 3,
                "usage": null
            })
        );
    }

    #[test]
    fn final_response_serializes_completion_calls_with_missing_usage() {
        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::final_response_with_completion_calls(
                OneOrMany::one(AssistantContent::text("done")),
                usage(3, 4),
                vec![
                    CompletionCall::new(0, None),
                    CompletionCall::new(1, Some(usage(3, 4))),
                ],
                None,
            );

        let value = serde_json::to_value(&item).expect("serialize final response");

        assert_eq!(
            value.get("completionCalls"),
            Some(&serde_json::json!([
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
    }

    fn streaming_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("hello"),
            MockStreamEvent::text(" world"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn calculator_provider_params() -> serde_json::Value {
        serde_json::json!({
            "tools": [
                {
                    "type": "function",
                    "name": "calculator",
                    "description": "Calculate a result",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        })
    }

    fn citation_metadata() -> serde_json::Value {
        serde_json::json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": "Claude Shannon",
                "encrypted_index": "encrypted-reference"
            }]
        })
    }

    fn streaming_cited_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text_start(Some(citation_metadata())),
            MockStreamEvent::text("cited "),
            MockStreamEvent::text_start(None),
            MockStreamEvent::text("answer"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn streaming_cited_text_then_tool_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text_start(Some(citation_metadata())),
                MockStreamEvent::text("I need a tool. "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
    }

    fn streaming_final_only_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::final_response_with_total_tokens(1),
        ]])
    }

    #[derive(Clone)]
    struct TerminateOnStreamFinish;

    impl PromptHook<MockCompletionModel> for TerminateOnStreamFinish {
        async fn on_stream_completion_response_finish(
            &self,
            _prompt: &Message,
            _response: &<MockCompletionModel as CompletionModel>::StreamingResponse,
        ) -> HookAction {
            HookAction::terminate("stop after completion call")
        }
    }

    type RecordedToolCallDelta = (String, String, Option<String>, String);

    #[derive(Clone, Default)]
    struct RecordingToolCallDeltaHook {
        deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
    }

    impl RecordingToolCallDeltaHook {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.deltas
                .lock()
                .expect("tool call delta hook records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for RecordingToolCallDeltaHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            let deltas = self.deltas.clone();
            let event = (
                tool_call_id.to_string(),
                internal_call_id.to_string(),
                tool_name.map(str::to_string),
                tool_call_delta.to_string(),
            );

            async move {
                deltas
                    .lock()
                    .expect("tool call delta hook records mutex was poisoned")
                    .push(event);
                HookAction::cont()
            }
        }
    }

    #[derive(Clone, Default)]
    struct TerminatingToolCallDeltaHook {
        deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
    }

    impl TerminatingToolCallDeltaHook {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.deltas
                .lock()
                .expect("tool call delta hook records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for TerminatingToolCallDeltaHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            let deltas = self.deltas.clone();
            let event = (
                tool_call_id.to_string(),
                internal_call_id.to_string(),
                tool_name.map(str::to_string),
                tool_call_delta.to_string(),
            );

            async move {
                deltas
                    .lock()
                    .expect("tool call delta hook records mutex was poisoned")
                    .push(event);
                HookAction::terminate("stop on tool call delta")
            }
        }
    }

    fn text_metadata(content: &OneOrMany<AssistantContent>) -> Option<&serde_json::Value> {
        content.iter().find_map(|item| match item {
            AssistantContent::Text(text) => text.additional_params.as_ref(),
            _ => None,
        })
    }

    #[tokio::test]
    async fn stream_prompt_continues_after_tool_call_turn() {
        let model = streaming_tool_then_text_model();
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut final_text = String::new();
        let mut final_response_text = None;
        let mut final_history = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    final_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    saw_final_response = true;
                    final_response_text = Some(res.response().to_owned());
                    final_history = res.history().map(|history| history.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(saw_tool_call);
        assert!(saw_tool_result);
        assert!(saw_final_response);
        assert_eq!(final_text, "done");
        assert_eq!(final_response_text.as_deref(), Some("done"));
        let history = final_history.expect("expected final response history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "done"
                ))
        )));
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert!(validate_follow_up_tool_history(&requests[1]).is_ok());
    }

    #[tokio::test]
    async fn stream_prompt_rejects_undeclared_tool_call_without_tool_result_or_follow_up_turn() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "missing_tool",
                    serde_json::json!({"input": "value"}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be called"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).build();

        let mut stream = agent.stream_prompt("do tool work").multi_turn(3).await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => saw_tool_call = true,
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => saw_tool_result = true,
                Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert!(!saw_final_response);
        assert!(
            matches!(
                error,
                Some(StreamingError::Completion(CompletionError::InvalidToolCall(ref error)))
                    if error.provider == "rig"
                        && error.tool_name == "missing_tool"
                        && error.declared_tool_names.is_empty()
                        && error.allowed_tool_names.is_none()
                        && error.reason == ToolCallValidationReason::Undeclared
            ),
            "expected invalid tool call error, got {error:?}"
        );
        assert_eq!(recorded.requests().len(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_rejects_registered_but_disallowed_tool_before_yield_or_dispatch() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "subtract",
                    serde_json::json!({"x": 4, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be called"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent.stream_prompt("do tool work").multi_turn(3).await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => saw_tool_call = true,
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => saw_tool_result = true,
                Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert!(!saw_final_response);
        assert!(
            matches!(
                error,
                Some(StreamingError::Completion(CompletionError::InvalidToolCall(ref invalid)))
                    if invalid.provider == "rig"
                        && invalid.tool_name == "subtract"
                        && invalid.declared_tool_names
                            == vec!["add".to_string(), "subtract".to_string()]
                        && invalid.allowed_tool_names == Some(vec!["add".to_string()])
                        && invalid.reason == ToolCallValidationReason::Disallowed
            ),
            "expected invalid tool-call error, got {error:?}"
        );
        assert_eq!(recorded.requests().len(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_rejects_disallowed_tool_name_delta_before_yield() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "subtract"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ]]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent.stream_prompt("do tool work").multi_turn(3).await;
        let mut saw_tool_delta = false;
        let mut saw_final_response = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => saw_tool_delta = true,
                Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_delta);
        assert!(!saw_final_response);
        assert!(
            matches!(
                error,
                Some(StreamingError::Completion(CompletionError::InvalidToolCall(ref invalid)))
                    if invalid.provider == "rig"
                        && invalid.tool_name == "subtract"
                        && invalid.reason == ToolCallValidationReason::Disallowed
            ),
            "expected invalid tool-call delta error, got {error:?}"
        );
        assert_eq!(recorded.requests().len(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_accepts_provider_extra_function_declared_in_params() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let tool_server_handle = ToolServer::new()
            .add_tools(ToolSet::from_tools(vec![MockAddTool]))
            .run();
        let agent = AgentBuilder::new(model)
            .additional_params(serde_json::json!({
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

        let mut stream = agent.stream_prompt("do tool work").multi_turn(3).await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item.expect("stream should succeed") {
                MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                    ..
                }) => saw_tool_call = true,
                MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. }) => {
                    saw_tool_result = true
                }
                MultiTurnStreamItem::FinalResponse(response) => final_response = Some(response),
                _ => {}
            }
        }

        assert!(saw_tool_call);
        assert!(saw_tool_result);
        assert!(matches!(final_response, Some(ref response) if response.response() == "done"));
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].tools.is_empty());
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_without_hook() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let agent = AgentBuilder::new(model)
            .additional_params(calculator_provider_params())
            .build();

        let mut stream = agent.stream_prompt("stream a tool call").await;
        let mut deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("calculator".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_after_hook_continue() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordingToolCallDeltaHook::default();
        let agent = AgentBuilder::new(model)
            .additional_params(calculator_provider_params())
            .build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .with_hook(hook.clone())
            .await;
        let mut stream_deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    stream_deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            hook.observed(),
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    Some("calculator".to_string()),
                    String::new()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "{\"x\":".to_string()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "1}".to_string()
                ),
            ]
        );
        assert_eq!(
            stream_deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("calculator".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_tool_call_deltas_hook_termination_prevents_delta_emit() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = TerminatingToolCallDeltaHook::default();
        let agent = AgentBuilder::new(model)
            .additional_params(calculator_provider_params())
            .build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .with_hook(hook.clone())
            .await;
        let mut saw_delta = false;
        let mut saw_final_response = false;
        let mut error_message = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error_message = Some(err.to_string());
                    break;
                }
            }
        }

        assert_eq!(
            hook.observed(),
            vec![(
                "tool_1".to_string(),
                "internal_1".to_string(),
                Some("calculator".to_string()),
                String::new()
            )]
        );
        assert!(!saw_delta);
        assert!(!saw_final_response);
        assert!(
            error_message
                .as_deref()
                .is_some_and(|message| message.contains("PromptCancelled: stop on tool call delta")),
            "expected hook termination error, got {error_message:?}"
        );
    }

    #[tokio::test]
    async fn stream_prompt_exposes_completion_calls() {
        let first_call_usage = usage(10, 2);
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response(first_call_usage),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            completion_calls_events,
            vec![
                CompletionCall::new(0, Some(first_call_usage)),
                CompletionCall::new(1, Some(second_call_usage))
            ]
        );

        let final_response = final_response.expect("expected final response");
        assert_eq!(
            final_response.usage(),
            Usage {
                input_tokens: 35,
                output_tokens: 7,
                total_tokens: 42,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            final_response.completion_calls(),
            &[
                CompletionCall::new(0, Some(first_call_usage)),
                CompletionCall::new(1, Some(second_call_usage))
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_emits_completion_call_before_finish_hook_termination() {
        let call_usage = usage(10, 2);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(call_usage),
        ]]);
        let agent = AgentBuilder::new(model).build();

        let mut stream = agent
            .stream_prompt("say done")
            .with_hook(TerminateOnStreamFinish)
            .await;
        let mut completion_calls = Vec::new();
        let mut saw_error = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_calls.push(completion_call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    panic!("unexpected final response after hook termination: {response:?}");
                }
                Ok(_) => {}
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }

        assert_eq!(
            completion_calls,
            vec![CompletionCall::new(0, Some(call_usage))]
        );
        assert!(saw_error);
    }

    #[tokio::test]
    async fn stream_prompt_completion_calls_records_unreported_usage() {
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let expected_usage = vec![
            CompletionCall::new(0, None),
            CompletionCall::new(1, Some(second_call_usage)),
        ];
        assert_eq!(completion_calls_events, expected_usage);

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.completion_calls(), expected_usage.as_slice());
    }

    #[tokio::test]
    async fn final_response_matches_streamed_text_when_provider_final_is_textless() {
        let agent = AgentBuilder::new(streaming_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("say hello").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.response().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(streamed_text, "hello world");
        assert_eq!(final_response_text.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn final_response_preserves_structured_text_metadata() {
        let agent = AgentBuilder::new(streaming_cited_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("answer with citations").await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.response(), "cited answer");
        let metadata = text_metadata(final_response.content())
            .expect("expected text metadata in final content");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_history_preserves_structured_text_metadata() {
        let agent = AgentBuilder::new(streaming_cited_text_then_final_model()).build();

        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("answer with citations")
            .with_history(empty_history)
            .await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        let history = final_response
            .history()
            .expect("with_history should include final history");
        let assistant_content = history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in history");
        let metadata =
            text_metadata(assistant_content).expect("expected text metadata in assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn tool_follow_up_history_preserves_structured_text_metadata() {
        let model = streaming_cited_text_then_tool_model();
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("use a tool with citations")
            .with_history(empty_history)
            .multi_turn(3)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().collect::<Vec<_>>();
        let assistant_content = follow_up_history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in follow-up history");
        let metadata = text_metadata(assistant_content)
            .expect("expected citation metadata in follow-up assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_can_remain_empty_for_truly_textless_turns() {
        let agent = AgentBuilder::new(streaming_final_only_model()).build();

        let mut stream = agent.stream_prompt("say nothing").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.response().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(streamed_text.is_empty());
        assert_eq!(final_response_text.as_deref(), Some(""));
    }

    /// Background task that logs periodically to detect span leakage.
    /// If span leakage occurs, these logs will be prefixed with `invoke_agent{...}`.
    async fn background_logger(stop: Arc<AtomicBool>, leak_count: Arc<AtomicU32>) {
        let mut interval = tokio::time::interval(Duration::from_millis(50));
        let mut count = 0u32;

        while !stop.load(Ordering::Relaxed) {
            interval.tick().await;
            count += 1;

            tracing::event!(
                target: "background_logger",
                tracing::Level::INFO,
                count = count,
                "Background tick"
            );

            // Check if we're inside an unexpected span
            let current = tracing::Span::current();
            if !current.is_disabled() && !current.is_none() {
                leak_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::info!(target: "background_logger", total_ticks = count, "Background logger stopped");
    }

    /// Test that span context doesn't leak to concurrent tasks during streaming.
    ///
    /// This test verifies that using `.instrument()` instead of `span.enter()` in
    /// async_stream prevents thread-local span context from leaking to other tasks.
    ///
    /// Uses single-threaded runtime to force all tasks onto the same thread,
    /// making the span leak deterministic (it only occurs when tasks share a thread).
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "This requires an API key"]
    async fn test_span_context_isolation() -> anyhow::Result<()> {
        let stop = Arc::new(AtomicBool::new(false));
        let leak_count = Arc::new(AtomicU32::new(0));

        // Start background logger
        let bg_stop = stop.clone();
        let bg_leak = leak_count.clone();
        let bg_handle = tokio::spawn(async move {
            background_logger(bg_stop, bg_leak).await;
        });

        // Small delay to let background logger start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Make streaming request WITHOUT an outer span so rig creates its own invoke_agent span
        // (rig reuses current span if one exists, so we need to ensure there's no current span)
        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant.")
            .temperature(0.1)
            .max_tokens(100)
            .build();

        let mut stream = agent
            .stream_prompt("Say 'hello world' and nothing else.")
            .await;

        let mut full_content = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    full_content.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    break;
                }
                Err(e) => {
                    tracing::warn!("Error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        tracing::info!("Got response: {:?}", full_content);

        // Stop background logger
        stop.store(true, Ordering::Relaxed);
        bg_handle.await?;

        let leaks = leak_count.load(Ordering::Relaxed);
        anyhow::ensure!(
            leaks == 0,
            "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times. \
             This indicates that span.enter() is being used inside async_stream instead of .instrument()"
        );

        Ok(())
    }

    /// Test that FinalResponse contains the updated chat history when with_history is used.
    ///
    /// This verifies that:
    /// 1. FinalResponse.history() returns Some when with_history was called
    /// 2. The history contains both the user prompt and assistant response
    #[tokio::test]
    #[ignore = "This requires an API key"]
    async fn test_chat_history_in_final_response() -> anyhow::Result<()> {
        use crate::message::Message;

        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant. Keep responses brief.")
            .temperature(0.1)
            .max_tokens(50)
            .build();

        // Send streaming request with history
        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("Say 'hello' and nothing else.")
            .with_history(empty_history)
            .await;

        // Consume the stream and collect FinalResponse
        let mut response_text = String::new();
        let mut final_history = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    response_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_history = res.history().map(|h| h.to_vec());
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
                _ => {}
            }
        }

        let history = final_history
            .ok_or_else(|| anyhow::anyhow!("final response should include history"))?;

        // Should contain at least the user message
        anyhow::ensure!(
            history.iter().any(|m| matches!(m, Message::User { .. })),
            "History should contain the user message"
        );

        // Should contain the assistant response
        anyhow::ensure!(
            history
                .iter()
                .any(|m| matches!(m, Message::Assistant { .. })),
            "History should contain the assistant response"
        );

        tracing::info!(
            "History after streaming: {} messages, response: {:?}",
            history.len(),
            response_text
        );

        Ok(())
    }

    #[tokio::test]
    async fn streaming_appends_to_memory_after_final_response() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .build();

        let mut stream = agent
            .stream_prompt("hi there")
            .conversation("stream-thread")
            .await;

        let mut history_in_final = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    history_in_final = res.history().map(|h| h.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_history = history_in_final
            .expect("FinalResponse.history should be populated when memory is configured");
        assert_eq!(
            final_history.len(),
            2,
            "user prompt + assistant response in final history: {final_history:?}"
        );

        let stored = memory.load("stream-thread").await.unwrap();
        assert_eq!(stored.len(), 2, "memory should contain user + assistant");
    }

    #[tokio::test]
    async fn streaming_with_history_overrides_memory() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        memory
            .append("t1", vec![Message::user("from-memory")])
            .await
            .unwrap();

        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .build();

        let mut stream = agent
            .stream_prompt("hi")
            .conversation("t1")
            .with_history(vec![Message::user("from-caller")])
            .await;

        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let stored = memory.load("t1").await.unwrap();
        assert_eq!(
            stored.len(),
            1,
            "with_history bypasses memory; only the pre-seeded entry remains: {stored:?}"
        );
    }

    #[tokio::test]
    async fn streaming_without_memory_disables_for_request() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .conversation_id("default")
            .build();

        let mut stream = agent.stream_prompt("hi").without_memory().await;

        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let stored = memory.load("default").await.unwrap();
        assert!(stored.is_empty(), "without_memory disables save");
    }

    #[tokio::test]
    async fn streaming_load_error_yields_memory_error() {
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(FailingMemory::default())
            .build();

        let mut stream = agent.stream_prompt("hi").conversation("t1").await;

        let first = stream.next().await.expect("at least one item");
        match first {
            Err(err) => {
                let msg = format!("{err:?}");
                assert!(
                    msg.contains("Memory") || msg.contains("memory") || msg.contains("load boom"),
                    "expected memory error, got: {msg}"
                );
            }
            Ok(other) => panic!("expected memory error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_with_filter_shapes_loaded_history() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

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

        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("ok"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ]]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).memory(memory).build();

        let mut stream = agent.stream_prompt("ping").conversation("t1").await;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let received = recorded.requests()[0]
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            received.len(),
            3,
            "window-truncated history (2) + current prompt: {received:?}"
        );
    }

    #[tokio::test]
    async fn streaming_append_error_does_not_suppress_final_response() {
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(AppendFailingMemory::default())
            .build();

        let mut stream = agent.stream_prompt("hi").conversation("t1").await;

        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                saw_final = true;
                break;
            }
        }
        assert!(
            saw_final,
            "FinalResponse must be yielded even when memory.append fails"
        );
    }
}
