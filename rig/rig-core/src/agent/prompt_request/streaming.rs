use crate::{
    OneOrMany,
    agent::completion::{DynamicContextStore, build_completion_request},
    agent::prompt_request::{HookAction, hooks::PromptHook},
    completion::{Document, GetTokenUsage},
    json_utils,
    message::{AssistantContent, ToolChoice, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent},
    tool::server::ToolServerHandle,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{Stream, StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tracing::info_span;
use tracing_futures::Instrument;

use super::{ToolCallHookAction, normalize_tool_concurrency};
use crate::{
    agent::Agent,
    completion::{CompletionError, CompletionModel, PromptError},
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
    /// The final result from the stream.
    FinalResponse(FinalResponse),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinalResponse {
    response: String,
    aggregated_usage: crate::completion::Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<Message>>,
}

impl FinalResponse {
    pub fn empty() -> Self {
        Self {
            response: String::new(),
            aggregated_usage: crate::completion::Usage::new(),
            history: None,
        }
    }

    pub fn response(&self) -> &str {
        &self.response
    }

    pub fn usage(&self) -> crate::completion::Usage {
        self.aggregated_usage
    }

    pub fn history(&self) -> Option<&[Message]> {
        self.history.as_deref()
    }
}

impl<R> MultiTurnStreamItem<R> {
    pub(crate) fn stream_item(item: StreamedAssistantContent<R>) -> Self {
        Self::StreamAssistantItem(item)
    }

    pub fn final_response(response: &str, aggregated_usage: crate::completion::Usage) -> Self {
        Self::FinalResponse(FinalResponse {
            response: response.to_string(),
            aggregated_usage,
            history: None,
        })
    }

    pub fn final_response_with_history(
        response: &str,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(FinalResponse {
            response: response.to_string(),
            aggregated_usage,
            history,
        })
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
    let content = OneOrMany::one(ToolResultContent::text(tool_result));
    let user_content = match call_id {
        Some(call_id) => UserContent::tool_result_with_call_id(id, call_id, content),
        None => UserContent::tool_result(id, content),
    };

    Message::User {
        content: OneOrMany::one(user_content),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
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
    /// How many tools should be executed at the same time (1 by default).
    concurrency: usize,
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
            concurrency: 1,
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
            concurrency: 1,
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

    /// Add concurrency to the streaming prompt request.
    ///
    /// When set to a value greater than 1, all tool calls from a single model
    /// response are collected and executed concurrently rather than one-by-one
    /// as they arrive in the stream. Values lower than 1 are clamped to 1.
    ///
    /// Pre-execution hooks ([`PromptHook::on_tool_call`]) still run
    /// sequentially before any tool is dispatched, so a
    /// [`ToolCallHookAction::Terminate`] prevents all subsequent execution.
    /// Post-execution hooks ([`PromptHook::on_tool_result`]) run sequentially
    /// as completed tool results are consumed from the bounded executor.
    ///
    /// Tool results may arrive in completion order rather than model tool-call
    /// order, and the per-turn [`StreamedAssistantContent::Final`] item is
    /// emitted only after all tool results for that turn have been yielded.
    ///
    /// Defaults to 1 (sequential execution, matching the previous behaviour).
    pub fn with_tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = normalize_tool_concurrency(concurrency);
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
            concurrency: self.concurrency,
        }
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
        let has_history = self.chat_history.is_some();
        let chat_history = self.chat_history;
        let concurrency = self.concurrency;
        let mut new_messages: Vec<Message> = vec![prompt.clone()];

        let mut current_max_turns = 0;
        let mut last_prompt_error = String::new();

        let mut last_text_response = String::new();
        let mut is_text_response = false;
        let mut max_turns_reached = false;
        let output_schema = self.output_schema;

        let mut aggregated_usage = crate::completion::Usage::new();

        // NOTE: We use .instrument(agent_span) instead of span.enter() to avoid
        // span context leaking to other concurrent tasks. Using span.enter() inside
        // async_stream::stream! holds the guard across yield points, which causes
        // thread-local span context to leak when other tasks run on the same thread.
        // See: https://docs.rs/tracing/latest/tracing/span/struct.Span.html#in-asynchronous-code
        // See also: https://github.com/rust-lang/rust-clippy/issues/8722
        let stream = async_stream::stream! {
            let mut current_prompt = prompt.clone();

            'outer: loop {
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

                if let Some(ref hook) = self.hook {
                    let history_snapshot: Vec<Message> = build_history_for_request(chat_history.as_deref(), &new_messages[..new_messages.len().saturating_sub(1)]);
                    if let HookAction::Terminate { reason } = hook.on_completion_call(&current_prompt, &history_snapshot)
                        .await {
                        yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                        break 'outer;
                    }
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
                    gen_ai.input.messages = tracing::field::Empty,
                    gen_ai.output.messages = tracing::field::Empty,
                );

                let history_snapshot: Vec<Message> = build_history_for_request(chat_history.as_deref(), &new_messages[..new_messages.len().saturating_sub(1)]);
                let mut stream = tracing::Instrument::instrument(
                    build_completion_request(
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
                    .stream(), chat_stream_span
                )

                .await?;

                new_messages.push(current_prompt.clone());

                let mut tool_calls = vec![];
                let mut tool_results = vec![];
                let mut accumulated_reasoning: Vec<rig::message::Reasoning> = vec![];
                // Kept separate from accumulated_reasoning so providers requiring
                // signatures (e.g. Anthropic) never see unsigned blocks.
                let mut pending_reasoning_delta_text = String::new();
                let mut pending_reasoning_delta_id: Option<String> = None;
                let mut saw_tool_call_this_turn = false;
                // When concurrency > 1, tool calls are buffered here during
                // streaming and executed in parallel after the stream ends.
                let mut deferred_tool_calls: Vec<(crate::message::ToolCall, String)> = vec![];
                // Keep the provider's per-turn final item until deferred tool
                // results have been emitted so stream consumers never observe a
                // "finished" turn before the tool outputs for that turn.
                let mut deferred_final_response = None;

                while let Some(content) = stream.next().await {
                    match content {
                        Ok(StreamedAssistantContent::Text(text)) => {
                            if !is_text_response {
                                last_text_response = String::new();
                                is_text_response = true;
                            }
                            last_text_response.push_str(&text.text);
                            if let Some(ref hook) = self.hook &&
                                let HookAction::Terminate { reason } = hook.on_text_delta(&text.text, &last_text_response).await {
                                    yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    break 'outer;
                            }

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Text(text)));
                        },
                        Ok(StreamedAssistantContent::ToolCall { tool_call, internal_call_id }) => {
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ToolCall { tool_call: tool_call.clone(), internal_call_id: internal_call_id.clone() }));

                            if concurrency > 1 {
                                // Buffer for parallel execution after stream completes.
                                deferred_tool_calls.push((tool_call, internal_call_id));
                            } else {
                                // Sequential (default): execute inline, matching prior behaviour.
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

                                let tc_result = async {
                                    let tool_span = tracing::Span::current();
                                    let tool_args = json_utils::value_to_json_string(&tool_call.function.arguments);
                                    if let Some(ref hook) = self.hook {
                                        let action = hook
                                            .on_tool_call(&tool_call.function.name, tool_call.call_id.clone(), &internal_call_id, &tool_args)
                                            .await;

                                        if let ToolCallHookAction::Terminate { reason } = action {
                                            return Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                        }

                                        if let ToolCallHookAction::Skip { reason } = action {
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
                            }
                        },
                        Ok(StreamedAssistantContent::ToolCallDelta { id, internal_call_id, content }) => {
                            if let Some(ref hook) = self.hook {
                                let (name, delta) = match &content {
                                    rig::streaming::ToolCallDeltaContent::Name(n) => (Some(n.as_str()), ""),
                                    rig::streaming::ToolCallDeltaContent::Delta(d) => (None, d.as_str()),
                                };

                                if let HookAction::Terminate { reason } = hook.on_tool_call_delta(&id, &internal_call_id, name, delta)
                                .await {
                                    yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    break 'outer;
                                }
                            }
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
                            if let Some(usage) = final_resp.token_usage() { aggregated_usage += usage; };
                            if is_text_response {
                                if let Some(ref hook) = self.hook &&
                                     let HookAction::Terminate { reason } = hook.on_stream_completion_response_finish(&prompt, &final_resp).await {
                                        yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                        break 'outer;
                                    }

                                tracing::Span::current().record("gen_ai.completion", &last_text_response);
                                if deferred_tool_calls.is_empty() {
                                    yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Final(final_resp)));
                                } else {
                                    deferred_final_response = Some(final_resp);
                                }
                                is_text_response = false;
                            }
                        }
                        Err(e) => {
                            yield Err(e.into());
                            break 'outer;
                        }
                    }
                }

                // ── Parallel tool execution (concurrency > 1) ──
                // Tool calls were buffered during the stream above.
                // Execute them concurrently, honouring hooks and cancellation.
                if !deferred_tool_calls.is_empty() {
                    saw_tool_call_this_turn = true;

                    // Phase 1 – pre-execution hooks (sequential).
                    // A Terminate stops everything; a Skip returns the
                    // reason as the tool result without executing.
                    let mut approved: Vec<(crate::message::ToolCall, String, String)> = vec![];
                    for (tc, internal_id) in deferred_tool_calls.drain(..) {
                        let tool_args = json_utils::value_to_json_string(&tc.function.arguments);
                        if let Some(ref hook) = self.hook {
                            let action = hook
                                .on_tool_call(&tc.function.name, tc.call_id.clone(), &internal_id, &tool_args)
                                .await;
                            match action {
                                ToolCallHookAction::Terminate { reason } => {
                                    yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                                    break 'outer;
                                }
                                ToolCallHookAction::Skip { reason } => {
                                    tracing::info!(
                                        tool_name = tc.function.name.as_str(),
                                        reason = reason,
                                        "Tool call rejected"
                                    );
                                    tool_calls.push(AssistantContent::ToolCall(tc.clone()));
                                    tool_results.push((tc.id.clone(), tc.call_id.clone(), reason.clone()));
                                    let tr = ToolResult { id: tc.id, call_id: tc.call_id, content: ToolResultContent::from_tool_output(reason) };
                                    yield Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { tool_result: tr, internal_call_id: internal_id }));
                                    continue;
                                }
                                ToolCallHookAction::Continue => {
                                    tool_calls.push(AssistantContent::ToolCall(tc.clone()));
                                }
                            }
                        } else {
                            tool_calls.push(AssistantContent::ToolCall(tc.clone()));
                        }
                        approved.push((tc, internal_id, tool_args));
                    }

                    // Phase 2 – execute approved tools with bounded concurrency.
                    let mut results = stream::iter(approved.into_iter().map(|(tc, internal_id, tool_args)| {
                        let tool_span = info_span!(
                            parent: tracing::Span::current(),
                            "execute_tool",
                            gen_ai.operation.name = "execute_tool",
                            gen_ai.tool.r#type = "function",
                            gen_ai.tool.name = tc.function.name.as_str(),
                            gen_ai.tool.call.id = tc.id.as_str(),
                            gen_ai.tool.call.arguments = tool_args.as_str(),
                            gen_ai.tool.call.result = tracing::field::Empty
                        );
                        let name = tc.function.name.clone();
                        let args = tool_args.clone();
                        let handle = tool_server_handle.clone();
                        async move {
                            let result = match handle.call_tool(&name, &args).await {
                                Ok(r) => r,
                                Err(e) => {
                                    tracing::warn!("Error while calling tool: {e}");
                                    e.to_string()
                                }
                            };
                            tracing::Span::current().record("gen_ai.tool.call.result", &result);
                            (tc, internal_id, tool_args, result)
                        }
                        .instrument(tool_span)
                    }))
                    .buffer_unordered(concurrency);

                    // Phase 3 – post-execution hooks (sequential) & yield results.
                    while let Some((tc, internal_id, tool_args, result)) = results.next().await {
                        if let Some(ref hook) = self.hook &&
                            let HookAction::Terminate { reason } =
                            hook.on_tool_result(
                                &tc.function.name,
                                tc.call_id.clone(),
                                &internal_id,
                                &tool_args,
                                &result,
                            ).await
                        {
                            yield Err(cancelled_prompt_error(chat_history.as_deref(), new_messages.clone(), reason).await);
                            break 'outer;
                        }

                        tool_results.push((tc.id.clone(), tc.call_id.clone(), result.clone()));

                        let tr = ToolResult { id: tc.id, call_id: tc.call_id, content: ToolResultContent::from_tool_output(result) };
                        yield Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { tool_result: tr, internal_call_id: internal_id }));
                    }
                }

                if let Some(final_resp) = deferred_final_response.take() {
                    yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Final(final_resp)));
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

                // Add text, reasoning, and tool calls to chat history.
                // OpenAI Responses API requires reasoning items to precede function_call items.
                if !tool_calls.is_empty() || !accumulated_reasoning.is_empty() {
                    let mut content_items: Vec<rig::message::AssistantContent> = vec![];

                    // Text before tool calls so the model sees its own prior output
                    if !last_text_response.is_empty() {
                        content_items.push(rig::message::AssistantContent::text(&last_text_response));
                        last_text_response.clear();
                    }

                    // Reasoning must come before tool calls (OpenAI requirement)
                    for reasoning in accumulated_reasoning.drain(..) {
                        content_items.push(rig::message::AssistantContent::Reasoning(reasoning));
                    }

                    content_items.extend(tool_calls.clone());

                    if !content_items.is_empty() {
                        new_messages.push(Message::Assistant {
                            id: stream.message_id.clone(),
                            content: OneOrMany::many(content_items).expect("Should have at least one item"),
                        });
                    }
                }

                for (id, call_id, tool_result) in tool_results {
                    new_messages.push(tool_result_to_user_message(id, call_id, tool_result));
                }

                // Set the current prompt to the last message in new_messages
                current_prompt = match new_messages.pop() {
                    Some(prompt) => prompt,
                    None => unreachable!("New messages should never be empty at this point"),
                };

                if !saw_tool_call_this_turn {
                    // Add user message and assistant response to history before finishing
                    new_messages.push(current_prompt.clone());
                    if !last_text_response.is_empty() {
                        new_messages.push(Message::assistant(&last_text_response));
                    }

                    let current_span = tracing::Span::current();
                    current_span.record("gen_ai.usage.input_tokens", aggregated_usage.input_tokens);
                    current_span.record("gen_ai.usage.output_tokens", aggregated_usage.output_tokens);
                    current_span.record("gen_ai.usage.cache_read.input_tokens", aggregated_usage.cached_input_tokens);
                    current_span.record("gen_ai.usage.cache_creation.input_tokens", aggregated_usage.cache_creation_input_tokens);
                    tracing::info!("Agent multi-turn stream finished");
                    let final_messages: Option<Vec<Message>> = if has_history {
                        Some(new_messages.clone())
                    } else {
                        None
                    };
                    yield Ok(MultiTurnStreamItem::final_response_with_history(
                        &last_text_response,
                        aggregated_usage,
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

/// Helper function to stream a completion request to stdout.
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
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
    use crate::completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Prompt,
    };
    use crate::message::{AssistantContent, ReasoningContent, ToolFunction, ToolResultContent};
    use crate::providers::anthropic;
    use crate::streaming::StreamingPrompt;
    use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse};
    use crate::tool::Tool;
    use futures::StreamExt;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::{Barrier, Notify, Semaphore};

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

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct MockStreamingResponse {
        usage: crate::completion::Usage,
    }

    impl MockStreamingResponse {
        fn new(total_tokens: u64) -> Self {
            let mut usage = crate::completion::Usage::new();
            usage.total_tokens = total_tokens;
            Self { usage }
        }
    }

    impl crate::completion::GetTokenUsage for MockStreamingResponse {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            Some(self.usage)
        }
    }

    #[derive(Clone, Default)]
    struct MultiTurnMockModel {
        turn_counter: Arc<AtomicUsize>,
    }

    #[allow(refining_impl_trait)]
    impl CompletionModel for MultiTurnMockModel {
        type Response = ();
        type StreamingResponse = MockStreamingResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::default()
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            Err(CompletionError::ProviderError(
                "completion is unused in this streaming test".to_string(),
            ))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst);
            let stream = async_stream::stream! {
                if turn == 0 {
                    yield Ok(RawStreamingChoice::ToolCall(
                        RawStreamingToolCall::new(
                            "tool_call_1".to_string(),
                            "missing_tool".to_string(),
                            serde_json::json!({"input": "value"}),
                        )
                        .with_call_id("call_1".to_string()),
                    ));
                    yield Ok(RawStreamingChoice::FinalResponse(MockStreamingResponse::new(4)));
                } else {
                    yield Ok(RawStreamingChoice::Message("done".to_string()));
                    yield Ok(RawStreamingChoice::FinalResponse(MockStreamingResponse::new(6)));
                }
            };

            let pinned_stream: crate::streaming::StreamingResult<Self::StreamingResponse> =
                Box::pin(stream);
            Ok(StreamingCompletionResponse::stream(pinned_stream))
        }
    }

    #[tokio::test]
    async fn stream_prompt_continues_after_tool_call_turn() {
        let model = MultiTurnMockModel::default();
        let turn_counter = model.turn_counter.clone();
        let agent = AgentBuilder::new(model).build();

        let mut stream = agent.stream_prompt("do tool work").multi_turn(3).await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut final_text = String::new();

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
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
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
        assert_eq!(turn_counter.load(Ordering::SeqCst), 2);
    }

    #[derive(Clone)]
    struct ScriptedStreamingModel {
        turns: Arc<Vec<Vec<RawStreamingChoice<MockStreamingResponse>>>>,
        turn_counter: Arc<AtomicUsize>,
    }

    impl ScriptedStreamingModel {
        fn new(turns: Vec<Vec<RawStreamingChoice<MockStreamingResponse>>>) -> Self {
            Self {
                turns: Arc::new(turns),
                turn_counter: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[allow(refining_impl_trait)]
    impl CompletionModel for ScriptedStreamingModel {
        type Response = ();
        type StreamingResponse = MockStreamingResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::new(Vec::new())
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            Err(CompletionError::ProviderError(
                "completion is unused in this scripted streaming test".to_string(),
            ))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst);
            let choices = self.turns.get(turn).cloned().ok_or_else(|| {
                CompletionError::ProviderError(format!("unexpected scripted streaming turn {turn}"))
            })?;

            let stream = async_stream::stream! {
                for choice in choices {
                    yield Ok(choice);
                }
            };

            let pinned_stream: crate::streaming::StreamingResult<Self::StreamingResponse> =
                Box::pin(stream);
            Ok(StreamingCompletionResponse::stream(pinned_stream))
        }
    }

    #[derive(Clone, Default)]
    struct NonStreamingZeroMockModel {
        turn_counter: Arc<AtomicUsize>,
    }

    #[allow(refining_impl_trait)]
    impl CompletionModel for NonStreamingZeroMockModel {
        type Response = ();
        type StreamingResponse = MockStreamingResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::default()
        }

        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            let turn = self.turn_counter.fetch_add(1, Ordering::SeqCst);
            let choice = if turn == 0 {
                OneOrMany::many(vec![
                    AssistantContent::ToolCall(
                        crate::message::ToolCall::new(
                            "tool_call_a".to_string(),
                            ToolFunction::new("probe_tool".to_string(), json!({ "slot": "a" })),
                        )
                        .with_call_id("call_a".to_string()),
                    ),
                    AssistantContent::ToolCall(
                        crate::message::ToolCall::new(
                            "tool_call_b".to_string(),
                            ToolFunction::new("probe_tool".to_string(), json!({ "slot": "b" })),
                        )
                        .with_call_id("call_b".to_string()),
                    ),
                ])
                .expect("tool call response should contain at least one item")
            } else {
                OneOrMany::one(AssistantContent::text("done"))
            };

            Ok(CompletionResponse {
                choice,
                usage: crate::completion::Usage::new(),
                raw_response: (),
                message_id: None,
            })
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            Err(CompletionError::ProviderError(
                "stream is unused in this non-streaming test".to_string(),
            ))
        }
    }

    #[derive(Clone, Debug, Deserialize)]
    struct ProbeArgs {
        slot: String,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("probe tool error")]
    struct ProbeToolError;

    fn update_max_in_flight(max: &AtomicUsize, current: usize) {
        let mut observed = max.load(Ordering::SeqCst);
        while current > observed {
            match max.compare_exchange(observed, current, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(actual) => observed = actual,
            }
        }
    }

    struct InFlightGuard {
        current: Arc<AtomicUsize>,
    }

    impl InFlightGuard {
        fn new(current: Arc<AtomicUsize>, max: Arc<AtomicUsize>) -> Self {
            let in_flight = current.fetch_add(1, Ordering::SeqCst) + 1;
            update_max_in_flight(&max, in_flight);
            Self { current }
        }
    }

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            self.current.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct ProbeToolState {
        current: Arc<AtomicUsize>,
        max: Arc<AtomicUsize>,
        started: AtomicUsize,
        started_notify: Notify,
        initial_batch_size: usize,
        initial_batch_barrier: Barrier,
        release: Semaphore,
        executed_slots: Mutex<Vec<String>>,
    }

    impl ProbeToolState {
        fn new(initial_batch_size: usize) -> Arc<Self> {
            Arc::new(Self {
                current: Arc::new(AtomicUsize::new(0)),
                max: Arc::new(AtomicUsize::new(0)),
                started: AtomicUsize::new(0),
                started_notify: Notify::new(),
                initial_batch_size,
                initial_batch_barrier: Barrier::new(initial_batch_size.max(1) + 1),
                release: Semaphore::new(0),
                executed_slots: Mutex::new(Vec::new()),
            })
        }

        async fn wait_for_initial_batch(&self) {
            if self.initial_batch_size == 0 {
                return;
            }
            self.initial_batch_barrier.wait().await;
        }

        fn release_all(&self, permits: usize) {
            self.release.add_permits(permits);
        }

        fn max_in_flight(&self) -> usize {
            self.max.load(Ordering::SeqCst)
        }

        fn started_count(&self) -> usize {
            self.started.load(Ordering::SeqCst)
        }

        fn executed_slots(&self) -> Vec<String> {
            self.executed_slots
                .lock()
                .expect("probe tool executed slots mutex should not be poisoned")
                .clone()
        }
    }

    #[derive(Clone)]
    struct ProbeTool {
        state: Arc<ProbeToolState>,
    }

    impl Tool for ProbeTool {
        const NAME: &'static str = "probe_tool";

        type Error = ProbeToolError;
        type Args = ProbeArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Test probe tool".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "slot": { "type": "string" }
                    },
                    "required": ["slot"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            let start_order = self.state.started.fetch_add(1, Ordering::SeqCst) + 1;
            self.state.started_notify.notify_waiters();

            let _guard = InFlightGuard::new(self.state.current.clone(), self.state.max.clone());
            if start_order <= self.state.initial_batch_size {
                self.state.initial_batch_barrier.wait().await;
            }

            let permit = self
                .state
                .release
                .acquire()
                .await
                .expect("probe tool semaphore should stay open during the test");
            permit.forget();

            self.state
                .executed_slots
                .lock()
                .expect("probe tool executed slots mutex should not be poisoned")
                .push(args.slot.clone());

            Ok(args.slot)
        }
    }

    struct SlotToolState {
        current: Arc<AtomicUsize>,
        max: Arc<AtomicUsize>,
        started: AtomicUsize,
        started_notify: Notify,
        release_by_slot: Mutex<HashMap<String, Arc<Semaphore>>>,
        executed_slots: Mutex<Vec<String>>,
    }

    impl SlotToolState {
        fn new(slots: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                current: Arc::new(AtomicUsize::new(0)),
                max: Arc::new(AtomicUsize::new(0)),
                started: AtomicUsize::new(0),
                started_notify: Notify::new(),
                release_by_slot: Mutex::new(
                    slots
                        .iter()
                        .map(|slot| ((*slot).to_string(), Arc::new(Semaphore::new(0))))
                        .collect(),
                ),
                executed_slots: Mutex::new(Vec::new()),
            })
        }

        async fn wait_for_started(&self, target: usize) {
            while self.started.load(Ordering::SeqCst) < target {
                self.started_notify.notified().await;
            }
        }

        fn release_slot(&self, slot: &str) {
            let semaphore = self
                .release_by_slot
                .lock()
                .expect("slot tool release map mutex should not be poisoned")
                .get(slot)
                .cloned()
                .expect("slot should exist in the release map");
            semaphore.add_permits(1);
        }
    }

    #[derive(Clone)]
    struct SlotTool {
        state: Arc<SlotToolState>,
    }

    impl Tool for SlotTool {
        const NAME: &'static str = "slot_tool";

        type Error = ProbeToolError;
        type Args = ProbeArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: Self::NAME.to_string(),
                description: "Slot-controlled test tool".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "slot": { "type": "string" }
                    },
                    "required": ["slot"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.state.started.fetch_add(1, Ordering::SeqCst);
            self.state.started_notify.notify_waiters();

            let _guard = InFlightGuard::new(self.state.current.clone(), self.state.max.clone());
            let semaphore = self
                .state
                .release_by_slot
                .lock()
                .expect("slot tool release map mutex should not be poisoned")
                .get(&args.slot)
                .cloned()
                .expect("slot should exist in the release map");
            let permit = semaphore
                .acquire()
                .await
                .expect("slot tool semaphore should stay open during the test");
            permit.forget();

            self.state
                .executed_slots
                .lock()
                .expect("slot tool executed slots mutex should not be poisoned")
                .push(args.slot.clone());

            Ok(args.slot)
        }
    }

    #[derive(Clone, Default)]
    struct TestHook {
        skip_slots: Arc<Vec<String>>,
        terminate_before_slot: Option<String>,
        terminate_after_slot: Option<String>,
    }

    impl TestHook {
        fn skip(slots: &[&str]) -> Self {
            Self {
                skip_slots: Arc::new(slots.iter().map(|slot| (*slot).to_string()).collect()),
                ..Self::default()
            }
        }

        fn terminate_before(slot: &str) -> Self {
            Self {
                terminate_before_slot: Some(slot.to_string()),
                ..Self::default()
            }
        }

        fn terminate_after(slot: &str) -> Self {
            Self {
                terminate_after_slot: Some(slot.to_string()),
                ..Self::default()
            }
        }
    }

    impl<M> PromptHook<M> for TestHook
    where
        M: CompletionModel,
    {
        fn on_tool_call(
            &self,
            _tool_name: &str,
            _tool_call_id: Option<String>,
            _internal_call_id: &str,
            args: &str,
        ) -> impl Future<Output = ToolCallHookAction> + WasmCompatSend {
            let args = args.to_owned();
            let skip_slots = self.skip_slots.clone();
            let terminate_before_slot = self.terminate_before_slot.clone();
            async move {
                let parsed: ProbeArgs =
                    serde_json::from_str(&args).expect("tool args should deserialize in tests");

                if terminate_before_slot.as_deref() == Some(parsed.slot.as_str()) {
                    ToolCallHookAction::terminate(format!("terminate-before-{}", parsed.slot))
                } else if skip_slots.iter().any(|slot| slot == &parsed.slot) {
                    ToolCallHookAction::skip(format!("skipped-{}", parsed.slot))
                } else {
                    ToolCallHookAction::cont()
                }
            }
        }

        fn on_tool_result(
            &self,
            _tool_name: &str,
            _tool_call_id: Option<String>,
            _internal_call_id: &str,
            args: &str,
            _result: &str,
        ) -> impl Future<Output = HookAction> + WasmCompatSend {
            let args = args.to_owned();
            let terminate_after_slot = self.terminate_after_slot.clone();
            async move {
                let parsed: ProbeArgs =
                    serde_json::from_str(&args).expect("tool args should deserialize in tests");

                if terminate_after_slot.as_deref() == Some(parsed.slot.as_str()) {
                    HookAction::terminate(format!("terminate-after-{}", parsed.slot))
                } else {
                    HookAction::cont()
                }
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum TurnEvent {
        ToolCall(String),
        ToolResult(String),
        Final,
    }

    fn raw_tool_call_for(tool_name: &str, id: &str, slot: &str) -> RawStreamingToolCall {
        RawStreamingToolCall::new(
            id.to_string(),
            tool_name.to_string(),
            json!({ "slot": slot }),
        )
        .with_call_id(format!("call_{id}"))
    }

    fn tool_turn(
        text_chunks: &[&str],
        tool_calls: Vec<RawStreamingToolCall>,
        total_tokens: u64,
    ) -> Vec<RawStreamingChoice<MockStreamingResponse>> {
        let mut turn: Vec<_> = text_chunks
            .iter()
            .map(|text| RawStreamingChoice::Message((*text).to_string()))
            .collect();
        turn.extend(tool_calls.into_iter().map(RawStreamingChoice::ToolCall));
        turn.push(RawStreamingChoice::FinalResponse(
            MockStreamingResponse::new(total_tokens),
        ));
        turn
    }

    fn text_turn(text: &str, total_tokens: u64) -> Vec<RawStreamingChoice<MockStreamingResponse>> {
        vec![
            RawStreamingChoice::Message(text.to_string()),
            RawStreamingChoice::FinalResponse(MockStreamingResponse::new(total_tokens)),
        ]
    }

    fn first_turn_events(
        items: &[Result<MultiTurnStreamItem<MockStreamingResponse>, StreamingError>],
    ) -> Vec<TurnEvent> {
        let mut events = Vec::new();

        for item in items {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { tool_call, .. },
                )) => events.push(TurnEvent::ToolCall(tool_call.id.clone())),
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => events.push(TurnEvent::ToolResult(tool_result.id.clone())),
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(
                    _,
                ))) => {
                    events.push(TurnEvent::Final);
                    break;
                }
                _ => {}
            }
        }

        events
    }

    fn tool_result_text(tool_result: &crate::message::ToolResult) -> Option<String> {
        match tool_result.content.first_ref() {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }
    }

    #[tokio::test]
    async fn stream_prompt_concurrency_is_bounded_when_configured() {
        let state = ProbeToolState::new(2);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("probe_tool", "tool_1", "a"),
                    raw_tool_call_for("probe_tool", "tool_2", "b"),
                    raw_tool_call_for("probe_tool", "tool_3", "c"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .tool(ProbeTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("run tools")
            .multi_turn(3)
            .with_tool_concurrency(2)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_initial_batch())
            .await
            .expect("expected the initial bounded tool batch to start");
        assert_eq!(state.max_in_flight(), 2);

        state.release_all(8);

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(items.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn stream_prompt_concurrency_one_stays_sequential() {
        let state = ProbeToolState::new(1);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("probe_tool", "tool_1", "a"),
                    raw_tool_call_for("probe_tool", "tool_2", "b"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .tool(ProbeTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("run tools sequentially")
            .multi_turn(3)
            .with_tool_concurrency(1)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_initial_batch())
            .await
            .expect("expected the sequential tool batch to start");
        assert_eq!(state.max_in_flight(), 1);

        state.release_all(4);

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(items.iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn zero_tool_concurrency_clamps_to_one_for_streaming_and_non_streaming() {
        let streaming_state = ProbeToolState::new(1);
        let streaming_model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("probe_tool", "tool_1", "a"),
                    raw_tool_call_for("probe_tool", "tool_2", "b"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let streaming_agent = AgentBuilder::new(streaming_model)
            .tool(ProbeTool {
                state: streaming_state.clone(),
            })
            .build();

        let mut stream = streaming_agent
            .stream_prompt("streaming zero concurrency")
            .multi_turn(3)
            .with_tool_concurrency(0)
            .await;

        let stream_handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(
            Duration::from_secs(1),
            streaming_state.wait_for_initial_batch(),
        )
        .await
        .expect("expected zero streaming concurrency to clamp to one");
        assert_eq!(streaming_state.max_in_flight(), 1);
        streaming_state.release_all(4);

        let stream_items = tokio::time::timeout(Duration::from_secs(1), stream_handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(stream_items.iter().all(Result::is_ok));

        let prompt_state = ProbeToolState::new(1);
        let prompt_agent = AgentBuilder::new(NonStreamingZeroMockModel::default())
            .tool(ProbeTool {
                state: prompt_state.clone(),
            })
            .build();

        let prompt_handle = tokio::spawn(async move {
            prompt_agent
                .prompt("non-streaming zero concurrency")
                .max_turns(3)
                .with_tool_concurrency(0)
                .await
        });

        tokio::time::timeout(
            Duration::from_secs(1),
            prompt_state.wait_for_initial_batch(),
        )
        .await
        .expect("expected zero non-streaming concurrency to clamp to one");
        assert_eq!(prompt_state.max_in_flight(), 1);
        prompt_state.release_all(4);

        let prompt_result = tokio::time::timeout(Duration::from_secs(1), prompt_handle)
            .await
            .expect("prompt future should finish")
            .expect("prompt task should join")
            .expect("prompt should succeed");
        assert_eq!(prompt_result, "done");
    }

    #[tokio::test]
    async fn stream_prompt_yields_tool_results_before_turn_final_in_concurrent_mode() {
        let state = ProbeToolState::new(2);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &["draft"],
                vec![
                    raw_tool_call_for("probe_tool", "tool_1", "a"),
                    raw_tool_call_for("probe_tool", "tool_2", "b"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .tool(ProbeTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("order please")
            .multi_turn(3)
            .with_tool_concurrency(2)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_initial_batch())
            .await
            .expect("expected the initial tool batch to start");
        state.release_all(4);

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(items.iter().all(Result::is_ok));
        let events = first_turn_events(&items);
        assert_eq!(events.len(), 5);
        assert_eq!(events[0], TurnEvent::ToolCall("tool_1".to_string()));
        assert_eq!(events[1], TurnEvent::ToolCall("tool_2".to_string()));
        assert!(matches!(events[2], TurnEvent::ToolResult(_)));
        assert!(matches!(events[3], TurnEvent::ToolResult(_)));
        assert_eq!(events[4], TurnEvent::Final);
    }

    #[tokio::test]
    async fn stream_prompt_yields_concurrent_tool_results_in_completion_order() {
        let state = SlotToolState::new(&["slow", "fast", "medium"]);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("slot_tool", "slow_id", "slow"),
                    raw_tool_call_for("slot_tool", "fast_id", "fast"),
                    raw_tool_call_for("slot_tool", "medium_id", "medium"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .tool(SlotTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("ordered release")
            .multi_turn(3)
            .with_tool_concurrency(3)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(3))
            .await
            .expect("expected all slot-controlled tools to start");
        state.release_slot("fast");
        tokio::task::yield_now().await;
        state.release_slot("medium");
        tokio::task::yield_now().await;
        state.release_slot("slow");

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(items.iter().all(Result::is_ok));

        let result_ids: Vec<_> = first_turn_events(&items)
            .into_iter()
            .filter_map(|event| match event {
                TurnEvent::ToolResult(id) => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(
            result_ids,
            vec![
                "fast_id".to_string(),
                "medium_id".to_string(),
                "slow_id".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_skip_hook_emits_synthetic_result_without_execution() {
        let state = ProbeToolState::new(1);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("probe_tool", "skip_id", "skip"),
                    raw_tool_call_for("probe_tool", "run_id", "run"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .hook(TestHook::skip(&["skip"]))
            .tool(ProbeTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("skip one tool")
            .multi_turn(3)
            .with_tool_concurrency(2)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_initial_batch())
            .await
            .expect("expected the continued tool call to start");
        state.release_all(2);

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");
        assert!(items.iter().all(Result::is_ok));
        assert_eq!(state.started_count(), 1);
        assert_eq!(state.executed_slots(), vec!["run".to_string()]);

        let mut tool_results = Vec::new();
        for item in &items {
            if let Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            })) = item
            {
                tool_results.push((tool_result.id.clone(), tool_result_text(tool_result)));
            }
        }

        assert_eq!(
            tool_results,
            vec![
                ("skip_id".to_string(), Some("skipped-skip".to_string())),
                ("run_id".to_string(), Some("\"run\"".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_pre_hook_terminate_prevents_dispatch_and_turn_final() {
        let state = ProbeToolState::new(1);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![raw_tool_call_for("probe_tool", "stop_id", "stop")],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .hook(TestHook::terminate_before("stop"))
            .tool(ProbeTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("terminate before execution")
            .multi_turn(3)
            .with_tool_concurrency(2)
            .await;

        let mut saw_turn_final = false;
        let mut saw_error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(
                    _,
                ))) => {
                    saw_turn_final = true;
                }
                Err(err) => {
                    saw_error = Some(err.to_string());
                    break;
                }
                _ => {}
            }
        }

        assert_eq!(state.started_count(), 0);
        assert!(!saw_turn_final);
        assert_eq!(
            saw_error,
            Some("PromptError: PromptCancelled: terminate-before-stop".to_string())
        );
    }

    #[tokio::test]
    async fn stream_prompt_post_hook_terminate_stops_before_turn_final() {
        let state = SlotToolState::new(&["fast", "slow"]);
        let model = ScriptedStreamingModel::new(vec![
            tool_turn(
                &[],
                vec![
                    raw_tool_call_for("slot_tool", "fast_id", "fast"),
                    raw_tool_call_for("slot_tool", "slow_id", "slow"),
                ],
                3,
            ),
            text_turn("done", 2),
        ]);

        let agent = AgentBuilder::new(model)
            .hook(TestHook::terminate_after("fast"))
            .tool(SlotTool {
                state: state.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("terminate after first result")
            .multi_turn(3)
            .with_tool_concurrency(2)
            .await;

        let handle = tokio::spawn(async move {
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
            }
            items
        });

        tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(2))
            .await
            .expect("expected both slot-controlled tools to start");
        state.release_slot("fast");

        let items = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("stream collector should finish")
            .expect("stream collector task should join");

        assert!(
            items.iter().all(|item| {
                !matches!(
                    item,
                    Ok(MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::Final(_)
                    ))
                )
            }),
            "turn final should not be emitted after post-hook termination"
        );

        let errors: Vec<_> = items
            .into_iter()
            .filter_map(Result::err)
            .map(|err| err.to_string())
            .collect();
        assert_eq!(
            errors,
            vec!["PromptError: PromptCancelled: terminate-after-fast".to_string()]
        );
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
    async fn test_span_context_isolation() {
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
        let client = anthropic::Client::from_env();
        let agent = client
            .agent(anthropic::completion::CLAUDE_3_5_HAIKU)
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
        bg_handle.await.unwrap();

        let leaks = leak_count.load(Ordering::Relaxed);
        assert_eq!(
            leaks, 0,
            "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times. \
             This indicates that span.enter() is being used inside async_stream instead of .instrument()"
        );
    }

    /// Test that FinalResponse contains the updated chat history when with_history is used.
    ///
    /// This verifies that:
    /// 1. FinalResponse.history() returns Some when with_history was called
    /// 2. The history contains both the user prompt and assistant response
    #[tokio::test]
    #[ignore = "This requires an API key"]
    async fn test_chat_history_in_final_response() {
        use crate::message::Message;

        let client = anthropic::Client::from_env();
        let agent = client
            .agent(anthropic::completion::CLAUDE_3_5_HAIKU)
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
                    panic!("Streaming error: {:?}", e);
                }
                _ => {}
            }
        }

        let history =
            final_history.expect("FinalResponse should contain history when with_history is used");

        // Should contain at least the user message
        assert!(
            history.iter().any(|m| matches!(m, Message::User { .. })),
            "History should contain the user message"
        );

        // Should contain the assistant response
        assert!(
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
    }
}
