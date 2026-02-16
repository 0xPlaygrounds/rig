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
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tracing::info_span;
use tracing_futures::Instrument;

use super::ToolCallHookAction;
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

async fn cancelled_prompt_error(
    chat_history: &Arc<RwLock<Vec<Message>>>,
    reason: String,
) -> StreamingError {
    StreamingError::Prompt(
        PromptError::prompt_cancelled(chat_history.read().await.to_vec(), reason).into(),
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
    /// Optional chat history to include with the prompt.
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
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = Some(history);
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
        let mut chat_history = self.chat_history.unwrap_or_default();

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
                    let history_snapshot = chat_history.clone();
                    if let HookAction::Terminate { reason } = hook.on_completion_call(&current_prompt, &history_snapshot)
                        .await {
                        yield Err(cancelled_prompt_error(&chat_history, reason).await);
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
                    gen_ai.input.messages = tracing::field::Empty,
                    gen_ai.output.messages = tracing::field::Empty,
                );

                let history_snapshot = chat_history.clone();
                let mut stream = tracing::Instrument::instrument(
                    build_completion_request(
                        &model,
                        current_prompt.clone(),
                        history_snapshot,
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

                chat_history.push(current_prompt.clone());

                let mut tool_calls = vec![];
                let mut tool_results = vec![];
                let mut accumulated_reasoning: Vec<rig::message::Reasoning> = vec![];
                let mut saw_tool_call_this_turn = false;

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
                                    yield Err(cancelled_prompt_error(&chat_history, reason).await);
                                    break 'outer;
                            }

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Text(text)));
                        },
                        Ok(StreamedAssistantContent::ToolCall { tool_call, internal_call_id }) => {
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
                                let tool_args = json_utils::value_to_json_string(&tool_call.function.arguments);
                                if let Some(ref hook) = self.hook {
                                    let action = hook
                                        .on_tool_call(&tool_call.function.name, tool_call.call_id.clone(), &internal_call_id, &tool_args)
                                        .await;

                                    if let ToolCallHookAction::Terminate { reason } = action {
                                        return Err(cancelled_prompt_error(&chat_history, reason).await);
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
                                        return Err(cancelled_prompt_error(&chat_history, reason).await);
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
                            if let Some(ref hook) = self.hook {
                                let (name, delta) = match &content {
                                    rig::streaming::ToolCallDeltaContent::Name(n) => (Some(n.as_str()), ""),
                                    rig::streaming::ToolCallDeltaContent::Delta(d) => (None, d.as_str()),
                                };

                                if let HookAction::Terminate { reason } = hook.on_tool_call_delta(&id, &internal_call_id, name, delta)
                                .await {
                                    yield Err(cancelled_prompt_error(&chat_history, reason).await);
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
                            // Deltas must be accumulated or reasoning evaporates
                            // in multi-turn tool call loops.
                            let delta_as_reasoning = crate::message::Reasoning::new(&reasoning)
                                .optional_id(id.clone());
                            merge_reasoning_blocks(&mut accumulated_reasoning, &delta_as_reasoning);
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ReasoningDelta { reasoning, id }));
                        },
                        Ok(StreamedAssistantContent::Final(final_resp)) => {
                            if let Some(usage) = final_resp.token_usage() { aggregated_usage += usage; };
                            if is_text_response {
                                if let Some(ref hook) = self.hook &&
                                     let HookAction::Terminate { reason } = hook.on_stream_completion_response_finish(&prompt, &final_resp).await {
                                        yield Err(cancelled_prompt_error(&chat_history, reason).await);
                                        break 'outer;
                                    }

                                tracing::Span::current().record("gen_ai.completion", &last_text_response);
                                yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Final(final_resp)));
                                is_text_response = false;
                            }
                        }
                        Err(e) => {
                            yield Err(e.into());
                            break 'outer;
                        }
                    }
                }

                // Add reasoning and tool calls to chat history.
                // OpenAI Responses API requires reasoning items to precede function_call items.
                let mut history = chat_history.write().await;
                if !tool_calls.is_empty() || !accumulated_reasoning.is_empty() {
                    let mut content_items: Vec<rig::message::AssistantContent> = vec![];

                    // Reasoning must come before tool calls (OpenAI requirement)
                    for reasoning in accumulated_reasoning.drain(..) {
                        content_items.push(rig::message::AssistantContent::Reasoning(reasoning));
                    }

                    content_items.extend(tool_calls.clone());

                    if !content_items.is_empty() {
                        chat_history.push(Message::Assistant {
                            id: None,
                            content: OneOrMany::many(content_items).expect("Should have at least one item"),
                        });
                    }
                }

                for (id, call_id, tool_result) in tool_results {
                    if let Some(call_id) = call_id {
                        chat_history.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result_with_call_id(
                                &id,
                                call_id.clone(),
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    } else {
                        chat_history.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result(
                                &id,
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    }
                }

                // Set the current prompt to the last message in the chat history
                current_prompt = match chat_history.pop() {
                    Some(prompt) => prompt,
                    None => unreachable!("Chat history should never be empty at this point"),
                };

                if !did_call_tool {
                    // Add user message and assistant response to history before finishing
                    chat_history.push(current_prompt.clone());
                    if !last_text_response.is_empty() {
                        chat_history.push(Message::assistant(&last_text_response));
                    }

                    let current_span = tracing::Span::current();
                    current_span.record("gen_ai.usage.input_tokens", aggregated_usage.input_tokens);
                    current_span.record("gen_ai.usage.output_tokens", aggregated_usage.output_tokens);
                    tracing::info!("Agent multi-turn stream finished");
                    let history_snapshot = if has_history {
                        Some(chat_history.clone())
                    } else {
                        None
                    };
                    yield Ok(MultiTurnStreamItem::final_response_with_history(
                        &last_text_response,
                        aggregated_usage,
                        history_snapshot,
                    ));
                    break;
                }
            }

            if max_turns_reached {
                yield Err(Box::new(PromptError::MaxTurnsError {
                    max_turns: self.max_turns,
                    chat_history: Box::new(chat_history.clone()),
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
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    };
    use crate::message::ReasoningContent;
    use crate::providers::anthropic;
    use crate::streaming::StreamingPrompt;
    use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse};
    use futures::StreamExt;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
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
        let mut stream = agent
            .stream_prompt("Say 'hello' and nothing else.")
            .with_history(vec![])
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
