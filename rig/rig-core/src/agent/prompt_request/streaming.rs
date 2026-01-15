use crate::{
    OneOrMany,
    agent::CancelSignal,
    completion::GetTokenUsage,
    json_utils,
    message::{AssistantContent, Reasoning, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingCompletion},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tokio::sync::RwLock;
use tracing::info_span;
use tracing_futures::Instrument;

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
}

impl FinalResponse {
    pub fn empty() -> Self {
        Self {
            response: String::new(),
            aggregated_usage: crate::completion::Usage::new(),
        }
    }

    pub fn response(&self) -> &str {
        &self.response
    }

    pub fn usage(&self) -> crate::completion::Usage {
        self.aggregated_usage
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
        })
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

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// If you expect to continuously call tools, you will want to ensure you use the `.multi_turn()`
/// argument to add more turns as by default, it is 0 (meaning only 1 tool round-trip). Otherwise,
/// attempting to await (which will send the prompt request) can potentially return
/// [`crate::completion::request::PromptError::MaxDepthError`] if the agent decides to call tools
/// back to back.
pub struct StreamingPromptRequest<M, P>
where
    M: CompletionModel,
    P: StreamingPromptHook<M> + 'static,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history to include with the prompt
    /// Note: chat history needs to outlive the agent as it might be used with other agents
    chat_history: Option<Vec<Message>>,
    /// Maximum depth for multi-turn conversations (0 means no multi-turn)
    max_depth: usize,
    /// The agent to use for execution
    agent: Arc<Agent<M>>,
    /// Optional per-request hook for events
    hook: Option<P>,
}

impl<M, P> StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend + GetTokenUsage,
    P: StreamingPromptHook<M>,
{
    /// Create a new PromptRequest with the given prompt and model
    pub fn new(agent: Arc<Agent<M>>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: agent.default_max_depth.unwrap_or_default(),
            agent,
            hook: None,
        }
    }

    /// Set the maximum depth for multi-turn conversations (ie, the maximum number of turns an LLM can have calling tools before writing a text response).
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxDepthError`].
    pub fn multi_turn(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Add chat history to the prompt request
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = Some(history);
        self
    }

    /// Attach a per-request hook for tool call events
    pub fn with_hook<P2>(self, hook: P2) -> StreamingPromptRequest<M, P2>
    where
        P2: StreamingPromptHook<M>,
    {
        StreamingPromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            hook: Some(hook),
        }
    }

    async fn send(self) -> StreamingResult<M::StreamingResponse> {
        let agent_span = if tracing::Span::current().is_disabled() {
            info_span!(
                "invoke_agent",
                gen_ai.operation.name = "invoke_agent",
                gen_ai.agent.name = self.agent.name(),
                gen_ai.system_instructions = self.agent.preamble,
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

        let agent = self.agent;

        let chat_history = if let Some(history) = self.chat_history {
            Arc::new(RwLock::new(history))
        } else {
            Arc::new(RwLock::new(vec![]))
        };

        let mut current_max_depth = 0;
        let mut last_prompt_error = String::new();

        let mut last_text_response = String::new();
        let mut is_text_response = false;
        let mut max_depth_reached = false;

        let mut aggregated_usage = crate::completion::Usage::new();

        let cancel_sig = CancelSignal::new();

        // NOTE: We use .instrument(agent_span) instead of span.enter() to avoid
        // span context leaking to other concurrent tasks. Using span.enter() inside
        // async_stream::stream! holds the guard across yield points, which causes
        // thread-local span context to leak when other tasks run on the same thread.
        // See: https://docs.rs/tracing/latest/tracing/span/struct.Span.html#in-asynchronous-code
        // See also: https://github.com/rust-lang/rust-clippy/issues/8722
        let stream = async_stream::stream! {
            let mut current_prompt = prompt.clone();
            let mut did_call_tool = false;

            'outer: loop {
                if current_max_depth > self.max_depth + 1 {
                    last_prompt_error = current_prompt.rag_text().unwrap_or_default();
                    max_depth_reached = true;
                    break;
                }

                current_max_depth += 1;

                if self.max_depth > 1 {
                    tracing::info!(
                        "Current conversation depth: {}/{}",
                        current_max_depth,
                        self.max_depth
                    );
                }

                if let Some(ref hook) = self.hook {
                    let reader = chat_history.read().await;
                    hook.on_completion_call(&current_prompt, &reader.to_vec(), cancel_sig.clone())
                        .await;

                    if cancel_sig.is_cancelled() {
                        yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                        ).into()));
                    }
                }

                let chat_stream_span = info_span!(
                    target: "rig::agent_chat",
                    parent: tracing::Span::current(),
                    "chat_streaming",
                    gen_ai.operation.name = "chat",
                    gen_ai.system_instructions = &agent.preamble,
                    gen_ai.provider.name = tracing::field::Empty,
                    gen_ai.request.model = tracing::field::Empty,
                    gen_ai.response.id = tracing::field::Empty,
                    gen_ai.response.model = tracing::field::Empty,
                    gen_ai.usage.output_tokens = tracing::field::Empty,
                    gen_ai.usage.input_tokens = tracing::field::Empty,
                    gen_ai.input.messages = tracing::field::Empty,
                    gen_ai.output.messages = tracing::field::Empty,
                );

                let mut stream = tracing::Instrument::instrument(
                    agent
                    .stream_completion(current_prompt.clone(), (*chat_history.read().await).clone())
                    .await?
                    .stream(), chat_stream_span
                )

                .await?;

                chat_history.write().await.push(current_prompt.clone());

                let mut tool_calls = vec![];
                let mut tool_results = vec![];

                while let Some(content) = stream.next().await {
                    match content {
                        Ok(StreamedAssistantContent::Text(text)) => {
                            if !is_text_response {
                                last_text_response = String::new();
                                is_text_response = true;
                            }
                            last_text_response.push_str(&text.text);
                            if let Some(ref hook) = self.hook {
                                hook.on_text_delta(&text.text, &last_text_response, cancel_sig.clone()).await;
                                if cancel_sig.is_cancelled() {
                                    yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                                        cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                                    ).into()));
                                }
                            }
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Text(text)));
                            did_call_tool = false;
                        },
                        Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
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

                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ToolCall(tool_call.clone())));

                            let tc_result = async {
                                let tool_span = tracing::Span::current();
                                let tool_args = json_utils::value_to_json_string(&tool_call.function.arguments);
                                if let Some(ref hook) = self.hook {
                                    hook.on_tool_call(&tool_call.function.name, tool_call.call_id.clone(), &tool_args, cancel_sig.clone()).await;
                                    if cancel_sig.is_cancelled() {
                                        return Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                                            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                                        ).into()));
                                    }
                                }

                                tool_span.record("gen_ai.tool.name", &tool_call.function.name);
                                tool_span.record("gen_ai.tool.call.arguments", &tool_args);

                                let tool_result = match
                                agent.tool_server_handle.call_tool(&tool_call.function.name, &tool_args).await {
                                    Ok(thing) => thing,
                                    Err(e) => {
                                        tracing::warn!("Error while calling tool: {e}");
                                        e.to_string()
                                    }
                                };

                                tool_span.record("gen_ai.tool.call.result", &tool_result);

                                if let Some(ref hook) = self.hook {
                                    hook.on_tool_result(&tool_call.function.name, tool_call.call_id.clone(), &tool_args, &tool_result.to_string(), cancel_sig.clone())
                                    .await;

                                    if cancel_sig.is_cancelled() {
                                        return Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                                            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                                        ).into()));
                                    }
                                }

                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id.clone(), tool_call.call_id.clone(), tool_result.clone()));

                                did_call_tool = true;
                                Ok(tool_result)
                            }.instrument(tool_span).await;

                            match tc_result {
                                Ok(text) => {
                                    let tr = ToolResult { id: tool_call.id, call_id: tool_call.call_id, content: OneOrMany::one(ToolResultContent::Text(Text { text })) };
                                    yield Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult(tr)));
                                }
                                Err(e) => {
                                    yield Err(e);
                                }
                            }
                        },
                        Ok(StreamedAssistantContent::ToolCallDelta { id, content }) => {
                            if let Some(ref hook) = self.hook {
                                let (name, delta) = match &content {
                                    rig::streaming::ToolCallDeltaContent::Name(n) => (Some(n.as_str()), ""),
                                    rig::streaming::ToolCallDeltaContent::Delta(d) => (None, d.as_str()),
                                };
                                hook.on_tool_call_delta(&id, name, delta, cancel_sig.clone())
                                .await;

                                if cancel_sig.is_cancelled() {
                                    yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                                        cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                                    ).into()));
                                }
                            }
                        }
                        Ok(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, id, signature })) => {
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, id, signature })));
                            did_call_tool = false;
                        },
                        Ok(StreamedAssistantContent::ReasoningDelta { reasoning, id }) => {
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::ReasoningDelta { reasoning, id }));
                            did_call_tool = false;
                        },
                        Ok(StreamedAssistantContent::Final(final_resp)) => {
                            if let Some(usage) = final_resp.token_usage() { aggregated_usage += usage; };
                            if is_text_response {
                                if let Some(ref hook) = self.hook {
                                    hook.on_stream_completion_response_finish(&prompt, &final_resp, cancel_sig.clone()).await;

                                    if cancel_sig.is_cancelled() {
                                        yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec(),
                                            cancel_sig.cancel_reason().unwrap_or("<no reason given>"),
                                        ).into()));
                                    }
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

                // Add (parallel) tool calls to chat history
                if !tool_calls.is_empty() {
                    chat_history.write().await.push(Message::Assistant {
                        id: None,
                        content: OneOrMany::many(tool_calls.clone()).expect("Impossible EmptyListError"),
                    });
                }

                // Add tool results to chat history
                for (id, call_id, tool_result) in tool_results {
                    if let Some(call_id) = call_id {
                        chat_history.write().await.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result_with_call_id(
                                &id,
                                call_id.clone(),
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    } else {
                        chat_history.write().await.push(Message::User {
                            content: OneOrMany::one(UserContent::tool_result(
                                &id,
                                OneOrMany::one(ToolResultContent::text(&tool_result)),
                            )),
                        });
                    }
                }

                // Set the current prompt to the last message in the chat history
                current_prompt = match chat_history.write().await.pop() {
                    Some(prompt) => prompt,
                    None => unreachable!("Chat history should never be empty at this point"),
                };

                if !did_call_tool {
                    let current_span = tracing::Span::current();
                    current_span.record("gen_ai.usage.input_tokens", aggregated_usage.input_tokens);
                    current_span.record("gen_ai.usage.output_tokens", aggregated_usage.output_tokens);
                    tracing::info!("Agent multi-turn stream finished");
                    yield Ok(MultiTurnStreamItem::final_response(&last_text_response, aggregated_usage));
                    break;
                }
            }

            if max_depth_reached {
                yield Err(Box::new(PromptError::MaxDepthError {
                    max_depth: self.max_depth,
                    chat_history: Box::new((*chat_history.read().await).clone()),
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
    P: StreamingPromptHook<M> + 'static,
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
                Reasoning { reasoning, .. },
            ))) => {
                let reasoning = reasoning.join("\n");
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

// dead code allowed because of functions being left empty to allow for users to not have to implement every single function
/// Trait for per-request hooks to observe tool call events.
pub trait StreamingPromptHook<M>: Clone + Send + Sync
where
    M: CompletionModel,
{
    #[allow(unused_variables)]
    /// Called before the prompt is sent to the model
    fn on_completion_call(
        &self,
        prompt: &Message,
        history: &[Message],
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called when receiving a text delta
    fn on_text_delta(
        &self,
        text_delta: &str,
        aggregated_text: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called when receiving a tool call delta.
    /// `tool_name` is Some on the first delta for a tool call, None on subsequent deltas.
    fn on_tool_call_delta(
        &self,
        tool_call_id: &str,
        tool_name: Option<&str>,
        tool_call_delta: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called after the model provider has finished streaming a text response from their completion API to the client.
    fn on_stream_completion_response_finish(
        &self,
        prompt: &Message,
        response: &<M as CompletionModel>::StreamingResponse,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called before a tool is invoked.
    fn on_tool_call(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called after a tool is invoked (and a result has been returned).
    fn on_tool_result(
        &self,
        tool_name: &str,
        tool_call_id: Option<String>,
        args: &str,
        result: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

impl<M> StreamingPromptHook<M> for () where M: CompletionModel {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ProviderClient;
    use crate::client::completion::CompletionClient;
    use crate::providers::anthropic;
    use crate::streaming::StreamingPrompt;
    use futures::StreamExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

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
}
