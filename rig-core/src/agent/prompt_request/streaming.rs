use crate::{
    OneOrMany,
    agent::CancelSignal,
    completion::GetTokenUsage,
    message::{AssistantContent, Reasoning, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamingCompletion},
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

#[cfg(not(target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Send>>;

#[cfg(target_arch = "wasm32")]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[non_exhaustive]
pub enum MultiTurnStreamItem<R> {
    StreamItem(StreamedAssistantContent<R>),
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
        Self::StreamItem(item)
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
            max_depth: 0,
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

    #[cfg_attr(feature = "worker", worker::send)]
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

        let cancel_signal = CancelSignal::new();

        Box::pin(async_stream::stream! {
            let _guard = agent_span.enter();
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
                    let prompt = reader.last().cloned().expect("there should always be at least one message in the chat history");
                    let chat_history_except_last = reader[..reader.len() - 1].to_vec();

                    hook.on_completion_call(&prompt, &chat_history_except_last, cancel_signal.clone())
                    .await;

                    if cancel_signal.is_cancelled() {
                        yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec()).into()));
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
                                hook.on_text_delta(&text.text, &last_text_response, cancel_signal.clone()).await;
                                if cancel_signal.is_cancelled() {
                                    yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec()).into()));
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

                            let res = async {
                                let tool_span = tracing::Span::current();
                                if let Some(ref hook) = self.hook {
                                    hook.on_tool_call(&tool_call.function.name, &tool_call.function.arguments.to_string(), cancel_signal.clone()).await;
                                    if cancel_signal.is_cancelled() {
                                        return Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec()).into()));
                                    }
                                }

                                tool_span.record("gen_ai.tool.name", &tool_call.function.name);
                                tool_span.record("gen_ai.tool.call.arguments", tool_call.function.arguments.to_string());

                                let tool_result = match
                                agent.tool_server_handle.call_tool(&tool_call.function.name, &tool_call.function.arguments.to_string()).await {
                                    Ok(thing) => thing,
                                    Err(e) => {
                                        tracing::warn!("Error while calling tool: {e}");
                                        e.to_string()
                                    }
                                };

                                tool_span.record("gen_ai.tool.call.result", &tool_result);

                                if let Some(ref hook) = self.hook {
                                    hook.on_tool_result(&tool_call.function.name, &tool_call.function.arguments.to_string(), &tool_result.to_string(), cancel_signal.clone())
                                    .await;

                                    if cancel_signal.is_cancelled() {
                                        return Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec()).into()));
                                    }
                                }

                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id, tool_call.call_id, tool_result));

                                did_call_tool = true;
                                Ok(())
                                // break;
                            }.instrument(tool_span).await;

                            if let Err(e) = res {
                                yield Err(e);
                            }
                        },
                        Ok(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, id, signature })) => {
                            chat_history.write().await.push(rig::message::Message::Assistant {
                                id: None,
                                content: OneOrMany::one(AssistantContent::Reasoning(Reasoning {
                                    reasoning: reasoning.clone(), id: id.clone(), signature: signature.clone()
                                }))
                            });
                            yield Ok(MultiTurnStreamItem::stream_item(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, id, signature })));
                            did_call_tool = false;
                        },
                        Ok(StreamedAssistantContent::Final(final_resp)) => {
                            if let Some(usage) = final_resp.token_usage() { aggregated_usage += usage; };
                            if is_text_response {
                                if let Some(ref hook) = self.hook {
                                    hook.on_stream_completion_response_finish(&prompt, &final_resp, cancel_signal.clone()).await;

                                    if cancel_signal.is_cancelled() {
                                        yield Err(StreamingError::Prompt(PromptError::prompt_cancelled(chat_history.read().await.to_vec()).into()));
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
                    prompt: last_prompt_error.clone().into(),
                }).into());
            }

        })
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

/// helper function to stream a completion selfuest to stdout
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamItem(StreamedAssistantContent::Text(Text { text }))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::StreamItem(StreamedAssistantContent::Reasoning(
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
        args: &str,
        result: &str,
        cancel_sig: CancelSignal,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

impl<M> StreamingPromptHook<M> for () where M: CompletionModel {}
