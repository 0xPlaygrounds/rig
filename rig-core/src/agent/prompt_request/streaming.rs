use crate::{
    OneOrMany,
    agent::prompt_request::PromptHook,
    completion::GetTokenUsage,
    message::{AssistantContent, Reasoning, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamingCompletion},
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tokio::sync::RwLock;

use crate::{
    agent::Agent,
    completion::{CompletionError, CompletionModel, PromptError},
    message::{Message, Text},
    tool::ToolSetError,
};

#[cfg(not(target_arch = "wasm32"))]
type StreamingResult =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>> + Send>>;

#[cfg(target_arch = "wasm32")]
type StreamingResult = Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MultiTurnStreamItem {
    Text(Text),
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

impl MultiTurnStreamItem {
    pub(crate) fn text(text: &str) -> Self {
        Self::Text(Text {
            text: text.to_string(),
        })
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
    Prompt(#[from] PromptError),
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
    P: PromptHook<M> + 'static,
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
    <M as CompletionModel>::StreamingResponse: Send + GetTokenUsage,
    P: PromptHook<M>,
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
        P2: PromptHook<M>,
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
    async fn send(self) -> StreamingResult {
        let agent_name = self.agent.name_owned();

        #[tracing::instrument(skip_all, fields(agent_name = agent_name))]
        fn inner<M, P>(req: StreamingPromptRequest<M, P>, agent_name: String) -> StreamingResult
        where
            M: CompletionModel + 'static,
            <M as CompletionModel>::StreamingResponse: Send,
            P: PromptHook<M> + 'static,
        {
            let prompt = req.prompt;
            let agent = req.agent;

            let chat_history = if let Some(mut history) = req.chat_history {
                history.push(prompt.clone());
                Arc::new(RwLock::new(history))
            } else {
                Arc::new(RwLock::new(vec![prompt.clone()]))
            };

            let mut current_max_depth = 0;
            let mut last_prompt_error = String::new();

            let mut last_text_response = String::new();
            let mut is_text_response = false;
            let mut max_depth_reached = false;

            let mut aggregated_usage = crate::completion::Usage::new();

            Box::pin(async_stream::stream! {
                let mut current_prompt = prompt.clone();
                let mut did_call_tool = false;

                'outer: loop {
                    if current_max_depth > req.max_depth + 1 {
                        last_prompt_error = current_prompt.rag_text().unwrap_or_default();
                        max_depth_reached = true;
                        break;
                    }

                    current_max_depth += 1;

                    if req.max_depth > 1 {
                        tracing::info!(
                            "Current conversation depth: {}/{}",
                            current_max_depth,
                            req.max_depth
                        );
                    }

                    if let Some(ref hook) = req.hook {
                        let reader = chat_history.read().await;
                        let prompt = reader.last().cloned().expect("there should always be at least one message in the chat history");
                        let chat_history_except_last = reader[..reader.len() - 1].to_vec();

                        hook.on_completion_call(&prompt, &chat_history_except_last)
                            .await;
                    }


                    let mut stream = agent
                        .stream_completion(current_prompt.clone(), (*chat_history.read().await).clone())
                        .await?
                        .stream()
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
                                yield Ok(MultiTurnStreamItem::text(&text.text));
                                did_call_tool = false;
                            },
                            Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
                                if let Some(ref hook) = req.hook {
                                    hook.on_tool_call(&tool_call.function.name, &tool_call.function.arguments.to_string()).await;
                                }
                                let tool_result =
                                    agent.tools.call(&tool_call.function.name, tool_call.function.arguments.to_string()).await?;

                                if let Some(ref hook) = req.hook {
                                    hook.on_tool_result(&tool_call.function.name, &tool_call.function.arguments.to_string(), &tool_result.to_string())
                                        .await;
                                }
                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id, tool_call.call_id, tool_result));

                                did_call_tool = true;
                                // break;
                            },
                            Ok(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, id })) => {
                                chat_history.write().await.push(rig::message::Message::Assistant {
                                    id: None,
                                    content: OneOrMany::one(AssistantContent::Reasoning(Reasoning {
                                        reasoning: reasoning.clone(), id
                                    }))
                                });
                                let text = reasoning.into_iter().collect::<Vec<String>>().join("");
                                yield Ok(MultiTurnStreamItem::text(&text));
                                did_call_tool = false;
                            },
                            Ok(StreamedAssistantContent::Final(final_resp)) => {
                                if is_text_response {
                                    if let Some(ref hook) = req.hook {
                                        hook.on_stream_completion_response_finish(&prompt, &final_resp).await;
                                    }
                                    yield Ok(MultiTurnStreamItem::text("\n"));
                                    is_text_response = false;
                                }
                                if let Some(usage) = final_resp.token_usage() { aggregated_usage += usage; };
                                // Do nothing here, since at the moment the final generic is actually unreachable.
                                // We need to implement a trait that aggregates token usage.
                                // TODO: Add a way to aggregate token responses from the generic variant
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
                        yield Ok(MultiTurnStreamItem::final_response(&last_text_response, aggregated_usage));
                        break;
                    }
                }

                    if max_depth_reached {
                        yield Err(PromptError::MaxDepthError {
                            max_depth: req.max_depth,
                            chat_history: (*chat_history.read().await).clone(),
                            prompt: last_prompt_error.into(),
                        }.into());
                    }

            })
        }

        inner(self, agent_name)
    }
}

impl<M, P> IntoFuture for StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: Send,
    P: PromptHook<M> + 'static,
{
    type Output = StreamingResult; // what `.await` returns
    type IntoFuture = Pin<Box<dyn futures::Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        // Wrap send() in a future, because send() returns a stream immediately
        Box::pin(async move { self.send().await })
    }
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout(
    stream: &mut StreamingResult,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::Text(Text { text })) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
        }
    }

    Ok(final_res)
}
