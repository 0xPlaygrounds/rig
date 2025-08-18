use crate::{
    OneOrMany,
    message::{AssistantContent, ToolResultContent, UserContent},
    streaming::StreamingCompletion,
};
use futures::{Stream, StreamExt};
use std::{pin::Pin, sync::Arc};
use tokio::sync::RwLock;

use crate::{
    agent::Agent,
    completion::{CompletionError, CompletionModel, PromptError},
    message::{Message, Text},
    streaming::StreamedAssistantContent,
    tool::ToolSetError,
};

type StreamingResult<'a> = Pin<Box<dyn Stream<Item = Result<Text, StreamingError>> + Send + 'a>>;

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
pub struct StreamingPromptRequest<'a, M>
where
    M: CompletionModel,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history to include with the prompt
    /// Note: chat history needs to outlive the agent as it might be used with other agents
    chat_history: Option<Vec<Message>>,
    /// Maximum depth for multi-turn conversations (0 means no multi-turn)
    max_depth: usize,
    /// The agent to use for execution
    agent: &'a Agent<M>,
    #[cfg(feature = "hooks")]
    /// Optional per-request hook for events
    hook: Option<&'a dyn PromptHook<M>>,
}

impl<'a, M> StreamingPromptRequest<'a, M>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: Send,
{
    /// Create a new PromptRequest with the given prompt and model
    pub fn new(agent: &'a Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: 0,
            agent,
            #[cfg(feature = "hooks")]
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

    #[cfg(feature = "hooks")]
    /// Attach a per-request hook for tool call events
    pub fn with_hook(self, hook: &'a dyn PromptHook<M>) -> PromptRequest<'a, S, M> {
        self.hook = Some(hook);
        self
    }

    fn send(self) -> StreamingResult<'a> {
        let agent_name = self.agent.name_owned();

        #[tracing::instrument(skip_all, fields(agent_name = agent_name))]
        fn inner<'a, M>(
            req: StreamingPromptRequest<'a, M>,
            agent_name: String,
        ) -> StreamingResult<'a>
        where
            M: CompletionModel + 'static,
            <M as CompletionModel>::StreamingResponse: Send,
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

            Box::pin(async_stream::stream! {
                let mut current_prompt = prompt;
                let mut did_call_tool = false;

                'outer: loop {
                    if current_max_depth > req.max_depth + 1 {
                        last_prompt_error = current_prompt.rag_text().unwrap_or_default();
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
                                yield Ok(Text { text: text.text });
                                did_call_tool = false;
                            },
                            Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
                                let tool_result =
                                    agent.tools.call(&tool_call.function.name, tool_call.function.arguments.to_string()).await?;

                                let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                                tool_calls.push(tool_call_msg);
                                tool_results.push((tool_call.id, tool_call.call_id, tool_result));

                                did_call_tool = true;
                                // break;
                            },
                            Ok(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, .. })) => {
                                let text = reasoning.into_iter().collect::<Vec<String>>().join("");
                                yield Ok(Text { text });
                                did_call_tool = false;
                            },
                            Ok(StreamedAssistantContent::Final(_)) => {
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
                        break;
                    }
                }

                    yield Err(PromptError::MaxDepthError {
                        max_depth: req.max_depth,
                        chat_history: (*chat_history.read().await).clone(),
                        prompt: last_prompt_error.into(),
                    }.into());

            })
        }

        inner(self, agent_name)
    }
}

impl<'a, M> IntoFuture for StreamingPromptRequest<'a, M>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: Send,
{
    type Output = StreamingResult<'a>; // what `.await` returns
    type IntoFuture = Pin<Box<dyn futures::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        // Wrap send() in a future, because send() returns a stream immediately
        Box::pin(async move { self.send() })
    }
}

/// helper function to stream a completion request to stdout
pub async fn stream_to_stdout(stream: &mut StreamingResult<'_>) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(Text { text }) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
        }
    }
    println!(); // New line after streaming completes

    Ok(())
}
