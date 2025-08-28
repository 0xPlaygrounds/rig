pub(crate) mod streaming;

use std::{future::IntoFuture, marker::PhantomData};

use futures::{FutureExt, StreamExt, future::BoxFuture, stream};

use crate::{
    OneOrMany,
    completion::{Completion, CompletionError, CompletionModel, Message, PromptError, Usage},
    message::{AssistantContent, UserContent},
    tool::ToolSetError,
};

use super::Agent;

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
/// [`crate::completion::request::PromptError::MaxDepthError`] if the agent decides to call tools
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
    max_depth: usize,
    /// The agent to use for execution
    agent: &'a Agent<M>,
    /// Phantom data to track the type of the request
    state: PhantomData<S>,
    /// Optional per-request hook for events
    hook: Option<P>,
}

impl<'a, M: CompletionModel> PromptRequest<'a, Standard, M, ()> {
    /// Create a new PromptRequest with the given prompt and model
    pub fn new(agent: &'a Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: 0,
            agent,
            state: PhantomData,
            hook: None,
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
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
        }
    }
    /// Set the maximum depth for multi-turn conversations (ie, the maximum number of turns an LLM can have calling tools before writing a text response).
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxDepthError`].
    pub fn multi_turn(self, depth: usize) -> PromptRequest<'a, S, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
        }
    }

    /// Add chat history to the prompt request
    pub fn with_history(self, history: &'a mut Vec<Message>) -> PromptRequest<'a, S, M, P> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: Some(history),
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: self.hook,
        }
    }

    /// Attach a per-request hook for tool call events
    pub fn with_hook<P2>(self, hook: P2) -> PromptRequest<'a, S, M, P2>
    where
        P2: PromptHook<M>,
    {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
            hook: Some(hook),
        }
    }
}

// dead code allowed because of functions being left empty to allow for users to not have to implement every single function
/// Trait for per-request hooks to observe tool call events.
pub trait PromptHook<M: CompletionModel>: Clone + Send + Sync {
    #[allow(unused_variables)]
    /// Called before the prompt is sent to the model
    fn on_completion_call(
        &self,
        prompt: &Message,
        history: &[Message],
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called after the prompt is sent to the model and a response is received.
    /// This function is for non-streamed responses. Please refer to `on_stream_completion_response_finish` for streamed responses.
    fn on_completion_response(
        &self,
        prompt: &Message,
        response: &crate::completion::CompletionResponse<M::Response>,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called after the model provider has finished streaming a text response from their completion API to the client.
    fn on_stream_completion_response_finish(
        &self,
        prompt: &Message,
        response: &<M as CompletionModel>::StreamingResponse,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called before a tool is invoked.
    fn on_tool_call(&self, tool_name: &str, args: &str) -> impl Future<Output = ()> + Send {
        async {}
    }

    #[allow(unused_variables)]
    /// Called after a tool is invoked (and a result has been returned).
    fn on_tool_result(
        &self,
        tool_name: &str,
        args: &str,
        result: &str,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

impl<M> PromptHook<M> for () where M: CompletionModel {}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl<'a, M, P> IntoFuture for PromptRequest<'a, Standard, M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type Output = Result<String, PromptError>;
    type IntoFuture = BoxFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        self.send().boxed()
    }
}

impl<'a, M, P> IntoFuture for PromptRequest<'a, Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = BoxFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        self.send().boxed()
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
    pub total_usage: Usage,
}

impl PromptResponse {
    pub fn new(output: impl Into<String>, total_usage: Usage) -> Self {
        Self {
            output: output.into(),
            total_usage,
        }
    }
}

impl<M, P> PromptRequest<'_, Extended, M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    #[tracing::instrument(skip(self), fields(agent_name = self.agent.name()))]
    async fn send(self) -> Result<PromptResponse, PromptError> {
        let agent = self.agent;
        let chat_history = if let Some(history) = self.chat_history {
            history.push(self.prompt);
            history
        } else {
            &mut vec![self.prompt]
        };

        let mut current_max_depth = 0;
        let mut usage = Usage::new();

        // We need to do at least 2 loops for 1 roundtrip (user expects normal message)
        let last_prompt = loop {
            let prompt = chat_history
                .last()
                .cloned()
                .expect("there should always be at least one message in the chat history");

            if current_max_depth > self.max_depth + 1 {
                break prompt;
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
                hook.on_completion_call(&prompt, &chat_history[..chat_history.len() - 1])
                    .await;
            }

            let resp = agent
                .completion(
                    prompt.clone(),
                    chat_history[..chat_history.len() - 1].to_vec(),
                )
                .await?
                .send()
                .await?;

            usage += resp.usage;

            if let Some(ref hook) = self.hook {
                hook.on_completion_response(&prompt, &resp).await;
            }

            let (tool_calls, texts): (Vec<_>, Vec<_>) = resp
                .choice
                .iter()
                .partition(|choice| matches!(choice, AssistantContent::ToolCall(_)));

            chat_history.push(Message::Assistant {
                id: None,
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

                if self.max_depth > 1 {
                    tracing::info!("Depth reached: {}/{}", current_max_depth, self.max_depth);
                }

                // If there are no tool calls, depth is not relevant, we can just return the merged text response.
                return Ok(PromptResponse::new(merged_texts, usage));
            }

            let hook = self.hook.clone();
            let tool_content = stream::iter(tool_calls)
                .then(|choice| {
                    let hook1 = hook.clone();
                    let hook2 = hook.clone();
                    async move {
                        if let AssistantContent::ToolCall(tool_call) = choice {
                            let tool_name = &tool_call.function.name;
                            let args = tool_call.function.arguments.to_string();
                            if let Some(hook) = hook1 {
                                hook.on_tool_call(tool_name, &args).await;
                            }
                            let output = agent.tools.call(tool_name, args.clone()).await?;
                            if let Some(hook) = hook2 {
                                hook.on_tool_result(tool_name, &args, &output.to_string())
                                    .await;
                            }
                            if let Some(call_id) = tool_call.call_id.clone() {
                                Ok(UserContent::tool_result_with_call_id(
                                    tool_call.id.clone(),
                                    call_id,
                                    OneOrMany::one(output.into()),
                                ))
                            } else {
                                Ok(UserContent::tool_result(
                                    tool_call.id.clone(),
                                    OneOrMany::one(output.into()),
                                ))
                            }
                        } else {
                            unreachable!(
                                "This should never happen as we already filtered for `ToolCall`"
                            )
                        }
                    }
                })
                .collect::<Vec<Result<UserContent, ToolSetError>>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

            chat_history.push(Message::User {
                content: OneOrMany::many(tool_content).expect("There is atleast one tool call"),
            });
        };

        // If we reach here, we never resolved the final tool call. We need to do ... something.
        Err(PromptError::MaxDepthError {
            max_depth: self.max_depth,
            chat_history: chat_history.clone(),
            prompt: last_prompt,
        })
    }
}
