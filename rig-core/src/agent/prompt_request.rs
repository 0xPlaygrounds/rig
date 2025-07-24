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
/// If you expect to continously call tools, you will want to ensure you use the `.multi_turn()`
/// argument to add more turns as by default, it is 0 (meaning only 1 tool round-trip). Otherwise,
/// attempting to await (which will send the prompt request) can potentially return
/// [`crate::completion::request::PromptError::MaxDepthError`] if the agent decides to call tools
/// back to back.
pub struct PromptRequest<'a, S: PromptType, M: CompletionModel> {
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
}

impl<'a, M: CompletionModel> PromptRequest<'a, Standard, M> {
    /// Create a new PromptRequest with the given prompt and model
    pub fn new(agent: &'a Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_depth: 0,
            agent,
            state: PhantomData,
        }
    }

    /// Enable returning extended details for responses (includes aggregated token usage)
    ///
    /// Note: This changes the type of the response from `.send` to return a `PromptResponse` struct
    /// instead of a simple `String`. This is useful for tracking token usage across multiple turns
    /// of conversation.
    pub fn extended_details(self) -> PromptRequest<'a, Extended, M> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
        }
    }
}

impl<'a, S: PromptType, M: CompletionModel> PromptRequest<'a, S, M> {
    /// Set the maximum depth for multi-turn conversations (ie, the maximum number of turns an LLM can have calling tools before writing a text response).
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxDepthError`].
    pub fn multi_turn(self, depth: usize) -> PromptRequest<'a, S, M> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_depth: depth,
            agent: self.agent,
            state: PhantomData,
        }
    }

    /// Add chat history to the prompt request
    pub fn with_history(self, history: &'a mut Vec<Message>) -> PromptRequest<'a, S, M> {
        PromptRequest {
            prompt: self.prompt,
            chat_history: Some(history),
            max_depth: self.max_depth,
            agent: self.agent,
            state: PhantomData,
        }
    }
}

/// Due to: [RFC 2515](https://github.com/rust-lang/rust/issues/63063), we have to use a `BoxFuture`
///  for the `IntoFuture` implementation. In the future, we should be able to use `impl Future<...>`
///  directly via the associated type.
impl<'a, M: CompletionModel> IntoFuture for PromptRequest<'a, Standard, M> {
    type Output = Result<String, PromptError>;
    type IntoFuture = BoxFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        self.send().boxed()
    }
}

impl<'a, M: CompletionModel> IntoFuture for PromptRequest<'a, Extended, M> {
    type Output = Result<PromptResponse, PromptError>;
    type IntoFuture = BoxFuture<'a, Self::Output>; // This future should not outlive the agent

    fn into_future(self) -> Self::IntoFuture {
        self.send().boxed()
    }
}

impl<M: CompletionModel> PromptRequest<'_, Standard, M> {
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

impl<M: CompletionModel> PromptRequest<'_, Extended, M> {
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

        // We need to do atleast 2 loops for 1 roundtrip (user expects normal message)
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

            let resp = agent
                .completion(prompt, chat_history[..chat_history.len() - 1].to_vec())
                .await?
                .send()
                .await?;

            usage += resp.usage;

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

            let tool_content = stream::iter(tool_calls)
                .then(|choice| async move {
                    if let AssistantContent::ToolCall(tool_call) = choice {
                        let output = agent
                            .tools
                            .call(
                                &tool_call.function.name,
                                tool_call.function.arguments.to_string(),
                            )
                            .await?;
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
