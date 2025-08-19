use super::prompt_request::{self, PromptRequest};
use crate::{
    agent::prompt_request::streaming::StreamingPromptRequest,
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        GetTokenUsage, Message, Prompt, PromptError,
    },
    streaming::{StreamingChat, StreamingCompletion, StreamingPrompt},
    tool::ToolSet,
    vector_store::{VectorStoreError, request::VectorSearchRequest},
};
use futures::{StreamExt, TryStreamExt, stream};
use std::collections::HashMap;

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// # Example
/// ```
/// use rig::{completion::Prompt, providers::openai};
///
/// let openai = openai::Client::from_env();
///
/// let comedian_agent = openai
///     .agent("gpt-4o")
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!")
///     .await
///     .expect("Failed to prompt the agent");
/// ```
#[non_exhaustive]
pub struct Agent<M: CompletionModel> {
    /// Name of the agent used for logging and debugging
    pub name: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub model: M,
    /// System prompt
    pub preamble: String,
    /// Context documents always available to the agent
    pub static_context: Vec<Document>,
    /// Tools that are always available to the agent (identified by their name)
    pub static_tools: Vec<String>,
    /// Temperature of the model
    pub temperature: Option<f64>,
    /// Maximum number of tokens for the completion
    pub max_tokens: Option<u64>,
    /// Additional parameters to be passed to the model
    pub additional_params: Option<serde_json::Value>,
    /// List of vector store, with the sample number
    pub dynamic_context: Vec<(usize, Box<dyn crate::vector_store::VectorStoreIndexDyn>)>,
    /// Dynamic tools
    pub dynamic_tools: Vec<(usize, Box<dyn crate::vector_store::VectorStoreIndexDyn>)>,
    /// Actual tool implementations
    pub tools: ToolSet,
}

impl<M: CompletionModel> Completion<M> for Agent<M> {
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let prompt = prompt.into();

        // Find the latest message in the chat history that contains RAG text
        let rag_text = prompt.rag_text();
        let rag_text = rag_text.or_else(|| {
            chat_history
                .iter()
                .rev()
                .find_map(|message| message.rag_text())
        });

        let completion_request = self
            .model
            .completion_request(prompt)
            .preamble(self.preamble.clone())
            .messages(chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());

        // If the agent has RAG text, we need to fetch the dynamic context and tools
        let agent = match &rag_text {
            Some(text) => {
                let dynamic_context = stream::iter(self.dynamic_context.iter())
                    .then(|(num_sample, index)| async {
                        let req = VectorSearchRequest::builder().query(text).samples(*num_sample as u64).build().expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n(req)
                                .await?
                                .into_iter()
                                .map(|(_, id, doc)| {
                                    // Pretty print the document if possible for better readability
                                    let text = serde_json::to_string_pretty(&doc)
                                        .unwrap_or_else(|_| doc.to_string());

                                    Document {
                                        id,
                                        text,
                                        additional_props: HashMap::new(),
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        acc.extend(docs);
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let dynamic_tools = stream::iter(self.dynamic_tools.iter())
                    .then(|(num_sample, index)| async {
                        let req = VectorSearchRequest::builder().query(text).samples(*num_sample as u64).build().expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n_ids(req)
                                .await?
                                .into_iter()
                                .map(|(_, id)| id)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        for doc in docs {
                            if let Some(tool) = self.tools.get(&doc) {
                                acc.push(tool.definition(text.into()).await)
                            } else {
                                tracing::warn!("Tool implementation not found in toolset: {}", doc);
                            }
                        }
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            Some(tool.definition(text.into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request
                    .documents(dynamic_context)
                    .tools([static_tools.clone(), dynamic_tools].concat())
            }
            None => {
                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            // TODO: tool definitions should likely take an `Option<String>`
                            Some(tool.definition("".into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request.tools(static_tools)
            }
        };

        Ok(agent)
    }
}

// Here, we need to ensure that usage of `.prompt` on agent uses these redefinitions on the opaque
//  `Prompt` trait so that when `.prompt` is used at the call-site, it'll use the more specific
//  `PromptRequest` implementation for `Agent`, making the builder's usage fluent.
//
// References:
//  - https://github.com/rust-lang/rust/issues/121718 (refining_impl_trait)

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for Agent<M> {
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> PromptRequest<'_, prompt_request::Standard, M> {
        PromptRequest::new(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Prompt for &Agent<M> {
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> PromptRequest<'_, prompt_request::Standard, M> {
        PromptRequest::new(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M: CompletionModel> Chat for Agent<M> {
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let mut cloned_history = chat_history.clone();
        PromptRequest::new(self, prompt)
            .with_history(&mut cloned_history)
            .await
    }
}

impl<M: CompletionModel> StreamingCompletion<M> for Agent<M> {
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        // Reuse the existing completion implementation to build the request
        // This ensures streaming and non-streaming use the same request building logic
        self.completion(prompt, chat_history).await
    }
}

impl<M> StreamingPrompt<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn stream_prompt(&self, prompt: impl Into<Message> + Send) -> StreamingPromptRequest<'_, M> {
        StreamingPromptRequest::new(self, prompt)
    }
}

impl<M> StreamingChat<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> StreamingPromptRequest<'_, M> {
        StreamingPromptRequest::new(self, prompt).with_history(chat_history)
    }
}

impl<M: CompletionModel> Agent<M> {
    /// Returns the name of the agent.
    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Returns the name of the agent as an owned variable.
    /// Useful in some cases where having the agent name as an owned variable is required.
    pub(crate) fn name_owned(&self) -> String {
        self.name.clone().unwrap_or(UNKNOWN_AGENT_NAME.to_string())
    }
}
