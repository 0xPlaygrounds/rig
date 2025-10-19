use super::prompt_request::{self, PromptRequest};
use crate::{
    agent::prompt_request::streaming::StreamingPromptRequest,
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        GetTokenUsage, Message, Prompt, PromptError,
    },
    message::ToolChoice,
    streaming::{StreamingChat, StreamingCompletion, StreamingPrompt},
    tool::server::ToolServerHandle,
    vector_store::{VectorStoreError, request::VectorSearchRequest},
    wasm_compat::WasmCompatSend,
};
use futures::{StreamExt, TryStreamExt, stream};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub type DynamicContextStore =
    Arc<RwLock<Vec<(usize, Box<dyn crate::vector_store::VectorStoreIndexDyn>)>>>;

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
#[derive(Clone)]
#[non_exhaustive]
pub struct Agent<M>
where
    M: CompletionModel,
{
    /// Name of the agent used for logging and debugging
    pub name: Option<String>,
    /// Agent description. Primarily useful when using sub-agents as part of an agent workflow and converting agents to other formats.
    pub description: Option<String>,
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    pub model: Arc<M>,
    /// System prompt
    pub preamble: Option<String>,
    /// Context documents always available to the agent
    pub static_context: Vec<Document>,
    /// Temperature of the model
    pub temperature: Option<f64>,
    /// Maximum number of tokens for the completion
    pub max_tokens: Option<u64>,
    /// Additional parameters to be passed to the model
    pub additional_params: Option<serde_json::Value>,
    pub tool_server_handle: ToolServerHandle,
    /// List of vector store, with the sample number
    pub dynamic_context: DynamicContextStore,
    /// Whether or not the underlying LLM should be forced to use a tool before providing a response.
    pub tool_choice: Option<ToolChoice>,
}

impl<M> Agent<M>
where
    M: CompletionModel,
{
    /// Returns the name of the agent.
    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }
}

impl<M> Completion<M> for Agent<M>
where
    M: CompletionModel,
{
    async fn completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
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
            .messages(chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());
        let completion_request = if let Some(preamble) = &self.preamble {
            completion_request.preamble(preamble.to_owned())
        } else {
            completion_request
        };

        // If the agent has RAG text, we need to fetch the dynamic context and tools
        let agent = match &rag_text {
            Some(text) => {
                let dynamic_context = stream::iter(self.dynamic_context.read().await.iter())
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

                let tooldefs = self
                    .tool_server_handle
                    .get_tool_defs(Some(text.to_string()))
                    .await
                    .map_err(|_| {
                        CompletionError::RequestError("Failed to get tool definitions".into())
                    })?;

                completion_request
                    .documents(dynamic_context)
                    .tools(tooldefs)
            }
            None => {
                let tooldefs = self
                    .tool_server_handle
                    .get_tool_defs(None)
                    .await
                    .map_err(|_| {
                        CompletionError::RequestError("Failed to get tool definitions".into())
                    })?;

                completion_request.tools(tooldefs)
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
impl<M> Prompt for Agent<M>
where
    M: CompletionModel,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, prompt_request::Standard, M, ()> {
        PromptRequest::new(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Prompt for &Agent<M>
where
    M: CompletionModel,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, prompt_request::Standard, M, ()> {
        PromptRequest::new(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M> Chat for Agent<M>
where
    M: CompletionModel,
{
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        mut chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        PromptRequest::new(self, prompt)
            .with_history(&mut chat_history)
            .await
    }
}

impl<M> StreamingCompletion<M> for Agent<M>
where
    M: CompletionModel,
{
    async fn stream_completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
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
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M, ()> {
        let arc = Arc::new(self.clone());
        StreamingPromptRequest::new(arc, prompt)
    }
}

impl<M> StreamingChat<M, M::StreamingResponse> for Agent<M>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> StreamingPromptRequest<M, ()> {
        let arc = Arc::new(self.clone());
        StreamingPromptRequest::new(arc, prompt).with_history(chat_history)
    }
}
