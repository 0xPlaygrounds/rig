use super::prompt_request::{self, PromptRequest, hooks::PromptHook};
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
use tokio::sync::RwLock as TokioRwLock;

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub type DynamicContextStore = Arc<
    TokioRwLock<
        Vec<(
            usize,
            Box<dyn crate::vector_store::VectorStoreIndexDyn + Send + Sync>,
        )>,
    >,
>;

/// Helper function to build a completion request from agent components.
/// This is used by both `Agent::completion()` and `PromptRequest::send()`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_completion_request<M: CompletionModel>(
    model: &Arc<M>,
    prompt: Message,
    chat_history: Vec<Message>,
    preamble: Option<&str>,
    static_context: &[Document],
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<&serde_json::Value>,
    tool_choice: Option<&ToolChoice>,
    tool_server_handle: &ToolServerHandle,
    dynamic_context: &DynamicContextStore,
    output_schema: Option<&schemars::Schema>,
) -> Result<CompletionRequestBuilder<M>, CompletionError> {
    // Find the latest message in the chat history that contains RAG text
    let rag_text = prompt.rag_text();
    let rag_text = rag_text.or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    let completion_request = model
        .completion_request(prompt)
        .messages(chat_history)
        .temperature_opt(temperature)
        .max_tokens_opt(max_tokens)
        .additional_params_opt(additional_params.cloned())
        .output_schema_opt(output_schema.cloned())
        .documents(static_context.to_vec());

    let completion_request = if let Some(preamble) = preamble {
        completion_request.preamble(preamble.to_owned())
    } else {
        completion_request
    };

    let completion_request = if let Some(tool_choice) = tool_choice {
        completion_request.tool_choice(tool_choice.clone())
    } else {
        completion_request
    };

    // If the agent has RAG text, we need to fetch the dynamic context and tools
    let result = match &rag_text {
        Some(text) => {
            let fetched_context = stream::iter(dynamic_context.read().await.iter())
                .then(|(num_sample, index)| async {
                    let req = VectorSearchRequest::builder()
                        .query(text)
                        .samples(*num_sample as u64)
                        .build()
                        .expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
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

            let tooldefs = tool_server_handle
                .get_tool_defs(Some(text.to_string()))
                .await
                .map_err(|_| {
                    CompletionError::RequestError("Failed to get tool definitions".into())
                })?;

            completion_request
                .documents(fetched_context)
                .tools(tooldefs)
        }
        None => {
            let tooldefs = tool_server_handle.get_tool_defs(None).await.map_err(|_| {
                CompletionError::RequestError("Failed to get tool definitions".into())
            })?;

            completion_request.tools(tooldefs)
        }
    };

    Ok(result)
}

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// The optional type parameter `P` represents a default hook that will be used for all
/// prompt requests unless overridden via `.with_hook()` on the request.
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
pub struct Agent<M, P = ()>
where
    M: CompletionModel,
    P: PromptHook<M>,
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
    /// Default maximum depth for recursive agent calls
    pub default_max_turns: Option<usize>,
    /// Default hook for this agent, used when no per-request hook is provided
    pub hook: Option<P>,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub output_schema: Option<schemars::Schema>,
}

impl<M, P> Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    /// Returns the name of the agent.
    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }
}

impl<M, P> Completion<M> for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    async fn completion(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        build_completion_request(
            &self.model,
            prompt.into(),
            chat_history,
            self.preamble.as_deref(),
            &self.static_context,
            self.temperature,
            self.max_tokens,
            self.additional_params.as_ref(),
            self.tool_choice.as_ref(),
            &self.tool_server_handle,
            &self.dynamic_context,
            self.output_schema.as_ref(),
        )
        .await
    }
}

// Here, we need to ensure that usage of `.prompt` on agent uses these redefinitions on the opaque
//  `Prompt` trait so that when `.prompt` is used at the call-site, it'll use the more specific
//  `PromptRequest` implementation for `Agent`, making the builder's usage fluent.
//
// References:
//  - https://github.com/rust-lang/rust/issues/121718 (refining_impl_trait)

#[allow(refining_impl_trait)]
impl<M, P> Prompt for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, prompt_request::Standard, M, P> {
        PromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> Prompt for &Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<'_, prompt_request::Standard, M, P> {
        PromptRequest::from_agent(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> Chat for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        mut chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        PromptRequest::from_agent(self, prompt)
            .with_history(&mut chat_history)
            .await
    }
}

impl<M, P> StreamingCompletion<M> for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
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

impl<M, P> StreamingPrompt<M, M::StreamingResponse> for Agent<M, P>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
    P: PromptHook<M> + 'static,
{
    type Hook = P;

    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> StreamingPromptRequest<M, P> {
        StreamingPromptRequest::<M, P>::from_agent(self, prompt)
    }
}

impl<M, P> StreamingChat<M, M::StreamingResponse> for Agent<M, P>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
    P: PromptHook<M> + 'static,
{
    type Hook = P;

    fn stream_chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> StreamingPromptRequest<M, P> {
        StreamingPromptRequest::<M, P>::from_agent(self, prompt).with_history(chat_history)
    }
}

use crate::{agent::prompt_request::TypedPromptRequest, completion::TypedPrompt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[allow(refining_impl_trait)]
impl<M, P> TypedPrompt for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type TypedRequest<'a, T>
        = TypedPromptRequest<'a, T, M, P>
    where
        Self: 'a,
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'a;

    /// Send a prompt and receive a typed structured response.
    ///
    /// The JSON schema for `T` is automatically generated and sent to the provider.
    /// Providers that support native structured outputs will constrain the model's
    /// response to match this schema.
    ///
    /// # Example
    /// ```rust,ignore
    /// use rig::prelude::*;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Debug, Deserialize, JsonSchema)]
    /// struct WeatherForecast {
    ///     city: String,
    ///     temperature_f: f64,
    ///     conditions: String,
    /// }
    ///
    /// let agent = client.agent("gpt-4o").build();
    ///
    /// // Type inferred from variable
    /// let forecast: WeatherForecast = agent
    ///     .prompt_typed("What's the weather in NYC?")
    ///     .await?;
    ///
    /// // Or explicit turbofish syntax
    /// let forecast = agent
    ///     .prompt_typed::<WeatherForecast>("What's the weather in NYC?")
    ///     .max_turns(3)
    ///     .await?;
    /// ```
    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<'_, T, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> TypedPrompt for &Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    type TypedRequest<'a, T>
        = TypedPromptRequest<'a, T, M, P>
    where
        Self: 'a,
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'a;

    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<'_, T, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(*self, prompt)
    }
}
