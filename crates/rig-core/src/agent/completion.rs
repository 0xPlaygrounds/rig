use super::prompt_request::{self, PromptRequest, hooks::PromptHook};
use crate::{
    agent::prompt_request::streaming::StreamingPromptRequest,
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        GetTokenUsage, Message, Prompt, PromptError, TypedPrompt,
    },
    message::ToolChoice,
    streaming::{StreamingChat, StreamingCompletion, StreamingPrompt},
    tool::server::ToolServerHandle,
    vector_store::{VectorStoreError, request::VectorSearchRequest},
    wasm_compat::WasmCompatSend,
};
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub type DynamicContextStore = Arc<
    Vec<(
        usize,
        Arc<dyn crate::vector_store::VectorStoreIndexDyn + Send + Sync>,
    )>,
>;

/// A prepared completion request plus the executable Rig tool names advertised
/// to the provider for this turn.
pub(crate) struct PreparedCompletionRequest<M: CompletionModel> {
    pub(crate) builder: CompletionRequestBuilder<M>,
    pub(crate) executable_tool_names: BTreeSet<String>,
    pub(crate) allowed_tool_names: BTreeSet<String>,
}

pub(crate) fn allowed_tool_names_for_choice(
    executable_tool_names: &BTreeSet<String>,
    tool_choice: Option<&ToolChoice>,
) -> Result<BTreeSet<String>, CompletionError> {
    let allowed = match tool_choice {
        None | Some(ToolChoice::Auto | ToolChoice::Required) => executable_tool_names.clone(),
        Some(ToolChoice::None) => BTreeSet::new(),
        Some(ToolChoice::Specific { function_names }) => {
            if function_names.is_empty() {
                return Err(CompletionError::RequestError(
                    "ToolChoice::Specific requires at least one function name".into(),
                ));
            }

            let requested = function_names.iter().cloned().collect::<BTreeSet<String>>();
            // Validation re-runs every turn against the tools available that
            // turn, and the available set can legitimately shift mid-run (an
            // MCP `tools/list_changed` swap, dynamic retrieval). So only error
            // when NONE of the requested names are available — nothing can be
            // forced; otherwise proceed with whichever named tools are present
            // and warn about the rest.
            let allowed = requested
                .intersection(executable_tool_names)
                .cloned()
                .collect::<BTreeSet<String>>();

            if allowed.is_empty() {
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Specific requested only unavailable tool names: {requested:?}. Available tools: {:?}",
                        executable_tool_names.iter().collect::<Vec<_>>()
                    )
                    .into(),
                ));
            }

            let missing = requested
                .difference(executable_tool_names)
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                tracing::warn!(
                    "ToolChoice::Specific names tools not currently available (proceeding with the rest): {missing:?}"
                );
            }

            allowed
        }
    };

    Ok(allowed)
}

/// Helper function to build a completion request from agent components.
/// This is used by `Agent::completion()` to preserve the public completion API.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_completion_request<M: CompletionModel>(
    model: &Arc<M>,
    prompt: Message,
    chat_history: &[Message],
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
    Ok(build_prepared_completion_request(
        model,
        prompt,
        chat_history,
        preamble,
        static_context,
        temperature,
        max_tokens,
        additional_params,
        tool_choice,
        tool_choice,
        tool_server_handle,
        dynamic_context,
        output_schema,
    )
    .await?
    .builder)
}

/// Helper function to build a completion request from agent components while
/// preserving the executable Rig tool names sent to the provider.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_prepared_completion_request<M: CompletionModel>(
    model: &Arc<M>,
    prompt: Message,
    chat_history: &[Message],
    preamble: Option<&str>,
    static_context: &[Document],
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<&serde_json::Value>,
    tool_choice: Option<&ToolChoice>,
    validation_tool_choice: Option<&ToolChoice>,
    tool_server_handle: &ToolServerHandle,
    dynamic_context: &DynamicContextStore,
    output_schema: Option<&schemars::Schema>,
) -> Result<PreparedCompletionRequest<M>, CompletionError> {
    // `tool_choice` is the WIRE choice (may be relaxed to Auto after the first
    // tool-call turn); `validation_tool_choice` is the configured choice that
    // bounds which tool calls Rig accepts. They differ only for relaxed
    // `Required`/`Specific` runs; `build_completion_request` passes the same
    // value for both.
    // Find the latest message in the chat history that contains RAG text
    let rag_text = prompt.rag_text();
    let rag_text = rag_text.or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    // Prepend preamble as system message if present
    let chat_history: Vec<Message> = if let Some(preamble) = preamble {
        std::iter::once(Message::system(preamble.to_owned()))
            .chain(chat_history.iter().cloned())
            .collect()
    } else {
        chat_history.to_vec()
    };

    let completion_request = model
        .completion_request(prompt)
        .messages(chat_history)
        .temperature_opt(temperature)
        .max_tokens_opt(max_tokens)
        .additional_params_opt(additional_params.cloned())
        .output_schema_opt(output_schema.cloned())
        .documents(static_context.to_vec());

    let completion_request = if let Some(tool_choice) = tool_choice {
        completion_request.tool_choice(tool_choice.clone())
    } else {
        completion_request
    };

    // If the agent has RAG text, we need to fetch the dynamic context and tools
    let (builder, executable_tool_names) = match &rag_text {
        Some(text) => {
            // Map over the vector to create async tasks
            let search_futures = dynamic_context.iter().map(|(num_sample, index)| {
                // Clone values to move into the async block
                let text = text.clone();
                let num_sample = *num_sample;
                let index = index.clone();

                async move {
                    let req = VectorSearchRequest::builder()
                        .query(text)
                        .samples(num_sample as u64)
                        .build();

                    let docs = index
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
                        .collect::<Vec<_>>();

                    Ok::<_, VectorStoreError>(docs)
                }
            });

            // Await all vector searches concurrently
            let fetched_context: Vec<Document> = futures::future::try_join_all(search_futures)
                .await
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?
                .into_iter()
                .flatten() // Flatten the Vec<Vec<Document>> into Vec<Document>
                .collect();

            let tooldefs = tool_server_handle
                .get_tool_defs(Some(text.to_string()))
                .await
                .map_err(|_| {
                    CompletionError::RequestError("Failed to get tool definitions".into())
                })?;
            let executable_tool_names = tooldefs.iter().map(|tool| tool.name.clone()).collect();

            (
                completion_request
                    .documents(fetched_context)
                    .tools(tooldefs),
                executable_tool_names,
            )
        }
        None => {
            let tooldefs = tool_server_handle.get_tool_defs(None).await.map_err(|_| {
                CompletionError::RequestError("Failed to get tool definitions".into())
            })?;
            let executable_tool_names = tooldefs.iter().map(|tool| tool.name.clone()).collect();

            (completion_request.tools(tooldefs), executable_tool_names)
        }
    };
    let allowed_tool_names =
        allowed_tool_names_for_choice(&executable_tool_names, validation_tool_choice)?;

    Ok(PreparedCompletionRequest {
        builder,
        executable_tool_names,
        allowed_tool_names,
    })
}

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// The optional type parameter `P` represents a default hook that will be used for all
/// prompt requests unless overridden via `.with_hook()` on the request.
///
/// # Example
/// ```no_run
/// use rig_core::{
///     client::{CompletionClient, ProviderClient},
///     completion::Prompt,
///     providers::openai,
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let openai = openai::Client::from_env()?;
///
/// let comedian_agent = openai
///     .agent(openai::GPT_5_2)
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!").await?;
/// # Ok(())
/// # }
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
    /// Optional conversation memory backend that loads/saves history per conversation id.
    pub memory: Option<Arc<dyn crate::memory::ConversationMemory>>,
    /// Optional default conversation id used when none is set per-request.
    pub default_conversation_id: Option<String>,
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
    async fn completion<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError>
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        let history: Vec<Message> = chat_history.into_iter().map(Into::into).collect();
        build_completion_request(
            &self.model,
            prompt.into(),
            &history,
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
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard, M, P> {
        PromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> Prompt for &Agent<M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    #[tracing::instrument(skip(self, prompt), fields(agent_name = self.name()))]
    fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> PromptRequest<prompt_request::Standard, M, P> {
        PromptRequest::from_agent(*self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> Chat for Agent<M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    #[tracing::instrument(skip(self, prompt, chat_history), fields(agent_name = self.name()))]
    async fn chat(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let response = PromptRequest::from_agent(self, prompt)
            .with_history(chat_history.clone())
            .extended_details()
            .await?;

        if let Some(messages) = response.messages {
            chat_history.extend(messages);
        }

        Ok(response.output)
    }
}

impl<M, P> StreamingCompletion<M> for Agent<M, P>
where
    M: CompletionModel,
    P: PromptHook<M>,
{
    async fn stream_completion<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError>
    where
        I: IntoIterator<Item = T> + WasmCompatSend,
        T: Into<Message>,
    {
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

    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<M, P>
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        StreamingPromptRequest::<M, P>::from_agent(self, prompt).with_history(chat_history)
    }
}

use crate::agent::prompt_request::TypedPromptRequest;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

#[allow(refining_impl_trait)]
impl<M, P> TypedPrompt for Agent<M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    /// Send a prompt and receive a typed structured response.
    ///
    /// The JSON schema for `T` is automatically generated and sent to the provider.
    /// Providers that support native structured outputs will constrain the model's
    /// response to match this schema.
    ///
    /// # Example
    /// ```rust,ignore
    /// use rig_core::prelude::*;
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
    ) -> TypedPromptRequest<T, prompt_request::Standard, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(self, prompt)
    }
}

#[allow(refining_impl_trait)]
impl<M, P> TypedPrompt for &Agent<M, P>
where
    M: CompletionModel + 'static,
    P: PromptHook<M> + 'static,
{
    type TypedRequest<T>
        = TypedPromptRequest<T, prompt_request::Standard, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend + 'static;

    fn prompt_typed<T>(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> TypedPromptRequest<T, prompt_request::Standard, M, P>
    where
        T: JsonSchema + DeserializeOwned + WasmCompatSend,
    {
        TypedPromptRequest::from_agent(*self, prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn allowed_tool_names_defaults_to_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, None).unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_auto_and_required_allow_all_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Auto)).unwrap(),
            executable
        );
        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::Required)).unwrap(),
            executable
        );
    }

    #[test]
    fn allowed_tool_names_none_allows_no_tools() {
        let executable = tool_names(&["add", "subtract"]);

        assert!(
            allowed_tool_names_for_choice(&executable, Some(&ToolChoice::None))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn allowed_tool_names_specific_allows_requested_executable_tools() {
        let executable = tool_names(&["add", "subtract"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["add".to_string()],
        };

        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&choice)).unwrap(),
            tool_names(&["add"])
        );
    }

    #[test]
    fn allowed_tool_names_specific_rejects_all_missing_tools() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["missing".to_string()],
        };

        // None of the requested names are available, so nothing can be
        // forced: error before the provider request.
        let err = allowed_tool_names_for_choice(&executable, Some(&choice))
            .expect_err("an all-unavailable specific choice should fail");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("missing")
                    && err.to_string().contains("add")
        ));
    }

    #[test]
    fn allowed_tool_names_specific_proceeds_with_available_subset() {
        let executable = tool_names(&["add", "subtract"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["add".to_string(), "vanished".to_string()],
        };

        // "vanished" is no longer available (e.g. dropped by an MCP
        // tools/list_changed refresh mid-run); validation proceeds with the
        // available subset instead of failing the whole run.
        assert_eq!(
            allowed_tool_names_for_choice(&executable, Some(&choice)).unwrap(),
            tool_names(&["add"])
        );
    }

    #[test]
    fn allowed_tool_names_specific_rejects_empty_names() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec![],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice))
            .expect_err("empty specific tool choice should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("requires at least one function name")
        ));
    }
}
