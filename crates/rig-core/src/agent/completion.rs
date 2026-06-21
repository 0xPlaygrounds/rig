use super::prompt_request::{self, PromptRequest, hooks::PromptHook};
use super::run::OutputMode;
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
    /// When Tool output mode is active, the name of the synthetic output tool
    /// advertised to the model (allowed but not executable). See #1928.
    pub(crate) output_tool_name: Option<String>,
}

/// Base name of the synthetic output tool used by [`OutputMode::Tool`].
const DEFAULT_OUTPUT_TOOL_NAME: &str = "final_result";

/// Whether the active [`ToolChoice`] lets the model call the synthetic output
/// tool. Tool output mode finalizes via that call, so when the choice forbids it
/// (`None`, or a `Specific` allow-list that lists only the caller's real tools)
/// Tool mode cannot work and must fall back to native structured output.
fn tool_choice_permits_output_tool(tool_choice: Option<&ToolChoice>) -> bool {
    matches!(
        tool_choice,
        None | Some(ToolChoice::Auto | ToolChoice::Required)
    )
}

/// Resolve the caller-facing [`OutputMode`] to a concrete mode for one request.
///
/// With no schema there is nothing to enforce, so the result is always `Native`
/// (the synthetic tool and prompt injection only make sense with a schema).
/// `Auto` becomes `Tool` only when a real executable tool is present, the tool
/// choice permits the output-tool call, AND the provider does *not* compose
/// native structured output with tools — i.e. only where the native constraint
/// would actually suppress tool calls (#1928). On providers that compose them
/// (OpenAI, Anthropic), `Auto` keeps guaranteed native structured output.
/// `Tool` (explicit or via `Auto`) requires that the active [`ToolChoice`]
/// permit the output-tool call; when it does not, it degrades to `Native` so
/// structured output is still enforced rather than silently dropped. Explicit
/// `Prompted`/`Native` are honored when a schema is present. The returned mode is
/// never `Auto`.
fn resolve_output_mode(
    has_schema: bool,
    has_executable_tools: bool,
    output_tool_callable: bool,
    provider_composes_native: bool,
    requested: &OutputMode,
) -> OutputMode {
    if !has_schema {
        return OutputMode::Native;
    }
    match requested {
        OutputMode::Native => OutputMode::Native,
        OutputMode::Prompted => OutputMode::Prompted,
        OutputMode::Tool if output_tool_callable => OutputMode::Tool,
        OutputMode::Tool => OutputMode::Native,
        OutputMode::Auto
            if has_executable_tools && output_tool_callable && !provider_composes_native =>
        {
            OutputMode::Tool
        }
        OutputMode::Auto => OutputMode::Native,
    }
}

/// Pick a collision-safe name for the synthetic output tool, never shadowing a
/// real executable tool (which would make the model's output call dispatchable).
fn pick_output_tool_name(executable_tool_names: &BTreeSet<String>) -> String {
    let mut name = DEFAULT_OUTPUT_TOOL_NAME.to_string();
    let mut suffix = 1u32;
    while executable_tool_names.contains(&name) {
        name = format!("{DEFAULT_OUTPUT_TOOL_NAME}_{suffix}");
        suffix += 1;
    }
    name
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
            let missing = requested
                .difference(executable_tool_names)
                .cloned()
                .collect::<Vec<_>>();

            if !missing.is_empty() {
                return Err(CompletionError::RequestError(
                    format!(
                        "ToolChoice::Specific requested unknown tool names: {missing:?}. Available tools: {:?}",
                        executable_tool_names.iter().collect::<Vec<_>>()
                    )
                    .into(),
                ));
            }

            requested
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
        tool_server_handle,
        dynamic_context,
        output_schema,
        // The single-shot `Agent::completion()` API has no run loop to consume an
        // output-tool call, so it always uses native structured output (#1928).
        &OutputMode::Native,
        None,
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
    tool_server_handle: &ToolServerHandle,
    dynamic_context: &DynamicContextStore,
    output_schema: Option<&schemars::Schema>,
    output_mode: &OutputMode,
    committed_output_tool: Option<&str>,
) -> Result<PreparedCompletionRequest<M>, CompletionError> {
    // Find the latest message in the chat history that contains RAG text
    let rag_text = prompt.rag_text();
    let rag_text = rag_text.or_else(|| {
        chat_history
            .iter()
            .rev()
            .find_map(|message| message.rag_text())
    });

    // Fetch dynamic (RAG) documents and the real executable tool set first, so we
    // can resolve the output mode (which depends on whether tools exist) before
    // building the preamble and request.
    let (mut tooldefs, fetched_context): (Vec<crate::completion::ToolDefinition>, Vec<Document>) =
        match &rag_text {
            Some(text) => {
                let search_futures = dynamic_context.iter().map(|(num_sample, index)| {
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

                let fetched_context: Vec<Document> = futures::future::try_join_all(search_futures)
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?
                    .into_iter()
                    .flatten()
                    .collect();

                let tooldefs = tool_server_handle
                    .get_tool_defs(Some(text.to_string()))
                    .await
                    .map_err(|_| {
                        CompletionError::RequestError("Failed to get tool definitions".into())
                    })?;

                (tooldefs, fetched_context)
            }
            None => {
                let tooldefs = tool_server_handle.get_tool_defs(None).await.map_err(|_| {
                    CompletionError::RequestError("Failed to get tool definitions".into())
                })?;

                (tooldefs, Vec::new())
            }
        };

    // Executable tools are the real tool-server tools, computed BEFORE any
    // synthetic output tool is appended.
    let executable_tool_names: BTreeSet<String> =
        tooldefs.iter().map(|tool| tool.name.clone()).collect();

    // Resolve the effective output mode (#1928). Once the run has committed to a
    // Tool-mode output tool on an earlier turn (signaled by `committed_output_
    // tool`, which is persisted on the run via `output_tool_name`), stay in Tool
    // mode and reuse that name — so a later turn whose tool set differs (e.g. RAG
    // retrieved no tools) can't flip Tool -> Native and re-apply the native
    // constraint that suppressed tools in the first place. Only Tool mode is
    // pinned; Native/Prompted re-resolve, so a tool-less first turn can still
    // become Tool once tools appear. Otherwise resolve from the request, the
    // schema, the tool set, whether the tool choice permits the output-tool call,
    // and whether the provider composes native structured output with tools.
    let resolved_mode = if committed_output_tool.is_some() && output_schema.is_some() {
        OutputMode::Tool
    } else {
        resolve_output_mode(
            output_schema.is_some(),
            !executable_tool_names.is_empty(),
            tool_choice_permits_output_tool(tool_choice),
            model.composes_native_output_with_tools(),
            output_mode,
        )
    };

    // In Tool mode, reuse the run's committed name or pick a collision-safe one.
    let output_tool_name = matches!(resolved_mode, OutputMode::Tool).then(|| {
        committed_output_tool
            .map(str::to_owned)
            .unwrap_or_else(|| pick_output_tool_name(&executable_tool_names))
    });

    // A freshly picked name never collides, but a name pinned on turn 1 can if a
    // real tool with that name is added mid-run (e.g. via dynamic/RAG tools).
    // The output-tool intercept matches by name, so surface the conflict — a
    // call to the real tool would otherwise finalize the run (see #1928, #3).
    if let Some(name) = &output_tool_name
        && executable_tool_names.contains(name)
    {
        tracing::warn!(
            output_tool = %name,
            "a real tool now shares the synthetic output-tool name; a call to it \
             will finalize the run instead of being dispatched"
        );
    }

    // Augment the preamble for Tool/Prompted modes, then prepend it as a system
    // message (deferred from the original position so it can reference the tool).
    let effective_preamble: Option<String> = {
        let base = preamble.map(str::to_owned);
        let instruction = match &resolved_mode {
            OutputMode::Tool => output_tool_name.as_deref().map(|name| {
                format!(
                    "When you have gathered enough information to answer, call the `{name}` \
                     tool exactly once with your final answer. Its arguments are the structured \
                     result and must satisfy the required schema. Do not return the final answer \
                     as plain text."
                )
            }),
            OutputMode::Prompted => output_schema.map(|schema| {
                let schema_json = serde_json::to_string(schema.as_value()).unwrap_or_default();
                format!(
                    "Respond with ONLY a single JSON object that conforms to this JSON Schema. \
                     Do not include any prose, explanation, or markdown code fences.\n{schema_json}"
                )
            }),
            OutputMode::Native | OutputMode::Auto => None,
        };
        match (base, instruction) {
            (Some(b), Some(i)) => Some(format!("{b}\n\n{i}")),
            (Some(b), None) => Some(b),
            (None, Some(i)) => Some(i),
            (None, None) => None,
        }
    };

    let chat_history: Vec<Message> = if let Some(preamble) = &effective_preamble {
        std::iter::once(Message::system(preamble.clone()))
            .chain(chat_history.iter().cloned())
            .collect()
    } else {
        chat_history.to_vec()
    };

    // In Tool mode, advertise the synthetic output tool to the provider (its name
    // is added to `allowed_tool_names` below but never to `executable_tool_names`,
    // so it is never dispatched to the tool server).
    // `output_tool_name` is only `Some` when `output_schema` is `Some` (Tool mode
    // requires a schema), so this match always fires in Tool mode.
    if let (Some(name), Some(schema)) = (&output_tool_name, output_schema) {
        tooldefs.push(crate::completion::ToolDefinition {
            name: name.clone(),
            description: "Call this tool exactly once with your final answer when you are done. \
                          Its arguments are the structured result and must satisfy the output \
                          schema."
                .to_string(),
            parameters: schema.clone().to_value(),
        });
    }

    let mut completion_request = model
        .completion_request(prompt)
        .messages(chat_history)
        .temperature_opt(temperature)
        .max_tokens_opt(max_tokens)
        .additional_params_opt(additional_params.cloned())
        .documents(static_context.to_vec())
        .tools(tooldefs);

    if !fetched_context.is_empty() {
        completion_request = completion_request.documents(fetched_context);
    }

    // Only Native mode sets the provider's native structured-output constraint.
    if matches!(resolved_mode, OutputMode::Native) {
        completion_request = completion_request.output_schema_opt(output_schema.cloned());
    }

    let completion_request = if let Some(tool_choice) = tool_choice {
        completion_request.tool_choice(tool_choice.clone())
    } else {
        completion_request
    };

    let mut allowed_tool_names =
        allowed_tool_names_for_choice(&executable_tool_names, tool_choice)?;
    // The output tool must be allowed (so it isn't flagged as an invalid tool
    // call) even though it is not executable.
    if let Some(name) = &output_tool_name {
        allowed_tool_names.insert(name.clone());
    }

    Ok(PreparedCompletionRequest {
        builder: completion_request,
        executable_tool_names,
        allowed_tool_names,
        output_tool_name,
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
    /// How `output_schema` is enforced — tool call, native structured output, or
    /// prompt injection (see [`OutputMode`] and issue #1928).
    pub output_mode: OutputMode,
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
    fn allowed_tool_names_specific_rejects_missing_tools() {
        let executable = tool_names(&["add"]);
        let choice = ToolChoice::Specific {
            function_names: vec!["missing".to_string()],
        };

        let err = allowed_tool_names_for_choice(&executable, Some(&choice))
            .expect_err("missing specific tool should fail before provider request");

        assert!(matches!(
            err,
            CompletionError::RequestError(err)
                if err.to_string().contains("missing")
                    && err.to_string().contains("add")
        ));
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

    #[test]
    fn resolve_output_mode_without_schema_is_always_native() {
        // No schema => nothing to enforce, regardless of the requested mode or tools.
        for requested in [
            OutputMode::Auto,
            OutputMode::Tool,
            OutputMode::Native,
            OutputMode::Prompted,
        ] {
            assert_eq!(
                resolve_output_mode(false, true, true, false, &requested),
                OutputMode::Native,
                "no schema should force Native for {requested:?}"
            );
            assert_eq!(
                resolve_output_mode(false, false, true, false, &requested),
                OutputMode::Native,
            );
        }
    }

    #[test]
    fn resolve_output_mode_auto_picks_tool_only_when_tools_present() {
        // This is the #1928 fix: with tools on a provider that does NOT compose
        // native output with tools, the schema must not be a native `format`
        // constraint on every turn, so Auto routes to Tool.
        assert_eq!(
            resolve_output_mode(true, true, true, false, &OutputMode::Auto),
            OutputMode::Tool,
        );
        // No tools => native structured output is safe and preferred.
        assert_eq!(
            resolve_output_mode(true, false, true, false, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_auto_keeps_native_when_provider_composes() {
        // On providers that compose native structured output with tools (OpenAI,
        // Anthropic), Auto keeps guaranteed native output even with tools present.
        assert_eq!(
            resolve_output_mode(true, true, true, true, &OutputMode::Auto),
            OutputMode::Native,
        );
    }

    #[test]
    fn resolve_output_mode_honors_explicit_choice_with_schema() {
        for (requested, expected) in [
            (OutputMode::Tool, OutputMode::Tool),
            (OutputMode::Native, OutputMode::Native),
            (OutputMode::Prompted, OutputMode::Prompted),
        ] {
            // Explicit modes are honored regardless of tools or provider support.
            assert_eq!(
                resolve_output_mode(true, true, true, false, &requested),
                expected
            );
            assert_eq!(
                resolve_output_mode(true, false, true, true, &requested),
                expected
            );
        }
    }

    #[test]
    fn resolve_output_mode_degrades_to_native_when_output_tool_not_callable() {
        // Tool mode finalizes via the output-tool call; when the tool choice
        // forbids it (None / Specific), structured output must still be enforced
        // via Native rather than silently dropped (#1928 regression guard).
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Auto),
            OutputMode::Native,
        );
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Tool),
            OutputMode::Native,
        );
        // Prompted does not rely on tools, so it is unaffected.
        assert_eq!(
            resolve_output_mode(true, true, false, false, &OutputMode::Prompted),
            OutputMode::Prompted,
        );
    }

    #[test]
    fn tool_choice_permits_output_tool_only_for_auto_required_or_unset() {
        assert!(tool_choice_permits_output_tool(None));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Auto)));
        assert!(tool_choice_permits_output_tool(Some(&ToolChoice::Required)));
        assert!(!tool_choice_permits_output_tool(Some(&ToolChoice::None)));
        assert!(!tool_choice_permits_output_tool(Some(
            &ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            }
        )));
    }

    #[test]
    fn pick_output_tool_name_defaults_when_unused() {
        let executable = tool_names(&["add", "subtract"]);
        assert_eq!(pick_output_tool_name(&executable), DEFAULT_OUTPUT_TOOL_NAME);
    }

    #[test]
    fn pick_output_tool_name_avoids_collision_with_real_tools() {
        // A user tool literally named `final_result` must not be shadowed, or
        // the model's output call would be dispatched to the tool server.
        let executable = tool_names(&["final_result"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_1");

        let executable = tool_names(&["final_result", "final_result_1"]);
        assert_eq!(pick_output_tool_name(&executable), "final_result_2");
    }
}
