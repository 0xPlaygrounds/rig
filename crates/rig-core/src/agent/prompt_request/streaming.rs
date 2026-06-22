use crate::{
    OneOrMany,
    agent::completion::{DynamicContextStore, build_prepared_completion_request},
    agent::prompt_request::{
        HookAction, assistant_text_from_choice,
        hooks::{InvalidToolCallHookAction, PromptHook},
        is_empty_assistant_turn, tool_result_user_content,
    },
    agent::run::{
        AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, OutputMode,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    completion::{Document, GetTokenUsage},
    json_utils,
    memory::ConversationMemory,
    message::{AssistantContent, ToolChoice, ToolResult, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamedUserContent, ToolCallDeltaContent},
    tool::server::ToolServerHandle,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, pin::Pin, sync::Arc};
use tracing::info_span;
use tracing_futures::Instrument;

use super::{CompletionCall, ToolCallHookAction};
use crate::{
    agent::Agent,
    completion::{CompletionError, CompletionModel, PromptError},
    message::{Message, Text},
    tool::ToolSetError,
};

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type StreamingResult<R> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[non_exhaustive]
pub enum MultiTurnStreamItem<R> {
    /// A streamed assistant content item.
    StreamAssistantItem(StreamedAssistantContent<R>),
    /// A streamed user content item (mostly for tool results).
    StreamUserItem(StreamedUserContent),
    /// Details for one successfully completed completion request made by this agent stream.
    ///
    /// This is emitted when a provider call finishes. Usage is the provider's
    /// final usage for that completion request when available; it is not
    /// incremental per streamed token.
    ///
    /// ```rust,ignore
    /// match item {
    ///     MultiTurnStreamItem::CompletionCall(completion_call) => {
    ///         // Zero-valued usage means the provider reported no metrics.
    ///         if completion_call.usage.has_values() {
    ///             let context_tokens = completion_call.usage.input_tokens;
    ///         }
    ///     }
    ///     _ => {}
    /// }
    /// ```
    CompletionCall(CompletionCall),
    /// The final result from the stream.
    FinalResponse(FinalResponse),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinalResponse {
    /// Structured assistant content for the final turn.
    content: OneOrMany<AssistantContent>,
    /// Concatenated assistant text for the final turn.
    /// This is empty only when the turn completed without emitting any text.
    response: String,
    aggregated_usage: crate::completion::Usage,
    /// Successfully completed completion requests made by this agent stream.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    completion_calls: Vec<CompletionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<Message>>,
}

impl FinalResponse {
    pub fn empty() -> Self {
        Self::new(
            OneOrMany::one(AssistantContent::text("")),
            crate::completion::Usage::new(),
            None,
        )
    }

    pub fn new(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        let response = assistant_text_from_choice(&content);
        Self {
            content,
            response,
            aggregated_usage,
            completion_calls: Vec::new(),
            history,
        }
    }

    /// Returns the concatenated assistant text for the final turn.
    pub fn response(&self) -> &str {
        &self.response
    }

    /// Returns the structured assistant content for the final turn.
    pub fn content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    /// Returns the structured assistant content for the final turn.
    pub fn assistant_content(&self) -> &OneOrMany<AssistantContent> {
        &self.content
    }

    pub fn usage(&self) -> crate::completion::Usage {
        self.aggregated_usage
    }

    /// Returns successfully completed completion requests made by this agent stream, with usage when available.
    ///
    /// Each entry represents one provider completion request. Usage is a
    /// whole-request provider snapshot, not incremental usage per streamed
    /// token. Streaming providers may omit usage for some calls; those calls
    /// have an entry with zero-valued usage.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Number of completion requests this agent run made.
    pub fn requests(&self) -> usize {
        self.completion_calls.len()
    }

    pub fn history(&self) -> Option<&[Message]> {
        self.history.as_deref()
    }
}

impl<R> MultiTurnStreamItem<R> {
    pub(crate) fn stream_item(item: StreamedAssistantContent<R>) -> Self {
        Self::StreamAssistantItem(item)
    }

    pub fn final_response(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(FinalResponse::new(content, aggregated_usage, None))
    }

    pub fn final_response_with_history(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(FinalResponse::new(content, aggregated_usage, history))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        completion_calls: Vec<CompletionCall>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let mut response = FinalResponse::new(content, aggregated_usage, history);
        response.completion_calls = completion_calls;
        Self::FinalResponse(response)
    }
}

/// Drain a provider stream abandoned by invalid tool-call recovery so the
/// reported usage for the recovered completion call is not lost.
async fn drain_stream_usage<R>(
    stream: &mut crate::streaming::StreamingCompletionResponse<R>,
) -> Result<crate::completion::Usage, StreamingError>
where
    R: Clone + Unpin + GetTokenUsage,
{
    while let Some(content) = stream.next().await {
        match content {
            Ok(StreamedAssistantContent::Final(final_resp)) => {
                return Ok(final_resp.token_usage());
            }
            Ok(_) => {}
            Err(err) => return Err(err.into()),
        }
    }

    Ok(crate::completion::Usage::new())
}

fn record_usage_on_span(span: &tracing::Span, usage: crate::completion::Usage) {
    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cached_input_tokens,
    );
    span.record(
        "gen_ai.usage.cache_creation.input_tokens",
        usage.cache_creation_input_tokens,
    );
    span.record(
        "gen_ai.usage.tool_use_prompt_tokens",
        usage.tool_use_prompt_tokens,
    );
    span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);
}

/// Build the final streamed content for a finished run (#1928).
///
/// When the finishing turn carries a tool call it is a Tool-mode output-tool
/// call (a real tool call would have routed to `CallTools`, not `Done`). In that
/// case the tool call AND the model's prose are dropped, any reasoning/image
/// content is kept, and `output` is appended as the final text — so the streamed
/// `response()` is the structured output rather than the prose, with no
/// unanswered tool_use, matching the non-streaming result. Otherwise returns
/// `None` and the caller surfaces the turn's content unchanged.
fn finalize_streamed_choice(
    last_final_choice: &OneOrMany<AssistantContent>,
    output: &str,
) -> Option<OneOrMany<AssistantContent>> {
    let finalized_via_output_tool = last_final_choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)));
    if !finalized_via_output_tool {
        return None;
    }
    let mut items: Vec<AssistantContent> = last_final_choice
        .iter()
        .filter(|item| {
            !matches!(
                item,
                AssistantContent::ToolCall(_) | AssistantContent::Text(_)
            )
        })
        .cloned()
        .collect();
    items.push(AssistantContent::text(output.to_string()));
    Some(
        OneOrMany::from_iter_optional(items)
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text(output.to_string()))),
    )
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

/// Surface [`crate::memory::ConversationMemory`] failures through the existing
/// [`CompletionError::RequestError`] variant so adding memory support does not
/// require a new top-level [`StreamingError`] arm.
impl From<crate::memory::MemoryError> for StreamingError {
    fn from(err: crate::memory::MemoryError) -> Self {
        Self::Completion(CompletionError::RequestError(Box::new(err)))
    }
}

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// If you expect to continuously call tools, you will want to ensure you use the `.multi_turn()`
/// argument to add more turns as by default, it is 0 (meaning only 1 tool round-trip). Otherwise,
/// attempting to await (which will send the prompt request) can potentially return
/// [`crate::completion::request::PromptError::MaxTurnsError`] if the agent decides to call tools
/// back to back.
pub struct StreamingPromptRequest<M, P>
where
    M: CompletionModel,
    P: PromptHook<M> + 'static,
{
    /// The prompt message to send to the model
    prompt: Message,
    /// Optional chat history provided by the caller.
    chat_history: Option<Vec<Message>>,
    /// Maximum Turns for multi-turn conversations (0 means no multi-turn)
    max_turns: usize,

    // Agent data (cloned from agent to allow hook type transitions):
    /// The completion model
    model: Arc<M>,
    /// Agent name for logging
    agent_name: Option<String>,
    /// System prompt
    preamble: Option<String>,
    /// Static context documents
    static_context: Vec<Document>,
    /// Temperature setting
    temperature: Option<f64>,
    /// Max tokens setting
    max_tokens: Option<u64>,
    /// Additional model parameters
    additional_params: Option<serde_json::Value>,
    /// Tool server handle for tool execution
    tool_server_handle: ToolServerHandle,
    /// Dynamic context store
    dynamic_context: DynamicContextStore,
    /// Tool choice setting
    tool_choice: Option<ToolChoice>,
    /// Optional JSON Schema for structured output
    output_schema: Option<schemars::Schema>,
    output_mode: OutputMode,
    /// Optional per-request hook for events
    hook: Option<P>,
    /// Maximum number of invalid tool-call retries for this request.
    max_invalid_tool_call_retries: usize,
    /// Optional conversation memory backend cloned from the agent.
    memory: Option<Arc<dyn ConversationMemory>>,
    /// Optional conversation id used for loading and saving memory.
    conversation_id: Option<String>,
}

impl<M, P> StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend + GetTokenUsage,
    P: PromptHook<M>,
{
    /// Create a new StreamingPromptRequest with the given prompt and model.
    /// Note: This creates a request without an agent hook. Use `from_agent` to include the agent's hook.
    pub fn new(agent: Arc<Agent<M>>, prompt: impl Into<Message>) -> StreamingPromptRequest<M, ()> {
        StreamingPromptRequest {
            prompt: prompt.into(),
            chat_history: None,
            max_turns: agent.default_max_turns.unwrap_or_default(),
            model: agent.model.clone(),
            agent_name: agent.name.clone(),
            preamble: agent.preamble.clone(),
            static_context: agent.static_context.clone(),
            temperature: agent.temperature,
            max_tokens: agent.max_tokens,
            additional_params: agent.additional_params.clone(),
            tool_server_handle: agent.tool_server_handle.clone(),
            dynamic_context: agent.dynamic_context.clone(),
            tool_choice: agent.tool_choice.clone(),
            output_schema: agent.output_schema.clone(),
            output_mode: agent.output_mode.clone(),
            hook: None,
            max_invalid_tool_call_retries: 0,
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
        }
    }

    /// Create a new StreamingPromptRequest from an agent, cloning the agent's data and default hook.
    pub fn from_agent<P2>(
        agent: &Agent<M, P2>,
        prompt: impl Into<Message>,
    ) -> StreamingPromptRequest<M, P2>
    where
        P2: PromptHook<M>,
    {
        StreamingPromptRequest {
            prompt: prompt.into(),
            chat_history: None,
            max_turns: agent.default_max_turns.unwrap_or_default(),
            model: agent.model.clone(),
            agent_name: agent.name.clone(),
            preamble: agent.preamble.clone(),
            static_context: agent.static_context.clone(),
            temperature: agent.temperature,
            max_tokens: agent.max_tokens,
            additional_params: agent.additional_params.clone(),
            tool_server_handle: agent.tool_server_handle.clone(),
            dynamic_context: agent.dynamic_context.clone(),
            tool_choice: agent.tool_choice.clone(),
            output_schema: agent.output_schema.clone(),
            output_mode: agent.output_mode.clone(),
            hook: agent.hook.clone(),
            max_invalid_tool_call_retries: 0,
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
        }
    }

    fn agent_name(&self) -> &str {
        self.agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Set the maximum Turns for multi-turn conversations (ie, the maximum number of turns an LLM can have calling tools before writing a text response).
    /// If the maximum turn number is exceeded, it will return a [`crate::completion::request::PromptError::MaxTurnsError`].
    pub fn multi_turn(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// Add chat history to the prompt request.
    ///
    /// When history is provided, the final [`FinalResponse`] will include the
    /// updated chat history (original messages + new user prompt + assistant response).
    /// ```ignore
    /// let mut stream = agent
    ///     .stream_prompt("Hello")
    ///     .with_history(vec![])
    ///     .await;
    /// // ... consume stream ...
    /// // Access updated history from FinalResponse::history()
    /// ```
    pub fn with_history<H, T>(mut self, history: H) -> Self
    where
        H: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.chat_history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Attach a per-request hook for tool call events.
    /// This overrides any default hook set on the agent.
    pub fn with_hook<P2>(self, hook: P2) -> StreamingPromptRequest<M, P2>
    where
        P2: PromptHook<M>,
    {
        StreamingPromptRequest {
            prompt: self.prompt,
            chat_history: self.chat_history,
            max_turns: self.max_turns,
            model: self.model,
            agent_name: self.agent_name,
            preamble: self.preamble,
            static_context: self.static_context,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            tool_server_handle: self.tool_server_handle,
            dynamic_context: self.dynamic_context,
            tool_choice: self.tool_choice,
            output_schema: self.output_schema,
            output_mode: self.output_mode,
            hook: Some(hook),
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            memory: self.memory,
            conversation_id: self.conversation_id,
        }
    }

    /// Set the retry budget for [`crate::agent::prompt_request::hooks::InvalidToolCallHookAction::Retry`].
    ///
    /// Invalid tool-call retries also consume normal multi-turn depth.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    /// Set the conversation id used to load and persist memory for this request.
    ///
    /// Overrides any default conversation id set on the agent. If memory is not
    /// configured on the agent, this has no effect.
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Disable conversation memory for this request.
    ///
    /// History will neither be loaded from nor saved to the agent's memory backend.
    pub fn without_memory(mut self) -> Self {
        self.memory = None;
        self.conversation_id = None;
        self
    }

    async fn send(self) -> StreamingResult<M::StreamingResponse> {
        let (agent_span, created_agent_span) = if tracing::Span::current().is_disabled() {
            (
                info_span!(
                    "invoke_agent",
                    gen_ai.operation.name = "invoke_agent",
                    gen_ai.agent.name = self.agent_name(),
                    gen_ai.system_instructions = self.preamble,
                    gen_ai.prompt = tracing::field::Empty,
                    gen_ai.completion = tracing::field::Empty,
                    gen_ai.usage.input_tokens = tracing::field::Empty,
                    gen_ai.usage.output_tokens = tracing::field::Empty,
                    gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                    gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                    gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                    gen_ai.usage.reasoning_tokens = tracing::field::Empty,
                ),
                true,
            )
        } else {
            (tracing::Span::current(), false)
        };

        let prompt = self.prompt;
        if let Some(text) = prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        // Clone fields needed inside the stream
        let model = self.model.clone();
        let preamble = self.preamble.clone();
        let static_context = self.static_context.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let additional_params = self.additional_params.clone();
        let tool_server_handle = self.tool_server_handle.clone();
        let dynamic_context = self.dynamic_context.clone();
        let tool_choice = self.tool_choice.clone();
        let agent_name = self.agent_name.clone();
        let output_schema = self.output_schema;
        let output_mode = self.output_mode.clone();
        // When the caller passes explicit history, memory is fully bypassed for
        // this request (no load AND no save). Otherwise, if a memory backend and
        // conversation id are both configured, load prior history; if either is
        // missing, behave as if no memory is configured.
        let (chat_history, memory_handle) = match self.chat_history {
            Some(history) => (Some(history), None),
            None => match (self.memory, self.conversation_id) {
                (Some(memory), Some(id)) => match memory.load(&id).await {
                    Ok(loaded) => (Some(loaded), Some((memory, id))),
                    Err(err) => {
                        let stream = async_stream::stream! {
                            yield Err(StreamingError::from(err));
                        };
                        return Box::pin(stream);
                    }
                },
                _ => (None, None),
            },
        };
        let has_history = chat_history.is_some();

        let mut run = AgentRun::new(prompt.clone())
            .max_turns(self.max_turns)
            .max_invalid_tool_call_retries(self.max_invalid_tool_call_retries)
            .with_output_validation(
                output_schema
                    .as_ref()
                    .map(|schema| schema.as_value().clone()),
                DEFAULT_OUTPUT_RETRIES,
            );
        if let Some(history) = chat_history {
            run = run.with_history(history);
        }
        if let Some(tool_choice) = tool_choice.clone() {
            run = run.with_tool_choice(tool_choice);
        }

        // NOTE: We use .instrument(agent_span) instead of span.enter() to avoid
        // span context leaking to other concurrent tasks. Using span.enter() inside
        // async_stream::stream! holds the guard across yield points, which causes
        // thread-local span context to leak when other tasks run on the same thread.
        // See: https://docs.rs/tracing/latest/tracing/span/struct.Span.html#in-asynchronous-code
        // See also: https://github.com/rust-lang/rust-clippy/issues/8722
        let stream = async_stream::stream! {
            // The raw provider choice of the most recent turn; the final
            // response surfaces it as-is, even when canonical reordering was
            // recorded in history.
            let mut last_final_choice: OneOrMany<AssistantContent> =
                OneOrMany::one(AssistantContent::text(""));
            let mut last_message_id: Option<String> = None;

            'outer: loop {
                let step = match run.next_step() {
                    Ok(step) => step,
                    Err(err) => {
                        yield Err(Box::new(err).into());
                        break 'outer;
                    }
                };

                match step {
                    AgentRunStep::CallModel { prompt: current_prompt, history, turn } => {
                        if self.max_turns > 1 {
                            tracing::info!(
                                "Current conversation Turns: {}/{}",
                                turn,
                                self.max_turns
                            );
                        }

                        if let Some(ref hook) = self.hook
                            && let HookAction::Terminate { reason } =
                                hook.on_completion_call(&current_prompt, &history).await
                        {
                            yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                            break 'outer;
                        }

                        let chat_stream_span = info_span!(
                            target: "rig::agent_chat",
                            parent: tracing::Span::current(),
                            "chat_streaming",
                            gen_ai.operation.name = "chat",
                            gen_ai.agent.name = agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                            gen_ai.system_instructions = preamble,
                            gen_ai.provider.name = tracing::field::Empty,
                            gen_ai.request.model = tracing::field::Empty,
                            gen_ai.response.id = tracing::field::Empty,
                            gen_ai.response.model = tracing::field::Empty,
                            gen_ai.usage.output_tokens = tracing::field::Empty,
                            gen_ai.usage.input_tokens = tracing::field::Empty,
                            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                            gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                            gen_ai.usage.reasoning_tokens = tracing::field::Empty,
                            gen_ai.input.messages = tracing::field::Empty,
                            gen_ai.output.messages = tracing::field::Empty,
                        );

                        // Pin Tool output mode once committed so later turns stay
                        // consistent even if the per-turn tool set changes (#1928).
                        // The pin rides on `output_tool_name`, which is persisted
                        // on the run, so it also survives a serialize/resume.
                        let committed_output_tool = run.output_tool_name().map(str::to_owned);
                        let prepared_request = build_prepared_completion_request(
                            &model,
                            current_prompt.clone(),
                            &history,
                            preamble.as_deref(),
                            &static_context,
                            temperature,
                            max_tokens,
                            additional_params.as_ref(),
                            tool_choice.as_ref(),
                            &tool_server_handle,
                            &dynamic_context,
                            output_schema.as_ref(),
                            &output_mode,
                            committed_output_tool.as_deref(),
                        )
                        .await?;

                        run.set_output_tool_name(prepared_request.output_tool_name.clone());

                        let mut stream = prepared_request
                            .builder
                            .stream()
                            .instrument(chat_stream_span.clone())
                            .await?;

                        let mut assembler = StreamedTurnAssembler::new(
                            prepared_request.executable_tool_names.clone(),
                            prepared_request.allowed_tool_names.clone(),
                        );
                        let mut completion_call_emitted = false;
                        let mut turn_abandoned = false;

                        'turn: while let Some(item) = stream.next().await {
                            let item = match item {
                                Ok(item) => item,
                                Err(err) => {
                                    yield Err(err.into());
                                    break 'outer;
                                }
                            };
                            let mut events: VecDeque<StreamedTurnEvent> =
                                match assembler.ingest(&item) {
                                    Ok(events) => events.into(),
                                    Err(err) => {
                                        yield Err(err.into());
                                        break 'outer;
                                    }
                                };
                            // At most one event per ingested item forwards the
                            // item itself; moving it out of the slot avoids a
                            // clone per streamed delta.
                            let mut item_slot = Some(item);
                            while let Some(event) = events.pop_front() {
                                match event {
                                    StreamedTurnEvent::EmitIngested => {
                                        if let Some(StreamedAssistantContent::Text(text)) =
                                            item_slot.as_ref()
                                            && let Some(ref hook) = self.hook
                                            && let HookAction::Terminate { reason } = hook
                                                .on_text_delta(
                                                    &text.text,
                                                    assembler.aggregated_text(),
                                                )
                                                .await
                                        {
                                            yield Err(StreamingError::Prompt(Box::new(
                                                run.cancel_error(reason),
                                            )));
                                            break 'outer;
                                        }
                                        if let Some(item) = item_slot.take() {
                                            yield Ok(MultiTurnStreamItem::stream_item(item));
                                        }
                                    }
                                    StreamedTurnEvent::EmitToolCallDelta {
                                        id,
                                        internal_call_id,
                                        content,
                                    } => {
                                        if let Some(ref hook) = self.hook {
                                            let (name, delta) = match &content {
                                                ToolCallDeltaContent::Name(name) => {
                                                    (Some(name.as_str()), "")
                                                }
                                                ToolCallDeltaContent::Delta(delta) => {
                                                    (None, delta.as_str())
                                                }
                                            };

                                            if let HookAction::Terminate { reason } = hook
                                                .on_tool_call_delta(
                                                    &id,
                                                    &internal_call_id,
                                                    name,
                                                    delta,
                                                )
                                                .await
                                            {
                                                yield Err(StreamingError::Prompt(Box::new(
                                                    run.cancel_error(reason),
                                                )));
                                                break 'outer;
                                            }
                                        }

                                        yield Ok(MultiTurnStreamItem::StreamAssistantItem(
                                            StreamedAssistantContent::ToolCallDelta {
                                                id,
                                                internal_call_id,
                                                content,
                                            },
                                        ));
                                    }
                                    StreamedTurnEvent::Completed { usage, emit_final } => {
                                        if !completion_call_emitted {
                                            if usage.has_values() {
                                                record_usage_on_span(&chat_stream_span, usage);
                                            }
                                            let completion_call =
                                                match run.record_streamed_completion_call(usage) {
                                                    Ok(call) => call,
                                                    Err(err) => {
                                                        yield Err(Box::new(err).into());
                                                        break 'outer;
                                                    }
                                                };
                                            completion_call_emitted = true;
                                            yield Ok(MultiTurnStreamItem::CompletionCall(
                                                completion_call,
                                            ));
                                        }

                                        if emit_final
                                            && let Some(StreamedAssistantContent::Final(
                                                final_resp,
                                            )) = item_slot.as_ref()
                                        {
                                            if let Some(ref hook) = self.hook
                                                && let HookAction::Terminate { reason } = hook
                                                    .on_stream_completion_response_finish(
                                                        &current_prompt,
                                                        final_resp,
                                                    )
                                                    .await
                                            {
                                                yield Err(StreamingError::Prompt(Box::new(
                                                    run.cancel_error(reason),
                                                )));
                                                break 'outer;
                                            }
                                            if let Some(item) = item_slot.take() {
                                                yield Ok(MultiTurnStreamItem::stream_item(item));
                                            }
                                        }
                                    }
                                    StreamedTurnEvent::InvalidToolCall(invalid) => {
                                        let partial =
                                            assembler.partial_turn(stream.message_id.clone());
                                        let action = match self.hook.as_ref() {
                                            Some(hook) => {
                                                let context = run
                                                    .streamed_invalid_tool_call_context(
                                                        &partial, &invalid,
                                                    );
                                                hook.on_invalid_tool_call(&context).await
                                            }
                                            None => InvalidToolCallHookAction::fail(),
                                        };

                                        let resolution = match run
                                            .resolve_streamed_invalid_tool_call(
                                                &partial, &invalid, action,
                                            ) {
                                            Ok(resolution) => resolution,
                                            Err(err) => {
                                                yield Err(Box::new(err).into());
                                                break 'outer;
                                            }
                                        };

                                        match resolution {
                                            StreamedResolution::Repaired { .. } => {
                                                // Replayed name/argument deltas flow through
                                                // the same event handling above.
                                                events.extend(
                                                    assembler.resolve_pending_invalid(&resolution),
                                                );
                                            }
                                            StreamedResolution::TurnAbandoned {
                                                ref skipped_tool_result,
                                            } => {
                                                let skipped_tool_result =
                                                    skipped_tool_result.clone();
                                                assembler.resolve_pending_invalid(&resolution);

                                                if let Some(err) = assembler.pending_delta_error() {
                                                    yield Err(err.into());
                                                    break 'outer;
                                                }
                                                let drained_usage =
                                                    match drain_stream_usage(&mut stream).await {
                                                        Ok(usage) => usage,
                                                        Err(err) => {
                                                            yield Err(err);
                                                            break 'outer;
                                                        }
                                                    };
                                                if !completion_call_emitted {
                                                    if drained_usage.has_values() {
                                                        record_usage_on_span(
                                                            &chat_stream_span,
                                                            drained_usage,
                                                        );
                                                    }
                                                    let completion_call = match run
                                                        .record_streamed_completion_call(
                                                            drained_usage,
                                                        ) {
                                                        Ok(call) => call,
                                                        Err(err) => {
                                                            yield Err(Box::new(err).into());
                                                            break 'outer;
                                                        }
                                                    };
                                                    completion_call_emitted = true;
                                                    yield Ok(MultiTurnStreamItem::CompletionCall(
                                                        completion_call,
                                                    ));
                                                }
                                                if let Some(tool_result) = skipped_tool_result {
                                                    yield Ok(MultiTurnStreamItem::StreamUserItem(
                                                        StreamedUserContent::ToolResult {
                                                            tool_result,
                                                            internal_call_id: invalid
                                                                .internal_call_id
                                                                .clone(),
                                                        },
                                                    ));
                                                }
                                                turn_abandoned = true;
                                                break 'turn;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if turn_abandoned {
                            continue 'outer;
                        }

                        if let Some(err) = assembler.pending_delta_error() {
                            yield Err(err.into());
                            break 'outer;
                        }

                        if !completion_call_emitted {
                            let completion_call =
                                match run
                                    .record_streamed_completion_call(crate::completion::Usage::new())
                                {
                                    Ok(call) => call,
                                    Err(err) => {
                                        yield Err(Box::new(err).into());
                                        break 'outer;
                                    }
                                };
                            yield Ok(MultiTurnStreamItem::CompletionCall(completion_call));
                        }

                        let final_turn_content = stream.choice.clone();
                        tracing::Span::current().record(
                            "gen_ai.completion",
                            assistant_text_from_choice(&final_turn_content),
                        );

                        last_message_id = stream.message_id.clone();
                        let streamed_turn =
                            assembler.finish(stream.message_id.clone(), &final_turn_content);
                        if let Err(err) = run.streamed_turn(streamed_turn) {
                            yield Err(Box::new(err).into());
                            break 'outer;
                        }
                        last_final_choice = final_turn_content;
                    }
                    AgentRunStep::CallTools { calls } => {
                        let full_history_for_errors = run.full_history();
                        let mut results: Vec<UserContent> = Vec::with_capacity(calls.len());

                        for pending in calls {
                            let tool_call = pending.tool_call;
                            if let Some(result) = pending.preresolved_result {
                                // Pre-resolved results only occur when invalid
                                // tool-call recovery suppressed execution; the
                                // streamed path abandons such turns instead, so
                                // this arm only serves machine-level drivers
                                // mixing streamed and non-streamed turns.
                                results.push(result);
                                continue;
                            }
                            let internal_call_id = pending
                                .internal_call_id
                                .unwrap_or_else(crate::id::generate);

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

                            yield Ok(MultiTurnStreamItem::stream_item(
                                StreamedAssistantContent::ToolCall {
                                    tool_call: tool_call.clone(),
                                    internal_call_id: internal_call_id.clone(),
                                },
                            ));

                            let tc_result = async {
                                let tool_span = tracing::Span::current();
                                let tool_args =
                                    json_utils::value_to_json_string(&tool_call.function.arguments);
                                if let Some(ref hook) = self.hook {
                                    let action = hook
                                        .on_tool_call(
                                            &tool_call.function.name,
                                            tool_call.call_id.clone(),
                                            &internal_call_id,
                                            &tool_args,
                                        )
                                        .await;

                                    if let ToolCallHookAction::Terminate { reason } = action {
                                        return Err(StreamingError::Prompt(Box::new(
                                            PromptError::prompt_cancelled(
                                                full_history_for_errors.clone(),
                                                reason,
                                            ),
                                        )));
                                    }

                                    if let ToolCallHookAction::Skip { reason } = action {
                                        // Tool execution rejected, return rejection message as tool result
                                        tracing::info!(
                                            tool_name = tool_call.function.name.as_str(),
                                            reason = reason,
                                            "Tool call rejected"
                                        );
                                        return Ok(reason);
                                    }
                                }

                                tool_span.record("gen_ai.tool.name", &tool_call.function.name);
                                tool_span.record("gen_ai.tool.call.arguments", &tool_args);

                                let tool_result = match tool_server_handle
                                    .call_tool(&tool_call.function.name, &tool_args)
                                    .await
                                {
                                    Ok(result) => result,
                                    Err(err) => {
                                        tracing::warn!("Error while calling tool: {err}");
                                        err.to_string()
                                    }
                                };

                                tool_span.record("gen_ai.tool.call.result", &tool_result);

                                if let Some(ref hook) = self.hook
                                    && let HookAction::Terminate { reason } = hook
                                        .on_tool_result(
                                            &tool_call.function.name,
                                            tool_call.call_id.clone(),
                                            &internal_call_id,
                                            &tool_args,
                                            &tool_result.to_string(),
                                        )
                                        .await
                                {
                                    return Err(StreamingError::Prompt(Box::new(
                                        PromptError::prompt_cancelled(
                                            full_history_for_errors.clone(),
                                            reason,
                                        ),
                                    )));
                                }

                                Ok(tool_result)
                            }
                            .instrument(tool_span)
                            .await;

                            match tc_result {
                                Ok(text) => {
                                    results.push(tool_result_user_content(
                                        tool_call.id.clone(),
                                        tool_call.call_id.clone(),
                                        text.clone(),
                                    ));
                                    let tool_result = ToolResult {
                                        id: tool_call.id,
                                        call_id: tool_call.call_id,
                                        content: ToolResultContent::from_tool_output(text),
                                    };
                                    yield Ok(MultiTurnStreamItem::StreamUserItem(
                                        StreamedUserContent::ToolResult {
                                            tool_result,
                                            internal_call_id,
                                        },
                                    ));
                                }
                                Err(err) => {
                                    yield Err(err);
                                    break 'outer;
                                }
                            }
                        }

                        if let Err(err) = run.tool_results(results) {
                            yield Err(Box::new(err).into());
                            break 'outer;
                        }
                    }
                    AgentRunStep::Done(response) => {
                        // Tool output mode (#1928): when the finishing turn made
                        // the output-tool call, surface the run's structured
                        // output as the final content (see `finalize_streamed_
                        // choice`). Otherwise keep the turn's content as-is.
                        let final_choice = finalize_streamed_choice(
                            &last_final_choice,
                            &response.output,
                        )
                        .unwrap_or_else(|| {
                            if is_empty_assistant_turn(&last_final_choice) {
                                tracing::warn!(
                                    agent_name =
                                        agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                                    message_id = ?last_message_id,
                                    "Streaming turn completed without assistant text; final response will be empty"
                                );
                            }
                            last_final_choice.clone()
                        });

                        if created_agent_span {
                            let current_span = tracing::Span::current();
                            record_usage_on_span(&current_span, response.usage);
                        }
                        tracing::info!("Agent multi-turn stream finished");
                        if let Some((memory, id)) = memory_handle.as_ref()
                            && let Err(err) = memory
                                .append(id, response.messages.clone().unwrap_or_default())
                                .await
                        {
                            tracing::warn!(
                                error = %err,
                                conversation_id = %id,
                                "conversation memory append failed; yielding final response anyway"
                            );
                        }
                        let final_messages: Option<Vec<Message>> = if has_history {
                            Some(response.messages.clone().unwrap_or_default())
                        } else {
                            None
                        };
                        yield Ok(MultiTurnStreamItem::final_response_with_completion_calls(
                            final_choice,
                            response.usage,
                            response.completion_calls.clone(),
                            final_messages,
                        ));
                        break 'outer;
                    }
                }
            }
        };

        Box::pin(stream.instrument(agent_span))
    }
}

impl<M, P> IntoFuture for StreamingPromptRequest<M, P>
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: WasmCompatSend,
    P: PromptHook<M> + 'static,
{
    type Output = StreamingResult<M::StreamingResponse>; // what `.await` returns
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        // Wrap send() in a future, because send() returns a stream immediately
        Box::pin(async move { self.send().await })
    }
}

/// Helper function to stream assistant-visible completion output to stdout.
///
/// This helper prints streamed assistant text and reasoning only. Streaming
/// metadata events, such as `MultiTurnStreamItem::CompletionCall`, are not
/// printed; metadata is returned on the `FinalResponse` via accessors such as
/// `FinalResponse::completion_calls`.
pub async fn stream_to_stdout<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text, .. },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::agent::prompt_request::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER;
    use crate::agent::prompt_request::hooks::{
        InvalidToolCallContext, InvalidToolCallHookAction, PromptHook, ToolCallHookAction,
    };
    use crate::agent::run::streamed::merge_reasoning_blocks;
    use crate::client::ProviderClient;
    use crate::client::completion::CompletionClient;
    use crate::completion::{CompletionRequest, PromptError, ToolDefinition, Usage};
    use crate::message::{
        AssistantContent, DocumentSourceKind, ImageMediaType, Message, ReasoningContent,
        ToolChoice, ToolResultContent, UserContent,
    };
    use crate::providers::anthropic;
    use crate::streaming::{StreamingPrompt, ToolCallDeltaContent};
    use crate::test_utils::{
        AppendFailingMemory, FailingMemory, MockAddTool, MockCompletionModel, MockResponse,
        MockStreamEvent, MockSubtractTool, MockToolError,
    };
    use crate::tool::Tool;
    use futures::{StreamExt, TryStreamExt};
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tracing::field::{Field, Visit};
    use tracing::{Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

    #[test]
    fn finalize_streamed_choice_surfaces_output_over_tool_call_and_prose() {
        use crate::message::{ToolCall, ToolFunction};

        let output_call = AssistantContent::ToolCall(ToolCall::new(
            "c1".to_string(),
            ToolFunction::new(
                "final_result".to_string(),
                serde_json::json!({"city": "Tokyo"}),
            ),
        ));

        // Prose + output-tool call (#1928): the streamed response text must be
        // the structured output, not the prose, with no orphan tool_use.
        let with_prose = OneOrMany::many(vec![
            AssistantContent::text("Sure, here is the weather:"),
            output_call.clone(),
        ])
        .expect("two items");
        let final_choice = finalize_streamed_choice(&with_prose, r#"{"city":"Tokyo"}"#)
            .expect("a turn with the output-tool call is finalized via it");
        assert_eq!(
            assistant_text_from_choice(&final_choice),
            r#"{"city":"Tokyo"}"#
        );
        assert!(
            !final_choice
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "no unanswered tool_use should remain in the final content"
        );

        // Output-tool call only.
        let only_call = OneOrMany::one(output_call);
        let final_choice = finalize_streamed_choice(&only_call, r#"{"city":"Tokyo"}"#)
            .expect("finalized via output tool");
        assert_eq!(
            assistant_text_from_choice(&final_choice),
            r#"{"city":"Tokyo"}"#
        );

        // A plain-text finalize (no tool call) is left to the caller.
        let text_only = OneOrMany::one(AssistantContent::text(r#"{"city":"Tokyo"}"#));
        assert!(finalize_streamed_choice(&text_only, r#"{"city":"Tokyo"}"#).is_none());
    }

    #[test]
    fn merge_reasoning_blocks_preserves_order_and_signatures() {
        let mut accumulated = Vec::new();
        let first = crate::message::Reasoning {
            id: Some("rs_1".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: Some("sig-1".to_string()),
            }],
        };
        let second = crate::message::Reasoning {
            id: Some("rs_1".to_string()),
            content: vec![
                ReasoningContent::Text {
                    text: "step-2".to_string(),
                    signature: Some("sig-2".to_string()),
                },
                ReasoningContent::Summary("summary".to_string()),
            ],
        };

        merge_reasoning_blocks(&mut accumulated, &first);
        merge_reasoning_blocks(&mut accumulated, &second);

        assert_eq!(accumulated.len(), 1);
        let merged = accumulated.first().expect("expected accumulated reasoning");
        assert_eq!(merged.id.as_deref(), Some("rs_1"));
        assert_eq!(merged.content.len(), 3);
        assert!(matches!(
            merged.content.first(),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-1" && sig == "sig-1"
        ));
        assert!(matches!(
            merged.content.get(1),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-2" && sig == "sig-2"
        ));
    }

    #[test]
    fn merge_reasoning_blocks_keeps_distinct_ids_as_separate_items() {
        let mut accumulated = vec![crate::message::Reasoning {
            id: Some("rs_a".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: None,
            }],
        }];
        let incoming = crate::message::Reasoning {
            id: Some("rs_b".to_string()),
            content: vec![ReasoningContent::Text {
                text: "step-2".to_string(),
                signature: None,
            }],
        };

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert_eq!(
            accumulated.first().and_then(|r| r.id.as_deref()),
            Some("rs_a")
        );
        assert_eq!(
            accumulated.get(1).and_then(|r| r.id.as_deref()),
            Some("rs_b")
        );
    }

    #[test]
    fn merge_reasoning_blocks_keeps_none_ids_separate_items() {
        let mut accumulated = vec![crate::message::Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: "first".to_string(),
                signature: None,
            }],
        }];
        let incoming = crate::message::Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: "second".to_string(),
                signature: None,
            }],
        };

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert!(matches!(
            accumulated.first(),
            Some(crate::message::Reasoning {
                id: None,
                content
            }) if matches!(
                content.first(),
                Some(ReasoningContent::Text { text, .. }) if text == "first"
            )
        ));
        assert!(matches!(
            accumulated.get(1),
            Some(crate::message::Reasoning {
                id: None,
                content
            }) if matches!(
                content.first(),
                Some(ReasoningContent::Text { text, .. }) if text == "second"
            )
        ));
    }

    #[test]
    fn tool_result_user_content_preserves_multimodal_tool_output() {
        let user_content = tool_result_user_content(
            "tool_call_1".to_string(),
            Some("call_1".to_string()),
            serde_json::json!({
                "response": {
                    "instruction": "Use the image part to answer."
                },
                "parts": [
                    {
                        "type": "image",
                        "data": "base64data==",
                        "mimeType": "image/png"
                    }
                ]
            })
            .to_string(),
        );

        let tool_result = match user_content {
            UserContent::ToolResult(tool_result) => tool_result,
            other => panic!("expected tool result content, got {other:?}"),
        };

        assert_eq!(tool_result.id, "tool_call_1");
        assert_eq!(tool_result.call_id.as_deref(), Some("call_1"));
        assert_eq!(tool_result.content.len(), 2);

        let mut items = tool_result.content.iter();
        match items.next() {
            Some(ToolResultContent::Text(text)) => {
                assert!(text.text.contains("Use the image part to answer."));
            }
            other => panic!("expected structured text payload first, got {other:?}"),
        }

        match items.next() {
            Some(ToolResultContent::Image(image)) => {
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert!(matches!(
                    image.data,
                    DocumentSourceKind::Base64(ref data) if data == "base64data=="
                ));
            }
            other => panic!("expected image payload second, got {other:?}"),
        }
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) -> Result<(), String> {
        let history = request.chat_history.iter().cloned().collect::<Vec<_>>();
        if history.len() != 3 {
            return Err(format!(
                "follow-up request should contain [original user prompt, assistant tool call, user tool result]: {history:?}"
            ));
        }

        if !matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ) {
            return Err(format!(
                "follow-up request should begin with the original user prompt: {history:?}"
            ));
        }

        if !matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request is missing the assistant tool call in position 2: {history:?}"
            ));
        }

        if !matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request should end with the user tool result: {history:?}"
            ));
        }

        Ok(())
    }

    fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.function.name == tool_name
                    ))
            )
        })
    }

    fn history_contains_text(history: &[Message], expected: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::Text(text) if text.text == expected
                    ))
            )
        })
    }

    fn assistant_reasoning_precedes_tool_call(
        history: &[Message],
        expected_reasoning: &str,
        tool_name: &str,
    ) -> bool {
        history.iter().any(|message| {
            let Message::Assistant { content, .. } = message else {
                return false;
            };

            let reasoning_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Reasoning(reasoning)
                        if reasoning.content.iter().any(|content| matches!(
                            content,
                            ReasoningContent::Text { text, .. }
                                if text == expected_reasoning
                        ))
                )
            });
            let tool_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                )
            });

            matches!((reasoning_index, tool_index), (Some(reasoning), Some(tool)) if reasoning < tool)
        })
    }

    fn assistant_reasoning_precedes_text_and_tool_call(
        history: &[Message],
        expected_reasoning: &str,
        expected_text: &str,
        tool_name: &str,
    ) -> bool {
        history.iter().any(|message| {
            let Message::Assistant { content, .. } = message else {
                return false;
            };

            let reasoning_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Reasoning(reasoning)
                        if reasoning.content.iter().any(|content| matches!(
                            content,
                            ReasoningContent::Text { text, .. }
                                if text == expected_reasoning
                        ))
                )
            });
            let text_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Text(text) if text.text == expected_text
                )
            });
            let tool_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                )
            });

            matches!(
                (reasoning_index, text_index, tool_index),
                (Some(reasoning), Some(text), Some(tool))
                    if reasoning < text && text < tool
            )
        })
    }

    #[derive(Clone)]
    struct PanicOnUnknownToolHook;

    impl PromptHook<MockCompletionModel> for PanicOnUnknownToolHook {
        async fn on_tool_call_delta(
            &self,
            _tool_call_id: &str,
            _internal_call_id: &str,
            _tool_name: Option<&str>,
            _tool_call_delta: &str,
        ) -> HookAction {
            panic!("unknown tool call delta should fail before delta hooks run")
        }

        async fn on_tool_call(
            &self,
            _tool_name: &str,
            _tool_call_id: Option<String>,
            _internal_call_id: &str,
            _args: &str,
        ) -> ToolCallHookAction {
            panic!("unknown tool call should fail before tool hooks run")
        }

        async fn on_stream_completion_response_finish(
            &self,
            _prompt: &Message,
            _response: &MockResponse,
        ) -> HookAction {
            panic!("unknown tool call should fail before stream finish hooks run")
        }
    }

    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    #[derive(Clone)]
    struct CountingSubtractTool {
        calls: Arc<AtomicU32>,
    }

    #[derive(Deserialize)]
    struct CountingOperationArgs {
        x: i32,
        y: i32,
    }

    fn arithmetic_tool_definition(name: &str, description: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first operand"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second operand"
                    }
                },
                "required": ["x", "y"],
            }),
        }
    }

    impl Tool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = CountingOperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            arithmetic_tool_definition(Self::NAME, "Add x and y together")
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(args.x + args.y)
        }
    }

    impl Tool for CountingSubtractTool {
        const NAME: &'static str = "subtract";
        type Error = MockToolError;
        type Args = CountingOperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            arithmetic_tool_definition(Self::NAME, "Subtract y from x")
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(args.x - args.y)
        }
    }

    fn streaming_tool_then_text_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
    }

    fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    #[derive(Clone, Debug, Default)]
    struct CapturedSpan {
        id: u64,
        name: String,
        parent_id: Option<u64>,
        fields: HashMap<String, u64>,
    }

    #[derive(Clone, Default)]
    struct CapturedSpans(Arc<Mutex<Vec<CapturedSpan>>>);

    impl CapturedSpans {
        fn clear(&self) {
            if let Ok(mut spans) = self.0.lock() {
                spans.clear();
            }
        }

        fn insert(&self, id: &Id, name: &str, parent_id: Option<u64>) {
            let id = id.into_u64();
            if let Ok(mut spans) = self.0.lock() {
                spans.push(CapturedSpan {
                    id,
                    name: name.to_string(),
                    parent_id,
                    fields: HashMap::new(),
                });
            }
        }

        fn record(&self, id: &Id, fields: Vec<(String, u64)>) {
            if let Ok(mut spans) = self.0.lock()
                && let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64())
            {
                span.fields.extend(fields);
            }
        }

        fn snapshot(&self) -> Vec<CapturedSpan> {
            self.0.lock().map(|spans| spans.clone()).unwrap_or_default()
        }
    }

    struct SpanCaptureLayer {
        spans: CapturedSpans,
    }

    impl<S> Layer<S> for SpanCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let parent_id = attrs
                .parent()
                .map(Id::into_u64)
                .or_else(|| ctx.current_span().id().map(Id::into_u64));
            self.spans.insert(id, attrs.metadata().name(), parent_id);
        }

        fn on_record(&self, span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            let mut fields = Vec::new();
            values.record(&mut SpanFieldCaptureVisitor {
                fields: &mut fields,
            });
            self.spans.record(span, fields);
        }
    }

    struct SpanFieldCaptureVisitor<'a> {
        fields: &'a mut Vec<(String, u64)>,
    }

    impl Visit for SpanFieldCaptureVisitor<'_> {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields.push((field.name().to_string(), value));
        }

        fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
    }

    async fn assert_stream_usage_recorded_on_chat_spans(
        agent: crate::agent::Agent<MockCompletionModel>,
        prompt: &str,
        max_turns: usize,
        expected_usages: &[Usage],
    ) {
        // Scoped-subscriber tests must not run concurrently; the warm-up
        // below explains the callsite-interest hazard this guards against.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        // Span callsites in the driver are shared with every other test in
        // this binary. The FIRST thread to hit a callsite caches its interest
        // from that thread's dispatcher (`Dispatchers::Rebuilder::JustOne`
        // consults `dispatcher::get_default`), so a parallel test without a
        // subscriber can permanently cache `Interest::never` for the very
        // spans this harness asserts on. Defend in two steps, both under the
        // isolation guard: (1) warm the whole driver path from THIS thread so
        // unregistered callsites first-register against this subscriber, then
        // (2) rebuild the interest cache to heal callsites a foreign thread
        // already poisoned.
        let warmup_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("warmup"),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let warmup_agent = crate::agent::AgentBuilder::new(warmup_model).build();
        let mut warmup_stream = warmup_agent.stream_prompt("warmup").multi_turn(1).await;
        while let Some(item) = warmup_stream
            .try_next()
            .await
            .expect("warmup stream should not error")
        {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let empty_history: &[Message] = &[];
        let outer_span = tracing::info_span!("outer");

        async {
            let mut stream = agent
                .stream_prompt(prompt)
                .with_history(empty_history)
                .multi_turn(max_turns)
                .await;

            while let Some(item) = stream.try_next().await.expect("stream should not error") {
                if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                    break;
                }
            }
        }
        .instrument(outer_span)
        .await;

        let span_snapshot = spans.snapshot();
        let outer_span_id = span_snapshot
            .iter()
            .find(|span| span.name == "outer")
            .map(|span| span.id)
            .expect("outer span should be captured");
        let chat_spans = span_snapshot
            .iter()
            .filter(|span| span.name == "chat_streaming")
            .collect::<Vec<_>>();

        assert_eq!(chat_spans.len(), expected_usages.len());
        assert!(
            span_snapshot.iter().all(|span| span.name != "invoke_agent"),
            "outer span path should not create invoke_agent"
        );

        for (chat_span, expected_usage) in chat_spans.into_iter().zip(expected_usages) {
            assert_eq!(chat_span.parent_id, Some(outer_span_id));
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.input_tokens"),
                Some(&expected_usage.input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.output_tokens"),
                Some(&expected_usage.output_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.cache_read.input_tokens"),
                Some(&expected_usage.cached_input_tokens)
            );
            assert_eq!(
                chat_span
                    .fields
                    .get("gen_ai.usage.cache_creation.input_tokens"),
                Some(&expected_usage.cache_creation_input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.tool_use_prompt_tokens"),
                Some(&expected_usage.tool_use_prompt_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.reasoning_tokens"),
                Some(&expected_usage.reasoning_tokens)
            );
        }

        let outer_span = span_snapshot
            .iter()
            .find(|span| span.id == outer_span_id)
            .expect("outer span should be present");
        assert!(
            outer_span
                .fields
                .keys()
                .all(|field| !field.starts_with("gen_ai.usage.")),
            "usage should not be recorded onto the caller's outer span"
        );
    }

    #[test]
    fn completion_calls_stream_item_serializes_and_deserializes_expected_shape() {
        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(2, usage(3, 4)));

        let value = serde_json::to_value(&item).expect("serialize completion call event");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 2,
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "tool_use_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                }
            })
        );

        let item: MultiTurnStreamItem<MockResponse> =
            serde_json::from_value(value).expect("deserialize completion call event");
        match item {
            MultiTurnStreamItem::CompletionCall(call_usage) => {
                assert_eq!(call_usage, CompletionCall::new(2, usage(3, 4)));
            }
            other => panic!("expected completion call event, got {other:?}"),
        }

        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(3, Usage::new()));
        let value = serde_json::to_value(&item).expect("serialize missing usage event");

        // Unreported usage serializes as a plain zero-valued object (Usage's
        // documented sentinel for missing provider metrics).
        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 3,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "tool_use_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                }
            })
        );

        // Stream items serialized before the Option encoding was dropped used
        // `"usage": null`; they must still deserialize.
        let legacy: MultiTurnStreamItem<MockResponse> = serde_json::from_value(serde_json::json!({
            "type": "completionCall",
            "call_index": 3,
            "usage": null
        }))
        .expect("legacy null-usage event should deserialize");
        match legacy {
            MultiTurnStreamItem::CompletionCall(call) => {
                assert_eq!(call, CompletionCall::new(3, Usage::new()));
            }
            other => panic!("expected completion call event, got {other:?}"),
        }
    }

    #[test]
    fn final_response_serializes_completion_calls_with_missing_usage() {
        let item: MultiTurnStreamItem<MockResponse> =
            MultiTurnStreamItem::final_response_with_completion_calls(
                OneOrMany::one(AssistantContent::text("done")),
                usage(3, 4),
                vec![
                    CompletionCall::new(0, Usage::new()),
                    CompletionCall::new(1, usage(3, 4)),
                ],
                None,
            );

        if let MultiTurnStreamItem::FinalResponse(response) = &item {
            assert_eq!(response.requests(), 2);
        }

        let value = serde_json::to_value(&item).expect("serialize final response");

        assert_eq!(
            value.get("completionCalls"),
            Some(&serde_json::json!([
                {
                    "call_index": 0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                },
                {
                    "call_index": 1,
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                }
            ]))
        );
    }

    fn streaming_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("hello"),
            MockStreamEvent::text(" world"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn citation_metadata() -> serde_json::Value {
        serde_json::json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": "Claude Shannon",
                "encrypted_index": "encrypted-reference"
            }]
        })
    }

    fn streaming_cited_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text_start(Some(citation_metadata())),
            MockStreamEvent::text("cited "),
            MockStreamEvent::text_start(None),
            MockStreamEvent::text("answer"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn streaming_cited_text_then_tool_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text_start(Some(citation_metadata())),
                MockStreamEvent::text("I need a tool. "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
    }

    fn streaming_final_only_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::final_response_with_total_tokens(1),
        ]])
    }

    #[derive(Clone)]
    struct TerminateOnStreamFinish;

    impl PromptHook<MockCompletionModel> for TerminateOnStreamFinish {
        async fn on_stream_completion_response_finish(
            &self,
            _prompt: &Message,
            _response: &<MockCompletionModel as CompletionModel>::StreamingResponse,
        ) -> HookAction {
            HookAction::terminate("stop after completion call")
        }
    }

    type RecordedToolCallDelta = (String, String, Option<String>, String);

    #[derive(Clone)]
    struct RepairDefaultApiHook;

    impl PromptHook<MockCompletionModel> for RepairDefaultApiHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            let tool_name = context.tool_name.clone();
            async move {
                assert_eq!(tool_name, "default_api");
                InvalidToolCallHookAction::repair("add")
            }
        }
    }

    #[derive(Clone)]
    struct RetryDefaultApiHook;

    impl PromptHook<MockCompletionModel> for RetryDefaultApiHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            let tool_name = context.tool_name.clone();
            let args = context.args.clone();
            async move {
                assert_eq!(tool_name, "default_api");
                if let Some(args) = args {
                    assert!(!args.is_empty());
                }
                InvalidToolCallHookAction::retry("Use the add tool instead")
            }
        }
    }

    #[derive(Clone)]
    struct SkipDefaultApiHook;

    impl PromptHook<MockCompletionModel> for SkipDefaultApiHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            let tool_name = context.tool_name.clone();
            async move {
                assert_eq!(tool_name, "default_api");
                InvalidToolCallHookAction::skip("default_api was skipped")
            }
        }
    }

    #[derive(Clone, Default)]
    struct RecordingInvalidToolCallHook {
        contexts: Arc<Mutex<Vec<InvalidToolCallContext>>>,
    }

    impl RecordingInvalidToolCallHook {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for RecordingInvalidToolCallHook {
        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            let contexts = self.contexts.clone();
            let context = context.clone();

            async move {
                contexts
                    .lock()
                    .expect("invalid tool context records mutex was poisoned")
                    .push(context);
                InvalidToolCallHookAction::fail()
            }
        }
    }

    #[derive(Clone, Default)]
    struct RecordingToolCallDeltaHook {
        deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
    }

    impl RecordingToolCallDeltaHook {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.deltas
                .lock()
                .expect("tool call delta hook records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for RecordingToolCallDeltaHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            let deltas = self.deltas.clone();
            let event = (
                tool_call_id.to_string(),
                internal_call_id.to_string(),
                tool_name.map(str::to_string),
                tool_call_delta.to_string(),
            );

            async move {
                deltas
                    .lock()
                    .expect("tool call delta hook records mutex was poisoned")
                    .push(event);
                HookAction::cont()
            }
        }
    }

    #[derive(Clone, Default)]
    struct RecordingTextDeltaHook {
        deltas: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl RecordingTextDeltaHook {
        fn observed(&self) -> Vec<(String, String)> {
            self.deltas
                .lock()
                .expect("text delta hook records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for RecordingTextDeltaHook {
        fn on_text_delta(
            &self,
            text_delta: &str,
            full_text: &str,
        ) -> impl Future<Output = HookAction> + Send {
            let deltas = self.deltas.clone();
            let event = (text_delta.to_string(), full_text.to_string());

            async move {
                deltas
                    .lock()
                    .expect("text delta hook records mutex was poisoned")
                    .push(event);
                HookAction::cont()
            }
        }
    }

    #[derive(Clone)]
    struct RecordingTextAndSkipInvalidToolHook {
        text: RecordingTextDeltaHook,
    }

    impl PromptHook<MockCompletionModel> for RecordingTextAndSkipInvalidToolHook {
        fn on_text_delta(
            &self,
            text_delta: &str,
            full_text: &str,
        ) -> impl Future<Output = HookAction> + Send {
            self.text.on_text_delta(text_delta, full_text)
        }

        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            SkipDefaultApiHook.on_invalid_tool_call(context)
        }
    }

    #[derive(Clone)]
    struct RecordingTextAndRetryInvalidToolHook {
        text: RecordingTextDeltaHook,
    }

    impl PromptHook<MockCompletionModel> for RecordingTextAndRetryInvalidToolHook {
        fn on_text_delta(
            &self,
            text_delta: &str,
            full_text: &str,
        ) -> impl Future<Output = HookAction> + Send {
            self.text.on_text_delta(text_delta, full_text)
        }

        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            RetryDefaultApiHook.on_invalid_tool_call(context)
        }
    }

    #[derive(Clone)]
    struct RecordingDeltaAndRetryInvalidToolHook {
        delta: RecordingToolCallDeltaHook,
    }

    impl PromptHook<MockCompletionModel> for RecordingDeltaAndRetryInvalidToolHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            self.delta.on_tool_call_delta(
                tool_call_id,
                internal_call_id,
                tool_name,
                tool_call_delta,
            )
        }

        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            RetryDefaultApiHook.on_invalid_tool_call(context)
        }
    }

    #[derive(Clone)]
    struct RecordingDeltaAndSkipInvalidToolHook {
        delta: RecordingToolCallDeltaHook,
    }

    impl PromptHook<MockCompletionModel> for RecordingDeltaAndSkipInvalidToolHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            self.delta.on_tool_call_delta(
                tool_call_id,
                internal_call_id,
                tool_name,
                tool_call_delta,
            )
        }

        fn on_invalid_tool_call(
            &self,
            context: &InvalidToolCallContext,
        ) -> impl Future<Output = InvalidToolCallHookAction> + Send {
            SkipDefaultApiHook.on_invalid_tool_call(context)
        }
    }

    #[derive(Clone, Default)]
    struct TerminatingToolCallDeltaHook {
        deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
    }

    impl TerminatingToolCallDeltaHook {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.deltas
                .lock()
                .expect("tool call delta hook records mutex was poisoned")
                .clone()
        }
    }

    impl PromptHook<MockCompletionModel> for TerminatingToolCallDeltaHook {
        fn on_tool_call_delta(
            &self,
            tool_call_id: &str,
            internal_call_id: &str,
            tool_name: Option<&str>,
            tool_call_delta: &str,
        ) -> impl Future<Output = HookAction> + Send {
            let deltas = self.deltas.clone();
            let event = (
                tool_call_id.to_string(),
                internal_call_id.to_string(),
                tool_name.map(str::to_string),
                tool_call_delta.to_string(),
            );

            async move {
                deltas
                    .lock()
                    .expect("tool call delta hook records mutex was poisoned")
                    .push(event);
                HookAction::terminate("stop on tool call delta")
            }
        }
    }

    fn text_metadata(content: &OneOrMany<AssistantContent>) -> Option<&serde_json::Value> {
        content.iter().find_map(|item| match item {
            AssistantContent::Text(text) => text.additional_params.as_ref(),
            _ => None,
        })
    }

    #[tokio::test]
    async fn stream_prompt_continues_after_tool_call_turn() {
        let model = streaming_tool_then_text_model();
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut final_text = String::new();
        let mut final_response_text = None;
        let mut final_history = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    final_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    saw_final_response = true;
                    final_response_text = Some(res.response().to_owned());
                    final_history = res.history().map(|history| history.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(saw_tool_call);
        assert!(saw_tool_result);
        assert!(saw_final_response);
        assert_eq!(final_text, "done");
        assert_eq!(final_response_text.as_deref(), Some("done"));
        let history = final_history.expect("expected final response history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "done"
                ))
        )));
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert!(validate_follow_up_tool_history(&requests[1]).is_ok());
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_streaming_second_request() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("unknown model-emitted tool should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_streaming_tool_name() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RepairDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;
        let mut saw_repaired_tool_call = false;
        let mut saw_tool_result = false;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { tool_call, .. },
                )) => {
                    assert_eq!(tool_call.function.name, "add");
                    saw_repaired_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    assert!(tool_result.content.iter().any(|content| {
                        matches!(
                            content,
                            ToolResultContent::Text(text) if text.text == "5"
                        )
                    }));
                    saw_tool_result = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(saw_repaired_tool_call);
        assert!(saw_tool_result);
        assert_eq!(final_response_text.as_deref(), Some("done"));
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_streaming_tool_call_provider_id() {
        let invalid_hook = RecordingInvalidToolCallHook::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("provider_call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(invalid_hook.clone())
            .multi_turn(3)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        assert!(error.is_some(), "invalid tool should fail");
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert!(context.internal_call_id.is_some());
        assert!(context.is_streaming);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skip_emits_streaming_tool_result() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(SkipDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    internal_call_id,
                })) => {
                    assert!(!internal_call_id.is_empty());
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_1");
        assert_eq!(skipped_tool_result.call_id.as_deref(), Some("call_1"));
        assert!(skipped_tool_result.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "default_api was skipped"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retries_mixed_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 4, "y": 5}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RetryDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;
        let mut completion_call_events = Vec::new();
        let mut final_response_text = None;
        let mut final_response_usage = Usage::new();
        let mut final_completion_calls = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_call_events.push(completion_call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    final_response_usage = response.usage();
                    final_completion_calls = response.completion_calls().to_vec();
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(final_response_text.as_deref(), Some("retried"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let mut first_usage = Usage::new();
        first_usage.total_tokens = 4;
        let mut second_usage = Usage::new();
        second_usage.total_tokens = 6;
        let expected_completion_calls = vec![
            CompletionCall::new(0, first_usage),
            CompletionCall::new(1, second_usage),
        ];
        assert_eq!(completion_call_events, expected_completion_calls);
        assert_eq!(final_completion_calls, expected_completion_calls);
        assert_eq!(final_response_usage.total_tokens, 10);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(retry_history.len(), 3);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_1"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == "Use the add tool instead"
                                ))
                    ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skips_mixed_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 4, "y": 5}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(SkipDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_2");
        assert_eq!(skipped_tool_result.call_id.as_deref(), Some("call_2"));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(follow_up_history.len(), 3);
        assert!(matches!(
            follow_up_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_1"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.call_id.as_deref() == Some("call_1")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.call_id.as_deref() == Some("call_2")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == "default_api was skipped"
                                ))
            ))
        ));
    }

    #[tokio::test]
    async fn invalid_completed_tool_call_skip_preserves_streaming_reasoning_history() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning("reasoned step").with_reasoning_id("rs_1"),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(SkipDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(history_contains_text(&follow_up_history, "checking "));
        assert!(assistant_reasoning_precedes_tool_call(
            &follow_up_history,
            "reasoned step",
            "default_api"
        ));
        assert!(
            assistant_reasoning_precedes_text_and_tool_call(
                &follow_up_history,
                "reasoned step",
                "checking ",
                "default_api"
            ),
            "{follow_up_history:?}"
        );
    }

    #[tokio::test]
    async fn invalid_name_delta_retry_preserves_streaming_reasoning_history() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::reasoning_delta(Some("rs_1"), "delta reason"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RetryDefaultApiHook)
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(assistant_reasoning_precedes_tool_call(
            &retry_history,
            "delta reason",
            "default_api"
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skip_resets_streaming_text_delta_state() {
        let text_hook = RecordingTextDeltaHook::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("stale "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("fresh"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RecordingTextAndSkipInvalidToolHook {
                text: text_hook.clone(),
            })
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            text_hook.observed(),
            vec![
                ("stale ".to_string(), "stale ".to_string()),
                ("fresh".to_string(), "fresh".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_retry_uses_structured_tool_feedback() {
        let delta_hook = RecordingToolCallDeltaHook::default();
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning_delta(Some("rs_1"), "diagnostic reason"),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RecordingDeltaAndRetryInvalidToolHook {
                delta: delta_hook.clone(),
            })
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;
        let mut completion_call_events = Vec::new();
        let mut final_response_text = None;
        let mut final_response_usage = Usage::new();
        let mut final_completion_calls = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_call_events.push(completion_call);
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => panic!("invalid tool-call delta should not be emitted"),
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    final_response_usage = response.usage();
                    final_completion_calls = response.completion_calls().to_vec();
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(final_response_text.as_deref(), Some("retried"));
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let mut first_usage = Usage::new();
        first_usage.total_tokens = 4;
        let mut second_usage = Usage::new();
        second_usage.total_tokens = 6;
        let expected_completion_calls = vec![
            CompletionCall::new(0, first_usage),
            CompletionCall::new(1, second_usage),
        ];
        assert_eq!(completion_call_events, expected_completion_calls);
        assert_eq!(final_completion_calls, expected_completion_calls);
        assert_eq!(final_response_usage.total_tokens, 10);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_0"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "default_api"
                            && tool_call.function.arguments == serde_json::json!({"x": 2, "y": 3})
                ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_0"
                                && result.call_id.as_deref() == Some("call_0")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "Use the add tool instead"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_context_includes_same_turn_history_and_tool_call_id() {
        let invalid_hook = RecordingInvalidToolCallHook::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning_delta(Some("rs_1"), "diagnostic reason"),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(invalid_hook.clone())
            .multi_turn(3)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        assert!(error.is_some(), "invalid name delta should fail");
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert_eq!(context.internal_call_id.as_deref(), Some("internal_1"));
        assert!(context.is_streaming);
        assert!(history_contains_text(&context.chat_history, "checking "));
        assert!(
            assistant_reasoning_precedes_tool_call(
                &context.chat_history,
                "diagnostic reason",
                "add"
            ),
            "{:?}",
            context.chat_history
        );
        assert!(history_contains_tool_call(&context.chat_history, "add"));
        assert!(history_contains_tool_call(
            &context.chat_history,
            "default_api"
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_retry_resets_streaming_text_delta_state() {
        let text_hook = RecordingTextDeltaHook::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("stale "),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("fresh"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RecordingTextAndRetryInvalidToolHook {
                text: text_hook.clone(),
            })
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            text_hook.observed(),
            vec![
                ("stale ".to_string(), "stale ".to_string()),
                ("fresh".to_string(), "fresh".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_skip_uses_structured_tool_feedback() {
        let delta_hook = RecordingToolCallDeltaHook::default();
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RecordingDeltaAndSkipInvalidToolHook {
                delta: delta_hook.clone(),
            })
            .multi_turn(3)
            .with_history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => panic!("invalid tool-call delta should not be emitted"),
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    internal_call_id,
                })) => {
                    assert_eq!(internal_call_id, "internal_1");
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_1");
        assert!(skipped_tool_result.call_id.is_none());
        assert!(skipped_tool_result.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            follow_up_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_0"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "default_api"
                            && tool_call.function.arguments == serde_json::json!({"x": 2, "y": 3})
                ))
        ));
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_0"
                                && result.call_id.as_deref() == Some("call_0")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "default_api was skipped"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn streaming_retry_budget_exhaustion_history_contains_invalid_tool_call() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RetryDefaultApiHook)
            .multi_turn(3)
            .max_invalid_tool_call_retries(0)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        let error = error.expect("retry budget exhaustion should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    chat_history,
                    ..
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn streaming_name_delta_retry_budget_exhaustion_history_includes_same_turn_context() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(RetryDefaultApiHook)
            .multi_turn(3)
            .max_invalid_tool_call_retries(0)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        let error = error.expect("retry budget exhaustion should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    chat_history,
                    ..
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert!(history_contains_text(&chat_history, "checking "));
                    assert!(history_contains_tool_call(&chat_history, "add"));
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn completed_unknown_tool_call_after_text_fails_before_finish_hook_or_later_emit() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("thinking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_text = false;
        let mut saw_completion_call = false;
        let mut saw_final_response = false;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(_))) => {
                    saw_text = true;
                }
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(
                    _,
                )))
                | Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(saw_text);
        assert!(!saw_completion_call);
        assert!(!saw_final_response);
        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("completed unknown tool call should fail immediately");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn mixed_streaming_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 3, "y": 4}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use tools")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_completion_call = false;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_completion_call);
        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("mixed unknown streamed tool call should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn multiple_valid_streaming_tool_calls_execute_after_batch_validation() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let subtract_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "subtract",
                    serde_json::json!({"x": 8, "y": 3}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(CountingSubtractTool {
                calls: subtract_calls.clone(),
            })
            .build();

        let mut stream = agent.stream_prompt("use tools").multi_turn(3).await;
        let mut tool_call_names = Vec::new();
        let mut tool_result_ids = Vec::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { tool_call, .. },
                )) => {
                    tool_call_names.push(tool_call.function.name);
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    tool_result_ids.push(tool_result.id);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.response().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            tool_call_names,
            vec!["add".to_string(), "subtract".to_string()]
        );
        assert_eq!(
            tool_result_ids,
            vec!["tool_call_1".to_string(), "tool_call_2".to_string()]
        );
        assert_eq!(add_calls.load(Ordering::SeqCst), 1);
        assert_eq!(subtract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(final_response_text.as_deref(), Some("done"));
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_streaming_second_request() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "subtract",
                    serde_json::json!({"x": 3, "y": 1}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the allowed tool")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("disallowed model-emitted tool should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "subtract");
                    assert_eq!(
                        available_tools,
                        vec!["add".to_string(), "subtract".to_string()]
                    );
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "subtract"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn mixed_specific_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "subtract",
                    serde_json::json!({"x": 3, "y": 1}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the allowed tool")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("mixed disallowed streamed tool call should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "subtract");
                    assert_eq!(
                        available_tools,
                        vec!["add".to_string(), "subtract".to_string()]
                    );
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "subtract"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("ToolChoice::None should reject returned tool calls");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call_name_delta_before_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("ToolChoice::None should reject returned tool-call deltas");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn unknown_tool_call_name_delta_fails_before_streaming_delta_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "default_api"),
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a bad tool call")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("unknown tool-call name delta should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_call_args_delta_before_unknown_name_fails_before_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a bad tool call")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("unknown tool-call name should reject buffered args");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_call_args_delta_before_valid_name_buffers_then_emits_in_safe_order() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordingToolCallDeltaHook::default();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .with_hook(hook.clone())
            .await;
        let mut stream_deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    stream_deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            hook.observed(),
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    Some("add".to_string()),
                    String::new()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "{\"x\":".to_string()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "1}".to_string()
                ),
            ]
        );
        assert_eq!(
            stream_deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn tool_call_args_delta_without_name_errors_at_stream_end() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream an incomplete tool call")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_delta = false;
        let mut saw_completion_call = false;
        let mut saw_final_response = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        assert!(!saw_completion_call);
        assert!(!saw_final_response);
        let error = error.expect("unterminated tool-call args delta should fail");
        match error {
            StreamingError::Completion(CompletionError::ResponseError(message)) => {
                assert!(
                    message.contains("streamed tool call arguments"),
                    "{message}"
                );
                assert!(message.contains("tool_1"), "{message}");
                assert!(message.contains("internal_1"), "{message}");
            }
            other => panic!("expected completion response error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_buffers_args_then_rejects_name_without_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .with_hook(PanicOnUnknownToolHook)
            .multi_turn(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("ToolChoice::None should reject buffered tool-call deltas");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_without_hook() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent.stream_prompt("stream a tool call").await;
        let mut deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_after_hook_continue() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordingToolCallDeltaHook::default();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .with_hook(hook.clone())
            .await;
        let mut stream_deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    stream_deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            hook.observed(),
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    Some("add".to_string()),
                    String::new()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "{\"x\":".to_string()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "1}".to_string()
                ),
            ]
        );
        assert_eq!(
            stream_deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_tool_call_deltas_hook_termination_prevents_delta_emit() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = TerminatingToolCallDeltaHook::default();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .with_hook(hook.clone())
            .await;
        let mut saw_delta = false;
        let mut saw_final_response = false;
        let mut error_message = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error_message = Some(err.to_string());
                    break;
                }
            }
        }

        assert_eq!(
            hook.observed(),
            vec![(
                "tool_1".to_string(),
                "internal_1".to_string(),
                Some("add".to_string()),
                String::new()
            )]
        );
        assert!(!saw_delta);
        assert!(!saw_final_response);
        assert!(
            error_message
                .as_deref()
                .is_some_and(|message| message.contains("PromptCancelled: stop on tool call delta")),
            "expected hook termination error, got {error_message:?}"
        );
    }

    #[tokio::test]
    async fn stream_prompt_exposes_completion_calls() {
        let first_call_usage = usage(10, 2);
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response(first_call_usage),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            completion_calls_events,
            vec![
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );

        let final_response = final_response.expect("expected final response");
        assert_eq!(
            final_response.usage(),
            Usage {
                input_tokens: 35,
                output_tokens: 7,
                total_tokens: 42,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            final_response.completion_calls(),
            &[
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_single_call_usage_on_chat_span_under_outer_span() {
        let call_usage = usage(10, 2);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(call_usage),
        ]]);
        let agent = AgentBuilder::new(model).build();

        assert_stream_usage_recorded_on_chat_spans(agent, "say done", 1, &[call_usage]).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_multi_turn_usage_on_chat_spans_under_outer_span() {
        let first_call_usage = usage(10, 2);
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response(first_call_usage),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();

        assert_stream_usage_recorded_on_chat_spans(
            agent,
            "do tool work",
            3,
            &[first_call_usage, second_call_usage],
        )
        .await;
    }

    #[tokio::test]
    async fn stream_prompt_emits_completion_call_before_finish_hook_termination() {
        let call_usage = usage(10, 2);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(call_usage),
        ]]);
        let agent = AgentBuilder::new(model).build();

        let mut stream = agent
            .stream_prompt("say done")
            .with_hook(TerminateOnStreamFinish)
            .await;
        let mut completion_calls = Vec::new();
        let mut saw_error = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_calls.push(completion_call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    panic!("unexpected final response after hook termination: {response:?}");
                }
                Ok(_) => {}
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }

        assert_eq!(completion_calls, vec![CompletionCall::new(0, call_usage)]);
        assert!(saw_error);
    }

    #[tokio::test]
    async fn stream_prompt_completion_calls_records_unreported_usage() {
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .with_history(empty_history)
            .multi_turn(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let expected_usage = vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, second_call_usage),
        ];
        assert_eq!(completion_calls_events, expected_usage);

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.completion_calls(), expected_usage.as_slice());
    }

    #[tokio::test]
    async fn final_response_matches_streamed_text_when_provider_final_is_textless() {
        let agent = AgentBuilder::new(streaming_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("say hello").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.response().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(streamed_text, "hello world");
        assert_eq!(final_response_text.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn final_response_preserves_structured_text_metadata() {
        let agent = AgentBuilder::new(streaming_cited_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("answer with citations").await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.response(), "cited answer");
        let metadata = text_metadata(final_response.content())
            .expect("expected text metadata in final content");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_history_preserves_structured_text_metadata() {
        let agent = AgentBuilder::new(streaming_cited_text_then_final_model()).build();

        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("answer with citations")
            .with_history(empty_history)
            .await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        let history = final_response
            .history()
            .expect("with_history should include final history");
        let assistant_content = history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in history");
        let metadata =
            text_metadata(assistant_content).expect("expected text metadata in assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn tool_follow_up_history_preserves_structured_text_metadata() {
        let model = streaming_cited_text_then_tool_model();
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("use a tool with citations")
            .with_history(empty_history)
            .multi_turn(3)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().collect::<Vec<_>>();
        let assistant_content = follow_up_history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in follow-up history");
        let metadata = text_metadata(assistant_content)
            .expect("expected citation metadata in follow-up assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_can_remain_empty_for_truly_textless_turns() {
        let agent = AgentBuilder::new(streaming_final_only_model()).build();

        let mut stream = agent.stream_prompt("say nothing").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.response().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(streamed_text.is_empty());
        assert_eq!(final_response_text.as_deref(), Some(""));
    }

    /// Background task that logs periodically to detect span leakage.
    /// If span leakage occurs, these logs will be prefixed with `invoke_agent{...}`.
    async fn background_logger(stop: Arc<AtomicBool>, leak_count: Arc<AtomicU32>) {
        let mut interval = tokio::time::interval(Duration::from_millis(50));
        let mut count = 0u32;

        while !stop.load(Ordering::Relaxed) {
            interval.tick().await;
            count += 1;

            tracing::event!(
                target: "background_logger",
                tracing::Level::INFO,
                count = count,
                "Background tick"
            );

            // Check if we're inside an unexpected span
            let current = tracing::Span::current();
            if !current.is_disabled() && !current.is_none() {
                leak_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::info!(target: "background_logger", total_ticks = count, "Background logger stopped");
    }

    /// Test that span context doesn't leak to concurrent tasks during streaming.
    ///
    /// This test verifies that using `.instrument()` instead of `span.enter()` in
    /// async_stream prevents thread-local span context from leaking to other tasks.
    ///
    /// Uses single-threaded runtime to force all tasks onto the same thread,
    /// making the span leak deterministic (it only occurs when tasks share a thread).
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "This requires an API key"]
    async fn test_span_context_isolation() -> anyhow::Result<()> {
        let stop = Arc::new(AtomicBool::new(false));
        let leak_count = Arc::new(AtomicU32::new(0));

        // Start background logger
        let bg_stop = stop.clone();
        let bg_leak = leak_count.clone();
        let bg_handle = tokio::spawn(async move {
            background_logger(bg_stop, bg_leak).await;
        });

        // Small delay to let background logger start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Make streaming request WITHOUT an outer span so rig creates its own invoke_agent span
        // (rig reuses current span if one exists, so we need to ensure there's no current span)
        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant.")
            .temperature(0.1)
            .max_tokens(100)
            .build();

        let mut stream = agent
            .stream_prompt("Say 'hello world' and nothing else.")
            .await;

        let mut full_content = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    full_content.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    break;
                }
                Err(e) => {
                    tracing::warn!("Error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        tracing::info!("Got response: {:?}", full_content);

        // Stop background logger
        stop.store(true, Ordering::Relaxed);
        bg_handle.await?;

        let leaks = leak_count.load(Ordering::Relaxed);
        anyhow::ensure!(
            leaks == 0,
            "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times. \
             This indicates that span.enter() is being used inside async_stream instead of .instrument()"
        );

        Ok(())
    }

    /// Test that FinalResponse contains the updated chat history when with_history is used.
    ///
    /// This verifies that:
    /// 1. FinalResponse.history() returns Some when with_history was called
    /// 2. The history contains both the user prompt and assistant response
    #[tokio::test]
    #[ignore = "This requires an API key"]
    async fn test_chat_history_in_final_response() -> anyhow::Result<()> {
        use crate::message::Message;

        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant. Keep responses brief.")
            .temperature(0.1)
            .max_tokens(50)
            .build();

        // Send streaming request with history
        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("Say 'hello' and nothing else.")
            .with_history(empty_history)
            .await;

        // Consume the stream and collect FinalResponse
        let mut response_text = String::new();
        let mut final_history = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    response_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_history = res.history().map(|h| h.to_vec());
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
                _ => {}
            }
        }

        let history = final_history
            .ok_or_else(|| anyhow::anyhow!("final response should include history"))?;

        // Should contain at least the user message
        anyhow::ensure!(
            history.iter().any(|m| matches!(m, Message::User { .. })),
            "History should contain the user message"
        );

        // Should contain the assistant response
        anyhow::ensure!(
            history
                .iter()
                .any(|m| matches!(m, Message::Assistant { .. })),
            "History should contain the assistant response"
        );

        tracing::info!(
            "History after streaming: {} messages, response: {:?}",
            history.len(),
            response_text
        );

        Ok(())
    }

    #[tokio::test]
    async fn streaming_appends_to_memory_after_final_response() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .build();

        let mut stream = agent
            .stream_prompt("hi there")
            .conversation("stream-thread")
            .await;

        let mut history_in_final = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    history_in_final = res.history().map(|h| h.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_history = history_in_final
            .expect("FinalResponse.history should be populated when memory is configured");
        assert_eq!(
            final_history.len(),
            2,
            "user prompt + assistant response in final history: {final_history:?}"
        );

        let stored = memory.load("stream-thread").await.unwrap();
        assert_eq!(stored.len(), 2, "memory should contain user + assistant");
    }

    #[tokio::test]
    async fn streaming_reasoning_without_tools_does_not_duplicate_final_history() {
        let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("final answer"),
            MockStreamEvent::reasoning("reasoned step").with_reasoning_id("rs_1"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]))
        .build();

        let mut stream = agent
            .stream_prompt("think before answering")
            .with_history(Vec::<Message>::new())
            .await;

        let mut history_in_final = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    history_in_final = res.history().map(|h| h.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_history = history_in_final
            .expect("FinalResponse.history should be populated when with_history is used");
        assert_eq!(
            final_history.len(),
            2,
            "user prompt + one assistant response in final history: {final_history:?}"
        );

        assert!(matches!(
            final_history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "think before answering"
                )
        ));

        let assistant_messages = final_history
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            assistant_messages.len(),
            1,
            "reasoning turn should produce exactly one assistant history message: {final_history:?}"
        );
        let assistant_content = assistant_messages
            .first()
            .expect("expected assistant history message");
        assert!(assistant_content.iter().any(|item| matches!(
            item,
            AssistantContent::Text(text) if text.text == "final answer"
        )));
        assert!(assistant_content.iter().any(|item| matches!(
            item,
            AssistantContent::Reasoning(reasoning)
                if reasoning.id.as_deref() == Some("rs_1")
                    && reasoning.content.iter().any(|content| matches!(
                        content,
                        ReasoningContent::Text { text, .. } if text == "reasoned step"
                    ))
        )));
        let reasoning_index = assistant_content
            .iter()
            .position(|item| matches!(item, AssistantContent::Reasoning(_)))
            .expect("assistant history should contain reasoning");
        let text_index = assistant_content
            .iter()
            .position(|item| matches!(item, AssistantContent::Text(_)))
            .expect("assistant history should contain text");
        assert!(
            reasoning_index < text_index,
            "assistant reasoning must be stored before assistant text: {assistant_content:?}"
        );
    }

    #[tokio::test]
    async fn streaming_with_history_overrides_memory() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        memory
            .append("t1", vec![Message::user("from-memory")])
            .await
            .unwrap();

        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .build();

        let mut stream = agent
            .stream_prompt("hi")
            .conversation("t1")
            .with_history(vec![Message::user("from-caller")])
            .await;

        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let stored = memory.load("t1").await.unwrap();
        assert_eq!(
            stored.len(),
            1,
            "with_history bypasses memory; only the pre-seeded entry remains: {stored:?}"
        );
    }

    #[tokio::test]
    async fn streaming_without_memory_disables_for_request() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new();
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(memory.clone())
            .conversation_id("default")
            .build();

        let mut stream = agent.stream_prompt("hi").without_memory().await;

        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let stored = memory.load("default").await.unwrap();
        assert!(stored.is_empty(), "without_memory disables save");
    }

    #[tokio::test]
    async fn streaming_load_error_yields_memory_error() {
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(FailingMemory::default())
            .build();

        let mut stream = agent.stream_prompt("hi").conversation("t1").await;

        let first = stream.next().await.expect("at least one item");
        match first {
            Err(err) => {
                let msg = format!("{err:?}");
                assert!(
                    msg.contains("Memory") || msg.contains("memory") || msg.contains("load boom"),
                    "expected memory error, got: {msg}"
                );
            }
            Ok(other) => panic!("expected memory error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streaming_with_filter_shapes_loaded_history() {
        use crate::memory::{ConversationMemory, InMemoryConversationMemory};

        let memory = InMemoryConversationMemory::new()
            .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).rev().collect());
        memory
            .append(
                "t1",
                vec![
                    Message::user("1"),
                    Message::assistant("2"),
                    Message::user("3"),
                    Message::assistant("4"),
                ],
            )
            .await
            .unwrap();

        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("ok"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ]]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model).memory(memory).build();

        let mut stream = agent.stream_prompt("ping").conversation("t1").await;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                break;
            }
        }

        let received = recorded.requests()[0]
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            received.len(),
            3,
            "window-truncated history (2) + current prompt: {received:?}"
        );
    }

    #[tokio::test]
    async fn streaming_append_error_does_not_suppress_final_response() {
        let agent = AgentBuilder::new(streaming_text_then_final_model())
            .memory(AppendFailingMemory::default())
            .build();

        let mut stream = agent.stream_prompt("hi").conversation("t1").await;

        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(_)) = item {
                saw_final = true;
                break;
            }
        }
        assert!(
            saw_final,
            "FinalResponse must be yielded even when memory.append fails"
        );
    }
}
