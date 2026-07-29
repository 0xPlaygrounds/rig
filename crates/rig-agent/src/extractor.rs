//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the OpenAI client
//! let openai = openai::Client::new("your-open-ai-api-key")?;
//!
//! // Define the structure of the data you want to extract
//! #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
//! struct Person {
//!    name: Option<String>,
//!    age: Option<u8>,
//!    profession: Option<String>,
//! }
//!
//! // Create the extractor
//! let extractor = openai.extractor::<Person>(openai::GPT_4O)
//!     .build();
//!
//! // Extract structured data from text
//! let person = extractor.extract("John Doe is a 30 year old doctor.").await?;
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use rig_core::{
    message::{Message, ToolChoice},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    agent::{Agent, AgentBuilder, OutputMode},
    completion::{CompletionError, PromptError, Usage},
    provider::ProviderConfig,
};

const SUBMIT_TOOL_NAME: &str = "submit";

/// Response from an extraction operation containing the extracted data and usage information.
#[derive(Debug, Clone)]
pub struct ExtractionResponse<T> {
    /// The extracted structured data
    pub data: T,
    /// Accumulated token usage across all attempts (including retries)
    pub usage: Usage,
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("No data extracted")]
    NoData,

    #[error("Failed to deserialize the extracted data: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    #[error("PromptError: {0}")]
    PromptError(#[from] PromptError),
}

/// Extractor for structured data from text
pub struct Extractor<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    agent: Agent,
    _t: PhantomData<T>,
    retries: u64,
}

impl<T> Extractor<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract(
        &self,
        text: impl Into<Message> + WasmCompatSend,
    ) -> Result<T, ExtractionError> {
        let (data, _usage) = self.retry_extract(text.into(), vec![]).await?;
        Ok(data)
    }

    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract_with_chat_history(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        let (data, _usage) = self.retry_extract(text.into(), chat_history).await?;
        Ok(data)
    }

    /// Attempts to extract data from the given text with a number of retries,
    /// returning both the extracted data and accumulated token usage.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    ///
    /// Usage accumulates across all retry attempts, including attempts that received
    /// a billed response but failed extraction (e.g. the model never called `submit`).
    /// Attempts whose completion call itself returned an error (e.g. network failures
    /// or unparseable provider responses) contribute no usage, and when every attempt
    /// fails the returned error carries no usage information at all.
    pub async fn extract_with_usage(
        &self,
        text: impl Into<Message> + WasmCompatSend,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        let (data, usage) = self.retry_extract(text.into(), vec![]).await?;
        Ok(ExtractionResponse { data, usage })
    }

    /// Attempts to extract data from the given text with a number of retries,
    /// providing chat history context, and returning both the extracted data
    /// and accumulated token usage.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    ///
    /// Usage accumulates across all retry attempts, including attempts that received
    /// a billed response but failed extraction (e.g. the model never called `submit`).
    /// Attempts whose completion call itself returned an error (e.g. network failures
    /// or unparseable provider responses) contribute no usage, and when every attempt
    /// fails the returned error carries no usage information at all.
    pub async fn extract_with_chat_history_with_usage(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        let (data, usage) = self.retry_extract(text.into(), chat_history).await?;
        Ok(ExtractionResponse { data, usage })
    }

    /// Runs the extraction with the retry semantics shared by all public
    /// `extract*` methods, returning the extracted data and the token usage
    /// accumulated across all attempts, including failed ones. The accumulated
    /// usage is only observable on success: when every attempt fails, the
    /// returned error cannot carry it.
    async fn retry_extract(
        &self,
        text: Message,
        chat_history: Vec<Message>,
    ) -> Result<(T, Usage), ExtractionError> {
        let mut last_error = None;
        let mut usage = Usage::new();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let (result, attempt_usage) = self.extract_json_with_usage(&text, &chat_history).await;
            usage += attempt_usage;
            match result {
                Ok(data) => return Ok((data, usage)),
                Err(e) => {
                    let suffix = if i < self.retries { " Retrying..." } else { "" };
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}.{suffix}");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
    }

    /// Performs a single extraction attempt, returning its outcome alongside
    /// the token usage it consumed. Usage is reported even when the attempt
    /// fails after a billed completion (e.g. the model never called `submit`);
    /// it is zero whenever the completion call itself returns an error, since
    /// `CompletionError` carries no usage — even if the provider billed the
    /// request (e.g. an unparseable response body).
    async fn extract_json_with_usage(
        &self,
        text: &Message,
        messages: &[Message],
    ) -> (Result<T, ExtractionError>, Usage) {
        let (result, error_usage) = self
            .agent
            .runner(text.clone())
            .history(messages.iter().cloned())
            .max_turns(1)
            .output_tool(
                SUBMIT_TOOL_NAME,
                "Submit the structured data you extracted from the provided text.",
                false,
            )
            .ignore_unhandled_invalid_tool_calls()
            .run_with_error_usage()
            .await;
        let response = match result {
            Ok(response) => response,
            Err(PromptError::CompletionError(e)) => {
                return (Err(ExtractionError::CompletionError(e)), error_usage);
            }
            Err(e) => return (Err(e.into()), error_usage),
        };
        let usage = response.usage;

        let submissions = response.output_tool_calls();
        if submissions == 0 {
            tracing::warn!(
                "The submit tool was not called. If this happens more than once, please ensure the model you are using is powerful enough to reliably call tools."
            );
            return (Err(ExtractionError::NoData), usage);
        }
        if submissions > 1 {
            tracing::warn!(
                "Multiple submit calls detected, using the first one. Providers / agents should only ensure one submit call."
            );
        }

        (
            serde_json::from_str(&response.output).map_err(ExtractionError::from),
            usage,
        )
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilder,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

impl<T> ExtractorBuilder<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(provider: ProviderConfig) -> Self {
        Self {
            agent_builder: AgentBuilder::new(provider)
                .preamble("\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, even with default values!!!.
                ")
                .output_schema::<T>()
                .tool_choice(ToolChoice::Required)
                .output_mode(OutputMode::Tool),
            retries: None,
            _t: PhantomData,
        }
    }

    /// Add additional preamble to the extractor
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.agent_builder = self.agent_builder.append_preamble(&format!(
            "\n=============== ADDITIONAL INSTRUCTIONS ===============\n{preamble}"
        ));
        self
    }

    /// Add a context document to the extractor
    pub fn context(mut self, doc: &str) -> Self {
        self.agent_builder = self.agent_builder.context(doc);
        self
    }

    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.agent_builder = self.agent_builder.additional_params(params);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.agent_builder = self.agent_builder.max_tokens(max_tokens);
        self
    }

    /// Use an existing provider [`Runtime`](crate::provider::Runtime) for the
    /// inner Agent instead of building a fresh one.
    pub fn runtime(mut self, rt: std::sync::Arc<crate::provider::Runtime>) -> Self {
        self.agent_builder = self.agent_builder.runtime(rt);
        self
    }

    /// Set the maximum number of retries for the extractor.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = Some(retries);
        self
    }

    /// Set the `tool_choice` option for the inner Agent.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.agent_builder = self.agent_builder.tool_choice(choice);
        self
    }

    /// Add a provider-independent lifecycle hook to every extraction attempt.
    ///
    /// Completion-response hooks receive canonical Rig content, usage, prompt,
    /// and message ID fields, just like hooks attached directly to an agent.
    pub fn add_hook(mut self, hook: crate::hooks::HookEntry) -> Self {
        self.agent_builder = self.agent_builder.add_hook(hook);
        self
    }

    /// Build the Extractor
    pub fn build(self) -> Extractor<T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
            retries: self.retries.unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use serde_json::json;

    use super::*;
    use crate::agent::{
        CompletionCallAction, InvalidToolCallAction, ModelTurnAction, ObservationAction,
    };
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::provider::MockScript;
    use rig_core::OneOrMany;
    use rig_core::completion::CompletionResponse as ModelResponse;
    use rig_core::embeddings::Embedding;
    use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
    use rig_core::vector_store::{
        StoreRecord, VectorSearchRequest, in_memory_store::InMemoryVectorStore,
    };

    #[derive(Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
    struct Person {
        name: String,
    }

    fn usage(total_tokens: u64) -> Usage {
        Usage {
            total_tokens,
            ..Usage::new()
        }
    }

    fn extractor(script: MockScript, retries: u64) -> Extractor<Person> {
        ExtractorBuilder::new(ProviderConfig::Mock(script))
            .retries(retries)
            .build()
    }

    fn response_from(contents: Vec<AssistantContent>, usage: Usage) -> ModelResponse {
        let choice = OneOrMany::many(contents).expect("at least one content item");
        ModelResponse::new(choice, usage, "mock")
    }

    fn text_response(text: &str) -> ModelResponse {
        response_from(vec![AssistantContent::text(text)], Usage::new())
    }

    fn submit_response(name: &str) -> ModelResponse {
        response_from(
            vec![tool_call("id1", SUBMIT_TOOL_NAME, json!({ "name": name }))],
            Usage::new(),
        )
    }

    /// Placeholder response occupying an index that `with_errors` fails.
    fn error_slot() -> ModelResponse {
        text_response("unused error slot")
    }

    fn with_usage(mut response: ModelResponse, usage: Usage) -> ModelResponse {
        response.usage = usage;
        response
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall::new(
            id.to_string(),
            ToolFunction::new(name.to_string(), arguments),
        ))
    }

    /// Named hook entry over a synchronous decision function.
    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::new(name, move |event| {
            let decision = decide(event);
            Box::pin(async move { decision })
        })
    }

    #[derive(Clone, Default)]
    struct LifecycleCounts {
        completion_calls: Arc<AtomicUsize>,
        completion_responses: Arc<AtomicUsize>,
        model_turns: Arc<AtomicUsize>,
        invalid_tool_calls: Arc<AtomicUsize>,
    }

    /// One entry counting every lifecycle event the classic `LifecycleCounts`
    /// hook counted, always deferring (`Continue` / `None`).
    fn lifecycle_counts_entry(counts: LifecycleCounts) -> HookEntry {
        hook_entry("lifecycle-counts", move |event| {
            match event {
                HookEvent::BeforeModelCall { .. } => {
                    counts.completion_calls.fetch_add(1, Ordering::SeqCst);
                }
                HookEvent::CompletionResponse { .. } => {
                    counts.completion_responses.fetch_add(1, Ordering::SeqCst);
                }
                HookEvent::ModelTurnFinished { .. } => {
                    counts.model_turns.fetch_add(1, Ordering::SeqCst);
                }
                HookEvent::InvalidToolCall(_) => {
                    counts.invalid_tool_calls.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
            HookDecision::Continue
        })
    }

    type ExtractorResponseSnapshot = (Message, Vec<AssistantContent>, Usage, Option<String>);

    #[derive(Clone, Default)]
    struct ExtractorResponseCapture {
        snapshot: Arc<Mutex<Option<ExtractorResponseSnapshot>>>,
    }

    fn response_capture_entry(capture: ExtractorResponseCapture) -> HookEntry {
        hook_entry("extractor-response-capture", move |event| {
            let HookEvent::CompletionResponse {
                prompt, response, ..
            } = event
            else {
                return HookDecision::Continue;
            };
            *capture
                .snapshot
                .lock()
                .expect("extractor response snapshot") = Some((
                prompt,
                response.choice.iter().cloned().collect(),
                response.usage,
                response.message_id.clone(),
            ));
            HookDecision::Observation(ObservationAction::continue_run())
        })
    }

    fn stop_before_completion_entry() -> HookEntry {
        hook_entry("stop-before-completion", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::stop("extractor stopped"))
        })
    }

    /// The passive-RAG hook recipe applied to an extractor: embed the prompt,
    /// query a concrete store, and inject the hits as per-turn context.
    struct ExtractorRagHook {
        embedder: crate::provider::EmbedderConfig,
        rt: Arc<crate::provider::Runtime>,
        store: Arc<InMemoryVectorStore>,
        samples: u64,
        queries: Arc<Mutex<Vec<(String, u64)>>>,
    }

    fn extractor_rag_entry(hook: ExtractorRagHook) -> HookEntry {
        let hook = Arc::new(hook);
        HookEntry::new("extractor-rag", move |event| {
            let hook = hook.clone();
            Box::pin(async move {
                use crate::agent::RequestPatch;

                let HookEvent::BeforeModelCall { prompt, .. } = event else {
                    return HookDecision::Continue;
                };
                let Some(query) = prompt.rag_text() else {
                    return HookDecision::CompletionCall(CompletionCallAction::continue_run());
                };
                hook.queries
                    .lock()
                    .expect("extractor query recorder")
                    .push((query.clone(), hook.samples));

                let embedded =
                    match crate::provider::embed(&hook.embedder, &hook.rt, vec![query]).await {
                        Ok(response) => response.embeddings.into_iter().next(),
                        Err(error) => {
                            return HookDecision::CompletionCall(CompletionCallAction::stop(
                                format!("query embedding failed: {error}"),
                            ));
                        }
                    };
                let Some(embedded) = embedded else {
                    return HookDecision::CompletionCall(CompletionCallAction::stop(
                        "query embedding was empty",
                    ));
                };

                let request = VectorSearchRequest::builder()
                    .query(embedded)
                    .samples(hook.samples)
                    .build();
                match hook.store.top_n(request).await {
                    Ok(hits) => HookDecision::CompletionCall(CompletionCallAction::patch(
                        RequestPatch::new().extra_context(hits.into_iter().map(|hit| {
                            crate::completion::Document {
                                id: hit.id,
                                text: serde_json::to_string_pretty(&hit.payload)
                                    .unwrap_or_else(|_| hit.payload.to_string()),
                                additional_props: Default::default(),
                            }
                        })),
                    )),
                    Err(error) => HookDecision::CompletionCall(CompletionCallAction::stop(
                        format!("context retrieval failed: {error}"),
                    )),
                }
            })
        })
    }

    #[derive(Clone, Copy)]
    enum StopFirstBilledResponseAt {
        CompletionResponse,
        ModelTurnFinished,
    }

    /// Stops the run at the first event of `phase`, deferring afterwards.
    fn stop_first_billed_response_entry(phase: StopFirstBilledResponseAt) -> HookEntry {
        let calls = Arc::new(AtomicUsize::new(0));
        hook_entry("stop-first-billed-response", move |event| match event {
            HookEvent::CompletionResponse { .. } => {
                if matches!(phase, StopFirstBilledResponseAt::CompletionResponse)
                    && calls.fetch_add(1, Ordering::SeqCst) == 0
                {
                    HookDecision::Observation(ObservationAction::stop("stop first billed response"))
                } else {
                    HookDecision::Observation(ObservationAction::continue_run())
                }
            }
            HookEvent::ModelTurnFinished { .. } => {
                if matches!(phase, StopFirstBilledResponseAt::ModelTurnFinished)
                    && calls.fetch_add(1, Ordering::SeqCst) == 0
                {
                    HookDecision::ModelTurn(ModelTurnAction::stop("stop first billed model turn"))
                } else {
                    HookDecision::ModelTurn(ModelTurnAction::continue_run())
                }
            }
            _ => HookDecision::Continue,
        })
    }

    fn stop_on_invalid_tool_call_entry() -> HookEntry {
        hook_entry("stop-on-invalid-tool-call", |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::stop(
                "unexpected extractor tool call",
            ))
        })
    }

    fn repair_unexpected_as_submit_entry() -> HookEntry {
        hook_entry("repair-unexpected-as-submit", |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair(SUBMIT_TOOL_NAME))
        })
    }

    fn skip_unexpected_entry() -> HookEntry {
        hook_entry("skip-unexpected", |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::skip("ignored by extractor hook"))
        })
    }

    #[tokio::test]
    async fn extractor_runs_through_full_response_lifecycle() {
        let script = MockScript::from_responses(vec![submit_response("John")]);
        let probe = script.clone();
        let counts = LifecycleCounts::default();
        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(lifecycle_counts_entry(counts.clone()))
            .build()
            .extract("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.name, "John");
        assert_eq!(probe.calls(), 1);
        assert_eq!(counts.completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counts.completion_responses.load(Ordering::SeqCst), 1);
        assert_eq!(counts.model_turns.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn extractor_hook_receives_canonical_response_fields() {
        let capture = ExtractorResponseCapture::default();
        let expected_usage = usage(23);
        let script = MockScript::from_responses(vec![
            with_usage(submit_response("John"), expected_usage)
                .with_message_id("extractor-message"),
        ]);
        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(response_capture_entry(capture.clone()))
            .build()
            .extract("John")
            .await
            .expect("extraction should succeed");
        assert_eq!(response.name, "John");

        let (prompt, content, observed_usage, message_id) = capture
            .snapshot
            .lock()
            .expect("extractor response snapshot")
            .clone()
            .expect("extractor response hook should fire");
        assert_eq!(prompt, Message::user("John"));
        assert_eq!(observed_usage, expected_usage);
        assert_eq!(message_id.as_deref(), Some("extractor-message"));
        assert!(matches!(
            content.as_slice(),
            [AssistantContent::ToolCall(tool_call)]
                if tool_call.function.name == SUBMIT_TOOL_NAME
                    && tool_call.function.arguments == json!({"name": "John"})
        ));
    }

    #[tokio::test]
    async fn extractor_rag_hook_uses_the_agent_hook_lifecycle() {
        let script = MockScript::from_responses(vec![submit_response("John")]);
        let probe = script.clone();
        let queries = Arc::new(Mutex::new(Vec::new()));

        let store = InMemoryVectorStore::new();
        store
            .insert(vec![StoreRecord {
                id: "extractor-context".to_string(),
                payload: json!({ "question": "retrieved" }),
                embeddings: OneOrMany::one(Embedding {
                    document: "retrieved".to_string(),
                    vec: vec![1.0, 0.0],
                }),
            }])
            .await
            .expect("store insert should succeed");

        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(extractor_rag_entry(ExtractorRagHook {
                embedder: crate::provider::EmbedderConfig::Mock(
                    crate::provider::MockEmbedder::from_responses(vec![vec![vec![1.0, 0.0]]]),
                ),
                rt: Arc::new(crate::provider::Runtime::new()),
                store: Arc::new(store),
                samples: 2,
                queries: queries.clone(),
            }))
            .build()
            .extract("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.name, "John");
        assert_eq!(
            *queries.lock().expect("extractor queries"),
            vec![("John".to_string(), 2)]
        );
        let requests = probe.requests();
        let request = requests.first().expect("one extractor request");
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.id == "extractor-context"
                    && document.text == "{\n  \"question\": \"retrieved\"\n}")
        );
    }

    #[tokio::test]
    async fn extractor_completion_call_stop_prevents_provider_io() {
        let script = MockScript::from_responses(vec![submit_response("John")]);
        let probe = script.clone();
        let error = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(stop_before_completion_entry())
            .build()
            .extract("John")
            .await
            .expect_err("terminating hook should cancel extraction");

        assert!(matches!(
            error,
            ExtractionError::PromptError(PromptError::PromptCancelled { reason, .. })
                if reason == "extractor stopped"
        ));
        assert_eq!(probe.calls(), 0);
    }

    #[tokio::test]
    async fn usage_accumulates_across_failed_attempts() {
        let script = MockScript::from_responses(vec![
            with_usage(text_response("no submit call"), usage(10)),
            with_usage(submit_response("John"), usage(5)),
        ]);

        let response = extractor(script, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(
            response.data,
            Person {
                name: "John".to_string()
            }
        );
        assert_eq!(response.usage.total_tokens, 15);
    }

    async fn assert_billed_hook_termination_usage(phase: StopFirstBilledResponseAt) {
        let script = MockScript::from_responses(vec![
            with_usage(submit_response("ignored"), usage(10)),
            with_usage(submit_response("John"), usage(5)),
        ]);
        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .retries(1)
            .add_hook(stop_first_billed_response_entry(phase))
            .build()
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn completion_response_hook_termination_preserves_billed_usage() {
        assert_billed_hook_termination_usage(StopFirstBilledResponseAt::CompletionResponse).await;
    }

    #[tokio::test]
    async fn model_turn_finished_hook_termination_preserves_billed_usage() {
        assert_billed_hook_termination_usage(StopFirstBilledResponseAt::ModelTurnFinished).await;
    }

    #[tokio::test]
    async fn unexpected_tool_call_preserves_usage_and_retries() {
        let script = MockScript::from_responses(vec![
            response_from(
                vec![tool_call("unknown", "unexpected", json!({}))],
                usage(10),
            ),
            with_usage(submit_response("John"), usage(5)),
        ]);

        let response = extractor(script, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn unexpected_tool_call_runs_hooks_before_extractor_fallback() {
        let script = MockScript::from_responses(vec![
            response_from(
                vec![tool_call("unknown", "unexpected", json!({}))],
                usage(10),
            ),
            with_usage(submit_response("John"), usage(5)),
        ]);
        let counts = LifecycleCounts::default();

        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .retries(1)
            .add_hook(lifecycle_counts_entry(counts.clone()))
            .build()
            .extract_with_usage("John")
            .await
            .expect("deferred invalid call should use extractor fallback");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(counts.invalid_tool_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counts.completion_responses.load(Ordering::SeqCst), 2);
        assert_eq!(counts.model_turns.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unexpected_tool_call_hook_can_stop_extraction() {
        let script = MockScript::from_responses(vec![response_from(
            vec![tool_call("unknown", "unexpected", json!({}))],
            Usage::new(),
        )]);

        let error = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(stop_on_invalid_tool_call_entry())
            .build()
            .extract("John")
            .await
            .expect_err("invalid-tool hook should retain control");

        assert!(matches!(
            error,
            ExtractionError::PromptError(PromptError::PromptCancelled { reason, .. })
                if reason == "unexpected extractor tool call"
        ));
    }

    #[tokio::test]
    async fn unexpected_tool_call_hook_can_repair_to_submit() {
        let script = MockScript::from_responses(vec![response_from(
            vec![tool_call(
                "unknown",
                "unexpected",
                json!({ "name": "John" }),
            )],
            Usage::new(),
        )]);

        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(repair_unexpected_as_submit_entry())
            .build()
            .extract("John")
            .await
            .expect("repaired output-tool call should finalize extraction");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn skip_hook_preserves_valid_submit_sibling() {
        let script = MockScript::from_responses(vec![response_from(
            vec![
                tool_call("unknown", "unexpected", json!({})),
                tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
            ],
            Usage::new(),
        )]);

        let response = ExtractorBuilder::<Person>::new(ProviderConfig::Mock(script))
            .add_hook(skip_unexpected_entry())
            .build()
            .extract("John")
            .await
            .expect("skipping an invalid sibling should preserve submit");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn submit_call_wins_over_unexpected_sibling_call() {
        let script = MockScript::from_responses(vec![response_from(
            vec![
                tool_call("unknown", "unexpected", json!({})),
                tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
            ],
            usage(7),
        )]);

        let response = extractor(script, 0)
            .extract_with_usage("John")
            .await
            .expect("submit should remain authoritative");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn submit_call_wins_before_unexpected_sibling_call() {
        let script = MockScript::from_responses(vec![response_from(
            vec![
                tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
                tool_call("unknown", "unexpected", json!({})),
            ],
            Usage::new(),
        )]);

        let response = extractor(script, 0)
            .extract("John")
            .await
            .expect("an earlier submit should remain authoritative");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn multiple_unexpected_calls_surrounding_submit_are_ignored() {
        let script = MockScript::from_responses(vec![response_from(
            vec![
                tool_call("unknown-before", "unexpected_before", json!({})),
                tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
                tool_call("unknown-after", "unexpected_after", json!({})),
            ],
            Usage::new(),
        )]);

        let response = extractor(script, 0)
            .extract("John")
            .await
            .expect("unexpected siblings should not displace submit");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn transport_errors_contribute_no_usage() {
        let script = MockScript::from_responses(vec![
            error_slot(),
            with_usage(submit_response("John"), usage(5)),
        ])
        .with_errors(vec![Some("boom".to_string()), None]);

        let response = extractor(script, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn single_successful_attempt_reports_its_own_usage() {
        let script =
            MockScript::from_responses(vec![with_usage(submit_response("John"), usage(7))]);

        let response = extractor(script, 0)
            .extract_with_usage("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn exhausted_retries_return_last_error() {
        let script = MockScript::from_responses(vec![with_usage(
            text_response("no submit call"),
            usage(10),
        )]);

        let err = extractor(script, 0)
            .extract("John")
            .await
            .expect_err("extraction should fail");

        assert!(matches!(err, ExtractionError::NoData));
    }

    #[tokio::test]
    async fn exhausted_retries_return_error_from_final_attempt() {
        let script = MockScript::from_responses(vec![error_slot(), error_slot()])
            .with_errors(vec![Some("first".to_string()), Some("second".to_string())]);

        let err = extractor(script, 1)
            .extract("John")
            .await
            .expect_err("extraction should fail");

        assert!(matches!(
            err,
            ExtractionError::CompletionError(CompletionError::ProviderError(message))
                if message == "second"
        ));
    }
}
