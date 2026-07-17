//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::openai};
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

use crate::{
    agent::{Agent, AgentBuilder, AgentHook, OutputMode},
    completion::{CompletionError, CompletionModel, PromptError, Usage},
    message::{Message, ToolChoice},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
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
pub struct Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    agent: Agent<M>,
    _t: PhantomData<T>,
    retries: u64,
}

impl<M, T> Extractor<M, T>
where
    M: CompletionModel,
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
pub struct ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilder<M>,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

impl<M, T> ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            agent_builder: AgentBuilder::new(model)
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
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.agent_builder = self.agent_builder.add_hook(hook);
        self
    }

    /// Build the Extractor
    pub fn build(self) -> Extractor<M, T> {
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
        CompletionCallAction, HookContext, ModelTurnPrepared, ObservationAction, RequestPatch,
    };
    use crate::message::{AssistantContent, ToolCall, ToolFunction};
    use crate::test_utils::{MockCompletionModel, MockTurn};

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

    fn extractor(
        model: MockCompletionModel,
        retries: u64,
    ) -> Extractor<MockCompletionModel, Person> {
        ExtractorBuilder::new(model).retries(retries).build()
    }

    fn submit_turn(name: &str) -> MockTurn {
        MockTurn::tool_call("id1", SUBMIT_TOOL_NAME, json!({ "name": name }))
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall::new(
            id.to_string(),
            ToolFunction::new(name.to_string(), arguments),
        ))
    }

    #[derive(Clone, Default)]
    struct LifecycleCounts {
        completion_calls: Arc<AtomicUsize>,
        prepared_turns: Arc<AtomicUsize>,
        invalid_tool_calls: Arc<AtomicUsize>,
    }

    impl AgentHook for LifecycleCounts {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> crate::agent::CompletionCallAction {
            self.completion_calls.fetch_add(1, Ordering::SeqCst);
            crate::agent::CompletionCallAction::Continue
        }

        async fn on_model_turn_prepared(
            &self,
            _ctx: &HookContext,
            _event: ModelTurnPrepared<'_>,
        ) -> ObservationAction {
            self.prepared_turns.fetch_add(1, Ordering::SeqCst);
            ObservationAction::Continue
        }

        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &crate::agent::InvalidToolCallContext,
        ) -> Option<crate::agent::InvalidToolCallAction> {
            self.invalid_tool_calls.fetch_add(1, Ordering::SeqCst);
            None
        }
    }

    type ExtractorResponseSnapshot = (Message, Vec<AssistantContent>, Usage, Option<String>);

    #[derive(Clone, Default)]
    struct ExtractorResponseCapture {
        snapshot: Arc<Mutex<Option<ExtractorResponseSnapshot>>>,
    }

    impl AgentHook for ExtractorResponseCapture {
        async fn on_model_turn_prepared(
            &self,
            _ctx: &HookContext,
            event: ModelTurnPrepared<'_>,
        ) -> ObservationAction {
            *self.snapshot.lock().expect("extractor response snapshot") = Some((
                event.prompt.clone(),
                event.content.iter().cloned().collect(),
                event.usage,
                event.message_id.map(str::to_owned),
            ));
            ObservationAction::continue_run()
        }
    }

    struct StopBeforeCompletion;

    impl AgentHook for StopBeforeCompletion {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> crate::agent::CompletionCallAction {
            crate::agent::CompletionCallAction::stop("extractor stopped")
        }
    }

    struct ExtractorRetrievalHook;

    impl AgentHook for ExtractorRetrievalHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            CompletionCallAction::patch(RequestPatch::new().context(crate::completion::Document {
                id: "application-retrieval".to_string(),
                text: "retrieved extractor context".to_string(),
                additional_props: Default::default(),
            }))
        }
    }

    #[derive(Clone)]
    struct StopFirstBilledResponse {
        calls: Arc<AtomicUsize>,
    }

    impl AgentHook for StopFirstBilledResponse {
        async fn on_model_turn_prepared(
            &self,
            _ctx: &HookContext,
            _event: ModelTurnPrepared<'_>,
        ) -> ObservationAction {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                ObservationAction::stop("stop first billed prepared turn")
            } else {
                ObservationAction::continue_run()
            }
        }
    }

    struct StopOnInvalidToolCall;

    impl AgentHook for StopOnInvalidToolCall {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &crate::agent::InvalidToolCallContext,
        ) -> Option<crate::agent::InvalidToolCallAction> {
            Some(crate::agent::InvalidToolCallAction::stop(
                "unexpected extractor tool call",
            ))
        }
    }

    struct RepairUnexpectedAsSubmit;

    impl AgentHook for RepairUnexpectedAsSubmit {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &crate::agent::InvalidToolCallContext,
        ) -> Option<crate::agent::InvalidToolCallAction> {
            Some(crate::agent::InvalidToolCallAction::repair(
                SUBMIT_TOOL_NAME,
            ))
        }
    }

    struct SkipUnexpected;

    impl AgentHook for SkipUnexpected {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &crate::agent::InvalidToolCallContext,
        ) -> Option<crate::agent::InvalidToolCallAction> {
            Some(crate::agent::InvalidToolCallAction::skip(
                "ignored by extractor hook",
            ))
        }
    }

    #[tokio::test]
    async fn extractor_runs_through_full_response_lifecycle() {
        let model = MockCompletionModel::new([submit_turn("John")]);
        let counts = LifecycleCounts::default();
        let response = ExtractorBuilder::<_, Person>::new(model.clone())
            .add_hook(counts.clone())
            .build()
            .extract("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.name, "John");
        assert_eq!(model.request_count(), 1);
        assert_eq!(counts.completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counts.prepared_turns.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn extractor_hook_receives_canonical_response_fields() {
        let capture = ExtractorResponseCapture::default();
        let expected_usage = usage(23);
        let response =
            ExtractorBuilder::<_, Person>::new(MockCompletionModel::new([submit_turn("John")
                .with_usage(expected_usage)
                .with_message_id("extractor-message")]))
            .add_hook(capture.clone())
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
    async fn extractor_accepts_application_defined_retrieval_hook() {
        let model = MockCompletionModel::new([submit_turn("John")]);
        let probe = model.clone();
        let response = ExtractorBuilder::<_, Person>::new(model)
            .add_hook(ExtractorRetrievalHook)
            .build()
            .extract("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.name, "John");
        let requests = probe.requests();
        let request = requests.first().expect("one extractor request");
        assert!(
            request
                .documents
                .iter()
                .any(|document| document.id == "application-retrieval"
                    && document.text == "retrieved extractor context")
        );
    }

    #[tokio::test]
    async fn extractor_completion_call_stop_prevents_provider_io() {
        let model = MockCompletionModel::new([submit_turn("John")]);
        let error = ExtractorBuilder::<_, Person>::new(model.clone())
            .add_hook(StopBeforeCompletion)
            .build()
            .extract("John")
            .await
            .expect_err("terminating hook should cancel extraction");

        assert!(matches!(
            error,
            ExtractionError::PromptError(PromptError::PromptCancelled { reason, .. })
                if reason == "extractor stopped"
        ));
        assert_eq!(model.request_count(), 0);
    }

    #[tokio::test]
    async fn usage_accumulates_across_failed_attempts() {
        let model = MockCompletionModel::new([
            MockTurn::text("no submit call").with_usage(usage(10)),
            submit_turn("John").with_usage(usage(5)),
        ]);

        let response = extractor(model, 1)
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

    async fn assert_billed_hook_termination_usage() {
        let model = MockCompletionModel::new([
            submit_turn("ignored").with_usage(usage(10)),
            submit_turn("John").with_usage(usage(5)),
        ]);
        let response = ExtractorBuilder::<_, Person>::new(model)
            .retries(1)
            .add_hook(StopFirstBilledResponse {
                calls: Arc::new(AtomicUsize::new(0)),
            })
            .build()
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn prepared_hook_termination_preserves_billed_usage() {
        assert_billed_hook_termination_usage().await;
    }

    #[tokio::test]
    async fn unexpected_tool_call_preserves_usage_and_retries() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("unknown", "unexpected", json!({})).with_usage(usage(10)),
            submit_turn("John").with_usage(usage(5)),
        ]);

        let response = extractor(model, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn unexpected_tool_call_runs_hooks_before_extractor_fallback() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("unknown", "unexpected", json!({})).with_usage(usage(10)),
            submit_turn("John").with_usage(usage(5)),
        ]);
        let counts = LifecycleCounts::default();

        let response = ExtractorBuilder::<_, Person>::new(model)
            .retries(1)
            .add_hook(counts.clone())
            .build()
            .extract_with_usage("John")
            .await
            .expect("deferred invalid call should use extractor fallback");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(counts.invalid_tool_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counts.prepared_turns.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unexpected_tool_call_hook_can_stop_extraction() {
        let model =
            MockCompletionModel::new([MockTurn::tool_call("unknown", "unexpected", json!({}))]);

        let error = ExtractorBuilder::<_, Person>::new(model)
            .add_hook(StopOnInvalidToolCall)
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
        let model = MockCompletionModel::new([MockTurn::tool_call(
            "unknown",
            "unexpected",
            json!({ "name": "John" }),
        )]);

        let response = ExtractorBuilder::<_, Person>::new(model)
            .add_hook(RepairUnexpectedAsSubmit)
            .build()
            .extract("John")
            .await
            .expect("repaired output-tool call should finalize extraction");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn skip_hook_preserves_valid_submit_sibling() {
        let turn = MockTurn::from_contents([
            tool_call("unknown", "unexpected", json!({})),
            tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
        ])
        .expect("two tool calls");
        let model = MockCompletionModel::new([turn]);

        let response = ExtractorBuilder::<_, Person>::new(model)
            .add_hook(SkipUnexpected)
            .build()
            .extract("John")
            .await
            .expect("skipping an invalid sibling should preserve submit");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn extractor_prepared_hook_receives_filtered_accepted_content() {
        let capture = ExtractorResponseCapture::default();
        let turn = MockTurn::from_contents([
            tool_call("unknown", "unexpected", json!({})),
            tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
        ])
        .expect("two tool calls");

        let response = ExtractorBuilder::<_, Person>::new(MockCompletionModel::new([turn]))
            .add_hook(capture.clone())
            .build()
            .extract("John")
            .await
            .expect("extractor filtering should preserve the valid submit call");

        assert_eq!(response.name, "John");
        let (_, accepted_content, _, _) = capture
            .snapshot
            .lock()
            .expect("extractor response snapshot")
            .clone()
            .expect("the accepted filtered turn should fire ModelTurnPrepared");
        assert!(matches!(
            accepted_content.as_slice(),
            [AssistantContent::ToolCall(tool_call)]
                if tool_call.function.name == SUBMIT_TOOL_NAME
                    && tool_call.function.arguments == json!({"name": "John"})
        ));
    }

    #[tokio::test]
    async fn submit_call_wins_over_unexpected_sibling_call() {
        let turn = MockTurn::from_contents([
            tool_call("unknown", "unexpected", json!({})),
            tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
        ])
        .expect("two tool calls")
        .with_usage(usage(7));
        let model = MockCompletionModel::new([turn]);

        let response = extractor(model, 0)
            .extract_with_usage("John")
            .await
            .expect("submit should remain authoritative");

        assert_eq!(response.data.name, "John");
        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn submit_call_wins_before_unexpected_sibling_call() {
        let turn = MockTurn::from_contents([
            tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
            tool_call("unknown", "unexpected", json!({})),
        ])
        .expect("two tool calls");

        let response = extractor(MockCompletionModel::new([turn]), 0)
            .extract("John")
            .await
            .expect("an earlier submit should remain authoritative");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn multiple_unexpected_calls_surrounding_submit_are_ignored() {
        let turn = MockTurn::from_contents([
            tool_call("unknown-before", "unexpected_before", json!({})),
            tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
            tool_call("unknown-after", "unexpected_after", json!({})),
        ])
        .expect("three tool calls");

        let response = extractor(MockCompletionModel::new([turn]), 0)
            .extract("John")
            .await
            .expect("unexpected siblings should not displace submit");

        assert_eq!(response.name, "John");
    }

    #[tokio::test]
    async fn transport_errors_contribute_no_usage() {
        let model = MockCompletionModel::new([
            MockTurn::error("boom"),
            submit_turn("John").with_usage(usage(5)),
        ]);

        let response = extractor(model, 1)
            .extract_with_usage("John")
            .await
            .expect("second attempt should succeed");

        assert_eq!(response.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn single_successful_attempt_reports_its_own_usage() {
        let model = MockCompletionModel::new([submit_turn("John").with_usage(usage(7))]);

        let response = extractor(model, 0)
            .extract_with_usage("John")
            .await
            .expect("extraction should succeed");

        assert_eq!(response.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn exhausted_retries_return_last_error() {
        let model =
            MockCompletionModel::new([MockTurn::text("no submit call").with_usage(usage(10))]);

        let err = extractor(model, 0)
            .extract("John")
            .await
            .expect_err("extraction should fail");

        assert!(matches!(err, ExtractionError::NoData));
    }

    #[tokio::test]
    async fn exhausted_retries_return_error_from_final_attempt() {
        let model = MockCompletionModel::new([MockTurn::error("first"), MockTurn::error("second")]);

        let err = extractor(model, 1)
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
