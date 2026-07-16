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

use std::{
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::{
        Agent, AgentBuilder, AgentHook, CompletionResponseEvent, HookContext, ObservationAction,
        OutputMode,
    },
    completion::{CompletionError, CompletionModel, PromptError, Usage},
    message::{AssistantContent, Message, ToolChoice},
    vector_store::VectorStoreIndexDyn,
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
        let submissions = Arc::new(AtomicUsize::new(0));
        let response = match self
            .agent
            .runner(text.clone())
            .history(messages.iter().cloned())
            .max_turns(1)
            .output_tool(
                SUBMIT_TOOL_NAME,
                "Submit the structured data you extracted from the provided text.",
                false,
            )
            .add_hook(SubmissionObserver(submissions.clone()))
            .run()
            .await
        {
            Ok(response) => response,
            Err(PromptError::CompletionError(e)) => {
                return (Err(ExtractionError::CompletionError(e)), Usage::new());
            }
            Err(e) => return (Err(e.into()), Usage::new()),
        };
        let usage = response.usage;

        let submissions = submissions.load(Ordering::Acquire);
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

    /// Add a lifecycle hook to every extraction attempt.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook<M> + 'static,
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

    /// Add dynamic context (RAG) to the extractor.
    ///
    /// On each prompt, `sample` documents will be retrieved from the index based on the RAG text
    /// and inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + Send + Sync + 'static,
    ) -> Self {
        self.agent_builder = self.agent_builder.dynamic_context(sample, dynamic_context);
        self
    }
}

struct SubmissionObserver(Arc<AtomicUsize>);

impl<M> AgentHook<M> for SubmissionObserver
where
    M: CompletionModel,
{
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: CompletionResponseEvent<'_, M>,
    ) -> ObservationAction {
        let submissions = event
            .response
            .choice
            .iter()
            .filter(|content| {
                matches!(
                    content,
                    AssistantContent::ToolCall(call) if call.function.name == SUBMIT_TOOL_NAME
                )
            })
            .count();
        self.0.fetch_add(submissions, Ordering::Release);
        ObservationAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use serde_json::json;

    use super::*;
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

    #[derive(Clone, Default)]
    struct LifecycleCounts {
        completion_calls: Arc<AtomicUsize>,
        completion_responses: Arc<AtomicUsize>,
        model_turns: Arc<AtomicUsize>,
    }

    impl<M> AgentHook<M> for LifecycleCounts
    where
        M: CompletionModel,
    {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> crate::agent::CompletionCallAction {
            self.completion_calls.fetch_add(1, Ordering::SeqCst);
            crate::agent::CompletionCallAction::Continue
        }

        async fn on_completion_response(
            &self,
            _ctx: &HookContext,
            _event: CompletionResponseEvent<'_, M>,
        ) -> ObservationAction {
            self.completion_responses.fetch_add(1, Ordering::SeqCst);
            ObservationAction::Continue
        }

        async fn on_model_turn_finished(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::ModelTurnFinished<'_>,
        ) -> ObservationAction {
            self.model_turns.fetch_add(1, Ordering::SeqCst);
            ObservationAction::Continue
        }
    }

    struct StopBeforeCompletion;

    impl<M> AgentHook<M> for StopBeforeCompletion
    where
        M: CompletionModel,
    {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> crate::agent::CompletionCallAction {
            crate::agent::CompletionCallAction::stop("extractor stopped")
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
        assert_eq!(counts.completion_responses.load(Ordering::SeqCst), 1);
        assert_eq!(counts.model_turns.load(Ordering::SeqCst), 1);
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
