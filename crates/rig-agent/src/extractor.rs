//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai;
//! use rig_reqwest::prelude::*;
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
use serde::{Serialize, de::DeserializeOwned};

use rig_core::{
    message::{Message, ToolChoice},
    vector_store::VectorStoreIndexDyn,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    agent::{Agent, AgentBuilder, AgentHook, ModelHandle, OutputMode},
    completion::{CompletionError, CompletionModel, PromptError, Usage},
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
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    agent: Agent,
    _t: PhantomData<T>,
    retries: u64,
}

/// A single extraction run with an overridden default model.
///
/// The model is the default candidate for every retry in this run;
/// model-selection hooks may replace it before each attempt. The originating
/// [`Extractor`]'s default model is unchanged.
#[must_use = "an extraction override does nothing until an extract method is awaited"]
pub struct ExtractorRun<'a, T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    extractor: &'a Extractor<T>,
    model: Option<ModelHandle>,
}

/// Generate the `Extractor` methods that delegate to the same-named
/// [`ExtractorRun`] method on [`Extractor::default_run`], appending the
/// retry-semantics doc paragraphs shared by all of them. Methods marked
/// `=> usage` also carry the shared usage-accounting paragraph.
macro_rules! forward_default_run {
    (@usage_doc usage) => {
        "\nUsage accumulates across all retry attempts, including attempts that received\n\
         a billed response but failed extraction (e.g. the model never called `submit`).\n\
         Attempts whose completion call itself returned an error (e.g. network failures\n\
         or unparseable provider responses) contribute no usage, and when every attempt\n\
         fails the returned error carries no usage information at all."
    };
    ($( $(#[$attr:meta])* $name:ident ( $($arg:ident : $ty:ty),* ) -> $ret:ty
        $(=> $usage:ident)?; )+) => {$(
        $(#[$attr])*
        ///
        /// The function will retry the extraction if the initial attempt fails or
        /// if the model does not call the `submit` tool.
        ///
        /// The number of retries is determined by the `retries` field on the Extractor struct.
        $(#[doc = forward_default_run!(@usage_doc $usage)])?
        pub async fn $name(
            &self,
            text: impl Into<Message> + WasmCompatSend,
            $($arg: $ty),*
        ) -> Result<$ret, ExtractionError> {
            self.default_run().$name(text $(, $arg)*).await
        }
    )+};
}

impl<T> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    /// Set a different default model for this extractor's subsequent attempts.
    pub fn with_model_handle(mut self, model: ModelHandle) -> Self {
        self.agent.set_model_handle(model);
        self
    }

    /// Set one extraction run's default model without changing this extractor.
    ///
    /// The handle is the default candidate for every retry of the run — not a
    /// hard pin: model-selection hooks may replace it before each attempt.
    pub fn using_model(&self, model: ModelHandle) -> ExtractorRun<'_, T> {
        ExtractorRun {
            extractor: self,
            model: Some(model),
        }
    }

    /// A run with no model override: the extractor's own default model.
    fn default_run(&self) -> ExtractorRun<'_, T> {
        ExtractorRun {
            extractor: self,
            model: None,
        }
    }

    /// Erase and set a typed default model for one extraction run.
    pub fn using_model_value<M>(&self, model: M) -> ExtractorRun<'_, T>
    where
        M: CompletionModel + 'static,
    {
        self.using_model(ModelHandle::new(model))
    }

    forward_default_run! {
        /// Attempts to extract data from the given text with a number of retries.
        extract() -> T;

        /// Attempts to extract data from the given text with a number of retries.
        extract_with_chat_history(chat_history: Vec<Message>) -> T;

        /// Attempts to extract data from the given text with a number of retries,
        /// returning both the extracted data and accumulated token usage.
        extract_with_usage() -> ExtractionResponse<T> => usage;

        /// Attempts to extract data from the given text with a number of retries,
        /// providing chat history context, and returning both the extracted data
        /// and accumulated token usage.
        extract_with_chat_history_with_usage(chat_history: Vec<Message>) -> ExtractionResponse<T>
            => usage;
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
        model: Option<&ModelHandle>,
    ) -> Result<(T, Usage), ExtractionError> {
        let mut last_error = None;
        let mut usage = Usage::new();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let (result, attempt_usage) = self
                .extract_json_with_usage(&text, &chat_history, model)
                .await;
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
        model: Option<&ModelHandle>,
    ) -> (Result<T, ExtractionError>, Usage) {
        let mut runner = self
            .agent
            .runner(text.clone())
            .history(messages.iter().cloned());
        // A run-local model is the default candidate for THIS attempt only;
        // model-selection hooks may still replace it per retry.
        if let Some(model) = model {
            runner = runner.using_model(model.clone());
        }
        let (result, error_usage) = runner
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

impl<T> ExtractorRun<'_, T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    /// Extract structured data with the run-local model.
    pub async fn extract(
        &self,
        text: impl Into<Message> + WasmCompatSend,
    ) -> Result<T, ExtractionError> {
        Ok(self.extract_with_usage(text).await?.data)
    }

    /// Extract structured data with chat history and the run-local model.
    pub async fn extract_with_chat_history(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        Ok(self
            .extract_with_chat_history_with_usage(text, chat_history)
            .await?
            .data)
    }

    /// Extract structured data and usage with the run-local model.
    pub async fn extract_with_usage(
        &self,
        text: impl Into<Message> + WasmCompatSend,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        self.extract_with_chat_history_with_usage(text, vec![])
            .await
    }

    /// Extract structured data with chat history and usage using the run-local model.
    pub async fn extract_with_chat_history_with_usage(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<ExtractionResponse<T>, ExtractionError> {
        let (data, usage) = self
            .extractor
            .retry_extract(text.into(), chat_history, self.model.as_ref())
            .await?;
        Ok(ExtractionResponse { data, usage })
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilder,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

/// Generate the `ExtractorBuilder` setters that forward verbatim to the inner
/// [`AgentBuilder`] method of the same name. Doc comments live at each
/// invocation; `preamble` (which wraps its argument) and `retries`
/// (builder-local) stay hand-written.
macro_rules! forward_agent_builder {
    ($( $(#[$attr:meta])* $name:ident $([$gen:ident : $($bound:tt)+])?
        ( $($arg:ident : $ty:ty),* );)+) => {$(
        $(#[$attr])*
        pub fn $name $(<$gen>)? (mut self, $($arg: $ty),*) -> Self
        $(where $gen: $($bound)+)?
        {
            self.agent_builder = self.agent_builder.$name($($arg),*);
            self
        }
    )+};
}

impl<T> ExtractorBuilder<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_model_handle(ModelHandle::new(model))
    }

    /// Create an extractor builder from an opaque runtime model handle.
    pub fn from_model_handle(model: ModelHandle) -> Self {
        Self {
            agent_builder: AgentBuilder::from_model_handle(model)
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

    forward_agent_builder! {
        /// Add a context document to the extractor
        context(doc: &str);

        /// Add dynamic context retrieved from a vector store on every extraction attempt.
        ///
        /// This delegates to [`AgentBuilder::dynamic_context`] and therefore uses the
        /// same completion-call hook lifecycle as an agent.
        dynamic_context[I: VectorStoreIndexDyn + 'static](samples: usize, index: I);

        additional_params(params: serde_json::Value);

        /// Set the maximum number of tokens for the completion
        max_tokens(max_tokens: u64);

        /// Set the `tool_choice` option for the inner Agent.
        tool_choice(choice: ToolChoice);

        /// Add a provider-independent lifecycle hook to every extraction attempt.
        ///
        /// Completion-response hooks receive canonical Rig content, usage, prompt,
        /// and message ID fields, just like hooks attached directly to an agent.
        add_hook[H: AgentHook + 'static](hook: H);
    }

    /// Set the maximum number of retries for the extractor.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = Some(retries);
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
mod tests;
