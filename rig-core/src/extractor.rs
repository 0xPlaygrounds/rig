//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```
//! use rig::providers::openai;
//!
//! // Initialize the OpenAI client
//! let openai = openai::Client::new("your-open-ai-api-key");
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
//! let person = extractor.extract("John Doe is a 30 year old doctor.")
//!     .await
//!     .expect("Failed to extract data from text");
//! ```

use std::marker::PhantomData;

use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    agent::{Agent, AgentBuilder, AgentBuilderSimple},
    completion::{Completion, CompletionError, CompletionModel, ToolDefinition},
    message::{AssistantContent, Message, ToolCall, ToolChoice, ToolFunction},
    tool::Tool,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

const SUBMIT_TOOL_NAME: &str = "submit";

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("No data extracted")]
    NoData,

    #[error("Failed to deserialize the extracted data: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),
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
        let mut last_error = None;
        let text_message = text.into();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let attempt_text = text_message.clone();
            match self.extract_json(attempt_text, vec![]).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}. Retrying...");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
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
        let mut last_error = None;
        let text_message = text.into();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let attempt_text = text_message.clone();
            match self.extract_json(attempt_text, chat_history.clone()).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}. Retrying...");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
    }

    async fn extract_json(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        messages: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        let response = self.agent.completion(text, messages).await?.send().await?;

        if !response.choice.iter().any(|x| {
            let AssistantContent::ToolCall(ToolCall {
                function: ToolFunction { name, .. },
                ..
            }) = x
            else {
                return false;
            };

            name == SUBMIT_TOOL_NAME
        }) {
            tracing::warn!(
                "The submit tool was not called. If this happens more than once, please ensure the model you are using is powerful enough to reliably call tools."
            );
        }

        let arguments = response
            .choice
            .into_iter()
            // We filter tool calls to look for submit tool calls
            .filter_map(|content| {
                if let AssistantContent::ToolCall(ToolCall {
                    function: ToolFunction { arguments, name },
                    ..
                }) = content
                {
                    if name == SUBMIT_TOOL_NAME {
                        Some(arguments)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if arguments.len() > 1 {
            tracing::warn!(
                "Multiple submit calls detected, using the last one. Providers / agents should only ensure one submit call."
            );
        }

        let raw_data = if let Some(arg) = arguments.into_iter().next() {
            arg
        } else {
            return Err(ExtractionError::NoData);
        };

        Ok(serde_json::from_value(raw_data)?)
    }

    pub async fn get_inner(&self) -> &Agent<M> {
        &self.agent
    }

    pub async fn into_inner(self) -> Agent<M> {
        self.agent
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilderSimple<M>,
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
                .tool(SubmitTool::<T> {_t: PhantomData})
                .tool_choice(ToolChoice::Required),
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

    /// Build the Extractor
    pub fn build(self) -> Extractor<M, T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
            retries: self.retries.unwrap_or(0),
        }
    }
}

#[derive(Deserialize, Serialize)]
struct SubmitTool<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
{
    _t: PhantomData<T>,
}

#[derive(Debug, thiserror::Error)]
#[error("SubmitError")]
struct SubmitError;

impl<T> Tool for SubmitTool<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
{
    const NAME: &'static str = SUBMIT_TOOL_NAME;
    type Error = SubmitError;
    type Args = T;
    type Output = T;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit the structured data you extracted from the provided text."
                .to_string(),
            parameters: json!(schema_for!(T)),
        }
    }

    async fn call(&self, data: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(data)
    }
}
