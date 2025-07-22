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
    agent::{Agent, AgentBuilder},
    completion::{Completion, CompletionError, CompletionModel, ToolDefinition},
    message::{AssistantContent, Message, ToolCall, ToolFunction},
    tool::Tool,
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
pub struct Extractor<M: CompletionModel, T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync> {
    agent: Agent<M>,
    _t: PhantomData<T>,
}

impl<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync, M: CompletionModel> Extractor<M, T>
where
    M: Sync,
{
    pub async fn extract(&self, text: impl Into<Message> + Send) -> Result<T, ExtractionError> {
        let response = self.agent.completion(text, vec![]).await?.send().await?;

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
pub struct ExtractorBuilder<
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
    M: CompletionModel,
> {
    agent_builder: AgentBuilder<M>,
    _t: PhantomData<T>,
}

impl<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync, M: CompletionModel>
    ExtractorBuilder<T, M>
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
                .tool(SubmitTool::<T> {_t: PhantomData}),

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

    /// Build the Extractor
    pub fn build(self) -> Extractor<M, T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct SubmitTool<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync> {
    _t: PhantomData<T>,
}

#[derive(Debug, thiserror::Error)]
#[error("SubmitError")]
struct SubmitError;

impl<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync> Tool for SubmitTool<T> {
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
