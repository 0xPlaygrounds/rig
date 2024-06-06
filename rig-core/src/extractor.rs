use std::marker::PhantomData;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    agent::{Agent, AgentBuilder},
    completion::{CompletionModel, Prompt, PromptError, ToolDefinition},
    tool::Tool,
};

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("No data extracted")]
    NoData,

    #[error("Failed to deserialize the extracted data: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("PromptError: {0}")]
    PromptError(#[from] PromptError),
}

pub struct Extractor<M: CompletionModel, T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync> {
    pub agent: Agent<M>,
    _t: PhantomData<T>,
}

impl<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync, M: CompletionModel> Extractor<M, T>
where
    M: Sync,
{
    pub async fn extract(&self, text: &str) -> Result<T, ExtractionError> {
        let summary = self.agent.prompt(text).await?;

        if summary.is_empty() {
            return Err(ExtractionError::NoData);
        }

        Ok(serde_json::from_str(&summary)?)
    }
}

pub struct ExtractorBuilder<
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
    M: CompletionModel,
> {
    agent_builder: AgentBuilder<M>,
    _t: PhantomData<T>,
}

impl<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync, M: CompletionModel>
    ExtractorBuilder<T, M>
{
    pub fn new(model: M) -> Self {
        Self {
            agent_builder: AgentBuilder::new(model)
                .preamble("\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, event with default values!!!.
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
pub struct SubmitError;

impl<T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync> Tool for SubmitTool<T> {
    const NAME: &'static str = "submit";
    type Error = SubmitError;
    type Args = String;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit the structured data you extracted from the provided text."
                .to_string(),
            parameters: json!(schema_for!(T)),
        }
    }

    async fn call(&self, data: Self::Args) -> Result<String, Self::Error> {
        Ok(data)
    }
}
