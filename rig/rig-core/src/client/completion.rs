use crate::agent::AgentBuilder;
use crate::completion::CompletionModel;
use crate::extractor::ExtractorBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A provider client with completion capabilities.
/// Clone is required for conversions between client types.
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel<Client = Self>;

    /// Create a completion model with the given model.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.completion_model(openai::GPT4);
    /// ```
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        Self::CompletionModel::make(self, model)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let agent = openai.agent(openai::GPT_4)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}
