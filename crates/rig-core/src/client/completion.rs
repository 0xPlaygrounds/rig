use crate::completion::CompletionModel;

/// A provider client with completion capabilities.
/// Clone is required for conversions between client types.
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel<Client = Self>;

    /// Create a completion model with the given model.
    ///
    /// # Example with OpenAI
    /// ```no_run
    /// use rig_core::prelude::*;
    /// use rig_core::providers::openai::{Client, self};
    ///
    /// # fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key")?;
    ///
    /// let gpt = openai.completion_model(openai::GPT_5_2);
    /// # Ok(())
    /// # }
    /// ```
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        Self::CompletionModel::make(self, model)
    }
}
