use super::{Client, ModelTransport, Provider};
use crate::completion::CompletionModel;

/// A provider client with completion capabilities.
///
/// Clients remain `Clone` for conversions between client types; the models
/// they construct no longer need to be.
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel;

    /// Create a completion model with the given model.
    ///
    /// Construction lives here rather than on [`CompletionModel`] so a model
    /// type can be implemented — and used — without any client type at all.
    /// Implement this by calling the model's own inherent constructor.
    ///
    /// # Example with OpenAI
    /// ```ignore
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
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel;
}

/// A [`Provider`] that offers completion models. Implementing this is what
/// makes [`CompletionClient`] available on `Client<Self, H>`.
pub trait HasCompletion: Provider {
    /// The concrete completion model built over transport `H`.
    type Model<H>: CompletionModel
    where
        H: ModelTransport;

    /// Build the completion model `model` from `client`.
    fn completion_model<H>(client: &Client<Self, H>, model: String) -> Self::Model<H>
    where
        H: ModelTransport;
}

impl<P, H> CompletionClient for Client<P, H>
where
    P: HasCompletion,
    H: ModelTransport,
{
    type CompletionModel = P::Model<H>;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        P::completion_model(self, model.into())
    }
}

#[cfg(test)]
mod tests;
