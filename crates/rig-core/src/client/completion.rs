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

/// Construction hook for the blanket [`CompletionClient`] implementation over
/// [`crate::client::Client`].
///
/// That blanket implementation is generic over whichever model type a provider
/// extension declares, so it needs some way to build that model. Coherence
/// rules out one blanket implementation per provider family — they would all
/// overlap on `Client<Ext, H>` — and the alternative of a public bound such as
/// `From<(Client<Ext, H>, String)>` would push a synthetic conversion into
/// every provider model's public API.
///
/// This trait is public because it is the extension point for out-of-tree
/// provider extensions built on the generic [`crate::client::Client`]: such a
/// crate cannot implement [`CompletionClient`] for rig's foreign
/// `Client<Ext, H>` type directly (orphan rule), so it implements this trait
/// on its own model type instead, and the blanket implementation supplies
/// `completion_model` for it. Providers with their own client type simply
/// implement [`CompletionClient`] directly and never need this trait.
pub trait ConstructCompletionModel<C>: Sized {
    /// Build this model from its provider client and a model identifier.
    fn construct(client: &C, model: String) -> Self;
}

#[cfg(test)]
mod tests;
