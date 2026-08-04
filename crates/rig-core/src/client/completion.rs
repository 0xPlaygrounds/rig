use crate::completion::CompletionModel;

/// A provider client with completion capabilities.
/// Clone is required for conversions between client types.
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
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel;
}

/// Crate-internal construction hook for the blanket [`CompletionClient`]
/// implementation over [`crate::client::Client`].
///
/// That blanket implementation is generic over whichever model type a provider
/// extension declares, so it needs some way to build that model. Coherence
/// rules out one blanket implementation per provider family — they would all
/// overlap on `Client<Ext, H>` — and the alternative of a public bound such as
/// `From<(Client<Ext, H>, String)>` would push a synthetic conversion into
/// every provider model's public API.
///
/// Keeping the hook crate-private means it constrains only rig's own generic
/// client. Provider crates outside `rig-core` implement [`CompletionClient`]
/// directly and never see this trait.
pub(crate) trait ConstructCompletionModel<C>: Sized {
    /// Build this model from its provider client and a model identifier.
    fn construct(client: &C, model: String) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionError, CompletionRequest, CompletionResponse};
    use crate::streaming::StreamingCompletionResponse;

    /// A model implemented entirely outside rig's provider machinery: no
    /// response associated types, no client associated type, and no
    /// construction hook.
    #[derive(Clone)]
    struct ExternalModel {
        name: String,
    }

    impl CompletionModel for ExternalModel {
        async fn completion(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, CompletionError> {
            Err(CompletionError::ResponseError(format!(
                "{} is a compile-coverage model",
                self.name
            )))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse, CompletionError> {
            Err(CompletionError::ResponseError(format!(
                "{} is a compile-coverage model",
                self.name
            )))
        }
    }

    struct ExternalClient;

    impl CompletionClient for ExternalClient {
        type CompletionModel = ExternalModel;

        fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
            ExternalModel { name: model.into() }
        }
    }

    #[test]
    fn external_model_needs_no_client_or_response_associated_types() {
        let model = ExternalClient.completion_model("external-model");
        assert_eq!(model.name, "external-model");
    }

    #[test]
    fn external_model_is_usable_without_a_client() {
        // A bare model with no client at all still satisfies `CompletionModel`.
        fn assert_completion_model<M: CompletionModel>(_: &M) {}

        assert_completion_model(&ExternalModel {
            name: "standalone".to_owned(),
        });
    }
}
