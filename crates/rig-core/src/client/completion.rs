use crate::completion::CompletionModel;

/// A provider client with completion capabilities.
/// Clone is required for conversions between client types.
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel;

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
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionError, CompletionRequest, CompletionResponse};
    use crate::streaming::StreamingCompletionResponse;

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
                "{} is a compile-only model",
                self.name
            )))
        }

        async fn stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse, CompletionError> {
            Err(CompletionError::ResponseError(format!(
                "{} is a compile-only model",
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
    fn external_model_and_client_opt_in_without_response_or_client_associated_types() {
        let model = ExternalClient.completion_model("external-model");
        assert_eq!(model.name, "external-model");
    }
}
