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
