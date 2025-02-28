use crate::client::Client;

use rig::{
    completion::{self, CompletionError, CompletionModel},
    json_utils::{delete_inplace, merge_inplace},
};

// All supported models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
/// `amazon.nova-lite-v1` foundational model
pub const AMAZON_NOVA_LITE_V1: &str = "amazon.nova-lite-v1:0";
/// `mistral.mixtral-8x7b-instruct-v0` foundational model
pub const MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0: &str = "mistral.mixtral-8x7b-instruct-v0:1";

#[derive(Clone)]
pub struct BedrockProvider<T: CompletionModel + Clone> {
    completion_model: T,
    client: Client,
    pub model: String,
}

impl<T: CompletionModel + Clone> BedrockProvider<T> {
    pub fn new(completion_model: T, client: Client, model: &str) -> Self {
        Self {
            completion_model,
            client,
            model: model.to_string(),
        }
    }
}

impl<T: CompletionModel + Clone> completion::CompletionModel for BedrockProvider<T>
where
    <T as rig::completion::CompletionModel>::Response: serde::de::DeserializeOwned,
    <T as rig::completion::CompletionModel>::Response:
        TryInto<completion::CompletionResponse<T::Response>>,
    <<T as rig::completion::CompletionModel>::Response as TryInto<
        completion::CompletionResponse<T::Response>,
    >>::Error: std::error::Error + Send + Sync + 'static,
{
    type Response = T::Response;

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let mut request = self.build_completion(completion_request).await?;
        for field in self.client.additional_fields.iter() {
            merge_inplace(&mut request, field.clone());
        }
        for field in self.client.deletable_fields.iter() {
            delete_inplace(&mut request, field);
        }

        let request = serde_json::to_string(&request).unwrap();
        let bytes = request.into_bytes();
        let response = self
            .client
            .aws_client
            .invoke_model()
            .accept("application/json")
            .content_type("application/json")
            .model_id(self.model.as_str())
            .body(bytes.into())
            .send()
            .await
            .unwrap();
        let body = response.body().as_ref();
        let response_body: T::Response =
            serde_json::from_slice(body).map_err(|e| CompletionError::RequestError(e.into()))?;

        // Create a CompletionResponse with the deserialized response body
        response_body
            .try_into()
            .map_err(|e| CompletionError::RequestError(Box::new(e)))
    }

    async fn build_completion(
        &self,
        request: completion::CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        self.completion_model.build_completion(request).await
    }
}
