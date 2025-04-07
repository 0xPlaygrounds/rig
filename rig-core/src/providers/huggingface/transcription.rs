use crate::providers::huggingface::completion::ApiResponse;
use crate::providers::huggingface::Client;
use crate::transcription;
use crate::transcription::TranscriptionError;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;

pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
pub const WHISPER_SMALL: &str = "openai/whisper-small";

#[derive(Debug, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

impl TryFrom<TranscriptionResponse>
    for transcription::TranscriptionResponse<TranscriptionResponse>
{
    type Error = TranscriptionError;

    fn try_from(value: TranscriptionResponse) -> Result<Self, Self::Error> {
        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct TranscriptionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}
impl transcription::TranscriptionModel for TranscriptionModel {
    type Response = TranscriptionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
        let data = request.data;
        let data = BASE64_STANDARD.encode(data);

        let request = json!({
            "inputs": data
        });

        let route = self
            .client
            .sub_provider
            .transcription_endpoint(&self.model)?;
        let response = self.client.post(&route).json(&request).send().await?;

        if response.status().is_success() {
            match response
                .json::<ApiResponse<TranscriptionResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(err) => Err(TranscriptionError::ProviderError(err.to_string())),
            }
        } else {
            Err(TranscriptionError::ProviderError(response.text().await?))
        }
    }
}
