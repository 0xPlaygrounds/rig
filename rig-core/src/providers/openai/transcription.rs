use crate::providers::openai::{ApiResponse, Client};
use crate::transcription;
use crate::transcription::TranscriptionError;
use reqwest::multipart::Part;
use serde::Deserialize;

// ================================================================
// OpenAI Transcription API
// ================================================================
pub const WHISPER_1: &str = "whisper-1";

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
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("language", request.language)
            .part(
                "file",
                Part::bytes(data).file_name(request.filename.clone()),
            );

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            for (key, value) in additional_params
                .as_object()
                .expect("Additional Parameters to OpenAI Transcription should be a map")
            {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let response = self
            .client
            .post("audio/transcriptions")
            .multipart(body)
            .send()
            .await?;

        if response.status().is_success() {
            match response
                .json::<ApiResponse<TranscriptionResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(response.text().await?))
        }
    }
}
