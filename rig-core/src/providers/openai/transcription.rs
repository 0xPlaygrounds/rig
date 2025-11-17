use bytes::Bytes;

use crate::http_client::HttpClientExt;
use crate::providers::openai::{Client, client::ApiResponse};
use crate::transcription::TranscriptionError;
use crate::{models, transcription};
use reqwest::multipart::Part;
use serde::Deserialize;

// ================================================================
// OpenAI Transcription API
// ================================================================

models! {
    pub enum TranscriptionModels {
        Whisper1 => "whisper-1",
    }
}
pub use TranscriptionModels::*;

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
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: TranscriptionModels,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: TranscriptionModels) -> Self {
        Self { client, model }
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    type Response = TranscriptionResponse;

    type Client = Client<T>;
    type Models = TranscriptionModels;

    fn make(client: &Self::Client, model: Self::Models) -> Self {
        Self::new(client.clone(), model)
    }

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
            .text(
                "model",
                <TranscriptionModels as Into<&str>>::into(self.model),
            )
            .part(
                "file",
                Part::bytes(data).file_name(request.filename.clone()),
            );

        if let Some(language) = request.language {
            body = body.text("language", language);
        }

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

        let req = self
            .client
            .post("/audio/transcriptions")?
            .body(body)
            .unwrap();

        let response = self
            .client
            .http_client()
            .send_multipart::<Bytes>(req)
            .await
            .unwrap();

        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();
        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            let str = String::from_utf8_lossy(&response_body).to_string();
            Err(TranscriptionError::ProviderError(str))
        }
    }
}
