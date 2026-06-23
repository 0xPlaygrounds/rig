use crate::http_client::HttpClientExt;
use crate::providers::huggingface::Client;
use crate::providers::huggingface::completion::ApiResponse;
use crate::transcription;
use crate::transcription::TranscriptionError;
use crate::wasm_compat::WasmCompatSync;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
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
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}
impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSync + 'static,
{
    type Response = TranscriptionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        TranscriptionModel::new(client.clone(), model)
    }

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
            .subprovider()
            .transcription_endpoint(&self.model)?;

        let request = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post(&route)?
            .header("Content-Type", "application/json")
            .body(request)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let status = response.status();
        let bytes: Vec<u8> = response.into_body().await?;
        let text = String::from_utf8_lossy(&bytes).into_owned();

        if status.is_success() {
            match serde_json::from_str::<ApiResponse<TranscriptionResponse>>(&text)? {
                ApiResponse::Ok(response) => response.try_into(),
                // Huggingface returns its error envelope with a 2xx status;
                // preserve the raw body alongside that status.
                ApiResponse::Err(err) => {
                    tracing::warn!(message = %err, "provider returned an error response");
                    Err(TranscriptionError::from_http_response(status, text))
                }
            }
        } else {
            Err(TranscriptionError::from_http_response(status, text))
        }
    }
}
