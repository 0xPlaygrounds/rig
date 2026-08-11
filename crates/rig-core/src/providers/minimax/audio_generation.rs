//! MiniMax text-to-speech API integration.

use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest,
    AudioGenerationResponse as RigAudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::providers::minimax::Client;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;

/// The `speech-2.8-hd` MiniMax speech model.
pub const SPEECH_2_8_HD: &str = "speech-2.8-hd";
/// The `speech-2.8-turbo` MiniMax speech model.
pub const SPEECH_2_8_TURBO: &str = "speech-2.8-turbo";

/// MiniMax text-to-speech response data.
#[derive(Clone, Debug, Deserialize)]
pub struct AudioGenerationData {
    /// Hex-encoded generated audio.
    pub audio: String,
    /// Provider generation status.
    pub status: i64,
}

/// MiniMax response status details.
#[derive(Clone, Debug, Deserialize)]
pub struct BaseResponse {
    /// Provider status code; zero indicates success.
    pub status_code: i64,
    /// Provider status description.
    #[serde(default)]
    pub status_msg: String,
}

/// MiniMax text-to-speech response envelope.
#[derive(Clone, Debug, Deserialize)]
pub struct AudioGenerationResponse {
    /// Generated audio payload, omitted in provider error envelopes.
    #[serde(default)]
    pub data: Option<AudioGenerationData>,
    /// Provider status details.
    pub base_resp: BaseResponse,
}

/// MiniMax text-to-speech model.
#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the speech model.
    pub model: String,
}

impl<T> AudioGenerationModel<T> {
    /// Creates a MiniMax text-to-speech model from a client and model name.
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
where
    T: HttpClientExt
        + Clone
        + Default
        + std::fmt::Debug
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = AudioGenerationResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<RigAudioGenerationResponse<Self::Response>, AudioGenerationError> {
        let mut body = json!({
            "model": self.model,
            "text": request.text,
            "stream": false,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": request.voice,
                "speed": request.speed,
            },
            "audio_setting": {
                "format": "mp3",
            },
        });

        if let Some(additional_params) = request.additional_params {
            merge_inplace(&mut body, additional_params);
        }

        let body = serde_json::to_vec(&body)?;
        let req = self
            .client
            .post("/t2a_v2")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send::<_, Bytes>(req).await?;
        let status = response.status();
        let response_body = response.into_body().await?;
        let response_text = String::from_utf8_lossy(&response_body);

        if !status.is_success() {
            return Err(AudioGenerationError::from_http_response(
                status,
                response_text,
            ));
        }

        let provider_response: AudioGenerationResponse = serde_json::from_slice(&response_body)?;
        if provider_response.base_resp.status_code != 0 {
            return Err(AudioGenerationError::from_http_response(
                status,
                response_text,
            ));
        }

        let data = provider_response.data.as_ref().ok_or_else(|| {
            AudioGenerationError::ResponseError(
                "MiniMax returned no audio data for a successful request".to_string(),
            )
        })?;
        let audio = decode_hex(&data.audio)?;

        Ok(RigAudioGenerationResponse {
            audio,
            response: provider_response,
        })
    }
}

fn decode_hex(value: &str) -> Result<Vec<u8>, AudioGenerationError> {
    let mut chars = value.chars();
    let mut audio = Vec::with_capacity(value.len() / 2);

    loop {
        let Some(high) = chars.next() else {
            break;
        };
        let Some(low) = chars.next() else {
            return Err(AudioGenerationError::ResponseError(
                "MiniMax returned an odd-length hexadecimal audio payload".to_string(),
            ));
        };

        let high = high.to_digit(16).ok_or_else(|| {
            AudioGenerationError::ResponseError(
                "MiniMax returned invalid hexadecimal audio".to_string(),
            )
        })?;
        let low = low.to_digit(16).ok_or_else(|| {
            AudioGenerationError::ResponseError(
                "MiniMax returned invalid hexadecimal audio".to_string(),
            )
        })?;
        audio.push(((high << 4) | low) as u8);
    }

    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_success_decodes_hex_and_maps_request() {
        let body = r#"{
            "data": {"audio": "000102ff", "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.audio_generation_model(SPEECH_2_8_HD);

        let response = model
            .audio_generation_request()
            .text("hello")
            .voice("English_expressive_narrator")
            .send()
            .await
            .expect("audio generation should succeed");

        assert_eq!(response.audio, vec![0, 1, 2, 255]);

        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert!(request.uri.ends_with("/v1/t2a_v2"));
        let request_body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request should be JSON");
        assert_eq!(request_body["model"], SPEECH_2_8_HD);
        assert_eq!(request_body["text"], "hello");
        assert_eq!(request_body["stream"], false);
        assert_eq!(request_body["output_format"], "hex");
        assert_eq!(
            request_body["voice_setting"]["voice_id"],
            "English_expressive_narrator"
        );
        assert_eq!(request_body["voice_setting"]["speed"], 1.0);
        assert_eq!(request_body["audio_setting"]["format"], "mp3");
    }

    #[tokio::test]
    async fn audio_generation_provider_error_preserves_status_and_body() {
        let body = r#"{
            "base_resp": {"status_code": 2013, "status_msg": "invalid input"}
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(SPEECH_2_8_TURBO);

        let error = model
            .audio_generation(
                model
                    .audio_generation_request()
                    .text("hello")
                    .voice("English_expressive_narrator")
                    .build(),
            )
            .await
            .err()
            .expect("provider error should be returned");

        match error {
            AudioGenerationError::ProviderResponse(response) => {
                assert_eq!(response.status, Some(http::StatusCode::OK));
                assert_eq!(response.body, body);
            }
            other => panic!("expected provider response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn audio_generation_http_error_preserves_status_and_body() {
        let body = r#"{"error":"temporarily unavailable"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(SPEECH_2_8_HD);

        let error = model
            .audio_generation(
                model
                    .audio_generation_request()
                    .text("hello")
                    .voice("English_expressive_narrator")
                    .build(),
            )
            .await
            .err()
            .expect("HTTP error should be returned");

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
