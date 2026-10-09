//! MiniMax voice cloning API integration.

use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm};
use crate::provider_response;
use crate::providers::minimax::Client;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

const VOICE_CLONE_PURPOSE: &str = "voice_clone";

/// Errors returned by MiniMax voice cloning operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VoiceCloneError {
    /// HTTP transport or status error.
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// JSON serialization or deserialization error.
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// The upload filename does not use a supported audio extension.
    #[error("Unsupported voice clone audio extension: {0}")]
    UnsupportedAudioFormat(String),

    /// A successful upload response did not contain file metadata.
    #[error("MiniMax returned no file metadata for a successful voice clone audio upload")]
    MissingUploadedFile,

    /// Raw error response preserved from MiniMax.
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(VoiceCloneError);

/// MiniMax response status details.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct BaseResponse {
    /// Provider status code; zero indicates success.
    pub status_code: i64,
    /// Provider status description.
    #[serde(default)]
    pub status_msg: String,
}

/// Metadata returned after uploading voice clone audio.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct VoiceCloneAudioFile {
    /// Provider-assigned file identifier.
    pub file_id: u64,
}

/// Successful voice clone audio upload response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceCloneAudioUploadResponse {
    /// Uploaded file metadata.
    pub file: VoiceCloneAudioFile,
    /// Provider status details.
    pub base_resp: BaseResponse,
}

/// Required parameters for creating a cloned voice.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct VoiceCloneRequest {
    /// Identifier returned by [`Client::upload_voice_clone_audio`].
    pub file_id: u64,
    /// Identifier to assign to the cloned voice.
    pub voice_id: String,
    /// Speech model used by the clone operation.
    pub model: String,
}

impl VoiceCloneRequest {
    /// Creates a voice clone request from its required fields.
    pub fn new(file_id: u64, voice_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            file_id,
            voice_id: voice_id.into(),
            model: model.into(),
        }
    }
}

/// Successful voice clone response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceCloneResponse {
    /// Identifier assigned by the request to the cloned voice.
    pub voice_id: String,
    /// Provider status details.
    pub base_resp: BaseResponse,
}

#[derive(Deserialize)]
struct VoiceCloneAudioUploadEnvelope {
    #[serde(default)]
    file: Option<VoiceCloneAudioFile>,
    base_resp: BaseResponse,
}

#[derive(Deserialize)]
struct VoiceCloneEnvelope {
    base_resp: BaseResponse,
}

impl<H> Client<H>
where
    H: HttpClientExt + 'static,
{
    /// Uploads an mp3, m4a, or wav file for voice cloning.
    pub async fn upload_voice_clone_audio(
        &self,
        filename: impl Into<String>,
        audio: impl Into<Bytes>,
    ) -> Result<VoiceCloneAudioUploadResponse, VoiceCloneError> {
        let filename = filename.into();
        let content_type = voice_clone_content_type(&filename)?;
        let form = MultipartForm::new()
            .text("purpose", VOICE_CLONE_PURPOSE)
            .part(
                Part::bytes("file", audio)
                    .filename(filename)
                    .content_type(content_type),
            );
        let request = self
            .post("/files/upload")?
            .body(form)
            .map_err(http_client::Error::from)?;
        let response = self.send_multipart::<Bytes>(request).await?;
        let status = response.status();
        let response_body = response.into_body().await?;
        let response_text = String::from_utf8_lossy(&response_body).into_owned();

        if !status.is_success() {
            return Err(VoiceCloneError::from_http_response(status, response_text));
        }

        let response: VoiceCloneAudioUploadEnvelope = serde_json::from_slice(&response_body)?;
        if response.base_resp.status_code != 0 {
            return Err(VoiceCloneError::from_http_response(status, response_text));
        }

        let file = response.file.ok_or(VoiceCloneError::MissingUploadedFile)?;
        Ok(VoiceCloneAudioUploadResponse {
            file,
            base_resp: response.base_resp,
        })
    }

    /// Creates a cloned voice from previously uploaded audio.
    pub async fn clone_voice(
        &self,
        request: VoiceCloneRequest,
    ) -> Result<VoiceCloneResponse, VoiceCloneError> {
        let voice_id = request.voice_id.clone();
        let request_body = serde_json::to_vec(&request)?;
        let request = self
            .post("/voice_clone")?
            .body(request_body)
            .map_err(http_client::Error::from)?;
        let response = self.send::<_, Bytes>(request).await?;
        let status = response.status();
        let response_body = response.into_body().await?;
        let response_text = String::from_utf8_lossy(&response_body).into_owned();

        if !status.is_success() {
            return Err(VoiceCloneError::from_http_response(status, response_text));
        }

        let response: VoiceCloneEnvelope = serde_json::from_slice(&response_body)?;
        if response.base_resp.status_code != 0 {
            return Err(VoiceCloneError::from_http_response(status, response_text));
        }

        Ok(VoiceCloneResponse {
            voice_id,
            base_resp: response.base_resp,
        })
    }
}

fn voice_clone_content_type(filename: &str) -> Result<mime::Mime, VoiceCloneError> {
    let extension = Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("mp3" | "m4a" | "wav") => Ok(mime_guess::from_path(filename).first_or_octet_stream()),
        _ => Err(VoiceCloneError::UnsupportedAudioFormat(
            extension.unwrap_or_default(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    /// Unit coverage verifies multipart construction without recording a user's voice sample.
    #[tokio::test]
    async fn upload_voice_clone_audio_maps_multipart_request_and_response() {
        let body = r#"{
            "file": {
                "file_id": 123456789,
                "bytes": 4,
                "created_at": 1700469398,
                "filename": "voice.mp3",
                "purpose": "voice_clone"
            },
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        let response = client
            .upload_voice_clone_audio("voice.mp3", Bytes::from_static(b"audio"))
            .await
            .expect("upload should succeed");

        assert_eq!(response.file.file_id, 123456789);
        assert_eq!(response.base_resp.status_code, 0);

        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert!(request.uri.ends_with("/v1/files/upload"));
        let multipart = String::from_utf8_lossy(&request.body);
        assert!(multipart.contains("name=\"purpose\""));
        assert!(multipart.contains(VOICE_CLONE_PURPOSE));
        assert!(multipart.contains("name=\"file\"; filename=\"voice.mp3\""));
        assert!(multipart.contains("Content-Type: audio/mpeg"));
        assert!(multipart.contains("audio"));
    }

    /// Unit coverage verifies the documented request boundary without calling the paid API.
    #[tokio::test]
    async fn clone_voice_maps_required_fields_and_returns_voice_id() {
        let body = r#"{
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        let response = client
            .clone_voice(VoiceCloneRequest::new(
                123456789,
                "OctopusVoice",
                "speech-2.8-hd",
            ))
            .await
            .expect("voice clone should succeed");

        assert_eq!(response.voice_id, "OctopusVoice");
        assert_eq!(response.base_resp.status_code, 0);

        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert!(request.uri.ends_with("/v1/voice_clone"));
        let request_body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request should be JSON");
        assert_eq!(request_body["file_id"], 123456789);
        assert_eq!(request_body["voice_id"], "OctopusVoice");
        assert_eq!(request_body["model"], "speech-2.8-hd");
    }

    /// Unit coverage verifies provider envelopes without relying on live error responses.
    #[tokio::test]
    async fn clone_voice_preserves_provider_error_envelope() {
        let body = r#"{
            "base_resp": {"status_code": 2013, "status_msg": "invalid input"}
        }"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");

        let error = client
            .clone_voice(VoiceCloneRequest::new(
                123456789,
                "OctopusVoice",
                "speech-2.8-hd",
            ))
            .await
            .expect_err("provider error should be returned");

        assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[test]
    fn upload_rejects_unsupported_audio_extension() {
        let error = voice_clone_content_type("voice.ogg")
            .expect_err("unsupported extension should be rejected");

        assert!(matches!(
            error,
            VoiceCloneError::UnsupportedAudioFormat(extension) if extension == "ogg"
        ));
    }
}
