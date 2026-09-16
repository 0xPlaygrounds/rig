use super::*;
use crate::client::transcription::TranscriptionClient;
use crate::providers::gemini::Client;
use crate::providers::gemini::completion::{GEMINI_2_0_FLASH, GEMINI_3_FLASH_PREVIEW};
use crate::test_utils::RecordingHttpClient;
use crate::transcription::TranscriptionModel as _;

fn transcription_request() -> transcription::TranscriptionRequest {
    transcription::TranscriptionRequest {
        data: b"audio bytes".to_vec(),
        filename: "audio.mp3".to_string(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    }
}

#[tokio::test]
async fn transcription_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(GEMINI_2_0_FLASH);

    let error = model
        .transcription(transcription_request())
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, TranscriptionError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// The bytes one transcription puts on the wire, pinned by
/// `tests/cassettes/gemini/transcription/transcription_smoke.yaml`: the path,
/// the `key` query parameter and the whole JSON document, down to the `null`
/// fields Gemini's request shape keeps. The cassette's `inlineData.data` is a
/// 32 KB mp3; this request's audio is [`transcription_request`]'s, so only
/// that one string differs.
#[test]
fn the_wire_encodes_the_recorded_generate_content_request() {
    let wire = Transcriptions::new(
        crate::providers::gemini::Gemini::new("test-key"),
        GEMINI_3_FLASH_PREVIEW,
    );

    let encoded = wire
        .encode(transcription_request(), Mode::Unary)
        .expect("the request encodes");

    let request = encoded.requests.first().expect("exactly one request");
    assert_eq!(request.method().as_str(), "POST");
    assert_eq!(
        request.uri().path(),
        "/v1beta/models/gemini-3-flash-preview:generateContent"
    );
    assert_eq!(request.uri().query(), Some("key=test-key"));
    assert_eq!(
        request
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(encoded.request_id_header, None);

    let crate::wire::Body::Bytes(body) = request.body() else {
        panic!("the transcription body is JSON, not a multipart form")
    };
    assert_eq!(
        serde_json::from_slice::<Value>(body).expect("the body is JSON"),
        serde_json::json!({
            "contents": [{
                "parts": [{
                    "inlineData": {
                        "data": "YXVkaW8gYnl0ZXM=",
                        "mimeType": "audio/mpeg"
                    },
                    "thought": false
                }],
                "role": "user"
            }],
            "generationConfig": {},
            "safetySettings": null,
            "systemInstruction": {
                "parts": [{ "text": TRANSCRIPTION_PREAMBLE, "thought": false }],
                "role": "model"
            },
            "toolConfig": null
        })
    );
}
