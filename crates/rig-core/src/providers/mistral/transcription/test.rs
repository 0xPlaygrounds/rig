use super::*;
use crate::providers::mistral::Client;

#[test]
fn test_mistral_transcription_response_deserialize() {
    let json = r#" {
          "model": "voxtral-mini-latest",
          "text": "The sun was setting slowly, casting long shadows across the empty field.",
          "language": null,
          "segments": [
            {
              "text": "The sun was setting slowly, casting long shadows across the empty field.",
              "start": 0.2,
              "end": 4.6,
              "speaker_id": "speaker_1",
              "type": "transcription_segment"
            }
          ],
          "usage": {
            "prompt_audio_seconds": 5,
            "prompt_tokens": 5,
            "total_tokens": 404,
            "completion_tokens": 24,
            "prompt_tokens_details": {
              "cached_tokens": 368
            }
          },
          "finish_reason": null
            }"#;

    let response: MistralTranscriptionResponse =
        serde_json::from_str(json).expect("should deserialize");

    assert_eq!(response.language, None);
    assert_eq!(response.model, VOXTRAL_MINI);
    assert_eq!(response.segments.len(), 1);

    let seg0 = &response.segments[0];
    assert_eq!(seg0.start, 0.2);
    assert_eq!(seg0.end, 4.6);
    assert_eq!(seg0.score, None);
    assert_eq!(seg0.speaker_id, Some("speaker_1".to_string()));
    assert_eq!(seg0.segment_type, "transcription_segment");

    assert_eq!(response.usage.prompt_audio_seconds, Some(5));
    assert_eq!(response.usage.prompt_tokens, 5);
    assert_eq!(response.usage.total_tokens, 404);
    let usage_token_details = response.usage.prompt_tokens_details.unwrap();
    let cached_token = usage_token_details.get("cached_tokens").unwrap();

    assert_eq!(cached_token.to_string().parse::<i32>().unwrap(), 368);
}

#[test]
fn test_response_conversion() {
    let mistral_response = MistralTranscriptionResponse {
        language: Some("en".to_string()),
        model: VOXTRAL_MINI.to_string(),
        segments: vec![SegmentChunk {
            start: 0.0,
            end: 1.0,
            text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                .into(),
            score: None,
            speaker_id: None,
            segment_type: "speech".to_string(),
        }],
        text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
            .to_string(),
        usage: TranscriptionUsage {
            prompt_audio_seconds: Some(1),
            prompt_tokens: 10,
            total_tokens: 20,
            completion_tokens: 10,
            prompt_tokens_details: None,
        },
    };

    let response = mistral_response
        .normalize("mistral")
        .expect("conversion should succeed");

    assert_eq!(
        response.text,
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
    );
    assert_eq!(response.provider, "mistral");
    assert_eq!(response.model.as_deref(), Some(VOXTRAL_MINI));
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.output_tokens, 10);
    assert_eq!(response.usage.total_tokens, 20);
}

#[tokio::test]
async fn transcription_non_success_preserves_status_and_body() {
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::{TranscriptionError, TranscriptionModel as _};

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(VOXTRAL_MINI);

    let Err(error) = model
        .transcription_request()
        .data(vec![0u8; 16])
        .send()
        .await
    else {
        panic!("transcription should fail with non-success status")
    };

    assert!(matches!(error, TranscriptionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
