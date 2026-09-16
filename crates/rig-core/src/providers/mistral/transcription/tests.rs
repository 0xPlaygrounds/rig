use super::*;

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
