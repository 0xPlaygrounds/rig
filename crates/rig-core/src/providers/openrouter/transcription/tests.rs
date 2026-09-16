use super::*;

#[test]
fn test_transcription_response_deserialization() {
    let json = r#"{"text": "Hello world", "usage": {"seconds": 1.5, "cost": 0.001}}"#;
    let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.text, "Hello world");
    let usage = resp.usage.unwrap();
    assert_eq!(usage.seconds, Some(1.5));
}

#[test]
fn test_transcription_response_without_usage() {
    let json = r#"{"text": "Hello world"}"#;
    let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.text, "Hello world");
    assert!(resp.usage.is_none());
}
