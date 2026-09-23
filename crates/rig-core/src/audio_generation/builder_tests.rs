use serde_json::json;

use super::*;

/// Requests match what the replaced typestate builder produced for the same
/// inputs, captured before it was removed.
#[test]
fn builds_the_requests_the_typestate_builder_built() {
    let request = AudioGenerationRequestBuilder::new((), "hi", "alloy").build();
    assert_eq!(
        (
            request.text.as_str(),
            request.voice.as_str(),
            request.speed,
            request.additional_params
        ),
        ("hi", "alloy", 1.0, None)
    );
    let request = AudioGenerationRequestBuilder::new((), "hi", "alloy")
        .speed(1.5)
        .build();
    assert_eq!(request.speed, 1.5);
    let request = AudioGenerationRequestBuilder::new((), "hi", "alloy")
        .additional_params(json!({"response_format": "wav"}))
        .build();
    assert_eq!(
        request.additional_params,
        Some(json!({"response_format": "wav"}))
    );
}

/// Repeated calls merge and `None` clears, matching the completion and
/// transcription builders. The replaced builder kept only the last call
/// (`{"b": 2, "nested": {"y": 2}}` here).
#[test]
fn additional_params_merge_and_none_clears() {
    let request = AudioGenerationRequestBuilder::new((), "hi", "alloy")
        .additional_params(json!({"a": 1, "nested": {"x": 1}}))
        .additional_params(json!({"b": 2, "nested": {"y": 2}}))
        .build();
    assert_eq!(
        request.additional_params,
        Some(json!({"a": 1, "b": 2, "nested": {"y": 2}}))
    );
    let request = AudioGenerationRequestBuilder::new((), "hi", "alloy")
        .additional_params(json!({"a": 1}))
        .additional_params(None)
        .build();
    assert_eq!(request.additional_params, None);
}
