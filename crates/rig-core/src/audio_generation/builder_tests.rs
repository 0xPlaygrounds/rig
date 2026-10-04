use serde_json::json;

use crate::audio_generation::AudioGenerationRequestBuilder;

/// Requests match what the replaced typestate builder produced for the same
/// inputs, captured before it was removed.
#[test]
fn builds_the_requests_the_typestate_builder_built() {
    let request = AudioGenerationRequestBuilder::new("hi", "alloy").build();
    assert_eq!(
        (
            request.text.as_str(),
            request.voice.as_str(),
            request.speed,
            request.additional_params
        ),
        ("hi", "alloy", 1.0, None)
    );
    let request = AudioGenerationRequestBuilder::new("hi", "alloy")
        .speed(1.5)
        .build();
    assert_eq!(request.speed, 1.5);
    let request = AudioGenerationRequestBuilder::new("hi", "alloy")
        .additional_params(json!({"response_format": "wav"}))
        .build();
    assert_eq!(
        request.additional_params,
        Some(json!({"response_format": "wav"}))
    );
}
