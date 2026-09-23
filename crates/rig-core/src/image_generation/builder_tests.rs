use serde_json::json;

use super::*;

/// Requests match what the replaced typestate builder produced for the same
/// inputs, captured before it was removed.
#[test]
fn builds_the_requests_the_typestate_builder_built() {
    let request = ImageGenerationRequestBuilder::new((), "lake").build();
    assert_eq!(
        (
            request.prompt.as_str(),
            request.width,
            request.height,
            request.additional_params
        ),
        ("lake", 256, 256, None)
    );
    let request = ImageGenerationRequestBuilder::new((), "lake")
        .width(1024)
        .height(512)
        .build();
    assert_eq!((request.width, request.height), (1024, 512));
    let request = ImageGenerationRequestBuilder::new((), "lake")
        .additional_params(json!({"quality": "low"}))
        .build();
    assert_eq!(request.additional_params, Some(json!({"quality": "low"})));
}

/// Repeated calls merge and `None` clears, matching the completion and
/// transcription builders. The replaced builder kept only the last call
/// (`{"b": 2, "nested": {"y": 2}}` here).
#[test]
fn additional_params_merge_and_none_clears() {
    let request = ImageGenerationRequestBuilder::new((), "lake")
        .additional_params(json!({"a": 1, "nested": {"x": 1}}))
        .additional_params(json!({"b": 2, "nested": {"y": 2}}))
        .build();
    assert_eq!(
        request.additional_params,
        Some(json!({"a": 1, "b": 2, "nested": {"y": 2}}))
    );
    let request = ImageGenerationRequestBuilder::new((), "lake")
        .additional_params(json!({"a": 1}))
        .additional_params(None)
        .build();
    assert_eq!(request.additional_params, None);
}
