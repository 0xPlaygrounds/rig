use super::*;
use aws_sdk_bedrockruntime::types::error::{
    InternalServerException, ModelTimeoutException, ValidationException,
};

// NOTE: These tests construct the *extracted* service-error enum variants
// directly via the AWS-provided builders and drive the gating helpers plus
// the `From` conversions. The `SdkError` wrapper (and thus the public
// `AwsSdk*Error` newtypes) cannot be constructed in a unit test, so the
// `From` contract is asserted on the helper + builder-routed error type
// rather than on the newtype. None of these paths are feature-gated in
// `rig-bedrock` (the crate exposes no `image`/`audio` features; the
// completion/embedding/image/streaming modules are always compiled), so the
// tests only need `#[cfg(test)]`.

/// Recorded, then replayed: a Bedrock 404 whose exception this SDK version
/// does not classify arrives as `Unhandled`, whose `meta()` is empty and
/// whose message hides in its source. Before the raw-body fallback, that
/// surfaced as `ProviderError("… Verify Internet connection or AWS keys")`
/// and the operator never saw "This model version has reached the end of
/// its life". Cassette replay covers the classified path; this covers the
/// unclassified one, which no cassette can produce once the transport
/// preserves `x-amzn-errortype`.
#[test]
fn unclassified_error_falls_back_to_the_raw_provider_body() {
    let raw_body =
        Some(r#"{"message":"This model version has reached the end of its life."}"#.to_string());
    let unclassified = (None, UNEXPECTED.to_string());

    let error: CompletionError = gated(
        with_raw_body(unclassified, raw_body.clone()),
        CompletionError::from_provider_body,
        CompletionError::ProviderError,
    );

    assert_eq!(error.provider_response_body(), raw_body.as_deref());
}

/// The exception's own message still wins: the raw body is a fallback, not
/// a replacement, so classified errors keep their existing wording.
#[test]
fn classified_error_message_wins_over_the_raw_body() {
    let classified = (Some("boom".to_string()), UNEXPECTED.to_string());

    let (message, _fallback) =
        with_raw_body(classified, Some(r#"{"message":"ignored"}"#.to_string()));

    assert_eq!(message, Some("boom".to_string()));
}

/// With neither a classified message nor a body, Rig prose is still the
/// fallback — and it must not masquerade as a provider response body.
#[test]
fn absent_message_and_body_yields_rig_prose_not_a_provider_body() {
    let error: CompletionError = gated(
        with_raw_body((None, UNEXPECTED.to_string()), None),
        CompletionError::from_provider_body,
        CompletionError::ProviderError,
    );

    assert_eq!(error.provider_response_body(), None);
    assert!(matches!(error, CompletionError::ProviderError(_)));
}

#[test]
fn invoke_model_message_returns_provider_message_when_present() {
    let err = InvokeModelError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback) = invoke_model_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn invoke_model_message_returns_none_when_message_absent() {
    let err = InvokeModelError::InternalServerException(InternalServerException::builder().build());
    let (message, fallback) = invoke_model_message(err);
    assert_eq!(message, None);
    assert_eq!(fallback, "An internal server error occurred.".to_string());
}

#[test]
fn image_generation_with_provider_message_yields_provider_response() {
    let err = InvokeModelError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: ImageGenerationError = match invoke_model_message(err) {
        (Some(msg), _) => ImageGenerationError::from_provider_body(msg),
        (None, fallback) => ImageGenerationError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn image_generation_without_provider_message_yields_provider_error() {
    // A matched variant with no message -> `(None, fallback)` -> `ProviderError`,
    // which must NOT surface Rig prose through `provider_response_body()`.
    let err = InvokeModelError::ValidationException(ValidationException::builder().build());
    let error: ImageGenerationError = match invoke_model_message(err) {
        (Some(msg), _) => ImageGenerationError::from_provider_body(msg),
        (None, fallback) => ImageGenerationError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn embedding_with_provider_message_yields_provider_response() {
    let err = InvokeModelError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: EmbeddingError = match invoke_model_message(err) {
        (Some(msg), _) => EmbeddingError::from_provider_body(msg),
        (None, fallback) => EmbeddingError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn embedding_without_provider_message_yields_provider_error() {
    let err = InvokeModelError::InternalServerException(InternalServerException::builder().build());
    let error: EmbeddingError = match invoke_model_message(err) {
        (Some(msg), _) => EmbeddingError::from_provider_body(msg),
        (None, fallback) => EmbeddingError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
}

#[test]
fn converse_message_returns_provider_message_when_present() {
    let err = ConverseError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback) = converse_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn converse_with_provider_message_yields_provider_response() {
    let err = ConverseError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let error: CompletionError = match converse_message(err) {
        (Some(msg), _) => CompletionError::from_provider_body(msg),
        (None, fallback) => CompletionError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_without_provider_message_yields_provider_error() {
    let err = ConverseError::ModelTimeoutException(ModelTimeoutException::builder().build());
    let error: CompletionError = match converse_message(err) {
        (Some(msg), _) => CompletionError::from_provider_body(msg),
        (None, fallback) => CompletionError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_message_returns_provider_message_when_present() {
    let err = ConverseStreamError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback) = converse_stream_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn converse_stream_with_provider_message_yields_provider_response() {
    let err = ConverseStreamError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: CompletionError = match converse_stream_message(err) {
        (Some(msg), _) => CompletionError::from_provider_body(msg),
        (None, fallback) => CompletionError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_without_provider_message_yields_provider_error() {
    let err = ConverseStreamError::ValidationException(ValidationException::builder().build());
    let error: CompletionError = match converse_stream_message(err) {
        (Some(msg), _) => CompletionError::from_provider_body(msg),
        (None, fallback) => CompletionError::ProviderError(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_output_with_provider_message_yields_provider_response() {
    let err = ConverseStreamOutputError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error = converse_stream_output_completion_error(err);
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_output_without_provider_message_yields_provider_error() {
    let err =
        ConverseStreamOutputError::ValidationException(ValidationException::builder().build());
    let error = converse_stream_output_completion_error(err);
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}
