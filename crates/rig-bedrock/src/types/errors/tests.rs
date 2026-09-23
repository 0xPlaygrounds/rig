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
    let unclassified = (None, UNEXPECTED.to_string(), None);

    let error: ProviderError = gated(
        with_raw_body(unclassified, raw_body.clone()),
        Transport::default(),
    );

    assert_eq!(error.provider_response_body(), raw_body.as_deref());
}

/// The exception's own message still wins: the raw body is a fallback, not
/// a replacement, so classified errors keep their existing wording.
#[test]
fn classified_error_message_wins_over_the_raw_body() {
    let classified = (Some("boom".to_string()), UNEXPECTED.to_string(), None);

    let (message, _fallback, _) =
        with_raw_body(classified, Some(r#"{"message":"ignored"}"#.to_string()));

    assert_eq!(message, Some("boom".to_string()));
}

/// With neither a classified message nor a body, Rig prose is still the
/// fallback — and it must not masquerade as a provider response body.
#[test]
fn absent_message_and_body_yields_rig_prose_not_a_provider_body() {
    let error: ProviderError = gated(
        with_raw_body((None, UNEXPECTED.to_string(), None), None),
        Transport::default(),
    );

    assert_eq!(error.provider_response_body(), None);
    assert!(matches!(error, ProviderError::Provider(_)));
}

#[test]
fn invoke_model_message_returns_provider_message_when_present() {
    let err = InvokeModelError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback, _) = invoke_model_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn invoke_model_message_returns_none_when_message_absent() {
    let err = InvokeModelError::InternalServerException(InternalServerException::builder().build());
    let (message, fallback, _) = invoke_model_message(err);
    assert_eq!(message, None);
    assert_eq!(fallback, "An internal server error occurred.".to_string());
}

#[test]
fn image_generation_with_provider_message_yields_provider_response() {
    let err = InvokeModelError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: ProviderError = match invoke_model_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn image_generation_without_provider_message_yields_provider_error() {
    // A matched variant with no message -> `(None, fallback)` -> `ProviderError`,
    // which must NOT surface Rig prose through `provider_response_body()`.
    let err = InvokeModelError::ValidationException(ValidationException::builder().build());
    let error: ProviderError = match invoke_model_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn embedding_with_provider_message_yields_provider_response() {
    let err = InvokeModelError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: ProviderError = match invoke_model_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn embedding_without_provider_message_yields_provider_error() {
    let err = InvokeModelError::InternalServerException(InternalServerException::builder().build());
    let error: ProviderError = match invoke_model_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
}

#[test]
fn converse_message_returns_provider_message_when_present() {
    let err = ConverseError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback, _) = converse_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn converse_with_provider_message_yields_provider_response() {
    let err = ConverseError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let error: ProviderError = match converse_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_without_provider_message_yields_provider_error() {
    let err = ConverseError::ModelTimeoutException(ModelTimeoutException::builder().build());
    let error: ProviderError = match converse_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_message_returns_provider_message_when_present() {
    let err = ConverseStreamError::ModelTimeoutException(
        ModelTimeoutException::builder().message("boom").build(),
    );
    let (message, _fallback, _) = converse_stream_message(err);
    assert_eq!(message, Some("boom".to_string()));
}

#[test]
fn converse_stream_with_provider_message_yields_provider_response() {
    let err = ConverseStreamError::ValidationException(
        ValidationException::builder().message("boom").build(),
    );
    let error: ProviderError = match converse_stream_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
    };
    assert_eq!(error.provider_response_body(), Some("boom"));
    assert_eq!(error.provider_response_status(), None);
}

#[test]
fn converse_stream_without_provider_message_yields_provider_error() {
    let err = ConverseStreamError::ValidationException(ValidationException::builder().build());
    let error: ProviderError = match converse_stream_message(err) {
        (Some(msg), _, _) => ProviderError::from_provider_body(msg),
        (None, fallback, _) => ProviderError::Provider(fallback),
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

/// Bedrock's exception type is the provider's code. Without the SDK's HTTP
/// status at hand the type decides retryability (throttled, unavailable,
/// not ready, timed out retry; a bad request does not); with the status,
/// the status decides, whatever the type says.
#[test]
fn exception_types_classify_without_a_status_and_the_status_wins_with_one() {
    use aws_sdk_bedrockruntime::types::error::ThrottlingException;
    let cells = [
        ("ThrottlingException", true),
        ("ServiceUnavailableException", true),
        ("InternalServerException", true),
        ("ModelNotReadyException", true),
        ("ModelTimeoutException", true),
        ("ModelStreamErrorException", true),
        ("ValidationException", false),
        ("AccessDeniedException", false),
        ("ResourceNotFoundException", false),
        ("ServiceQuotaExceededException", false),
        ("ModelErrorException", false),
    ];
    for (code, transient) in cells {
        let classified = || {
            (
                Some("boom".to_string()),
                UNEXPECTED.to_string(),
                Some(code.to_string()),
            )
        };
        let err: ProviderError = gated(classified(), Transport::default());
        assert_eq!(err.is_retryable(), transient, "{code} without a status");
        let report = err.report();
        assert_eq!(report.code.as_deref(), Some(code));
        assert_eq!(report.http_status, None);

        let throttled: ProviderError = gated(
            classified(),
            Transport {
                status: Some(StatusCode::TOO_MANY_REQUESTS),
                transient: None,
            },
        );
        assert!(throttled.is_retryable(), "{code} under a 429 retries");
        assert_eq!(throttled.report().http_status, Some(429));

        let rejected: ProviderError = gated(
            classified(),
            Transport {
                status: Some(StatusCode::BAD_REQUEST),
                transient: None,
            },
        );
        assert!(
            !rejected.is_retryable(),
            "{code} under a 400 does not retry"
        );
    }

    // The macro names the typed variant as the code.
    let (_, _, code) = converse_message(ConverseError::ThrottlingException(
        ThrottlingException::builder().message("slow down").build(),
    ));
    assert_eq!(code.as_deref(), Some("ThrottlingException"));

    // Rig prose never carries a code: a message-less exception is a
    // diagnostic, not a reply.
    let plain: ProviderError = gated(
        (
            None,
            UNEXPECTED.to_string(),
            Some("ThrottlingException".to_string()),
        ),
        Transport::default(),
    );
    assert!(matches!(plain, ProviderError::Provider(_)));
    assert!(!plain.is_retryable());
}

/// Without a provider message: a status the SDK saw is still the reply
/// (an empty body under that status, classified by it); a timeout or
/// dispatch failure is a transport failure and retries; only a failure
/// with neither is Rig prose.
#[test]
fn a_message_less_failure_keeps_its_status_or_its_transport_verdict() {
    let none = || (None, UNEXPECTED.to_string(), None);
    let throttled: ProviderError = gated(
        none(),
        Transport {
            status: Some(StatusCode::TOO_MANY_REQUESTS),
            transient: None,
        },
    );
    assert!(throttled.is_retryable());
    assert_eq!(
        throttled.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(throttled.provider_response_body(), Some(""));

    let timed_out: ProviderError = gated(
        none(),
        Transport {
            status: None,
            transient: Some(true),
        },
    );
    assert!(matches!(timed_out, ProviderError::Http(_)), "{timed_out:?}");
    assert!(timed_out.is_retryable());

    let plain: ProviderError = gated(none(), Transport::default());
    assert!(matches!(plain, ProviderError::Provider(_)));
}
