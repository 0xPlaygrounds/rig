use super::*;
use aws_sdk_bedrockruntime::error::ErrorMetadata;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::types::error::{ThrottlingException, ValidationException};
use aws_smithy_types::body::SdkBody;

/// The raw response the SDK keeps beside a service error.
fn raw(status: StatusCode, body: &str) -> HttpResponse {
    let mut raw = HttpResponse::new(status.into(), SdkBody::from(body));
    raw.headers_mut().insert("x-amzn-requestid", "aws-req-1");
    raw
}

/// A modeled exception as the SDK deserializes it: the exception type is
/// the metadata's code.
fn throttled(message: Option<&str>) -> ConverseError {
    let mut meta = ErrorMetadata::builder().code("ThrottlingException");
    if let Some(message) = message {
        meta = meta.message(message);
    }
    ConverseError::ThrottlingException(
        ThrottlingException::builder()
            .set_message(message.map(str::to_owned))
            .meta(meta.build())
            .build(),
    )
}

/// A modeled exception keeps its message, type, status and request id, and
/// its message wins over the raw body.
#[test]
fn a_modeled_exception_is_the_provider_reply() {
    let error = sdk_error(SdkError::service_error(
        throttled(Some("slow down")),
        raw(StatusCode::TOO_MANY_REQUESTS, r#"{"message":"ignored"}"#),
    ));

    assert_eq!(error.provider_response_body(), Some("slow down"));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_request_id(), Some("aws-req-1"));
    assert_eq!(error.report().code.as_deref(), Some("ThrottlingException"));
    assert!(error.is_retryable());
}

/// Recorded, then replayed: a Bedrock 404 whose exception this SDK version
/// does not model has no message in its metadata. The raw body carries the
/// service's diagnostic, so it becomes the reply rather than Rig prose.
#[test]
fn an_unmodeled_exception_falls_back_to_the_raw_body() {
    let body = r#"{"message":"This model version has reached the end of its life."}"#;
    let error = sdk_error(SdkError::service_error(
        InvokeModelError::generic(ErrorMetadata::builder().build()),
        raw(StatusCode::NOT_FOUND, body),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::NOT_FOUND)
    );
    assert!(!error.is_retryable());
}

/// Without a message or body, the status the SDK saw is still the reply:
/// an empty body under that status, classified by it.
#[test]
fn a_message_less_exception_keeps_its_status() {
    let error = sdk_error(SdkError::service_error(
        throttled(None),
        raw(StatusCode::TOO_MANY_REQUESTS, ""),
    ));

    assert_eq!(error.provider_response_body(), Some(""));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert!(error.is_retryable());
}

/// A timeout is a retryable transport failure; a failure with neither a
/// reply nor a transport verdict is Rig prose, never a provider body.
#[test]
fn failures_without_a_reply_are_transport_errors_or_rig_prose() {
    let timed_out = sdk_error(SdkError::<ConverseError, HttpResponse>::timeout_error(
        std::io::Error::other("timed out"),
    ));
    assert!(matches!(timed_out, ProviderError::Http(_)), "{timed_out:?}");
    assert!(timed_out.is_retryable());

    let unbuilt = sdk_error(
        SdkError::<ConverseError, HttpResponse>::construction_failure(std::io::Error::other(
            "bad input",
        )),
    );
    assert!(matches!(&unbuilt, ProviderError::Provider(message) if message == UNEXPECTED));
    assert_eq!(unbuilt.provider_response_body(), None);
    assert!(!unbuilt.is_retryable());
}

#[test]
fn a_stream_exception_with_a_message_is_the_provider_reply() {
    let error = stream_error(ConverseStreamOutputError::ThrottlingException(
        ThrottlingException::builder().message("slow down").build(),
    ));

    assert_eq!(error.provider_response_body(), Some("slow down"));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.report().code.as_deref(), Some("ThrottlingException"));
    assert!(error.is_retryable());
}

/// A message-less stream exception is a diagnostic naming its type, not a
/// reply.
#[test]
fn a_message_less_stream_exception_is_rig_prose() {
    let error = stream_error(ConverseStreamOutputError::ValidationException(
        ValidationException::builder().build(),
    ));

    assert!(
        matches!(&error, ProviderError::Provider(message)
            if message == "Bedrock event stream failed (ValidationException)"),
        "{error:?}"
    );
    assert_eq!(error.provider_response_body(), None);
    assert!(!error.is_retryable());
}

/// Bedrock's exception type is the provider's code. Without an HTTP status
/// the type decides retryability; with one, the status decides, whatever
/// the type says.
#[test]
fn exception_types_classify_without_a_status_and_the_status_wins_with_one() {
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
        let classified = |status| {
            reply(
                Some("boom".to_string()),
                Some(code.to_string()),
                status,
                None,
                UNEXPECTED,
            )
        };
        let err = classified(None);
        assert_eq!(err.is_retryable(), transient, "{code} without a status");
        let report = err.report();
        assert_eq!(report.code.as_deref(), Some(code));
        assert_eq!(report.http_status, None);

        let throttled = classified(Some(StatusCode::TOO_MANY_REQUESTS));
        assert!(throttled.is_retryable(), "{code} under a 429 retries");
        assert_eq!(throttled.report().http_status, Some(429));

        let rejected = classified(Some(StatusCode::BAD_REQUEST));
        assert!(
            !rejected.is_retryable(),
            "{code} under a 400 does not retry"
        );
    }
}
