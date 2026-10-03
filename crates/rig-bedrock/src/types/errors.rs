use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::{ProvideErrorMetadata, SdkError};
use aws_sdk_bedrockruntime::operation::RequestId;
use rig_core::error::ProviderError;
use rig_core::http_client::StatusCode;
use rig_core::json_utils::Lenient;

/// Rig's diagnostic for a failed call that carried neither a provider
/// message nor an HTTP status.
const UNEXPECTED: &str = "An unexpected error occurred. Verify Internet connection or AWS keys";

/// Converts a failed Bedrock SDK call into a provider error. The exception's
/// message is the provider's reply, or the raw response body when this SDK
/// version does not model the exception. The reply carries the HTTP status,
/// the exception type as its code, and the AWS request id.
pub(crate) fn sdk_error<E: ProvideErrorMetadata>(
    error: SdkError<E, HttpResponse>,
) -> ProviderError {
    let raw = error.raw_response();
    let status = raw.and_then(|raw| StatusCode::from_u16(raw.status().as_u16()).ok());
    let message = error
        .message()
        .map(str::to_owned)
        .or_else(|| raw.and_then(raw_body));
    // Timeout and dispatch failures are retry hints; delivery is not guaranteed.
    let transient = matches!(
        error,
        SdkError::TimeoutError(_) | SdkError::DispatchFailure(_)
    )
    .then_some(true);
    let code = error.code().map(str::to_owned);
    reply(message, code, status, transient, UNEXPECTED)
        .with_provider_request_id(error.request_id().map(str::to_owned))
}

/// Converts an exception Bedrock sent mid-stream, `{"<type>": <payload>}`
/// as its event-stream message states it. The type names the code.
pub(crate) fn exception(kind: &str, payload: &serde_json::Value) -> ProviderError {
    let mut code = kind.to_owned();
    if let Some(first) = code.get_mut(..1) {
        first.make_ascii_uppercase();
    }
    let message = payload.str("message").or_else(|| payload.str("Message"));
    reply(
        message.map(str::to_owned),
        Some(code),
        None,
        None,
        STREAM_FAILED,
    )
}

/// Converts a stream the SDK could not read on, which no exception message
/// explained.
pub(crate) fn stream_error<E: ProvideErrorMetadata, R>(error: SdkError<E, R>) -> ProviderError {
    let message = error.message().map(str::to_owned);
    reply(
        message,
        error.code().map(str::to_owned),
        None,
        None,
        STREAM_FAILED,
    )
}

const STREAM_FAILED: &str = "Bedrock event stream failed";

/// The trimmed, nonempty UTF-8 body the SDK retained. It carries the
/// service's diagnostic for exceptions absent from the parsed metadata.
fn raw_body(raw: &HttpResponse) -> Option<String> {
    let body = std::str::from_utf8(raw.body().bytes()?).ok()?.trim();
    (!body.is_empty()).then(|| body.to_owned())
}

/// Classifies known transient exception codes when no HTTP status is available.
/// Unlisted codes are non-transient.
fn transient_exception(code: &str) -> bool {
    matches!(
        code,
        "ThrottlingException"
            | "ServiceUnavailableException"
            | "InternalServerException"
            | "ModelNotReadyException"
            | "ModelTimeoutException"
            | "ModelStreamErrorException"
    )
}

/// The provider's reply when it sent a message or an HTTP status, with the
/// exception type deciding retryability when no status does. Otherwise a
/// transport error for a `transient` failure, or Rig's `fallback`
/// diagnostic, which never poses as a provider body.
fn reply(
    message: Option<String>,
    code: Option<String>,
    status: Option<StatusCode>,
    transient: Option<bool>,
    fallback: &str,
) -> ProviderError {
    if message.is_none() && status.is_none() {
        let fallback = match code {
            Some(code) => format!("{fallback} ({code})"),
            None => fallback.to_owned(),
        };
        return if transient == Some(true) {
            ProviderError::Http(
                rig_core::http_client::Error::instance(std::io::Error::other(fallback)).into(),
            )
        } else {
            ProviderError::Provider(fallback)
        };
    }
    let transient = code.as_deref().map(transient_exception).or(transient);
    ProviderError::from_provider_body(message.unwrap_or_default())
        .with_provider_status(status)
        .with_provider_code(code)
        .with_transient(transient)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod request_id_tests;
