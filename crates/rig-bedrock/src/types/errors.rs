use std::fmt;

use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use rig_core::error::ProviderError;
use rig_core::http_client::StatusCode;

/// What a service error said about itself: the provider's message when it
/// supplied one, Rig's fallback prose otherwise, and the exception type as
/// the provider's own code (`ThrottlingException`).
type Classified = (Option<String>, String, Option<String>);

/// Generates classifiers returning provider message, fallback diagnostic, and
/// exception code separately. Fallback prose must not become a provider body.
macro_rules! service_error_message {
    ($fn_name:ident, $err_ty:ty, $default:expr, { $($variant:ident => $msg:expr),+ $(,)? }) => {
        fn $fn_name(err: $err_ty) -> Classified {
            type E = $err_ty;
            // Unhandled exceptions may retain provider metadata; preserve it
            // before considering fallback diagnostics.
            let metadata_message =
                ::aws_smithy_types::error::metadata::ProvideErrorMetadata::message(&err)
                    .map(str::to_string);
            // The exception type is the provider's code: the service's
            // `x-amzn-errortype` when this SDK version does not model it.
            let metadata_code =
                ::aws_smithy_types::error::metadata::ProvideErrorMetadata::code(&err)
                    .map(str::to_string);
            match err {
                $(E::$variant(e) => (e.message, $msg.into(), Some(stringify!($variant).to_string())),)+
                _ => (metadata_message, $default.into(), metadata_code),
            }
        }
    };
}

/// Returns a trimmed, nonempty UTF-8 response body when retained by the SDK.
/// Raw bodies preserve service diagnostics absent from parsed error metadata.
fn raw_response_body<E, R>(error: &SdkError<E, R>) -> Option<String>
where
    R: RawResponseBody,
{
    let body = error.raw_response()?.body_text()?;
    let body = body.trim();

    (!body.is_empty()).then(|| body.to_string())
}

/// The raw-body accessor for the response type an `SdkError` carries.
trait RawResponseBody {
    fn body_text(&self) -> Option<&str>;
}

impl RawResponseBody for HttpResponse {
    fn body_text(&self) -> Option<&str> {
        std::str::from_utf8(self.body().bytes()?).ok()
    }
}

/// Prefer the exception's own message; fall back to the raw provider body for
/// the failures this SDK version cannot classify.
fn with_raw_body((message, fallback, code): Classified, raw_body: Option<String>) -> Classified {
    (message.or(raw_body), fallback, code)
}

/// The HTTP status the SDK saw beside the exception, when it kept the raw
/// response: then the reply classifies by status like any HTTP reply.
fn raw_response_status<E, R>(error: &SdkError<E, R>) -> Option<StatusCode>
where
    R: RawResponseStatus,
{
    StatusCode::from_u16(error.raw_response()?.status_u16()).ok()
}

/// The raw-status accessor for the response type an `SdkError` carries.
trait RawResponseStatus {
    fn status_u16(&self) -> u16;
}

impl RawResponseStatus for HttpResponse {
    fn status_u16(&self) -> u16 {
        self.status().as_u16()
    }
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

/// The facts an SDK error carries beside its message, for [`gated`] to
/// stamp on the provider's reply.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Transport {
    status: Option<StatusCode>,
    /// Retry hint for SDK timeout or dispatch failures; delivery is not guaranteed.
    transient: Option<bool>,
}

impl Transport {
    fn of<E, R: RawResponseStatus>(error: &SdkError<E, R>) -> Self {
        Self {
            status: raw_response_status(error),
            transient: matches!(
                error,
                SdkError::TimeoutError(_) | SdkError::DispatchFailure(_)
            )
            .then_some(true),
        }
    }
}

/// Constructs a provider response error when a message or status is available,
/// preserving status, code, and retry hints. Otherwise uses a transport error for
/// SDK timeout/dispatch failures or a plain fallback diagnostic.
fn gated((message, fallback, code): Classified, transport: Transport) -> ProviderError {
    let transient = code
        .as_deref()
        .map(transient_exception)
        .or(transport.transient);
    let reply = |body: String, status: Option<StatusCode>| {
        ProviderError::from_provider_body(body)
            .with_provider_status(status)
            .with_provider_code(code)
            .with_transient(transient)
    };
    match (message, transport.status) {
        (Some(body), status) => reply(body, status),
        (None, Some(status)) => reply(String::new(), Some(status)),
        (None, None) if transport.transient == Some(true) => ProviderError::Http(
            rig_core::http_client::Error::instance(std::io::Error::other(fallback)),
        ),
        (None, None) => ProviderError::Provider(fallback),
    }
}

const UNEXPECTED: &str = "An unexpected error occurred. Verify Internet connection or AWS keys";

service_error_message!(invoke_model_message, InvokeModelError, UNEXPECTED, {
    ModelTimeoutException => "The request took too long to process. Processing time exceeded the model timeout length.",
    AccessDeniedException => "The request is denied because you do not have sufficient permissions to perform the requested action.",
    ResourceNotFoundException => "The specified resource ARN was not found.",
    ThrottlingException => "Your request was denied due to exceeding the account quotas for Amazon Bedrock.",
    ServiceUnavailableException => "The service isn't currently available.",
    InternalServerException => "An internal server error occurred.",
    ValidationException => "The input fails to satisfy the constraints specified by Amazon Bedrock.",
    ModelNotReadyException => "The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.",
    ModelErrorException => "The request failed due to an error while processing the model.",
    ServiceQuotaExceededException => "Your request exceeds the service quota for your account.",
});

service_error_message!(converse_message, ConverseError, UNEXPECTED, {
    ModelTimeoutException => "The request took too long to process. Processing time exceeded the model timeout length.",
    AccessDeniedException => "The request is denied because you do not have sufficient permissions to perform the requested action.",
    ResourceNotFoundException => "The specified resource ARN was not found.",
    ThrottlingException => "Your request was denied due to exceeding the account quotas for AWS Bedrock.",
    ServiceUnavailableException => "The service isn't currently available.",
    InternalServerException => "An internal server error occurred.",
    ValidationException => "The input fails to satisfy the constraints specified by AWS Bedrock.",
    ModelNotReadyException => "The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.",
    ModelErrorException => "The request failed due to an error while processing the model.",
});

service_error_message!(converse_stream_message, ConverseStreamError, UNEXPECTED, {
    ModelTimeoutException => "Bedrock model timed out",
    AccessDeniedException => "Bedrock access denied",
    ResourceNotFoundException => "Bedrock resource not found",
    ThrottlingException => "Bedrock request throttled",
    ServiceUnavailableException => "Bedrock service unavailable",
    InternalServerException => "Bedrock internal server error",
    ModelStreamErrorException => "Bedrock streaming model error",
    ValidationException => "Bedrock validation error",
    ModelNotReadyException => "Bedrock model not ready",
    ModelErrorException => "Bedrock model error",
});

service_error_message!(
    converse_stream_output_message,
    ConverseStreamOutputError,
    "Bedrock event stream failed",
    {
        InternalServerException => "Bedrock internal server error",
        ModelStreamErrorException => "Bedrock streaming model error",
        ValidationException => "Bedrock validation error",
        ThrottlingException => "Bedrock request throttled",
        ServiceUnavailableException => "Bedrock service unavailable",
    }
);

pub struct AwsSdkInvokeModelError(pub SdkError<InvokeModelError, HttpResponse>);

impl From<AwsSdkInvokeModelError> for ProviderError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        let raw_body = raw_response_body(&value.0);
        let transport = Transport::of(&value.0);
        gated(
            with_raw_body(invoke_model_message(value.0.into_service_error()), raw_body),
            transport,
        )
    }
}

pub struct AwsSdkConverseError(pub SdkError<ConverseError, HttpResponse>);

impl From<AwsSdkConverseError> for ProviderError {
    fn from(value: AwsSdkConverseError) -> Self {
        let raw_body = raw_response_body(&value.0);
        let transport = Transport::of(&value.0);
        let request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&value.0).map(str::to_string);
        gated(
            with_raw_body(converse_message(value.0.into_service_error()), raw_body),
            transport,
        )
        .with_provider_request_id(request_id)
    }
}

pub(crate) fn converse_stream_output_completion_error(
    err: ConverseStreamOutputError,
) -> ProviderError {
    gated(converse_stream_output_message(err), Transport::default())
}

pub struct AwsSdkConverseStreamError(pub SdkError<ConverseStreamError, HttpResponse>);
impl From<AwsSdkConverseStreamError> for ProviderError {
    fn from(value: AwsSdkConverseStreamError) -> Self {
        let raw_body = raw_response_body(&value.0);
        let transport = Transport::of(&value.0);
        let request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&value.0).map(str::to_string);
        gated(
            with_raw_body(
                converse_stream_message(value.0.into_service_error()),
                raw_body,
            ),
            transport,
        )
        .with_provider_request_id(request_id)
    }
}

#[derive(Debug)]
pub struct TypeConversionError(String);

impl TypeConversionError {
    pub fn new(input: &str) -> Self {
        Self(input.to_string())
    }
}

impl fmt::Display for TypeConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = self.0.clone();
        write!(f, "{message}")
    }
}

impl std::error::Error for TypeConversionError {}

impl From<std::convert::Infallible> for TypeConversionError {
    fn from(value: std::convert::Infallible) -> Self {
        match value {}
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod request_id_tests;
