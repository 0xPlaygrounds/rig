use std::fmt;

use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::types::error::ConverseStreamOutputError;
use rig_core::completion::CompletionError;
use rig_core::embeddings::EmbeddingError;
use rig_core::image_generation::ImageGenerationError;

/// Emit a `fn(err) -> (Option<String>, String)` that extracts the
/// provider-supplied message from an AWS service error.
///
/// Each generated fn returns `(Some(message), _)` when the service supplied a
/// genuine error message (which should be surfaced as a provider response
/// body), otherwise `(None, fallback)` where `fallback` is Rig-authored
/// diagnostic prose. The caller decides which arm to surface (see [`gated`])
/// so that Rig prose never leaks into `provider_response_body()`.
macro_rules! service_error_message {
    ($fn_name:ident, $err_ty:ty, $default:expr, { $($variant:ident => $msg:expr),+ $(,)? }) => {
        fn $fn_name(err: $err_ty) -> (Option<String>, String) {
            type E = $err_ty;
            // The catch-all arm is not only "an exception we chose not to
            // name": `SdkError::into_service_error` funnels *every*
            // non-service failure (timeout, dispatch error, unparseable
            // response) and every exception this SDK version does not model
            // into `Unhandled`. Those still carry the service's own message in
            // their error metadata, so read it before falling back to Rig
            // prose — dropping it reported a Bedrock end-of-life notice as
            // "verify Internet connection or AWS keys".
            let metadata_message =
                ::aws_smithy_types::error::metadata::ProvideErrorMetadata::message(&err)
                    .map(str::to_string);
            match err {
                $(E::$variant(e) => (e.message, $msg.into()),)+
                _ => (metadata_message, $default.into()),
            }
        }
    };
}

/// The raw HTTP body Bedrock answered with, when the failure carries one.
///
/// `SdkError::into_service_error` funnels every failure this SDK version does
/// not model — a new exception type, or any response whose `x-amzn-errortype`
/// the transport did not preserve — into `Unhandled`, whose *source* holds the
/// parsed message while its `meta()` is empty. Reading the raw body recovers
/// what the service actually said instead of reporting Bedrock's end-of-life
/// notice as "verify Internet connection or AWS keys".
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
fn with_raw_body(
    (message, fallback): (Option<String>, String),
    raw_body: Option<String>,
) -> (Option<String>, String) {
    (message.or(raw_body), fallback)
}

/// Route a `(provider_message, fallback)` pair into an error type: a genuine
/// provider message becomes a provider response body, otherwise the
/// Rig-authored fallback becomes a plain provider error.
fn gated<E>(
    (message, fallback): (Option<String>, String),
    from_body: impl FnOnce(String) -> E,
    provider_error: impl FnOnce(String) -> E,
) -> E {
    message.map_or_else(|| provider_error(fallback), from_body)
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

impl From<AwsSdkInvokeModelError> for ImageGenerationError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        let raw_body = raw_response_body(&value.0);
        gated(
            with_raw_body(invoke_model_message(value.0.into_service_error()), raw_body),
            ImageGenerationError::from_provider_body,
            ImageGenerationError::ProviderError,
        )
    }
}

impl From<AwsSdkInvokeModelError> for EmbeddingError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        let raw_body = raw_response_body(&value.0);
        gated(
            with_raw_body(invoke_model_message(value.0.into_service_error()), raw_body),
            EmbeddingError::from_provider_body,
            EmbeddingError::ProviderError,
        )
    }
}

pub struct AwsSdkConverseError(pub SdkError<ConverseError, HttpResponse>);

/// Attach the AWS request id (from the SDK error's response metadata) to a
/// preserved provider response body — the id AWS support asks for on failed
/// calls (rig#2314). Rig-authored `ProviderError` diagnostics are left
/// untouched: the id belongs with what the provider actually said.
fn attach_request_id(error: CompletionError, request_id: Option<String>) -> CompletionError {
    match error {
        CompletionError::ProviderResponse(response) => {
            CompletionError::ProviderResponse(response.with_provider_request_id(request_id))
        }
        other => other,
    }
}

impl From<AwsSdkConverseError> for CompletionError {
    fn from(value: AwsSdkConverseError) -> Self {
        let raw_body = raw_response_body(&value.0);
        let request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&value.0).map(str::to_string);
        attach_request_id(
            gated(
                with_raw_body(converse_message(value.0.into_service_error()), raw_body),
                CompletionError::from_provider_body,
                CompletionError::ProviderError,
            ),
            request_id,
        )
    }
}

pub(crate) fn converse_stream_output_completion_error(
    err: ConverseStreamOutputError,
) -> CompletionError {
    gated(
        converse_stream_output_message(err),
        CompletionError::from_provider_body,
        CompletionError::ProviderError,
    )
}

pub struct AwsSdkConverseStreamError(pub SdkError<ConverseStreamError, HttpResponse>);
impl From<AwsSdkConverseStreamError> for CompletionError {
    fn from(value: AwsSdkConverseStreamError) -> Self {
        let raw_body = raw_response_body(&value.0);
        let request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&value.0).map(str::to_string);
        attach_request_id(
            gated(
                with_raw_body(
                    converse_stream_message(value.0.into_service_error()),
                    raw_body,
                ),
                CompletionError::from_provider_body,
                CompletionError::ProviderError,
            ),
            request_id,
        )
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
