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
    match message {
        Some(msg) => from_body(msg),
        None => provider_error(fallback),
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
mod tests {
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
        let raw_body = Some(
            r#"{"message":"This model version has reached the end of its life."}"#.to_string(),
        );
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
        let err =
            InvokeModelError::InternalServerException(InternalServerException::builder().build());
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
        let err =
            InvokeModelError::InternalServerException(InternalServerException::builder().build());
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
}

#[cfg(test)]
mod request_id_tests {
    use super::*;
    use rig_core::ProviderResponseError;

    /// rig#2314: the AWS request id attaches to preserved provider bodies and
    /// leaves Rig-authored diagnostics untouched.
    #[test]
    fn attach_request_id_targets_provider_responses_only() {
        let attached = attach_request_id(
            CompletionError::ProviderResponse(ProviderResponseError::without_status("aws said no")),
            Some("aws-req-1".to_string()),
        );
        assert_eq!(attached.provider_request_id(), Some("aws-req-1"));
        assert!(
            attached.to_string().contains("request id: aws-req-1"),
            "the id appears in the logged message: {attached}"
        );

        let untouched = attach_request_id(
            CompletionError::ProviderError("rig diagnostic".to_string()),
            Some("aws-req-1".to_string()),
        );
        assert_eq!(untouched.provider_request_id(), None);
    }
}
