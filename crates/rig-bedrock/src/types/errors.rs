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
            match err {
                $(E::$variant(e) => (e.message, $msg.into()),)+
                _ => (None, $default.into()),
            }
        }
    };
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
        gated(
            invoke_model_message(value.0.into_service_error()),
            ImageGenerationError::from_provider_body,
            ImageGenerationError::ProviderError,
        )
    }
}

impl From<AwsSdkInvokeModelError> for EmbeddingError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        gated(
            invoke_model_message(value.0.into_service_error()),
            EmbeddingError::from_provider_body,
            EmbeddingError::ProviderError,
        )
    }
}

pub struct AwsSdkConverseError(pub SdkError<ConverseError, HttpResponse>);

impl From<AwsSdkConverseError> for CompletionError {
    fn from(value: AwsSdkConverseError) -> Self {
        gated(
            converse_message(value.0.into_service_error()),
            CompletionError::from_provider_body,
            CompletionError::ProviderError,
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
        gated(
            converse_stream_message(value.0.into_service_error()),
            CompletionError::from_provider_body,
            CompletionError::ProviderError,
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
