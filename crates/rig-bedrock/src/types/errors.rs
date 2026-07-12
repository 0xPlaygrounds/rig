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

/// Extracts the provider-supplied message from an [`InvokeModelError`] service
/// error.
///
/// Returns `(Some(message), _)` when the service supplied a genuine error
/// message (which should be surfaced as a provider response body), otherwise
/// `(None, fallback)` where `fallback` is Rig-authored diagnostic prose. The
/// caller decides which arm to surface so that Rig prose never leaks into
/// `provider_response_body()`.
fn invoke_model_message(err: InvokeModelError) -> (Option<String>, String) {
    match err {
        InvokeModelError::ModelTimeoutException(e) => (e.message, "The request took too long to process. Processing time exceeded the model timeout length.".into()),
        InvokeModelError::AccessDeniedException(e) => (e.message, "The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
        InvokeModelError::ResourceNotFoundException(e) => (e.message, "The specified resource ARN was not found.".into()),
        InvokeModelError::ThrottlingException(e) => (e.message, "Your request was denied due to exceeding the account quotas for Amazon Bedrock.".into()),
        InvokeModelError::ServiceUnavailableException(e) => (e.message, "The service isn't currently available.".into()),
        InvokeModelError::InternalServerException(e) => (e.message, "An internal server error occurred.".into()),
        InvokeModelError::ValidationException(e) => (e.message, "The input fails to satisfy the constraints specified by Amazon Bedrock.".into()),
        InvokeModelError::ModelNotReadyException(e) => (e.message, "The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
        InvokeModelError::ModelErrorException(e) => (e.message, "The request failed due to an error while processing the model.".into()),
        InvokeModelError::ServiceQuotaExceededException(e) => (e.message, "Your request exceeds the service quota for your account.".into()),
        _ => (None, "An unexpected error occurred. Verify Internet connection or AWS keys".into()),
    }
}

/// Extracts the provider-supplied message from a [`ConverseError`] service
/// error. See [`invoke_model_message`] for the gating contract.
fn converse_message(err: ConverseError) -> (Option<String>, String) {
    match err {
        ConverseError::ModelTimeoutException(e) => (e.message, "The request took too long to process. Processing time exceeded the model timeout length.".into()),
        ConverseError::AccessDeniedException(e) => (e.message, "The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
        ConverseError::ResourceNotFoundException(e) => (e.message, "The specified resource ARN was not found.".into()),
        ConverseError::ThrottlingException(e) => (e.message, "Your request was denied due to exceeding the account quotas for AWS Bedrock.".into()),
        ConverseError::ServiceUnavailableException(e) => (e.message, "The service isn't currently available.".into()),
        ConverseError::InternalServerException(e) => (e.message, "An internal server error occurred.".into()),
        ConverseError::ValidationException(e) => (e.message, "The input fails to satisfy the constraints specified by AWS Bedrock.".into()),
        ConverseError::ModelNotReadyException(e) => (e.message, "The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
        ConverseError::ModelErrorException(e) => (e.message, "The request failed due to an error while processing the model.".into()),
        _ => (None, "An unexpected error occurred. Verify Internet connection or AWS keys".into()),
    }
}

/// Extracts the provider-supplied message from a [`ConverseStreamError`]
/// service error. See [`invoke_model_message`] for the gating contract.
fn converse_stream_message(err: ConverseStreamError) -> (Option<String>, String) {
    match err {
        ConverseStreamError::ModelTimeoutException(e) => {
            (e.message, "Bedrock model timed out".into())
        }
        ConverseStreamError::AccessDeniedException(e) => {
            (e.message, "Bedrock access denied".into())
        }
        ConverseStreamError::ResourceNotFoundException(e) => {
            (e.message, "Bedrock resource not found".into())
        }
        ConverseStreamError::ThrottlingException(e) => {
            (e.message, "Bedrock request throttled".into())
        }
        ConverseStreamError::ServiceUnavailableException(e) => {
            (e.message, "Bedrock service unavailable".into())
        }
        ConverseStreamError::InternalServerException(e) => {
            (e.message, "Bedrock internal server error".into())
        }
        ConverseStreamError::ModelStreamErrorException(e) => {
            (e.message, "Bedrock streaming model error".into())
        }
        ConverseStreamError::ValidationException(e) => {
            (e.message, "Bedrock validation error".into())
        }
        ConverseStreamError::ModelNotReadyException(e) => {
            (e.message, "Bedrock model not ready".into())
        }
        ConverseStreamError::ModelErrorException(e) => (e.message, "Bedrock model error".into()),
        _ => (
            None,
            "An unexpected error occurred. Verify Internet connection or AWS keys".into(),
        ),
    }
}

pub struct AwsSdkInvokeModelError(pub SdkError<InvokeModelError, HttpResponse>);

impl From<AwsSdkInvokeModelError> for ImageGenerationError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        match invoke_model_message(value.0.into_service_error()) {
            (Some(msg), _) => ImageGenerationError::from_provider_body(msg),
            (None, fallback) => ImageGenerationError::ProviderError(fallback),
        }
    }
}

impl From<AwsSdkInvokeModelError> for EmbeddingError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        match invoke_model_message(value.0.into_service_error()) {
            (Some(msg), _) => EmbeddingError::from_provider_body(msg),
            (None, fallback) => EmbeddingError::ProviderError(fallback),
        }
    }
}

pub struct AwsSdkConverseError(pub SdkError<ConverseError, HttpResponse>);

impl From<AwsSdkConverseError> for CompletionError {
    fn from(value: AwsSdkConverseError) -> Self {
        match converse_message(value.0.into_service_error()) {
            (Some(msg), _) => CompletionError::from_provider_body(msg),
            (None, fallback) => CompletionError::ProviderError(fallback),
        }
    }
}

pub(crate) fn converse_stream_output_completion_error(
    err: ConverseStreamOutputError,
) -> CompletionError {
    let (message, fallback) = match err {
        ConverseStreamOutputError::InternalServerException(err) => {
            (err.message, "Bedrock internal server error".into())
        }
        ConverseStreamOutputError::ModelStreamErrorException(err) => {
            (err.message, "Bedrock streaming model error".into())
        }
        ConverseStreamOutputError::ValidationException(err) => {
            (err.message, "Bedrock validation error".into())
        }
        ConverseStreamOutputError::ThrottlingException(err) => {
            (err.message, "Bedrock request throttled".into())
        }
        ConverseStreamOutputError::ServiceUnavailableException(err) => {
            (err.message, "Bedrock service unavailable".into())
        }
        _ => (None, "Bedrock event stream failed".into()),
    };

    match message {
        Some(message) => CompletionError::from_provider_body(message),
        None => CompletionError::ProviderError(fallback),
    }
}

fn converse_stream_completion_error(err: ConverseStreamError) -> CompletionError {
    match converse_stream_message(err) {
        (Some(msg), _) => CompletionError::from_provider_body(msg),
        (None, fallback) => CompletionError::ProviderError(fallback),
    }
}

pub struct AwsSdkConverseStreamError(pub SdkError<ConverseStreamError, HttpResponse>);
impl From<AwsSdkConverseStreamError> for CompletionError {
    fn from(value: AwsSdkConverseStreamError) -> Self {
        converse_stream_completion_error(value.0.into_service_error())
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
