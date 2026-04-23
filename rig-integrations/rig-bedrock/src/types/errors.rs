use std::fmt;

use aws_sdk_bedrockruntime::config::http::HttpResponse;
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use rig::completion::CompletionError;
use rig::embeddings::EmbeddingError;
use rig::image_generation::ImageGenerationError;

pub struct AwsSdkInvokeModelError(pub SdkError<InvokeModelError, HttpResponse>);

impl AwsSdkInvokeModelError {
    pub fn into_service_error(self) -> String {
        let error: String = match self.0.into_service_error() {
            InvokeModelError::ModelTimeoutException(e) => e.message.unwrap_or("The request took too long to process. Processing time exceeded the model timeout length.".into()),
            InvokeModelError::AccessDeniedException(e) => e.message.unwrap_or("The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
            InvokeModelError::ResourceNotFoundException(e) => e.message.unwrap_or("The specified resource ARN was not found.".into()),
            InvokeModelError::ThrottlingException(e) => e.message.unwrap_or("Your request was denied due to exceeding the account quotas for Amazon Bedrock.".into()),
            InvokeModelError::ServiceUnavailableException(e) => e.message.unwrap_or("The service isn't currently available.".into()),
            InvokeModelError::InternalServerException(e) => e.message.unwrap_or("An internal server error occurred.".into()),
            InvokeModelError::ValidationException(e) => e.message.unwrap_or("The input fails to satisfy the constraints specified by Amazon Bedrock.".into()),
            InvokeModelError::ModelNotReadyException(e) => e.message.unwrap_or("The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
            InvokeModelError::ModelErrorException(e) => e.message.unwrap_or("The request failed due to an error while processing the model.".into()),
            InvokeModelError::ServiceQuotaExceededException(e) => e.message.unwrap_or("Your request exceeds the service quota for your account.".into()),
            _ => "An unexpected error occurred. Verify Internet connection or AWS keys".into(),
        };
        error
    }
}

impl From<AwsSdkInvokeModelError> for ImageGenerationError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        ImageGenerationError::ProviderError(value.into_service_error())
    }
}

impl From<AwsSdkInvokeModelError> for EmbeddingError {
    fn from(value: AwsSdkInvokeModelError) -> Self {
        EmbeddingError::ProviderError(value.into_service_error())
    }
}

pub struct AwsSdkConverseError(pub SdkError<ConverseError, HttpResponse>);

impl From<AwsSdkConverseError> for CompletionError {
    fn from(value: AwsSdkConverseError) -> Self {
        let error: String = match value.0.into_service_error() {
            ConverseError::ModelTimeoutException(e) => e.message.unwrap_or("The request took too long to process. Processing time exceeded the model timeout length.".into()),
            ConverseError::AccessDeniedException(e) => e.message.unwrap_or("The request is denied because you do not have sufficient permissions to perform the requested action.".into()),
            ConverseError::ResourceNotFoundException(e) => e.message.unwrap_or("The specified resource ARN was not found.".into()),
            ConverseError::ThrottlingException(e) => e.message.unwrap_or("Your request was denied due to exceeding the account quotas for AWS Bedrock.".into()),
            ConverseError::ServiceUnavailableException(e) => e.message.unwrap_or("The service isn't currently available.".into()),
            ConverseError::InternalServerException(e) => e.message.unwrap_or("An internal server error occurred.".into()),
            ConverseError::ValidationException(e) => e.message.unwrap_or("The input fails to satisfy the constraints specified by AWS Bedrock.".into()),
            ConverseError::ModelNotReadyException(e) => e.message.unwrap_or("The model specified in the request is not ready to serve inference requests. The AWS SDK will automatically retry the operation up to 5 times.".into()),
            ConverseError::ModelErrorException(e) => e.message.unwrap_or("The request failed due to an error while processing the model.".into()),
            _ => String::from("An unexpected error occurred. Verify Internet connection or AWS keys")
        };
        CompletionError::ProviderError(error)
    }
}

pub struct AwsSdkConverseStreamError(pub SdkError<ConverseStreamError, HttpResponse>);
impl From<AwsSdkConverseStreamError> for CompletionError {
    fn from(value: AwsSdkConverseStreamError) -> Self {
        let error: String = match value.0.into_service_error() {
            ConverseStreamError::ModelTimeoutException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock model timed out".to_string()),
            ConverseStreamError::AccessDeniedException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock access denied".to_string()),
            ConverseStreamError::ResourceNotFoundException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock resource not found".to_string()),
            ConverseStreamError::ThrottlingException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock request throttled".to_string()),
            ConverseStreamError::ServiceUnavailableException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock service unavailable".to_string()),
            ConverseStreamError::InternalServerException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock internal server error".to_string()),
            ConverseStreamError::ModelStreamErrorException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock streaming model error".to_string()),
            ConverseStreamError::ValidationException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock validation error".to_string()),
            ConverseStreamError::ModelNotReadyException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock model not ready".to_string()),
            ConverseStreamError::ModelErrorException(e) => e
                .message
                .unwrap_or_else(|| "Bedrock model error".to_string()),
            _ => "An unexpected error occurred. Verify Internet connection or AWS keys".into(),
        };
        CompletionError::ProviderError(error)
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
