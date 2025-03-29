use aws_sdk_bedrockruntime::types::DocumentFormat;
use rig::{
    completion::CompletionError,
    message::{DocumentMediaType, MimeType},
};

pub struct RigDocumentMediaType(pub DocumentMediaType);

impl TryFrom<RigDocumentMediaType> for DocumentFormat {
    type Error = CompletionError;

    fn try_from(value: RigDocumentMediaType) -> Result<Self, Self::Error> {
        match value.0 {
            DocumentMediaType::PDF => Ok(DocumentFormat::Pdf),
            DocumentMediaType::TXT => Ok(DocumentFormat::Txt),
            DocumentMediaType::HTML => Ok(DocumentFormat::Html),
            DocumentMediaType::MARKDOWN => Ok(DocumentFormat::Md),
            DocumentMediaType::CSV => Ok(DocumentFormat::Csv),
            e => Err(CompletionError::ProviderError(format!(
                "Unsupported media type {}",
                e.to_mime_type()
            ))),
        }
    }
}

impl TryFrom<DocumentFormat> for RigDocumentMediaType {
    type Error = CompletionError;

    fn try_from(value: DocumentFormat) -> Result<Self, Self::Error> {
        match value {
            DocumentFormat::Csv => Ok(RigDocumentMediaType(DocumentMediaType::CSV)),
            DocumentFormat::Html => Ok(RigDocumentMediaType(DocumentMediaType::HTML)),
            DocumentFormat::Md => Ok(RigDocumentMediaType(DocumentMediaType::MARKDOWN)),
            DocumentFormat::Pdf => Ok(RigDocumentMediaType(DocumentMediaType::PDF)),
            DocumentFormat::Txt => Ok(RigDocumentMediaType(DocumentMediaType::TXT)),
            e => Err(CompletionError::ProviderError(format!(
                "Unsupported media type {}",
                e
            ))),
        }
    }
}
