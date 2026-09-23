use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::error::ProviderError;
use rig_core::message::{DocumentMediaType, MimeType};

use super::converse_output::DocumentFormat;

pub struct RigDocumentMediaType(pub DocumentMediaType);

impl TryFrom<RigDocumentMediaType> for aws_bedrock::DocumentFormat {
    type Error = ProviderError;

    fn try_from(value: RigDocumentMediaType) -> Result<Self, Self::Error> {
        match value.0 {
            DocumentMediaType::PDF => Ok(aws_bedrock::DocumentFormat::Pdf),
            DocumentMediaType::TXT => Ok(aws_bedrock::DocumentFormat::Txt),
            DocumentMediaType::HTML => Ok(aws_bedrock::DocumentFormat::Html),
            DocumentMediaType::MARKDOWN => Ok(aws_bedrock::DocumentFormat::Md),
            DocumentMediaType::CSV => Ok(aws_bedrock::DocumentFormat::Csv),
            e => Err(ProviderError::Provider(format!(
                "Unsupported media type {}",
                e.to_mime_type()
            ))),
        }
    }
}

impl TryFrom<DocumentFormat> for RigDocumentMediaType {
    type Error = ProviderError;

    fn try_from(value: DocumentFormat) -> Result<Self, Self::Error> {
        // Preserve wire spellings in errors for unsupported media types.
        fn unsupported(format: &str) -> ProviderError {
            ProviderError::Provider(format!("Unsupported media type {format}"))
        }

        match value {
            DocumentFormat::Csv => Ok(RigDocumentMediaType(DocumentMediaType::CSV)),
            DocumentFormat::Html => Ok(RigDocumentMediaType(DocumentMediaType::HTML)),
            DocumentFormat::Md => Ok(RigDocumentMediaType(DocumentMediaType::MARKDOWN)),
            DocumentFormat::Pdf => Ok(RigDocumentMediaType(DocumentMediaType::PDF)),
            DocumentFormat::Txt => Ok(RigDocumentMediaType(DocumentMediaType::TXT)),
            DocumentFormat::Doc => Err(unsupported("doc")),
            DocumentFormat::Docx => Err(unsupported("docx")),
            DocumentFormat::Xls => Err(unsupported("xls")),
            DocumentFormat::Xlsx => Err(unsupported("xlsx")),
            DocumentFormat::Unknown(value) => Err(unsupported(&value.to_string())),
        }
    }
}
