use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{ContentFormat, Document, DocumentMediaType, MimeType},
};

use base64::{prelude::BASE64_STANDARD, Engine};

pub struct RigDocument(pub Document);

impl TryFrom<RigDocument> for aws_bedrock::DocumentBlock {
    type Error = CompletionError;

    fn try_from(value: RigDocument) -> Result<Self, Self::Error> {
        let format = value
            .0
            .media_type
            .map(|doc| match doc {
                DocumentMediaType::PDF => Ok(aws_bedrock::DocumentFormat::Pdf),
                DocumentMediaType::TXT => Ok(aws_bedrock::DocumentFormat::Txt),
                DocumentMediaType::HTML => Ok(aws_bedrock::DocumentFormat::Html),
                DocumentMediaType::MARKDOWN => Ok(aws_bedrock::DocumentFormat::Md),
                DocumentMediaType::CSV => Ok(aws_bedrock::DocumentFormat::Csv),
                e => Err(CompletionError::ProviderError(format!(
                    "Unsupported media type {}",
                    e.to_mime_type()
                ))),
            })
            .and_then(|doc| doc.ok());

        let document_data = BASE64_STANDARD
            .decode(value.0.data)
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        let data = aws_smithy_types::Blob::new(document_data);
        let document_source = aws_bedrock::DocumentSource::Bytes(data);

        let result = aws_bedrock::DocumentBlock::builder()
            .source(document_source)
            .name("Document")
            .set_format(format)
            .build()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        Ok(result)
    }
}

impl TryFrom<aws_bedrock::DocumentBlock> for RigDocument {
    type Error = CompletionError;

    fn try_from(value: aws_bedrock::DocumentBlock) -> Result<Self, Self::Error> {
        let media_type = match value.format {
            aws_bedrock::DocumentFormat::Csv => Ok(DocumentMediaType::CSV),
            aws_bedrock::DocumentFormat::Html => Ok(DocumentMediaType::HTML),
            aws_bedrock::DocumentFormat::Md => Ok(DocumentMediaType::MARKDOWN),
            aws_bedrock::DocumentFormat::Pdf => Ok(DocumentMediaType::PDF),
            aws_bedrock::DocumentFormat::Txt => Ok(DocumentMediaType::TXT),
            e => Err(CompletionError::ProviderError(e.to_string())),
        };
        let data = match value.source {
            Some(aws_bedrock::DocumentSource::Bytes(blob)) => {
                let encoded_data = BASE64_STANDARD.encode(blob.into_inner());
                Ok(encoded_data)
            }
            _ => Err(CompletionError::ProviderError(
                "Document source is missing".into(),
            )),
        }?;

        Ok(RigDocument(Document {
            data,
            format: Some(ContentFormat::Base64),
            media_type: media_type.ok(),
        }))
    }
}
