use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{ContentFormat, Document, DocumentMediaType, MimeType},
};

use base64::{prelude::BASE64_STANDARD, Engine};

#[derive(Clone)]
pub struct RigDocument(pub Document);

impl TryFrom<RigDocument> for aws_bedrock::DocumentBlock {
    type Error = CompletionError;

    fn try_from(value: RigDocument) -> Result<Self, Self::Error> {
        let maybe_format = value.0.media_type.map(|doc| match doc {
            DocumentMediaType::PDF => Ok(aws_bedrock::DocumentFormat::Pdf),
            DocumentMediaType::TXT => Ok(aws_bedrock::DocumentFormat::Txt),
            DocumentMediaType::HTML => Ok(aws_bedrock::DocumentFormat::Html),
            DocumentMediaType::MARKDOWN => Ok(aws_bedrock::DocumentFormat::Md),
            DocumentMediaType::CSV => Ok(aws_bedrock::DocumentFormat::Csv),
            e => Err(CompletionError::ProviderError(format!(
                "Unsupported media type {}",
                e.to_mime_type()
            ))),
        });

        let format = match maybe_format {
            Some(Ok(document_format)) => Ok(Some(document_format)),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }?;

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
            e => Err(CompletionError::ProviderError(format!(
                "Unsupported media type {}",
                e
            ))),
        }?;

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
            media_type: Some(media_type),
        }))
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use base64::{prelude::BASE64_STANDARD, Engine};
    use rig::{
        completion::CompletionError,
        message::{ContentFormat, Document, DocumentMediaType},
    };

    use crate::types::document::RigDocument;

    #[test]
    fn test_document_to_aws_document() {
        let rig_document = RigDocument(Document {
            data: "data".into(),
            format: Some(ContentFormat::Base64),
            media_type: Some(DocumentMediaType::PDF),
        });
        let aws_document: Result<aws_bedrock::DocumentBlock, _> = rig_document.clone().try_into();
        assert_eq!(aws_document.is_ok(), true);
        let aws_document = aws_document.unwrap();
        assert_eq!(aws_document.format, aws_bedrock::DocumentFormat::Pdf);
        let document_data = BASE64_STANDARD.decode(rig_document.0.data).unwrap();
        let aws_document_bytes = aws_document
            .source()
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_ref()
            .to_owned();
        assert_eq!(aws_document_bytes, document_data)
    }

    #[test]
    fn test_unsupported_document_to_aws_document() {
        let rig_document = RigDocument(Document {
            data: "data".into(),
            format: Some(ContentFormat::Base64),
            media_type: Some(DocumentMediaType::Javascript),
        });
        let aws_document: Result<aws_bedrock::DocumentBlock, _> = rig_document.clone().try_into();
        assert_eq!(
            aws_document.err().unwrap().to_string(),
            CompletionError::ProviderError(
                "Unsupported media type application/x-javascript".into()
            )
            .to_string()
        )
    }

    #[test]
    fn test_aws_document_to_rig_document() {
        let data = aws_smithy_types::Blob::new("document_data");
        let document_source = aws_bedrock::DocumentSource::Bytes(data);
        let aws_document = aws_bedrock::DocumentBlock::builder()
            .format(aws_bedrock::DocumentFormat::Pdf)
            .name("Document")
            .source(document_source)
            .build()
            .unwrap();
        let rig_document: Result<RigDocument, _> = aws_document.clone().try_into();
        assert_eq!(rig_document.is_ok(), true);
        let rig_document = rig_document.unwrap().0;
        assert_eq!(rig_document.media_type.unwrap(), DocumentMediaType::PDF)
    }

    #[test]
    fn test_unsupported_aws_document_to_rig_document() {
        let data = aws_smithy_types::Blob::new("document_data");
        let document_source = aws_bedrock::DocumentSource::Bytes(data);
        let aws_document = aws_bedrock::DocumentBlock::builder()
            .format(aws_bedrock::DocumentFormat::Xlsx)
            .name("Document")
            .source(document_source)
            .build()
            .unwrap();
        let rig_document: Result<RigDocument, _> = aws_document.clone().try_into();
        assert_eq!(rig_document.is_ok(), false);
        assert_eq!(
            rig_document.err().unwrap().to_string(),
            CompletionError::ProviderError("Unsupported media type xlsx".into()).to_string()
        )
    }
}
