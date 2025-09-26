use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig::{
    completion::CompletionError,
    message::{Document, DocumentSourceKind},
};

pub(crate) use crate::types::media_types::RigDocumentMediaType;
use base64::{Engine, prelude::BASE64_STANDARD};
use uuid::Uuid;

#[derive(Clone)]
pub struct RigDocument(pub Document);

impl TryFrom<RigDocument> for aws_bedrock::DocumentBlock {
    type Error = CompletionError;

    fn try_from(
        RigDocument(Document {
            data, media_type, ..
        }): RigDocument,
    ) -> Result<Self, Self::Error> {
        let document_media_type = media_type.map(|doc| RigDocumentMediaType(doc).try_into());

        let document_media_type = match document_media_type {
            Some(Ok(document_format)) => Ok(Some(document_format)),
            Some(Err(err)) => Err(err),
            None => Ok(None),
        }?;

        let data = match data {
            DocumentSourceKind::Base64(blob) => BASE64_STANDARD
                .decode(blob)
                .map_err(|e| CompletionError::RequestError(e.into()))?,
            DocumentSourceKind::Raw(bytes) => bytes,
            doc => {
                return Err(CompletionError::RequestError(
                    format!("Unsupported document kind: {doc}").into(),
                ));
            }
        };

        let data = aws_smithy_types::Blob::new(data);
        let document_source = aws_bedrock::DocumentSource::Bytes(data);

        let random_string = Uuid::new_v4().simple().to_string();
        let document_name = format!("document-{random_string}");
        let result = aws_bedrock::DocumentBlock::builder()
            .source(document_source)
            .name(document_name)
            .set_format(document_media_type)
            .build()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        Ok(result)
    }
}

impl TryFrom<aws_bedrock::DocumentBlock> for RigDocument {
    type Error = CompletionError;

    fn try_from(value: aws_bedrock::DocumentBlock) -> Result<Self, Self::Error> {
        let media_type: RigDocumentMediaType = value.format.try_into()?;
        let media_type = media_type.0;

        let data = match value.source {
            Some(aws_bedrock::DocumentSource::Bytes(blob)) => {
                let encoded_data = BASE64_STANDARD.encode(blob.into_inner());
                Ok(DocumentSourceKind::Base64(encoded_data))
            }
            _ => Err(CompletionError::ProviderError(
                "Document source is missing".into(),
            )),
        }?;

        Ok(RigDocument(Document {
            data,
            media_type: Some(media_type),
            additional_params: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use base64::{Engine, prelude::BASE64_STANDARD};
    use rig::{
        completion::CompletionError,
        message::{Document, DocumentMediaType, DocumentSourceKind},
    };

    use crate::types::document::RigDocument;

    #[test]
    fn test_document_to_aws_document() {
        let rig_document = RigDocument(Document {
            data: DocumentSourceKind::Base64("data".into()),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        });

        let aws_document: Result<aws_bedrock::DocumentBlock, _> = rig_document.clone().try_into();
        assert!(aws_document.is_ok());

        let aws_document = aws_document.unwrap();
        assert_eq!(aws_document.format, aws_bedrock::DocumentFormat::Pdf);

        let document_data = rig_document
            .0
            .data
            .try_into_inner()
            .unwrap()
            .as_bytes()
            .to_vec();

        let document_data = BASE64_STANDARD.decode(document_data).unwrap();

        let aws_document_bytes = aws_document
            .source()
            .unwrap()
            .as_bytes()
            .unwrap()
            .as_ref()
            .to_owned();

        let doc_name = aws_document.name;
        assert!(doc_name.starts_with("document-"));
        assert_eq!(aws_document_bytes, document_data)
    }

    #[test]
    fn test_base64_document_to_aws_document() {
        let rig_document = RigDocument(Document {
            data: DocumentSourceKind::Base64("data".into()),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        });

        let aws_document: aws_bedrock::DocumentBlock = rig_document.clone().try_into().unwrap();
        let document_data = BASE64_STANDARD
            .decode(rig_document.0.data.try_into_inner().unwrap())
            .unwrap();
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
            data: DocumentSourceKind::Base64("data".into()),
            media_type: Some(DocumentMediaType::Javascript),
            additional_params: None,
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
        assert!(rig_document.is_ok());
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
        assert!(rig_document.is_err());
        assert_eq!(
            rig_document.err().unwrap().to_string(),
            CompletionError::ProviderError("Unsupported media type xlsx".into()).to_string()
        )
    }
}
