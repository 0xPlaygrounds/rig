use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::{
    completion::CompletionError,
    message::{Document, DocumentSourceKind},
};

pub(crate) use crate::types::media_types::RigDocumentMediaType;
use base64::{Engine, prelude::BASE64_STANDARD};
use uuid::Uuid;

use super::converse_output::{DocumentBlock, DocumentSource};

#[derive(Clone)]
pub struct RigDocument(pub Document);

impl TryFrom<RigDocument> for aws_bedrock::DocumentBlock {
    type Error = CompletionError;

    fn try_from(
        RigDocument(Document {
            data, media_type, ..
        }): RigDocument,
    ) -> Result<Self, Self::Error> {
        let document_media_type = media_type
            .map(|doc| RigDocumentMediaType(doc).try_into())
            .transpose()?;

        let document_source = match data {
            DocumentSourceKind::Base64(blob) => {
                let bytes = BASE64_STANDARD
                    .decode(blob)
                    .map_err(|e| CompletionError::RequestError(e.into()))?;

                aws_bedrock::DocumentSource::Bytes(aws_smithy_types::Blob::new(bytes))
            }
            // NOTE: until [aws-sdk-bedrockruntime DocumentSource bug #1365](https://github.com/awslabs/aws-sdk-rust/issues/1365)
            // is resolved we will use this as a workaround
            // DocumentSourceKind::String(str) => aws_bedrock::DocumentSource::Text(str),
            DocumentSourceKind::String(str) => {
                aws_bedrock::DocumentSource::Bytes(aws_smithy_types::Blob::new(str.as_bytes()))
            }
            doc => {
                return Err(CompletionError::RequestError(
                    format!("Unsupported document kind: {doc}").into(),
                ));
            }
        };

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

impl TryFrom<DocumentBlock> for RigDocument {
    type Error = CompletionError;

    fn try_from(value: DocumentBlock) -> Result<Self, Self::Error> {
        let media_type: RigDocumentMediaType = value.format.try_into()?;
        let media_type = media_type.0;

        let data = match value.source {
            Some(DocumentSource::Bytes(blob)) => {
                let encoded_data = BASE64_STANDARD.encode(blob.inner);
                Ok(DocumentSourceKind::Base64(encoded_data))
            }
            Some(DocumentSource::Text(str)) => Ok(DocumentSourceKind::String(str)),
            doc => Err(CompletionError::ProviderError(format!(
                "Unsupported document type: {doc:?}"
            ))),
        }?;

        Ok(RigDocument(Document {
            data,
            media_type: Some(media_type),
            additional_params: None,
        }))
    }
}

#[cfg(test)]
mod tests;
