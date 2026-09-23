use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::error::ProviderError;
use rig_core::message::{Document, DocumentSourceKind};

pub(crate) use crate::types::media_types::RigDocumentMediaType;
use base64::{Engine, prelude::BASE64_STANDARD};
use sha2::{Digest, Sha256};

use super::converse_output::{DocumentBlock, DocumentSource};

#[derive(Clone)]
pub struct RigDocument(pub Document);

impl TryFrom<RigDocument> for aws_bedrock::DocumentBlock {
    type Error = ProviderError;

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
                    .map_err(|e| ProviderError::Request(e.into()))?;

                aws_bedrock::DocumentSource::Bytes(aws_smithy_types::Blob::new(bytes))
            }
            // Use the byte source for string documents to avoid SDK text-source
            // compatibility limitations.
            DocumentSourceKind::String(str) => {
                aws_bedrock::DocumentSource::Bytes(aws_smithy_types::Blob::new(str.as_bytes()))
            }
            doc => {
                return Err(ProviderError::Request(
                    format!("Unsupported document kind: {doc}").into(),
                ));
            }
        };

        let document_name = document_name(&document_source);
        let result = aws_bedrock::DocumentBlock::builder()
            .source(document_source)
            .name(document_name)
            .set_format(document_media_type)
            .build()
            .map_err(|e| ProviderError::Provider(e.to_string()))?;
        Ok(result)
    }
}

/// Bedrock requires a name on every document and Rig's `Document` carries
/// none. Naming by content keeps a request byte-stable across turns and runs,
/// which prompt-cache prefixes and recorded replays both rely on; names that
/// repeat within one request are made unique by
/// [`disambiguate_document_names`].
fn document_name(source: &aws_bedrock::DocumentSource) -> String {
    let bytes: &[u8] = match source {
        aws_bedrock::DocumentSource::Bytes(blob) => blob.as_ref(),
        aws_bedrock::DocumentSource::Text(text) => text.as_bytes(),
        aws_bedrock::DocumentSource::S3Location(location) => location.uri().as_bytes(),
        _ => &[],
    };
    let digest = Sha256::digest(bytes);
    let hex: String = digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("document-{hex}")
}

/// Suffix the second and later occurrences of a document name within one
/// request (`document-ab12`, `document-ab12-2`, …), so the same content sent
/// twice stays two distinct, deterministically named documents.
pub(crate) fn disambiguate_document_names(messages: &mut [aws_bedrock::Message]) {
    let mut seen = std::collections::HashMap::<String, usize>::new();
    for message in messages {
        for block in &mut message.content {
            if let aws_bedrock::ContentBlock::Document(document) = block {
                let count = seen.entry(document.name.clone()).or_insert(0);
                *count += 1;
                if *count > 1 {
                    document.name = format!("{}-{count}", document.name);
                }
            }
        }
    }
}

impl TryFrom<DocumentBlock> for RigDocument {
    type Error = ProviderError;

    fn try_from(value: DocumentBlock) -> Result<Self, Self::Error> {
        let media_type: RigDocumentMediaType = value.format.try_into()?;
        let media_type = media_type.0;

        let data = match value.source {
            Some(DocumentSource::Bytes(blob)) => {
                let encoded_data = BASE64_STANDARD.encode(blob.inner);
                Ok(DocumentSourceKind::Base64(encoded_data))
            }
            Some(DocumentSource::Text(str)) => Ok(DocumentSourceKind::String(str)),
            doc => Err(ProviderError::Provider(format!(
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
