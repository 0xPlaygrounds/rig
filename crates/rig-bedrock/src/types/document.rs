use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::error::ProviderError;
use rig_core::message::{Document, DocumentMediaType};

use sha2::{Digest, Sha256};

use super::source::{self, Source};

/// The Converse document block for `document`, as base64 data or an S3
/// object. Converse rejects a document's text source, so a string document
/// is an error.
pub(crate) fn to_aws(
    Document {
        data, media_type, ..
    }: Document,
) -> Result<aws_bedrock::DocumentBlock, ProviderError> {
    let format = media_type
        .map(document_format)
        .ok_or_else(|| ProviderError::request("Converse needs a document's format"))?;
    let source = match source::of(data)? {
        Source::Bytes(blob) => aws_bedrock::DocumentSource::Bytes(blob),
        Source::Stored(location) => aws_bedrock::DocumentSource::S3Location(location),
    };
    aws_bedrock::DocumentBlock::builder()
        .name(document_name(&source))
        .source(source)
        .format(format)
        .build()
        .map_err(|e| ProviderError::Provider(e.to_string()))
}

/// The Converse format for `media_type`. A text format Converse does not
/// list is plain text.
fn document_format(media_type: DocumentMediaType) -> aws_bedrock::DocumentFormat {
    match media_type {
        DocumentMediaType::PDF => aws_bedrock::DocumentFormat::Pdf,
        DocumentMediaType::HTML => aws_bedrock::DocumentFormat::Html,
        DocumentMediaType::MARKDOWN => aws_bedrock::DocumentFormat::Md,
        DocumentMediaType::CSV => aws_bedrock::DocumentFormat::Csv,
        DocumentMediaType::TXT
        | DocumentMediaType::RTF
        | DocumentMediaType::CSS
        | DocumentMediaType::XML
        | DocumentMediaType::Javascript
        | DocumentMediaType::Python => aws_bedrock::DocumentFormat::Txt,
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

#[cfg(test)]
mod tests;
