use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::{
    completion::CompletionError,
    message::{Document, DocumentMediaType, DocumentSourceKind},
};

use crate::types::{converse_output::DocumentBlock, document::RigDocument};

/// The inbound path reads the mirror, but what Bedrock sends is the SDK
/// block, so the tests still start there and mirror it first.
fn mirrored(block: aws_bedrock::DocumentBlock) -> DocumentBlock {
    block.try_into().expect("the SDK block mirrors")
}

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
    assert_eq!(aws_document_bytes, document_data);
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
    assert_eq!(aws_document_bytes, document_data);
}

#[test]
fn test_unsupported_document_to_aws_document() {
    let rig_document = RigDocument(Document {
        data: DocumentSourceKind::Base64("data".into()),
        media_type: Some(DocumentMediaType::Javascript),
        additional_params: None,
    });
    let aws_document: Result<aws_bedrock::DocumentBlock, _> = rig_document.try_into();
    assert_eq!(
        aws_document.err().unwrap().to_string(),
        CompletionError::ProviderError("Unsupported media type application/x-javascript".into())
            .to_string()
    );
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
    let rig_document: Result<RigDocument, _> = mirrored(aws_document).try_into();
    assert!(rig_document.is_ok());
    let rig_document = rig_document.unwrap().0;
    assert_eq!(rig_document.media_type.unwrap(), DocumentMediaType::PDF);
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
    let rig_document: Result<RigDocument, _> = mirrored(aws_document).try_into();
    assert!(rig_document.is_err());
    assert_eq!(
        rig_document.err().unwrap().to_string(),
        CompletionError::ProviderError("Unsupported media type xlsx".into()).to_string()
    );
}
