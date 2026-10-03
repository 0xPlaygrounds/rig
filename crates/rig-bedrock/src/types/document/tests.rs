use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::message::{Document, DocumentMediaType, DocumentSourceKind};

use crate::types::document;

#[test]
fn test_document_to_aws_document() {
    let rig_document = Document {
        data: DocumentSourceKind::Base64("data".into()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    };

    let aws_document: Result<aws_bedrock::DocumentBlock, _> =
        document::to_aws(rig_document.clone());
    assert!(aws_document.is_ok());

    let aws_document = aws_document.unwrap();
    assert_eq!(aws_document.format, aws_bedrock::DocumentFormat::Pdf);

    let document_data = rig_document
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
    let rig_document = Document {
        data: DocumentSourceKind::Base64("data".into()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    };

    let aws_document: aws_bedrock::DocumentBlock = document::to_aws(rig_document.clone()).unwrap();
    let document_data = BASE64_STANDARD
        .decode(rig_document.data.try_into_inner().unwrap())
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

/// A text format Converse does not list goes as plain text; a string
/// document is refused, since Converse rejects a document's text source.
#[test]
fn unlisted_text_formats_go_as_txt() {
    let document = |data, media_type| Document {
        data,
        media_type: Some(media_type),
        additional_params: None,
    };
    let script = document::to_aws(document(
        DocumentSourceKind::Base64("bGV0IGEgPSAxOw==".into()),
        DocumentMediaType::Javascript,
    ))
    .unwrap();
    assert_eq!(script.format, aws_bedrock::DocumentFormat::Txt);
    let text = document(
        DocumentSourceKind::String("notes".into()),
        DocumentMediaType::TXT,
    );
    assert!(document::to_aws(text).is_err());
}
