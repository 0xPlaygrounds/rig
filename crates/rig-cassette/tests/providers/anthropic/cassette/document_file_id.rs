//! Cassette-backed Anthropic coverage for provider file IDs in generic document messages.

use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message, Text, UserContent as RigUserContent,
};
use rig::providers::anthropic;
use serde_json::Value;

const PAGE_ONE_VERIFIER: &str = "rig-file-id-page-one-verifier-3a91";
const PAGE_TWO_VERIFIER: &str = "rig-file-id-page-two-verifier-8c27";
const PAGE_THREE_VERIFIER: &str = "rig-file-id-page-three-verifier-f54e";
const PAGE_VERIFIERS: [&str; 3] = [PAGE_ONE_VERIFIER, PAGE_TWO_VERIFIER, PAGE_THREE_VERIFIER];

fn file_id_document(file_id: &str) -> Document {
    Document {
        data: DocumentSourceKind::file_id(file_id).into(),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    }
}

/// A document carrying only the provider's file id, no generic media type:
/// the shape a file reference ingested from Anthropic's own wire has.
fn provider_file_content_as_generic_document(file_id: &str) -> RigUserContent {
    RigUserContent::Document(Document {
        data: DocumentSourceKind::file_id(file_id).into(),
        media_type: None,
        additional_params: None,
    })
}

fn document_question(content: RigUserContent, page_number: u8) -> Message {
    Message::User {
        content: vec![
            content,
            RigUserContent::Text(Text::new(format!(
                "What verifier token is printed on page {page_number}? Reply with only the exact token."
            ))),
        ],
    }
}

fn direct_file_id_document_question(file_id: &str, page_number: u8) -> Message {
    document_question(
        RigUserContent::Document(file_id_document(file_id)),
        page_number,
    )
}

fn message_contains_file_id(message: &Message, expected_file_id: &str) -> bool {
    let Message::User { content } = message else {
        return false;
    };

    content.iter().any(|content| {
        matches!(
            content,
            RigUserContent::Document(Document {
                data: rig::message::DocumentData::File(DocumentSourceKind::FileId(file_id)),
                ..
            }) if file_id == expected_file_id
        )
    })
}

/// `message` as the Messages wire encodes it, alone in a request.
fn anthropic_wire_json(message: Message) -> Value {
    use rig::wire::{Body, Mode, Wire};

    let wire = anthropic::Anthropic::new("unused")
        .completion(anthropic::completion::CLAUDE_SONNET_4_6)
        .wire;
    let request = rig::completion::CompletionRequest::from(vec![message]).max_tokens(16);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("generic message should encode for Anthropic");
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("the Messages endpoint takes JSON");
    };
    let body: Value = serde_json::from_slice(bytes).expect("the body is JSON");
    body["messages"][0].clone()
}

fn assert_anthropic_wire_file_source(message: Message, expected_file_id: &str) {
    let json = anthropic_wire_json(message);

    assert_eq!(json["role"], "user");
    assert_wire_json_has_exact_file_source(&json, expected_file_id);
    assert_no_text_file_id_fallback(&json, expected_file_id);
}

fn assert_wire_json_has_exact_file_source(json: &Value, expected_file_id: &str) {
    let content = json["content"]
        .as_array()
        .unwrap_or_else(|| panic!("expected content array, got {json:#}"));
    assert!(
        !content.is_empty(),
        "expected non-empty content array, got {json:#}"
    );

    let document_blocks = content
        .iter()
        .filter(|block| block["type"] == "document")
        .collect::<Vec<_>>();
    assert_eq!(
        document_blocks.len(),
        1,
        "expected exactly one document block, got {json:#}"
    );

    let document = document_blocks[0]
        .as_object()
        .unwrap_or_else(|| panic!("expected document object, got {json:#}"));
    assert_eq!(
        document.len(),
        2,
        "expected document block to contain only type and source, got {json:#}"
    );
    let source = document_blocks[0]["source"]
        .as_object()
        .unwrap_or_else(|| panic!("expected document source object, got {json:#}"));
    assert_eq!(
        source.len(),
        2,
        "expected file source to contain only type and file_id, got {json:#}"
    );
    assert_eq!(source["type"], "file");
    assert_eq!(source["file_id"], expected_file_id);
}

fn assert_no_text_file_id_fallback(json: &Value, expected_file_id: &str) {
    let serialized = serde_json::to_string(json).expect("json should serialize");
    assert!(
        !serialized.contains("[file_id:"),
        "file id fallback text marker leaked into Anthropic JSON: {json:#}"
    );

    let Some(content) = json["content"].as_array() else {
        return;
    };

    for block in content {
        if block["type"] == "text" {
            let text = block["text"].as_str().unwrap_or_default();
            assert!(
                !text.contains(expected_file_id),
                "file id appeared in a text block instead of document source: {json:#}"
            );
        }
    }
}

fn assert_no_verifier_leaked_into_prompt(message: &Message) {
    let Message::User { content } = message else {
        return;
    };

    for content in content.iter() {
        let RigUserContent::Text(Text { text, .. }) = content else {
            continue;
        };

        for verifier in PAGE_VERIFIERS {
            assert!(
                !text.contains(verifier),
                "prompt text leaked verifier {verifier}: {text}"
            );
        }
    }
}

fn assert_generic_message_has_file_id(message: &Message, expected_file_id: &str) {
    assert!(
        message_contains_file_id(message, expected_file_id),
        "expected generic message to preserve document file ID {expected_file_id}: {message:?}"
    );
}

#[test]
fn document_file_id_wire_assertions_cover_roundtrip_paths() {
    let file_id = "file_test";

    let direct_message = direct_file_id_document_question(file_id, 2);
    assert_no_verifier_leaked_into_prompt(&direct_message);
    assert_anthropic_wire_file_source(direct_message, file_id);

    let provider_native_content = provider_file_content_as_generic_document(file_id);
    let provider_native_roundtrip_message = document_question(provider_native_content, 2);
    assert_no_verifier_leaked_into_prompt(&provider_native_roundtrip_message);
    assert_generic_message_has_file_id(&provider_native_roundtrip_message, file_id);
    assert_anthropic_wire_file_source(provider_native_roundtrip_message, file_id);
}
