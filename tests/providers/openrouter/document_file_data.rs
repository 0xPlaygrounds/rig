//! OpenRouter wire coverage for PDF `file_data` document messages.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message as RigMessage, Text,
    UserContent as RigUserContent,
};
use rig::providers::openrouter::Message as OpenRouterMessage;
use serde_json::Value;

const VERIFIER_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/file-id-verifiers.pdf"
);
const PAGE_ONE_VERIFIER: &str = "rig-file-id-page-one-verifier-3a91";
const PAGE_TWO_VERIFIER: &str = "rig-file-id-page-two-verifier-8c27";
const PAGE_THREE_VERIFIER: &str = "rig-file-id-page-three-verifier-f54e";
const PAGE_VERIFIERS: [&str; 3] = [PAGE_ONE_VERIFIER, PAGE_TWO_VERIFIER, PAGE_THREE_VERIFIER];

fn verifier_document() -> Document {
    let bytes =
        std::fs::read(VERIFIER_FIXTURE_PATH).expect("verifier fixture PDF should be readable");
    Document {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(bytes)),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    }
}

fn document_question(page_number: u8) -> RigMessage {
    RigMessage::User {
        content: OneOrMany::many(vec![
            RigUserContent::Document(verifier_document()),
            RigUserContent::Text(Text::new(format!(
                "What verifier token is printed on page {page_number}? Reply with only the exact token."
            ))),
        ])
        .expect("content should be non-empty"),
    }
}

fn openrouter_wire_messages(message: RigMessage) -> Vec<Value> {
    let messages: Vec<OpenRouterMessage> = message
        .try_into()
        .expect("generic message should convert to OpenRouter messages");
    messages
        .into_iter()
        .map(|message| serde_json::to_value(message).expect("message should serialize"))
        .collect()
}

fn assert_openrouter_wire_file_data(message: RigMessage) {
    let messages = openrouter_wire_messages(message);
    assert_eq!(messages.len(), 1, "expected one OpenRouter message");

    let json = &messages[0];
    assert_eq!(json["role"], "user");

    let content = json["content"]
        .as_array()
        .unwrap_or_else(|| panic!("expected content array, got {json:#}"));
    assert!(
        !content.is_empty(),
        "expected non-empty content array, got {json:#}"
    );

    let file_blocks = content
        .iter()
        .filter(|block| block["type"] == "file")
        .collect::<Vec<_>>();
    assert_eq!(
        file_blocks.len(),
        1,
        "expected exactly one file block, got {json:#}"
    );

    let file = file_blocks[0]["file"]
        .as_object()
        .unwrap_or_else(|| panic!("expected file object, got {json:#}"));
    assert_eq!(
        file.get("filename").and_then(Value::as_str),
        Some("document.pdf")
    );
    let file_data = file
        .get("file_data")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("expected file_data, got {json:#}"));
    assert!(
        file_data.starts_with("data:application/pdf;base64,"),
        "expected PDF data URI, got {file_data:?}"
    );
    assert!(
        file.get("file_id").is_none(),
        "OpenRouter wire JSON must not contain provider file IDs: {json:#}"
    );
}

fn assert_no_verifier_leaked_into_prompt(message: &RigMessage) {
    let RigMessage::User { content } = message else {
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

#[test]
fn document_file_data_wire_assertions_cover_pdf_prompt() {
    let message = document_question(2);
    assert_no_verifier_leaked_into_prompt(&message);
    assert_openrouter_wire_file_data(message);
}
