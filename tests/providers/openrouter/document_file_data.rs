//! Live OpenRouter coverage for PDF `file_data` document messages.
//!
//! Run with:
//! `cargo test -p rig --test openrouter openrouter::document_file_data -- --ignored --nocapture --test-threads=1`

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Prompt};
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message as RigMessage, Text,
    UserContent as RigUserContent,
};
use rig::providers::openrouter::{self, Message as OpenRouterMessage};
use rig::streaming::StreamingPrompt;
use serde_json::Value;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

const DOCUMENT_MODEL: &str = "google/gemini-2.5-flash";
const DOCUMENT_PREAMBLE: &str =
    "Answer using only the attached PDF. Keep answers short and return exact visible tokens.";
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
            RigUserContent::Text(Text {
                text: format!(
                    "What verifier token is printed on page {page_number}? Reply with only the exact token."
                ),
            }),
        ])
        .expect("content should be non-empty"),
    }
}

fn message_contains_base64_document(message: &RigMessage) -> bool {
    let RigMessage::User { content } = message else {
        return false;
    };

    content.iter().any(|content| {
        matches!(
            content,
            RigUserContent::Document(Document {
                data: DocumentSourceKind::Base64(_),
                media_type: Some(DocumentMediaType::PDF),
                ..
            })
        )
    })
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

fn assert_no_file_id_leaked_into_wire(message: RigMessage) {
    let messages = openrouter_wire_messages(message);
    for json in messages {
        let serialized = serde_json::to_string(&json).expect("json should serialize");
        assert!(
            !serialized.contains("file_id"),
            "provider file ID field leaked into OpenRouter JSON: {json:#}"
        );
    }
}

fn assert_history_preserves_single_file_data_document(history: &[RigMessage]) {
    let mut document_message_count = 0;

    for message in history {
        assert_no_file_id_leaked_into_wire(message.clone());
        if message_contains_base64_document(message) {
            document_message_count += 1;
            assert_openrouter_wire_file_data(message.clone());
        }
    }

    assert_eq!(
        document_message_count, 1,
        "expected exactly one history message to preserve the PDF file_data document: {history:?}"
    );
}

fn assert_no_verifier_leaked_into_prompt(message: &RigMessage) {
    let RigMessage::User { content } = message else {
        return;
    };

    for content in content.iter() {
        let RigUserContent::Text(Text { text }) = content else {
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

fn assert_verifier_response(response: &str, expected_verifier: &str) {
    assert_nonempty_response(response);
    let expected_suffix = verifier_suffix(expected_verifier);
    assert!(
        response.contains(expected_verifier) || response.contains(expected_suffix),
        "expected response to contain verifier {expected_verifier} or suffix {expected_suffix}, got {response:?}"
    );

    for verifier in PAGE_VERIFIERS {
        if verifier != expected_verifier {
            let suffix = verifier_suffix(verifier);
            assert!(
                !response.contains(verifier) && !response.contains(suffix),
                "response included wrong-page verifier {verifier} or suffix {suffix}; response was {response:?}"
            );
        }
    }
}

fn verifier_suffix(verifier: &str) -> &str {
    verifier
        .rsplit('-')
        .next()
        .expect("verifier should contain a suffix")
}

#[test]
fn document_file_data_wire_assertions_cover_pdf_prompt() {
    let message = document_question(2);
    assert_no_verifier_leaked_into_prompt(&message);
    assert_openrouter_wire_file_data(message);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn document_file_data_roundtrip_live() {
    let client = openrouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(DOCUMENT_MODEL)
        .preamble(DOCUMENT_PREAMBLE)
        .build();
    let mut history = Vec::new();

    let direct_message = document_question(2);
    assert_no_verifier_leaked_into_prompt(&direct_message);
    assert_openrouter_wire_file_data(direct_message.clone());

    let response = agent
        .chat(direct_message, &mut history)
        .await
        .expect("OpenRouter should read PDF file_data document");
    assert_verifier_response(&response, PAGE_TWO_VERIFIER);
    assert_history_preserves_single_file_data_document(&history);

    let follow_up = agent
        .chat(
            "Using the same PDF from the conversation history, what verifier token is printed on page 3? Reply with only the exact token.",
            &mut history,
        )
        .await
        .expect("OpenRouter should reuse PDF file_data document from chat history");
    assert_verifier_response(&follow_up, PAGE_THREE_VERIFIER);
    assert_history_preserves_single_file_data_document(&history);

    let direct_prompt = document_question(1);
    assert_no_verifier_leaked_into_prompt(&direct_prompt);
    assert_openrouter_wire_file_data(direct_prompt.clone());
    let direct_response = agent
        .prompt(direct_prompt)
        .await
        .expect("OpenRouter should read direct generic PDF file_data document");
    assert_verifier_response(&direct_response, PAGE_ONE_VERIFIER);
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming_document_file_data_roundtrip_live() {
    let client = openrouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(DOCUMENT_MODEL)
        .preamble(DOCUMENT_PREAMBLE)
        .build();

    let stream_prompt = document_question(2);
    assert_no_verifier_leaked_into_prompt(&stream_prompt);
    assert_openrouter_wire_file_data(stream_prompt.clone());

    let mut stream = agent.stream_prompt(stream_prompt).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("OpenRouter streaming should read PDF file_data document");
    assert_verifier_response(&response, PAGE_TWO_VERIFIER);
}
