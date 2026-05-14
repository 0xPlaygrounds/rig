//! Cassette-backed Anthropic coverage for provider file IDs in generic document messages.

use futures::FutureExt;
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{Chat, Prompt};
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message, Text, UserContent as RigUserContent,
};
use rig::providers::anthropic;
use rig::providers::anthropic::completion::{
    ANTHROPIC_VERSION_2023_06_01, Content as AnthropicContent,
    DocumentSource as AnthropicDocumentSource, Message as AnthropicMessage, Role as AnthropicRole,
};
use rig::streaming::StreamingPrompt;
use serde::Deserialize;
use serde_json::Value;
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use super::super::support::with_anthropic_files_cassette;
use crate::support::{assert_nonempty_response, collect_stream_final_response};

const ANTHROPIC_FILES_BETA: &str = "files-api-2025-04-14";
const DOCUMENT_PREAMBLE: &str =
    "Answer using only the attached PDF. Keep answers short and return exact visible tokens.";
const VERIFIER_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/file-id-verifiers.pdf"
);
const VERIFIER_FIXTURE_FILENAME: &str = "rig-file-id-verifiers.pdf";
const PAGE_ONE_VERIFIER: &str = "rig-file-id-page-one-verifier-3a91";
const PAGE_TWO_VERIFIER: &str = "rig-file-id-page-two-verifier-8c27";
const PAGE_THREE_VERIFIER: &str = "rig-file-id-page-three-verifier-f54e";
const PAGE_VERIFIERS: [&str; 3] = [PAGE_ONE_VERIFIER, PAGE_TWO_VERIFIER, PAGE_THREE_VERIFIER];

#[derive(Debug, Deserialize)]
struct UploadedFile {
    id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    mime_type: String,
    size_bytes: u64,
    downloadable: bool,
}

async fn upload_pdf_for_file_id_test(base_url: &str, api_key: &str) -> UploadedFile {
    let bytes = tokio::fs::read(VERIFIER_FIXTURE_PATH)
        .await
        .expect("verifier fixture PDF should be readable");
    let file_part = reqwest::multipart::Part::bytes(bytes)
        .file_name(VERIFIER_FIXTURE_FILENAME)
        .mime_str("application/pdf")
        .expect("verifier fixture PDF MIME should be valid");
    let form = reqwest::multipart::Form::new().part("file", file_part);

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/files"))
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION_2023_06_01)
        .header("anthropic-beta", ANTHROPIC_FILES_BETA)
        .multipart(form)
        .send()
        .await
        .expect("file upload request should be sent");

    let status = response.status();
    let body = response
        .text()
        .await
        .expect("file upload response body should be readable");
    assert!(
        status.is_success(),
        "file upload failed with {status}: {body}"
    );

    let uploaded: UploadedFile =
        serde_json::from_str(&body).expect("file upload response should deserialize");
    assert_uploaded_file_metadata(&uploaded);
    uploaded
}

fn assert_uploaded_file_metadata(uploaded: &UploadedFile) {
    assert!(
        uploaded.id.starts_with("file_"),
        "expected Anthropic file id to start with file_, got {}",
        uploaded.id
    );
    assert_eq!(uploaded.file_type, "file");
    assert_eq!(uploaded.filename, VERIFIER_FIXTURE_FILENAME);
    assert_eq!(uploaded.mime_type, "application/pdf");
    assert!(
        uploaded.size_bytes > 0,
        "uploaded file size should be nonzero"
    );
    assert!(
        !uploaded.downloadable,
        "user-uploaded Anthropic file should not be downloadable"
    );
}

async fn delete_uploaded_file(base_url: &str, api_key: &str, file_id: &str) {
    let response = reqwest::Client::new()
        .delete(format!("{base_url}/v1/files/{file_id}"))
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION_2023_06_01)
        .header("anthropic-beta", ANTHROPIC_FILES_BETA)
        .send()
        .await;

    if let Ok(response) = response
        && !response.status().is_success()
    {
        eprintln!(
            "cleanup failed for uploaded Anthropic file {file_id}: {}",
            response.status()
        );
    }
}

async fn with_uploaded_pdf<F, Fut>(base_url: &str, api_key: &str, test_body: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    let uploaded = upload_pdf_for_file_id_test(base_url, api_key).await;
    let file_id = uploaded.id;

    let result = AssertUnwindSafe(test_body(file_id.clone()))
        .catch_unwind()
        .await;

    delete_uploaded_file(base_url, api_key, &file_id).await;

    if let Err(payload) = result {
        resume_unwind(payload);
    }
}

fn file_id_document(file_id: &str) -> Document {
    Document {
        data: DocumentSourceKind::file_id(file_id),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    }
}

fn provider_file_content_as_generic_document(file_id: &str) -> RigUserContent {
    let provider_message = AnthropicMessage {
        role: AnthropicRole::User,
        content: OneOrMany::one(AnthropicContent::Document {
            source: AnthropicDocumentSource::File {
                file_id: file_id.to_string(),
            },
            cache_control: None,
        }),
    };
    let generic_message: Message = provider_message
        .try_into()
        .expect("Anthropic file source should convert into generic message");
    let Message::User { content } = generic_message else {
        panic!("expected generic user message");
    };
    let content = content
        .into_iter()
        .next()
        .expect("generic user message should contain document");

    assert_file_id_user_content(&content, file_id);
    content
}

fn document_question(content: RigUserContent, page_number: u8) -> Message {
    Message::User {
        content: OneOrMany::many(vec![
            content,
            RigUserContent::Text(Text {
                text: format!(
                    "What verifier token is printed on page {page_number}? Reply with only the exact token."
                ),
            }),
        ])
        .expect("content should be non-empty"),
    }
}

fn direct_file_id_document_question(file_id: &str, page_number: u8) -> Message {
    document_question(
        RigUserContent::Document(file_id_document(file_id)),
        page_number,
    )
}

fn assert_file_id_user_content(content: &RigUserContent, expected_file_id: &str) {
    let RigUserContent::Document(Document {
        data,
        media_type,
        additional_params,
    }) = content
    else {
        panic!("expected generic document content, got {content:?}");
    };

    assert!(
        matches!(data, DocumentSourceKind::FileId(file_id) if file_id == expected_file_id),
        "expected file ID document source {expected_file_id}, got {data:?}"
    );
    assert_eq!(
        *media_type, None,
        "provider file-id content should not invent a generic media type"
    );
    assert_eq!(
        *additional_params, None,
        "provider file-id content should not invent additional params"
    );
}

fn message_contains_file_id(message: &Message, expected_file_id: &str) -> bool {
    let Message::User { content } = message else {
        return false;
    };

    content.iter().any(|content| {
        matches!(
            content,
            RigUserContent::Document(Document {
                data: DocumentSourceKind::FileId(file_id),
                ..
            }) if file_id == expected_file_id
        )
    })
}

fn assert_history_preserves_single_file_id(history: &[Message], expected_file_id: &str) {
    let mut file_id_message_count = 0;

    for message in history {
        let json = anthropic_wire_json(message.clone());
        assert_no_text_file_id_fallback(&json, expected_file_id);

        if message_contains_file_id(message, expected_file_id) {
            file_id_message_count += 1;
            assert_wire_json_has_exact_file_source(&json, expected_file_id);
        }
    }

    assert_eq!(
        file_id_message_count, 1,
        "expected exactly one history message to preserve document file ID {expected_file_id}: {history:?}"
    );
}

fn anthropic_wire_json(message: Message) -> Value {
    let anthropic_message: AnthropicMessage = message
        .try_into()
        .expect("generic message should convert to Anthropic message");
    serde_json::to_value(&anthropic_message).expect("message should serialize")
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
    assert!(
        response.contains(expected_verifier),
        "expected response to contain verifier {expected_verifier}, got {response:?}"
    );

    for verifier in PAGE_VERIFIERS {
        if verifier != expected_verifier {
            assert!(
                !response.contains(verifier),
                "response included wrong-page verifier {verifier}; response was {response:?}"
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

#[tokio::test]
async fn messages_document_file_id_roundtrip_live() {
    with_anthropic_files_cassette(
        "document_file_id/messages_document_file_id_roundtrip_live",
        ANTHROPIC_FILES_BETA,
        |parts| async move {
            let client = parts.client;
            let base_url = parts.base_url;
            let api_key = parts.api_key;
            with_uploaded_pdf(&base_url, &api_key, |file_id| async move {
                let agent = client
                    .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                    .preamble(DOCUMENT_PREAMBLE)
                    .build();
                let mut history = Vec::new();

                let direct_message = direct_file_id_document_question(&file_id, 2);
                assert_no_verifier_leaked_into_prompt(&direct_message);
                assert_anthropic_wire_file_source(direct_message, &file_id);

                let provider_native_content = provider_file_content_as_generic_document(&file_id);
                let provider_native_roundtrip_message =
                    document_question(provider_native_content, 2);
                assert_no_verifier_leaked_into_prompt(&provider_native_roundtrip_message);
                assert_generic_message_has_file_id(&provider_native_roundtrip_message, &file_id);
                assert_anthropic_wire_file_source(
                    provider_native_roundtrip_message.clone(),
                    &file_id,
                );

                let response = agent
                    .chat(provider_native_roundtrip_message, &mut history)
                    .await
                    .expect("Messages API should read uploaded PDF by file_id");
                assert_verifier_response(&response, PAGE_TWO_VERIFIER);
                assert_history_preserves_single_file_id(&history, &file_id);

                let follow_up = agent
                    .chat(
                        "Using the same PDF from the conversation history, what verifier token is printed on page 3? Reply with only the exact token.",
                        &mut history,
                    )
                    .await
                    .expect("Messages API should reuse file_id document from chat history");
                assert_verifier_response(&follow_up, PAGE_THREE_VERIFIER);
                assert_history_preserves_single_file_id(&history, &file_id);

                let direct_prompt = direct_file_id_document_question(&file_id, 1);
                assert_no_verifier_leaked_into_prompt(&direct_prompt);
                assert_anthropic_wire_file_source(direct_prompt.clone(), &file_id);
                let direct_response = agent
                    .prompt(direct_prompt)
                    .await
                    .expect("Messages API should read direct generic file_id document");
                assert_verifier_response(&direct_response, PAGE_ONE_VERIFIER);
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_document_file_id_roundtrip_live() {
    with_anthropic_files_cassette(
        "document_file_id/streaming_document_file_id_roundtrip_live",
        ANTHROPIC_FILES_BETA,
        |parts| async move {
            let client = parts.client;
            let base_url = parts.base_url;
            let api_key = parts.api_key;
            with_uploaded_pdf(&base_url, &api_key, |file_id| async move {
                let agent = client
                    .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                    .preamble(DOCUMENT_PREAMBLE)
                    .build();

                let stream_prompt = direct_file_id_document_question(&file_id, 2);
                assert_no_verifier_leaked_into_prompt(&stream_prompt);
                assert_anthropic_wire_file_source(stream_prompt.clone(), &file_id);

                let mut stream = agent.stream_prompt(stream_prompt).await;
                let response = collect_stream_final_response(&mut stream)
                    .await
                    .expect("streaming Messages API should read uploaded PDF by file_id");
                assert_verifier_response(&response, PAGE_TWO_VERIFIER);
            })
            .await;
        },
    )
    .await;
}
