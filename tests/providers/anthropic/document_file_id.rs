//! Live Anthropic coverage for provider file IDs in generic document messages.
//!
//! Run with:
//! `cargo test -p rig --test anthropic anthropic::document_file_id -- --ignored --nocapture --test-threads=1`

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
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use crate::support::{
    PDF_FIXTURE_PATH, assert_contains_all_case_insensitive, assert_nonempty_response,
    collect_stream_final_response,
};

const ANTHROPIC_FILES_BETA: &str = "files-api-2025-04-14";
const DOCUMENT_PREAMBLE: &str =
    "Answer using only the attached PDF. Keep answers short and quote exact visible text.";

#[derive(Debug, Deserialize)]
struct UploadedFile {
    id: String,
}

fn anthropic_base_url() -> String {
    let base_url =
        std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".into());
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

fn anthropic_api_key() -> String {
    std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY should be set for this ignored test")
}

fn anthropic_files_client() -> anthropic::Client {
    let mut builder = anthropic::Client::builder()
        .api_key(anthropic_api_key())
        .anthropic_beta(ANTHROPIC_FILES_BETA);

    if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    builder.build().expect("client should build")
}

async fn upload_pdf_for_file_id_test() -> UploadedFile {
    let bytes = tokio::fs::read(PDF_FIXTURE_PATH)
        .await
        .expect("fixture PDF should be readable");
    let file_part = reqwest::multipart::Part::bytes(bytes)
        .file_name("rig-pages.pdf")
        .mime_str("application/pdf")
        .expect("fixture PDF MIME should be valid");
    let form = reqwest::multipart::Form::new().part("file", file_part);

    let response = reqwest::Client::new()
        .post(format!("{}/v1/files", anthropic_base_url()))
        .header("x-api-key", anthropic_api_key())
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

    serde_json::from_str(&body).expect("file upload response should deserialize")
}

async fn delete_uploaded_file(file_id: &str) {
    let response = reqwest::Client::new()
        .delete(format!("{}/v1/files/{file_id}", anthropic_base_url()))
        .header("x-api-key", anthropic_api_key())
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

async fn with_uploaded_pdf<F, Fut>(test_body: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    let uploaded = upload_pdf_for_file_id_test().await;
    let file_id = uploaded.id;

    let result = AssertUnwindSafe(test_body(file_id.clone()))
        .catch_unwind()
        .await;

    delete_uploaded_file(&file_id).await;

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
                    "What exact visible text appears on page {page_number}? Reply with only that text."
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
    let RigUserContent::Document(Document { data, .. }) = content else {
        panic!("expected generic document content, got {content:?}");
    };

    assert!(
        matches!(data, DocumentSourceKind::FileId(file_id) if file_id == expected_file_id),
        "expected file ID document source {expected_file_id}, got {data:?}"
    );
}

fn assert_history_contains_file_id(history: &[Message], expected_file_id: &str) {
    let found = history.iter().any(|message| {
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
    });

    assert!(
        found,
        "expected chat history to preserve document file ID {expected_file_id}: {history:?}"
    );
}

fn assert_anthropic_wire_file_source(message: Message, expected_file_id: &str) {
    let anthropic_message: AnthropicMessage = message
        .try_into()
        .expect("generic file_id document should convert to Anthropic message");
    let json = serde_json::to_value(&anthropic_message).expect("message should serialize");

    assert_eq!(json["content"][0]["type"], "document");
    assert_eq!(json["content"][0]["source"]["type"], "file");
    assert_eq!(json["content"][0]["source"]["file_id"], expected_file_id);
}

fn assert_page_label(response: &str, page_number: u8) {
    assert_nonempty_response(response);
    assert_contains_all_case_insensitive(response, &["page", &page_number.to_string()]);
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn messages_document_file_id_roundtrip_live() {
    with_uploaded_pdf(|file_id| async move {
        let client = anthropic_files_client();
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(DOCUMENT_PREAMBLE)
            .build();
        let mut history = Vec::new();

        assert_anthropic_wire_file_source(direct_file_id_document_question(&file_id, 2), &file_id);

        let provider_native_content = provider_file_content_as_generic_document(&file_id);
        let response = agent
            .chat(document_question(provider_native_content, 2), &mut history)
            .await
            .expect("Messages API should read uploaded PDF by file_id");
        assert_page_label(&response, 2);
        assert_history_contains_file_id(&history, &file_id);

        let follow_up = agent
            .chat(
                "Using the same PDF from the conversation history, what exact visible text appears on page 3? Reply with only that text.",
                &mut history,
            )
            .await
            .expect("Messages API should reuse file_id document from chat history");
        assert_page_label(&follow_up, 3);
        assert_history_contains_file_id(&history, &file_id);

        let direct_response = agent
            .prompt(direct_file_id_document_question(&file_id, 1))
            .await
            .expect("Messages API should read direct generic file_id document");
        assert_page_label(&direct_response, 1);
    })
    .await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming_document_file_id_roundtrip_live() {
    with_uploaded_pdf(|file_id| async move {
        let client = anthropic_files_client();
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(DOCUMENT_PREAMBLE)
            .build();

        let mut stream = agent
            .stream_prompt(direct_file_id_document_question(&file_id, 2))
            .await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming Messages API should read uploaded PDF by file_id");
        assert_page_label(&response, 2);
    })
    .await;
}
