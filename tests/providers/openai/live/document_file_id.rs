//! Live OpenAI coverage for provider file IDs in generic document messages.
//!
//! Run with:
//! `cargo test -p rig --test openai openai::live::document_file_id -- --ignored --nocapture --test-threads=1`

use futures::FutureExt;
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Prompt};
use rig::message::{
    Document, DocumentMediaType, DocumentSourceKind, Message, Text, UserContent as RigUserContent,
};
use rig::providers::openai::{self, FileData, UserContent as OpenAiUserContent};
use serde::Deserialize;
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use crate::support::{
    PDF_FIXTURE_PATH, assert_contains_all_case_insensitive, assert_nonempty_response,
};

const DOCUMENT_PREAMBLE: &str =
    "Answer using only the attached PDF. Keep answers short and quote exact visible text.";

#[derive(Debug, Deserialize)]
struct UploadedFile {
    id: String,
}

fn openai_base_url() -> String {
    std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string())
}

fn openai_api_key() -> String {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY should be set for this ignored test")
}

async fn upload_pdf_for_file_id_test() -> UploadedFile {
    let bytes = tokio::fs::read(PDF_FIXTURE_PATH)
        .await
        .expect("fixture PDF should be readable");
    let file_part = reqwest::multipart::Part::bytes(bytes)
        .file_name("rig-pages.pdf")
        .mime_str("application/pdf")
        .expect("fixture PDF MIME should be valid");
    let form = reqwest::multipart::Form::new()
        .text("purpose", "user_data")
        .text("expires_after[anchor]", "created_at")
        .text("expires_after[seconds]", "3600")
        .part("file", file_part);

    let response = reqwest::Client::new()
        .post(format!("{}/files", openai_base_url()))
        .bearer_auth(openai_api_key())
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
        .delete(format!("{}/files/{file_id}", openai_base_url()))
        .bearer_auth(openai_api_key())
        .send()
        .await;

    if let Ok(response) = response
        && !response.status().is_success()
    {
        eprintln!(
            "cleanup failed for uploaded OpenAI file {file_id}: {}",
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
    let content = OpenAiUserContent::File {
        file: FileData {
            file_data: None,
            file_id: Some(file_id.to_string()),
            filename: Some("rig-pages.pdf".to_string()),
        },
    };
    let content: RigUserContent = content.into();
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

fn assert_page_label(response: &str, page_number: u8) {
    assert_nonempty_response(response);
    assert_contains_all_case_insensitive(response, &["page", &page_number.to_string()]);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_document_file_id_roundtrip_live() {
    with_uploaded_pdf(|file_id| async move {
        let client = openai::Client::from_env().expect("client should build");
        let agent = client
            .agent(openai::GPT_5_5)
            .preamble(DOCUMENT_PREAMBLE)
            .build();
        let mut history = Vec::new();

        let provider_native_content = provider_file_content_as_generic_document(&file_id);
        let response = agent
            .chat(document_question(provider_native_content, 2), &mut history)
            .await
            .expect("Responses API should read uploaded PDF by file_id");
        assert_page_label(&response, 2);
        assert_history_contains_file_id(&history, &file_id);

        let follow_up = agent
            .chat(
                "Using the same PDF from the conversation history, what exact visible text appears on page 3? Reply with only that text.",
                &mut history,
            )
            .await
            .expect("Responses API should reuse file_id document from chat history");
        assert_page_label(&follow_up, 3);
        assert_history_contains_file_id(&history, &file_id);

        let direct_response = agent
            .prompt(direct_file_id_document_question(&file_id, 1))
            .await
            .expect("Responses API should read direct generic file_id document");
        assert_page_label(&direct_response, 1);
    })
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_document_file_id_roundtrip_live() {
    with_uploaded_pdf(|file_id| async move {
        let client = openai::Client::from_env()
            .expect("client should build")
            .completions_api();
        let agent = client
            .agent(openai::GPT_5_5)
            .preamble(DOCUMENT_PREAMBLE)
            .build();
        let mut history = Vec::new();

        let response = agent
            .chat(direct_file_id_document_question(&file_id, 2), &mut history)
            .await
            .expect("Chat Completions API should read uploaded PDF by file_id");
        assert_page_label(&response, 2);
        assert_history_contains_file_id(&history, &file_id);

        let follow_up = agent
            .chat(
                "Using the same PDF from the conversation history, what exact visible text appears on page 3? Reply with only that text.",
                &mut history,
            )
            .await
            .expect("Chat Completions API should reuse file_id document from chat history");
        assert_page_label(&follow_up, 3);
        assert_history_contains_file_id(&history, &file_id);

        let provider_native_content = provider_file_content_as_generic_document(&file_id);
        let native_roundtrip_response = agent
            .prompt(document_question(provider_native_content, 1))
            .await
            .expect("Chat Completions API should read provider-native file_id round-tripped through Rig");
        assert_page_label(&native_roundtrip_response, 1);
    })
    .await;
}
