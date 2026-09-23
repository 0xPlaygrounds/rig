//! xAI file-id chain, asserted from the recorded bytes: a PDF is uploaded to
//! the Files API, referenced by its id in Responses content on two turns,
//! then deleted. xAI files do not expire, so the delete runs whether the
//! body passes or panics. The id is account-scoped and deleted by the
//! recording, so the fixture replays but cannot seed a live call.

use std::panic::{AssertUnwindSafe, resume_unwind};

use futures::FutureExt;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{
    AssistantContent, Document, DocumentMediaType, DocumentSourceKind, Message, UserContent,
};
use rig::providers::xai;
use serde_json::{Value, json};

use super::support::with_xai_cassette;

fn request(history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(256),
        tool_choice: None,
        additional_params: Some(json!({ "store": false })),
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn file_id_chain() {
    const SCENARIO: &str = "stateful_chain_matrix/file_id_chain";
    with_xai_cassette("stateful_chain_matrix/file_id_chain", |client| async move {
        let base = client.wire.base_url.trim_end_matches('/').to_owned();
        let key = client.wire.api_key.expose().to_owned();
        let http = reqwest::Client::new();
        let bytes = std::fs::read(crate::support::PDF_FIXTURE_PATH).expect("fixture PDF");
        let form = reqwest::multipart::Form::new()
            .text("purpose", "assistants")
            .part(
                "file",
                reqwest::multipart::Part::bytes(bytes)
                    .file_name("rig-pages.pdf")
                    .mime_str("application/pdf")
                    .expect("PDF MIME"),
            );
        let uploaded: Value = http
            .post(format!("{base}/v1/files"))
            .bearer_auth(&key)
            .multipart(form)
            .send()
            .await
            .expect("upload is sent")
            .json()
            .await
            .expect("upload reply is JSON");
        let file_id = uploaded["id"]
            .as_str()
            .expect("an uploaded file id")
            .to_owned();

        let body = async {
            let document = Message::User {
                content: vec![
                    UserContent::Document(Document {
                        data: DocumentSourceKind::file_id(&file_id),
                        media_type: Some(DocumentMediaType::PDF),
                        additional_params: None,
                    }),
                    UserContent::text("How many pages does this PDF have? Answer with a number."),
                ],
            };
            let model = client.completion(xai::GROK_4);
            let first = model
                .completion(request(vec![document.clone()]))
                .await
                .expect("turn one reads the file by id");
            assert!(text(&first.choice).contains('3'), "{:?}", first.choice);
            let history = vec![
                document,
                Message::Assistant {
                    id: first.message_id.clone(),
                    content: first.choice.clone(),
                },
                Message::user("Is the attached PDF longer than two pages? Answer yes or no."),
            ];
            let second = model
                .completion(request(history))
                .await
                .expect("turn two still reads the file by id");
            assert!(
                text(&second.choice).to_ascii_lowercase().contains("yes"),
                "{:?}",
                second.choice
            );
        };
        let outcome = AssertUnwindSafe(body).catch_unwind().await;
        let deleted = http
            .delete(format!("{base}/v1/files/{file_id}"))
            .bearer_auth(&key)
            .send()
            .await
            .map(|response| response.status());
        if let Err(panic) = outcome {
            eprintln!("cleanup of {file_id}: {deleted:?}");
            resume_unwind(panic);
        }
        assert!(
            deleted.as_ref().is_ok_and(|status| status.is_success()),
            "cleanup failed, delete {file_id} by hand: {deleted:?}"
        );
    })
    .await;

    let paths = crate::cassettes::recorded_request_paths("xai", SCENARIO);
    let bodies = crate::cassettes::recorded_interaction_bodies("xai", SCENARIO);
    assert_eq!(paths.len(), 4, "upload, two turns, delete");
    let upload: Value = serde_json::from_str(&bodies[0].1).expect("upload reply");
    let file_id = upload["id"].as_str().expect("the upload's id");
    assert!(!file_id.contains("REDACTED"));
    for (path, (request, _)) in paths.iter().zip(&bodies).skip(1).take(2) {
        assert!(path.ends_with("/v1/responses"), "{path}");
        let request: Value = serde_json::from_str(request).expect("JSON");
        let referenced: Vec<&str> = request["input"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|item| item["content"].as_array().into_iter().flatten())
            .filter(|part| part["type"] == "input_file")
            .filter_map(|part| part["file_id"].as_str())
            .collect();
        assert_eq!(
            referenced,
            [file_id],
            "the input_file part carries the uploaded id"
        );
    }
    assert!(
        paths[3].ends_with(&format!("/v1/files/{file_id}")),
        "the upload is deleted"
    );
}
