use serde_json::json;

use super::*;
use crate::providers::gemini::GeminiConfig;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

#[test]
fn a_batch_renders_each_request_as_the_model_would() {
    let model = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3.8-flash");
    let batch = batch(
        &model,
        "nightly",
        [
            CompletionRequest::new("Summarize A"),
            CompletionRequest::new("Summarize B").max_tokens(64),
        ],
    )
    .expect("renders");
    assert_eq!(
        serde_json::to_value(&batch).expect("JSON"),
        json!({
            "displayName": "nightly",
            "model": "models/gemini-3.8-flash",
            "inputConfig": {"requests": {"requests": [
                {"metadata": {"key": "0"}, "request": {"contents": [{"role": "user", "parts": [{"text": "Summarize A"}]}]}},
                {"metadata": {"key": "1"}, "request": {
                    "contents": [{"role": "user", "parts": [{"text": "Summarize B"}]}],
                    "generationConfig": {"maxOutputTokens": 64}
                }}
            ]}}
        })
    );
}

#[tokio::test]
async fn the_verbs_address_the_batch_resource() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(
            r#"{"name":"batches/b1","metadata":{"state":"BATCH_STATE_PENDING"}}"#,
        ),
        MockHttpResponse::success(r#"{"name":"batches/b1","done":true}"#),
        MockHttpResponse::success("{}"),
        MockHttpResponse::success("{}"),
    ]);
    let batches = crate::driver::Model::new(GeminiConfig::new("test-key").batches(), http.clone());
    let model = GenerateContent::new(GeminiConfig::new("test-key"), "gemini-3.8-flash");
    let created = batches
        .create(&model, "nightly", [CompletionRequest::new("Summarize A")])
        .await
        .expect("created");
    assert_eq!(created.name.as_deref(), Some("batches/b1"));
    let done = batches.get("batches/b1").await.expect("read");
    assert_eq!(done.done, Some(true));
    batches.cancel("batches/b1").await.expect("cancelled");
    batches.delete("b1").await.expect("deleted");
    let uris: Vec<_> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert_eq!(
        uris,
        [
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:batchGenerateContent?key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1?key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1:cancel?key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1?key=test-key",
        ]
    );
}
