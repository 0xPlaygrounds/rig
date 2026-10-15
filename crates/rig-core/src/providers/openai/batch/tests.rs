//! Unit tests over documented replies: a batch takes minutes to hours to
//! end, and its failed and expired lines cannot be produced on demand, so a
//! recording cannot cover them. The shapes are those of
//! <https://developers.openai.com/api/docs/guides/batch>.

use std::collections::BTreeMap;

use super::*;
use crate::completion::CompletionRequest;
use crate::completion::batch::{Batch, BatchItem};
use crate::json_utils::Lenient;
use crate::message::AssistantContent;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

const MODEL: &str = crate::providers::openai::GPT_4_1_NANO;

fn batch_object(status: &str) -> String {
    format!(
        r#"{{"id":"batch_01","object":"batch","endpoint":"/v1/chat/completions","errors":null,"input_file_id":"file-in","completion_window":"24h","status":"{status}","output_file_id":"file-out","error_file_id":"file-err","created_at":1,"request_counts":{{"total":3,"completed":1,"failed":2}}}}"#
    )
}

const OUTPUT: &str = r#"{"id":"batch_req_1","custom_id":"answer","response":{"status_code":200,"request_id":"req_1","body":{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4.1-nano","choices":[{"index":0,"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}},"error":null}
"#;

const ERRORS: &str = concat!(
    r#"{"id":"batch_req_2","custom_id":"late","response":null,"error":{"code":"batch_expired","message":"This request could not be executed before the completion window expired."}}"#,
    "\n",
    r#"{"id":"batch_req_3","custom_id":"broken","response":{"status_code":400,"request_id":"req_3","body":{"error":{"message":"Invalid model","type":"invalid_request_error","code":"model_not_found"}}},"error":null}"#,
    "\n",
);

#[tokio::test]
async fn a_batch_uploads_its_requests_then_creates_the_batch_over_them() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(
            r#"{"id":"file-in","object":"file","bytes":10,"created_at":1,"filename":"batch.jsonl","purpose":"batch"}"#,
        ),
        MockHttpResponse::success(batch_object("validating")),
    ]);
    let model = OpenAIConfig::new("key")
        .connect(http.clone())
        .completion(MODEL);
    let item = BatchItem::new("answer", CompletionRequest::new("Capital of France?"));
    let batch = model.submit_batch(vec![item]).await?;

    anyhow::ensure!(batch.id == "batch_01");
    let requests = http.requests();
    let uris: Vec<&str> = requests
        .iter()
        .map(|request| request.uri.as_str())
        .collect();
    anyhow::ensure!(
        uris == [
            "https://api.openai.com/v1/files",
            "https://api.openai.com/v1/batches"
        ]
    );
    let create: Value = serde_json::from_slice(
        &requests
            .get(1)
            .map_or_else(Default::default, |request| request.body.clone()),
    )?;
    anyhow::ensure!(
        create
            == json!({
                "input_file_id": "file-in",
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            })
    );
    Ok(())
}

/// Each line is the live request's body under its endpoint, and a batch
/// keeps to one endpoint and one model.
#[test]
fn the_input_file_has_a_line_per_request() -> anyhow::Result<()> {
    let item = |id: &str, path: &str, model: &str| -> anyhow::Result<EncodedItem> {
        Ok(EncodedItem {
            id: id.to_owned(),
            request: http::Request::post(format!("https://api.openai.com/v1{path}"))
                .body(Body::Bytes(serde_json::to_vec(&json!({ "model": model }))?))?,
        })
    };
    let (lines, endpoint) = input_file(vec![
        item("a", "/chat/completions", "m")?,
        item("b", "/chat/completions", "m")?,
    ])?;
    anyhow::ensure!(endpoint == "/v1/chat/completions");
    let line: Value =
        serde_json::from_str(String::from_utf8(lines)?.lines().nth(1).unwrap_or_default())?;
    anyhow::ensure!(
        line == json!({
            "custom_id": "b",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": { "model": "m" },
        })
    );
    let mixed = input_file(vec![
        item("a", "/chat/completions", "m")?,
        item("b", "/chat/completions", "n")?,
    ]);
    anyhow::ensure!(mixed.is_err());
    anyhow::ensure!(input_file(vec![item("a", "/embeddings", "m")?]).is_err());
    Ok(())
}

#[tokio::test]
async fn results_read_the_output_and_error_files() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(batch_object("expired")),
        MockHttpResponse::success(OUTPUT),
        MockHttpResponse::success(ERRORS),
    ]);
    let model = OpenAIConfig::new("key").connect(http.clone()).chat(MODEL);
    let question = CompletionRequest::new("Capital of France?");
    let batch = Batch {
        id: "batch_01".to_owned(),
        requests: BTreeMap::from(
            ["answer", "late", "broken"].map(|id| (id.to_owned(), question.clone())),
        ),
    };
    let results = model.batch_results(&batch).await?;

    let uris: Vec<String> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    anyhow::ensure!(
        uris == [
            "https://api.openai.com/v1/batches/batch_01",
            "https://api.openai.com/v1/files/file-out/content",
            "https://api.openai.com/v1/files/file-err/content",
        ]
    );
    let Some(BatchOutcome::Succeeded(response)) = results.get("answer") else {
        anyhow::bail!("the answer decodes: {results:?}");
    };
    anyhow::ensure!(matches!(
        response.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "Paris"
    ));
    anyhow::ensure!(response.provider_request_id.as_deref() == Some("req_1"));
    anyhow::ensure!(matches!(results.get("late"), Some(BatchOutcome::Expired)));
    let Some(BatchOutcome::Failed(ProviderError::ProviderResponse(error))) = results.get("broken")
    else {
        anyhow::bail!("the error is the provider's: {results:?}");
    };
    anyhow::ensure!(error.status == Some(http::StatusCode::BAD_REQUEST));
    anyhow::ensure!(error.provider_request_id.as_deref() == Some("req_3"));
    Ok(())
}

#[tokio::test]
async fn results_of_a_running_batch_are_not_ready() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([MockHttpResponse::success(batch_object("in_progress"))]);
    let model = OpenAIConfig::new("key").connect(http.clone()).chat(MODEL);
    let state = model.batch_status("batch_01").await?;
    anyhow::ensure!(state.counts.pending == 0 && state.counts.failed == 2);
    anyhow::ensure!(state.raw.str("id") == Some("batch_01"));
    let http = SequencedHttpClient::new([MockHttpResponse::success(batch_object("in_progress"))]);
    let model = OpenAIConfig::new("key").connect(http).chat(MODEL);
    let batch = Batch {
        id: "batch_01".to_owned(),
        requests: BTreeMap::new(),
    };
    anyhow::ensure!(model.batch_results(&batch).await.is_err());
    Ok(())
}

#[test]
fn only_openai_itself_has_batches() {
    let groq = OpenAIConfig::with_key(&crate::providers::openai::wire::GROQ, "key").chat(MODEL);
    assert!(groq.batches().is_err());
}
