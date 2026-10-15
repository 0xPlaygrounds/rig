//! Unit tests over replies shaped as a live batch's: a batch takes minutes
//! to hours to end, and its failed items cannot be produced on demand, so a
//! recording cannot cover them. The shapes are those of
//! <https://ai.google.dev/api/batch-api>.

use std::collections::BTreeMap;

use super::*;
use crate::completion::CompletionRequest;
use crate::completion::batch::{Batch, BatchItem};
use crate::message::AssistantContent;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

const MODEL: &str = super::super::completion::GEMINI_2_5_FLASH;

fn operation(state: &str, done: bool) -> String {
    json!({
        "name": "batches/b1",
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
            "model": "models/gemini-2.5-flash",
            "batchStats": { "requestCount": "2", "successfulRequestCount": "1", "failedRequestCount": "1" },
            "state": state,
            "name": "batches/b1",
        },
        "done": done,
        "response": {
            "inlinedResponses": { "inlinedResponses": [
                {
                    "response": {
                        "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Paris" }] }, "finishReason": "STOP", "index": 0 }],
                        "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11 },
                        "modelVersion": "gemini-2.5-flash",
                    },
                    "metadata": { "key": "answer" },
                },
                {
                    "error": { "code": 3, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT" },
                    "metadata": { "key": "broken" },
                },
            ] },
        },
    })
    .to_string()
}

fn model(http: &SequencedHttpClient) -> Model<GenerateContent> {
    GeminiConfig::new("key")
        .connect(http.clone())
        .completion(MODEL)
}

#[tokio::test]
async fn a_batch_sends_its_requests_inline_and_reads_their_replies_back() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(operation("BATCH_STATE_PENDING", false)),
        MockHttpResponse::success(operation("BATCH_STATE_SUCCEEDED", true)),
    ]);
    let model = model(&http);
    let question = CompletionRequest::new("Capital of France?");
    let items = vec![
        BatchItem::new("answer", question.clone()),
        BatchItem::new("broken", question),
    ];
    let batch = model.submit_batch(items).await?;
    let results = model.batch_results(&batch).await?;

    anyhow::ensure!(batch.id == "batches/b1");
    let requests = http.requests();
    let uris: Vec<&str> = requests
        .iter()
        .map(|request| request.uri.as_str())
        .collect();
    anyhow::ensure!(
        uris == [
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:batchGenerateContent?key=key",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1?key=key",
        ]
    );
    let create: Value = serde_json::from_slice(
        &requests
            .first()
            .map_or_else(Default::default, |request| request.body.clone()),
    )?;
    let inline = create
        .at("/batch/inputConfig/requests/requests/1")
        .cloned()
        .unwrap_or_default();
    anyhow::ensure!(inline.at("/metadata/key") == Some(&json!("broken")));
    anyhow::ensure!(inline.at("/request/contents/0/role") == Some(&json!("user")));

    let Some(BatchOutcome::Succeeded(response)) = results.get("answer") else {
        anyhow::bail!("the answer decodes: {results:?}");
    };
    anyhow::ensure!(matches!(
        response.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "Paris"
    ));
    let Some(BatchOutcome::Failed(ProviderError::ProviderResponse(error))) = results.get("broken")
    else {
        anyhow::bail!("the error is the provider's: {results:?}");
    };
    anyhow::ensure!(error.machine_code().as_deref() == Some("INVALID_ARGUMENT"));
    Ok(())
}

#[test]
fn a_batch_reports_its_state_and_counts() -> anyhow::Result<()> {
    let state = |state: &str, done: bool| -> anyhow::Result<BatchState> {
        let text = operation(state, done);
        let operation: BatchOperation = serde_json::from_str(&text)?;
        Ok(operation.state(serde_json::from_str(&text)?)?)
    };
    let running = state("BATCH_STATE_RUNNING", false)?;
    anyhow::ensure!(running.status == BatchStatus::Running);
    anyhow::ensure!(
        running.counts
            == BatchCounts {
                pending: 0,
                succeeded: 1,
                failed: 1
            }
    );
    anyhow::ensure!(state("JOB_STATE_CANCELLED", true)?.status == BatchStatus::Cancelled);
    anyhow::ensure!(state("BATCH_STATE_EXPIRED", true)?.status == BatchStatus::Expired);
    anyhow::ensure!(state("BATCH_STATE_FAILED", true)?.status == BatchStatus::Failed);
    anyhow::ensure!(state("BATCH_STATE_SOMETHING_NEW", false).is_err());
    Ok(())
}

/// Cancelling answers `{}`, so the batch is read back after it.
#[tokio::test]
async fn cancelling_reads_the_batch_back() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success("{}"),
        MockHttpResponse::success(operation("BATCH_STATE_CANCELLED", true)),
    ]);
    let state = model(&http).cancel_batch("batches/b1").await?;
    anyhow::ensure!(state.status == BatchStatus::Cancelled);
    let uris: Vec<String> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    anyhow::ensure!(
        uris == [
            "https://generativelanguage.googleapis.com/v1beta/batches/b1:cancel?key=key",
            "https://generativelanguage.googleapis.com/v1beta/batches/b1?key=key",
        ]
    );
    Ok(())
}

#[tokio::test]
async fn a_batch_addresses_one_model_and_results_wait_for_the_end() -> anyhow::Result<()> {
    let http = SequencedHttpClient::new([MockHttpResponse::success(operation(
        "BATCH_STATE_RUNNING",
        false,
    ))]);
    let model = model(&http);
    let other = CompletionRequest::new("Hi").model("gemini-2.5-pro");
    let mixed = vec![
        BatchItem::new("a", CompletionRequest::new("Hi")),
        BatchItem::new("b", other),
    ];
    anyhow::ensure!(model.submit_batch(mixed).await.is_err());
    let batch = Batch {
        id: "batches/b1".to_owned(),
        requests: BTreeMap::new(),
    };
    anyhow::ensure!(model.batch_results(&batch).await.is_err());
    anyhow::ensure!(http.requests().len() == 1);
    Ok(())
}
