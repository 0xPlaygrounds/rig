//! Unit tests over documented replies: a batch takes minutes to hours to
//! end, and its errored, cancelled and expired items cannot be produced on
//! demand, so a recording cannot cover them. The shapes are those of
//! <https://platform.claude.com/docs/en/api/messages/batches/results>.

use super::*;
use crate::completion::CompletionRequest;
use crate::completion::batch::{BatchItem, BatchStatus};
use crate::message::AssistantContent;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

const MODEL: &str = crate::providers::anthropic::CLAUDE_HAIKU_4_5;

fn batch_object(status: &str, counts: &str, cancel_initiated_at: &str) -> String {
    format!(
        r#"{{"id":"msgbatch_01","type":"message_batch","processing_status":"{status}","request_counts":{counts},"ended_at":null,"created_at":"2026-10-15T10:00:00Z","expires_at":"2026-10-16T10:00:00Z","cancel_initiated_at":{cancel_initiated_at},"results_url":null,"archived_at":null}}"#
    )
}

const RESULTS: &str = concat!(
    r#"{"custom_id":"answer","result":{"type":"succeeded","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[{"type":"text","text":"Paris"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":3}}}}"#,
    "\n",
    r#"{"custom_id":"broken","result":{"type":"errored","error":{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: Field required"},"request_id":"req_02"}}}"#,
    "\n",
    r#"{"custom_id":"late","result":{"type":"expired"}}"#,
    "\n",
    r#"{"custom_id":"dropped","result":{"type":"canceled"}}"#,
    "\n",
);

fn model(http: &SequencedHttpClient) -> Model<Messages> {
    AnthropicConfig::new("key")
        .connect(http.clone())
        .completion(MODEL)
}

/// The body a live unary call sends for `request`, and what its reply,
/// the batch's first, costs.
async fn live_call(request: CompletionRequest) -> anyhow::Result<(Value, Option<f64>)> {
    let message = serde_json::from_str::<Value>(RESULTS.lines().next().unwrap_or_default())?;
    let reply = message.at("/result/message").cloned().unwrap_or_default();
    let http = SequencedHttpClient::new([MockHttpResponse::success(reply.to_string())]);
    let response = model(&http).call(request).await?;
    let sent = http
        .requests()
        .first()
        .map_or_else(Default::default, |request| request.body.clone());
    Ok((
        serde_json::from_slice(&sent)?,
        response.usage.cost.map(|cost| cost.total),
    ))
}

#[tokio::test]
async fn a_batch_sends_each_request_as_a_live_call_would_and_decodes_each_result()
-> anyhow::Result<()> {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(batch_object(
            "in_progress",
            r#"{"processing":4,"succeeded":0,"errored":0,"canceled":0,"expired":0}"#,
            "null",
        )),
        MockHttpResponse::success(RESULTS),
    ]);
    let model = model(&http);
    let question = CompletionRequest::new("Capital of France?").max_tokens(16);
    let items = ["answer", "broken", "late", "dropped"]
        .map(|id| BatchItem::new(id, question.clone()))
        .to_vec();
    let batch = model.submit_batch(items).await?;
    let results = model.batch_results(&batch).await?;

    let requests = http.requests();
    let create = requests.first().map(|request| request.uri.as_str());
    anyhow::ensure!(create == Some("https://api.anthropic.com/v1/messages/batches"));
    let body: Value = serde_json::from_slice(
        &requests
            .first()
            .map_or_else(Default::default, |request| request.body.clone()),
    )?;
    anyhow::ensure!(body.at("/requests/0/custom_id") == Some(&json!("answer")));
    let (live, live_cost) = live_call(question).await?;
    anyhow::ensure!(body.at("/requests/0/params") == Some(&live), "{body}");
    let results_uri = requests.get(1).map(|request| request.uri.as_str());
    anyhow::ensure!(
        results_uri == Some("https://api.anthropic.com/v1/messages/batches/msgbatch_01/results")
    );

    anyhow::ensure!(batch.id == "msgbatch_01");
    let Some(BatchOutcome::Succeeded(response)) = results.get("answer") else {
        anyhow::bail!("the answer decodes: {results:?}");
    };
    anyhow::ensure!(matches!(
        response.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "Paris"
    ));
    let cost = response.usage.cost.map(|cost| cost.total);
    anyhow::ensure!(live_cost.is_some() && cost == live_cost.map(|total| total * 0.5));
    let Some(BatchOutcome::Failed(ProviderError::ProviderResponse(error))) = results.get("broken")
    else {
        anyhow::bail!("the error is the provider's: {results:?}");
    };
    anyhow::ensure!(error.provider_request_id.as_deref() == Some("req_02"));
    anyhow::ensure!(error.machine_code().as_deref() == Some("invalid_request_error"));
    anyhow::ensure!(matches!(results.get("late"), Some(BatchOutcome::Expired)));
    anyhow::ensure!(matches!(
        results.get("dropped"),
        Some(BatchOutcome::Canceled)
    ));
    Ok(())
}

/// The request-level beta flags go on the create call, once each.
#[test]
fn the_create_call_carries_every_beta_flag_its_requests_need() -> anyhow::Result<()> {
    let item = |id: &str, betas: &str| -> anyhow::Result<EncodedItem> {
        Ok(EncodedItem {
            id: id.to_owned(),
            request: http::Request::post("https://api.anthropic.com/v1/messages")
                .header(BETA_HEADER, betas)
                .body(Body::Bytes(br#"{"model":"m"}"#.to_vec()))?,
        })
    };
    let request = create_request(vec![item("a", "one,two")?, item("b", "two, three")?])?;
    let MessageBatchRequest::Create { requests, betas } = request else {
        anyhow::bail!("a create call");
    };
    anyhow::ensure!(betas == ["one", "two", "three"]);
    anyhow::ensure!(requests.get(1) == Some(&json!({"custom_id": "b", "params": {"model": "m"}})));
    Ok(())
}

#[test]
fn an_ended_batch_reports_how_it_ended() -> anyhow::Result<()> {
    let ended = |counts: &str, cancelled: &str| -> anyhow::Result<BatchState> {
        let text = batch_object("ended", counts, cancelled);
        let batch: MessageBatch = serde_json::from_str(&text)?;
        Ok(batch.state(serde_json::from_str(&text)?))
    };
    let all = r#"{"processing":0,"succeeded":2,"errored":1,"canceled":0,"expired":0}"#;
    let completed = ended(all, "null")?;
    anyhow::ensure!(completed.status == BatchStatus::Completed);
    anyhow::ensure!(completed.counts.succeeded == 2 && completed.counts.failed == 1);
    let some_expired = r#"{"processing":0,"succeeded":1,"errored":0,"canceled":0,"expired":1}"#;
    anyhow::ensure!(ended(some_expired, "null")?.status == BatchStatus::Expired);
    let cancelled = ended(all, r#""2026-10-15T10:05:00Z""#)?;
    anyhow::ensure!(cancelled.status == BatchStatus::Cancelled);
    Ok(())
}

#[test]
fn a_messages_format_provider_other_than_anthropic_has_no_batches() {
    let wire =
        AnthropicConfig::with_key(&crate::providers::anthropic::wire::ZAI, "key").completion(MODEL);
    assert!(wire.batches().is_err());
}
