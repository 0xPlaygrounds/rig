//! The Responses wire projects its boundary facts into observation.
//!
//! `Decoder::project` is the only path left: the per-request payload observer
//! the client layer attached is gone, so the driver projects each reply
//! payload itself ([`rig_core::driver`]) and the decoder writes through
//! `ObservationSink`. Nothing else in the suite covers *this* wire's
//! projector — `observe/adapter/tests.rs` exercises the seam through gemini —
//! so emptying `responses_api::wire::project_payload` would otherwise cost
//! usage, verdict and response-id telemetry silently, with every test green.
#![allow(clippy::expect_used)]

use std::sync::Arc;

use rig_core::completion::{CompletionModel, CompletionRequest};
use rig_core::driver::Bind;
use rig_core::observe::{
    Action, AdapterContext, AdapterEvent, AdapterObservation, AdapterUsage, ObservationLog, Subject,
};
use rig_core::providers::openai::OpenAI;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

/// One completed unary Responses reply, carrying every count the projection
/// reads: the two totals, the cached input detail and the reasoning detail.
const BODY: &str = r#"{
  "id": "resp_probe_1",
  "object": "response",
  "created_at": 1730000000,
  "model": "gpt-4o-2024-08-06",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "id": "msg_1",
      "role": "assistant",
      "status": "completed",
      "content": [{ "type": "output_text", "text": "hello" }]
    }
  ],
  "usage": {
    "input_tokens": 11,
    "output_tokens": 7,
    "total_tokens": 18,
    "input_tokens_details": { "cached_tokens": 3 },
    "output_tokens_details": { "reasoning_tokens": 5 }
  }
}"#;

#[tokio::test]
async fn a_unary_responses_reply_projects_usage_verdict_and_id() {
    let http = SequencedHttpClient::new(vec![MockHttpResponse::success(BODY)]);
    let model = OpenAI::new("test-key").responses("gpt-4o").bind(http);

    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "projection");
    let request: CompletionRequest = model.completion_request("hi").build();
    let response = model
        .completion_with_context(request, Some(context))
        .await
        .expect("the scripted reply must fold");
    assert!(!response.choice.is_empty(), "the turn must carry content");

    let trace = log.trace();
    let facts: Vec<&AdapterObservation> = trace
        .observations
        .iter()
        .filter_map(|observed| match &observed.action {
            Action::Adapter { observation } => Some(observation),
            _ => None,
        })
        .collect();

    // A present zero is a reported zero and unknown stays unknown, so the
    // usage snapshot is compared whole rather than field by field.
    assert!(
        facts.iter().any(|fact| fact.event
            == AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: Some(11),
                    output_tokens: Some(7),
                    total_tokens: Some(18),
                    cached_input_tokens: Some(3),
                    reasoning_tokens: Some(5),
                    tool_input_tokens: None,
                },
            }),
        "usage must reach observation: {facts:?}"
    );

    // The verdict and the response id travel together: the id rides the
    // observation's analysis, the status and model ride the event.
    assert!(
        facts.iter().any(|fact| matches!(&fact.event,
            AdapterEvent::Provider { verdict }
                if verdict.finish_reason.as_deref() == Some("completed")
                    && verdict.model.as_deref() == Some("gpt-4o-2024-08-06"))
            && fact
                .analysis
                .as_ref()
                .and_then(|analysis| analysis.response_id.as_deref())
                == Some("resp_probe_1")),
        "the verdict and response id must reach observation: {facts:?}"
    );
}
