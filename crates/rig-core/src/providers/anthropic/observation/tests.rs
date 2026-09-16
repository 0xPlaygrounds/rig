//! The Messages projection, driven through [`crate::driver`].
//!
//! The projector is reached as [`crate::wire::Decoder::project`], which the
//! driver calls on every raw payload — the rejection body included — so
//! these cells exercise the wire and the driver together rather than the
//! projector in isolation. That is the only way the *closure* facts
//! (`Started`, `Response`, `Finished`) are observable at all: they belong to
//! the attempt, not to the payload.

use std::sync::Arc;

use crate::completion::CompletionRequest;
use crate::observe::{
    Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
};
use crate::providers::anthropic::wire::{Anthropic, Messages};
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use futures::StreamExt;

fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
    log.trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect()
}

fn context(log: &Arc<ObservationLog>) -> Option<AdapterContext> {
    Some(AdapterContext::new(log.clone(), Subject::default(), "call"))
}

fn wire() -> Messages {
    Anthropic::new("test-key").messages("claude-test")
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("hello")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// A rejected Messages call: the envelope's type and message, and the
/// closure with the one funnel's classification.
#[tokio::test]
async fn messages_rejection_projects_the_envelope() {
    let http = RecordingHttpClient::with_error(
        http::StatusCode::SERVICE_UNAVAILABLE,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    );
    let log = Arc::new(ObservationLog::default());
    let error = crate::driver::call(&wire(), &http, request(), context(&log))
        .await
        .expect_err("the transport rejects the call");
    assert!(error.is_retryable());
    let events = adapter_events(&log);
    assert!(
        matches!(&events[0], AdapterEvent::Started { method, route } if method == "POST" && route == "/v1/messages")
    );
    assert!(events.contains(&AdapterEvent::Response { status: 503 }));
    assert!(events.contains(&AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code: None,
            status: Some("overloaded_error".into()),
            message: Some("Overloaded".into()),
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(503),
                retryable: true,
            }
        })
    );
}

/// A Messages stream: `message_start` carries the id, the model and the
/// prompt usage; `message_delta` carries the stop reason and the answer's
/// usage; the terminal closes the attempt.
#[tokio::test]
async fn messages_stream_projects_usage_stop_reason_and_model() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":9,\"output_tokens\":1,\"cache_read_input_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let http = MockStreamingClient {
        sse_bytes: bytes::Bytes::from(sse),
    };
    let log = Arc::new(ObservationLog::default());
    let stream = crate::driver::stream(&wire(), &http, request(), context(&log))
        .expect("the streamed request encodes");
    let mut stream = Box::pin(stream);
    while let Some(item) = stream.next().await {
        item.expect("the recorded stream decodes without an in-band error");
    }
    drop(stream);

    let events = adapter_events(&log);
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cached_input_tokens: Some(0),
            ..AdapterUsage::default()
        }
    }));
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            output_tokens: Some(3),
            ..AdapterUsage::default()
        }
    }));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { model: Some(model), finish_reason: None, .. } }
            if model == "claude-sonnet-4-6"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { finish_reason: Some(reason), .. } }
            if reason == "end_turn"
    )));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    );
}
