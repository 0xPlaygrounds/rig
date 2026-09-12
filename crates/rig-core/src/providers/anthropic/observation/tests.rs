use std::sync::Arc;

use crate::client::CompletionClient;
use crate::completion::CompletionModel as _;
use crate::observe::{
    Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
};
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

/// A rejected Messages call: the envelope's type and message, and the
/// closure with the one funnel's classification.
#[tokio::test]
async fn messages_rejection_projects_the_envelope() {
    let http = RecordingHttpClient::with_error(
        http::StatusCode::SERVICE_UNAVAILABLE,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    );
    let client = crate::providers::anthropic::Client::builder()
        .api_key("test-key")
        .http_client(http)
        .build()
        .unwrap();
    let model = client.completion_model("claude-test");
    let log = Arc::new(ObservationLog::default());
    let error = model
        .completion_with_context(
            model.completion_request("hello").max_tokens(64).build(),
            context(&log),
        )
        .await
        .unwrap_err();
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
    let client = crate::providers::anthropic::Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(sse),
        })
        .build()
        .unwrap();
    let model = client.completion_model("claude-test");
    let log = Arc::new(ObservationLog::default());
    let mut stream = model
        .stream_with_context(
            model.completion_request("hello").max_tokens(64).build(),
            context(&log),
        )
        .await
        .unwrap();
    while let Some(item) = stream.next().await {
        item.unwrap();
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
