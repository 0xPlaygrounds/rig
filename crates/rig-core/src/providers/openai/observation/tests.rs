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

/// A rejected Chat Completions call: the envelope, the usage the provider
/// billed anyway, and the closure with the one funnel's classification.
#[tokio::test]
async fn chat_completions_rejection_projects_envelope_and_usage() {
    let http = RecordingHttpClient::with_error(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"},"usage":{"prompt_tokens":7,"completion_tokens":0,"total_tokens":7}}"#,
    );
    let client = crate::providers::openai::Client::builder()
        .api_key("test-key")
        .http_client(http)
        .build()
        .unwrap()
        .completions_api();
    let model = client.completion_model("gpt-4o");
    let log = Arc::new(ObservationLog::default());
    let error = model
        .completion_with_context(model.completion_request("hello").build(), context(&log))
        .await
        .unwrap_err();
    assert!(error.is_retryable());

    let events = adapter_events(&log);
    assert!(
        matches!(&events[0], AdapterEvent::Started { method, route } if method == "POST" && route == "/chat/completions")
    );
    assert!(events.contains(&AdapterEvent::Response { status: 429 }));
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            input_tokens: Some(7),
            output_tokens: Some(0),
            total_tokens: Some(7),
            ..AdapterUsage::default()
        }
    }));
    assert!(events.contains(&AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code: Some("rate_limit_exceeded".into()),
            status: Some("rate_limit_error".into()),
            message: Some("slow down".into()),
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(429),
                retryable: true,
            }
        })
    );
}

/// A Chat Completions stream: the verdict and the usage ride on the last
/// chunks, and `[DONE]` is not a payload.
#[tokio::test]
async fn chat_completions_stream_projects_finish_reason_model_and_usage() {
    let sse = "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6,\"completion_tokens_details\":{\"reasoning_tokens\":0}}}\n\n\
data: [DONE]\n\n";
    let client = crate::providers::openai::Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(sse),
        })
        .build()
        .unwrap()
        .completions_api();
    let model = client.completion_model("gpt-4o");
    let log = Arc::new(ObservationLog::default());
    let mut stream = model
        .stream_with_context(model.completion_request("hello").build(), context(&log))
        .await
        .unwrap();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }
    drop(stream);

    let events = adapter_events(&log);
    assert!(events.contains(&AdapterEvent::Response { status: 200 }));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { finish_reason: Some(reason), model: Some(model), .. } }
            if reason == "stop" && model == "gpt-4o-2024-08-06"
    )));
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            input_tokens: Some(5),
            output_tokens: Some(1),
            total_tokens: Some(6),
            reasoning_tokens: Some(0),
            ..AdapterUsage::default()
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    );
}

/// A Responses stream: the completed event carries the response object with
/// its status, model and usage; an `error` event carries the envelope.
#[tokio::test]
async fn responses_stream_projects_status_usage_and_error_events() {
    let completed = "data: {\"type\":\"response.output_text.delta\",\"content_index\":0,\"delta\":\"hi\",\"item_id\":\"msg_1\",\"output_index\":0,\"sequence_number\":1}\n\n\
data: {\"type\":\"response.completed\",\"sequence_number\":99,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":0,\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\",\"annotations\":[]}]}],\"tools\":[],\"usage\":{\"input_tokens\":4,\"output_tokens\":1,\"total_tokens\":5,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens_details\":{\"reasoning_tokens\":0}}}}\n\n";
    let log = Arc::new(ObservationLog::default());
    {
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(completed),
            })
            .build()
            .unwrap();
        let model = client.completion_model("gpt-5.4");
        let mut stream = model
            .stream_with_context(model.completion_request("hello").build(), context(&log))
            .await
            .unwrap();
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
    }
    let events = adapter_events(&log);
    assert!(matches!(&events[0], AdapterEvent::Started { route, .. } if route == "/responses"));
    assert!(events.iter().any(|event| matches!(
        event,
        AdapterEvent::Provider { verdict: AdapterVerdict { finish_reason: Some(reason), model: Some(model), .. } }
            if reason == "completed" && model == "gpt-5.4"
    )));
    assert!(events.contains(&AdapterEvent::Usage {
        usage: AdapterUsage {
            input_tokens: Some(4),
            output_tokens: Some(1),
            total_tokens: Some(5),
            cached_input_tokens: Some(0),
            reasoning_tokens: Some(0),
            ..AdapterUsage::default()
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    );

    let failed = "data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"boom\",\"param\":null,\"sequence_number\":1}\n\n";
    let log = Arc::new(ObservationLog::default());
    {
        let client = crate::providers::openai::Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(failed),
            })
            .build()
            .unwrap();
        let model = client.completion_model("gpt-5.4");
        let mut stream = model
            .stream_with_context(model.completion_request("hello").build(), context(&log))
            .await
            .unwrap();
        let mut errors = 0;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                errors += 1;
            }
        }
        assert_eq!(errors, 1);
    }
    let events = adapter_events(&log);
    assert!(events.contains(&AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code: Some("server_error".into()),
            status: None,
            message: Some("boom".into()),
        }
    }));
}

/// A rejected Responses call: the unary envelope, projected off the reply.
#[tokio::test]
async fn responses_rejection_projects_the_envelope() {
    let http = RecordingHttpClient::with_error(
        http::StatusCode::BAD_REQUEST,
        r#"{"error":{"message":"The requested model does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#,
    );
    let client = crate::providers::openai::Client::builder()
        .api_key("test-key")
        .http_client(http)
        .build()
        .unwrap();
    let model = client.completion_model("gpt-nonexistent");
    let log = Arc::new(ObservationLog::default());
    let error = model
        .completion_with_context(model.completion_request("hello").build(), context(&log))
        .await
        .unwrap_err();
    assert!(!error.is_retryable());
    let events = adapter_events(&log);
    assert!(events.contains(&AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code: Some("model_not_found".into()),
            status: Some("invalid_request_error".into()),
            message: Some("The requested model does not exist.".into()),
        }
    }));
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(400),
                retryable: false,
            }
        })
    );
}
