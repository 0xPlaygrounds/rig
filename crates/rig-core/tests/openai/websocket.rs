//! Migrated from `examples/openai_websocket_mode.rs`.

use anyhow::Result;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::AssistantContent;
use rig_core::providers::openai;
use rig_core::providers::openai::responses_api::streaming::{ItemChunkKind, ResponseChunkKind};
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;

use crate::support::assert_nonempty_response;

fn extract_text(choice: &rig_core::OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_session_roundtrip() -> Result<()> {
    let client = openai::Client::from_env().expect("client should build");
    let model_name = openai::GPT_4O_MINI;
    let model = client.completion_model(model_name);
    let mut session = client.responses_websocket(model_name).await?;

    let warmup_request = model
        .completion_request("You will answer a follow-up question about websocket mode.")
        .preamble("Be precise and concise.".to_string())
        .build();
    let warmup_id = session.warmup(warmup_request).await?;
    anyhow::ensure!(!warmup_id.is_empty(), "warmup should return a response id");

    let request = model
        .completion_request("Explain the benefit of websocket mode in one sentence.")
        .build();
    session.send(request).await?;

    let mut streamed_text = String::new();
    loop {
        match session.next_event().await? {
            ResponsesWebSocketEvent::Item(item) => {
                if let ItemChunkKind::OutputTextDelta(delta) = item.data {
                    streamed_text.push_str(&delta.delta);
                }
            }
            ResponsesWebSocketEvent::Response(chunk) => {
                if matches!(
                    chunk.kind,
                    ResponseChunkKind::ResponseCompleted
                        | ResponseChunkKind::ResponseFailed
                        | ResponseChunkKind::ResponseIncomplete
                ) {
                    break;
                }
            }
            ResponsesWebSocketEvent::Done(_) => {}
            ResponsesWebSocketEvent::Error(error) => {
                return Err(anyhow::anyhow!(error.to_string()));
            }
        }
    }
    assert_nonempty_response(&streamed_text);

    let chained_request = model
        .completion_request("Now restate that as three very short bullet points.")
        .build();
    let response = session.completion(chained_request).await?;
    let text = extract_text(&response.choice);
    assert_nonempty_response(&text);

    session.close().await?;
    Ok(())
}
