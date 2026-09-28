//! Migrated from `examples/openai_websocket_mode.rs`.

use anyhow::Result;
use futures::StreamExt;
use rig::driver::Model;
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::streaming::{Item, StreamEvent};
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::assert_nonempty_response;
use rig::completion::CompletionRequest;

fn extract_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Warm up, then send only the new input each turn: the chaining opt-in
/// carries the conversation on the connection.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn websocket_chained_roundtrip() -> Result<()> {
    let client = OpenAiModels::from_env().expect("config should build from env");
    let model = client.responses(openai::GPT_4O_MINI);
    let socket = model.responses_websocket().chaining().connect().await?;

    let warmup = Model::new(socket.wire.clone().warmup(), socket.transport.clone());
    let warmup_request =
        CompletionRequest::new("You will answer a follow-up question about websocket mode.")
            .preamble("Be precise and concise.");
    let warmed = warmup.call(warmup_request).await?;
    anyhow::ensure!(
        warmed
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty()),
        "warmup should return a response id"
    );

    let request = CompletionRequest::new("Explain the benefit of websocket mode in one sentence.");
    let mut stream = socket.stream(request)?;
    let mut streamed_text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text, .. }) = item? {
            streamed_text.push_str(&text);
        }
    }
    stream.finish().await?;
    assert_nonempty_response(&streamed_text);

    let chained_request =
        CompletionRequest::new("Now restate that as three very short bullet points.");
    let response = socket.call(chained_request).await?;
    let text = extract_text(&response.choice);
    assert_nonempty_response(&text);

    socket.transport.close().await?;
    Ok(())
}
