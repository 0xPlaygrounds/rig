//! Migrated from `examples/ollama_streaming_pause_control.rs`.

use futures::StreamExt;
use rig::http_runtime::HttpRuntime;
use rig::providers::ollama;
use rig::streaming::StreamedAssistantContent;
use tokio::time::{Duration, sleep};

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn streaming_pause_and_resume() {
    let cfg = ollama::functions::Config::from_env("gemma3:4b").expect("config should build");
    let rt = HttpRuntime::new();
    let request = rig::completion::CompletionRequest {
        temperature: Some(0.7),
        ..rig::completion::CompletionRequest::with_history(
            Some("You are a helpful AI assistant. Provide concise explanations."),
            Vec::new(),
            "Explain backpropagation in neural networks.",
        )
    };
    let mut stream = ollama::functions::open_stream(&cfg, &rt, request)
        .await
        .expect("stream should start");

    let mut chunk_count = 0usize;
    let mut paused_once = false;
    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream chunk should succeed") {
            StreamedAssistantContent::Text(text) => {
                chunk_count += usize::from(!text.text.is_empty());
            }
            StreamedAssistantContent::ToolCall { .. } | StreamedAssistantContent::Reasoning(_) => {
                chunk_count += 1
            }
            StreamedAssistantContent::Final(_) => break,
            _ => {}
        }

        if !paused_once && chunk_count > 0 {
            stream.pause();
            sleep(Duration::from_millis(50)).await;
            stream.resume();
            paused_once = true;
        }
    }

    assert!(paused_once, "expected to exercise pause/resume");
    assert!(
        chunk_count > 0,
        "expected to process at least one stream chunk"
    );
}
