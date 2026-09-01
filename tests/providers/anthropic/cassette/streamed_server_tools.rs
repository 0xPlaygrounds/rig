//! Streamed server-tool blocks, recorded from the real API.
//!
//! `streaming.rs` models `server_tool_use`, `web_search_tool_result` and
//! `code_execution_tool_result` on `content_block_start`/`content_block_stop`,
//! including assembling a `server_tool_use`'s `input_json_delta` fragments —
//! but no cassette in the suite carried streamed server-tool traffic, so that
//! assembly had only hand-built unit streams behind it. Every recorded
//! `content_block_start` in the tree was `text`, `tool_use`, `thinking` or
//! `redacted_thinking`.
//!
//! This closes that gap: one recorded streaming web-search turn, asserted
//! against its own fixture's frames so it cannot pass vacuously, plus its
//! blocking twin for parity.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::completion::{CompletionModel, ProviderToolDefinition};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
use rig::streaming::RawStreamingChoice;
use serde_json::json;

use super::super::support::with_anthropic_cassette;

const WEB_SEARCH_PROMPT: &str = "Use web search to check the color of a clear daytime sky. Keep the final answer under five words.";

fn web_search_tool() -> ProviderToolDefinition {
    ProviderToolDefinition::new("web_search_20250305").with_config("name", json!("web_search"))
}

/// The `content_block_start` block types a streamed fixture recorded.
///
/// Read back from the fixture so the cell asserts its own premise: if the
/// provider stops emitting server-tool blocks on this prompt, the cell fails
/// instead of silently covering an ordinary text turn.
fn recorded_block_types(scenario: &str) -> Vec<String> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    contents
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data: "))
        // Parse the frame rather than pattern-matching its text: a block's
        // own `"type"` sits behind nested objects (`input`, `caller`) that
        // string splitting walks straight into.
        .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
        .filter(|frame| frame["type"] == "content_block_start")
        .filter_map(|frame| frame["content_block"]["type"].as_str().map(str::to_string))
        .collect()
}

/// The raw Anthropic block type rig preserved on a normalized text part.
fn raw_block_type(content: &AssistantContent) -> Option<String> {
    let AssistantContent::Text(text) = content else {
        return None;
    };
    let params = text.additional_params.as_ref()?;
    let raw = params.get("anthropic_content")?;
    raw.get("type")?.as_str().map(str::to_string)
}

#[tokio::test]
async fn streamed_web_search_preserves_server_tool_blocks() {
    with_anthropic_cassette(
        "streamed_server_tools/streamed_web_search_preserves_server_tool_blocks",
        |client| async move {
            let model = client.completion_model(CLAUDE_OPUS_4_8);
            let request = model
                .completion_request(WEB_SEARCH_PROMPT)
                .provider_tool(web_search_tool())
                .max_tokens(1024)
                .build();

            let mut stream = model
                .raw_stream(request)
                .await
                .expect("streaming web-search request should open");

            let mut raw_types = Vec::new();
            let mut terminal_seen = false;
            while let Some(item) = stream.next().await {
                match item.expect("stream item should not error") {
                    RawStreamingChoice::TextStart {
                        additional_params: Some(params),
                        ..
                    } => {
                        if let Some(raw) = params
                            .get("anthropic_content")
                            .and_then(|raw| raw.get("type"))
                            .and_then(|value| value.as_str())
                        {
                            raw_types.push(raw.to_string());
                        }
                    }
                    RawStreamingChoice::FinalResponse(_) => terminal_seen = true,
                    _ => {}
                }
            }

            assert!(terminal_seen, "the stream must produce a terminal record");
            assert!(
                raw_types.iter().any(|kind| kind == "server_tool_use"),
                "the streamed turn must surface its server_tool_use block, got {raw_types:?}",
            );
            assert!(
                raw_types
                    .iter()
                    .any(|kind| kind == "web_search_tool_result"),
                "the streamed turn must surface its web_search_tool_result block, got {raw_types:?}",
            );
        },
    )
    .await;

    let recorded = recorded_block_types(
        "streamed_server_tools/streamed_web_search_preserves_server_tool_blocks",
    );
    assert!(
        recorded.iter().any(|kind| kind == "server_tool_use"),
        "the recorded stream must contain a server_tool_use content_block_start, got {recorded:?}",
    );
    assert!(
        recorded.iter().any(|kind| kind == "web_search_tool_result"),
        "the recorded stream must contain a web_search_tool_result content_block_start, got {recorded:?}",
    );
}

/// Blocking twin: the same prompt through the unary path must preserve the
/// same server-tool block kinds, so the streamed turn is not carrying less.
#[tokio::test]
async fn blocking_web_search_preserves_server_tool_blocks() {
    with_anthropic_cassette(
        "streamed_server_tools/blocking_web_search_preserves_server_tool_blocks",
        |client| async move {
            let model = client.completion_model(CLAUDE_OPUS_4_8);
            let request = model
                .completion_request(WEB_SEARCH_PROMPT)
                .provider_tool(web_search_tool())
                .max_tokens(1024)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("blocking web-search request should succeed");

            let raw_types: Vec<_> = response
                .choice
                .iter()
                .filter_map(raw_block_type)
                .collect();

            assert!(
                raw_types.iter().any(|kind| kind == "server_tool_use"),
                "the blocking turn must surface its server_tool_use block, got {raw_types:?}",
            );
            assert!(
                raw_types
                    .iter()
                    .any(|kind| kind == "web_search_tool_result"),
                "the blocking turn must surface its web_search_tool_result block, got {raw_types:?}",
            );
        },
    )
    .await;
}
