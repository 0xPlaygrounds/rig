//! Streamed server-tool blocks, recorded from the real API.
//!
//! `streaming.rs` keeps every block it has no canonical form for, such as
//! `server_tool_use` and `web_search_tool_result`, as a provider item,
//! assembling a block's `input_json_delta` fragments into its `input`.
//!
//! One recorded streaming web-search turn is asserted against its own
//! fixture's frames so it cannot pass vacuously, plus its blocking twin for
//! parity. Both prove the kept items replay verbatim to the same dialect.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::completion::ProviderToolDefinition;
use rig::message::AssistantContent;
use rig::providers::anthropic::completion::{CLAUDE_OPUS_4_8, DIALECT};
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
use rig::completion::CompletionRequest;

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

/// Each recorded block the stream states whole in its `content_block_start`
/// and never extends with a delta, by index.
fn recorded_whole_blocks(scenario: &str) -> Vec<serde_json::Value> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));
    let frames: Vec<serde_json::Value> = contents
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data: "))
        .filter_map(|frame| serde_json::from_str(frame).ok())
        .collect();
    let extended: Vec<&serde_json::Value> = frames
        .iter()
        .filter(|frame| frame["type"] == "content_block_delta")
        .map(|frame| &frame["index"])
        .collect();
    frames
        .iter()
        .filter(|frame| {
            frame["type"] == "content_block_start" && !extended.contains(&&frame["index"])
        })
        .map(|frame| frame["content_block"].clone())
        .collect()
}

/// The Messages provider item rig kept for a block with no canonical form.
fn native_item(content: &AssistantContent) -> Option<serde_json::Value> {
    let AssistantContent::Native(native) = content else {
        return None;
    };
    native
        .open_native(DIALECT, &[native.issuer().clone()])
        .map(|item| item.item().clone())
}

/// `block` as a replay states it: a `caller` stating the documented
/// default (`direct`) is omitted.
fn as_replayed(block: &serde_json::Value) -> serde_json::Value {
    let mut block = block.clone();
    if block["caller"] == json!({ "type": "direct" })
        && let Some(block) = block.as_object_mut()
    {
        block.remove("caller");
    }
    block
}

/// The raw Anthropic block type rig kept as a provider item.
fn raw_block_type(content: &AssistantContent) -> Option<String> {
    native_item(content)?["type"].as_str().map(str::to_string)
}

/// The assistant blocks a follow-up request replays `choice` as, encoded by
/// `model`'s own wire without sending anything.
fn replayed_blocks(
    model: &rig::driver::Model<rig::providers::anthropic::wire::Messages>,
    choice: Vec<AssistantContent>,
) -> Vec<serde_json::Value> {
    use rig::wire::Wire;
    let request = CompletionRequest::new("Thanks.")
        .messages([
            rig::message::Message::user(WEB_SEARCH_PROMPT),
            rig::message::Message::Assistant {
                id: None,
                content: choice,
            },
        ])
        .max_tokens(16);
    let encoded = model
        .wire
        .encode(request, rig::wire::Mode::Unary)
        .expect("the follow-up encodes");
    let rig::wire::Body::Bytes(body) = encoded.request.body() else {
        panic!("a JSON body");
    };
    let body: serde_json::Value = serde_json::from_slice(body).expect("JSON");
    body["messages"][1]["content"]
        .as_array()
        .cloned()
        .expect("the assistant turn replays")
}

#[tokio::test]
async fn streamed_web_search_preserves_server_tool_blocks() {
    with_anthropic_cassette(
        "streamed_server_tools/streamed_web_search_preserves_server_tool_blocks",
        |client| async move {
            let model = client.completion(CLAUDE_OPUS_4_8);
            let request = CompletionRequest::new(WEB_SEARCH_PROMPT)
                .provider_tool(web_search_tool())
                .max_tokens(1024);

            let mut stream = model
                .stream(request)
                .expect("streaming web-search request should open");

            let mut raw_types = Vec::new();
            let mut natives = Vec::new();
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::End { content, .. }) =
                    item.expect("stream item should not error")
                    && let Some(item) = native_item(&content)
                {
                    raw_types.extend(item["type"].as_str().map(str::to_string));
                    natives.push(item);
                }
            }
            let response = stream
                .finish()
                .await
                .expect("the stream must produce a terminal record");
            // A block the stream stated whole is kept exactly as stated.
            for block in recorded_whole_blocks(
                "streamed_server_tools/streamed_web_search_preserves_server_tool_blocks",
            ) {
                if block["type"] == "web_search_tool_result" {
                    assert!(natives.contains(&block), "{block} kept verbatim");
                }
            }
            // The assembled `server_tool_use` carries the streamed input.
            assert!(
                natives
                    .iter()
                    .any(|item| item["type"] == "server_tool_use" && item["input"]["query"].is_string()),
                "{natives:?}"
            );
            // Same-dialect replay sends every kept item back as it was kept.
            let replayed = replayed_blocks(&model, response.choice);
            for item in &natives {
                assert!(replayed.contains(&as_replayed(item)), "{item} replays verbatim");
            }
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
            let model = client.completion(CLAUDE_OPUS_4_8);
            let request = CompletionRequest::new(WEB_SEARCH_PROMPT)
                .provider_tool(web_search_tool())
                .max_tokens(1024);

            let response = model
                .call(request)
                .await
                .expect("blocking web-search request should succeed");

            let raw_types: Vec<_> = response
                .choice
                .iter()
                .filter_map(raw_block_type)
                .collect();
            // The whole reply's own blocks replay exactly as it stated them.
            let recorded: serde_json::Value = serde_json::from_value(response.raw.clone())
                .expect("the reply is JSON");
            let replayed = replayed_blocks(&model, response.choice.clone());
            for block in recorded["content"].as_array().expect("content") {
                if block["type"] != "text" {
                    assert!(
                        replayed.contains(&as_replayed(block)),
                        "{block} replays verbatim"
                    );
                }
            }

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
