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
use rig::completion::ProviderToolDefinition;
use rig::message::{AssistantContent, ProviderItem};
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde_json::{Value, json};

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

/// The type of the Anthropic block rig kept verbatim as a provider part.
fn raw_block_type(content: &AssistantContent) -> Option<String> {
    let AssistantContent::Provider(item) = content else {
        return None;
    };
    let issuer = item.issuer().clone();
    match item.open(&issuer)? {
        ProviderItem::AnthropicMessages(block) => Some(block.kind().to_string()),
        _ => None,
    }
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
            while let Some(item) = stream.next().await {
                if let Item::Event(StreamEvent::End { content, .. }) =
                    item.expect("stream item should not error")
                    && let Some(raw) = raw_block_type(&content)
                {
                    raw_types.push(raw);
                }
            }
            stream
                .finish()
                .await
                .expect("the stream must produce a terminal record");
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

const BLOCKING_SCENARIO: &str =
    "streamed_server_tools/blocking_web_search_preserves_server_tool_blocks";

/// The recorded blocking reply restated as the SSE stream Anthropic sends
/// for it: text arrives by `text_delta` and `citations_delta`, a
/// `server_tool_use` opens with empty input and streams it as
/// `input_json_delta`, and every other block arrives whole on its start.
fn restated_stream(message: &Value) -> Vec<String> {
    let frame = |data: Value| {
        format!(
            "event: {}\ndata: {data}",
            data["type"].as_str().unwrap_or("")
        )
    };
    let mut opening = message.clone();
    opening["content"] = json!([]);
    opening["stop_reason"] = Value::Null;
    let mut frames = vec![frame(json!({"type": "message_start", "message": opening}))];
    for (index, block) in message["content"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let delta = |delta: Value| {
            frame(json!({"type": "content_block_delta", "index": index, "delta": delta}))
        };
        match block["type"].as_str() {
            Some("text") => {
                frames.push(frame(json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}})));
                frames.push(delta(json!({"type": "text_delta", "text": block["text"]})));
                for citation in block["citations"].as_array().into_iter().flatten() {
                    frames.push(delta(
                        json!({"type": "citations_delta", "citation": citation}),
                    ));
                }
            }
            Some("server_tool_use") => {
                let mut opened = block.clone();
                opened["input"] = json!({});
                frames.push(frame(json!({"type": "content_block_start", "index": index,
                    "content_block": opened})));
                frames.push(delta(json!({"type": "input_json_delta",
                    "partial_json": block["input"].to_string()})));
            }
            _ => frames.push(frame(json!({"type": "content_block_start", "index": index,
                "content_block": block}))),
        }
        frames.push(frame(json!({"type": "content_block_stop", "index": index})));
    }
    frames.push(frame(json!({"type": "message_delta",
        "delta": {"stop_reason": message["stop_reason"], "stop_sequence": message["stop_sequence"]},
        "usage": {"output_tokens": message["usage"]["output_tokens"]}})));
    frames.push(frame(json!({"type": "message_stop"})));
    frames
}

/// Stream and whole reply agree by construction: the recorded blocking
/// reply, decoded whole and decoded as the stream that states it, folds to
/// the same choice, provider blocks and citations included. The hosted
/// blocks go through the catch-all arm, not a modeled variant.
#[tokio::test]
async fn the_recorded_reply_folds_identically_whole_and_streamed() {
    let (_, body) = crate::cassettes::recorded_interaction_bodies("anthropic", BLOCKING_SCENARIO)
        .into_iter()
        .next()
        .expect("the blocking scenario records one exchange");
    let message: Value = serde_json::from_str(&body).expect("the body is JSON");
    let request = || {
        CompletionRequest::new(WEB_SEARCH_PROMPT)
            .provider_tool(web_search_tool())
            .max_tokens(1024)
    };

    let unary = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = unary.clone();
    with_anthropic_cassette(
        "streamed_server_tools/blocking_web_search_preserves_server_tool_blocks",
        |client| async move {
            let response = client
                .completion(CLAUDE_OPUS_4_8)
                .call(request())
                .await
                .expect("the blocking reply decodes");
            *sink.lock().expect("lock") = Some(response.choice);
        },
    )
    .await;
    let unary = unary.lock().expect("lock").take().expect("the cell ran");

    let model = rig_test_support::cassette_models::AnthropicModels::new(
        rig::providers::anthropic::AnthropicConfig::new("sk-scripted-parity"),
        crate::stream_faults::scripted(vec![crate::stream_faults::sse_bytes(&restated_stream(
            &message,
        ))]),
    )
    .completion(CLAUDE_OPUS_4_8);
    let mut stream = model.stream(request()).expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("every restated frame decodes");
    }
    let streamed = stream.finish().await.expect("the stream ends").choice;

    let kinds: Vec<String> = unary.iter().filter_map(raw_block_type).collect();
    assert!(
        kinds.iter().any(|kind| kind == "server_tool_use"),
        "{kinds:?}"
    );
    assert!(
        kinds.iter().any(|kind| kind == "web_search_tool_result"),
        "{kinds:?}"
    );
    assert_eq!(
        streamed, unary,
        "one history whichever way the reply arrived"
    );
}

/// The request body `wire` would send for `history`.
fn encoded_body<
    W: rig::wire::Wire<Op = rig::operation::Completion, Payload = rig::wire::Encoded>,
>(
    wire: &W,
    history: Vec<rig::message::Message>,
) -> Value {
    let encoded = wire
        .encode(CompletionRequest::from(history), rig::wire::Mode::Unary)
        .expect("the history encodes");
    let rig::wire::Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("the body is JSON")
}

/// The cross-dialect rule on the recorded hosted-tool reply: its provider
/// blocks replay only to the dialect and service that issued them. Another
/// service on the same dialect (Z.AI) and another dialect (OpenAI Responses)
/// receive the answer text alone, without the blocks and without Anthropic's
/// citations.
#[tokio::test]
async fn recorded_provider_blocks_follow_the_cross_dialect_rule() {
    let unary = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = unary.clone();
    with_anthropic_cassette(
        "streamed_server_tools/blocking_web_search_preserves_server_tool_blocks",
        |client| async move {
            let response = client
                .completion(CLAUDE_OPUS_4_8)
                .call(
                    CompletionRequest::new(WEB_SEARCH_PROMPT)
                        .provider_tool(web_search_tool())
                        .max_tokens(1024),
                )
                .await
                .expect("the blocking reply decodes");
            *sink.lock().expect("lock") = Some(response);
        },
    )
    .await;
    let response = unary.lock().expect("lock").take().expect("the cell ran");
    let mut history = vec![rig::message::Message::user(WEB_SEARCH_PROMPT)];
    history.extend(response.message());
    history.push(rig::message::Message::user("And at night?"));

    let no_transport = || crate::stream_faults::scripted(Vec::new());
    let anthropic = rig_test_support::cassette_models::AnthropicModels::new(
        rig::providers::anthropic::AnthropicConfig::new("sk-offline"),
        no_transport(),
    )
    .completion(CLAUDE_OPUS_4_8);
    let zai = rig_test_support::cassette_models::AnthropicModels::new(
        rig::providers::anthropic::AnthropicConfig::with_key(
            &rig::providers::anthropic::wire::ZAI,
            "sk-offline",
        ),
        no_transport(),
    )
    .completion("glm-5");
    let responses = rig_test_support::cassette_models::OpenAiModels::new(
        rig::providers::openai::OpenAIConfig::new("sk-offline"),
        no_transport(),
    )
    .completion(rig::providers::openai::GPT_5_4_MINI);

    let block_types = |body: &Value| -> Vec<String> {
        body["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|message| message["content"].as_array().into_iter().flatten())
            .filter_map(|block| block["type"].as_str().map(str::to_owned))
            .collect()
    };

    // Same dialect, same service: every block replays, citations included.
    let own = encoded_body(&anthropic.wire, history.clone());
    let own_types = block_types(&own);
    assert!(
        own_types.iter().any(|kind| kind == "server_tool_use"),
        "{own_types:?}"
    );
    assert!(
        own_types
            .iter()
            .any(|kind| kind == "web_search_tool_result"),
        "{own_types:?}"
    );

    // Same dialect, another service: the blocks stay home.
    let other_service = encoded_body(&zai.wire, history.clone());
    let other_types = block_types(&other_service);
    assert!(
        other_types.iter().all(|kind| kind == "text"),
        "only text reaches Z.AI: {other_types:?}"
    );

    // Another dialect: no Anthropic block, key or citation leaks.
    let other_dialect = encoded_body(&responses.wire, history);
    let input = other_dialect["input"].to_string();
    for leak in [
        "server_tool_use",
        "web_search_tool_result",
        "srvtoolu_",
        "encrypted_index",
        "citations",
    ] {
        assert!(
            !input.contains(leak),
            "{leak} leaked into Responses: {input}"
        );
    }
    assert!(
        other_dialect["input"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item["role"] == "assistant")),
        "the answer text still replays: {input}"
    );
}
