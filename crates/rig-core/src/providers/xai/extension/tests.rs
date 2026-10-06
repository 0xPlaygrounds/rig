//! xAI's options as the bodies they encode to on both routes, and its
//! extras. The encoder tests are unit tests because no recording sends
//! typed provider options. Every xAI recording is on Responses, so the
//! extras tests decode those, and a reply built here stands in for Chat.

use serde_json::json;

use super::*;
use crate::providers::openai::wire::OpenAIConfig;
use crate::providers::xai::DIALECT;
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, chat_reply, recorded_reply, reply_of,
};

const MODEL: &str = "grok-4.3";

fn config() -> OpenAIConfig {
    OpenAIConfig::with_key(&DIALECT, "key")
}

#[test]
fn prompt_cache_key_lands_on_chat() {
    let body = body_with::<Xai, _>(
        &config().chat(MODEL),
        &XaiOptions::new().prompt_cache_key("k"),
    );
    assert_eq!(body["prompt_cache_key"], "k");
}

#[test]
fn prompt_cache_key_lands_on_responses() {
    let body = body_with::<Xai, _>(
        &config().responses(MODEL),
        &XaiOptions::new().prompt_cache_key("k"),
    );
    assert_eq!(body["prompt_cache_key"], "k");
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = XaiOptions::new().prompt_cache_key("k");
    assert_no_reserved_leaf::<Xai, _>(&[config().chat(MODEL), config().chat("grok-3")], &options);
    assert_no_reserved_leaf::<Xai, _>(&[config().responses(MODEL)], &options);
}

#[tokio::test]
async fn responses_extras_from_unary_recordings() {
    let reply = reply_of(
        config().responses(MODEL),
        recorded_reply("xai", "agent/completion_smoke", 0),
    )
    .await;
    let extras = reply
        .extras::<Xai>()
        .unwrap_or_else(|| panic!("an xAI reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.cost_in_usd_ticks, Some(6_734_000));
    assert_eq!(extras.num_sources_used, Some(0));
    assert_eq!(extras.num_server_side_tools_used, Some(0));

    let reply = reply_of(
        config().responses(MODEL),
        recorded_reply("xai", "web_search_citations/streamed_and_unary", 1),
    )
    .await;
    let extras = reply
        .extras::<Xai>()
        .unwrap_or_else(|| panic!("an xAI reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.num_server_side_tools_used, Some(2));
    assert_eq!(
        extras
            .server_side_tool_usage_details
            .as_ref()
            .and_then(|details| details.get("web_search_calls")),
        Some(&json!(2))
    );
    assert_eq!(extras.cost_in_usd_ticks, Some(188_046_000));
}

#[tokio::test]
async fn chat_extras_from_a_built_reply() {
    let reply = reply_of(
        config().chat(MODEL),
        chat_reply(json!({"usage": {"cost_in_usd_ticks": 1200, "num_sources_used": 3}})),
    )
    .await;
    let extras = reply
        .extras::<Xai>()
        .unwrap_or_else(|| panic!("an xAI reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.cost_in_usd_ticks, Some(1200));
    assert_eq!(extras.num_sources_used, Some(3));
}

/// No recording streams xAI's Chat route: a hand-built stream whose last
/// chunk carries the usage reads as the unary reply stating it.
#[tokio::test]
async fn chat_extras_read_alike_from_a_built_stream() {
    use crate::test_utils::provider_extensions::{chat_stream, streamed_reply_of};

    let usage = json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
        "cost_in_usd_ticks": 1200, "num_sources_used": 3});
    let chunk = |choices: serde_json::Value, usage: serde_json::Value| {
        json!({"id": "reply-1", "object": "chat.completion.chunk", "created": 0,
            "model": MODEL, "choices": choices, "usage": usage})
    };
    let stream = chat_stream(&[
        chunk(
            json!([{"index": 0, "delta": {"role": "assistant", "content": "pong"}}]),
            json!(null),
        ),
        chunk(
            json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            json!(null),
        ),
        chunk(json!([]), usage.clone()),
    ]);
    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<Xai>()
            .unwrap_or_else(|| panic!("an xAI reply"))
            .unwrap_or_else(|error| panic!("{error}"))
    };
    let streamed = read(streamed_reply_of(config().chat(MODEL), stream).await);
    let unary = read(reply_of(config().chat(MODEL), chat_reply(json!({"usage": usage}))).await);
    assert_eq!(streamed, unary);
    assert_eq!(streamed.cost_in_usd_ticks, Some(1200));
    assert_eq!(streamed.num_sources_used, Some(3));
}
