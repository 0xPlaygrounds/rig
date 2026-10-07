//! Groq's options as the bodies they encode to, and its extras read from a
//! recorded reply. The encoder tests are unit tests because no recording
//! sends typed provider options.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{Chat, GROQ, OpenAIConfig};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, recorded_reply, reply_of,
};

fn chat_wire(model: &str) -> Chat {
    OpenAIConfig::with_key(&GROQ, "key").chat(model)
}

fn body(options: &GroqOptions) -> Value {
    body_with::<GroqExt, _>(&chat_wire("qwen/qwen3-32b"), options)
}

#[test]
fn reasoning_format_lands_at_top_level() {
    let body = body(&GroqOptions::new().reasoning_format(ReasoningFormat::Hidden));
    assert_eq!(body["reasoning_format"], "hidden");
}

#[test]
fn include_reasoning_lands_at_top_level() {
    let body = body(&GroqOptions::new().include_reasoning(false));
    assert_eq!(body["include_reasoning"], false);
}

/// Groq refuses both at once, so setting one clears the other.
#[test]
fn include_reasoning_clears_reasoning_format() {
    let body = body(
        &GroqOptions::new()
            .reasoning_format(ReasoningFormat::Parsed)
            .include_reasoning(true),
    );
    assert_eq!(body["include_reasoning"], true);
    assert!(body.get("reasoning_format").is_none(), "{body}");
    let body = self::body(
        &GroqOptions::new()
            .include_reasoning(true)
            .reasoning_format(ReasoningFormat::Raw),
    );
    assert_eq!(body["reasoning_format"], "raw");
    assert!(body.get("include_reasoning").is_none(), "{body}");
}

#[test]
fn search_settings_land_at_top_level() {
    let settings = SearchSettings::new()
        .exclude_domains(["example.com"])
        .include_domains(["rust-lang.org"])
        .country("canada");
    let body = body(&GroqOptions::new().search_settings(settings));
    assert_eq!(
        body["search_settings"],
        json!({
            "exclude_domains": ["example.com"],
            "include_domains": ["rust-lang.org"],
            "country": "canada"
        })
    );
}

#[test]
fn citation_options_land_at_top_level() {
    let body = body(&GroqOptions::new().citation_options(CitationOptions::Disabled));
    assert_eq!(body["citation_options"], "disabled");
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = GroqOptions::new()
        .include_reasoning(true)
        .search_settings(SearchSettings::new().country("us"))
        .citation_options(CitationOptions::Enabled);
    assert_no_reserved_leaf::<GroqExt, _>(
        &[
            chat_wire("qwen/qwen3-32b"),
            chat_wire("openai/gpt-oss-120b"),
        ],
        &options,
    );
    assert_no_reserved_leaf::<GroqExt, _>(
        &[chat_wire("qwen/qwen3-32b")],
        &GroqOptions::new().reasoning_format(ReasoningFormat::Parsed),
    );
}

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        chat_wire("qwen/qwen3.8-27b"),
        recorded_reply(
            "groq",
            "agent_tool_sessions/json_object_response_format_roundtrip",
            0,
        ),
    )
    .await;
    let extras = reply
        .extras::<GroqExt>()
        .unwrap_or_else(|| panic!("a Groq reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        extras
            .x_groq
            .as_ref()
            .and_then(|envelope| envelope.get("id")),
        Some(&json!("req_01m36hjf21egaasf8cq1afm781"))
    );
    assert_eq!(extras.queue_time, Some(0.004410393));
    assert_eq!(extras.prompt_time, Some(0.003381439));
    assert_eq!(extras.completion_time, Some(0.088502995));
    assert_eq!(extras.total_time, Some(0.091884434));
    assert_eq!(extras.service_tier.as_deref(), Some("on_demand"));
    assert_eq!(extras.usage_breakdown, None);
    assert_eq!(extras.system_fingerprint.as_deref(), Some("fp_57c7e760a9"));
    assert_eq!(extras.executed_tools, None);
}

/// A stream's `raw` is the unary document: the recorded stream's extras
/// read as the recorded unary answer of the same prompt states them. Each
/// answer times itself.
#[tokio::test]
async fn extras_read_alike_from_a_recorded_stream() {
    use crate::test_utils::provider_extensions::{recorded_stream, streamed_reply_of};

    const SCENARIO: &str = "agent_tool_sessions/parallel_tool_calls_single_turn_{}";
    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<GroqExt>()
            .unwrap_or_else(|| panic!("a Groq reply"))
            .unwrap_or_else(|error| panic!("{error}"))
    };
    let streamed = read(
        streamed_reply_of(
            chat_wire("model"),
            recorded_stream("groq", &SCENARIO.replace("{}", "streaming"), 0),
        )
        .await,
    );
    let unary = read(
        reply_of(
            chat_wire("model"),
            recorded_reply("groq", &SCENARIO.replace("{}", "nonstreaming"), 0),
        )
        .await,
    );
    for extras in [&streamed, &unary] {
        assert!(extras.x_groq.is_some());
        assert!(extras.queue_time.is_some() && extras.total_time.is_some());
    }
    assert_eq!(streamed.service_tier, unary.service_tier);
    assert_eq!(streamed.system_fingerprint, unary.system_fingerprint);
}
