//! Anthropic prompt caching cassette tests.

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::{
    AssistantContent, CompletionModel, CompletionResponse as RigCompletionResponse, GetTokenUsage,
    ToolDefinition, Usage,
};
use rig::message::ToolChoice;
use rig::providers::anthropic;
use rig::streaming::StreamedAssistantContent;
use serde_json::json;

use super::super::support::with_anthropic_cassette;

const CACHE_PROBE_RESPONSE: &str = "cache probe ready";
const CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these three words: cache probe ready";
const STREAMING_CACHE_PROBE_RESPONSE: &str = "stream cache probe ready";
const STREAMING_CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these four words: stream cache probe ready";
const AUTOMATIC_CACHE_PROBE_RESPONSE: &str = "automatic cache probe ready";
const AUTOMATIC_CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these four words: automatic cache probe ready";
const CACHE_PADDING_REPETITIONS: usize = 180;
const CACHE_PADDING_SENTENCE: &str = "\
This cache fixture paragraph is stable provider test padding about request routing, \
tool schemas, system instructions, and deterministic replay behavior.";

#[tokio::test]
async fn manual_prompt_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/manual_prompt_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching();
            let tools = cache_probe_tools();

            let first = send_cache_probe(
                model.clone(),
                CACHE_PROBE_PROMPT,
                cache_probe_preamble(),
                tools.clone(),
            )
            .await;
            assert_response_contains_cache_probe(&first, CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first prompt-cached request");

            let second =
                send_cache_probe(model, CACHE_PROBE_PROMPT, cache_probe_preamble(), tools).await;
            assert_response_contains_cache_probe(&second, CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second prompt-cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_prompt_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/streaming_prompt_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching();
            let tools = cache_probe_tools_for("streaming prompt caching");

            let first = send_streaming_cache_probe(
                model.clone(),
                STREAMING_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("streaming prompt caching"),
                tools.clone(),
            )
            .await;
            assert_text_contains_cache_probe(&first.text, STREAMING_CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first streaming prompt-cached request");

            let second = send_streaming_cache_probe(
                model,
                STREAMING_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("streaming prompt caching"),
                tools,
            )
            .await;
            assert_text_contains_cache_probe(&second.text, STREAMING_CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second streaming prompt-cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn prompt_and_automatic_caching_reuses_tool_cache() {
    with_anthropic_cassette(
        "prompt_caching/prompt_and_automatic_caching_reuses_tool_cache",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_automatic_caching();
            let tools = cache_probe_tools_for("manual plus automatic prompt caching");

            let first = send_cache_probe(
                model.clone(),
                AUTOMATIC_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("manual plus automatic prompt caching"),
                tools.clone(),
            )
            .await;
            assert_response_contains_cache_probe(&first, AUTOMATIC_CACHE_PROBE_RESPONSE);
            assert_cache_created_or_read(&first.usage, "first prompt+automatic cached request");

            let second = send_cache_probe(
                model,
                AUTOMATIC_CACHE_PROBE_PROMPT,
                cache_probe_preamble_for("manual plus automatic prompt caching"),
                tools,
            )
            .await;
            assert_response_contains_cache_probe(&second, AUTOMATIC_CACHE_PROBE_RESPONSE);
            assert!(
                second.usage.cached_input_tokens > 0,
                "second prompt+automatic cached request should read cached tokens, got usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

async fn send_cache_probe(
    model: anthropic::completion::CompletionModel,
    prompt: &'static str,
    preamble: String,
    tools: Vec<ToolDefinition>,
) -> RigCompletionResponse<anthropic::completion::CompletionResponse> {
    model
        .completion_request(prompt)
        .preamble(preamble)
        .tools(tools)
        .tool_choice(ToolChoice::None)
        .temperature(0.0)
        .max_tokens(16)
        .send()
        .await
        .expect("prompt-cached Anthropic request should succeed")
}

struct StreamingCacheProbeResponse {
    text: String,
    usage: Usage,
}

async fn send_streaming_cache_probe(
    model: anthropic::completion::CompletionModel,
    prompt: &'static str,
    preamble: String,
    tools: Vec<ToolDefinition>,
) -> StreamingCacheProbeResponse {
    let mut stream = model
        .completion_request(prompt)
        .preamble(preamble)
        .tools(tools)
        .additional_params(json!({
            "tool_choice": { "type": "none" }
        }))
        .temperature(0.0)
        .max_tokens(16)
        .stream()
        .await
        .expect("streaming prompt-cached Anthropic request should start");
    let mut text = String::new();
    let mut usage = None;

    while let Some(item) = stream.next().await {
        match item.expect("streaming prompt-cached Anthropic item should succeed") {
            StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
            StreamedAssistantContent::Final(response) => {
                usage = Some(response.token_usage());
            }
            _ => {}
        }
    }

    StreamingCacheProbeResponse {
        text,
        usage: usage.expect("stream should yield final token usage"),
    }
}

fn assert_response_contains_cache_probe(
    response: &RigCompletionResponse<anthropic::completion::CompletionResponse>,
    expected: &str,
) {
    let text = response_text(response);
    assert_text_contains_cache_probe(&text, expected);
}

fn assert_text_contains_cache_probe(text: &str, expected: &str) {
    assert!(
        text.to_ascii_lowercase()
            .contains(&expected.to_ascii_lowercase()),
        "response should contain the requested cache probe text {expected:?}, got: {text:?}"
    );
}

fn assert_cache_created_or_read(usage: &Usage, context: &str) {
    assert!(
        usage.cache_creation_input_tokens > 0 || usage.cached_input_tokens > 0,
        "{context} should create or read cache tokens, got usage: {usage:?}"
    );
}

fn response_text(
    response: &RigCompletionResponse<anthropic::completion::CompletionResponse>,
) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn cache_probe_preamble() -> String {
    format!(
        "You are a deterministic cassette test assistant. {}\n{}",
        "Never call tools for the cache probe prompt; answer only with the requested phrase.",
        cache_padding(CACHE_PADDING_REPETITIONS)
    )
}

fn cache_probe_preamble_for(label: &str) -> String {
    format!(
        "You are a deterministic cassette test assistant for {label}. {}\n{}",
        "Never call tools for the cache probe prompt; answer only with the requested phrase.",
        cache_padding(CACHE_PADDING_REPETITIONS)
    )
}

fn cache_probe_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lookup_cache_policy".to_string(),
            description: format!(
                "Return internal prompt cache policy notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy topic to look up."
                    }
                },
                "required": ["topic"]
            }),
        },
        ToolDefinition {
            name: "lookup_cache_fixture".to_string(),
            description: format!(
                "Return prompt cache fixture notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "fixture": {
                        "type": "string",
                        "description": "Fixture identifier to look up."
                    }
                },
                "required": ["fixture"]
            }),
        },
    ]
}

fn cache_probe_tools_for(label: &str) -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lookup_cache_policy".to_string(),
            description: format!(
                "Return {label} internal prompt cache policy notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Policy topic to look up."
                    }
                },
                "required": ["topic"]
            }),
        },
        ToolDefinition {
            name: "lookup_cache_fixture".to_string(),
            description: format!(
                "Return prompt cache fixture notes. {}",
                cache_padding(CACHE_PADDING_REPETITIONS / 2)
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "fixture": {
                        "type": "string",
                        "description": "Fixture identifier to look up."
                    }
                },
                "required": ["fixture"]
            }),
        },
    ]
}

fn cache_padding(repetitions: usize) -> String {
    std::iter::repeat_n(CACHE_PADDING_SENTENCE, repetitions)
        .collect::<Vec<_>>()
        .join(" ")
}
