//! What only Ollama's native `/api/chat` (`Ollama::native_completion`)
//! carries: model `options` such as `num_ctx`, `think` as a level,
//! `keep_alive`, content that only a leading `<think>` tag splits, and tool
//! calls streamed as NDJSON records.
//!
//! Replays by default; record with a local daemon serving `qwen3:4b`:
//! `cargo xtask cassette record ollama/native/<scenario>.yaml`.

use futures::StreamExt;
use rig::completion::{
    CompletionRequest, CompletionResponse, FinishReason, GenerationOptions, ProviderOptions,
    Reasoning,
};
use rig::message::AssistantContent;
use rig::providers::ollama::extension::{KeepAlive, OllamaExt, OllamaOptions};
use rig::streaming::{Item, PartKind, StreamEvent};
use serde_json::{Value, json};

use super::super::{CASSETTE_MODEL, support::with_ollama_cassette};
use crate::cassettes::{recorded_interaction_bodies, recorded_json_request};
use crate::support::{Adder, TOOLS_PREAMBLE};

const PROVIDER: &str = "ollama";

/// The text and reasoning of a turn's blocks, joined by kind.
fn texts(choice: &[AssistantContent]) -> (String, String) {
    let (mut text, mut reasoning) = (String::new(), String::new());
    for block in choice {
        match block {
            AssistantContent::Text(block) => text.push_str(&block.text),
            AssistantContent::Reasoning(block) => reasoning.push_str(&block.text),
            _ => {}
        }
    }
    (text, reasoning)
}

/// The content a recorded reply carried, joined across its records.
fn recorded_content(scenario: &str) -> String {
    let interactions = recorded_interaction_bodies(PROVIDER, scenario);
    let [(_, body)] = interactions.as_slice() else {
        panic!("{scenario}: one interaction was recorded");
    };
    body.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter_map(|record| {
            record
                .pointer("/message/content")?
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// The reason and counts a reply's typed extras read.
fn counted(response: &CompletionResponse) -> (Option<String>, Option<u64>, Option<u64>) {
    let extras = response
        .extras::<OllamaExt>()
        .expect("an Ollama reply")
        .expect("the extras read");
    assert_eq!(extras.model.as_deref(), Some("qwen3:4b"));
    (
        extras.done_reason,
        extras.prompt_eval_count,
        extras.eval_count,
    )
}

/// Model parameters go in `options`, `max_tokens` as `num_predict` beside
/// them, and `keep_alive` and `think` at the top level, where the daemon
/// reads them.
#[tokio::test]
async fn options_reach_the_daemon() {
    let scenario = "native/options_and_keep_alive";
    with_ollama_cassette("native/options_and_keep_alive", |client| async move {
        let response = client
            .native_completion(CASSETTE_MODEL)
            .call(
                CompletionRequest::new("Reply with exactly the single word: pong")
                    .temperature(0.0)
                    .max_tokens(64)
                    .additional_params(json!({
                        "think": false,
                        "keep_alive": "5m",
                        "num_ctx": 4096,
                        "seed": 7,
                        "top_k": 20,
                    })),
            )
            .await
            .expect("the request succeeds");
        assert!(!texts(&response.choice).0.trim().is_empty(), "an answer");
    })
    .await;
    let sent = recorded_json_request(PROVIDER, scenario);
    assert_eq!(
        sent["options"],
        json!({"temperature": 0.0, "num_predict": 64, "num_ctx": 4096, "seed": 7, "top_k": 20})
    );
    assert_eq!(sent["keep_alive"], "5m");
    assert_eq!(sent["think"], false);
    assert_eq!(sent["stream"], false);
}

/// The same request as [`options_reach_the_daemon`] made with typed
/// options sends the recorded body, and the typed extras read the reply's
/// timings and counts.
#[tokio::test]
async fn typed_options_reach_the_daemon_and_extras_read_the_reply() {
    with_ollama_cassette("native/options_and_keep_alive", |client| async move {
        let options = OllamaOptions::default()
            .keep_alive(KeepAlive::duration("5m"))
            .num_ctx(4096)
            .top_k(20);
        let response = client
            .native_completion(CASSETTE_MODEL)
            .call(
                CompletionRequest::new("Reply with exactly the single word: pong")
                    .temperature(0.0)
                    .max_tokens(64)
                    .options(
                        GenerationOptions::default()
                            .reasoning(Reasoning::Off)
                            .seed(7),
                    )
                    .provider_options(
                        ProviderOptions::new()
                            .with::<OllamaExt>(&options)
                            .expect("the options serialize"),
                    ),
            )
            .await
            .expect("the request succeeds");
        let extras = response
            .extras::<OllamaExt>()
            .expect("an Ollama reply")
            .expect("the extras read");
        assert_eq!(extras.model.as_deref(), Some("qwen3:4b"));
        assert_eq!(extras.created_at.as_deref(), Some("1970-01-01T00:00:00Z"));
        assert_eq!(extras.done_reason.as_deref(), Some("length"));
        assert_eq!(extras.total_duration, Some(5_497_823_250));
        assert_eq!(extras.load_duration, Some(1_068_742));
        assert_eq!(extras.prompt_eval_count, Some(18));
        assert_eq!(extras.prompt_eval_cached_count, Some(1));
        assert_eq!(extras.prompt_eval_duration, Some(76_722_000));
        assert_eq!(extras.eval_count, Some(64));
        assert_eq!(extras.eval_duration, Some(5_412_245_000));
        assert_eq!(extras.logprobs, None);
    })
    .await;
}

/// A thinking level is sent as `think` and the reply's `thinking` comes back
/// as reasoning ahead of the answer.
#[tokio::test]
async fn a_thinking_level_returns_reasoning() {
    let scenario = "native/think_level";
    with_ollama_cassette("native/think_level", |client| async move {
        let response = client
            .native_completion(CASSETTE_MODEL)
            .call(
                CompletionRequest::new("What is 2 + 2? Answer with one number.")
                    .temperature(0.0)
                    .additional_params(json!({"think": "low", "options": {"seed": 7}})),
            )
            .await
            .expect("the request succeeds");
        let (text, reasoning) = texts(&response.choice);
        assert!(!reasoning.trim().is_empty(), "the reply thought");
        assert!(text.contains('4'), "{text}");
        assert!(
            matches!(
                response.choice.first(),
                Some(AssistantContent::Reasoning(_))
            ),
            "{:?}",
            response.choice
        );
    })
    .await;
    assert_eq!(recorded_json_request(PROVIDER, scenario)["think"], "low");
}

/// With thinking off, `qwen3:4b` writes its reasoning into the content with
/// no opening `<think>` tag, closed by a bare `</think>`. Only a leading
/// `<think>` tag opens a block, so all of it is text, whole and streamed,
/// and nothing is reasoning. Leave `think` on to get the native `thinking`.
#[tokio::test]
async fn reasoning_without_an_opening_tag_is_text_whole() {
    let scenario = "native/inline_think_whole";
    with_ollama_cassette("native/inline_think_whole", |client| async move {
        let response = client
            .native_completion(CASSETTE_MODEL)
            .call(
                CompletionRequest::new("What is 2 + 2? Answer with one number.")
                    .temperature(0.0)
                    .additional_params(json!({"think": false, "seed": 7})),
            )
            .await
            .expect("the request succeeds");
        let (text, reasoning) = texts(&response.choice);
        assert!(reasoning.is_empty(), "{reasoning}");
        assert_eq!(text, recorded_content(scenario));
        assert_eq!(
            counted(&response),
            (Some("stop".to_owned()), Some(23), Some(246))
        );
    })
    .await;
    let content = recorded_content(scenario);
    assert!(
        content.contains("\n</think>\n\n") && !content.trim_start().starts_with("<think>"),
        "{scenario}: the premise, a block with no opening tag"
    );
}

#[tokio::test]
async fn reasoning_without_an_opening_tag_is_text_streamed() {
    let scenario = "native/inline_think_streamed";
    with_ollama_cassette("native/inline_think_streamed", |client| async move {
        let mut stream = client
            .native_completion(CASSETTE_MODEL)
            .stream(
                CompletionRequest::new("What is 2 + 2? Answer with one number.")
                    .temperature(0.0)
                    .additional_params(json!({"think": false, "seed": 7})),
            )
            .expect("the stream starts");
        let (mut streamed_text, mut first) = (String::new(), None);
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("an item") {
                first.get_or_insert_with(|| text.clone());
                streamed_text.push_str(&text);
            }
        }
        let response = stream.finish().await.expect("the stream ends");
        let (text, reasoning) = texts(&response.choice);
        assert!(reasoning.is_empty(), "{reasoning}");
        assert_eq!(streamed_text, text);
        assert_eq!(text, recorded_content(scenario));
        // Nothing is held: the first record's content streams on its own.
        assert_eq!(first.as_deref(), Some("Okay"));
        // The streamed reply's extras read what the unary reply's do.
        assert_eq!(
            counted(&response),
            (Some("stop".to_owned()), Some(23), Some(246))
        );
    })
    .await;
    assert!(
        recorded_content(scenario).contains("\n</think>\n\n"),
        "{scenario}: the premise, a block with no opening tag"
    );
}

/// A streamed tool call arrives whole in one NDJSON record: it starts, its
/// arguments stream through the shared writer, it keeps the daemon's id,
/// and the `stop` reply ends on `ToolCalls`.
#[tokio::test]
async fn ndjson_tool_calls_stream() {
    with_ollama_cassette("native/streamed_tool_call", |client| async move {
        let mut stream = client
            .native_completion(CASSETTE_MODEL)
            .stream(
                CompletionRequest::new("Use the add tool to add 2 and 3.")
                    .preamble(TOOLS_PREAMBLE.to_owned())
                    .tool(rig::tool::tool_definition(&Adder))
                    .temperature(0.0)
                    .additional_params(json!({"think": false, "seed": 7})),
            )
            .expect("the stream starts");
        let (mut started, mut arguments) = (false, String::new());
        while let Some(item) = stream.next().await {
            match item.expect("an item") {
                Item::Event(StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    name: Some(name),
                    ..
                }) => started |= name.as_str() == "add",
                Item::Event(StreamEvent::Arguments { json, .. }) => arguments.push_str(&json),
                _ => {}
            }
        }
        let response = stream.finish().await.expect("the stream ends");
        assert!(started, "the call started on the stream");
        let call = response
            .tool_calls()
            .find(|call| call.function.name.as_str() == "add")
            .expect("an add call");
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).expect("the arguments streamed whole"),
            Value::Object(call.function.arguments.clone())
        );
        assert!(call.id.provider().is_some(), "the daemon's id is kept");
        assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    })
    .await;
}
