//! Edge matrix for rig#2354's third fix, plus the wire census the hunt turned
//! up.
//!
//! **The fix.** DeepSeek takes message `content` as a plain string, so
//! `finalize_request_body` flattened content-part arrays. It passed
//! `only_if_all_text = false`, which *drops* every non-text part — so an
//! attached image, audio clip or PDF was silently deleted and DeepSeek
//! answered the question from the remaining text alone, with nothing anywhere
//! saying the attachment was gone. Perplexity, the tree's other plain-text-only
//! provider, passes `true` so the part survives to the wire and the API's own
//! rejection reaches the caller. DeepSeek now does the same: verified live,
//! `{"type":"image_url",...}` comes back `400 Failed to deserialize the JSON
//! body into the target type: messages[0]: unknown variant `image_url`,
//! expected `text``. All-text arrays still flatten to a plain string, byte for
//! byte as before — which is why no existing fixture moved.
//!
//! **The census** (confirmed non-bugs, recorded so they stay confirmed):
//! DeepSeek really does reject a forced `tool_choice` while thinking is on, so
//! `finalize_request_body`'s suppression is justified; the completion path
//! preserves DeepSeek's error envelope; and the `prompt_cache_hit_tokens` /
//! `prompt_cache_miss_tokens` split reaches `Usage::cached_input_tokens` on
//! both transports.

use rig::completion::{CompletionModel, Document, Message};
use rig::message::{DocumentMediaType, DocumentSourceKind, ToolChoice, UserContent};
use rig::prelude::*;
use rig::providers::deepseek;
use serde_json::{Value, json};

use super::support::{
    collect_raw_stream_outcome, recorded_interactions, recorded_request,
    with_deepseek_cassette_bogus_key_result, with_deepseek_wire_shape_cassette_result,
};

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQkAAAgEsAtx/ZMZxgi+hcEKLNO+FgEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQGBywKqxUDxqh7TUQAAAABJRU5ErkJggg==";
/// A ~600-token filler so a repeated prompt crosses DeepSeek's cache-block
/// boundary and the hit/miss split is non-zero on the second request.
const CACHE_FILLER: &str = "Cache probe paragraph. The nightly build pipeline warms the artifact cache, uploads the artifacts, verifies the checksums, publishes the manifest, and finally notifies the release channel. ";

fn non_thinking_params() -> Value {
    json!({ "thinking": { "type": "disabled" } })
}

fn thinking_params() -> Value {
    json!({ "thinking": { "type": "enabled" } })
}

fn red_png() -> UserContent {
    UserContent::image_base64(
        RED_PNG_BASE64,
        Some(rig::message::ImageMediaType::PNG),
        None,
    )
}

fn multimodal_prompt(part: UserContent) -> Message {
    Message::User {
        content: vec![UserContent::text("What colour is the attachment?"), part],
    }
}

/// The `type` tags of a recorded request's first user message content array,
/// or `None` when the wire carried a plain string.
fn recorded_first_user_part_types(scenario: &str) -> Option<Vec<String>> {
    let body = recorded_request(scenario);
    let messages = body["messages"].as_array()?;
    let user = messages
        .iter()
        .find(|message| message["role"] == "user")
        .expect("a user message should be recorded");
    user["content"].as_array().map(|parts| {
        parts
            .iter()
            .map(|part| part["type"].as_str().unwrap_or("?").to_owned())
            .collect()
    })
}

fn assert_rejected_by_deepseek(error: &rig::completion::CompletionError, context: &str) {
    let rendered = error.to_string();
    assert!(
        rendered.contains("unknown variant") || rendered.contains("expected `text`"),
        "{context}: expected DeepSeek's own rejection of the non-text part, got {rendered}"
    );
}

// ================================================================
// A. Non-text parts now reach the wire
// ================================================================

#[tokio::test]
async fn blocking_image_base64_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_image_base64_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_image_base64_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(multimodal_prompt(red_png()))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("DeepSeek rejects an image part rather than answering without it");
            assert_rejected_by_deepseek(&error, "blocking base64 image");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_image_base64_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "image_url".to_owned()]),
        "the image part must survive finalization instead of being deleted"
    );
}

#[tokio::test]
async fn blocking_image_url_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_image_url_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_image_url_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(multimodal_prompt(UserContent::image_url(
                            "https://example.invalid/red.png",
                            Some(rig::message::ImageMediaType::PNG),
                            None,
                        )))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("DeepSeek rejects an image part rather than answering without it");
            assert_rejected_by_deepseek(&error, "blocking url image");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_image_url_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "image_url".to_owned()])
    );
}

#[tokio::test]
async fn blocking_pdf_document_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_pdf_document_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_pdf_document_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(multimodal_prompt(UserContent::Document(
                            rig::message::Document {
                                data: DocumentSourceKind::Base64("JVBERi0xLjQK".to_owned()),
                                media_type: Some(DocumentMediaType::PDF),
                                additional_params: None,
                            },
                        )))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("DeepSeek rejects a file part rather than answering without it");
            assert_rejected_by_deepseek(&error, "blocking pdf document");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_pdf_document_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "file".to_owned()])
    );
}

#[tokio::test]
async fn blocking_audio_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_audio_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_audio_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(multimodal_prompt(UserContent::audio(
                            "aGVsbG8=",
                            Some(rig::message::AudioMediaType::MP3),
                        )))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("DeepSeek rejects an audio part rather than answering without it");
            assert_rejected_by_deepseek(&error, "blocking audio");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_audio_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "input_audio".to_owned()])
    );
}

#[tokio::test]
async fn blocking_video_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_video_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_video_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(multimodal_prompt(UserContent::Video(
                            rig::message::Video {
                                data: DocumentSourceKind::Url(
                                    "https://example.invalid/clip.mp4".to_owned(),
                                ),
                                media_type: Some(rig::message::VideoMediaType::MP4),
                                additional_params: None,
                            },
                        )))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("DeepSeek rejects a video part rather than answering without it");
            assert_rejected_by_deepseek(&error, "blocking video");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_video_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "video_url".to_owned()])
    );
}

/// The boundary the flatten predicate turns on: an array with **no** text part
/// at all. `parts.iter().all(is_text)` is false here, so the array survives
/// whole rather than flattening to the empty string — which is what the old
/// `false` argument produced, an empty user turn.
#[tokio::test]
async fn blocking_image_only_message_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_image_only_message_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_image_only_message_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![red_png()],
                        })
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
                .expect_err("an image-only turn is rejected, not silently emptied");
            assert_rejected_by_deepseek(&error, "blocking image-only");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_image_only_message_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["image_url".to_owned()]),
        "the lone image part must not be flattened away to an empty string"
    );
}

#[tokio::test]
async fn streaming_image_part_reaches_the_wire() {
    const SCENARIO: &str = "wire_shape_matrix/streaming_image_part_reaches_the_wire";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/streaming_image_part_reaches_the_wire",
        |client| async move {
            let model = client.completion_model(MODEL);
            // The SSE connect may surface the rejection as a connect error or
            // as the stream's first item, depending on how the transport
            // reports a 400 on an event-stream request; both are the provider
            // rejecting the part rather than answering without it.
            let rendered = match model
                .stream(
                    model
                        .completion_request(multimodal_prompt(red_png()))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await
            {
                Err(error) => error.to_string(),
                Ok(stream) => {
                    let outcome = collect_raw_stream_outcome(stream).await;
                    assert!(
                        outcome.tool_calls.is_empty() && outcome.text.is_empty(),
                        "a rejected request must produce no content: {:?}",
                        outcome.order
                    );
                    outcome.errors.join("\n")
                }
            };
            assert!(
                rendered.contains("unknown variant") || rendered.contains("expected `text`"),
                "streaming image: expected DeepSeek's own rejection, got {rendered}"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_image_part_reaches_the_wire should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        Some(vec!["text".to_owned(), "image_url".to_owned()])
    );
}

// ================================================================
// B. Controls: every all-text shape still flattens, byte for byte
// ================================================================

#[tokio::test]
async fn blocking_all_text_parts_still_flatten_to_a_string() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_all_text_parts_still_flatten_to_a_string";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_all_text_parts_still_flatten_to_a_string",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(Message::User {
                            content: vec![
                                UserContent::text("Reply with exactly: parts-ok"),
                                UserContent::text("Nothing else."),
                            ],
                        })
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await?;
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_all_text_parts_still_flatten_to_a_string should replay from its cassette");

    assert_eq!(
        recorded_first_user_part_types(SCENARIO),
        None,
        "an all-text array must still reach the wire as a plain string"
    );
    let body = recorded_request(SCENARIO);
    let user = body["messages"]
        .as_array()
        .and_then(|messages| messages.iter().find(|message| message["role"] == "user"))
        .expect("a user message should be recorded")
        .clone();
    assert_eq!(
        user["content"],
        "Reply with exactly: parts-ok\nNothing else."
    );
}

#[tokio::test]
async fn blocking_text_document_still_flattens_to_a_string() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_text_document_still_flattens_to_a_string";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_text_document_still_flattens_to_a_string",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("What is the code word? Answer with just the word.")
                        .document(Document {
                            id: "code-word".to_owned(),
                            text: "The code word is periwinkle.".to_owned(),
                            additional_props: Default::default(),
                        })
                        .additional_params(non_thinking_params())
                        .max_tokens(24)
                        .build(),
                )
                .await?;
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_text_document_still_flattens_to_a_string should replay from its cassette");

    let body = recorded_request(SCENARIO);
    for message in body["messages"].as_array().expect("messages") {
        assert!(
            message["content"].is_string(),
            "every message must reach the wire as a plain string: {message}"
        );
    }
}

#[tokio::test]
async fn blocking_assistant_and_tool_history_still_flattens() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_assistant_and_tool_history_still_flattens";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_assistant_and_tool_history_still_flattens",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("Now say: history-ok")
                        .message(Message::Assistant {
                            id: None,
                            content: vec![
                                rig::message::AssistantContent::text("Checking the ledger."),
                                rig::message::AssistantContent::tool_call(
                                    "call_history_1",
                                    "ping",
                                    json!({}),
                                ),
                            ],
                        })
                        .message(Message::User {
                            content: vec![UserContent::tool_result(
                                "call_history_1",
                                "ping",
                                vec![rig::message::ToolResultContent::text("pong")],
                            )],
                        })
                        .tool(crate::support::zero_arg_tool_definition("ping"))
                        .additional_params(non_thinking_params())
                        .max_tokens(16)
                        .build(),
                )
                .await?;
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_assistant_and_tool_history_still_flattens should replay from its cassette");

    let body = recorded_request(SCENARIO);
    let messages = body["messages"].as_array().expect("messages");
    for message in messages {
        assert!(
            message["content"].is_string(),
            "assistant and tool-result history must still flatten: {message}"
        );
    }
    assert!(
        messages.iter().any(|message| message["role"] == "tool"),
        "premise: the history carried a tool-result message: {body}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message["tool_calls"].is_array()),
        "premise: the history carried an assistant tool call: {body}"
    );
}

// ================================================================
// C. Census: the forced-tool-choice suppression is justified
// ================================================================

#[tokio::test]
async fn forced_tool_choice_under_thinking_is_rejected_upstream() {
    const SCENARIO: &str =
        "wire_shape_matrix/forced_tool_choice_under_thinking_is_rejected_upstream";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/forced_tool_choice_under_thinking_is_rejected_upstream",
        |client| async move {
            // `finalize_request_body` rewrites any forced `tool_choice` rig
            // itself would send, so the only way to learn what DeepSeek does
            // with one is to hand-build the body.
            let url = format!(
                "{}/chat/completions",
                client.base_url().trim_end_matches('/')
            );
            let api_key =
                std::env::var("DEEPSEEK_API_KEY").unwrap_or_else(|_| "[REDACTED]".to_owned());
            let tools = json!([{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]);

            for tool_choice in [
                json!("required"),
                json!({"type": "function", "function": {"name": "ping"}}),
            ] {
                let response = reqwest::Client::new()
                    .post(&url)
                    .bearer_auth(&api_key)
                    .json(&json!({
                        "model": MODEL,
                        "thinking": {"type": "enabled"},
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "ping"}],
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }))
                    .send()
                    .await?;

                let status = response.status();
                let body: Value = response.json().await?;
                assert_eq!(
                    status.as_u16(),
                    400,
                    "a forced tool choice under thinking is a hard error: {body}"
                );
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains("thinking mode does not support this tool_choice"),
                    "the rejection names the thinking-mode constraint: {body}"
                );
            }
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "forced_tool_choice_under_thinking_is_rejected_upstream should replay from its cassette",
    );

    assert_eq!(
        recorded_interactions(SCENARIO).len(),
        2,
        "both forced shapes are recorded"
    );
}

#[tokio::test]
async fn rig_suppresses_a_forced_tool_choice_while_thinking_is_on() {
    const SCENARIO: &str =
        "wire_shape_matrix/rig_suppresses_a_forced_tool_choice_while_thinking_is_on";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/rig_suppresses_a_forced_tool_choice_while_thinking_is_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("ping")
                        .tool(crate::support::zero_arg_tool_definition("ping"))
                        .tool_choice(ToolChoice::Required)
                        .additional_params(thinking_params())
                        .max_tokens(64)
                        .build(),
                )
                .await?;
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "rig_suppresses_a_forced_tool_choice_while_thinking_is_on should replay from its cassette",
    );

    assert_eq!(
        recorded_request(SCENARIO)["tool_choice"],
        Value::Null,
        "the forced choice is suppressed to an explicit null, which the API accepts"
    );
}

#[tokio::test]
async fn rig_keeps_a_forced_tool_choice_when_thinking_is_disabled() {
    const SCENARIO: &str =
        "wire_shape_matrix/rig_keeps_a_forced_tool_choice_when_thinking_is_disabled";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/rig_keeps_a_forced_tool_choice_when_thinking_is_disabled",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("ping")
                        .tool(crate::support::zero_arg_tool_definition("ping"))
                        .tool_choice(ToolChoice::Required)
                        .additional_params(non_thinking_params())
                        .max_tokens(32)
                        .build(),
                )
                .await?;
            assert!(!response.choice.is_empty());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "rig_keeps_a_forced_tool_choice_when_thinking_is_disabled should replay from its cassette",
    );

    assert_eq!(
        recorded_request(SCENARIO)["tool_choice"],
        json!("required"),
        "with thinking off the caller's constraint rides the wire untouched"
    );
}

// ================================================================
// D. Census: the completion path preserves DeepSeek's error envelope
// ================================================================

#[tokio::test]
async fn chat_completion_rejects_an_unknown_model_with_the_provider_body() {
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/chat_completion_rejects_an_unknown_model_with_the_provider_body",
        |client| async move {
            let model = client.completion_model("deepseek-v9-nonexistent");
            let error = model
                .completion(
                    model
                        .completion_request("hi")
                        .additional_params(non_thinking_params())
                        .max_tokens(8)
                        .build(),
                )
                .await
                .expect_err("an unknown model is rejected");
            let rendered = error.to_string();
            assert!(
                rendered.contains("deepseek-v9-nonexistent"),
                "the provider's own message survives: {rendered}"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("chat_completion_rejects_an_unknown_model_with_the_provider_body should replay from its cassette");
}

#[tokio::test]
async fn chat_completion_rejects_a_bogus_key_with_the_provider_body() {
    with_deepseek_cassette_bogus_key_result(
        "wire_shape_matrix/chat_completion_rejects_a_bogus_key_with_the_provider_body",
        |client| async move {
            let model = client.completion_model(MODEL);
            let error = model
                .completion(
                    model
                        .completion_request("hi")
                        .additional_params(non_thinking_params())
                        .max_tokens(8)
                        .build(),
                )
                .await
                .expect_err("a rejected key is an error");
            let rendered = error.to_string().to_lowercase();
            assert!(
                rendered.contains("authentication fails"),
                "the provider's own 401 body survives: {rendered}"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("chat_completion_rejects_a_bogus_key_with_the_provider_body should replay from its cassette");
}

// ================================================================
// E. Census: the cache hit/miss split reaches rig's usage
// ================================================================

fn cache_probe_prompt() -> String {
    format!(
        "{}Answer with exactly one word: cached.",
        CACHE_FILLER.repeat(24)
    )
}

#[tokio::test]
async fn blocking_repeated_prompt_reports_the_cache_split() {
    const SCENARIO: &str = "wire_shape_matrix/blocking_repeated_prompt_reports_the_cache_split";
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/blocking_repeated_prompt_reports_the_cache_split",
        |client| async move {
            let model = client.completion_model(MODEL);
            let build = || {
                model
                    .completion_request(cache_probe_prompt())
                    .additional_params(non_thinking_params())
                    .max_tokens(8)
                    .build()
            };

            let first = model.raw_completion(build()).await?;
            assert_eq!(
                first.usage.prompt_cache_hit_tokens + first.usage.prompt_cache_miss_tokens,
                first.usage.prompt_tokens,
                "hit + miss accounts for the whole prompt: {:?}",
                first.usage
            );

            let second = model.raw_completion(build()).await?;
            assert!(
                second.usage.prompt_cache_hit_tokens > 0,
                "the repeated prompt should hit DeepSeek's cache: {:?}",
                second.usage
            );
            let normalized = rig::completion::Usage::from(&second.usage);
            assert_eq!(
                normalized.cached_input_tokens,
                u64::from(second.usage.prompt_cache_hit_tokens),
                "the native cache-hit counter reaches rig's usage"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_repeated_prompt_reports_the_cache_split should replay from its cassette");

    assert_eq!(
        recorded_interactions(SCENARIO).len(),
        2,
        "the cache split needs both requests recorded"
    );
}

#[tokio::test]
async fn streaming_repeated_prompt_reports_the_cache_split() {
    with_deepseek_wire_shape_cassette_result(
        "wire_shape_matrix/streaming_repeated_prompt_reports_the_cache_split",
        |client| async move {
            let model = client.completion_model(MODEL);
            let build = || {
                model
                    .completion_request(cache_probe_prompt())
                    .additional_params(non_thinking_params())
                    .max_tokens(8)
                    .build()
            };

            let _ = collect_raw_stream_outcome(model.stream(build()).await?).await;
            let second = collect_raw_stream_outcome(model.stream(build()).await?).await;
            let usage = second
                .final_record
                .as_ref()
                .map(|record| record.usage)
                .expect("the stream should yield a terminal record");
            assert!(
                usage.cached_input_tokens > 0,
                "the streamed terminal carries DeepSeek's cache-hit counter: {usage:?}"
            );
            assert!(usage.input_tokens > usage.cached_input_tokens);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_repeated_prompt_reports_the_cache_split should replay from its cassette");
}
