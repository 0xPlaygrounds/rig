//! Wire-sequence conformance suite over the provider streaming pipelines.
//!
//! Scenarios live in `rig_core::test_utils::streaming_conformance`; this file
//! instantiates each sequence family per provider fixture. Every family pins a
//! shipped bug from the #2257 review rounds — the scenario doc comments cite
//! the `rig-2257-code-review-findings-*.md` entry.
//!
//! Run the suite with:
//! `cargo test -p rig --test core core::streaming_conformance`

use rig_core::test_utils::streaming_conformance::{
    self as conformance,
    fixtures::{
        anthropic, cohere, gemini_rest, interactions, ollama, openai_chat, openai_responses,
    },
};

/// Lower fixture frames onto the byte transport a
/// `SequencedStreamingHttpClient` replays (the byte-driver contract the
/// in-crate fixtures apply; event frames are a fixture authoring error here).
fn byte_chunks(
    chunks: conformance::WireChunks,
) -> Result<Vec<rig_core::http_client::Result<bytes::Bytes>>, rig_core::completion::CompletionError>
{
    chunks
        .into_iter()
        .map(|chunk| match chunk {
            Ok(frame) => frame.as_bytes().cloned().map(Ok).ok_or_else(|| {
                rig_core::completion::CompletionError::ProviderError(
                    "typed-event frame fed to a byte-transport driver".to_string(),
                )
            }),
            Err(error) => Ok(Err(error)),
        })
        .collect()
}

fn sse(frame: &serde_json::Value) -> conformance::WireInput {
    conformance::WireInput::Bytes(bytes::Bytes::from(format!("data: {frame}\n\n")))
}

/// xAI relays the OpenAI Responses SSE wire through
/// `ResponsesStreamOptions::strict_with_immediate_tool_calls` (completed tool
/// calls are emitted the moment their done item arrives instead of buffering
/// until the terminal). The wire frames are the shared Responses fixture's;
/// only the pipeline under test differs.
mod xai {
    use super::*;
    use rig_core::client::CompletionClient as _;
    use rig_core::completion::CompletionModel as _;
    use rig_core::test_utils::SequencedStreamingHttpClient;

    fn driver() -> conformance::WireDriver {
        conformance::WireDriver::new("xai", |chunks| {
            Box::pin(async move {
                let client = rig_core::providers::xai::Client::builder()
                    .api_key("test-key")
                    .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                    .build()
                    .map_err(|error| {
                        rig_core::completion::CompletionError::ProviderError(error.to_string())
                    })?;
                let model = client.completion_model(rig_core::providers::xai::completion::GROK_4);
                let request = model.completion_request("hello").build();
                let stream = rig_core::completion::CompletionModel::stream(&model, request).await?;
                Ok(conformance::fixtures::drain(stream).await)
            })
        })
    }

    pub fn fixture() -> conformance::ProviderWireFixture {
        conformance::ProviderWireFixture {
            driver: driver(),
            ..openai_responses::fixture()
        }
    }
}

/// Copilot's two routes: Codex-class models stream the OpenAI Responses SSE
/// wire over `/responses`; conversational models stream the chat-completions
/// wire over `/chat/completions`. Each route reuses its wire family's shared
/// fixture frames with a Copilot pipeline driver.
mod copilot {
    use super::*;
    use rig_core::client::CompletionClient as _;
    use rig_core::completion::CompletionModel as _;
    use rig_core::test_utils::SequencedStreamingHttpClient;

    fn driver(provider: &'static str, model_name: &'static str) -> conformance::WireDriver {
        conformance::WireDriver::new(provider, move |chunks| {
            Box::pin(async move {
                let client = rig_core::providers::copilot::Client::builder()
                    .api_key("copilot-token")
                    .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                    .build()
                    .map_err(|error| {
                        rig_core::completion::CompletionError::ProviderError(error.to_string())
                    })?;
                let model = client.completion_model(model_name);
                let request = model.completion_request("hello").build();
                let stream = model.stream(request).await?;
                Ok(conformance::fixtures::drain(stream).await)
            })
        })
    }

    /// The `/responses` route (Codex-class models).
    pub fn responses_fixture() -> conformance::ProviderWireFixture {
        conformance::ProviderWireFixture {
            driver: driver("copilot-responses", "gpt-5.3-codex"),
            ..openai_responses::fixture()
        }
    }

    /// The `/chat/completions` route (conversational models).
    pub fn chat_fixture() -> conformance::ProviderWireFixture {
        conformance::ProviderWireFixture {
            driver: driver("copilot-chat", "gpt-4o"),
            ..openai_chat::fixture()
        }
    }
}

/// The ChatGPT backend's LIVE SSE route: `ResponsesCompletionModel::stream`
/// relays the OpenAI Responses SSE wire (the envelope-repairing buffered
/// replay is a separate pipeline, exercised by
/// `terminal_body_content_merges_per_kind` below). The wire frames are the
/// shared Responses fixture's; only the pipeline under test differs.
mod chatgpt {
    use super::*;
    use rig_core::client::CompletionClient as _;
    use rig_core::completion::CompletionModel as _;
    use rig_core::test_utils::SequencedStreamingHttpClient;

    fn driver() -> conformance::WireDriver {
        conformance::WireDriver::new("chatgpt", |chunks| {
            Box::pin(async move {
                let client = rig_core::providers::chatgpt::Client::builder()
                    .api_key(rig_core::providers::chatgpt::ChatGPTAuth::AccessToken {
                        access_token: "test-token".to_string(),
                        account_id: Some("account-id".to_string()),
                    })
                    .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                    .build()
                    .map_err(|error| {
                        rig_core::completion::CompletionError::ProviderError(error.to_string())
                    })?;
                let model = client.completion_model("gpt-5.4");
                let request = model.completion_request("hello").build();
                let stream = model.stream(request).await?;
                Ok(conformance::fixtures::drain(stream).await)
            })
        })
    }

    pub fn fixture() -> conformance::ProviderWireFixture {
        conformance::ProviderWireFixture {
            driver: driver(),
            ..openai_responses::fixture()
        }
    }
}

// The in-crate wire families (openai_chat, openai_responses, gemini_rest,
// gemini_interactions, anthropic, cohere, ollama) expand their canonical
// suites in `streaming_conformance_suites.rs`; the families below reuse those
// fixtures' frames with their own pipeline drivers, through the same
// declarative macro (with anti-tamper and capability checks). Capability rows
// mirror the wire family whose frames each fixture reuses.
mod xai_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "xai",
        fixture: xai::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: true,
        },
    }
}

mod copilot_responses_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "copilot",
        fixture: copilot::responses_fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: true,
        },
    }
}

mod chatgpt_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "chatgpt",
        fixture: chatgpt::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: true,
        },
    }
}

mod copilot_chat_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "copilot",
        fixture: copilot::chat_fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: true,
            malformed_frame: true,
            unknown_event_frame: false,
            defective_known_frame: true,
            delta_less_prelude: true,
            refusal: false,
        },
    }
}

// Grammar-level guards (#2258 Part III (c) item 5). The other two scenarios
// that item names — per-wire unknown-frame-warn-and-skip and
// corrupt-frame-error — are the `unknown_event_is_skipped`,
// `malformed_frame_surfaces_err_and_terminal_still_completes`, and
// `defective_known_event_surfaces_err` families every suite above already
// instantiates; illegal-transition cases live once in parts.rs unit tests.
mod grammar_guards {
    use super::*;

    /// langchain's aggregation guard (`integration_tests/chat_models.py:873`):
    /// a plain streamed answer aggregates to exactly ONE text block, however
    /// many deltas delivered it — consecutive text deltas must never fragment
    /// the aggregated choice.
    async fn exactly_one_text_block(fixture: conformance::ProviderWireFixture) {
        let mut frames = fixture.text_frames.clone();
        frames.extend(fixture.terminal_frames.iter().cloned());
        let drained = fixture
            .driver
            .drive(conformance::ok_chunks(frames))
            .await
            .expect("stream should drive");
        let expected = fixture.expected_texts.concat();
        assert_eq!(
            drained.choice_texts(),
            vec![expected.as_str()],
            "aggregation must yield exactly one text block for {}",
            fixture.driver.provider
        );
    }

    macro_rules! exactly_one_text_block_test {
        ($name:ident, $fixture:path) => {
            #[tokio::test]
            async fn $name() {
                exactly_one_text_block($fixture()).await;
            }
        };
    }

    exactly_one_text_block_test!(openai_chat, openai_chat::fixture);
    exactly_one_text_block_test!(openai_responses, openai_responses::fixture);
    exactly_one_text_block_test!(gemini_rest, gemini_rest::fixture);
    exactly_one_text_block_test!(cohere, cohere::fixture);
    exactly_one_text_block_test!(ollama, ollama::fixture);
    exactly_one_text_block_test!(anthropic, anthropic::fixture);
    exactly_one_text_block_test!(interactions, interactions::fixture);
    exactly_one_text_block_test!(xai, super::xai::fixture);
    exactly_one_text_block_test!(copilot_responses, super::copilot::responses_fixture);
    exactly_one_text_block_test!(copilot_chat, super::copilot::chat_fixture);

    /// The item-4 identity pin (#2258): two `message` output items on the
    /// OpenAI Responses wire carry distinct `item_id`s and must aggregate as
    /// two distinct text parts, in wire order — before TextStart carried
    /// identity, their deltas concatenated boundary-less into one block.
    #[tokio::test]
    async fn two_message_items_aggregate_as_two_text_parts_on_the_responses_wire() {
        use serde_json::json;

        let delta = |item_id: &str, text: &str, sequence: u64| {
            sse(&json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": text,
                "item_id": item_id,
                "output_index": if item_id == "msg_1" { 0 } else { 1 },
                "sequence_number": sequence,
            }))
        };
        let frames = vec![
            delta("msg_1", "first ", 1),
            delta("msg_1", "message", 2),
            delta("msg_2", "second ", 3),
            delta("msg_2", "message", 4),
            sse(&json!({
                "type": "response.completed",
                "sequence_number": 5,
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                    "tools": [],
                    "usage": null,
                },
            })),
        ];

        let driver = openai_responses::driver();
        let drained = driver
            .drive(conformance::ok_chunks(frames))
            .await
            .expect("stream should drive");
        assert_eq!(
            drained.choice_texts(),
            vec!["first message", "second message"],
            "two message items must aggregate as two distinct text parts, in order"
        );
    }

    /// The item-0 identity pin (#2258) on the wire: several llama.cpp/vllm-
    /// style chat-compat gateways stream tool-call deltas with NO ids, keyed
    /// by chunk index alone. Two such calls interleaving their argument
    /// fragments must aggregate as two distinct uncorrupted calls under
    /// boundary-minted `tool-{index}` ids — a shared empty grammar id would
    /// interleave both calls' fragments into one garbled argument buffer.
    #[tokio::test]
    async fn id_less_parallel_tool_calls_assemble_distinct_on_the_chat_wire() {
        use rig_core::message::AssistantContent;
        use serde_json::json;

        let chunk = |tool_calls: serde_json::Value| {
            sse(&json!({
                "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": null}],
            }))
        };
        let frames = vec![
            chunk(
                json!([{"index": 0, "type": "function", "function": {"name": "get_weather", "arguments": ""}}]),
            ),
            chunk(
                json!([{"index": 1, "type": "function", "function": {"name": "get_time", "arguments": ""}}]),
            ),
            chunk(json!([{"index": 0, "function": {"arguments": "{\"city\":"}}])),
            chunk(json!([{"index": 1, "function": {"arguments": "{\"zone\":"}}])),
            chunk(json!([{"index": 0, "function": {"arguments": "\"Tokyo\"}"}}])),
            chunk(json!([{"index": 1, "function": {"arguments": "\"UTC\"}"}}])),
            sse(&json!({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": null,
            })),
            conformance::WireInput::Bytes(bytes::Bytes::from("data: [DONE]\n\n".to_string())),
        ];

        let fixture = openai_chat::fixture();
        let drained = fixture
            .driver
            .drive(conformance::ok_chunks(frames))
            .await
            .expect("stream should drive");
        assert_eq!(
            drained.error_count(),
            0,
            "no frame may error: {:?}",
            drained.items
        );

        let calls: Vec<_> = drained
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(tool_call) => Some(tool_call),
                _ => None,
            })
            .collect();
        let shape: Vec<(&str, &str, &serde_json::Value)> = calls
            .iter()
            .map(|call| {
                (
                    call.id.as_str(),
                    call.function.name.as_str(),
                    &call.function.arguments,
                )
            })
            .collect();
        assert_eq!(
            shape,
            vec![
                ("tool-0", "get_weather", &json!({"city": "Tokyo"})),
                ("tool-1", "get_time", &json!({"zone": "UTC"})),
            ],
            "two id-less indexed calls must assemble distinct and uncorrupted"
        );
    }

    /// Sibling-reasoning supersede on the wire (the corpus counterpart of the
    /// parts.rs `deltas_then_sibling_full_blocks_keep_each_part_once` unit
    /// test): a summary delta followed by its item's multi-part done block —
    /// the first part supersedes the delta accumulation (no duplicate), the
    /// remaining sibling parts append once each, in wire order.
    #[tokio::test]
    async fn sibling_reasoning_supersede_on_the_responses_wire() {
        use rig_core::message::ReasoningContent;
        use serde_json::json;

        let driver = openai_responses::driver();
        let frames = vec![
            sse(&json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "sequence_number": 1,
                "delta": "s1",
            })),
            sse(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 2,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [
                        {"type": "summary_text", "text": "s1"},
                        {"type": "summary_text", "text": "s2"},
                    ],
                    "content": [],
                    "status": "completed",
                    "encrypted_content": "enc",
                },
            })),
            sse(&json!({
                "type": "response.completed",
                "sequence_number": 3,
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [],
                    "tools": [],
                    "usage": null,
                },
            })),
        ];

        let drained = driver
            .drive(conformance::ok_chunks(frames))
            .await
            .expect("stream should drive");
        let reasoning_texts: Vec<String> = drained
            .choice_reasoning()
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .map(|content| match content {
                ReasoningContent::Summary(text) => text.clone(),
                ReasoningContent::Text { text, .. } => text.clone(),
                ReasoningContent::Encrypted(data) => data.clone(),
                ReasoningContent::Redacted { data } => data.clone(),
                // `ReasoningContent` is non-exhaustive; no other variant is
                // reachable from these frames.
                _ => String::new(),
            })
            .collect();
        assert_eq!(
            reasoning_texts,
            ["s1", "s2", "enc"],
            "the first sibling supersedes the delta build; the rest append once each"
        );
    }
}

// The terminal-body/delta per-kind merge lives on the buffered-body pipeline
// (the ChatGPT backend re-parses the SSE body after the fact); the live
// streaming path delivers message text only through deltas.
#[tokio::test]
async fn terminal_body_content_merges_per_kind() {
    let driver = openai_responses::buffered_driver();
    conformance::terminal_body_content_merges_per_kind(
        &driver,
        vec![
            (
                "body-only",
                openai_responses::terminal_body_only_sse_body("from body"),
            ),
            (
                "body+delta",
                openai_responses::terminal_body_and_delta_sse_body("from body"),
            ),
            (
                "delta-only",
                openai_responses::delta_only_sse_body("from body"),
            ),
        ],
        "from body",
    )
    .await
    .expect("scenario should hold");
}

// Pins #2258 F3 on the real buffered pipeline: an envelope-less reasoning
// delta (ChatGPT replay; repair injects `output_index: 0`, minting the
// `output-0` identity) followed by the item's envelope-full
// `output_item.done` must aggregate the restated summary exactly once — the
// done item's blocks adopt the minted per-slot identity instead of their
// `rs_*` id, superseding the delta build rather than appending beside it.
#[tokio::test]
async fn envelope_less_reasoning_deltas_are_superseded_without_duplication() {
    use rig_core::message::{AssistantContent, ReasoningContent};

    let driver = openai_responses::buffered_driver();
    let (body, summary) = openai_responses::envelope_less_reasoning_supersede_sse_body();
    let choice = driver.drive(body).await.expect("replay should normalize");

    let reasoning: Vec<_> = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning.len(),
        1,
        "deltas and their done block must collapse to one reasoning item: {reasoning:?}"
    );
    let occurrences = reasoning
        .iter()
        .flat_map(|item| item.content.iter())
        .filter(|content| match content {
            ReasoningContent::Summary(text) | ReasoningContent::Text { text, .. } => {
                text.contains(summary)
            }
            _ => false,
        })
        .count();
    assert_eq!(
        occurrences, 1,
        "the restated summary must supersede its deltas, not duplicate them"
    );
}

// Reasoning sequence families. Only the OpenAI Responses wire spells
// multi-part reasoning items, so these run against that fixture alone.
mod reasoning {
    use super::*;

    // Pins P1-1 of `rig-2257-code-review-findings-34ee8ba5.md`: the
    // `reasoning_summary_text.delta` handler drops `item_id`, so the strict
    // same-item table appends the full block beside the delta-built item and
    // every o-series reasoning-summary stream aggregates the summary twice.
    #[tokio::test]
    async fn summary_deltas_are_superseded_without_duplication() {
        let driver = openai_responses::driver();
        let (frames, summary) = openai_responses::reasoning_summary_supersede_frames();
        conformance::reasoning_summary_deltas_are_superseded_without_duplication(
            &driver, frames, summary,
        )
        .await
        .expect("scenario should hold");
    }

    // Pins P1-2 of `rig-2257-code-review-findings-34ee8ba5.md`: the
    // `rposition` by-id fallback replaces the just-appended same-id sibling,
    // so a multi-part reasoning item survives only as its last part.
    #[tokio::test]
    async fn multi_part_same_id_reasoning_keeps_every_part() {
        let driver = openai_responses::driver();
        let (frames, expected) = openai_responses::multi_part_reasoning_frames();
        conformance::multi_part_same_id_reasoning_keeps_every_part(&driver, frames, &expected)
            .await
            .expect("scenario should hold");
    }

    #[tokio::test]
    async fn interleaved_reasoning_aggregates_to_one_item() {
        let driver = openai_responses::driver();
        let (frames, expected) = openai_responses::interleaved_reasoning_frames();
        conformance::interleaved_reasoning_aggregates_to_one_item(&driver, frames, expected)
            .await
            .expect("scenario should hold");
    }
}

// The F1b boundary family (#2258 review): constant-id wires mint a per-stream
// reasoning identity, so the only interleaving signal is other output — main's
// "other output closes the reasoning item" boundary must survive aggregation.
// The Responses fixture stays on `interleaved_reasoning_aggregates_to_one_item`
// above: real per-item ids must keep collapsing across boundaries.
mod interleaved_constant_id_reasoning {
    use super::*;

    #[tokio::test]
    async fn gemini_rest_preserves_order_around_a_tool_call() {
        let driver = gemini_rest::fixture().driver;
        let (frames, first, tool, second) = gemini_rest::interleaved_thought_frames();
        conformance::interleaved_constant_id_reasoning_preserves_order(
            &driver, frames, first, tool, second,
        )
        .await
        .expect("scenario should hold");
    }

    #[tokio::test]
    async fn gemini_rest_signed_full_does_not_erase_prior_thought() {
        let driver = gemini_rest::fixture().driver;
        let (frames, first, tool, second) = gemini_rest::interleaved_signed_thought_frames();
        conformance::interleaved_signed_full_reasoning_does_not_erase_prior_thought(
            &driver, frames, first, tool, second,
        )
        .await
        .expect("scenario should hold");
    }

    #[tokio::test]
    async fn interactions_preserves_order_around_a_tool_call() {
        let driver = interactions::fixture().driver;
        let (frames, first, tool, second) = interactions::interleaved_thought_frames();
        conformance::interleaved_constant_id_reasoning_preserves_order(
            &driver, frames, first, tool, second,
        )
        .await
        .expect("scenario should hold");
    }

    #[tokio::test]
    async fn ollama_preserves_order_around_a_tool_call() {
        let driver = ollama::fixture().driver;
        let (frames, first, tool, second) = ollama::interleaved_thinking_frames();
        conformance::interleaved_constant_id_reasoning_preserves_order(
            &driver, frames, first, tool, second,
        )
        .await
        .expect("scenario should hold");
    }
}
