//! Canonical streaming-grammar coverage for the OpenAI Responses API,
//! asserted through the *normalized* path: the aggregated
//! `choice` from `CompletionStream::finish`, the terminal `CompletionResponse`
//! record, usage, IDs, and finish reason — real recorded wire traffic, not
//! synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record OPENAI_API_KEY=... cargo test --test openai streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders (prefixes such as `msg_`/`resp_` are
//! preserved); assertions derive expected IDs from the recorded turn and never
//! mint literal IDs.

use futures::StreamExt;
use rig::completion::CompletionResponse;
use rig::completion::FinishReason;
use rig::message::{
    AssistantContent, AssistantMessage, Message, Reasoning, ToolCall, ToolResultContent,
    UserContent,
};
use rig::providers::openai;
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde_json::json;

use rig::completion::Effort;
use rig::providers::openai::extension::{
    Include, OpenAiOptions, OpenAiResponsesOptions, OpenAiShared,
};

use super::super::support::{
    effort, openai_options, shared_options, stateless, with_openai_cassette,
};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
};
use rig::completion::CompletionRequest;

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
struct StreamRun {
    /// Streamed text deltas, concatenated.
    text: String,
    /// Full reasoning blocks yielded as stream events, in order.
    reasoning_blocks: Vec<Reasoning>,
    /// Concatenated reasoning-delta text.
    reasoning_delta: String,
    /// Complete tool calls yielded as stream events, in order.
    tool_calls: Vec<ToolCall>,
    /// Terminal records yielded by the stream.
    /// The aggregated choice built by the normalized stream.
    choice: Vec<AssistantContent>,
    /// The normalized terminal record retained on the stream.
    response: Option<CompletionResponse>,
}

impl StreamRun {
    /// The turn's origin and stop, for the history its blocks go back in.
    fn head(&self) -> AssistantMessage {
        self.response
            .as_ref()
            .map(CompletionResponse::head)
            .unwrap_or_default()
    }
}

async fn drain_stream(mut stream: rig::streaming::CompletionStream) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_blocks: Vec::new(),
        reasoning_delta: String::new(),
        tool_calls: Vec::new(),
        choice: vec![AssistantContent::text("")],
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            Item::Event(StreamEvent::Text { text, .. }) => run.text.push_str(&text),
            Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) => {
                run.reasoning_blocks.push(reasoning);
            }
            Item::Event(StreamEvent::Reasoning {
                text: reasoning, ..
            }) => {
                run.reasoning_delta.push_str(&reasoning);
            }
            Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            }) => {
                run.tool_calls.push(tool_call);
            }
            _ => {}
        }
    }
    let response = stream.finish().await.expect("the stream ends");

    run.choice = response.choice.clone();
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
    run.response = Some(response);
    run
}

fn assert_terminal(run: &StreamRun, expected_finish: FinishReason) {
    let terminal = run
        .response
        .as_ref()
        .expect("aggregated stream should retain the terminal record");
    assert_eq!(
        terminal.finish_reason(),
        Some(expected_finish),
        "unexpected finish reason"
    );
    assert!(
        terminal.usage.total_tokens.is_some_and(|n| n > 0),
        "terminal record should carry non-zero usage, got {:?}",
        terminal.usage
    );
    // ID contract: the Responses API names both the response (`resp_`) and the
    // assistant output message (`msg_`); prefixes survive cassette scrubbing.
    let response_id = terminal
        .response_id()
        .expect("Responses API should report a response-scoped ID");
    assert!(
        response_id.starts_with("resp_"),
        "response_id should be response-scoped, got {response_id}"
    );
    let message_ids = run
        .choice
        .iter()
        .filter(|content| matches!(content, AssistantContent::Text(_)))
        .filter_map(|content| content.native_item()?["id"].as_str());
    for message_id in message_ids {
        assert!(
            message_id.starts_with("msg_") || message_id.starts_with("rs_"),
            "message_id should be an output-item ID, got {message_id}"
        );
        assert_ne!(
            message_id, response_id,
            "message-scoped and response-scoped IDs must not be conflated"
        );
    }
}

/// Parallel tool calls streamed in one turn: both calls must land in the
/// aggregated choice and the terminal record must report `ToolCalls`.
#[tokio::test]
async fn parallel_tool_calls_both_survive_aggregation() {
    with_openai_cassette(
        "streaming_grammar/parallel_tool_calls",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request = CompletionRequest::new(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .options(effort(Effort::Low).parallel_tool_calls(true));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);

            let aggregated_calls: Vec<&ToolCall> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .collect();
            for name in ["lookup_harbor_label", "lookup_orchard_label"] {
                let streamed = run
                    .tool_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("stream should yield a {name} call"));
                let aggregated = aggregated_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
                // IDs derived from the recorded turn, never minted literally.
                assert_eq!(aggregated.id, streamed.id, "{name} id should aggregate");
            }
            assert_eq!(
                aggregated_calls.len(),
                2,
                "aggregated choice should contain exactly the two parallel calls"
            );
        },
    )
    .await;
}

/// Tool call then follow-up text across turns: turn one ends in `ToolCalls`
/// with the call aggregated; the follow-up turn (fed the tool result) ends in
/// `Stop` with text that uses the result.
#[tokio::test]
async fn tool_call_then_followup_text_across_turns() {
    with_openai_cassette(
        "streaming_grammar/tool_then_followup_text",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request = CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .options(effort(Effort::Low))
                .provider_options(stateless());
            let first = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&first, FinishReason::ToolCalls);
            let tool_call = first
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == "lookup_harbor_label" =>
                    {
                        Some(tool_call.clone())
                    }
                    _ => None,
                })
                .expect("aggregated first turn should contain the lookup_harbor_label call");

            let assistant_message = Message::Assistant(
                first
                    .head()
                    .with_content(vec![AssistantContent::ToolCall(tool_call.clone())]),
            );
            let tool_result = Message::from(UserContent::tool_result(
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
            ));
            let followup_request = CompletionRequest::new(
                "Now reply in one short sentence using the provided tool result. \
                     Do not call any tools.",
            )
            .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
            .messages(vec![
                Message::user(ORDERED_TOOL_STREAM_PROMPT),
                assistant_message,
                tool_result,
            ])
            .options(effort(Effort::Low))
            .provider_options(stateless());
            let second = drain_stream(
                model
                    .stream(followup_request)
                    .expect("follow-up stream should start"),
            )
            .await;

            assert_terminal(&second, FinishReason::Stop);
            assert!(
                second.tool_calls.is_empty(),
                "follow-up turn should not call tools"
            );
            assert!(
                second.text.contains(ALPHA_SIGNAL_OUTPUT),
                "follow-up text should use the tool result, got {:?}",
                second.text
            );
        },
    )
    .await;
}

/// Three-turn tool session with encrypted reasoning (`store: false`): every
/// turn's reasoning items carry real `rs_*` ids that are sent back verbatim on
/// the following turn, exercising the Responses provenance gate on real ids
/// across turns (reasoning + tool call → tool result → follow-up reasoning +
/// text → follow-up text).
#[tokio::test]
async fn three_turn_tool_session_replays_rs_ids_across_turns() {
    with_openai_cassette(
        "streaming_grammar/three_turn_tool_session",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            // A trivial tool turn skips the reasoning item entirely on the
            // wire (verified via a direct probe, even at medium effort); a
            // math sub-task at high effort reliably yields the `rs_*`
            // reasoning item the provenance assertions need.
            const FIRST_TURN_PROMPT: &str =
                "First work out how many positive integers n < 90 are divisible by 7 \
                 (do this carefully). Then call `lookup_harbor_label` exactly once. After \
                 the tool result arrives, answer with the count and the exact tool output.";
            let generation = effort(Effort::High);
            let provider = openai_options(
                &OpenAiOptions::new()
                    .shared(OpenAiShared::default().store(false))
                    .responses(
                        OpenAiResponsesOptions::default()
                            .include([Include::ReasoningEncryptedContent]),
                    ),
            );

            // Turn 1: forced tool work with encrypted reasoning.
            let request = CompletionRequest::new(FIRST_TURN_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .options(generation.clone())
                .provider_options(provider.clone());
            let first =
                drain_stream(model.stream(request).expect("stream should start")).await;
            assert_terminal(&first, FinishReason::ToolCalls);

            // The reasoning item the wire produced carries a real `rs_*` id;
            // the id is derived from the recorded turn, never minted.
            let reasoning_ids: Vec<&str> = first
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Reasoning(reasoning) => {
                        reasoning.native.as_ref()?.item["id"].as_str()
                    }
                    _ => None,
                })
                .collect();
            assert!(
                !reasoning_ids.is_empty(),
                "store:false encrypted turn should aggregate a reasoning part with an id, got {:?}",
                first.choice
            );
            for id in &reasoning_ids {
                assert!(
                    id.starts_with("rs_"),
                    "reasoning part should carry the wire's rs_* id, got {id}"
                );
            }
            let tool_call = first
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(call)
                        if call.function.name == "lookup_harbor_label" =>
                    {
                        Some(call.clone())
                    }
                    _ => None,
                })
                .expect("aggregated first turn should contain the lookup_harbor_label call");

            // Turn 2: the full aggregated choice — reasoning items with their
            // recorded rs_* ids included — goes back through the provenance
            // gate together with the tool result.
            let first_assistant = Message::Assistant(first.head().with_content(first.choice.clone()));
            let tool_result = Message::from(UserContent::tool_result(tool_call.id.clone(), tool_call.function.name.clone(), vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)]));
            let second_request = CompletionRequest::new(
                    "Answer in one short sentence that includes the exact tool output. \
                     Do not call any tools.",
                )
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .messages(vec![
                    Message::user(FIRST_TURN_PROMPT),
                    first_assistant.clone(),
                    tool_result.clone(),
                ])
                .options(generation.clone())
                .provider_options(provider.clone());
            let second = drain_stream(
                model
                    .stream(second_request)
                    .expect("second-turn stream should start"),
            )
            .await;
            assert_terminal(&second, FinishReason::Stop);
            assert!(
                second.text.contains(ALPHA_SIGNAL_OUTPUT),
                "second turn should use the tool result, got {:?}",
                second.text
            );

            // Turn 3: both prior assistant turns' rs_* items replay together.
            let second_assistant = Message::Assistant(second.head().with_content(second.choice.clone()));
            let third_request = CompletionRequest::new(
                    "Repeat the exact tool output one more time, alone on a single line.",
                )
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .messages(vec![
                    Message::user(FIRST_TURN_PROMPT),
                    first_assistant,
                    tool_result,
                    Message::user(
                        "Answer in one short sentence that includes the exact tool output. \
                         Do not call any tools.",
                    ),
                    second_assistant,
                ])
                .options(generation)
                .provider_options(provider);
            let third = drain_stream(
                model
                    .stream(third_request)
                    .expect("third-turn stream should start"),
            )
            .await;
            assert_terminal(&third, FinishReason::Stop);
            assert!(
                third.text.contains(ALPHA_SIGNAL_OUTPUT),
                "third turn should still carry the tool output, got {:?}",
                third.text
            );
            assert!(
                third.tool_calls.is_empty(),
                "third turn should not call tools"
            );
        },
    )
    .await;
}

/// `response.incomplete` cut mid-tool-call: forced tool use with a minimal
/// `max_output_tokens` budget. The stream must still terminate cleanly with a
/// `Length` finish, and any tool call that did surface must carry object
/// arguments (never a corrupted fragment).
///
/// The cassette captures the wire's genuine incomplete shape: the function
/// call's `arguments` are cut mid-JSON (`"arguments":"{\"x\":48151"`, item
/// and response status `incomplete`). The typed models keep `arguments` as
/// the raw wire string (`FunctionCallArguments`) and parse at consumption
/// time, so both the `response.output_item.done` and `response.incomplete`
/// frames decode; the truncation policy drops the partial call instead of
/// fabricating one, and the terminal normalizes to `Length`.
#[tokio::test]
async fn incomplete_mid_tool_call_normalizes_to_length() {
    with_openai_cassette(
        "streaming_grammar/incomplete_mid_tool_call",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request = CompletionRequest::new(
                "Add 48151.62342 and 27182.81828 using the add tool. You must call the tool.",
            )
            .tool(rig::tool::tool_definition(&Adder))
            .tool_choice(rig::message::ToolChoice::Required)
            .max_tokens(16)
            .options(effort(Effort::Low));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Length);
            // Whatever partial output survived must be well-formed part-wise:
            // a truncated tool call keeps what its arguments state, as an
            // object, with the text they arrived as beside it.
            for call in &run.tool_calls {
                assert!(
                    call.function
                        .arguments
                        .values()
                        .all(|value| !value.is_null())
                        && call.function.invalid_arguments.is_some(),
                    "a cut-off call keeps an object and its raw text, got {:?}",
                    call.function
                );
            }
            for content in run.choice.iter() {
                if let AssistantContent::ToolCall(call) = content {
                    assert!(
                        call.function.invalid_arguments.is_some(),
                        "an aggregated cut-off call keeps its raw text beside its object, got {:?}",
                        call.function
                    );
                }
            }
        },
    )
    .await;
}

/// Structured-output streaming (`text.format` json_schema): the streamed text
/// parses as the requested schema and matches the aggregated text part.
#[tokio::test]
async fn structured_output_stream_yields_schema_conformant_text() {
    with_openai_cassette(
        "streaming_grammar/structured_output_stream",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request = CompletionRequest::new(
                "Return a concise event object for a local Rust meetup in Seattle.",
            )
            .options(effort(Effort::Low))
            .additional_params(json!({
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "smoke_event",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "category": { "type": "string" },
                                "summary": { "type": "string" }
                            },
                            "required": ["title", "category", "summary"],
                            "additionalProperties": false
                        }
                    }
                }
            }));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Stop);
            let parsed: serde_json::Value = serde_json::from_str(run.text.trim())
                .expect("structured-output stream should yield valid JSON text");
            for field in ["title", "category", "summary"] {
                assert!(
                    parsed
                        .get(field)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty()),
                    "structured output should carry a non-empty {field}, got {parsed}"
                );
            }
            // The aggregated text part is exactly the streamed text — one
            // discrete text part, not a re-concatenation.
            let aggregated_text_parts: Vec<&str> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                aggregated_text_parts.concat(),
                run.text,
                "aggregated text should match the streamed structured output"
            );
        },
    )
    .await;
}

/// `previous_response_id`-chained turn: turn one is stored, turn two chains
/// off its recorded `resp_*` id (derived from the turn, never minted) and can
/// see the earlier turn's content.
#[tokio::test]
async fn previous_response_id_chains_server_side_state() {
    with_openai_cassette(
        "streaming_grammar/previous_response_id_chain",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request = CompletionRequest::new("Reply with exactly one word: quartz")
                .options(effort(Effort::Low))
                .provider_options(shared_options(Some(true), None));
            let first = drain_stream(model.stream(request).expect("stream should start")).await;
            assert_terminal(&first, FinishReason::Stop);
            assert!(
                first.text.to_ascii_lowercase().contains("quartz"),
                "first turn should echo the word, got {:?}",
                first.text
            );
            let previous_response_id = first
                .response
                .as_ref()
                .and_then(|terminal| terminal.response_id().map(str::to_owned))
                .expect("stored turn should report a resp_* id");

            let second_request = CompletionRequest::new(
                "What exact word did you reply with just now? Reply with only that word.",
            )
            .options(effort(Effort::Low))
            .provider_options(shared_options(Some(true), None))
            .additional_params(json!({ "previous_response_id": previous_response_id }));
            let second = drain_stream(
                model
                    .stream(second_request)
                    .expect("chained stream should start"),
            )
            .await;
            assert_terminal(&second, FinishReason::Stop);
            assert!(
                second.text.to_ascii_lowercase().contains("quartz"),
                "chained turn should see the stored turn's content, got {:?}",
                second.text
            );
            let second_id = second
                .response
                .as_ref()
                .and_then(|terminal| terminal.response_id())
                .expect("chained turn should report its own resp_* id");
            assert_ne!(
                second_id,
                first
                    .response
                    .as_ref()
                    .and_then(|terminal| terminal.response_id())
                    .expect("first turn id"),
                "each turn carries its own response-scoped id"
            );
        },
    )
    .await;
}

/// `response.incomplete` via a small `max_output_tokens` budget: the stream
/// still terminates with a terminal record, the finish reason normalizes to
/// `Length`, and whatever partial output arrived is kept.
#[tokio::test]
async fn incomplete_max_output_tokens_normalizes_to_length() {
    with_openai_cassette(
        "streaming_grammar/incomplete_max_output_tokens",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6);
            let request =
                CompletionRequest::new("Write a 300-word essay about the history of lighthouses.")
                    .max_tokens(32)
                    .options(effort(Effort::Low));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Length);
            // Partial content is kept: whatever the stream surfaced before the
            // cutoff (text and/or reasoning) must match the aggregated choice.
            let aggregated_text: String = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                aggregated_text, run.text,
                "aggregated choice should keep exactly the streamed partial text"
            );
        },
    )
    .await;
}
