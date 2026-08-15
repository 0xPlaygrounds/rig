//! Edge matrix for the premature stream-terminal bug.
//!
//! # The bug
//!
//! `GeminiRestAdapter::interpret` treated *any* chunk carrying a
//! `finishReason` as the provider completing the turn and pushed
//! `RawStreamingChoice::FinalResponse` there; the shared driver stops reading
//! as soon as it sees a terminal record. Gemini's `streamGenerateContent`
//! does not honour that assumption: when a built-in tool runs a round it
//! emits an **intermediate** `finishReason` and keeps streaming. A recorded
//! two-round code-execution turn reads
//!
//! ```text
//! [executableCode] [codeExecutionResult] [executableCode + finishReason:STOP]
//! [codeExecutionResult] [text] [text + finishReason:STOP]
//! ```
//!
//! so rig ended the stream on frame 3 and dropped the model's entire answer —
//! while still yielding a terminal record that reported a clean `STOP`. The
//! blocking twin of the same request returns the full answer.
//!
//! The fix defers the terminal record to EOF, which the adapter contract
//! names as deferral rather than synthesis ("a terminal the provider *did*
//! signal earlier … may be emitted here"). EOF with no `finishReason` at all
//! is still truncation and still yields no terminal.
//!
//! # The matrix
//!
//! The fixed code path's inputs are: how many `finishReason` chunks arrive,
//! what follows the first one, which terminal metadata each chunk carries,
//! how the stream ends (EOF / transport error / in-band failure), and which
//! stream entry point the caller used.
//!
//! | # | cell | kind | dimension pinned |
//! |---|------|------|------------------|
//! | 1 | `two_terminal_stream_keeps_the_text_after_the_first_finish` | recorded | the bug itself |
//! | 2 | `two_terminal_stream_blocking_twin_has_the_same_answer` | recorded | the blocking yardstick cell 1 is measured against |
//! | 3 | `two_terminal_stream_agent_prompt_keeps_the_answer` | recorded | `Agent::stream_prompt` |
//! | 4 | `two_terminal_stream_terminal_carries_the_last_usage` | recorded | terminal metadata comes from the last chunk |
//! | 5 | `two_terminal_stream_with_visible_thoughts` | recorded | reasoning spanning the boundary |
//! | 6 | `gemini_3_flash_does_not_emit_the_intermediate_finish` | recorded | second model family: control, shape absent |
//! | 7 | `two_terminal_stream_through_raw_stream` | recorded | the provider-native `raw_stream` entry |
//! | 8 | `two_terminal_stream_unicode_answer_after_the_first_finish` | recorded | multi-byte text after the boundary |
//! | 9 | `single_terminal_text_stream_is_unchanged` | recorded | regression guard: ordinary stream |
//! | 10 | `single_terminal_tool_call_stream_is_unchanged` | recorded | regression guard: tool-call stream |
//! | 11 | `max_tokens_truncated_stream_still_reports_length` | recorded | a non-`STOP` terminal reason |
//! | 12 | `thinking_stream_terminal_is_unchanged` | recorded | regression guard: reasoning stream |
//! | 13 | `an_intermediate_finish_reason_does_not_end_the_stream` | unit | minimal statement of the rule |
//! | 14 | `the_last_finish_reason_wins_over_an_earlier_one` | unit | STOP→MAX_TOKENS ordering |
//! | 15 | `a_usage_only_trailer_after_the_finish_reason_reaches_the_terminal` | unit | trailing usage chunk |
//! | 16 | `later_metadata_wins_on_the_terminal_record` | unit | responseId/modelVersion from the last chunk |
//! | 17 | `three_finish_reason_chunks_still_yield_one_terminal` | unit | exactly-one-terminal invariant |
//! | 18 | `eof_without_any_finish_reason_yields_no_terminal_record` | unit | truncation is still truncation |
//! | 19 | `a_transport_error_after_a_finish_reason_yields_no_terminal_record` | unit | error path |
//! | 20 | `a_tool_protocol_failure_still_ends_the_stream_immediately` | unit | in-band failure still short-circuits |
//! | 21 | `a_tool_call_after_the_first_finish_reason_reaches_the_choice` | unit | tool content past the boundary |
//! | 22 | `reasoning_after_the_first_finish_reason_reaches_the_choice` | unit | reasoning past the boundary |
//! | 23 | `an_unknown_frame_after_the_first_finish_reason_is_passed_through` | unit | passthrough past the boundary |
//! | 24 | `the_terminal_record_is_the_last_item_yielded` | unit | ordering invariant |
//! | 25 | `a_transport_error_after_the_real_terminal_also_reports_truncation` | unit | the deliberate trade-off, stated |
//!
//! Cells 13–25 drive synthetic SSE frames through the real client rather than
//! recording, because the provider cannot be asked for these shapes: Gemini
//! never emits `STOP` followed by `MAX_TOKENS`, never sends a usage-only
//! trailer *after* an intermediate finish, and cannot be made to fail its
//! transport mid-turn on demand. Each frame sequence is written from the
//! bytes recorded by cells 1–12.
//!
//! Recording note: the two-terminal shape needs the model to take **two**
//! code-execution rounds. Measured 4/4 with the prompt below and thinking at
//! its default; forcing `thinkingBudget: 0` makes gemini-2.5-flash narrate
//! the code as text instead of calling the tool, and the shape never appears.
//! `assert_recorded_stream_finishes_early` asserts the shape's presence (or,
//! for cell 6, its absence) against each fixture's own bytes, so a cell that
//! stopped covering what it claims fails instead of quietly passing.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test -p rig --all-features --test gemini stream_terminal_matrix -- --test-threads=1`

use futures::StreamExt;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use serde_json::{Value, json};

use super::super::support::{
    assert_recorded_stream_finishes_early, last_frame_total_tokens,
    with_gemini_stream_terminal_cassette,
};

/// The prompt that reliably makes Gemini take two code-execution rounds, and
/// therefore emit an intermediate `finishReason`.
const TWO_ROUND_PROMPT: &str = "You must call the code execution tool twice as two separate executions. \
     Execution 1: run only print(987654321 * 123456789). Then, after seeing that number, \
     Execution 2: run only print(sum(int(d) for d in str(N))) where N is the exact number \
     from Execution 1. Never combine them. Finally state both numbers.";

/// The product printed by the first round, which the answer must restate.
const FIRST_ROUND_VALUE: &str = "121932631112635269";

fn code_execution_params() -> Value {
    json!({ "tools": [{ "codeExecution": {} }] })
}

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Whether `text` states `value`.
///
/// Digit grouping the model may add is ignored ("1,048,576" states
/// "1048576"), but only separators *between* digits are removed, so
/// `"items 12, 345"` does not silently become the number `12345`. A numeric
/// `value` must not be a fragment of a longer number — "2880" in "28800" is
/// not the answer — so digit-adjacency is rejected. An empty or non-numeric
/// `value` keeps plain substring semantics.
fn states(text: &str, value: &str) -> bool {
    let text: String = text
        .char_indices()
        .filter(|(index, ch)| {
            !(matches!(ch, ',' | '_')
                && text[..*index]
                    .chars()
                    .next_back()
                    .is_some_and(|previous| previous.is_ascii_digit())
                && text[index + ch.len_utf8()..]
                    .chars()
                    .next()
                    .is_some_and(|next| next.is_ascii_digit()))
        })
        .map(|(_, ch)| ch)
        .collect();

    if value.is_empty() || !value.chars().all(|ch| ch.is_ascii_digit()) {
        return text.contains(value);
    }
    text.match_indices(value).any(|(index, _)| {
        let before_ok = text[..index]
            .chars()
            .next_back()
            .is_none_or(|ch| !ch.is_ascii_digit());
        let after_ok = text[index + value.len()..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_digit());
        before_ok && after_ok
    })
}

/// Everything a drained normalized stream is asserted about here.
struct Drained {
    text: String,
    choice: Vec<AssistantContent>,
    terminals: usize,
    terminal: Option<rig::streaming::StreamFinal>,
    last_item_was_terminal: bool,
}

async fn drain(mut stream: rig::streaming::StreamingCompletionResponse) -> Drained {
    let mut text = String::new();
    let mut terminals = 0;
    let mut last_item_was_terminal = false;
    while let Some(item) = stream.next().await {
        let item = item.expect("no stream item should be an error");
        last_item_was_terminal = matches!(item, StreamedAssistantContent::Final(_));
        match item {
            StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
            StreamedAssistantContent::Final(_) => terminals += 1,
            _ => {}
        }
    }
    Drained {
        text,
        choice: stream.choice.clone(),
        terminals,
        terminal: stream.response.clone(),
        last_item_was_terminal,
    }
}

// --- 1-8: the bug, over the shapes live traffic produces ------------------

#[tokio::test]
async fn two_terminal_stream_keeps_the_text_after_the_first_finish() {
    const SCENARIO: &str =
        "stream_terminal_matrix/two_terminal_stream_keeps_the_text_after_the_first_finish";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_keeps_the_text_after_the_first_finish",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                states(&drained.text, FIRST_ROUND_VALUE),
                "the answer arrives only AFTER the intermediate finishReason; ending the stream \
             there dropped it entirely. got {:?}",
                drained.text
            );
            assert_eq!(
                drained.terminals, 1,
                "exactly one terminal record, however many finishReason chunks the wire sent"
            );
            assert!(
                drained.last_item_was_terminal,
                "the terminal record must still be the last item yielded"
            );
            assert_eq!(
                text_of(&drained.choice),
                drained.text,
                "the aggregated choice must equal the streamed deltas"
            );
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

#[tokio::test]
async fn two_terminal_stream_blocking_twin_has_the_same_answer() {
    // The yardstick, not a comparison: the blocking transport never had this
    // bug, so recording its answer to the same request fixes what cell 1's
    // stream is required to produce. Both assert the same
    // `FIRST_ROUND_VALUE`, which is what ties them together.
    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_blocking_twin_has_the_same_answer",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let response = CompletionModel::completion(&model, request)
                .await
                .expect("blocking turn should convert");
            assert!(
                states(&text_of(&response.choice), FIRST_ROUND_VALUE),
                "the blocking twin must carry the answer, got {:?}",
                text_of(&response.choice)
            );
        },
    )
    .await;
}

#[tokio::test]
async fn two_terminal_stream_agent_prompt_keeps_the_answer() {
    const SCENARIO: &str =
        "stream_terminal_matrix/two_terminal_stream_agent_prompt_keeps_the_answer";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_agent_prompt_keeps_the_answer",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let mut stream = agent.stream_prompt(TWO_ROUND_PROMPT).await;
            let mut answer = String::new();
            while let Some(item) = stream.next().await {
                if let rig::agent::MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::Text(text),
                ) = item.expect("no stream item should be an error")
                {
                    answer.push_str(&text.text);
                }
            }

            assert!(
                states(&answer, FIRST_ROUND_VALUE),
                "the agent's streamed answer must survive the intermediate finishReason, got \
             {answer:?}"
            );
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

#[tokio::test]
async fn two_terminal_stream_terminal_carries_the_last_usage() {
    const SCENARIO: &str =
        "stream_terminal_matrix/two_terminal_stream_terminal_carries_the_last_usage";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_terminal_carries_the_last_usage",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            let terminal = drained.terminal.expect("the turn should complete");
            // Gemini's usage is cumulative per chunk, so a terminal built at
            // the *first* finishReason under-reported the turn. Asserting the
            // exact last-frame total is what makes this cell about last-chunk
            // metadata: the recorded stream carries a smaller total at the
            // intermediate finish, so `> 0` would have passed before the fix
            // too. The expectation is read from the fixture, not hardcoded, so
            // a re-record cannot leave it behind.
            assert_eq!(
                terminal.usage.total_tokens,
                last_frame_total_tokens(
                    "stream_terminal_matrix/two_terminal_stream_terminal_carries_the_last_usage"
                ),
                "the terminal must carry the last frame's cumulative usage, not the \
                 intermediate finish's"
            );
            assert_eq!(
                terminal.finish_reason,
                Some(FinishReason::Stop),
                "the reason the turn actually ended on"
            );
            assert!(
                terminal
                    .response_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "the terminal should carry the responseId"
            );
            assert!(
                terminal.model.as_deref().is_some_and(|m| !m.is_empty()),
                "the terminal should carry the model version"
            );
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

#[tokio::test]
async fn two_terminal_stream_with_visible_thoughts() {
    const SCENARIO: &str = "stream_terminal_matrix/two_terminal_stream_with_visible_thoughts";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_with_visible_thoughts",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(3000)
                .additional_params(json!({
                    "tools": [{ "codeExecution": {} }],
                    "generationConfig": {
                        "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
                    }
                }))
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                states(&drained.text, FIRST_ROUND_VALUE),
                "the answer must survive, got {:?}",
                drained.text
            );
            assert!(
                drained
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "reasoning spanning the boundary must still aggregate as reasoning"
            );
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

#[tokio::test]
async fn gemini_3_flash_does_not_emit_the_intermediate_finish() {
    const SCENARIO: &str =
        "stream_terminal_matrix/gemini_3_flash_does_not_emit_the_intermediate_finish";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/gemini_3_flash_does_not_emit_the_intermediate_finish",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(3000)
                .additional_params(code_execution_params())
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                states(&drained.text, FIRST_ROUND_VALUE),
                "the answer must survive on this model family too, got {:?}",
                drained.text
            );
            assert_eq!(drained.terminals, 1, "exactly one terminal record");
        },
    )
    .await;

    // The control: measured 0/3 on this family for the request that produces
    // the shape 4/4 on gemini-2.5-flash. Asserting the *absence* keeps the
    // scope of the finding honest — the intermediate finish is something
    // Gemini 2.5 does, not something every Gemini model does — and makes the
    // cell fail loudly if a future model starts emitting it here.
    assert_recorded_stream_finishes_early(SCENARIO, false);
}

#[tokio::test]
async fn two_terminal_stream_through_raw_stream() {
    const SCENARIO: &str = "stream_terminal_matrix/two_terminal_stream_through_raw_stream";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_through_raw_stream",
        |client| async move {
            use rig::streaming::RawStreamingChoice;

            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(TWO_ROUND_PROMPT)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            // `raw_stream` keeps Gemini's own terminal type, so this pins the fix
            // on the provider-native entry point as well as the normalized one.
            let mut stream = model
                .raw_stream(request)
                .await
                .expect("raw stream should open");
            let mut text = String::new();
            let mut natives = 0;
            while let Some(item) = stream.next().await {
                match item.expect("no raw item should be an error") {
                    RawStreamingChoice::Message(chunk) => text.push_str(&chunk),
                    RawStreamingChoice::FinalResponse(_) => natives += 1,
                    _ => {}
                }
            }

            assert!(
                states(&text, FIRST_ROUND_VALUE),
                "the raw stream must carry the answer, got {text:?}"
            );
            assert_eq!(natives, 1, "exactly one provider-native terminal record");
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

#[tokio::test]
async fn two_terminal_stream_unicode_answer_after_the_first_finish() {
    const SCENARIO: &str =
        "stream_terminal_matrix/two_terminal_stream_unicode_answer_after_the_first_finish";

    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/two_terminal_stream_unicode_answer_after_the_first_finish",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "You must call the code execution tool twice as two separate executions. \
                 Execution 1: run only print(6 * 7). Then, after seeing that number, \
                 Execution 2: run only print(6 * 7 + 100). Never combine them. \
                 Finally reply with exactly: résultats ✅ <first> et <second>.",
                )
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                drained.text.contains("résultats"),
                "multi-byte text after the boundary must survive chunk splitting, got {:?}",
                drained.text
            );
            assert_eq!(
                text_of(&drained.choice),
                drained.text,
                "the aggregated choice must equal the streamed deltas"
            );
        },
    )
    .await;

    assert_recorded_stream_finishes_early(SCENARIO, true);
}

// --- 9-12: regression guards for ordinary single-terminal streams ---------

#[tokio::test]
async fn single_terminal_text_stream_is_unchanged() {
    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/single_terminal_text_stream_is_unchanged",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Reply with exactly the word PONG.")
                .temperature(0.0)
                .max_tokens(400)
                .additional_params(json!({
                    "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
                }))
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(drained.text.contains("PONG"), "got {:?}", drained.text);
            assert_eq!(drained.terminals, 1);
            assert!(drained.last_item_was_terminal);
            assert_eq!(
                drained
                    .terminal
                    .as_ref()
                    .and_then(|terminal| terminal.finish_reason.clone()),
                Some(FinishReason::Stop)
            );
        },
    )
    .await;
}

#[tokio::test]
async fn single_terminal_tool_call_stream_is_unchanged() {
    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/single_terminal_tool_call_stream_is_unchanged",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("What is 41 plus 1? Use the add tool.")
                .temperature(0.0)
                .max_tokens(1000)
                .tools(vec![rig::completion::ToolDefinition {
                    name: "add".to_string(),
                    description: "Add x and y together".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "x": { "type": "number", "description": "first operand" },
                            "y": { "type": "number", "description": "second operand" }
                        },
                        "required": ["x", "y"]
                    }),
                }])
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                drained
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                "the tool call must still aggregate"
            );
            assert_eq!(drained.terminals, 1);
            assert!(drained.last_item_was_terminal);
        },
    )
    .await;
}

#[tokio::test]
async fn max_tokens_truncated_stream_still_reports_length() {
    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/max_tokens_truncated_stream_still_reports_length",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Write 300 words about lighthouses.")
                .temperature(0.0)
                .max_tokens(32)
                .additional_params(json!({
                    "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
                }))
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert_eq!(
                drained
                    .terminal
                    .as_ref()
                    .and_then(|terminal| terminal.finish_reason.clone()),
                Some(FinishReason::Length),
                "a non-STOP terminal reason must survive the deferral"
            );
            assert_eq!(drained.terminals, 1);
        },
    )
    .await;
}

#[tokio::test]
async fn thinking_stream_terminal_is_unchanged() {
    with_gemini_stream_terminal_cassette(
        "stream_terminal_matrix/thinking_stream_terminal_is_unchanged",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("How many minutes are in two days? Answer with the number.")
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(json!({
                    "generationConfig": {
                        "thinkingConfig": { "thinkingBudget": 512, "includeThoughts": true }
                    }
                }))
                .build();

            let drained = drain(
                CompletionModel::stream(&model, request)
                    .await
                    .expect("stream should open"),
            )
            .await;

            assert!(
                states(&drained.text, "2880"),
                "the answer must arrive, got {:?}",
                drained.text
            );
            assert!(
                drained
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "reasoning must still aggregate"
            );
            assert_eq!(drained.terminals, 1);
            assert!(drained.last_item_was_terminal);
        },
    )
    .await;
}

// --- 13-25: shapes the provider cannot be asked for ----------------------

mod unit {
    use futures::StreamExt;
    use rig::completion::{CompletionModel, FinishReason};
    use rig::prelude::*;
    use rig::providers::gemini;
    use rig::streaming::StreamedAssistantContent;
    use rig_core::test_utils::{MockStreamingClient, SequencedStreamingHttpClient};

    /// Frames written from the bytes recorded by cells 1–12.
    const CODE_ROUND: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(6*7)"}}],"role":"model"},"index":0}],"responseId":"resp-first","modelVersion":"gemini-2.5-flash"}"#;
    /// The intermediate terminal: a finishReason on a chunk that is not last.
    /// The part kind matches the recorded bytes, where the early finish rides
    /// the *second* `executableCode` frame (see the module doc).
    const INTERMEDIATE_TERMINAL: &str = r#"{"candidates":[{"content":{"parts":[{"executableCode":{"language":"PYTHON","code":"print(6*7+100)"}}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"responseId":"resp-first","modelVersion":"gemini-2.5-flash"}"#;
    const ANSWER: &str = r#"{"candidates":[{"content":{"parts":[{"text":"The answer is 42."}],"role":"model"},"index":0}]}"#;
    const REAL_TERMINAL: &str = r#"{"candidates":[{"content":{"parts":[{"text":" Done."}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":40,"totalTokenCount":50},"responseId":"resp-last","modelVersion":"gemini-2.5-flash-002"}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    struct Run {
        text: String,
        reasoning: usize,
        tool_calls: usize,
        unknowns: usize,
        terminals: Vec<rig::streaming::StreamFinal>,
        errors: usize,
        last_was_terminal: bool,
        response: Option<rig::streaming::StreamFinal>,
    }

    async fn run(frames: &[&str]) -> Run {
        run_client(MockStreamingClient {
            sse_bytes: sse(frames),
        })
        .await
    }

    async fn run_client<T>(http_client: T) -> Run
    where
        T: rig::http_client::HttpClientExt + Clone + std::fmt::Debug + Send + Sync + 'static,
    {
        let client = gemini::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("client should build");
        let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
        let request = model.completion_request("hello").build();
        let mut stream = CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut run = Run {
            text: String::new(),
            reasoning: 0,
            tool_calls: 0,
            unknowns: 0,
            terminals: Vec::new(),
            errors: 0,
            last_was_terminal: false,
            response: None,
        };
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => {
                    run.last_was_terminal = matches!(item, StreamedAssistantContent::Final(_));
                    match item {
                        StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
                        StreamedAssistantContent::Reasoning { .. } => run.reasoning += 1,
                        StreamedAssistantContent::ToolCall { .. } => run.tool_calls += 1,
                        StreamedAssistantContent::Unknown(_) => run.unknowns += 1,
                        StreamedAssistantContent::Final(final_record) => {
                            run.terminals.push(final_record)
                        }
                        _ => {}
                    }
                }
                Err(_) => {
                    run.last_was_terminal = false;
                    run.errors += 1;
                }
            }
        }
        run.response = stream.response.clone();
        run
    }

    #[tokio::test]
    async fn an_intermediate_finish_reason_does_not_end_the_stream() {
        let run = run(&[CODE_ROUND, INTERMEDIATE_TERMINAL, ANSWER, REAL_TERMINAL]).await;
        assert_eq!(run.text, "The answer is 42. Done.");
        assert_eq!(run.terminals.len(), 1);
    }

    #[tokio::test]
    async fn the_last_finish_reason_wins_over_an_earlier_one() {
        // Gemini never pairs these two reasons, so only synthetic frames can
        // show which one the terminal reports.
        const TRUNCATED: &str = r#"{"candidates":[{"content":{"parts":[{"text":" more"}],"role":"model"},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, ANSWER, TRUNCATED]).await;
        assert_eq!(
            run.terminals
                .first()
                .and_then(|terminal| terminal.finish_reason.clone()),
            Some(FinishReason::Length)
        );
    }

    #[tokio::test]
    async fn a_usage_only_trailer_after_the_finish_reason_reaches_the_terminal() {
        const TRAILER: &str = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":99,"totalTokenCount":109}}"#;
        let run = run(&[ANSWER, REAL_TERMINAL, TRAILER]).await;
        let terminal = run.terminals.first().expect("one terminal");
        assert_eq!(
            terminal.usage.total_tokens, 109,
            "a usage trailer after the finish chunk must reach the terminal record"
        );
    }

    #[tokio::test]
    async fn later_metadata_wins_on_the_terminal_record() {
        let run = run(&[CODE_ROUND, INTERMEDIATE_TERMINAL, ANSWER, REAL_TERMINAL]).await;
        let terminal = run.terminals.first().expect("one terminal");
        assert_eq!(terminal.response_id.as_deref(), Some("resp-last"));
        assert_eq!(terminal.model.as_deref(), Some("gemini-2.5-flash-002"));
        assert_eq!(terminal.usage.total_tokens, 50);
    }

    #[tokio::test]
    async fn three_finish_reason_chunks_still_yield_one_terminal() {
        let run = run(&[
            INTERMEDIATE_TERMINAL,
            ANSWER,
            INTERMEDIATE_TERMINAL,
            ANSWER,
            REAL_TERMINAL,
        ])
        .await;
        assert_eq!(run.terminals.len(), 1, "exactly one terminal record");
    }

    #[tokio::test]
    async fn eof_without_any_finish_reason_yields_no_terminal_record() {
        let run = run(&[CODE_ROUND, ANSWER]).await;
        assert!(run.terminals.is_empty(), "truncation is still truncation");
        assert!(run.response.is_none());
        assert_eq!(run.text, "The answer is 42.");
    }

    #[tokio::test]
    async fn a_transport_error_after_a_finish_reason_yields_no_terminal_record() {
        let run = run_client(SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[INTERMEDIATE_TERMINAL, ANSWER])),
            Err(rig::http_client::Error::InvalidStatusCodeWithMessage(
                reqwest::StatusCode::BAD_GATEWAY,
                "connection reset".to_string(),
            )),
        ]))
        .await;

        assert_eq!(
            run.errors, 1,
            "the transport failure must reach the consumer"
        );
        // The deliberate half of the trade-off. Before the fix, a connection
        // that dropped after *any* finishReason still handed the consumer a
        // terminal record, because the stream had already ended there. It now
        // reports truncation instead — and it must: on this wire a
        // finishReason is not proof the turn finished (that is the whole bug),
        // so the two cases are indistinguishable in-band and the safe reading
        // is the module's stated truncation semantics. A terminal here would
        // report a successful completion for a turn that may have been cut in
        // half.
        assert!(
            run.terminals.is_empty(),
            "a failed stream must not be dressed up as a completed turn by the deferred terminal"
        );
        assert!(run.response.is_none());
    }

    /// The other half of cell 19, and the regressing direction: the provider
    /// signalled its *real* finish and the transport then failed before EOF.
    /// The turn's bytes all arrived, and rig still reports truncation —
    /// deliberately, because on this wire a `finishReason` is not proof the
    /// turn finished (that is the whole bug), so "final finish" and
    /// "intermediate finish" are indistinguishable in-band and reporting a
    /// completed turn for one that may have been cut in half is the worse
    /// error. Pinned so the trade-off cannot change unnoticed.
    #[tokio::test]
    async fn a_transport_error_after_the_real_terminal_also_reports_truncation() {
        let run = run_client(SequencedStreamingHttpClient::new(vec![
            Ok(sse(&[ANSWER, REAL_TERMINAL])),
            Err(rig::http_client::Error::InvalidStatusCodeWithMessage(
                reqwest::StatusCode::BAD_GATEWAY,
                "connection reset".to_string(),
            )),
        ]))
        .await;

        assert_eq!(run.errors, 1, "the transport failure reaches the consumer");
        assert_eq!(run.text, "The answer is 42. Done.", "content still arrives");
        assert!(
            run.terminals.is_empty(),
            "no terminal record: the stream never reached EOF"
        );
        assert!(run.response.is_none());
    }

    #[tokio::test]
    async fn a_tool_protocol_failure_still_ends_the_stream_immediately() {
        const FAILURE: &str = r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"malformed function call","index":0}]}"#;
        let run = run(&[ANSWER, FAILURE, ANSWER, REAL_TERMINAL]).await;

        assert_eq!(run.errors, 1);
        assert_eq!(
            run.text, "The answer is 42.",
            "nothing after the in-band failure is interpreted"
        );
        assert!(
            run.terminals.is_empty(),
            "an in-band failure must not acquire a terminal record from the EOF deferral"
        );
        assert!(run.response.is_none());
    }

    #[tokio::test]
    async fn a_tool_call_after_the_first_finish_reason_reaches_the_choice() {
        const TOOL_CALL: &str = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"add","args":{"x":1,"y":2},"id":"call-1"}}],"role":"model"},"index":0}]}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, TOOL_CALL, REAL_TERMINAL]).await;
        assert_eq!(
            run.tool_calls, 1,
            "a tool call past the boundary must still be delivered"
        );
    }

    #[tokio::test]
    async fn reasoning_after_the_first_finish_reason_reaches_the_choice() {
        const THOUGHT: &str = r#"{"candidates":[{"content":{"parts":[{"text":"thinking on","thought":true,"thoughtSignature":"sig"}],"role":"model"},"index":0}]}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, THOUGHT, REAL_TERMINAL]).await;
        assert!(
            run.reasoning > 0,
            "reasoning past the boundary must still be delivered"
        );
    }

    #[tokio::test]
    async fn an_unknown_frame_after_the_first_finish_reason_is_passed_through() {
        const UNKNOWN: &str = r#"{"someFutureField":{"x":1}}"#;
        let run = run(&[INTERMEDIATE_TERMINAL, UNKNOWN, ANSWER, REAL_TERMINAL]).await;
        assert_eq!(
            run.unknowns, 1,
            "the raw passthrough channel must still see frames past the boundary"
        );
        assert_eq!(run.text, "The answer is 42. Done.");
    }

    #[tokio::test]
    async fn the_terminal_record_is_the_last_item_yielded() {
        let run = run(&[CODE_ROUND, INTERMEDIATE_TERMINAL, ANSWER, REAL_TERMINAL]).await;
        assert!(
            run.last_was_terminal,
            "the deferred terminal must still close the stream"
        );
    }
}
