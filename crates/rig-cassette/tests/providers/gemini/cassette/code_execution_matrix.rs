//! Edge matrix for the `codeExecution` response-part bug.
//!
//! # The bug
//!
//! Rig lets a caller enable Gemini's built-in code-execution tool by putting
//! it in `additional_params.tools` — `create_request_body` lifts that array
//! straight onto the request (`extract_tools_from_additional_params`). Gemini
//! then answers with `executableCode` and `codeExecutionResult` parts
//! interleaved with the model's text. The blocking mapper had no arm for
//! either kind, so it fell through to
//! `ResponseError("Response did not contain a message or tool call")` and
//! **discarded the whole turn**, final text answer included, while the
//! streaming adapter skipped the same parts and returned the answer. Blocking
//! and streaming disagreed about the same bytes.
//!
//! # The matrix
//!
//! The fixed code path is the part mapper the GenerateContent decoder runs on
//! every part it reads, on both transports. Its inputs are the *kinds* of
//! parts in a candidate, their *order*, and which surface is reading them —
//! so the matrix multiplies part composition × transport × surface, and pins
//! the streaming twin of every blocking cell so parity is asserted rather
//! than assumed.
//!
//! | # | cell | transport | surface | dimension pinned |
//! |---|------|-----------|---------|------------------|
//! | 11 | `streaming_code_execution_with_visible_thoughts` | streaming | `stream` | parity twin of 10 |
//! | 22 | `blocking_code_execution_replayed_in_chat_history` | blocking | `Agent::chat` | a code-execution turn replayed as history, code parts included |
//! | 23 | `code_execution_only_turn_is_a_turn_of_its_code_parts` | — | unit | see below |
//! | 24 | `code_execution_parts_are_skipped_in_every_position` | — | unit | see below |
//! | 25 | `unmodeled_part_kinds_still_fail_loudly` | — | unit | see below |
//!
//! Cells 23–25 are unit tests rather than recordings because a live turn
//! cannot be made to produce their states: Gemini never returns a candidate
//! whose *only* parts are code-execution parts (it always narrates a result),
//! never returns the same code-execution parts in every ordering, and never
//! returns the `functionResponse`/`fileData` kinds that must keep failing.
//! Each states its shape from bytes recorded in this matrix.
//!
//! No cell was dropped for cost: every recording here is a flash-class model
//! with a capped output budget.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test -p rig --all-features --test gemini code_execution_matrix -- --test-threads=1`

use futures::StreamExt;
use rig::message::AssistantContent;
use rig::providers::gemini;
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde_json::{Value, json};

use super::super::support::{
    assert_recorded_response_contains, with_gemini_code_execution_cassette,
};
use rig::completion::CompletionRequest;

/// Wire markers of the two code-execution part kinds, as Gemini spells them.
pub(super) const CODE_PART_MARKERS: &[&str] = &["executableCode", "codeExecutionResult"];

/// `additional_params` enabling Gemini's built-in code-execution tool.
///
/// Thinking is deliberately left at the model's default. Forcing
/// `thinkingBudget: 0` makes gemini-2.5-flash skip the tool and *narrate* the
/// code instead — recorded responses come back as the literal text
/// `"tool_code\nprint(2**20)\n\n"` with no `executableCode` part at all — so a
/// zero budget would silently record cells that never exercise the part kinds
/// this matrix is about.
pub(super) fn code_execution_params() -> Value {
    json!({ "tools": [{ "codeExecution": {} }] })
}

/// The same, with the model's reasoning surfaced as `thought: true` parts so
/// a cell can pin code parts sitting next to thought parts.
fn code_execution_params_with_thoughts() -> Value {
    json!({
        "tools": [{ "codeExecution": {} }],
        "generationConfig": {
            "thinkingConfig": { "thinkingBudget": 512, "includeThoughts": true }
        }
    })
}

/// Whether `text` states `value`.
///
/// Digit grouping the model may add is ignored ("1,048,576" states
/// "1048576"), but only separators *between* digits are removed, so
/// `"items 12, 345"` does not silently become the number `12345`. A numeric
/// `value` must not be a fragment of a longer number — "2880" in "28800" is
/// not the answer — so digit-adjacency is rejected. An empty or non-numeric
/// `value` keeps plain substring semantics.
pub(super) fn states(text: &str, value: &str) -> bool {
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

/// Drain a normalized stream into its text and terminal record.
async fn drain(
    mut stream: rig::streaming::CompletionStream,
) -> (String, Vec<AssistantContent>, bool) {
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: chunk, .. }) =
            item.expect("no stream item should be an error")
        {
            text.push_str(&chunk)
        }
    }
    let response = stream.finish().await;
    let saw_terminal = response.is_ok();
    (
        text,
        response.map(|response| response.choice).unwrap_or_default(),
        saw_terminal,
    )
}

// --- 1/2: baseline, raw model, blocking and streaming ---------------------

// --- 3/4: agent surface ---------------------------------------------------

// --- 5: provider-native escape hatch --------------------------------------

// --- 6/7: failed code outcome ---------------------------------------------

// --- 8/9: several code rounds in one turn ---------------------------------

// --- 10/11: thought parts adjacent to code parts --------------------------

const THINKING_PROMPT: &str = "Use the code execution tool to compute the 20th Fibonacci number. \
     State the number in your answer.";

#[tokio::test]
async fn streaming_code_execution_with_visible_thoughts() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_with_visible_thoughts";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_with_visible_thoughts",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request = CompletionRequest::new(THINKING_PROMPT)
                .temperature(0.0)
                .max_tokens(2500)
                .additional_params(code_execution_params_with_thoughts());

            let stream = model.stream(request).expect("stream should open");
            let (streamed, choice, saw_terminal) = drain(stream).await;

            assert!(
                states(&streamed, "6765"),
                "streamed answer must carry the value, got {streamed:?}"
            );
            assert!(
                choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "streamed reasoning must aggregate as reasoning, not text"
            );
            assert!(saw_terminal, "the turn must produce a terminal record");
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
    assert_recorded_response_contains(SCENARIO, &["\"thought\""]);
}

// --- 12/13: system instruction alongside the tool -------------------------

// --- 14/15: non-ASCII stdout ----------------------------------------------

// --- 16/17: long stdout ---------------------------------------------------

// --- 18/19: sampling knobs alongside the tool -----------------------------

// --- 20/21: a second model family -----------------------------------------

// --- 22: a code-execution turn replayed as chat history -------------------

#[tokio::test]
async fn blocking_code_execution_replayed_in_chat_history() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_replayed_in_chat_history";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_replayed_in_chat_history",
        |client| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .temperature(0.0)
                    .max_tokens(2000)
                    .additional_params(code_execution_params())
                    .build();

            // Turn one produces the code-execution turn; turn two replays it
            // to the same model, code parts included.
            let mut history = Vec::new();
            let first = agent
                .chat(
                    "Use the code execution tool to compute 13 times 13. State the number.",
                    &mut history,
                )
                .await
                .expect("first code-execution turn should convert");
            assert!(
                states(&first.output(), "169"),
                "first answer should carry 169, got {first:?}"
            );
            let second = agent
                .chat("Now double that number and state the result.", &mut history)
                .await
                .expect("history replay after a code-execution turn should succeed");
            assert!(
                states(&second.output(), "338"),
                "second answer should carry the doubled value, got {second:?}"
            );
        },
    )
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies("gemini", SCENARIO);
    let replayed = &bodies.last().expect("a continuation").0;
    for marker in CODE_PART_MARKERS {
        assert!(
            replayed.contains(marker),
            "the continuation replays the turn's {marker} part"
        );
    }
    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 23-25: states a live turn cannot be made to produce ------------------

mod unit {
    use rig::completion::CompletionResponse;
    use rig::error::ProviderError;
    use rig::message::AssistantContent;
    use rig::providers::gemini::GeminiConfig;
    use rig::test_utils::RecordingHttpClient;
    use serde_json::{Value, json};

    /// One `executableCode` part, exactly as recorded in
    /// `blocking_raw_model_answers_after_code_execution`.
    fn executable_code_part() -> Value {
        json!({
            "executableCode": {
                "language": "PYTHON",
                "code": "import math\n\nresult = math.factorial(7)\nprint(f'{result=}')"
            }
        })
    }

    /// One `codeExecutionResult` part, from the same recording.
    fn code_result_part() -> Value {
        json!({
            "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "result=5040\n" }
        })
    }

    /// The recorded reply shape, with `parts` as the candidate's content.
    fn reply_with(parts: Vec<Value>) -> String {
        json!({
            "candidates": [{
                "content": { "parts": parts, "role": "model" },
                "finishReason": "STOP",
                "index": 0
            }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "unit-response",
            "usageMetadata": { "promptTokenCount": 13, "candidatesTokenCount": 65, "totalTokenCount": 152 }
        })
        .to_string()
    }

    /// One turn through the one seam, answered by a stub transport with
    /// `parts`.
    ///
    /// These part compositions cannot be produced live, so the bytes are
    /// stated here and carried by the real wire, driver and decoder — the
    /// same path every recorded cell above runs, with the reply substituted.
    async fn completion_of(parts: Vec<Value>) -> Result<CompletionResponse, ProviderError> {
        let model = GeminiConfig::new("unit-key")
            .connect(RecordingHttpClient::new(reply_with(parts)))
            .completion("gemini-2.5-flash");
        let request = rig::completion::CompletionRequest::new("unit");
        model.call(request).await
    }

    /// Not a recording: Gemini always narrates a code round, so a candidate
    /// whose only parts are code-execution parts cannot be forced live. Such
    /// a turn is a success, as pi takes an empty reply: emptiness is decided
    /// once by the fold, never by a decoder. Skipping the code parts invents
    /// no text: they stay opaque blocks that replay to the same model.
    #[tokio::test]
    async fn code_execution_only_turn_is_a_turn_of_its_code_parts() {
        let response = completion_of(vec![executable_code_part(), code_result_part()])
            .await
            .expect("an executable-code-only reply is a turn");
        assert!(!response.stop().is_failure(), "{:?}", response.stop());
        assert!(
            response
                .choice
                .iter()
                .all(|block| matches!(block, AssistantContent::Opaque(opaque) if opaque.replay)),
            "{:?}",
            response.choice
        );
        assert_eq!(response.choice.len(), 2, "{:?}", response.choice);
    }

    /// Not a recording: one live turn emits one ordering. Each code part
    /// stays an opaque block where it sat, whatever its position, and the
    /// text stays the only answer.
    #[tokio::test]
    async fn code_execution_parts_keep_their_position_as_opaque_blocks() {
        let text = json!({ "text": "The 7 factorial is 5040." });
        let orderings = [
            vec![executable_code_part(), code_result_part(), text.clone()],
            vec![text.clone(), executable_code_part(), code_result_part()],
            vec![executable_code_part(), text.clone(), code_result_part()],
            vec![
                executable_code_part(),
                code_result_part(),
                text,
                executable_code_part(),
                code_result_part(),
            ],
        ];

        for (index, parts) in orderings.into_iter().enumerate() {
            let response = completion_of(parts.clone())
                .await
                .unwrap_or_else(|error| panic!("ordering {index} should convert: {error:?}"));
            let kept: Vec<Value> = response
                .choice
                .iter()
                .map(|block| match block {
                    rig::message::AssistantContent::Opaque(opaque) => {
                        assert!(opaque.replay, "ordering {index}: code parts replay");
                        opaque.item.clone()
                    }
                    rig::message::AssistantContent::Text(text) => json!({ "text": text.text }),
                    other => panic!("ordering {index}: unexpected block {other:?}"),
                })
                .collect();
            assert_eq!(kept, parts, "ordering {index}: every part keeps its place");
        }
    }

    /// Not a recording: `generateContent` never answers with a
    /// `functionResponse` or `fileData` part (they are request-side kinds),
    /// so neither has a live source. Rig has no block for them, and they
    /// survive as opaque blocks rather than being dropped.
    #[tokio::test]
    async fn unmodeled_part_kinds_survive_as_opaque_blocks() {
        for part in [
            json!({ "functionResponse": { "name": "add", "response": { "result": 3 } } }),
            json!({ "fileData": { "mimeType": "text/plain", "fileUri": "https://example.invalid/f" } }),
        ] {
            let response = completion_of(vec![part.clone(), json!({ "text": "done" })])
                .await
                .unwrap_or_else(|error| panic!("part {part} should convert: {error:?}"));
            assert!(
                matches!(
                    response.choice.first(),
                    Some(rig::message::AssistantContent::Opaque(opaque)) if opaque.item == part
                ),
                "part {part} survives as an opaque block: {:?}",
                response.choice
            );
        }
    }
}
