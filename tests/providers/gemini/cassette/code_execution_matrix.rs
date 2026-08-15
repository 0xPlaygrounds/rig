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
//! The fixed code path is `map_response_part`, which every blocking caller
//! reaches through `TryFrom<GenerateContentResponse>`. Its inputs are the
//! *kinds* of parts in a candidate, their *order*, and which surface is
//! reading them — so the matrix multiplies part composition × transport ×
//! surface, and pins the streaming twin of every blocking cell so parity is
//! asserted rather than assumed.
//!
//! | # | cell | transport | surface | dimension pinned |
//! |---|------|-----------|---------|------------------|
//! | 1 | `blocking_raw_model_answers_after_code_execution` | blocking | `CompletionModel::completion` | baseline: code parts skipped, text survives |
//! | 2 | `streaming_raw_model_answers_after_code_execution` | streaming | `CompletionModel::stream` | parity twin of 1 |
//! | 3 | `blocking_agent_prompt_answers_after_code_execution` | blocking | `Agent::prompt` | agent surface |
//! | 4 | `streaming_agent_prompt_answers_after_code_execution` | streaming | `Agent::stream_prompt` | parity twin of 3 |
//! | 5 | `blocking_raw_completion_keeps_native_code_parts` | blocking | `CompletionModel::raw_completion` | escape hatch still exposes the parts |
//! | 6 | `blocking_failed_code_execution_outcome` | blocking | `completion` | `OUTCOME_FAILED` result part |
//! | 7 | `streaming_failed_code_execution_outcome` | streaming | `stream` | parity twin of 6 |
//! | 8 | `blocking_multiple_code_execution_rounds` | blocking | `completion` | several code/result pairs in one turn |
//! | 9 | `streaming_multiple_code_execution_rounds` | streaming | `stream` | parity twin of 8 |
//! | 10 | `blocking_code_execution_with_visible_thoughts` | blocking | `completion` | thought parts adjacent to code parts |
//! | 11 | `streaming_code_execution_with_visible_thoughts` | streaming | `stream` | parity twin of 10 |
//! | 12 | `blocking_code_execution_with_preamble` | blocking | `completion` | `systemInstruction` + `codeExecution` |
//! | 13 | `streaming_code_execution_with_preamble` | streaming | `stream` | parity twin of 12 |
//! | 14 | `blocking_code_execution_unicode_output` | blocking | `completion` | non-ASCII stdout in the result part |
//! | 15 | `streaming_code_execution_unicode_output` | streaming | `stream` | parity twin of 14; multi-byte chunk splitting |
//! | 16 | `blocking_code_execution_large_stdout` | blocking | `completion` | long `codeExecutionResult.output` |
//! | 17 | `streaming_code_execution_large_stdout` | streaming | `stream` | parity twin of 16 |
//! | 18 | `blocking_code_execution_capped_by_max_tokens` | blocking | `completion` | sampling knobs alongside the tool |
//! | 19 | `streaming_code_execution_capped_by_max_tokens` | streaming | `stream` | parity twin of 18 |
//! | 20 | `blocking_code_execution_on_gemini_3_flash` | blocking | `completion` | second model family |
//! | 21 | `streaming_code_execution_on_gemini_3_flash` | streaming | `stream` | parity twin of 20 |
//! | 22 | `blocking_code_execution_replayed_in_chat_history` | blocking | `Agent::chat` | a code-execution turn replayed as history |
//! | 23 | `code_execution_only_turn_is_an_empty_response` | — | unit | see below |
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
use rig::completion::{CompletionModel, Prompt};
use rig::message::{AssistantContent, Message};
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use serde_json::{Value, json};

use super::super::support::{
    assert_recorded_response_contains, with_gemini_code_execution_cassette,
};

/// Wire markers of the two code-execution part kinds, as Gemini spells them.
const CODE_PART_MARKERS: &[&str] = &["executableCode", "codeExecutionResult"];

/// `additional_params` enabling Gemini's built-in code-execution tool.
///
/// Thinking is deliberately left at the model's default. Forcing
/// `thinkingBudget: 0` makes gemini-2.5-flash skip the tool and *narrate* the
/// code instead — recorded responses come back as the literal text
/// `"tool_code\nprint(2**20)\n\n"` with no `executableCode` part at all — so a
/// zero budget would silently record cells that never exercise the part kinds
/// this matrix is about.
fn code_execution_params() -> Value {
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

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Drain a normalized stream into its text and terminal record.
async fn drain(
    mut stream: rig::streaming::StreamingCompletionResponse,
) -> (String, Vec<AssistantContent>, bool) {
    let mut text = String::new();
    let mut saw_terminal = false;
    while let Some(item) = stream.next().await {
        match item.expect("no stream item should be an error") {
            StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
            StreamedAssistantContent::Final(_) => saw_terminal = true,
            _ => {}
        }
    }
    (text, stream.choice.clone(), saw_terminal)
}

/// One blocking cell: run the prompt with code execution enabled and assert
/// the turn survived with its answer intact.
async fn blocking_body(
    client: gemini::Client,
    scenario: &'static str,
    model_id: &'static str,
    prompt: &'static str,
    params: Value,
    max_tokens: Option<u64>,
    expected_substring: &'static str,
) {
    let model = client.completion_model(model_id);
    let mut request = model
        .completion_request(prompt)
        .temperature(0.0)
        .additional_params(params);
    if let Some(max_tokens) = max_tokens {
        request = request.max_tokens(max_tokens);
    }

    let response = CompletionModel::completion(&model, request.build())
        .await
        .expect("a turn carrying code-execution parts must still convert");

    let text = text_of(&response.choice);
    assert!(
        states(&text, expected_substring),
        "{scenario}: the model's answer must survive the code-execution parts; \
             expected {expected_substring:?} in {text:?}"
    );
}

/// The streaming twin of [`blocking_body`], over the same request.
async fn streaming_body(
    client: gemini::Client,
    scenario: &'static str,
    model_id: &'static str,
    prompt: &'static str,
    params: Value,
    max_tokens: Option<u64>,
    expected_substring: &'static str,
) {
    let model = client.completion_model(model_id);
    let mut request = model
        .completion_request(prompt)
        .temperature(0.0)
        .additional_params(params);
    if let Some(max_tokens) = max_tokens {
        request = request.max_tokens(max_tokens);
    }

    let stream = CompletionModel::stream(&model, request.build())
        .await
        .expect("stream should open");
    let (streamed, choice, saw_terminal) = drain(stream).await;

    assert!(
        states(&streamed, expected_substring),
        "{scenario}: streamed text must carry the answer; expected \
             {expected_substring:?} in {streamed:?}"
    );
    assert_eq!(
        text_of(&choice),
        streamed,
        "{scenario}: the aggregated choice must equal the streamed deltas"
    );
    assert!(
        saw_terminal,
        "{scenario}: a completed code-execution turn must yield a terminal record"
    );
}

// --- 1/2: baseline, raw model, blocking and streaming ---------------------

#[tokio::test]
async fn blocking_raw_model_answers_after_code_execution() {
    const SCENARIO: &str = "code_execution_matrix/blocking_raw_model_answers_after_code_execution";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette("code_execution_matrix/blocking_raw_model_answers_after_code_execution", |client| {
        blocking_body(
            client,
            SCENARIO,
            gemini::completion::GEMINI_2_5_FLASH,
            "Use the code execution tool to compute 7 factorial. State the number in your answer.",
            code_execution_params(),
            Some(2000),
            "5040",
        )
    })
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_raw_model_answers_after_code_execution() {
    const SCENARIO: &str = "code_execution_matrix/streaming_raw_model_answers_after_code_execution";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette("code_execution_matrix/streaming_raw_model_answers_after_code_execution", |client| {
        streaming_body(
            client,
            SCENARIO,
            gemini::completion::GEMINI_2_5_FLASH,
            "Use the code execution tool to compute 7 factorial. State the number in your answer.",
            code_execution_params(),
            Some(2000),
            "5040",
        )
    })
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 3/4: agent surface ---------------------------------------------------

#[tokio::test]
async fn blocking_agent_prompt_answers_after_code_execution() {
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_agent_prompt_answers_after_code_execution",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let answer = agent
                .prompt("Use the code execution tool to compute 2 to the power of 20. State the number in your answer.")
                .await
                .expect("agent prompt must survive code-execution parts");

            assert!(
                states(&answer, "1048576"),
                "agent answer should carry the computed value, got {answer:?}"
            );
        },
    )
    .await;

    assert_recorded_response_contains(
        "code_execution_matrix/blocking_agent_prompt_answers_after_code_execution",
        CODE_PART_MARKERS,
    );
}

#[tokio::test]
async fn streaming_agent_prompt_answers_after_code_execution() {
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_agent_prompt_answers_after_code_execution",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let mut stream = agent
                .stream_prompt("Use the code execution tool to compute 2 to the power of 20. State the number in your answer.")
                .await;

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
                states(&answer, "1048576"),
                "streamed agent answer should carry the computed value, got {answer:?}"
            );
        },
    )
    .await;

    assert_recorded_response_contains(
        "code_execution_matrix/streaming_agent_prompt_answers_after_code_execution",
        CODE_PART_MARKERS,
    );
}

// --- 5: provider-native escape hatch --------------------------------------

#[tokio::test]
async fn blocking_raw_completion_keeps_native_code_parts() {
    const SCENARIO: &str = "code_execution_matrix/blocking_raw_completion_keeps_native_code_parts";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_raw_completion_keeps_native_code_parts",
        |client| async move {
            use rig::providers::gemini::completion::gemini_api_types::PartKind;

            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "Use the code execution tool to sum the integers from 1 to 100. \
                 State the number in your answer.",
                )
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("raw_completion should succeed");

            let parts = raw
                .candidates
                .first()
                .and_then(|candidate| candidate.content.as_ref())
                .map(|content| content.parts.as_slice())
                .expect("the recorded candidate carries content");

            // The escape hatch is the supported way to reach what the normalized
            // choice cannot represent: the code and its output are still here.
            assert!(
                parts
                    .iter()
                    .any(|part| matches!(part.part, PartKind::ExecutableCode(_))),
                "raw_completion must expose the executableCode part verbatim"
            );
            assert!(
                parts
                    .iter()
                    .any(|part| matches!(part.part, PartKind::CodeExecutionResult(_))),
                "raw_completion must expose the codeExecutionResult part verbatim"
            );

            // And the same payload normalizes rather than erroring.
            let normalized: rig::completion::CompletionResponse =
                raw.try_into().expect("the same payload must normalize");
            assert!(
                states(&text_of(&normalized.choice), "5050"),
                "normalized answer should carry the computed value"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 6/7: failed code outcome ---------------------------------------------

const FAILING_CODE_PROMPT: &str = "Use the code execution tool to run exactly `print(1/0)` first. After it fails, \
     say the word DIVISIONERROR in your answer.";

#[tokio::test]
async fn blocking_failed_code_execution_outcome() {
    const SCENARIO: &str = "code_execution_matrix/blocking_failed_code_execution_outcome";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_failed_code_execution_outcome",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                FAILING_CODE_PROMPT,
                code_execution_params(),
                Some(2000),
                "DIVISIONERROR",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_failed_code_execution_outcome() {
    const SCENARIO: &str = "code_execution_matrix/streaming_failed_code_execution_outcome";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_failed_code_execution_outcome",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                FAILING_CODE_PROMPT,
                code_execution_params(),
                Some(2000),
                "DIVISIONERROR",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 8/9: several code rounds in one turn ---------------------------------

const MULTI_ROUND_PROMPT: &str = "Use the code execution tool twice, in two separate runs: first compute 6*7, \
     then in a second run compute 6*7 plus 100. State both numbers in your answer.";

#[tokio::test]
async fn blocking_multiple_code_execution_rounds() {
    const SCENARIO: &str = "code_execution_matrix/blocking_multiple_code_execution_rounds";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_multiple_code_execution_rounds",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                MULTI_ROUND_PROMPT,
                code_execution_params(),
                Some(2000),
                "142",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_multiple_code_execution_rounds() {
    const SCENARIO: &str = "code_execution_matrix/streaming_multiple_code_execution_rounds";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_multiple_code_execution_rounds",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                MULTI_ROUND_PROMPT,
                code_execution_params(),
                Some(2000),
                "142",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 10/11: thought parts adjacent to code parts --------------------------

const THINKING_PROMPT: &str = "Use the code execution tool to compute the 20th Fibonacci number. \
     State the number in your answer.";

#[tokio::test]
async fn blocking_code_execution_with_visible_thoughts() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_with_visible_thoughts";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_with_visible_thoughts",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(THINKING_PROMPT)
                .temperature(0.0)
                .max_tokens(2500)
                .additional_params(code_execution_params_with_thoughts())
                .build();

            let response = CompletionModel::completion(&model, request)
                .await
                .expect("code-execution parts next to thought parts must still convert");

            assert!(
                states(&text_of(&response.choice), "6765"),
                "the answer must survive, got {:?}",
                text_of(&response.choice)
            );
            // Thought parts stay reasoning; they are never folded into the text.
            assert!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "includeThoughts should surface reasoning blocks"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
    assert_recorded_response_contains(SCENARIO, &["\"thought\""]);
}

#[tokio::test]
async fn streaming_code_execution_with_visible_thoughts() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_with_visible_thoughts";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_with_visible_thoughts",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(THINKING_PROMPT)
                .temperature(0.0)
                .max_tokens(2500)
                .additional_params(code_execution_params_with_thoughts())
                .build();

            let stream = CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");
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

const PREAMBLE: &str = "You are a calculator. Always use the code execution tool, then \
                        state the numeric result in plain text.";
const PREAMBLE_PROMPT: &str = "What is 123 multiplied by 456?";

#[tokio::test]
async fn blocking_code_execution_with_preamble() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_with_preamble";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_with_preamble",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(PREAMBLE_PROMPT)
                .preamble(PREAMBLE.to_string())
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let response = CompletionModel::completion(&model, request)
                .await
                .expect("systemInstruction + codeExecution must still convert");
            assert!(
                states(&text_of(&response.choice), "56088"),
                "answer should carry the product, got {:?}",
                text_of(&response.choice)
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_code_execution_with_preamble() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_with_preamble";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_with_preamble",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(PREAMBLE_PROMPT)
                .preamble(PREAMBLE.to_string())
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            let stream = CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");
            let (streamed, _, saw_terminal) = drain(stream).await;

            assert!(
                states(&streamed, "56088"),
                "streamed answer should carry the product, got {streamed:?}"
            );
            assert!(saw_terminal, "the turn must produce a terminal record");
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 14/15: non-ASCII stdout ----------------------------------------------

const UNICODE_PROMPT: &str = "Use the code execution tool to run exactly \
     `print('θερμοκρασία 🌡️ 25°C')` and then repeat its output verbatim in your answer.";

#[tokio::test]
async fn blocking_code_execution_unicode_output() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_unicode_output";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_unicode_output",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                UNICODE_PROMPT,
                code_execution_params(),
                Some(2000),
                "θερμοκρασία",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_code_execution_unicode_output() {
    // The cassette replay server fragments SSE bodies on byte boundaries that
    // fall inside multi-byte characters, so this cell also pins that a code
    // round's non-ASCII output survives chunk splitting.
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_unicode_output";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_unicode_output",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                UNICODE_PROMPT,
                code_execution_params(),
                Some(2000),
                "θερμοκρασία",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 16/17: long stdout ---------------------------------------------------

const LARGE_OUTPUT_PROMPT: &str = "Use the code execution tool to run exactly `print(list(range(200)))`. \
     Then say the word PRINTED in your answer.";

#[tokio::test]
async fn blocking_code_execution_large_stdout() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_large_stdout";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_large_stdout",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                LARGE_OUTPUT_PROMPT,
                code_execution_params(),
                Some(2000),
                "PRINTED",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_code_execution_large_stdout() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_large_stdout";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_large_stdout",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                LARGE_OUTPUT_PROMPT,
                code_execution_params(),
                Some(2000),
                "PRINTED",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 18/19: sampling knobs alongside the tool -----------------------------

const CAPPED_PROMPT: &str =
    "Use the code execution tool to compute 999 times 999. State the number in your answer.";

/// The caller's sampling knobs must ride the same request as the tool: an
/// `additional_params` payload carrying `tools` is merged into the request
/// beside `generationConfig`, and this asserts the merge kept both.
fn assert_capped_request_carried_both_knobs(scenario: &str) {
    super::super::support::assert_recorded_sampling_fields(
        scenario,
        &[
            ("maxOutputTokens", serde_json::json!(1500)),
            ("temperature", serde_json::json!(0.0)),
        ],
    );
}

#[tokio::test]
async fn blocking_code_execution_capped_by_max_tokens() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_capped_by_max_tokens";
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_capped_by_max_tokens",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                CAPPED_PROMPT,
                code_execution_params(),
                Some(1500),
                "998001",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
    assert_capped_request_carried_both_knobs(SCENARIO);
}

#[tokio::test]
async fn streaming_code_execution_capped_by_max_tokens() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_capped_by_max_tokens";
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_capped_by_max_tokens",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                CAPPED_PROMPT,
                code_execution_params(),
                Some(1500),
                "998001",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
    assert_capped_request_carried_both_knobs(SCENARIO);
}

// --- 20/21: a second model family -----------------------------------------

const SECOND_MODEL_PROMPT: &str =
    "Use the code execution tool to compute 17 squared. State the number in your answer.";

#[tokio::test]
async fn blocking_code_execution_on_gemini_3_flash() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_on_gemini_3_flash";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_on_gemini_3_flash",
        |client| {
            blocking_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_3_FLASH_PREVIEW,
                SECOND_MODEL_PROMPT,
                json!({ "tools": [{ "codeExecution": {} }] }),
                Some(2000),
                "289",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

#[tokio::test]
async fn streaming_code_execution_on_gemini_3_flash() {
    const SCENARIO: &str = "code_execution_matrix/streaming_code_execution_on_gemini_3_flash";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_code_execution_cassette(
        "code_execution_matrix/streaming_code_execution_on_gemini_3_flash",
        |client| {
            streaming_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_3_FLASH_PREVIEW,
                SECOND_MODEL_PROMPT,
                json!({ "tools": [{ "codeExecution": {} }] }),
                Some(2000),
                "289",
            )
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 22: a code-execution turn replayed as chat history -------------------

#[tokio::test]
async fn blocking_code_execution_replayed_in_chat_history() {
    const SCENARIO: &str = "code_execution_matrix/blocking_code_execution_replayed_in_chat_history";

    with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_replayed_in_chat_history",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(code_execution_params())
                .build();

            // Turn one produces the code-execution turn; turn two replays the
            // normalized assistant message back to Gemini as history. The
            // normalized message holds only the text — the code parts have no
            // slot — so this cell pins that the trimmed history is still a legal
            // request.
            let first = agent
                .prompt("Use the code execution tool to compute 13 times 13. State the number.")
                .await
                .expect("first code-execution turn should convert");
            assert!(
                states(&first, "169"),
                "first answer should carry 169, got {first:?}"
            );

            let mut history = vec![
                Message::user(
                    "Use the code execution tool to compute 13 times 13. State the number.",
                ),
                Message::assistant(first),
            ];
            let second = agent
                .chat("Now double that number and state the result.", &mut history)
                .await
                .expect("history replay after a code-execution turn should succeed");
            assert!(
                states(&second, "338"),
                "second answer should carry the doubled value, got {second:?}"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, CODE_PART_MARKERS);
}

// --- 23-25: states a live turn cannot be made to produce ------------------

mod unit {
    use rig::completion::CompletionResponse;
    use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
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

    fn response_with(parts: Vec<Value>) -> GenerateContentResponse {
        serde_json::from_value(json!({
            "candidates": [{
                "content": { "parts": parts, "role": "model" },
                "finishReason": "STOP",
                "index": 0
            }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "unit-response",
            "usageMetadata": { "promptTokenCount": 13, "candidatesTokenCount": 65, "totalTokenCount": 152 }
        }))
        .expect("recorded-shape payload should deserialize")
    }

    /// Not a recording: Gemini always narrates a code round, so a candidate
    /// whose only parts are code-execution parts cannot be forced live. The
    /// contract is that skipping them does not invent content — an otherwise
    /// empty choice must still be rejected by rig's shared empty-response
    /// rule, not silently returned as a blank answer.
    #[test]
    fn code_execution_only_turn_is_an_empty_response() {
        let response = response_with(vec![executable_code_part(), code_result_part()]);
        let error = CompletionResponse::try_from(response)
            .expect_err("a turn with no modeled content must not convert to a blank answer");
        assert!(
            matches!(error, rig::completion::CompletionError::ResponseError(_)),
            "expected the shared empty-response rejection, got {error:?}"
        );
    }

    /// Not a recording: one live turn emits one ordering. The skip must not
    /// depend on where the code parts sit relative to the text, so every
    /// position is asserted from the recorded part shapes.
    #[test]
    fn code_execution_parts_are_skipped_in_every_position() {
        let text = json!({ "text": "The 7 factorial is 5040." });
        let orderings = [
            vec![executable_code_part(), code_result_part(), text.clone()],
            vec![text.clone(), executable_code_part(), code_result_part()],
            vec![executable_code_part(), text.clone(), code_result_part()],
            vec![
                executable_code_part(),
                code_result_part(),
                text.clone(),
                executable_code_part(),
                code_result_part(),
            ],
        ];

        for (index, parts) in orderings.into_iter().enumerate() {
            let response = CompletionResponse::try_from(response_with(parts))
                .unwrap_or_else(|error| panic!("ordering {index} should convert: {error:?}"));
            assert_eq!(
                response.choice.len(),
                1,
                "ordering {index}: only the text part carries assistant content"
            );
            assert!(
                matches!(
                    response.choice.first(),
                    Some(rig::message::AssistantContent::Text(text)) if text.text == "The 7 factorial is 5040."
                ),
                "ordering {index}: the surviving part must be the model's text, got {:?}",
                response.choice
            );
        }
    }

    /// Not a recording: `generateContent` never answers with a
    /// `functionResponse` or `fileData` part (they are request-side kinds), so
    /// the arm that must keep failing has no live source. Widening the skip to
    /// every unmodeled kind would turn a genuinely unaccountable payload into
    /// a silent content drop.
    #[test]
    fn unmodeled_part_kinds_still_fail_loudly() {
        for part in [
            json!({ "functionResponse": { "name": "add", "response": { "result": 3 } } }),
            json!({ "fileData": { "mimeType": "text/plain", "fileUri": "https://example.invalid/f" } }),
        ] {
            let response = response_with(vec![part.clone(), json!({ "text": "done" })]);
            let error = CompletionResponse::try_from(response)
                .expect_err("an unaccountable part must still fail the response");
            assert!(
                matches!(error, rig::completion::CompletionError::ResponseError(_)),
                "part {part} should be a ResponseError, got {error:?}"
            );
        }
    }
}
