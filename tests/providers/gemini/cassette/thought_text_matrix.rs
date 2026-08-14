//! Edge matrix for the "a thought part is not output text" bug.
//!
//! # The bug
//!
//! `thinkingConfig.includeThoughts` makes Gemini return its chain-of-thought
//! in the *same* `parts` array as the answer, distinguished only by
//! `"thought": true`. The completion mapper honours that flag (thought parts
//! become `AssistantContent::Reasoning`); two other readers of the same
//! payload did not:
//!
//! * `TryFrom<GenerateContentResponse> for TranscriptionResponse` read
//!   `parts.first()`. With thoughts on, parts[0] is the reasoning — so
//!   `response.text` was **the model's private reasoning** and the actual
//!   transcript, sitting in parts[1], was dropped. A transcript split across
//!   several text parts lost everything after the first, too.
//! * `ProviderResponseExt::get_text_response` collected *every* text part,
//!   gluing the chain-of-thought onto the answer. Rig's other Gemini
//!   transport (the Interactions API) and Anthropic both exclude reasoning
//!   from that method, so the REST wire was the odd one out.
//!
//! Both now go through `visible_text_parts`, the one place the skip rule
//! lives.
//!
//! # The matrix
//!
//! The fixed code path's inputs are: whether thought parts are present, where
//! they sit, which reader is asking, and which model produced them (2.5's
//! `thinkingBudget` and 3's `thinkingLevel` are different wire dialects).
//!
//! | # | cell | reader | dimension pinned |
//! |---|------|--------|------------------|
//! | 1 | `transcription_with_visible_thoughts_returns_the_transcript` | transcription | the bug itself |
//! | 2 | `transcription_with_thinking_disabled_is_unchanged` | transcription | no-thoughts regression guard |
//! | 3 | `transcription_with_default_params_is_unchanged` | transcription | no `additional_params` at all |
//! | 4 | `transcription_with_thoughts_on_gemini_3_flash` | transcription | `thinkingLevel` dialect |
//! | 5 | `transcription_with_thoughts_and_temperature` | transcription | temperature alongside thoughts |
//! | 6 | `transcription_with_a_large_thinking_budget` | transcription | long reasoning before the transcript |
//! | 7 | `text_response_skips_thoughts_on_gemini_2_5_flash` | `get_text_response` | the bug's second reader |
//! | 8 | `text_response_skips_thoughts_on_gemini_3_flash` | `get_text_response` | `thinkingLevel` dialect |
//! | 9 | `text_response_with_thinking_disabled_is_unchanged` | `get_text_response` | no-thoughts regression guard |
//! | 10 | `text_response_on_gemini_2_5_flash_lite` | `get_text_response` | third model |
//! | 11 | `text_response_on_gemini_3_1_flash_lite` | `get_text_response` | pre-thinking model family |
//! | 12 | `text_response_with_a_large_thinking_budget` | `get_text_response` | long reasoning |
//! | 13 | `text_response_with_a_preamble` | `get_text_response` | `systemInstruction` present |
//! | 14 | `text_response_on_a_tool_call_turn` | `get_text_response` | thought part beside a `functionCall` |
//! | 15 | `text_response_with_structured_output` | `get_text_response` | `responseJsonSchema` turn |
//! | 16 | `text_response_across_two_candidates` | `get_text_response` | `candidateCount: 2` |
//! | 17 | `text_response_is_none_when_the_turn_is_all_thought` | `get_text_response` | budget spent entirely on thinking |
//! | 18 | `streaming_twin_keeps_reasoning_out_of_the_text` | stream | streaming parity for the same request |
//! | 19 | `text_response_matches_the_choice_text` (unit) | both | see below |
//! | 20 | `transcription_joins_every_visible_text_part` (unit) | transcription | see below |
//! | 21 | `transcription_rejects_a_thought_only_candidate` (unit) | transcription | see below |
//! | 22 | `transcription_rejects_a_candidate_with_no_text_part` (unit) | transcription | see below |
//! | 23 | `text_response_is_none_for_a_thought_only_candidate` (unit) | `get_text_response` | see below |
//! | 24 | `text_response_still_ignores_non_model_roles` (unit) | `get_text_response` | see below |
//! | 25 | `transcription_keeps_an_empty_visible_text_part` (unit) | transcription | see below |
//!
//! Cells 19–25 are unit tests because a live turn cannot be made to produce
//! their states: Gemini does not split a short transcript across several
//! visible text parts on demand, does not return a transcription candidate
//! consisting only of thoughts, and never labels a `generateContent`
//! candidate with a non-model role. Every one states its shape from bytes
//! recorded elsewhere in this matrix.
//!
//! No cell was dropped for cost.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test -p rig --all-features --test gemini thought_text_matrix -- --test-threads=1`

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::StreamedAssistantContent;
use rig::telemetry::ProviderResponseExt;
use rig::transcription::TranscriptionModel;
use serde_json::{Value, json};

use super::super::support::{
    assert_recorded_response_contains, assert_recorded_response_excludes,
    with_gemini_thought_text_cassette,
};
use crate::support::AUDIO_FIXTURE_PATH;

/// The sentence spoken in `tests/data/en-us-natural-speech.mp3`, as recorded
/// by this matrix's own fixtures.
const SPOKEN_WORDS: &str = "casting long shadows";

/// The wire marker of a thought part.
///
/// The key alone, not `"thought": true`: `generateContent` returns compact
/// JSON (`"thought":true`) while `streamGenerateContent` pretty-prints with a
/// space, and only responses are scanned, so the key's presence is exactly
/// "this turn carried a thought part".
const THOUGHT_MARKER: &[&str] = &["\"thought\""];

/// A transcription `additional_params` value: Gemini's transcription surface
/// deserializes it straight into `GenerationConfig`, so the thinking config
/// is the top level here (not nested under `generationConfig` as it is on the
/// completion surface).
fn transcription_thinking(budget: u32, include: bool) -> Value {
    json!({ "thinkingConfig": { "thinkingBudget": budget, "includeThoughts": include } })
}

/// A completion `additional_params` value carrying a Gemini 2.5 thinking
/// config.
fn completion_thinking(budget: u32, include: bool) -> Value {
    json!({
        "generationConfig": {
            "thinkingConfig": { "thinkingBudget": budget, "includeThoughts": include }
        }
    })
}

fn choice_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn has_reasoning(choice: &[AssistantContent]) -> bool {
    choice
        .iter()
        .any(|content| matches!(content, AssistantContent::Reasoning(_)))
}

/// A cell that expected thought parts must prove its fixture still carries
/// them; one that expected none must prove the opposite.
fn assert_thought_parts_recorded(scenario: &str, expected: bool) {
    if expected {
        assert_recorded_response_contains(scenario, THOUGHT_MARKER);
    } else {
        assert_recorded_response_excludes(scenario, THOUGHT_MARKER);
    }
}

// --- 1-6: the transcription reader ----------------------------------------

/// The text of a candidate's parts, split into (visible, thought).
///
/// Derived from the payload the cell actually recorded, so every assertion
/// below is re-derivable from the fixture rather than from an expectation
/// about what the model chose to say.
fn split_parts(
    response: &rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse,
) -> (String, Vec<String>) {
    use rig::providers::gemini::completion::gemini_api_types::PartKind;

    let mut visible = String::new();
    let mut thoughts = Vec::new();
    for part in response
        .candidates
        .iter()
        .filter_map(|candidate| candidate.content.as_ref())
        .flat_map(|content| content.parts.iter())
    {
        if let PartKind::Text(text) = &part.part {
            if part.thought.unwrap_or(false) {
                thoughts.push(text.clone());
            } else {
                visible.push_str(text);
            }
        }
    }
    (visible, thoughts)
}

/// One transcription cell: run the fixture audio through the model and assert
/// the returned text is the transcript, not the reasoning.
///
/// `expected_words` is checked only where the recorded turn answers in
/// English. Gemini 3 reads rig's fixed transcription preamble ("Translate the
/// provided audio exactly…") as an instruction to *translate*, so that cell
/// pins the mapping without pinning the language.
async fn transcription_body(
    client: gemini::Client,
    scenario: &'static str,
    model_id: &'static str,
    params: Option<Value>,
    temperature: Option<f64>,
    thoughts_expected: bool,
    expected_words: Option<&'static str>,
) {
    let model = client.transcription_model(model_id);
    let mut request = model
        .transcription_request()
        .load_file(AUDIO_FIXTURE_PATH)
        .expect("audio fixture should load");
    if let Some(params) = params {
        request = request.additional_params(params);
    }
    if let Some(temperature) = temperature {
        request = request.temperature(temperature);
    }

    let response = request.send().await.expect("transcription should succeed");
    let (visible, thoughts) = split_parts(&response.response);

    assert_eq!(
        !thoughts.is_empty(),
        thoughts_expected,
        "{scenario}: recorded thought parts should be {}",
        if thoughts_expected {
            "present"
        } else {
            "absent"
        }
    );
    assert!(
        !visible.is_empty(),
        "{scenario}: the recorded turn carried no visible text, so it pins nothing"
    );

    // The invariant, re-derived from the recorded bytes: the transcript is
    // exactly the turn's visible text — every visible part, and only those.
    assert_eq!(
        response.text, visible,
        "{scenario}: the transcript must be the turn's visible text; returning the \
             chain-of-thought (or only the first part) is the bug"
    );
    for thought in &thoughts {
        assert!(
            !response.text.contains(thought.as_str()),
            "{scenario}: reasoning text must not appear in the transcript"
        );
    }
    if let Some(words) = expected_words {
        assert!(
            response.text.contains(words),
            "{scenario}: expected the spoken words {words:?} in {:?}",
            response.text
        );
    }
}

#[tokio::test]
async fn transcription_with_visible_thoughts_returns_the_transcript() {
    const SCENARIO: &str =
        "thought_text_matrix/transcription_with_visible_thoughts_returns_the_transcript";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_visible_thoughts_returns_the_transcript",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                Some(transcription_thinking(512, true)),
                None,
                true,
                Some(SPOKEN_WORDS),
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn transcription_with_thinking_disabled_is_unchanged() {
    const SCENARIO: &str = "thought_text_matrix/transcription_with_thinking_disabled_is_unchanged";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_thinking_disabled_is_unchanged",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                Some(transcription_thinking(0, false)),
                None,
                false,
                Some(SPOKEN_WORDS),
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, false);
}

#[tokio::test]
async fn transcription_with_default_params_is_unchanged() {
    const SCENARIO: &str = "thought_text_matrix/transcription_with_default_params_is_unchanged";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_default_params_is_unchanged",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                None,
                None,
                false,
                Some(SPOKEN_WORDS),
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, false);
}

#[tokio::test]
async fn transcription_with_thoughts_on_gemini_3_flash() {
    const SCENARIO: &str = "thought_text_matrix/transcription_with_thoughts_on_gemini_3_flash";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_thoughts_on_gemini_3_flash",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_3_FLASH_PREVIEW,
                Some(json!({
                    "thinkingConfig": { "thinkingLevel": "high", "includeThoughts": true }
                })),
                None,
                true,
                // Gemini 3 obeys rig's "Translate the provided audio exactly"
                // preamble literally and answers in another language, so this cell
                // pins the mapping, not the wording.
                None,
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn transcription_with_thoughts_and_temperature() {
    const SCENARIO: &str = "thought_text_matrix/transcription_with_thoughts_and_temperature";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_thoughts_and_temperature",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                Some(transcription_thinking(512, true)),
                Some(0.0),
                true,
                Some(SPOKEN_WORDS),
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn transcription_with_a_large_thinking_budget() {
    const SCENARIO: &str = "thought_text_matrix/transcription_with_a_large_thinking_budget";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/transcription_with_a_large_thinking_budget",
        |client| {
            transcription_body(
                client,
                SCENARIO,
                gemini::completion::GEMINI_2_5_FLASH,
                Some(transcription_thinking(4096, true)),
                None,
                true,
                Some(SPOKEN_WORDS),
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

// --- 7-17: the `get_text_response` reader ---------------------------------

/// One `get_text_response` cell: the provider-native reader and the
/// normalized choice must agree about what the *text* of the turn was, and
/// reasoning must appear in neither's text.
struct TextResponseCell {
    model_id: &'static str,
    prompt: &'static str,
    preamble: Option<&'static str>,
    params: Option<Value>,
    max_tokens: Option<u64>,
    thoughts_expected: bool,
}

async fn text_response_body(
    client: gemini::Client,
    scenario: &'static str,
    cell: TextResponseCell,
) {
    let TextResponseCell {
        model_id,
        prompt,
        preamble,
        params,
        max_tokens,
        thoughts_expected,
    } = cell;

    let model = client.completion_model(model_id);
    let mut request = model.completion_request(prompt).temperature(0.0);
    if let Some(preamble) = preamble {
        request = request.preamble(preamble.to_string());
    }
    if let Some(params) = params {
        request = request.additional_params(params);
    }
    if let Some(max_tokens) = max_tokens {
        request = request.max_tokens(max_tokens);
    }
    let request = request.build();

    let raw = model
        .raw_completion(request)
        .await
        .expect("raw_completion should succeed");

    let recorded_thoughts: Vec<String> = raw
        .candidates
        .iter()
        .filter_map(|candidate| candidate.content.as_ref())
        .flat_map(|content| content.parts.iter())
        .filter(|part| part.thought.unwrap_or(false))
        .filter_map(|part| match &part.part {
            rig::providers::gemini::completion::gemini_api_types::PartKind::Text(text) => {
                Some(text.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        !recorded_thoughts.is_empty(),
        thoughts_expected,
        "{scenario}: recorded thought parts should be {}",
        if thoughts_expected {
            "present"
        } else {
            "absent"
        }
    );

    let text_response = raw.get_text_response();
    for thought in &recorded_thoughts {
        if let Some(text) = text_response.as_deref() {
            assert!(
                !text.contains(thought.as_str()),
                "{scenario}: get_text_response must not carry the chain-of-thought"
            );
        }
    }

    // The two readers of one payload must agree. `choice_text` joins the
    // normalized text blocks the same way `get_text_response` joins parts.
    let normalized: rig::completion::CompletionResponse =
        raw.try_into().expect("payload should normalize");
    assert_eq!(
        text_response.as_deref().unwrap_or_default(),
        choice_text(&normalized.choice),
        "{scenario}: the provider-native text reader and the normalized choice disagree"
    );
    assert_eq!(
        has_reasoning(&normalized.choice),
        thoughts_expected,
        "{scenario}: thought parts must normalize to reasoning blocks"
    );
}

const THINKING_PROMPT: &str =
    "Work out how many minutes are in three and a half days, then answer with only the number.";

#[tokio::test]
async fn text_response_skips_thoughts_on_gemini_2_5_flash() {
    const SCENARIO: &str = "thought_text_matrix/text_response_skips_thoughts_on_gemini_2_5_flash";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_skips_thoughts_on_gemini_2_5_flash",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: gemini::completion::GEMINI_2_5_FLASH,
                    prompt: THINKING_PROMPT,
                    preamble: None,
                    params: Some(completion_thinking(512, true)),
                    max_tokens: Some(2000),
                    thoughts_expected: true,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_skips_thoughts_on_gemini_3_flash() {
    const SCENARIO: &str = "thought_text_matrix/text_response_skips_thoughts_on_gemini_3_flash";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_skips_thoughts_on_gemini_3_flash",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: gemini::completion::GEMINI_3_FLASH_PREVIEW,
                    prompt: THINKING_PROMPT,
                    preamble: None,
                    params: Some(json!({
                    "generationConfig": {
                    "thinkingConfig": { "thinkingLevel": "low", "includeThoughts": true }
                    }
                    })),
                    max_tokens: Some(2000),
                    thoughts_expected: true,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_with_thinking_disabled_is_unchanged() {
    const SCENARIO: &str = "thought_text_matrix/text_response_with_thinking_disabled_is_unchanged";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_with_thinking_disabled_is_unchanged",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: gemini::completion::GEMINI_2_5_FLASH,
                    prompt: THINKING_PROMPT,
                    preamble: None,
                    params: Some(completion_thinking(0, false)),
                    max_tokens: Some(200),
                    thoughts_expected: false,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, false);
}

#[tokio::test]
async fn text_response_on_gemini_2_5_flash_lite() {
    const SCENARIO: &str = "thought_text_matrix/text_response_on_gemini_2_5_flash_lite";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_on_gemini_2_5_flash_lite",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: // Spelled out: the crate exports no constant for this model, and the
// point of the cell is a third thinking-capable family.
"gemini-2.5-flash-lite",
                    prompt: THINKING_PROMPT,
                    preamble: None,
                    params: Some(completion_thinking(512, true)),
                    max_tokens: Some(2000),
                    thoughts_expected: true,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_on_gemini_3_1_flash_lite() {
    // Thinking on but `includeThoughts` off: the flag never appears even
    // though the model reasoned, so the reader must behave identically.
    const SCENARIO: &str = "thought_text_matrix/text_response_on_gemini_3_1_flash_lite";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_on_gemini_3_1_flash_lite",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: // No exported constant for this model; a fourth family widens the
// dialect coverage past 2.5's budget and 3-flash's level.
"gemini-3.1-flash-lite",
                    prompt: THINKING_PROMPT,
                    preamble: None,
                    params: Some(json!({
"generationConfig": { "thinkingConfig": { "thinkingLevel": "low" } }
})),
                    max_tokens: Some(400),
                    thoughts_expected: false,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, false);
}

#[tokio::test]
async fn text_response_with_a_large_thinking_budget() {
    const SCENARIO: &str = "thought_text_matrix/text_response_with_a_large_thinking_budget";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette("thought_text_matrix/text_response_with_a_large_thinking_budget", |client| {
        text_response_body(
            client,
            SCENARIO,
            TextResponseCell {
                model_id: gemini::completion::GEMINI_2_5_FLASH,
                prompt:
                    "Explain in one sentence why the sky is blue, after reasoning it through carefully.",
                preamble: None,
                params: Some(completion_thinking(4096, true)),
                max_tokens: Some(6000),
                thoughts_expected: true,
            },
        )
    })
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_with_a_preamble() {
    const SCENARIO: &str = "thought_text_matrix/text_response_with_a_preamble";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_with_a_preamble",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: gemini::completion::GEMINI_2_5_FLASH,
                    prompt: THINKING_PROMPT,
                    preamble: Some("You are a terse calculator. Answer with digits only."),
                    params: Some(completion_thinking(512, true)),
                    max_tokens: Some(2000),
                    thoughts_expected: true,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_on_a_tool_call_turn() {
    const SCENARIO: &str = "thought_text_matrix/text_response_on_a_tool_call_turn";

    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_on_a_tool_call_turn",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("What is 41 plus 1? Use the add tool.")
                .temperature(0.0)
                .max_tokens(2000)
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
                .additional_params(completion_thinking(512, true))
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("raw_completion should succeed");

            // A tool-call turn's *text* is whatever visible text parts it has —
            // never the reasoning that preceded the call.
            let text_response = raw.get_text_response().unwrap_or_default();
            let normalized: rig::completion::CompletionResponse =
                raw.try_into().expect("payload should normalize");
            assert_eq!(
                text_response,
                choice_text(&normalized.choice),
                "the two readers must agree on a tool-call turn too"
            );
            assert!(
                normalized
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                "the recorded turn should carry a tool call"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, &["functionCall"]);
}

#[tokio::test]
async fn text_response_with_structured_output() {
    const SCENARIO: &str = "thought_text_matrix/text_response_with_structured_output";
    // The literal is repeated here on purpose: the cassette safety
    // scan reads each scenario out of this call site's AST.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_with_structured_output",
        |client| {
            text_response_body(
                client,
                SCENARIO,
                TextResponseCell {
                    model_id: gemini::completion::GEMINI_2_5_FLASH,
                    prompt: "Give the city and country of the Eiffel Tower.",
                    preamble: None,
                    params: Some(json!({
                        "generationConfig": {
                            "thinkingConfig": { "thinkingBudget": 512, "includeThoughts": true },
                            "responseMimeType": "application/json",
                            "responseJsonSchema": {
                                "type": "object",
                                "properties": {
                                    "city": { "type": "string" },
                                    "country": { "type": "string" }
                                },
                                "required": ["city", "country"]
                            }
                        }
                    })),
                    max_tokens: Some(2000),
                    thoughts_expected: true,
                },
            )
        },
    )
    .await;

    assert_thought_parts_recorded(SCENARIO, true);
}

#[tokio::test]
async fn text_response_across_two_candidates() {
    const SCENARIO: &str = "thought_text_matrix/text_response_across_two_candidates";

    // `candidateCount: 2` is the one shape where the two readers legitimately
    // differ: `get_text_response` folds every candidate, while the normalized
    // choice is candidate 0 alone. The thought filter must still apply per
    // candidate rather than only to the first.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_across_two_candidates",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Name one primary colour. Answer with the single word.")
                .temperature(0.0)
                .max_tokens(400)
                .additional_params(json!({
                    "generationConfig": {
                        "candidateCount": 2,
                        "thinkingConfig": { "thinkingBudget": 512, "includeThoughts": true }
                    }
                }))
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("raw_completion should succeed");

            assert_eq!(
                raw.candidates.len(),
                2,
                "this cell's premise is a two-candidate turn"
            );
            let (_, thoughts) = split_parts(&raw);
            assert!(
                !thoughts.is_empty(),
                "the recorded turn should carry thought parts"
            );

            let text_response = raw
                .get_text_response()
                .expect("a two-candidate turn has text");
            for thought in &thoughts {
                assert!(
                    !text_response.contains(thought.as_str()),
                    "no candidate's reasoning may reach the folded text response"
                );
            }

            // Every candidate's visible text is represented, and only that.
            let per_candidate: Vec<String> = raw
                .candidates
                .iter()
                .filter_map(|candidate| candidate.content.as_ref())
                .map(|content| {
                    content
                        .parts
                        .iter()
                        .filter(|part| !part.thought.unwrap_or(false))
                        .filter_map(|part| {
                            match &part.part {
                        rig::providers::gemini::completion::gemini_api_types::PartKind::Text(
                            text,
                        ) => Some(text.as_str()),
                        _ => None,
                    }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .collect();
            assert_eq!(
                text_response,
                per_candidate.join("\n"),
                "the folded text response must be exactly every candidate's visible text"
            );

            // And rig's documented normalization still keeps candidate 0 only.
            let normalized: rig::completion::CompletionResponse =
                raw.try_into().expect("payload should normalize");
            assert_eq!(
                choice_text(&normalized.choice),
                per_candidate[0],
                "the normalized choice is the first candidate"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, THOUGHT_MARKER);
}

#[tokio::test]
async fn text_response_is_none_when_the_turn_is_all_thought() {
    const SCENARIO: &str = "thought_text_matrix/text_response_is_none_when_the_turn_is_all_thought";

    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_is_none_when_the_turn_is_all_thought",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            // A budget large enough to start thinking and far too small to answer:
            // the turn truncates with reasoning and no visible text.
            let request = model
                .completion_request(
                    "Prove rigorously, with full detail, that there are infinitely many primes.",
                )
                .temperature(0.0)
                .max_tokens(64)
                .additional_params(completion_thinking(512, true))
                .build();

            let raw = model
                .raw_completion(request)
                .await
                .expect("raw_completion should succeed");

            let visible: Vec<&rig::providers::gemini::completion::gemini_api_types::Part> = raw
                .candidates
                .iter()
                .filter_map(|candidate| candidate.content.as_ref())
                .flat_map(|content| content.parts.iter())
                .filter(|part| !part.thought.unwrap_or(false))
                .collect();

            assert!(
                visible.is_empty(),
                "this cell's premise is a turn whose parts are all thoughts; got {visible:?}"
            );
            assert_eq!(
                raw.get_text_response(),
                None,
                "a turn that produced no visible text has no text response — returning the \
             chain-of-thought here is exactly the bug"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, THOUGHT_MARKER);
}

// --- 18: streaming parity for the same request ----------------------------

#[tokio::test]
async fn streaming_twin_keeps_reasoning_out_of_the_text() {
    const SCENARIO: &str = "thought_text_matrix/streaming_twin_keeps_reasoning_out_of_the_text";

    with_gemini_thought_text_cassette(
        "thought_text_matrix/streaming_twin_keeps_reasoning_out_of_the_text",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(THINKING_PROMPT)
                .temperature(0.0)
                .max_tokens(2000)
                .additional_params(completion_thinking(512, true))
                .build();

            let mut stream = CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");

            let mut text = String::new();
            let mut reasoning = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("no stream item should be an error") {
                    StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                    StreamedAssistantContent::ReasoningDelta { reasoning: r, .. } => {
                        reasoning.push_str(&r)
                    }
                    _ => {}
                }
            }

            assert!(
                !reasoning.is_empty(),
                "the recorded stream should carry reasoning deltas"
            );
            assert!(
                !text.contains(reasoning.trim()),
                "streamed text must not contain the reasoning"
            );
            assert!(
                has_reasoning(&stream.choice),
                "the aggregated choice should keep reasoning as reasoning"
            );
            assert_eq!(
                choice_text(&stream.choice),
                text,
                "the aggregated text must be exactly the streamed text deltas"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, THOUGHT_MARKER);
}

// --- 19-24: states a live turn cannot be made to produce ------------------

#[cfg(test)]
mod unit {
    use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
    use rig::telemetry::ProviderResponseExt;
    use rig::transcription::TranscriptionResponse;
    use serde_json::{Value, json};

    /// A thought part with the shape recorded in
    /// `transcription_with_visible_thoughts_returns_the_transcript`.
    fn thought_part(text: &str) -> Value {
        json!({ "text": text, "thought": true })
    }

    fn text_part(text: &str) -> Value {
        json!({ "text": text })
    }

    fn response_with(parts: Vec<Value>, role: &str) -> GenerateContentResponse {
        serde_json::from_value(json!({
            "candidates": [{
                "content": { "parts": parts, "role": role },
                "finishReason": "STOP",
                "index": 0
            }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "unit-response",
            "usageMetadata": { "promptTokenCount": 190, "candidatesTokenCount": 14, "totalTokenCount": 228 }
        }))
        .expect("recorded-shape payload should deserialize")
    }

    /// Not a recording: a live turn produces one part layout, and the claim
    /// under test is that the transcription reader and the text reader agree
    /// on *every* layout of the same two part kinds.
    #[test]
    fn text_response_matches_the_choice_text() {
        let layouts = [
            vec![thought_part("reasoning"), text_part("answer")],
            vec![text_part("answer"), thought_part("reasoning")],
            vec![
                thought_part("reasoning"),
                text_part("ans"),
                thought_part("more reasoning"),
                text_part("wer"),
            ],
        ];

        for (index, parts) in layouts.into_iter().enumerate() {
            let response = response_with(parts, "model");
            let text_response = response.get_text_response();
            let normalized: rig::completion::CompletionResponse =
                response.try_into().expect("payload should normalize");
            let choice_text = normalized
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            assert_eq!(
                text_response.as_deref().unwrap_or_default(),
                choice_text,
                "layout {index}: readers disagree"
            );
            assert!(
                !text_response
                    .as_deref()
                    .unwrap_or_default()
                    .contains("reasoning"),
                "layout {index}: reasoning leaked into the text response"
            );
        }
    }

    /// Not a recording: Gemini returns the fixture's one-sentence transcript
    /// as a single part, so a split transcript has no live source. The rule
    /// under test is that no part after the first is dropped.
    #[test]
    fn transcription_joins_every_visible_text_part() {
        let response = response_with(
            vec![
                thought_part("I should transcribe this."),
                text_part("The sun was setting slowly, "),
                text_part("casting long shadows across the empty field."),
            ],
            "model",
        );

        let transcription: TranscriptionResponse<GenerateContentResponse> =
            response.try_into().expect("transcription should convert");
        assert_eq!(
            transcription.text,
            "The sun was setting slowly, casting long shadows across the empty field.",
            "every visible text part belongs to the transcript, joined without an \
             invented separator"
        );
    }

    /// Not a recording: a transcription turn always emits the transcript, so
    /// a thought-only candidate cannot be forced. Skipping thoughts must
    /// leave "no transcript" as an error rather than returning the reasoning.
    #[test]
    fn transcription_rejects_a_thought_only_candidate() {
        let response = response_with(vec![thought_part("Let me listen again...")], "model");
        assert_transcription_response_error(
            TranscriptionResponse::<GenerateContentResponse>::try_from(response),
            "a thought-only candidate has no transcript",
        );
    }

    /// `TranscriptionResponse` carries the provider payload and is not
    /// `Debug`, so `expect_err` is unavailable; assert on the `Err` directly.
    fn assert_transcription_response_error(
        result: Result<
            TranscriptionResponse<GenerateContentResponse>,
            rig::transcription::TranscriptionError,
        >,
        context: &str,
    ) {
        match result {
            Ok(response) => panic!("{context}; got transcript {:?}", response.text),
            Err(error) => assert!(
                matches!(
                    error,
                    rig::transcription::TranscriptionError::ResponseError(_)
                ),
                "{context}: expected a ResponseError, got {error:?}"
            ),
        }
    }

    /// Not a recording: Gemini answers this fixture's audio with a real
    /// transcript, so an empty-but-present text part has no live source. "No
    /// text" is a structural question — are there visible text parts at all —
    /// not "is the joined string empty", so a turn whose text part is
    /// genuinely empty converts, exactly as it did before the rewrite.
    #[test]
    fn transcription_keeps_an_empty_visible_text_part() {
        let response = response_with(vec![thought_part("hmm"), text_part("")], "model");
        let transcription: TranscriptionResponse<GenerateContentResponse> = response
            .try_into()
            .expect("an empty visible text part is still a (blank) transcript");
        assert_eq!(transcription.text, "");
    }

    /// Not a recording: `generateContent` does not answer a transcription
    /// request with a media part. The pre-existing rejection for a candidate
    /// carrying no text at all must survive the rewrite.
    #[test]
    fn transcription_rejects_a_candidate_with_no_text_part() {
        let response = response_with(
            vec![json!({ "inlineData": { "mimeType": "image/png", "data": "aGVsbG8=" } })],
            "model",
        );
        assert_transcription_response_error(
            TranscriptionResponse::<GenerateContentResponse>::try_from(response),
            "a candidate with no text part has no transcript",
        );
    }

    /// Not a recording: the live counterpart (cell 17) needs the model to
    /// truncate mid-thought, which is timing-dependent. This states the same
    /// rule directly.
    #[test]
    fn text_response_is_none_for_a_thought_only_candidate() {
        let response = response_with(vec![thought_part("Still working it out...")], "model");
        assert_eq!(
            response.get_text_response(),
            None,
            "a candidate with no visible text has no text response"
        );
    }

    /// Not a recording: `generateContent` never labels a candidate with the
    /// user role. The role filter predates this fix and must not be lost.
    #[test]
    fn text_response_still_ignores_non_model_roles() {
        let response = response_with(vec![text_part("answer")], "user");
        assert_eq!(
            response.get_text_response(),
            None,
            "only model-role candidates contribute text"
        );
    }
}
