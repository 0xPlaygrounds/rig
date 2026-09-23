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
//! * `NormalizeTranscriptionResponse for GenerateContentResponse` read
//!   `parts.first()`. With thoughts on, parts[0] is the reasoning — so
//!   `response.text` was **the model's private reasoning** and the actual
//!   transcript, sitting in parts[1], was dropped. A transcript split across
//!   several text parts lost everything after the first, too.
//! * `ProviderResponseExt::text_response` collected *every* text part,
//!   gluing the chain-of-thought onto the answer — a second reader of the
//!   same document, disagreeing with the first.
//!
//! The transcription reader goes through `visible_text_parts`, the one place
//! the skip rule lives. The second reader is gone with the client layer:
//! `text_response` had no caller left once the driver started recording
//! telemetry off the normalized response. So the cells that held the two
//! readers to each other now hold the one that remains — the GenerateContent
//! decoder — to the same rule: **the normalized text of a turn is exactly
//! its visible text parts, and carries no thought text.**
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
//! | 7 | `text_response_skips_thoughts_on_gemini_2_5_flash` | `text_response` | the bug's second reader |
//! | 8 | `text_response_skips_thoughts_on_gemini_3_flash` | `text_response` | `thinkingLevel` dialect |
//! | 9 | `text_response_with_thinking_disabled_is_unchanged` | `text_response` | no-thoughts regression guard |
//! | 10 | `text_response_on_gemini_2_5_flash_lite` | `text_response` | third model |
//! | 11 | `text_response_on_gemini_3_1_flash_lite` | `text_response` | pre-thinking model family |
//! | 12 | `text_response_with_a_large_thinking_budget` | `text_response` | long reasoning |
//! | 13 | `text_response_with_a_preamble` | `text_response` | `systemInstruction` present |
//! | 14 | `text_response_on_a_tool_call_turn` | `text_response` | thought part beside a `functionCall` |
//! | 15 | `text_response_with_structured_output` | `text_response` | `responseJsonSchema` turn |
//! | 16 | `text_response_across_two_candidates` | `text_response` | `candidateCount: 2` |
//! | 17 | `text_response_is_none_when_the_turn_is_all_thought` | `text_response` | budget spent entirely on thinking |
//! | 18 | `streaming_twin_keeps_reasoning_out_of_the_text` | stream | streaming parity for the same request |
//! | 19 | `text_response_matches_the_choice_text` (unit) | both | see below |
//! | 20 | `transcription_joins_every_visible_text_part` (unit) | transcription | see below |
//! | 21 | `transcription_rejects_a_thought_only_candidate` (unit) | transcription | see below |
//! | 22 | `transcription_rejects_a_candidate_with_no_text_part` (unit) | transcription | see below |
//! | 23 | `text_response_is_none_for_a_thought_only_candidate` (unit) | `text_response` | see below |
//! | 24 | — | — | deleted with the reader it tested; see below |
//! | 25 | `transcription_keeps_an_empty_visible_text_part` (unit) | transcription | see below |
//! | 26 | `blocking_keeps_a_trailing_thought_signature` | blocking choice | Gemini 3's no-`thought`-flag signature |
//! | 27 | `streaming_twin_agrees_on_a_trailing_thought_signature` | stream | the signature stays on its own part |
//! | 28 | `a_trailing_signature_stays_on_its_text` (unit) | blocking | see below |
//! | 29 | `a_thought_flagged_part_still_signs_its_own_reasoning` (unit) | blocking | see below |
//! | 30 | `a_text_part_without_a_signature_yields_no_reasoning` (unit) | blocking | see below |
//! | 31 | `transcription_rejects_a_candidate_with_no_parts_at_all` (unit) | transcription | see below |
//! | 32 | `a_text_signature_does_not_sign_the_chain_of_thought` (unit) | blocking | see below |
//!
//! Cells 26–32 cover Gemini 3's `thoughtSignature` on an answer part carrying
//! no `thought` flag. Gemini requires every signature back inside the part
//! that carried it, never merged into another part, so it stays on that
//! answer text (`gemini::text_thought_signature`) rather than on reasoning.
//! Cells 26–27 are recorded; 28–32 state orderings one live turn cannot
//! emit.
//!
//! The cells marked `(unit)` are unit tests because a live turn cannot be
//! made to produce their states: Gemini does not split a short transcript
//! across several visible text parts on demand, does not return a
//! transcription candidate consisting only of thoughts, never labels a
//! `generateContent` candidate with a non-model role, and emits one part
//! ordering per turn. Every one states its shape from bytes recorded
//! elsewhere in this matrix.
//!
//! No cell was dropped for cost.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test -p rig --all-features --test gemini thought_text_matrix -- --test-threads=1`
//!
//! The `reader` column's `text_response` entries name the cells (whose
//! function names are frozen), not a surviving method: on those cells the
//! reader under test is now the decoder's normalized `choice`.
//!
//! Cell 24, `text_response_still_ignores_non_model_roles`, is deleted rather
//! than restated. The role filter it pinned existed only inside
//! `ProviderResponseExt::text_response` (`content.role != Role::Model` →
//! contribute nothing); the decoder reads the first candidate's parts without
//! consulting `role`, and `visible_text_parts` — the surviving skip rule —
//! never filtered on it either. Nothing replaces the cell because nothing
//! replaces the filter.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
use rig::streaming::{Delta, StreamEvent};
use rig::transcription::TranscriptionRequestBuilder;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{
    BoundGemini, assert_recorded_response_contains, assert_recorded_response_excludes,
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

/// The text of a normalized choice, concatenated: the decoder's text blocks
/// are the turn's text parts, and a part boundary is not a separator the
/// provider sent.
fn choice_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
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
fn split_parts(response: &GenerateContentResponse) -> (String, Vec<String>) {
    use rig::providers::gemini::completion::gemini_api_types::PartKind;

    let mut visible = String::new();
    let mut thoughts = Vec::new();
    // The first candidate only — that is the one the transcription mapper
    // reads, and a helper that ranged wider would stop describing the rule
    // under test the moment a fixture had two candidates.
    for part in response
        .candidates
        .first()
        .and_then(|candidate| candidate.content.as_ref())
        .into_iter()
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
    client: BoundGemini,
    scenario: &'static str,
    model_id: &'static str,
    params: Option<Value>,
    temperature: Option<f64>,
    thoughts_expected: bool,
    expected_words: Option<&'static str>,
) {
    let model = client.transcription(model_id);
    let mut request = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
        .expect("audio fixture should load");
    if let Some(params) = params {
        request = request.additional_params(params);
    }
    if let Some(temperature) = temperature {
        request = request.temperature(temperature);
    }

    let response = request.send().await.expect("transcription should succeed");
    let raw: GenerateContentResponse = serde_json::from_value(response.raw.clone())
        .expect("raw payload should round-trip to Gemini's own response type");
    let (visible, thoughts) = split_parts(&raw);

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

// --- 7-17: the `text_response` reader ---------------------------------

/// One `text_response` cell: the provider-native reader and the
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

async fn text_response_body(client: BoundGemini, scenario: &'static str, cell: TextResponseCell) {
    let TextResponseCell {
        model_id,
        prompt,
        preamble,
        params,
        max_tokens,
        thoughts_expected,
    } = cell;

    let model = client.completion(model_id);
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

    let response = model
        .completion(request)
        .await
        .expect("completion should succeed");

    let document = GenerateContentResponse::deserialize(&response.raw)
        .expect("raw is Gemini's own generateContent document");
    let (visible, recorded_thoughts) = split_parts(&document);
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

    // The one reader: the normalized text of the turn is exactly its visible
    // text parts, and none of the chain-of-thought.
    let text = choice_text(&response.choice);
    for thought in &recorded_thoughts {
        assert!(
            !text.contains(thought.as_str()),
            "{scenario}: the turn's text must not carry the chain-of-thought"
        );
    }
    assert_eq!(
        text, visible,
        "{scenario}: the normalized text must be every visible text part and only those"
    );

    // Reasoning blocks come from `thought: true` parts only. A signature on
    // an answer part stays on that answer's text, to return in its own part.
    assert_eq!(
        has_reasoning(&response.choice),
        thoughts_expected,
        "{scenario}: reasoning blocks must appear exactly when the turn carried thought text"
    );
    let answer_signatures: Vec<&str> = document
        .candidates
        .first()
        .and_then(|candidate| candidate.content.as_ref())
        .into_iter()
        .flat_map(|content| content.parts.iter())
        .filter(|part| !part.thought.unwrap_or(false))
        .filter_map(|part| part.thought_signature.as_deref())
        .collect();
    assert_eq!(
        text_signatures(&response.choice),
        answer_signatures,
        "{scenario}: every answer-part signature stays on its text"
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
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
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

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            // A tool-call turn's *text* is whatever visible text parts it has
            // — never the reasoning that preceded the call.
            let document = GenerateContentResponse::deserialize(&response.raw)
                .expect("raw is Gemini's own generateContent document");
            let (visible, _) = split_parts(&document);
            assert_eq!(
                choice_text(&response.choice),
                visible,
                "the turn's text is its visible text parts on a tool-call turn too"
            );
            assert!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                "the recorded turn should carry a tool call"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, &["functionCall"]);
    // The cell's premise is a thought part *beside* the call; without this it
    // silently degrades into a plain tool-call cell if a re-record drops the
    // thought.
    assert_thought_parts_recorded(SCENARIO, true);
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

    // `candidateCount: 2` is where a folding reader and the normalized choice
    // legitimately differ: the choice is candidate 0 alone. The thought
    // filter must still apply, and no second candidate may be folded in.
    with_gemini_thought_text_cassette(
        "thought_text_matrix/text_response_across_two_candidates",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
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

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            let document = GenerateContentResponse::deserialize(&response.raw)
                .expect("raw is Gemini's own generateContent document");
            assert_eq!(
                document.candidates.len(),
                2,
                "this cell's premise is a two-candidate turn"
            );
            let (_, thoughts) = split_parts(&document);
            assert!(
                !thoughts.is_empty(),
                "the recorded turn should carry thought parts"
            );

            // Every candidate's visible text, per candidate.
            let per_candidate: Vec<String> = document
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
                        .collect::<String>()
                })
                .collect();
            assert!(
                per_candidate.len() == 2 && !per_candidate[1].is_empty(),
                "premise: the second candidate carries visible text of its own, got \
                 {per_candidate:?}"
            );

            let text = choice_text(&response.choice);
            for thought in &thoughts {
                assert!(
                    !text.contains(thought.as_str()),
                    "no candidate's reasoning may reach the turn's text"
                );
            }
            assert_eq!(
                text, per_candidate[0],
                "the normalized choice is the first candidate's visible text"
            );
            assert_ne!(
                text,
                per_candidate.concat(),
                "and it is candidate 0 alone — a reader that folded every candidate would land \
                 here"
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
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
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

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            let document = GenerateContentResponse::deserialize(&response.raw)
                .expect("raw is Gemini's own generateContent document");
            let (visible, thoughts) = split_parts(&document);
            assert!(
                visible.is_empty(),
                "this cell's premise is a turn whose parts are all thoughts; got {visible:?}"
            );
            assert!(
                !thoughts.is_empty(),
                "the recorded turn should carry thought parts"
            );
            assert_eq!(
                choice_text(&response.choice),
                "",
                "a turn that produced no visible text has no text — returning the \
                 chain-of-thought here is exactly the bug"
            );
            assert!(
                has_reasoning(&response.choice),
                "the thoughts reach the caller as reasoning, not as the answer"
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
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
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
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text: chunk },
                        ..
                    } => text.push_str(&chunk),
                    StreamEvent::BlockDelta {
                        delta: Delta::Reasoning { text: r },
                        ..
                    } => {
                        reasoning.push_str(&r);
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
                has_reasoning(&stream.snapshot()),
                "the aggregated choice should keep reasoning as reasoning"
            );
            assert_eq!(
                choice_text(&stream.snapshot()),
                text,
                "the aggregated text must be exactly the streamed text deltas"
            );
        },
    )
    .await;

    assert_recorded_response_contains(SCENARIO, THOUGHT_MARKER);
}

// --- 26-27: Gemini 3's trailing thought signature -------------------------

/// A Gemini 3 prompt short enough that the turn is one text part — which is
/// where the wire hangs the trailing `thoughtSignature`.
const SIGNATURE_PROMPT: &str = "What is 17 squared? Answer with the number only.";

/// The signatures the answer texts of `choice` carry, in order.
fn text_signatures(choice: &[AssistantContent]) -> Vec<&str> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => gemini::text_thought_signature(text),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn blocking_keeps_a_trailing_thought_signature() {
    with_gemini_thought_text_cassette(
        "thought_text_matrix/blocking_keeps_a_trailing_thought_signature",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW);
            let request = model
                .completion_request(SIGNATURE_PROMPT)
                .temperature(0.0)
                .max_tokens(1000)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            // The premise, from the recorded bytes: a text part with a
            // signature and no `thought` flag at all.
            let document = GenerateContentResponse::deserialize(&response.raw)
                .expect("raw is Gemini's own generateContent document");
            let signed_text_part = document
                .candidates
                .first()
                .and_then(|candidate| candidate.content.as_ref())
                .is_some_and(|content| {
                    content.parts.iter().any(|part| {
                        part.thought_signature.is_some() && !part.thought.unwrap_or(false)
                    })
                });
            assert!(
                signed_text_part,
                "this cell's premise is a signed part with no thought flag"
            );

            assert_eq!(
                text_signatures(&response.choice).len(),
                1,
                "the signature is replay-required state and stays on the answer text: {:?}",
                response.choice
            );
            assert!(!has_reasoning(&response.choice));
            assert!(
                choice_text(&response.choice).contains("289"),
                "the answer must still be there, got {:?}",
                choice_text(&response.choice)
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_twin_agrees_on_a_trailing_thought_signature() {
    with_gemini_thought_text_cassette(
        "thought_text_matrix/streaming_twin_agrees_on_a_trailing_thought_signature",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW);
            let request = model
                .completion_request(SIGNATURE_PROMPT)
                .temperature(0.0)
                .max_tokens(1000)
                .build();

            let mut stream = CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");
            while stream.next().await.is_some() {}

            // The stream sends the answer, then an empty part carrying the
            // signature: that empty part keeps it, on its own text.
            let snapshot = stream.snapshot();
            assert_eq!(text_signatures(&snapshot).len(), 1, "{snapshot:?}");
            assert!(!has_reasoning(&snapshot));
            assert!(
                choice_text(&stream.snapshot()).contains("289"),
                "the streamed answer must be there, got {:?}",
                choice_text(&stream.snapshot())
            );
        },
    )
    .await;
}

// --- unit cells: states a live turn cannot be made to produce -------------

mod unit {
    use rig::completion::{CompletionModel, CompletionResponse};
    use rig::message::{AssistantContent, ReasoningContent};
    use rig::prelude::*;
    use rig::providers::gemini::Gemini;
    use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
    use rig::test_utils::RecordingHttpClient;
    use rig::transcription::{NormalizeTranscriptionResponse, TranscriptionResponse};
    use serde_json::{Value, json};

    /// A thought part with the shape recorded in
    /// `transcription_with_visible_thoughts_returns_the_transcript`.
    fn thought_part(text: &str) -> Value {
        json!({ "text": text, "thought": true })
    }

    fn text_part(text: &str) -> Value {
        json!({ "text": text })
    }

    /// The recorded reply shape, with `parts` as the candidate's content.
    fn reply_with(parts: Vec<Value>, role: &str) -> Value {
        json!({
            "candidates": [{
                "content": { "parts": parts, "role": role },
                "finishReason": "STOP",
                "index": 0
            }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "unit-response",
            "usageMetadata": { "promptTokenCount": 190, "candidatesTokenCount": 14, "totalTokenCount": 228 }
        })
    }

    fn response_with(parts: Vec<Value>, role: &str) -> GenerateContentResponse {
        serde_json::from_value(reply_with(parts, role))
            .expect("recorded-shape payload should deserialize")
    }

    /// One completion turn through the one seam, answered by a stub transport
    /// with `parts`.
    ///
    /// These part layouts cannot be produced live, so the bytes are stated
    /// here and carried by the real wire, driver and decoder — the same path
    /// every recorded cell above runs, with the reply substituted.
    async fn completion_of(parts: Vec<Value>, role: &str) -> CompletionResponse {
        let model = Gemini::new("unit-key")
            .bind(RecordingHttpClient::new(
                reply_with(parts, role).to_string(),
            ))
            .completion("gemini-2.5-flash");
        let request = model.completion_request("unit").build();
        model
            .completion(request)
            .await
            .expect("the stubbed reply should convert")
    }

    fn choice_text(response: &CompletionResponse) -> String {
        response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Not a recording: a live turn produces one part layout, and the claim
    /// under test is that the decoder applies the skip rule to *every* layout
    /// of the same two part kinds. Each layout's visible parts spell
    /// `answer`.
    #[tokio::test]
    async fn text_response_matches_the_choice_text() {
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
            let response = completion_of(parts, "model").await;
            let text = choice_text(&response);
            assert_eq!(
                text, "answer",
                "layout {index}: the turn's text is its visible parts, in order"
            );
            assert!(
                !text.contains("reasoning"),
                "layout {index}: reasoning leaked into the turn's text"
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

        let transcription = response
            .normalize("gcp.gemini")
            .expect("transcription should convert");
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
            response.normalize("gcp.gemini"),
            "a thought-only candidate has no transcript",
        );
    }

    fn assert_transcription_response_error(
        result: Result<TranscriptionResponse, rig::error::ProviderError>,
        context: &str,
    ) {
        match result {
            Ok(response) => panic!("{context}; got transcript {:?}", response.text),
            Err(error) => assert!(
                matches!(error, rig::error::ProviderError::Response(_)),
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
        let transcription = response
            .normalize("gcp.gemini")
            .expect("an empty visible text part is still a (blank) transcript");
        assert_eq!(transcription.text, "");
    }

    /// Not a recording as a *transcription* turn, though the shape is real:
    /// `generateContent` answers `{"content":{"role":"model"}}` — content
    /// present, `parts` absent entirely — when the output budget runs out
    /// before any part is produced (confirmed live with
    /// `maxOutputTokens: 1`). The structural "no visible text part" check has
    /// to reject that as firmly as it rejects a media-only candidate, rather
    /// than returning a blank transcript.
    #[test]
    fn transcription_rejects_a_candidate_with_no_parts_at_all() {
        let response: GenerateContentResponse = serde_json::from_value(json!({
            "candidates": [{ "content": { "role": "model" }, "finishReason": "MAX_TOKENS" }],
            "modelVersion": "gemini-2.5-flash",
            "responseId": "unit-response",
            "usageMetadata": { "promptTokenCount": 14, "totalTokenCount": 14 }
        }))
        .expect("recorded-shape payload should deserialize");
        assert_transcription_response_error(
            response.normalize("gcp.gemini"),
            "a candidate with no parts at all has no transcript",
        );
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
            response.normalize("gcp.gemini"),
            "a candidate with no text part has no transcript",
        );
    }

    /// Not a recording: the live counterpart (cell 17) needs the model to
    /// truncate mid-thought, which is timing-dependent. This states the same
    /// rule directly: a candidate with no visible text contributes no text,
    /// and its thoughts arrive as reasoning.
    #[tokio::test]
    async fn text_response_is_none_for_a_thought_only_candidate() {
        let response = completion_of(vec![thought_part("Still working it out...")], "model").await;
        assert_eq!(
            choice_text(&response),
            "",
            "a candidate with no visible text has no text"
        );
        assert!(
            response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::Reasoning(_))),
            "the thought reaches the caller as reasoning, got {:?}",
            response.choice
        );
    }

    /// Not a recording: one live turn emits one part ordering. When a
    /// visible thought part precedes the signed answer, the signature stays
    /// on the answer: Gemini requires it back inside the part that carried
    /// it, and merging it into the thought part is what corrupts replay.
    #[tokio::test]
    async fn a_text_signature_does_not_sign_the_chain_of_thought() {
        let response = completion_of(
            vec![
                json!({ "text": "the chain", "thought": true }),
                json!({ "text": "answer", "thoughtSignature": "sig-trailing" }),
            ],
            "model",
        )
        .await;
        assert_eq!(response.choice.len(), 2, "{:?}", response.choice);
        assert!(
            matches!(
                response.choice.first(),
                Some(AssistantContent::Reasoning(reasoning))
                    if matches!(reasoning.content.first(),
                        Some(ReasoningContent::Text { text, signature: None }) if text == "the chain")
            ),
            "the chain-of-thought block stays unsigned, got {:?}",
            response.choice
        );
        assert!(matches!(
            response.choice.get(1),
            Some(AssistantContent::Text(text))
                if text.text == "answer"
                    && rig::providers::gemini::text_thought_signature(text) == Some("sig-trailing")
        ));
    }

    /// Not a recording: one part carrying both the answer and its signature
    /// stays one text with that signature.
    #[tokio::test]
    async fn a_trailing_signature_stays_on_its_text() {
        let response = completion_of(
            vec![json!({ "text": "17 squared is 289.", "thoughtSignature": "sig-trailing" })],
            "model",
        )
        .await;
        assert_eq!(response.choice.len(), 1, "{:?}", response.choice);
        assert!(matches!(
            response.choice.first(),
            Some(AssistantContent::Text(text))
                if text.text == "17 squared is 289."
                    && rig::providers::gemini::text_thought_signature(text) == Some("sig-trailing")
        ));
    }

    /// Not a recording: the counterpart ordering. A `thought: true` part signs
    /// its own reasoning block and must not also grow a sibling.
    #[tokio::test]
    async fn a_thought_flagged_part_still_signs_its_own_reasoning() {
        let response = completion_of(
            vec![
                json!({ "text": "thinking", "thought": true, "thoughtSignature": "sig-own" }),
                json!({ "text": "answer" }),
            ],
            "model",
        )
        .await;
        assert_eq!(response.choice.len(), 2, "one reasoning block, one text");
        assert!(matches!(
            response.choice.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if matches!(reasoning.content.first(),
                    Some(ReasoningContent::Text { text, signature })
                        if text == "thinking" && signature.as_deref() == Some("sig-own"))
        ));
    }

    /// Not a recording: the negative case. No signature, no reasoning block —
    /// the fix must not manufacture one for every text part.
    #[tokio::test]
    async fn a_text_part_without_a_signature_yields_no_reasoning() {
        let response = completion_of(vec![text_part("plain answer")], "model").await;
        assert_eq!(response.choice.len(), 1);
        assert!(matches!(
            response.choice.first(),
            Some(AssistantContent::Text(_))
        ));
    }
}
