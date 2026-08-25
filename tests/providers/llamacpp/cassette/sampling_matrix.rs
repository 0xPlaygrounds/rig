//! Sampling parameters, crossed against what `llama-server` does with them.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10499-6d05498 —
//! except where a cell names another. The smoke tier is enough for every cell
//! here: what is under test is whether a parameter reaches the wire and what
//! the server does with it, not whether the model is clever.
//!
//! | Cell | Parameter | Pinned |
//! | --- | --- | --- |
//! | [`temperature_zero_and_nonzero_both_reach_the_wire`] | `temperature` | 0 serializes as `0.0` rather than being dropped as falsy; a non-zero value round-trips |
//! | [`a_one_token_cap_truncates_with_finish_reason_length`] | `max_tokens: 1` | exactly one completion token, `FinishReason::Length` |
//! | [`a_normal_cap_lets_the_turn_stop_on_its_own`] | `max_tokens` | `FinishReason::Stop`, fewer tokens than the cap |
//! | [`a_cap_past_the_context_is_clamped`] | `max_tokens` past `-c` | recorded in the error matrix; cross-referenced here |
//! | [`a_single_stop_sequence_truncates_the_answer`] | `stop: ["…"]` | the stop text is absent from the answer, `finish_reason: stop` |
//! | [`several_stop_sequences_fire_on_whichever_comes_first`] | `stop: [a, b]` | the earlier one wins |
//! | [`a_stop_sequence_that_never_matches_changes_nothing`] | `stop: ["…"]` | the turn ends exactly as it would have without the sequence |
//! | [`stop_matching_is_case_sensitive`] | `stop` | a case-mismatched sequence does not fire — a real footgun |
//! | [`a_fixed_seed_and_an_absent_seed_are_both_accepted`] | `seed` | present round-trips; absent falls back to the server's `--seed` |
//! | [`additional_params_wins_over_the_typed_field_it_collides_with`] | precedence | `additional_params` overrides a typed builder call, silently |
//!
//! # Two things worth knowing
//!
//! `stop` is **not** a field on rig's [`CompletionRequest`], so every stop cell
//! goes through `additional_params`. That is the supported route — the shared
//! OpenAI request merges `additional_params` into the body — but it means stop
//! sequences are untyped for every provider, and a caller gets no help with
//! the case-sensitivity footgun [`stop_matching_is_case_sensitive`] records.
//!
//! `temperature: 0.0` is the interesting half of the temperature cell.
//! Serializing a zero as "absent" is a classic defect in OpenAI-compatible
//! clients — it turns a deterministic request into a sampled one and nothing
//! in the response says so. The cell reads the recorded request bytes rather
//! than trusting the builder.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, FinishReason};
use serde_json::{Value, json};

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};
use crate::support::assistant_text_response;

use super::super::cassette_support::*;

/// Qwen3 emits a `<think>` trace before answering and the chat-completions
/// route has no switch for it, so prompts that need a short literal answer
/// prefix `/no_think`, which the model's own template honours.
const NO_THINK: &str = "/no_think ";

fn recorded_completion_text(scenario: &str) -> String {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded
        .last()
        .unwrap_or_else(|| panic!("{scenario} should have recorded an interaction"));
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn recorded_finish_reason(scenario: &str) -> String {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (_, body) = recorded.last().expect("an interaction");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

// ---------------------------------------------------------------------------
// temperature
// ---------------------------------------------------------------------------

/// `temperature: 0.0` must arrive as `0.0`, not vanish.
///
/// A client that skips falsy values silently turns every "deterministic
/// please" request into a sampled one. The claim is about the bytes, so it is
/// asserted against the recorded request rather than against the builder.
#[tokio::test]
async fn temperature_zero_and_nonzero_both_reach_the_wire() {
    with_llamacpp_cassette("sampling_matrix/temperature_zero", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Say ok."))
                    .temperature(0.0)
                    .max_tokens(32)
                    .build(),
            )
            .await
            .expect("temperature 0 should be accepted");
    })
    .await;

    with_llamacpp_cassette("sampling_matrix/temperature_nonzero", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Say ok."))
                    .temperature(0.7)
                    .max_tokens(32)
                    .build(),
            )
            .await
            .expect("a non-zero temperature should be accepted");
    })
    .await;

    let zero = recorded_json_request("llamacpp", "sampling_matrix/temperature_zero");
    assert_eq!(
        zero["temperature"],
        json!(0.0),
        "temperature 0 must serialize as 0.0, not be dropped as falsy"
    );
    let nonzero = recorded_json_request("llamacpp", "sampling_matrix/temperature_nonzero");
    assert_eq!(nonzero["temperature"], json!(0.7));
}

// ---------------------------------------------------------------------------
// max_tokens
// ---------------------------------------------------------------------------

/// `max_tokens: 1` stops after exactly one token, reported as `Length`.
#[tokio::test]
async fn a_one_token_cap_truncates_with_finish_reason_length() {
    with_llamacpp_cassette("sampling_matrix/max_tokens_one", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request("Count from one to ten.")
                    .max_tokens(1)
                    .build(),
            )
            .await
            .expect("a one-token cap is a normal request");

        assert_eq!(
            response.finish_reason(),
            Some(FinishReason::Length),
            "a cap that truncates must be reported as Length, not Stop"
        );
        assert_eq!(
            response.usage.output_tokens, 1,
            "the server must honour the cap exactly"
        );
    })
    .await;

    assert_eq!(
        recorded_finish_reason("sampling_matrix/max_tokens_one"),
        "length"
    );
    assert_eq!(
        recorded_json_request("llamacpp", "sampling_matrix/max_tokens_one")["max_tokens"],
        json!(1)
    );
}

/// A cap the turn does not reach leaves `finish_reason: stop`.
#[tokio::test]
async fn a_normal_cap_lets_the_turn_stop_on_its_own() {
    with_llamacpp_cassette("sampling_matrix/max_tokens_normal", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Reply with the single word: ok"))
                    .max_tokens(512)
                    .build(),
            )
            .await
            .expect("a generous cap is a normal request");

        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert!(
            response.usage.output_tokens < 512,
            "the turn stopped on its own, so it used fewer tokens than the cap"
        );
    })
    .await;

    assert_eq!(
        recorded_finish_reason("sampling_matrix/max_tokens_normal"),
        "stop"
    );
}

/// A cap past the context window is clamped rather than refused.
///
/// The recorded evidence lives in the error matrix, which owns the `-c 512`
/// server; this cell exists so the sampling table has a row for the third
/// `max_tokens` arm rather than an unexplained gap. Cross-referencing a
/// fixture instead of recording a second copy of it is the deliberate choice.
#[test]
fn a_cap_past_the_context_is_clamped() {
    let request = recorded_json_request("llamacpp", "error_matrix/oversized_output_cap");
    assert_eq!(request["max_tokens"], json!(100_000));
    let recorded = recorded_statuses_and_bodies("llamacpp", "error_matrix/oversized_output_cap");
    assert_eq!(recorded[0].0, 200, "clamped, not refused");
}

// ---------------------------------------------------------------------------
// stop sequences
// ---------------------------------------------------------------------------

/// One stop sequence truncates the answer before the sequence itself.
#[tokio::test]
async fn a_single_stop_sequence_truncates_the_answer() {
    with_llamacpp_cassette("sampling_matrix/stop_single", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!(
                        "{NO_THINK}Write exactly this and nothing else: Alpha Bravo Charlie Delta"
                    ))
                    .max_tokens(64)
                    .additional_params(json!({ "stop": ["Charlie"] }))
                    .build(),
            )
            .await
            .expect("a stop sequence is a normal request");

        let text = assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            !text.contains("Charlie"),
            "the stop text must not appear in the answer: {text:?}"
        );
        assert_eq!(
            response.finish_reason(),
            Some(FinishReason::Stop),
            "a stop sequence terminates the turn as a stop, not a length cut"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "sampling_matrix/stop_single");
    assert_eq!(
        request["stop"],
        json!(["Charlie"]),
        "additional_params must merge the stop list into the body"
    );
    let text = recorded_completion_text("sampling_matrix/stop_single");
    assert!(!text.contains("Charlie"), "{text:?}");
    assert_eq!(
        recorded_finish_reason("sampling_matrix/stop_single"),
        "stop"
    );
}

/// With several stop sequences, whichever the model reaches first wins.
#[tokio::test]
async fn several_stop_sequences_fire_on_whichever_comes_first() {
    with_llamacpp_cassette("sampling_matrix/stop_multiple", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!(
                        "{NO_THINK}Write exactly this and nothing else: Alpha Bravo Charlie Delta"
                    ))
                    .max_tokens(64)
                    // `Zulu` never appears; `Bravo` appears before `Charlie`.
                    .additional_params(json!({ "stop": ["Zulu", "Charlie", "Bravo"] }))
                    .build(),
            )
            .await
            .expect("several stop sequences are a normal request");

        let text = assistant_text_response(&response.choice).unwrap_or_default();
        for sequence in ["Bravo", "Charlie"] {
            assert!(
                !text.contains(sequence),
                "the answer stopped at the first match, so neither later sequence \
                 can appear: {text:?}"
            );
        }
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    })
    .await;

    let request = recorded_json_request("llamacpp", "sampling_matrix/stop_multiple");
    assert_eq!(request["stop"], json!(["Zulu", "Charlie", "Bravo"]));
    let text = recorded_completion_text("sampling_matrix/stop_multiple");
    assert!(text.contains("Alpha"), "the prefix survives: {text:?}");
    assert!(!text.contains("Bravo"), "{text:?}");
}

/// A stop sequence the model never produces changes nothing.
#[tokio::test]
async fn a_stop_sequence_that_never_matches_changes_nothing() {
    with_llamacpp_cassette("sampling_matrix/stop_never_fires", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!(
                        "{NO_THINK}Write exactly this and nothing else: Alpha Bravo Charlie Delta"
                    ))
                    .max_tokens(64)
                    .additional_params(json!({ "stop": ["QQZZXX-never-emitted"] }))
                    .build(),
            )
            .await
            .expect("an unmatched stop sequence is a normal request");

        let text = assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.contains("Delta"),
            "nothing truncated the answer, so the last word survives: {text:?}"
        );
    })
    .await;

    let text = recorded_completion_text("sampling_matrix/stop_never_fires");
    assert!(text.contains("Delta"), "{text:?}");
}

/// llama.cpp matches stop sequences **case-sensitively**.
///
/// A footgun with no client-side guard: the same sequence in the wrong case
/// silently does nothing, and the only symptom is a longer answer than
/// expected. The cell sends a lowercase sequence for text the model writes
/// capitalized and pins that the answer runs past it.
#[tokio::test]
async fn stop_matching_is_case_sensitive() {
    with_llamacpp_cassette("sampling_matrix/stop_case_sensitive", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request(format!(
                        "{NO_THINK}Write exactly this and nothing else: Alpha Bravo Charlie Delta"
                    ))
                    .max_tokens(64)
                    .additional_params(json!({ "stop": ["charlie"] }))
                    .build(),
            )
            .await
            .expect("a case-mismatched stop sequence is still a valid request");

        let text = assistant_text_response(&response.choice).unwrap_or_default();
        assert!(
            text.contains("Charlie"),
            "a lowercase stop sequence must not fire on capitalized text — that is \
             the behaviour this cell records: {text:?}"
        );
    })
    .await;

    let request = recorded_json_request("llamacpp", "sampling_matrix/stop_case_sensitive");
    assert_eq!(request["stop"], json!(["charlie"]));
    let text = recorded_completion_text("sampling_matrix/stop_case_sensitive");
    assert!(
        text.contains("Charlie"),
        "the recorded answer must run past the mismatched sequence: {text:?}"
    );
}

// ---------------------------------------------------------------------------
// seed
// ---------------------------------------------------------------------------

/// `seed` present round-trips; absent falls back to the server's `--seed`.
///
/// Both halves are worth a cell because a recording harness depends on them:
/// the corpus is reproducible only because the *server* was started with
/// `--seed 42`, and a request that silently injected a different seed would
/// make every fixture in this suite unreproducible without saying so.
#[tokio::test]
async fn a_fixed_seed_and_an_absent_seed_are_both_accepted() {
    with_llamacpp_cassette("sampling_matrix/seed_fixed", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Say ok."))
                    .max_tokens(32)
                    .additional_params(json!({ "seed": 7 }))
                    .build(),
            )
            .await
            .expect("an explicit seed should be accepted");
    })
    .await;

    with_llamacpp_cassette("sampling_matrix/seed_absent", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        model
            .completion(
                model
                    .completion_request(format!("{NO_THINK}Say ok."))
                    .max_tokens(32)
                    .build(),
            )
            .await
            .expect("no seed should be accepted");
    })
    .await;

    assert_eq!(
        recorded_json_request("llamacpp", "sampling_matrix/seed_fixed")["seed"],
        json!(7)
    );
    assert!(
        recorded_json_request("llamacpp", "sampling_matrix/seed_absent")
            .get("seed")
            .is_none(),
        "rig must not invent a seed the caller did not ask for; the corpus's \
         determinism comes from the server's --seed"
    );
}

// ---------------------------------------------------------------------------
// Precedence
// ---------------------------------------------------------------------------

/// `additional_params` **overrides** a typed field of the same name.
///
/// Half this matrix reaches the wire through `additional_params` — `stop`,
/// `seed`, `grammar`, `n`, `logprobs` all have no typed home — so which side
/// wins when the two collide is load-bearing for reading any of these
/// fixtures, and nothing stated it.
///
/// It is the escape hatch's `#[serde(flatten)]` that decides: the typed fields
/// serialize first and the flattened map is written over them, so
/// `additional_params` wins. That is defensible as an escape hatch, and it is
/// silent — a caller who sets `max_tokens(7)` and then passes an
/// `additional_params` blob that happens to carry `max_tokens` gets 99 with no
/// warning.
///
/// Definitional rather than observed: this is rig's serialization, not
/// llama.cpp's parsing, so it is checked without a server.
#[test]
fn additional_params_wins_over_the_typed_field_it_collides_with() {
    use rig::providers::openai::completion::OpenAICompatibleProvider as _;

    let request = rig::completion::CompletionRequest {
        model: None,
        chat_history: vec![rig::message::Message::User {
            content: vec![rig::message::UserContent::text("hi")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(7),
        tool_choice: None,
        additional_params: Some(json!({ "max_tokens": 99, "top_k": 3 })),
        output_schema: None,
        record_telemetry_content: false,
    };

    let typed = rig::providers::llamacpp::LlamacppExt
        .build_completion_request("m".to_string(), request, Default::default())
        .expect("the request should build");
    let body = serde_json::to_value(&typed).expect("the request should serialize");

    assert_eq!(
        body["max_tokens"],
        json!(99),
        "the escape hatch overrides the typed field it collides with"
    );
    assert_eq!(
        body["top_k"],
        json!(3),
        "and a key with no typed counterpart passes through unchanged — which is \
         how `stop`, `seed`, `grammar`, `n` and `logprobs` reach the wire in this \
         matrix"
    );
    assert_eq!(
        body["temperature"],
        json!(0.0),
        "a typed field the blob does not name is untouched"
    );
}
