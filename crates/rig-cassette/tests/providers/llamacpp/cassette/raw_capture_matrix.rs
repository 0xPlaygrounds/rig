//! Matrix for raw response capture on llama.cpp's blocking path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. llama.cpp is the `LLAMACPP` dialect of the OpenAI
//! `Chat` wire, and the driver sets `raw` from the reply's bytes, so `raw` is
//! the server's response *document* — not a re-serialization of whatever type
//! the decoder parsed. Nothing about it is sent to the server.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by
//! hand with no provider payload behind it, which no cell here can produce.
//!
//! Two things follow from `raw` being the document, and this file pins both.
//!
//! The chat-completions body carries fields the normalized
//! [`rig::completion::CompletionResponse`] has no home for — `object`,
//! `created`, `system_fingerprint`, and llama.cpp's own `timings` — and a
//! caller reaches them through `raw`. That is cells 2 and 4.
//!
//! And `raw` reads back into llama.cpp's own response type,
//! [`llamacpp::CompletionResponse`](rig::providers::llamacpp::CompletionResponse),
//! which is the documented typed escape hatch. Cells 1 and 3 use it: 1 that
//! the document parses, 3 that every normalized field the decoder produced
//! agrees with the provider-native field it came from. Cell 3 is deliberately
//! *not* a comparison of two normalizations — there is one decoder and one
//! mapping now, so comparing it to a second mapping would compare it to a
//! copy of itself.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns: a fixture without the envelope fields fails
//! loudly.
//!
//! Every cell runs its one turn through
//! [`capture_completion`](crate::raw_capture::capture_completion) and keeps
//! its cassette wrapper call, scenario literal included, at the test site:
//! `cassette_safety` reads every scenario out of the AST and accepts only a
//! literal there, so a shared runner that took the scenario as a parameter
//! would register nothing and orphan four fixtures.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_reads_back_as_the_provider_type` | typed access | `raw` is the recorded reply document and `llamacpp::CompletionResponse::deserialize(&raw)` parses it | recorded |
//! | 2 | `raw_exposes_envelope_fields` | provider-only fields | `object`/`created`/`system_fingerprint` in `raw` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_the_typed_raw` | normalized view | every normalized field equals the provider-native field on `raw` that produced it | recorded |
//! | 4 | `raw_preserves_the_timings_the_openai_type_drops` | Part 4: dropped fields | `timings` survives into `raw`; the same bytes read as `openai::CompletionResponse` lose it | recorded |
//!
//! Cell 4 is the one that justifies this provider having its own response
//! type at all. `timings` is llama.cpp's server-side latency accounting and
//! the only such accounting a caller gets — for local inference,
//! `predicted_per_second` is the number people watch. The shared
//! `openai::CompletionResponse` has neither a field for it nor a catch-all, so
//! a caller who reads `raw` through that type drops it silently, while
//! `llamacpp::CompletionResponse` keeps it. The asymmetry is real and cell 4
//! plus `raw_stream_capture_matrix`'s timings cell are what pin both halves.
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; two cell names changed because the old
//! ones described a re-serialization fixed point and a re-normalization that
//! the code no longer performs.
//!
//! **Server**: the default configuration — `unsloth/Qwen3-1.7B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 4096`, `llama-server` b10964-b29c606e2.
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test llamacpp raw_capture_matrix -- --test-threads=1`

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::{llamacpp, openai};
use serde::Deserialize;
use serde_json::Value;

use super::super::cassette_support::*;
use crate::cassettes::{CassetteMode, recorded_json_turn};
use crate::raw_capture::{assert_no_request_id, assert_normalized_lacks, capture_completion, chat};
use crate::support::{Observed, assert_wire_value_matches, assistant_text, normalized_without_raw};

const LLAMACPP_PROVIDER: &str = "llamacpp";
const PROMPT: &str = "Reply with exactly the single word: pong";

/// Qwen3 spends tokens on a reasoning trace before the one-word answer and the
/// chat-completions route has no `think` switch, so the cap is generous
/// enough that the turn stops on its own (`finish_reason: "stop"`).
fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(1024).build()
}

/// The premise every envelope cell rests on: the recorded body is a
/// chat-completions response carrying the envelope fields.
fn assert_recorded_envelope(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("chat.completion"),
        "{scenario}: the recorded body must be a chat.completion envelope"
    );
    assert!(
        body.get("created").and_then(Value::as_u64).is_some(),
        "{scenario}: the recorded body must carry `created`"
    );
    assert!(
        body.get("system_fingerprint")
            .and_then(Value::as_str)
            .is_some(),
        "{scenario}: the recorded body must carry `system_fingerprint` — without \
         it this cell cannot prove raw exposes a provider-only field"
    );
    assert!(
        body.get("usage").is_some(),
        "{scenario}: the recorded body must report usage"
    );
}

// ---------------------------------------------------------------------------
// 1: raw is the reply document, and it reads back as the provider's type
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_reads_back_as_the_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    let sink = Observed::default();
    with_llamacpp_cassette_result(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| capture_completion(client.completion(CASSETTE_MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_round_trips_provider_type should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(LLAMACPP_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);

    // `raw` is the reply document. On replay the harness serves the scrubbed
    // fixture back, so the two are the same bytes; a recording pass sees the
    // live values behind the scrubber's placeholders, and the claim it can
    // make is that the document has the same fields.
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(
            response.raw, body,
            "raw must be the reply document the server sent, verbatim"
        ),
        CassetteMode::Record => {
            let (live, recorded) = (
                response.raw.as_object().expect("raw is an object"),
                body.as_object().expect("the recorded body is an object"),
            );
            // The fixture stores canonical JSON with sorted keys, while `raw`
            // keeps the server's key order, so the shape is the key set.
            assert_eq!(
                live.keys().collect::<std::collections::BTreeSet<_>>(),
                recorded.keys().collect::<std::collections::BTreeSet<_>>(),
                "raw and the recording must be the same document shape"
            );
        }
    }

    // And the document reads back through llama.cpp's own response type,
    // which is the typed escape hatch `raw`'s documentation points at.
    let typed = llamacpp::CompletionResponse::deserialize(&response.raw)
        .expect("raw must deserialize into llamacpp::CompletionResponse");
    assert_eq!(Some(typed.openai.model.as_str()), response.model.as_deref());
    assert_eq!(response.provider, LLAMACPP_PROVIDER);
    assert!(!response.choice.is_empty());
    llamacpp::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let sink = Observed::default();
    with_llamacpp_cassette_result("raw_capture_matrix/raw_exposes_envelope_fields", |client| {
        capture_completion(client.completion(CASSETTE_MODEL), request, sink.clone())
    })
    .await
    .expect("raw_exposes_envelope_fields should replay from its cassette");
    let response = sink.take();

    let normalized = normalized_without_raw(response.clone());
    assert_normalized_lacks(&normalized, &["object", "created", "system_fingerprint"]);

    let raw = response.raw;
    let (_, body) = recorded_json_turn(LLAMACPP_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);
    for field in ["object", "system_fingerprint"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    // `created` and `id` are volatile: the cassette scrubber placeholders
    // them on the way to disk, so only a replay — which reads the scrubbed
    // bytes back — can compare them exactly. A live recording proves the
    // weaker shape claim: raw carries them with the wire's types.
    for field in ["created", "id"] {
        assert_wire_value_matches(&raw, &body, field);
    }
    let typed = llamacpp::CompletionResponse::deserialize(&raw)
        .expect("raw must deserialize into llamacpp::CompletionResponse");
    assert_eq!(Some(typed.openai.object.as_str()), body["object"].as_str());
    assert!(typed.openai.created > 0 || matches!(CassetteMode::current(), CassetteMode::Replay));
    assert_eq!(
        typed.openai.system_fingerprint.as_deref(),
        body["system_fingerprint"].as_str()
    );
}

// ---------------------------------------------------------------------------
// 3: the normalized view agrees with the provider-native one it came from
// ---------------------------------------------------------------------------

/// Every normalized field must equal the provider-native field on `raw` that
/// produced it.
///
/// This is the decoder's mapping, asserted against the document it read
/// rather than against a second mapping of the same document: there is one
/// decoder now, so `raw.normalize(..)` would have compared the mapping to a
/// copy of itself.
#[tokio::test]
async fn normalized_fields_match_the_typed_raw() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let sink = Observed::default();
    with_llamacpp_cassette_result(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| capture_completion(client.completion(CASSETTE_MODEL), request, sink.clone()),
    )
    .await
    .expect("normalized_fields_equal_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let typed = llamacpp::CompletionResponse::deserialize(&response.raw)
        .expect("raw must deserialize into llamacpp::CompletionResponse");
    let native = &typed.openai;

    assert_eq!(response.provider, LLAMACPP_PROVIDER);
    chat::assert_native_matches_normalized(&response, native, "the typed view of raw");
    // Both sides of this one come from the live document, so the id compares
    // exactly in either mode — the format contract's token-aware form is the
    // weaker claim, and there is nothing here for a scrubber to displace.
    assert_eq!(response.response_id.as_deref(), Some(native.id.as_str()));
    // llama.cpp reports no request-id response header, so the driver has
    // nothing to attach — a documented outcome rather than a gap.
    assert_no_request_id(response.provider_request_id.as_deref(), "llama.cpp");

    // And the same fields against the fixture bytes, so a recording that
    // stopped carrying them fails here rather than silently agreeing with an
    // empty document.
    let (_, body) = recorded_json_turn(LLAMACPP_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);
    assert_eq!(
        assistant_text(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("the recorded turn must carry assistant text")
    );
    assert_eq!(
        response.finish_reason(),
        Some(chat::recorded_chat_finish_reason(&body))
    );
    // The response id is a generated per-call id the scrubber placeholders on
    // disk; only a replay compares it exactly. `model` is the same situation
    // for a different reason: llama.cpp echoes the *filesystem path* of the
    // loaded GGUF, and `scrub_local_filesystem_paths` rewrites it to
    // `/REDACTED_PATH/<basename>` on the way to disk, so the live value and
    // the recorded value cannot be equal on a recording pass.
    for field in ["id", "model"] {
        assert_wire_value_matches(&response.raw, &body, field);
    }
}

// ---------------------------------------------------------------------------
// 4: `timings` — the field the shared OpenAI type has nowhere to put
// ---------------------------------------------------------------------------

/// `raw` carries llama.cpp's `timings`; reading the same bytes as the shared
/// OpenAI type loses them.
///
/// This is the whole argument for `llamacpp::CompletionResponse` existing,
/// made against real recorded bytes rather than a hand-written body. The
/// second half is deliberately a *negative* assertion about
/// `openai::CompletionResponse`: if that type ever grows a catch-all, this
/// cell fails and tells whoever is looking that the provider-local type is no
/// longer carrying its weight.
#[tokio::test]
async fn raw_preserves_the_timings_the_openai_type_drops() {
    let scenario = "raw_capture_matrix/raw_preserves_timings";
    let sink = Observed::default();
    with_llamacpp_cassette_result("raw_capture_matrix/raw_preserves_timings", |client| {
        capture_completion(client.completion(CASSETTE_MODEL), request, sink.clone())
    })
    .await
    .expect("raw_preserves_timings should replay from its cassette");
    let response = sink.take();

    let typed = llamacpp::CompletionResponse::deserialize(&response.raw)
        .expect("raw must deserialize into llamacpp::CompletionResponse");
    let timings = typed
        .timings
        .clone()
        .expect("llama.cpp reports timings on every chat completion");
    assert!(
        timings.predicted_n.is_some_and(|n| n > 0),
        "the turn generated tokens, so predicted_n must be positive: {timings:?}"
    );
    assert!(
        timings.predicted_per_second.is_some_and(|rate| rate > 0.0),
        "tokens-per-second is the accounting this field exists for: {timings:?}"
    );

    // `cache_n` and the normalized cached-token count are populated
    // independently by the server; they must agree.
    assert_eq!(
        timings.cache_n, response.usage.cached_input_tokens,
        "timings.cache_n and usage.prompt_tokens_details.cached_tokens \
         describe the same thing"
    );

    let (_, body) = recorded_json_turn(LLAMACPP_PROVIDER, scenario);
    assert_eq!(
        response.raw.get("timings"),
        body.get("timings"),
        "raw.timings must be the recorded wire value verbatim"
    );

    // The negative half: the shared OpenAI type reads these same bytes and
    // silently drops the field.
    let as_openai = openai::CompletionResponse::deserialize(&body)
        .expect("the recorded body is still a valid chat-completions response");
    let reserialized = serde_json::to_value(&as_openai).expect("the OpenAI type should serialize");
    assert!(
        body.get("timings").is_some(),
        "the fixture must carry timings for this cell to mean anything"
    );
    assert!(
        reserialized.get("timings").is_none(),
        "openai::CompletionResponse has no home for `timings`; if it grew one, \
         llamacpp::CompletionResponse may no longer be needed"
    );
}
