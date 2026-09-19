//! Matrix for raw terminal-record capture on both Copilot streaming routes
//! ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. The terminal record of every stream the driver
//! yields carries `raw`: the route's own terminal record, serialized by the
//! decoder that built it from the stream's frames. On the chat-completions
//! route that is the shared chat terminal type, on the Responses route the
//! shared Responses one, and each cell reads `raw` back through the type its
//! route owns. It is the terminal record only, and nothing about it is sent
//! to Copilot. `raw == Value::Null` means only that a `StreamFinal` was built
//! by hand without a provider terminal behind it, which no cell here can
//! produce. Which route a stream took is a fact about the wire rather than
//! about `raw`, so each typed-access cell asserts it on the bound wire
//! itself.
//!
//! A stream has no single reply body, so this `raw` is unlike its blocking
//! twin: it is `serde_json::to_value` of the terminal record the decoder
//! assembled (`openai::wire::chat`'s `emit_terminal`, and the Responses
//! decoder's `terminal_record`), not a document read off the socket. That is
//! why a cell here may assert the record re-serializes *equal* to `raw`,
//! which on the blocking path would be false: nothing the stream carried and
//! the terminal type does not model is silently dropped, because the shared
//! terminal type accumulates it under `additional_params`.
//!
//! Terminal-only fields per route: on the chat route the shared terminal type
//! accumulates unknown top-level chunk fields under `additional_params`,
//! which is where Copilot's own `copilot_usage` block (with `total_nano_aiu`)
//! and the `system_fingerprint` land — neither has a home on the normalized
//! [`StreamFinal`](rig::streaming::StreamFinal); on the Responses route the
//! terminal `status`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_terminal_round_trips_provider_type` | chat route, typed access | the wire is `CopilotWire::Chat`; `raw` reads back as the chat terminal record and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_stream_raw_exposes_copilot_usage` | chat route, terminal-only fields | `raw.additional_params.copilot_usage` equals the terminal frame's; usage equals the frame's | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `responses_stream_raw_terminal_round_trips_provider_type` | responses route, typed access | the wire is `CopilotWire::Responses`; `raw` reads back as the Responses terminal record and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `responses_stream_raw_exposes_terminal_status` | responses route, terminal-only field | `raw.status == "completed"` as the recorded `response.completed` frame says | unrecorded (no COPILOT credentials in this environment) |
//!
//! Every cell is unrecorded: none of `GITHUB_COPILOT_API_KEY`,
//! `COPILOT_API_KEY`, `COPILOT_GITHUB_ACCESS_TOKEN`/`GITHUB_TOKEN` nor a Copilot
//! OAuth cache was present when this matrix was written, and a fixture is
//! never fabricated. To record: export `GITHUB_COPILOT_API_KEY`, remove the
//! `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test copilot copilot::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/copilot/raw_stream_capture_matrix/`.

use rig::completion::CompletionModel as _;
use rig::driver::Bound;
use rig::providers::copilot;
use rig::providers::copilot::wire::CopilotWire;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use rig::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};
use crate::copilot::with_copilot_cassette_result;
use crate::raw_capture::{
    assert_normalized_lacks, capture_sole_terminal, chat, responses, stream_normalized_without_raw,
};
use crate::support::Observed;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &Bound<CopilotWire>) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// Every cell records exactly one interaction; a scenario with more has
/// drifted from the matrix's premise.
fn assert_single_interaction(scenario: &str) {
    assert_eq!(
        recorded_interaction_bodies(COPILOT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
}

/// Chat-route premise: the recorded SSE stream's last frame carries `usage`
/// (and Copilot's `copilot_usage`). Returns `(all frames, terminal frame)`.
///
/// Neither shared premise reader fits: `chat::recorded_sole_usage_frame`
/// requires the usage frame to be the stream's last data frame and
/// `chat::recorded_agreeing_usage_frames` requires every usage-bearing frame
/// to agree, while the rule here is "the last usage-bearing frame, which must
/// also carry `copilot_usage`".
fn recorded_chat_frames(scenario: &str) -> (Vec<Value>, Value) {
    assert_single_interaction(scenario);
    let frames = recorded_sse_json_frames(COPILOT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("usage").is_some_and(Value::is_object))
        .cloned()
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must carry a usage-bearing frame")
        });
    assert!(
        terminal.get("copilot_usage").is_some_and(Value::is_object),
        "{scenario}: the usage frame must carry Copilot's `copilot_usage` block — \
         without it this cell cannot prove raw exposes a provider-only field"
    );
    (frames, terminal)
}

/// Responses-route premise: the recorded SSE stream ends with a
/// `response.completed` frame carrying usage. Returns its `response`.
fn recorded_responses_terminal(scenario: &str) -> Value {
    assert_single_interaction(scenario);
    let frames = recorded_sse_json_frames(COPILOT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must end with response.completed")
        });
    let response = terminal["response"].clone();
    assert!(
        response.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the terminal frame must report usage"
    );
    response
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/chat_stream_raw_terminal_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_stream_capture_matrix/chat_stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            assert!(
                matches!(model.wire.wire, OpenAiWire::Chat(_)),
                "premise: gpt-4o routes through chat completions"
            );
            capture_sole_terminal(model, request, sink).await
        },
    )
    .await
    .expect("chat_stream_raw_terminal_round_trips_provider_type should replay from its cassette");

    let terminal = captured.take();
    // The record carries the wire's own accounting, which flattens both the
    // OpenAI-compatible counters and whatever else the dialect added — so the
    // round trip is exact, and the accounting's own normalization is what the
    // terminal must carry. The transport id is the header's, so the native
    // record has no slot filled for it.
    let typed = chat::assert_terminal_round_trips(&terminal);
    // Copilot's recorded chat stream reports no transport id, so the native
    // record and the normalized terminal agree on its absence — the claim
    // this cell made before the round trip became shared.
    assert_eq!(typed.provider_request_id, terminal.provider_request_id);

    let (_, terminal_frame) = recorded_chat_frames(scenario);
    assert_eq!(
        terminal.raw["usage"]["prompt_tokens"], terminal_frame["usage"]["prompt_tokens"],
        "raw usage must be the terminal frame's usage"
    );
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_raw_exposes_copilot_usage() {
    let scenario = "raw_stream_capture_matrix/chat_stream_raw_exposes_copilot_usage";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_stream_capture_matrix/chat_stream_raw_exposes_copilot_usage",
        |client| async move {
            capture_sole_terminal(client.completion(CHAT_MODEL), request, sink).await
        },
    )
    .await
    .expect("chat_stream_raw_exposes_copilot_usage should replay from its cassette");

    let terminal = captured.take();
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(
        &normalized,
        &["copilot_usage", "system_fingerprint", "additional_params"],
    );

    let raw = &terminal.raw;
    let (frames, terminal_frame) = recorded_chat_frames(scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated chunk envelope under additional_params");
    assert_eq!(
        params.get("copilot_usage"),
        terminal_frame.get("copilot_usage"),
        "raw.additional_params.copilot_usage must equal the recorded terminal frame's block"
    );
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    // `fp_…` fingerprints are placeholdered on disk; only a replay compares
    // them exactly, a live recording checks the field is there.
    let recorded_fingerprint = frames
        .iter()
        .find_map(|frame| frame.get("system_fingerprint"))
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: recorded chunks must carry system_fingerprint"));
    match CassetteMode::current() {
        CassetteMode::Replay => {
            assert_eq!(
                params.get("system_fingerprint"),
                Some(&recorded_fingerprint)
            );
        }
        CassetteMode::Record => assert!(
            params
                .get("system_fingerprint")
                .is_some_and(Value::is_string),
            "raw.additional_params.system_fingerprint must carry the chunk fingerprint"
        ),
    }
    let typed = StreamingCompletionResponse::<ChatUsage>::deserialize(raw)
        .expect("chat-route raw must read back as the chat terminal record");
    let typed_params = typed
        .additional_params
        .expect("typed terminal must carry additional_params");
    assert_eq!(
        typed_params
            .get("copilot_usage")
            .and_then(|usage| usage.get("total_nano_aiu")),
        terminal_frame.pointer("/copilot_usage/total_nano_aiu")
    );
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_raw_terminal_round_trips_provider_type() {
    let scenario =
        "raw_stream_capture_matrix/responses_stream_raw_terminal_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_stream_capture_matrix/responses_stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            assert!(
                matches!(model.wire.wire, OpenAiWire::Responses(_)),
                "premise: the codex model routes through the Responses API"
            );
            capture_sole_terminal(model, request, sink).await
        },
    )
    .await
    .expect(
        "responses_stream_raw_terminal_round_trips_provider_type should replay from its cassette",
    );

    let terminal = captured.take();
    let raw = &terminal.raw;
    let typed = responses::assert_terminal_round_trips(&terminal);
    // Copilot's recorded Responses stream reports no transport id either, so
    // the native record and the normalized terminal agree on its absence —
    // the claim this cell made before the round trip became shared.
    assert_eq!(typed.provider_request_id, terminal.provider_request_id);

    let recorded_terminal = recorded_responses_terminal(scenario);
    assert_eq!(
        raw["usage"]["total_tokens"],
        recorded_terminal["usage"]["total_tokens"]
    );
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_raw_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/responses_stream_raw_exposes_terminal_status";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_stream_capture_matrix/responses_stream_raw_exposes_terminal_status",
        |client| async move {
            capture_sole_terminal(client.completion(RESPONSES_MODEL), request, sink).await
        },
    )
    .await
    .expect("responses_stream_raw_exposes_terminal_status should replay from its cassette");

    let terminal = captured.take();
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(&normalized, &["status"]);

    let raw = &terminal.raw;
    let recorded_terminal = recorded_responses_terminal(scenario);
    assert_eq!(
        recorded_terminal["status"],
        Value::String("completed".to_string())
    );
    assert_eq!(raw["status"], recorded_terminal["status"]);
    assert_eq!(raw["usage"], recorded_terminal["usage"]);
    let typed: responses_api::streaming::StreamingCompletionResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(typed.status, Some(responses_api::ResponseStatus::Completed));
}
