//! Matrix for raw terminal-record capture on both Copilot streaming routes
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every stream the driver yields carries `raw`: the
//! document the route's reassembler rebuilds from the stream's frames, on
//! the chat-completions route the `chat.completion` a unary call returns.
//! Nothing about it is sent to Copilot. `raw == Value::Null` means only that
//! a `CompletionResponse` was built by hand without a provider reply behind
//! it, which no cell here can produce. Which route a stream took is a fact
//! about the wire rather than about `raw`, so each typed-access cell asserts
//! it on the bound wire itself.
//!
//! Terminal-only fields per route: on the chat route the rebuilt
//! `chat.completion` keeps every top-level chunk field where a unary body
//! states it, which is where Copilot's own `copilot_usage` block (with
//! `total_nano_aiu`) and the `system_fingerprint` land — neither has a home on the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse); on the Responses route the
//! terminal `status`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_terminal_round_trips_provider_type` | chat route, typed access | the wire is `CopilotWire::Chat`; `raw` is the `chat.completion` document the stream rebuilds | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_stream_raw_exposes_copilot_usage` | chat route, terminal-only fields | `raw.copilot_usage` equals the terminal frame's; usage equals the frame's | unrecorded (no COPILOT credentials in this environment) |
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

use rig::providers::copilot;
use rig::providers::openai::wire::OpenAiWire;
use serde_json::Value;

use crate::cassettes::CassetteMode;
use crate::cassettes::{recorded_interaction_bodies, recorded_sse_json_frames};
use crate::copilot::with_copilot_cassette_result;
use crate::raw_capture::chat;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal, responses};
use crate::support::Observed;
use crate::support::normalized_without_raw;
use rig::completion::CompletionRequest;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
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
            capture_terminal(model, request(), sink).await
        },
    )
    .await
    .expect("chat_stream_raw_terminal_round_trips_provider_type should replay from its cassette");

    let terminal = captured.take();
    // The rebuilt document carries the wire's own accounting, the
    // OpenAI-compatible counters and whatever else the dialect added, and
    // the accounting's own normalization is what the terminal must carry.
    chat::assert_terminal_round_trips(&terminal);

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
            capture_terminal(client.completion(CHAT_MODEL), request(), sink).await
        },
    )
    .await
    .expect("chat_stream_raw_exposes_copilot_usage should replay from its cassette");

    let terminal = captured.take();
    let normalized = normalized_without_raw(terminal.clone());
    assert_normalized_lacks(&normalized, &["copilot_usage", "system_fingerprint"]);

    let raw = &terminal.raw;
    let (frames, terminal_frame) = recorded_chat_frames(scenario);
    // The rebuilt document states the chunks' top-level fields where a unary
    // body does.
    assert_eq!(
        raw.get("copilot_usage"),
        terminal_frame.get("copilot_usage"),
        "raw.copilot_usage must equal the recorded terminal frame's block"
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
            assert_eq!(raw.get("system_fingerprint"), Some(&recorded_fingerprint));
        }
        CassetteMode::Record => assert!(
            raw.get("system_fingerprint").is_some_and(Value::is_string),
            "raw.system_fingerprint must carry the chunk fingerprint"
        ),
    }
    assert_eq!(
        raw.get("copilot_usage")
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
            capture_terminal(model, request(), sink).await
        },
    )
    .await
    .expect(
        "responses_stream_raw_terminal_round_trips_provider_type should replay from its cassette",
    );

    let terminal = captured.take();
    let raw = &terminal.raw;
    responses::assert_terminal_round_trips(&terminal);
    // Copilot's recorded Responses stream reports no transport id.
    assert_eq!(terminal.provider_request_id, None);

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
            capture_terminal(client.completion(RESPONSES_MODEL), request(), sink).await
        },
    )
    .await
    .expect("responses_stream_raw_exposes_terminal_status should replay from its cassette");

    let terminal = captured.take();
    let normalized = normalized_without_raw(terminal.clone());
    assert_normalized_lacks(&normalized, &["status"]);

    let raw = &terminal.raw;
    let recorded_terminal = recorded_responses_terminal(scenario);
    assert_eq!(
        recorded_terminal["status"],
        Value::String("completed".to_string())
    );
    assert_eq!(raw["status"], recorded_terminal["status"]);
    assert_eq!(raw["usage"], recorded_terminal["usage"]);
    assert_eq!(raw["status"], "completed");
}
