//! Matrix for raw terminal-record capture on mistral.rs's streaming
//! `/v1/chat/completions` route
//! ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. mistral.rs streams through the plain `OPENAI`
//! chat-completions wire pointed at its base URL, and the decoder assembles
//! a terminal record — [`StreamingCompletionResponse`] over [`ChatUsage`] —
//! from the stream's final `data:` frame plus the envelope fields the chunks
//! carried (`object`, `created`, `system_fingerprint`) accumulated under
//! `additional_params`. Every terminal record carries `raw`: that record
//! serialized. It is the terminal record only — an SSE reply is many frames
//! and no single one is the answer, so a typed round trip through `raw` is
//! exact here, unlike the unary path where `raw` is the reply document.
//! Nothing about it is sent to the server. `raw == Value::Null` means only
//! that a `StreamFinal` was built by hand without a provider terminal behind
//! it, which no cell here can produce.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | typed access | `StreamingCompletionResponse::<ChatUsage>::deserialize(&raw)` re-serializes equal | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `stream_raw_exposes_envelope_fields` | terminal-only fields | `additional_params.system_fingerprint`/`object` in `raw` equal the recorded chunks; usage equals the terminal frame | unrecorded (no mistral.rs server in this environment) |
//!
//! Every cell is unrecorded: no mistral.rs server was listening on
//! `127.0.0.1:1234` when this matrix was written, and a fixture is never
//! fabricated. To record: start `mistralrs-server` on that port serving
//! `Qwen/Qwen3-4B` (or export `MISTRALRS_BASE_URL`/`MISTRALRS_MODEL`), remove
//! the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test mistralrs mistralrs::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/mistralrs/raw_stream_capture_matrix/`.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::CassetteMode;
use crate::raw_capture::{
    assert_normalized_lacks, capture_sole_terminal, chat, stream_normalized_without_raw,
};
use crate::support::Observed;

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The plain OpenAI dialect names itself `openai`, and a terminal record is
/// attributed to the dialect that produced it.
const NORMALIZED_PROVIDER: &str = "openai";
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

/// The wire's terminal record over the wire's own accounting, which for
/// mistral.rs carries its per-second throughput counters in `ChatUsage`'s
/// flattened extras.
type MistralRsTerminal = StreamingCompletionResponse<ChatUsage>;

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose SSE stream's last JSON frame carries `usage`. Returns
/// `(all frames, terminal frame)`.
///

// ---------------------------------------------------------------------------
// 1: raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            capture_sole_terminal(client.chat(model_name()), request, sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let typed = chat::assert_terminal_round_trips(&terminal);
    // mistral.rs keeps its per-second throughput counters beside the
    // OpenAI-compatible ones, so the flattened shared counts are the halves
    // the normalized usage is made of.
    let usage = typed
        .usage
        .as_ref()
        .expect("the terminal record carries the reply's accounting");
    assert_eq!(
        Some(usage.openai.prompt_tokens as u64),
        terminal.usage.input_tokens
    );
    assert_eq!(
        usage.openai.completion_tokens.map(|tokens| tokens as u64),
        terminal.usage.output_tokens
    );
    assert_eq!(terminal.provider, NORMALIZED_PROVIDER);

    let (_, terminal_frame) = chat::recorded_frames_with_terminal(MISTRALRS_PROVIDER, scenario);
    let raw = &terminal.raw;
    assert_eq!(
        raw["usage"]["prompt_tokens"], terminal_frame["usage"]["prompt_tokens"],
        "raw usage must be the terminal frame's usage"
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        terminal_frame["usage"]["completion_tokens"]
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_raw_exposes_envelope_fields() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_envelope_fields";
    let captured = Observed::default();
    let sink = captured.clone();
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_envelope_fields",
        |client| async move {
            capture_sole_terminal(client.chat(model_name()), request, sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let normalized = stream_normalized_without_raw(&terminal);
    assert_normalized_lacks(
        &normalized,
        &[
            "system_fingerprint",
            "object",
            "created",
            "additional_params",
        ],
    );

    let raw = &terminal.raw;
    let (frames, terminal_frame) =
        chat::recorded_frames_with_terminal(MISTRALRS_PROVIDER, scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated envelope under additional_params");
    for key in ["system_fingerprint", "object"] {
        assert_eq!(
            params.get(key),
            Some(&chat::recorded_envelope_field(&frames, key, scenario)),
            "raw.additional_params.{key} must equal the recorded chunk envelope"
        );
    }
    // `created` is volatile: the scrubber placeholders it on disk, so only a
    // replay compares it exactly.
    let created = chat::recorded_envelope_field(&frames, "created", scenario);
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(params.get("created"), Some(&created)),
        CassetteMode::Record => assert!(
            params.get("created").is_some_and(Value::is_u64) && created.is_u64(),
            "raw.additional_params.created must carry the chunk envelope's integer"
        ),
    }
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    let typed = MistralRsTerminal::deserialize(raw)
        .expect("raw must deserialize into the wire's terminal record");
    let typed_params = typed
        .additional_params
        .expect("typed terminal must carry additional_params");
    assert_eq!(
        typed_params.get("system_fingerprint"),
        frames[0].get("system_fingerprint")
    );
}
