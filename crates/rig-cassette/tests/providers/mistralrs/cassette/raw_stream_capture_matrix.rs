//! Matrix for raw document capture on mistral.rs's streaming
//! `/v1/chat/completions` route
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. mistral.rs streams through the plain `OPENAI`
//! chat-completions wire pointed at its base URL, and every stream's `raw`
//! is the `chat.completion` document its chunks rebuild: the envelope fields
//! the chunks carried (`object` renamed to the unary tag, `created`,
//! `system_fingerprint`) where a unary body states them, and the final
//! frame's usage. Nothing about it is sent to the server. `raw ==
//! Value::Null` means only that a `CompletionResponse` was built by hand
//! without a provider reply behind it, which no cell here can produce.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | typed access | `raw` is the rebuilt `chat.completion` and carries the reply's accounting | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `stream_raw_exposes_envelope_fields` | terminal-only fields | `system_fingerprint` in `raw` equals the recorded chunks' and `object` is `chat.completion`; usage equals the terminal frame | unrecorded (no mistral.rs server in this environment) |
//!
//! Every cell is unrecorded: no mistral.rs server was listening on
//! `127.0.0.1:1234` when this matrix was written, and a fixture is never
//! fabricated. To record: start `mistralrs-server` on that port serving
//! `Qwen/Qwen3-4B` (or export `MISTRALRS_BASE_URL`/`MISTRALRS_MODEL`), remove
//! the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test mistralrs mistralrs::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/mistralrs/raw_stream_capture_matrix/`.

use rig::completion::CompletionRequest;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::CassetteMode;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal, chat};
use crate::support::Observed;
use crate::support::normalized_without_raw;

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The plain OpenAI dialect names itself `openai`, and a terminal record is
/// attributed to the dialect that produced it.
const NORMALIZED_PROVIDER: &str = "openai";
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose SSE stream's last JSON frame carries `usage`. Returns
/// `(all frames, terminal frame)`.
///

// ---------------------------------------------------------------------------
// 1: raw is the rebuilt chat.completion
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
            capture_terminal(client.chat(model_name()), request(), sink)
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
    let usage = &typed["usage"];
    assert_eq!(usage["prompt_tokens"].as_u64(), terminal.usage.input_tokens);
    assert_eq!(
        usage["completion_tokens"].as_u64(),
        terminal.usage.output_tokens
    );
    assert_eq!(terminal.provider(), NORMALIZED_PROVIDER);

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
            capture_terminal(client.chat(model_name()), request(), sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let normalized = normalized_without_raw(terminal.clone());
    assert_normalized_lacks(&normalized, &["system_fingerprint", "object", "created"]);

    let raw = &terminal.raw;
    let (frames, terminal_frame) =
        chat::recorded_frames_with_terminal(MISTRALRS_PROVIDER, scenario);
    assert_eq!(
        raw.get("system_fingerprint"),
        Some(&chat::recorded_envelope_field(
            &frames,
            "system_fingerprint",
            scenario
        )),
        "raw.system_fingerprint must equal the recorded chunk envelope"
    );
    assert_eq!(raw["object"], "chat.completion");
    // `created` is volatile: the scrubber placeholders it on disk, so only a
    // replay compares it exactly.
    let created = chat::recorded_envelope_field(&frames, "created", scenario);
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(raw.get("created"), Some(&created)),
        CassetteMode::Record => assert!(
            raw.get("created").is_some_and(Value::is_u64) && created.is_u64(),
            "raw.created must carry the chunk envelope's integer"
        ),
    }
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    assert_eq!(
        raw.get("system_fingerprint"),
        frames[0].get("system_fingerprint")
    );
}
