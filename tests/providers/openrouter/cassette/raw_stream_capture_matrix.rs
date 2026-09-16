//! Raw provider response capture on OpenRouter's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the record the chat decoder
//! reassembled from the reply's frames — the shared chat terminal
//! (`openai::wire::StreamingCompletionResponse`) over the shared chat usage
//! shape, serialized. Capture is always on: there is no flag to request it,
//! nothing about it reaches the wire, and a `Value::Null` only ever means a
//! terminal built by hand with no provider record behind it. It is the terminal
//! record only, never the stream's frames.
//!
//! **Why it matters here.** A gateway reports things the normalized terminal
//! has no slot for: OpenRouter's terminal usage carries the turn's `cost`, and
//! the terminal's accumulated `additional_params` carries the routed
//! `provider` the frames repeat. `ChatUsage` flattens a dialect's extra usage
//! fields, so both reach a caller through `raw` — which is the capability
//! these two cells pin.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_reads_back_as_terminal_type` | typed read-back | terminal `raw` deserializes into the shared chat terminal, and the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_cost_and_provider` | terminal-only fields | `raw.usage.cost` and `raw.additional_params.provider` equal the recorded terminal frame's | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last data
//! frame — so the raw terminal record's usage is knowable from the bytes and a
//! recording whose stream stopped reporting usage fails loudly instead of
//! covering nothing. OpenRouter contracts no request-id header, so the
//! terminal's `provider_request_id` is `None` — pinned as the documented
//! outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};
use rig::streaming::StreamFinal;
use serde::Deserialize as _;
use serde_json::{Value, json};

use super::super::DEFAULT_MODEL;
use super::super::support::{assert_matches_recorded_token, with_openrouter_cassette_result};
use crate::support::collect_text_and_terminal;

type OpenRouterTerminal = StreamingCompletionResponse<ChatUsage>;

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The single recorded frame that carries usage — the stream's last data
/// frame.
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let mut with_usage = frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| !frame["usage"].is_null());
    let (index, terminal) = with_usage
        .next()
        .expect("the recorded stream must carry usage on its terminal frame");
    assert!(
        with_usage.next().is_none(),
        "usage must be reported on exactly one (terminal) frame"
    );
    assert_eq!(
        index + 1,
        frames.len(),
        "the usage-bearing frame must be the stream's last data frame"
    );
    terminal.clone()
}

fn assert_terminal_reproduces_frame(terminal: &StreamFinal, frame: &Value) {
    assert_eq!(terminal.provider, PROVIDER, "provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        frame["id"].as_str(),
        "response id",
    );
    assert_eq!(terminal.model.as_deref(), frame["model"].as_str(), "model");
    assert_eq!(
        terminal.usage.input_tokens,
        frame["usage"]["prompt_tokens"].as_u64(),
        "input tokens"
    );
    assert_eq!(
        terminal.usage.output_tokens,
        frame["usage"]["completion_tokens"].as_u64(),
        "output tokens"
    );
    assert_eq!(
        terminal.usage.total_tokens,
        frame["usage"]["total_tokens"].as_u64(),
        "total tokens"
    );
    assert_eq!(
        terminal.provider_request_id, None,
        "OpenRouter contracts no id header"
    );
}

/// Drain one recorded stream and hand back its terminal record.
async fn observe(scenario: &'static str) -> StreamFinal {
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_openrouter_cassette_result(scenario, |client| async move {
        let model = client.completion(DEFAULT_MODEL);
        let stream = model.stream(request(&model)).await?;
        let (text, terminal) = collect_text_and_terminal(stream).await;
        let terminal = terminal.expect("stream should end with a terminal record");
        assert!(!text.is_empty());
        *sink.lock().expect("observation lock") = Some(terminal);
        Ok::<(), anyhow::Error>(())
    })
    .await
    .unwrap_or_else(|error| panic!("{scenario} should replay from its cassette: {error}"));

    observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record")
}

// ================================================================
// 1. raw reads back as the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_reads_back_as_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let terminal = observe("raw_stream_capture_matrix/stream_raw_round_trips_terminal_type").await;

    let raw = &terminal.raw;
    let typed = OpenRouterTerminal::deserialize(raw)
        .expect("raw is the shared chat terminal over the chat usage shape");
    assert_eq!(typed.response_id, terminal.response_id);
    assert_eq!(typed.finish_reason, terminal.finish_reason);
    assert_eq!(typed.provider_request_id, terminal.provider_request_id);
    assert_eq!(typed.model, terminal.model);

    let frame = recorded_terminal_frame(SCENARIO);
    assert_terminal_reproduces_frame(&terminal, &frame);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_cost_and_provider() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider";
    let terminal =
        observe("raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider").await;

    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_cost = frame["usage"]["cost"]
        .as_f64()
        .expect("OpenRouter's terminal usage reports cost");
    let recorded_provider = frame["provider"]
        .as_str()
        .expect("OpenRouter's frames name the routed provider");

    let raw = &terminal.raw;
    assert_eq!(raw["usage"]["cost"], json!(recorded_cost));
    assert_eq!(
        raw["additional_params"]["provider"],
        json!(recorded_provider)
    );
    // The normalized terminal has no slot for either: its `provider` is rig's
    // descriptor name, not the routed upstream.
    assert_eq!(terminal.provider, PROVIDER);
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("cost").is_none(),
        "the normalized usage has no cost slot: {normalized_usage}"
    );
}
