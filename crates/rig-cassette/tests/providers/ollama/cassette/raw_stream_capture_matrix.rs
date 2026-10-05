//! Raw response capture on Ollama's streamed native `/api/chat` route.
//!
//! The stream is NDJSON: every line is a chat record, and the last one says
//! `done`. A streamed reply's `raw` is that record as the daemon sent it, and
//! its usage is the record's counters.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_terminal_reproduces_the_usage_chunk` | terminal record | `raw` is the recorded `done` record, and usage and the model are its own | recorded |

use rig::completion::CompletionRequest;
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_ollama_cassette};
use crate::cassettes::recorded_interaction_bodies;
use crate::raw_capture::{assert_no_request_id, capture_text_and_terminal};
use crate::support::{Observed, assert_wire_value_matches};

const PROVIDER: &str = "ollama";

/// The recorded stream's records, one per NDJSON line.
fn recorded_records(scenario: &str) -> Vec<Value> {
    let interactions = recorded_interaction_bodies(PROVIDER, scenario);
    let [(_, body)] = interactions.as_slice() else {
        panic!("{scenario}: one interaction was recorded");
    };
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("each line is a record"))
        .collect()
}

#[tokio::test]
async fn stream_terminal_reproduces_the_usage_chunk() {
    let scenario = "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk";
    let sink = Observed::default();
    let parked = sink.clone();
    with_ollama_cassette(
        "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk",
        |client| async move {
            capture_text_and_terminal(
                client.completion(CASSETTE_MODEL),
                CompletionRequest::new(
                    "Reply with exactly this one word and nothing else: streamed",
                )
                .temperature(0.0)
                .max_tokens(1024),
                parked,
            )
            .await
            .expect("the recorded stream replays");
        },
    )
    .await;
    let (text, terminal) = sink.take();
    assert!(!text.trim().is_empty(), "the stream carried text");
    let records = recorded_records(scenario);
    assert!(records.len() > 1, "{scenario}: the premise, a stream");
    let done = records.last().expect("a record");
    assert_eq!(done["done"], true, "{scenario}: the stream ends on `done`");
    for field in [
        "done",
        "done_reason",
        "model",
        "prompt_eval_count",
        "eval_count",
    ] {
        assert_eq!(terminal.raw.get(field), done.get(field), "{field}");
    }
    assert_wire_value_matches(&terminal.raw, done, "created_at");
    assert_eq!(
        terminal.usage.input_tokens,
        done["prompt_eval_count"].as_u64()
    );
    assert_eq!(terminal.usage.output_tokens, done["eval_count"].as_u64());
    assert_eq!(terminal.model(), done["model"].as_str());
    assert_no_request_id(terminal.provider_request_id.as_deref(), PROVIDER);
}
