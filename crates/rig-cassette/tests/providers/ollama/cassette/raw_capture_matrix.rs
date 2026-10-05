//! Raw response capture on Ollama's native `/api/chat` route.
//!
//! The driver sets `raw` from the reply's bytes, so a whole reply's `raw` is
//! the `/api/chat` body itself: `done`, `done_reason` and the daemon's
//! timings ride on it, and only the counters reach the normalized usage.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `raw_exposes_envelope_fields` | provider-only fields | `done`, `total_duration`, `eval_duration` and `created_at` are on `raw` and absent from the normalized response | recorded |

use rig::completion::{CompletionRequest, FinishReason};
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_ollama_cassette};
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{Observed, assert_wire_value_matches, normalized_without_raw};

const PROVIDER: &str = "ollama";

fn request() -> CompletionRequest {
    CompletionRequest::new("Reply with exactly this one word and nothing else: captured")
        .temperature(0.0)
        .max_tokens(1024)
}

#[tokio::test]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let sink = Observed::default();
    let parked = sink.clone();
    with_ollama_cassette(
        "raw_capture_matrix/raw_exposes_envelope_fields",
        |client| async move {
            capture_completion(client.completion(CASSETTE_MODEL), request(), parked)
                .await
                .expect("the recorded turn replays");
        },
    )
    .await;
    let response = sink.take();
    let (_, body) = recorded_json_turn(PROVIDER, scenario);
    assert_eq!(body["done"], true, "{scenario}: the premise");

    assert_normalized_lacks(
        &normalized_without_raw(response.clone()),
        &["done", "total_duration", "eval_duration", "created_at"],
    );
    for field in ["done", "done_reason", "total_duration", "eval_duration"] {
        assert_eq!(response.raw.get(field), body.get(field), "{field}");
    }
    assert_wire_value_matches(&response.raw, &body, "created_at");
    assert!(
        response.raw["eval_duration"].is_u64(),
        "{}",
        Value::clone(&response.raw)
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    let count = |key: &str| body[key].as_u64();
    assert_eq!(response.usage.input_tokens, count("prompt_eval_count"));
    assert_eq!(response.usage.output_tokens, count("eval_count"));
}
