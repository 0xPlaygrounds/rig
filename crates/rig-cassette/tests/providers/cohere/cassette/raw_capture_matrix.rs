//! Raw response capture on Cohere's Chat Completions route.
//!
//! Cohere is the `COHERE` dialect of the shared Chat wire, and the driver
//! sets `raw` from the reply's bytes, so `raw` is the reply document itself.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `raw_exposes_envelope_fields` | provider-only fields | `object` and `created` are on `raw` and absent from the normalized response | recorded |

use rig::completion::CompletionRequest;
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{Observed, assert_wire_value_matches, normalized_without_raw};

const PROVIDER: &str = "cohere";

fn request() -> CompletionRequest {
    CompletionRequest::new("Reply with exactly this one word and nothing else: captured")
        .temperature(0.0)
        .max_tokens(16)
}

#[tokio::test]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let sink = Observed::default();
    let parked = sink.clone();
    with_cohere_cassette(
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
    assert_eq!(body["object"], "chat.completion", "{scenario}: the premise");

    assert_normalized_lacks(
        &normalized_without_raw(response.clone()),
        &["object", "created"],
    );
    assert_eq!(response.raw.get("object"), body.get("object"));
    assert_wire_value_matches(&response.raw, &body, "created");
    assert!(
        response.raw["created"].is_u64(),
        "{}",
        Value::clone(&response.raw)
    );
}
