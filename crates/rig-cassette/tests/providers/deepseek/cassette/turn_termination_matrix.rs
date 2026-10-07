//! The model-turn termination metadata a hook sees (rig#2184), against the
//! real DeepSeek wire: a turn the provider cut short at the cap reaches the
//! hook as `FinishReason::Length` with the cap that attempt ran under, on
//! both surfaces. Each cell re-reads its own fixture and fails if the
//! recorded turn stopped carrying the wire reason or the request cap the
//! cell is about.
//!
//! `ContentFilter`, `Other(_)` and a missing reason have no benign live
//! trigger on this endpoint. Unit cells pin them in
//! `crates/rig-agent/src/agent/runner.rs` (`model_turn_finished_*`) and
//! `crates/rig-core/src/completion/request.rs` (`truncated_output_*`).

use rig::completion::FinishReason;
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes;
use crate::deepseek::support::with_deepseek_cassette;
use crate::support::{TurnTerminationProbe, collect_stream_final_response};

const MODEL: &str = "deepseek-chat";
const TINY_CAP: u64 = 16;
/// Truncates at `TINY_CAP`.
const TRUNCATING_PROMPT: &str = "Write two sentences about maple trees.";
const CONCISE_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: turn_termination_matrix_case;
    # [tokio :: test]
    blocking_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap", blocking_truncated_turn_reports_length_and_cap_15);
    # [tokio :: test]
    streaming_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap", streaming_truncated_turn_reports_length_and_cap_16);
}

// ---------------------------------------------------------------------------
// Fixture-premise checks: the recorded bytes must still say what the cell
// claims, or the cell passes while covering nothing.
// ---------------------------------------------------------------------------

fn recorded_interactions(scenario: &str) -> Vec<serde_yaml::Value> {
    {
        let path = cassettes::cassette_path("deepseek", scenario);
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            {
                panic!(
                    "provider cassette {} should be readable after recording: {error}",
                    path.display()
                )
            }
        });
        serde_yaml::Deserializer::from_str(&contents)
            .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
            .collect()
    }
}

fn interaction_bodies(scenario: &str, side: &str) -> Vec<String> {
    {
        recorded_interactions(scenario)
            .iter()
            .filter_map(|interaction| {
                {
                    interaction
                        .get(side)
                        .and_then(|side| side.get("body"))
                        .and_then(serde_yaml::Value::as_str)
                        .map(ToOwned::to_owned)
                }
            })
            .collect()
    }
}

/// Every JSON object in a recorded body. A blocking response is one object; a
/// streamed one is a sequence of `data:` frames, so both are handled by
/// scanning line by line and keeping whatever parses.
fn body_json_objects(body: &str) -> Vec<Value> {
    {
        body.lines()
            .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
            .filter(|line| !line.is_empty() && *line != "[DONE]")
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect()
    }
}

/// One wire finish reason per recorded interaction, in order — the first
/// non-null one the body carries. A stream repeats `null` on every chunk until
/// the terminal one, so taking the first non-null entry yields exactly one
/// reason per call on both surfaces.
fn recorded_wire_reasons(scenario: &str) -> Vec<String> {
    {
        interaction_bodies(scenario, "then")
            .iter()
            .filter_map(|body| {
                {
                    body_json_objects(body).iter().find_map(|json| {
                        json.get("choices")?.as_array()?.iter().find_map(|choice| {
                            choice
                                .get("finish_reason")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                    })
                }
            })
            .collect()
    }
}

fn assert_recorded_wire_reason(scenario: &str, expected: &str) {
    {
        let reasons = recorded_wire_reasons(scenario);
        assert!(
            reasons.contains(&expected.to_owned()),
            "cassette {scenario} no longer records a `{expected}` finish reason (recorded: \
         {reasons:?}); this cell would pass while covering nothing"
        );
    }
}

/// The output-token cap of every recorded *request*, in order — proof that the
/// cap the hook reported is the cap that actually went on the wire.
fn recorded_request_caps(scenario: &str) -> Vec<u64> {
    {
        interaction_bodies(scenario, "when")
            .iter()
            .filter_map(|body| serde_json::from_str::<Value>(body).ok())
            .filter_map(|body| {
                body.get("max_tokens")
                    .or_else(|| body.get("max_completion_tokens"))
                    .and_then(Value::as_u64)
            })
            .collect()
    }
}

fn assert_recorded_request_cap(scenario: &str, expected: u64) {
    {
        let caps = recorded_request_caps(scenario);
        assert!(
            caps.contains(&expected),
            "cassette {scenario} no longer records a request capped at {expected} (recorded: {caps:?})"
        );
    }
}
