//! Model-turn termination metadata against llama.cpp.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>`: the normalized reason the provider stopped, and the
//! effective output-token cap that attempt ran under.
//! [`streaming_truncated_turn_reports_length_and_cap`] pins `Length` and the
//! request's cap on the streaming surface. llama.cpp's wire values `stop`,
//! `length` and `tool_calls` map directly to `Stop`, `Length` and `ToolCalls`;
//! `server-task.cpp` builds the field from exactly those three, and
//! `response_shape_matrix`'s finish-reason cell fails if a fourth appears.
//!
//! Recorded against the competent tier: `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10964-b29c606e2, so one
//! server that reliably calls tools records the whole matrix. Every cell re-reads
//! its own fixture and fails if the recorded turn stopped carrying the wire
//! reason the cell is about.

use rig::completion::FinishReason;
use serde_json::Value;

use crate::cassettes::recorded_interaction_bodies;
use crate::support::{TurnTerminationProbe, collect_stream_final_response};

use super::super::cassette_support::*;

/// Far below what the prompt below needs, so the turn is cut short with
/// partial text kept. Large enough that Qwen3 gets past its `<think>` block —
/// a turn that spends the whole cap on hidden tokens produces no answer at
/// all, which is a different subject.
const TINY_CAP: u64 = 24;
const TRUNCATING_PROMPT: &str = "/no_think Write two sentences about maple trees.";
const CONCISE_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";

/// Every recorded response's `finish_reason`, in order.
fn recorded_wire_reasons(scenario: &str) -> Vec<String> {
    recorded_interaction_bodies("llamacpp", scenario)
        .iter()
        .filter_map(|(_, response)| {
            response
                .lines()
                .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
                .filter(|line| !line.is_empty() && *line != "[DONE]")
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find_map(|json| {
                    json.get("choices")?.as_array()?.iter().find_map(|choice| {
                        choice
                            .get("finish_reason")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                })
        })
        .collect()
}

fn assert_recorded_wire_reason(scenario: &str, expected: &str) {
    let reasons = recorded_wire_reasons(scenario);
    assert!(
        reasons.contains(&expected.to_owned()),
        "cassette {scenario} no longer records a `{expected}` finish reason \
         (recorded: {reasons:?}); this cell would pass while covering nothing"
    );
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_truncated_turn_reports_length_and_cap() {
    let probe = TurnTerminationProbe::default();
    let observed = probe.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/streaming_truncated_turn",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
                .preamble(CONCISE_PREAMBLE)
                .temperature(0.0)
                .max_tokens(TINY_CAP)
                .build();

            let mut stream = agent.prompt(TRUNCATING_PROMPT).add_hook(probe).stream();
            let _ = collect_stream_final_response(&mut stream).await;
        },
    )
    .await;

    assert_eq!(
        observed.first_reason(),
        Some(FinishReason::Length),
        "the streaming surface must report the same reason as the blocking one"
    );
    assert_eq!(observed.first_max_tokens(), Some(TINY_CAP));
    assert_recorded_wire_reason("turn_termination_matrix/streaming_truncated_turn", "length");
}
