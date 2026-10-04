//! The root package's own tests: guards that scan the source tree and the
//! fixture runners, which need the repository root. Behaviour of the bus and
//! the agent over it is verified in `crates/rig-cassette`; provider behaviour
//! in `tests/providers`; anything needing crate-private types stays a unit
//! test in its crate.

mod agent_run_stepper;
mod dependency_graph;
#[cfg(feature = "derive")]
mod embed_macro;
mod fixtures_hold_no_key;
mod golden_causal;
mod golden_delta;
mod golden_endings;
mod golden_hooks;
mod golden_invalid;
mod golden_layers;
mod golden_leftovers;
mod golden_memory;
mod golden_oracle;
mod golden_outcome;
mod golden_output;
mod golden_pairing;
mod golden_recovery;
mod history_conformance;
mod history_conformance_registry;
mod loaders;
mod no_random_ids;
mod prompt_response_messages;
mod reasoning_stream_stats;
#[cfg(feature = "derive")]
mod rig_tool_facade;
mod streaming_conformance;
mod streaming_conformance_registry;
mod streaming_conformance_suites;
#[allow(dead_code)]
#[path = "../../xtask/src/verify/checks.rs"]
mod verification_checks;

/// The text of every tool result in the history of the completion recorded
/// at `at`.
fn tool_result_texts(log: &rig_cassette::effect_log::EffectLog, at: usize) -> Vec<String> {
    match &log.records[at].kind {
        rig::effect::EffectKind::Completion { request, .. } => request
            .chat_history
            .iter()
            .filter_map(|message| match message {
                rig::message::Message::User { content } => Some(content.iter()),
                _ => None,
            })
            .flatten()
            .filter_map(|content| match content {
                rig::message::UserContent::ToolResult(result) => Some(
                    result
                        .content
                        .iter()
                        .map(|part| match part {
                            rig::message::ToolResultContent::Text(text) => text.text.clone(),
                            rig::message::ToolResultContent::Json { value } => value.to_string(),
                            other => format!("{other:?}"),
                        })
                        .collect::<String>(),
                ),
                _ => None,
            })
            .collect(),
        other => panic!("a completion, not {other:?}"),
    }
}

/// `events` closed by a final response with default usage.
fn stream_turn(
    mut events: Vec<rig::test_utils::MockStreamEvent>,
) -> Vec<rig::test_utils::MockStreamEvent> {
    events.push(rig::test_utils::MockStreamEvent::final_response_with_default_usage());
    events
}

/// The hook's reason a run was cancelled with.
fn cancelled_reason(error: &rig::completion::PromptError) -> &str {
    match error {
        rig::completion::PromptError::Cancelled { reason, .. } => reason,
        other => panic!("a cancelled run, not {other:?}"),
    }
}
