//! Native complex-tool sessions with original validators and real tools.
use super::agent_tool_sessions::*;
use super::support::with_xai_cassette_result;
use crate::ecs_agent::EcsAgent;
use crate::ecs_observation::install_observers;
use crate::ecs_session::run_session;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive,
};
use anyhow::Result;
use rig::tool::Tool;
use serde_json::json;
use std::sync::{Arc, Mutex};
#[tokio::test]
#[ignore = "stale cassette: the shared session cell sends no `store: false`, so xAI stores its responses and the recorder refuses the re-record"]
async fn sequential_complex_tool_calls_nonstreaming() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette_result(
                "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming",
                |client| async move {
                    let log = Arc::new(Mutex::new(Vec::new()));
                    let (ping, manifest, labels, echo) = complex_tools(&log);
                    let mut agent = {
                        let mut ecs = EcsAgent::new(
                            client.completion(SESSION_MODEL),
                            COMPLEX_SESSION_PREAMBLE,
                            10,
                        );
                        ecs.app.world_mut().entity_mut(ecs.agent).insert((
                            rig_ecs::agent::DefaultMaxTurns(Some(10)),
                            rig_ecs::agent::AdditionalParams(Some(
                                json!({ "parallel_tool_calls" : false }),
                            )),
                        ));
                        ecs.tool(ping);
                        ecs.tool(manifest);
                        ecs.tool(labels);
                        ecs.tool(echo);
                        install_observers(&mut ecs);
                        ecs
                    };
                    let response =
                        run_session(&mut agent, COMPLEX_SESSION_PROMPT, false, None).await?;
                    let history = response.history;
                    assert_contains_all_case_insensitive(
                        &response.output,
                        &["EMPTY-OK", "MANIFEST-OK", "LABELS-OK", "ESCAPE-OK"],
                    );
                    assert_complex_invocations(&log);
                    assert_history_records_sequential_tool_roundtrips(
                        &history,
                        &[
                            PingEmpty::NAME,
                            InspectManifest::NAME,
                            JoinLabels::NAME,
                            EscapeEcho::NAME,
                        ],
                    );
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "xai_tool_sessions_sequential_complex_tool_calls_nonstreaming",
                log,
            )
        },
    )
    .await
}
#[tokio::test]
#[ignore = "stale cassette: the shared session cell sends no `store: false`, so xAI stores its responses and the recorder refuses the re-record"]
async fn parallel_tool_calls_single_turn_nonstreaming() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_xai_cassette_result(
                "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
                |client| async move {
                    let mut agent = {
                        let mut ecs = EcsAgent::new(
                            client.completion(SESSION_MODEL),
                            TWO_TOOL_STREAM_PREAMBLE,
                            5,
                        );
                        ecs.app.world_mut().entity_mut(ecs.agent).insert((
                            rig_ecs::agent::DefaultMaxTurns(Some(5)),
                            rig_ecs::agent::AdditionalParams(Some(
                                json!({ "parallel_tool_calls" : true }),
                            )),
                        ));
                        ecs.tool(AlphaSignal);
                        ecs.tool(BetaSignal);
                        install_observers(&mut ecs);
                        ecs
                    };
                    let response =
                        run_session(&mut agent, TWO_TOOL_STREAM_PROMPT, false, None).await?;
                    let history = response.history;
                    assert_contains_all_case_insensitive(
                        &response.output,
                        &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
                    );
                    let calls = history_tool_calls(&history);
                    let call_names = calls
                        .iter()
                        .map(|call| call.name.as_str())
                        .collect::<Vec<_>>();
                    anyhow::ensure!(
                        calls.len() == 2
                            && call_names.contains(&AlphaSignal::NAME)
                            && call_names.contains(&BetaSignal::NAME),
                        "expected both zero-argument tools, saw {call_names:?}"
                    );
                    anyhow::ensure!(
                        calls[0].message_index == calls[1].message_index,
                        "parallel tool calls should be recorded on one assistant message"
                    );
                    anyhow::ensure!(
                        history_tool_results(&history).len() == 2,
                        "expected two tool results"
                    );
                    Ok(())
                },
            )
            .await
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "xai_tool_sessions_parallel_tool_calls_single_turn_nonstreaming",
                log,
            )
        },
    )
    .await
}
