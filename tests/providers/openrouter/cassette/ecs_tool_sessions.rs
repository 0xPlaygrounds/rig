//! Native complex-tool sessions with original validators and real tools.
use super::super::support::with_openrouter_cassette_result;
use super::agent_tool_sessions::*;
use crate::ecs_agent::EcsAgent;
use crate::ecs_observation::{install_observers, observation};
use crate::ecs_session::run_session;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive,
    assert_two_tool_roundtrip_contract,
};
use anyhow::Result;
use rig::prelude::*;
use rig::tool::Tool;
use serde_json::json;
use std::sync::{Arc, Mutex};
#[tokio::test]
async fn sequential_complex_tool_calls_nonstreaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, echo) = complex_tools(&log);
            let mut agent = {
                let mut ecs = EcsAgent::new(
                    client.completion_model(SESSION_MODEL),
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
            let response = run_session(&mut agent, COMPLEX_SESSION_PROMPT, false, None).await?;
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
}
#[tokio::test]
async fn sequential_complex_tool_calls_streaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_streaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, echo) = complex_tools(&log);
            let mut agent = {
                let mut ecs = EcsAgent::new(
                    client.completion_model(SESSION_MODEL),
                    COMPLEX_SESSION_PREAMBLE,
                    1,
                );
                ecs.app.world_mut().entity_mut(ecs.agent).insert((
                    rig_ecs::agent::DefaultMaxTurns(None),
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
            run_session(&mut agent, COMPLEX_SESSION_PROMPT, true, Some(10)).await?;
            let observation = observation(&agent);
            anyhow::ensure!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            let expected_tool_calls = vec![
                PingEmpty::NAME.to_string(),
                InspectManifest::NAME.to_string(),
                JoinLabels::NAME.to_string(),
                EscapeEcho::NAME.to_string(),
            ];
            anyhow::ensure!(
                observation.tool_calls == expected_tool_calls,
                "stream should expose the same ordered tool calls as non-streaming; saw {:?}",
                observation.tool_calls
            );
            anyhow::ensure!(
                observation.tool_results == 4,
                "expected 4 streamed tool results, saw {}",
                observation.tool_results
            );
            anyhow::ensure!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            let response = observation
                .final_response_text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("stream should produce final response text"))?;
            assert_contains_all_case_insensitive(
                response,
                &["EMPTY-OK", "MANIFEST-OK", "LABELS-OK", "ESCAPE-OK"],
            );
            assert_complex_invocations(&log);
            Ok(())
        },
    )
    .await
}
#[tokio::test]
async fn parallel_tool_calls_single_turn_nonstreaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let mut agent = {
                let mut ecs = EcsAgent::new(
                    client.completion_model(SESSION_MODEL),
                    TWO_TOOL_STREAM_PREAMBLE,
                    5,
                );
                ecs.app.world_mut().entity_mut(ecs.agent).insert((
                    rig_ecs::agent::DefaultMaxTurns(Some(5)),
                    rig_ecs::agent::AdditionalParams(None),
                ));
                ecs.tool(AlphaSignal);
                ecs.tool(BetaSignal);
                install_observers(&mut ecs);
                ecs
            };
            let response = run_session(&mut agent, TWO_TOOL_STREAM_PROMPT, false, None).await?;
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
                "expected both zero-argument tools in one model turn, saw {call_names:?}"
            );
            anyhow::ensure!(
                calls[0].message_index == calls[1].message_index,
                "parallel tool calls should be recorded on one assistant message"
            );
            let result_count = history_tool_results(&history).len();
            anyhow::ensure!(
                result_count == 2,
                "expected two tool results, saw {result_count}"
            );
            Ok(())
        },
    )
    .await
}
#[tokio::test]
async fn parallel_tool_calls_single_turn_streaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let mut agent = {
                let mut ecs = EcsAgent::new(
                    client.completion_model(SESSION_MODEL),
                    TWO_TOOL_STREAM_PREAMBLE,
                    1,
                );
                ecs.app.world_mut().entity_mut(ecs.agent).insert((
                    rig_ecs::agent::DefaultMaxTurns(None),
                    rig_ecs::agent::AdditionalParams(None),
                ));
                ecs.tool(AlphaSignal);
                ecs.tool(BetaSignal);
                install_observers(&mut ecs);
                ecs
            };
            run_session(&mut agent, TWO_TOOL_STREAM_PROMPT, true, Some(5)).await?;
            let observation = observation(&agent);
            assert_two_tool_roundtrip_contract(
                observation,
                &[AlphaSignal::NAME, BetaSignal::NAME],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn nested_structured_output_schema_roundtrip() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/nested_structured_output_schema_roundtrip",
        |client| async move {
            let mut agent = EcsAgent::new(
                client.completion_model(STRUCTURED_MODEL),
                "Return only data that satisfies the requested schema. Use lane canary, risk low, \
                 and checks compile=true and replay=true.",
                1,
            );
            agent.app.world_mut().entity_mut(agent.agent).insert((
                rig_ecs::agent::DefaultMaxTurns(None),
                rig_ecs::agent::AdditionalParams(Some(json!({
                    "provider": {
                        "require_parameters": true,
                        "order": ["Google AI Studio", "Google Vertex"]
                    }
                }))),
            ));
            let run = rig_ecs::systems::spawn_run(
                agent.app.world_mut(),
                agent.agent,
                &[],
                "Create the OpenRouter cassette release validation plan.",
                false,
                None,
            );
            agent
                .app
                .world_mut()
                .entity_mut(run)
                .insert(rig_ecs::agent::Output {
                    mode: rig_ecs::agent::OutputKind::Native,
                    schema: Some(schemars::schema_for!(NestedPlan).into()),
                });
            let output = agent
                .wait_for_outcome(run)
                .await
                .map_err(|error| anyhow::anyhow!("native typed run failed: {error:?}"))?;
            let plan: NestedPlan = crate::ecs_session::parse_native_output(&output)?;
            anyhow::ensure!(plan.release.lane.eq_ignore_ascii_case("canary"));
            anyhow::ensure!(plan.release.risk.eq_ignore_ascii_case("low"));
            anyhow::ensure!(
                plan.checks
                    .iter()
                    .any(|check| check.name.eq_ignore_ascii_case("compile") && check.required),
                "structured output should include the compile check"
            );
            anyhow::ensure!(
                plan.checks
                    .iter()
                    .any(|check| check.name.eq_ignore_ascii_case("replay") && check.required),
                "structured output should include the replay check"
            );
            Ok(())
        },
    )
    .await
}
