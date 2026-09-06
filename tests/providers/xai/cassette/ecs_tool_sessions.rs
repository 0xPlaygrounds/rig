//! Native complex-tool sessions with original validators and real tools.
use super::agent_tool_sessions::*;
use super::support::with_xai_cassette_result;
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
    with_xai_cassette_result(
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
    with_xai_cassette_result(
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
            anyhow::ensure!(
                observation.tool_calls
                    == vec![
                        PingEmpty::NAME.to_string(),
                        InspectManifest::NAME.to_string(),
                        JoinLabels::NAME.to_string(),
                        EscapeEcho::NAME.to_string(),
                    ],
                "stream should expose ordered tool calls, saw {:?}",
                observation.tool_calls
            );
            anyhow::ensure!(
                observation.tool_results == 4,
                "expected 4 streamed tool results, saw {}",
                observation.tool_results
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
    with_xai_cassette_result(
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
                    rig_ecs::agent::AdditionalParams(Some(json!({ "parallel_tool_calls" : true }))),
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
}
#[tokio::test]
async fn parallel_tool_calls_single_turn_streaming() -> Result<()> {
    with_xai_cassette_result(
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
                    rig_ecs::agent::AdditionalParams(Some(json!({ "parallel_tool_calls" : true }))),
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
async fn multimodal_image_input_mixed_text_ordering() -> Result<()> {
    use bevy_ecs::prelude::*;
    use rig::message::UserContent;
    use rig_ecs::agent::{DefaultMaxTurns, MessageParts, Parts, Utterance};
    with_xai_cassette_result(
        "agent_tool_sessions/multimodal_image_input_mixed_text_ordering",
        |client| async move {
            let mut agent = EcsAgent::new(
                client.completion_model(VISION_MODEL),
                "You answer image questions concisely and directly.",
                1,
            );
            agent
                .app
                .world_mut()
                .entity_mut(agent.agent)
                .insert(DefaultMaxTurns(None));
            let run = rig_ecs::systems::spawn_run(
                agent.app.world_mut(),
                agent.agent,
                &[],
                "",
                false,
                None,
            );
            // Before scheduling, replace the fresh user utterance with its
            // actual multimodal parts. This is the native graph input surface.
            let (parent, mut parts) = agent
                .app
                .world_mut()
                .query_filtered::<(&ChildOf, &mut Parts), With<Utterance>>()
                .single_mut(agent.app.world_mut())
                .map_err(|error| anyhow::anyhow!("expected one fresh prompt: {error}"))?;
            anyhow::ensure!(parent.parent() == run);
            parts.0 = MessageParts::User {
                content: vec![
                    UserContent::text("First, note this is an image-analysis cassette test."),
                    image_content(),
                    UserContent::text(
                        "Then answer in one short sentence naming the main visible subject.",
                    ),
                ],
            };
            let output = agent
                .wait_for_outcome(run)
                .await
                .map_err(|error| anyhow::anyhow!("native vision run failed: {error:?}"))?;
            crate::support::assert_nonempty_response(&output);
            Ok(())
        },
    )
    .await
}
