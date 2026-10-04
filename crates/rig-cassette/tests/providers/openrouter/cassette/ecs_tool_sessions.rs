//! Native complex-tool sessions with original validators and real tools.
use super::super::support::with_openrouter_cassette_result;
use super::agent_tool_sessions::*;
use crate::ecs_agent::EcsAgent;
use anyhow::Result;
use rig_ecs::systems::RunCommands;
use serde_json::json;

#[tokio::test]
async fn nested_structured_output_schema_roundtrip() -> Result<()> {
    rig_test_support::goldens::world_golden_test(
        async {
            with_openrouter_cassette_result(
        "agent_tool_sessions/nested_structured_output_schema_roundtrip",
        |client| async move {
            let mut agent = EcsAgent::new(
                client.completion(STRUCTURED_MODEL),
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
            let run = agent.app.world_mut().spawn_run(
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "openrouter_tool_sessions_nested_structured_output_schema_roundtrip",
                log,
            )
        },
    )
    .await
}
