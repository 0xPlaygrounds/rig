//! Native ECS counterparts of the twelve `Surface::Agent` cells in
//! `tool_lifecycle_matrix`: the same cassette, the same one-turn budget, the
//! same real tools, and the unchanged original `assert_cell` run on native
//! observations. The original cells remain independent baselines.
//!
//! A one-turn budget with a real tool call cannot settle: `advance` refuses
//! the second model turn and the run fails with `Failure::MaxTurns`. That is
//! the expected ending here, the native spelling of the original's tolerated
//! post-tool error; every other failure kind fails the test.

use std::sync::Arc;

use anyhow::Result;
use rig::prelude::*;
use rig::providers::mistral;
use rig_core::effect::EffectKind;
use rig_ecs::agent::{AdditionalParams, DefaultMaxTurns, Failure, MaxTokens};
use rig_ecs::bus::{PendingEffect, Streamed};
use rig_ecs::systems::spawn_run;
use serde_json::json;

use super::support::with_mistral_tool_lifecycle_cassette_result;
use super::tool_lifecycle_matrix::{
    Alpha, Beta, Cell, InvocationLog, Model, Observation, PREAMBLE, Ping, RecordPayload, Shape,
    SharedObservation, Surface, Transport, assert_cell, cell, expected_names, model_name, prompt,
};
use crate::{ecs_agent::EcsAgent, ecs_observation};

/// Drive one agent cell natively and store the original `Observation` shape.
async fn run_native(
    client: mistral::Client,
    cell: Cell,
    observed: SharedObservation,
) -> Result<()> {
    let invocations = InvocationLog::default();
    let mut ecs = EcsAgent::new(client.completion_model(model_name(cell.model)), PREAMBLE, 1);
    // The original builder: default_max_turns(1), max_tokens(128) and the
    // forced tool choice with the shape's parallel policy.
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        DefaultMaxTurns(Some(1)),
        MaxTokens(Some(128)),
        AdditionalParams(Some(
            json!({ "tool_choice": "any", "parallel_tool_calls": cell.shape == Shape::Parallel }),
        )),
    ));
    match cell.shape {
        Shape::Zero => ecs.tool(Ping {
            log: Arc::clone(&invocations),
        }),
        Shape::Nested => ecs.tool(RecordPayload {
            log: Arc::clone(&invocations),
        }),
        Shape::Parallel => {
            ecs.tool(Alpha {
                log: Arc::clone(&invocations),
            });
            ecs.tool(Beta {
                log: Arc::clone(&invocations),
            });
        }
    }
    ecs_observation::install_observers(&mut ecs);

    // Streaming originals add the run override max_turns(1); blocking ones
    // rely on the builder default alone.
    let streamed = cell.transport == Transport::Streaming;
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt(cell.shape),
        streamed,
        streamed.then_some(1),
    );
    let mut errors = Vec::new();
    match ecs.wait_for_outcome(run).await {
        Ok(_) => {}
        Err(failure @ Failure::MaxTurns { limit: 1 }) => errors.push(format!("{failure:?}")),
        Err(failure) => panic!("native lifecycle run must end at its budget, not {failure:?}"),
    }
    // The original streaming collector drains to EOF and keeps item errors;
    // the success path already rejects them, the budget path must too.
    let world = ecs.app.world_mut();
    let mut streams = world.query::<&Streamed>();
    for stream in streams.iter(world) {
        errors.extend(stream.errors.iter().map(|(_, error)| error.to_string()));
    }
    let mut effects = world.query::<&PendingEffect>();
    let completions = effects
        .iter(world)
        .filter(|effect| matches!(effect.kind, EffectKind::Completion { .. }))
        .count();
    assert_eq!(
        completions, 1,
        "one recorded interaction, one completion effect"
    );

    // Native-only facts the original could not see: the model's calls were
    // materialised in wire order and each produced exactly one result.
    let expected = expected_names(cell.shape);
    let native = ecs_observation::observation(&ecs);
    assert_eq!(
        native
            .tool_calls
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        expected,
        "materialised tool calls follow the recorded call order"
    );
    assert_eq!(native.tool_results, expected.len(), "one result per call");

    let invocations = invocations.lock().expect("invocation log poisoned").clone();
    *observed.lock().expect("observation mutex poisoned") = Some(Observation {
        errors,
        invocations,
        ..Default::default()
    });
    Ok(())
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

#[tokio::test]
async fn blocking_mistral_small_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_mistral_small_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_mistral_small_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_mistral_small_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_mistral_small_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn blocking_mistral_small_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_mistral_small_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_mistral_small_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_ministral_3b_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_ministral_3b_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_ministral_3b_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_ministral_3b_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn blocking_ministral_3b_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/blocking_ministral_3b_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/blocking_ministral_3b_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_mistral_small_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_mistral_small_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_mistral_small_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_mistral_small_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_mistral_small_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_mistral_small_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_mistral_small_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_zero_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_ministral_3b_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_ministral_3b_zero_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_nested_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_ministral_3b_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_ministral_3b_nested_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}

#[tokio::test]
async fn streaming_ministral_3b_parallel_agent() -> Result<()> {
    const S: &str = "tool_lifecycle_matrix/streaming_ministral_3b_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_lifecycle_cassette_result(
        "tool_lifecycle_matrix/streaming_ministral_3b_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}
