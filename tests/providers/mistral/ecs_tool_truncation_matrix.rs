//! Native ECS counterparts of the twelve `Surface::Agent` cells in
//! `tool_truncation_matrix`: the same cassette, budget cap, real
//! `FileReport` tool and the unchanged original `assert_cell` on native
//! observations. The original cells remain independent baselines.
//!
//! Complete cells carry a real call under a one-turn budget, so the run ends
//! in `Failure::MaxTurns` after the tool ran: the expected ending, as in the
//! lifecycle counterparts. Truncated cells carry no complete call, so the run
//! must settle; native success strengthens the original tolerated-error
//! check, as the DeepSeek truncation counterparts already require.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use rig::prelude::*;
use rig::providers::mistral;
use rig_core::effect::EffectKind;
use rig_ecs::agent::{AdditionalParams, DefaultMaxTurns, Failure, MaxTokens};
use rig_ecs::bus::{PendingEffect, Streamed};
use rig_ecs::systems::spawn_run;
use serde_json::json;

use super::support::with_mistral_tool_truncation_cassette_result;
use super::tool_truncation_matrix::{
    Budget, Cell, FileReport, Model, Observation, PREAMBLE, PROMPT, SharedObservation, Surface,
    Transport, assert_cell, cell, max_tokens, model_name,
};
use crate::{ecs_agent::EcsAgent, ecs_observation};

/// Drive one agent cell natively and store the original `Observation` shape.
async fn run_native(
    client: mistral::Client,
    cell: Cell,
    observed: SharedObservation,
) -> Result<()> {
    let invocations = Arc::new(AtomicUsize::new(0));
    let mut ecs = EcsAgent::new(client.completion_model(model_name(cell.model)), PREAMBLE, 1);
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        DefaultMaxTurns(Some(1)),
        MaxTokens(Some(max_tokens(cell.budget))),
        AdditionalParams(Some(json!({ "tool_choice": "any" }))),
    ));
    ecs.tool(FileReport {
        invocations: Arc::clone(&invocations),
    });
    ecs_observation::install_observers(&mut ecs);

    let streamed = cell.transport == Transport::Streaming;
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        PROMPT,
        streamed,
        streamed.then_some(1),
    );
    let complete = cell.budget == Budget::Complete;
    let mut errors = Vec::new();
    match ecs.wait_for_outcome(run).await {
        Ok(_) => {}
        Err(failure @ Failure::MaxTurns { limit: 1 }) if complete => {
            errors.push(format!("{failure:?}"));
        }
        Err(failure) => panic!("native truncation run must not fail with {failure:?}"),
    }
    // The success path already rejects stream item errors; the budget path
    // of the complete cells must fold them too, as the original drained to
    // EOF and kept every item error.
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

    let native = ecs_observation::observation(&ecs);
    if complete {
        assert_eq!(native.tool_calls, vec![FileReport::NAME.to_owned()]);
        assert_eq!(
            native.tool_results, 1,
            "the complete call produced one result"
        );
    } else {
        assert!(
            native.tool_calls.is_empty(),
            "a truncated call must never materialise: {:?}",
            native.tool_calls
        );
    }

    *observed.lock().expect("observation mutex poisoned") = Some(Observation {
        errors,
        invocations: invocations.load(Ordering::SeqCst),
        ..Default::default()
    });
    Ok(())
}

// The literal wrapper calls below are intentionally explicit: cassette safety
// parses source rather than macro expansion.

#[tokio::test]
async fn blocking_mistral_small_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_mistral_small_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_mistral_small_low_agent",
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
async fn blocking_mistral_small_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_mistral_small_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_mistral_small_mid_agent",
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
async fn blocking_mistral_small_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_mistral_small_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::MistralSmall,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_mistral_small_complete_agent",
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
async fn blocking_ministral_3b_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_ministral_3b_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_ministral_3b_low_agent",
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
async fn blocking_ministral_3b_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_ministral_3b_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_ministral_3b_mid_agent",
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
async fn blocking_ministral_3b_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/blocking_ministral_3b_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Ministral3b,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/blocking_ministral_3b_complete_agent",
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
async fn streaming_mistral_small_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_mistral_small_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_mistral_small_low_agent",
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
async fn streaming_mistral_small_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_mistral_small_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_mistral_small_mid_agent",
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
async fn streaming_mistral_small_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_mistral_small_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::MistralSmall,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_mistral_small_complete_agent",
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
async fn streaming_ministral_3b_low_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_ministral_3b_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_ministral_3b_low_agent",
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
async fn streaming_ministral_3b_mid_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_ministral_3b_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_ministral_3b_mid_agent",
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
async fn streaming_ministral_3b_complete_agent() -> Result<()> {
    const S: &str = "tool_truncation_matrix/streaming_ministral_3b_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Ministral3b,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_mistral_tool_truncation_cassette_result(
        "tool_truncation_matrix/streaming_ministral_3b_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_native(x, c, o)
        },
    )
    .await?;
    assert_cell(S, c, o);
    Ok(())
}
