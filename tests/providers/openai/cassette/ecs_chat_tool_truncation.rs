//! Native OpenAI Chat tool-truncation agent cells with original shared assertions.
//!
//! The twelve agent cells of `chat_tool_truncation_matrix` replay their
//! original fixtures through the native runtime. A `length`-truncated required
//! call must never materialise or dispatch; the complete control must dispatch
//! exactly once before the one-turn budget ends the run.
use super::super::support::with_openai_tool_truncation_cassette_result;
use super::chat_tool_truncation_matrix::{
    Budget, Cell, FileReport, Model, Observation, PREAMBLE, PROMPT, Surface, Transport, cell,
    execute, max_tokens, model_name,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_observation,
    ecs_termination::{self, NativeProbe},
};
use anyhow::Result;
use rig::{completion::FinishReason, prelude::*, providers::openai};
use rig_ecs::{
    agent::{AdditionalParams, Failure, MaxTokens, ToolCallSlot},
    bus::{PendingEffect, Streamed},
    systems::spawn_run,
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

type SharedObservation = Arc<Mutex<Option<Observation>>>;

async fn run_cell(client: openai::Client, cell: Cell, observed: SharedObservation) -> Result<()> {
    assert_eq!(cell.surface, Surface::Agent);
    let invocations = Arc::new(AtomicUsize::new(0));
    let probe = NativeProbe::default();
    let mut ecs = EcsAgent::new(
        client
            .completions_api()
            .completion_model(model_name(cell.model)),
        PREAMBLE,
        1,
    );
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        MaxTokens(Some(max_tokens(cell.budget))),
        AdditionalParams(Some(json!({ "tool_choice": "required" }))),
    ));
    // The original's own counting tool: its schema is the advertised
    // `file_report` definition and its counter is the invocation obligation.
    ecs.tool(FileReport {
        invocations: Arc::clone(&invocations),
    });
    ecs_observation::install_observers(&mut ecs);
    ecs_termination::install(&mut ecs, probe.clone(), None);
    // Blocking declares the one-turn budget on the agent only; streaming also
    // declares it on the run, as the original `stream_chat(..).max_turns(1)`.
    let streamed = cell.transport == Transport::Streaming;
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        PROMPT,
        streamed,
        streamed.then_some(1),
    );
    let outcome = ecs.wait_for_outcome(run).await;
    // Streamed exists only for the streaming transport; blocking cells have no
    // stream entity, so this guard is a no-op for them.
    assert!(
        ecs.app
            .world_mut()
            .query::<&Streamed>()
            .iter(ecs.app.world())
            .all(|stream| stream.errors.is_empty()),
        "native provider streams have no item errors"
    );
    // Supplemental native graph observation: what actually reached dispatch.
    let world = ecs.app.world_mut();
    let mut slots = world.query::<(&ToolCallSlot, &PendingEffect)>();
    let dispatched: Vec<(String, Value)> = slots
        .iter(world)
        .map(|(slot, pending)| match &pending.kind {
            rig::effect::EffectKind::ToolCall { args, .. } => (
                slot.name.clone(),
                serde_json::from_str(args).expect("actual tool args JSON"),
            ),
            _ => panic!("tool slot owns tool effect"),
        })
        .collect();
    let errors = match cell.budget {
        Budget::Complete => {
            // The required call runs, then the one-turn budget ends the run.
            // The original tolerates that ending and only counts the call.
            let failure = outcome.expect_err("complete required call exhausts the one-turn budget");
            assert_eq!(failure, Failure::MaxTurns { limit: 1 });
            assert_eq!(probe.first_reason(), Some(FinishReason::ToolCalls));
            assert_eq!(
                dispatched.len(),
                1,
                "one native dispatch for the complete call"
            );
            assert_eq!(dispatched[0].0, "file_report");
            assert_eq!(
                dispatched[0].1["summary"], PROMPT,
                "exact complete arguments dispatched"
            );
            vec![format!("{failure:?}")]
        }
        Budget::Low | Budget::Mid => {
            // Decoding drops the partial call, so the turn carries no call and
            // the run settles. The original only requires the loop to survive
            // without a provider error; native success is the stronger check.
            outcome.expect("truncated turn settles without a materialised call");
            assert_eq!(probe.first_reason(), Some(FinishReason::Length));
            assert!(dispatched.is_empty(), "a partial call never materialises");
            assert!(
                ecs_observation::observation(&ecs).tool_calls.is_empty(),
                "a partial call is never observed as a call"
            );
            vec![]
        }
    };
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
async fn blocking_gpt4o_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt4o_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt4o_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_low_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_mid_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn blocking_gpt41_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/blocking_gpt41_complete_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/blocking_gpt41_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt4o_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt4o_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt4o_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_low_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_low_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Low,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_low_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_mid_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_mid_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Mid,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_mid_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
#[tokio::test]
async fn streaming_gpt41_complete_agent() -> Result<()> {
    const S: &str = "chat_tool_truncation_matrix/streaming_gpt41_complete_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Budget::Complete,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_truncation_cassette_result(
        "chat_tool_truncation_matrix/streaming_gpt41_complete_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
