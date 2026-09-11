//! Native OpenAI Chat tool-lifecycle agent cells with original shared assertions.
use super::super::support::with_openai_tool_lifecycle_cassette_result;
use super::chat_tool_lifecycle_matrix::{
    Cell, Model, Observation, PREAMBLE, Shape, Surface, Transport, assert_nonempty_distinct_ids,
    cell, execute, expected_arguments, expected_names, model_name, prompt, tool_definition,
};
use crate::ecs_agent::EcsAgent;
use anyhow::Result;
use rig::{prelude::*, providers::openai, tool::Tool};
use rig_ecs::{
    agent::{AdditionalParams, Failure, MaxTokens, ToolCallSlot},
    bus::{PendingEffect, Streamed},
    commands::Prompt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
type SharedObservation = Arc<Mutex<Option<Observation>>>;

type InvocationLog = Arc<Mutex<Vec<String>>>;

#[derive(Debug, thiserror::Error)]
#[error("matrix tool failed")]
struct MatrixToolError;

#[derive(Debug, Deserialize, Serialize)]
struct EmptyArgs {}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadArgs {
    label: String,
    values: Vec<i64>,
    meta: PayloadMeta,
}

#[derive(Debug, Deserialize, Serialize)]
struct PayloadMeta {
    active: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ValueArgs {
    value: String,
}

fn note(log: &InvocationLog, name: &str) {
    log.lock()
        .expect("invocation log poisoned")
        .push(name.to_owned());
}

macro_rules! impl_matrix_tool {
    ($ty:ident, $name:literal, $args:ty) => {
        #[derive(Clone)]
        struct $ty {
            log: InvocationLog,
        }

        impl Tool for $ty {
            const NAME: &'static str = $name;
            type Error = MatrixToolError;
            type Args = $args;
            type Output = String;

            fn description(&self) -> String {
                format!("Matrix tool {}", Self::NAME)
            }
            fn parameters(&self) -> Value {
                tool_definition(Self::NAME).parameters
            }

            async fn call(
                &self,
                _context: &mut rig::tool::ToolContext,
                _args: Self::Args,
            ) -> std::result::Result<Self::Output, Self::Error> {
                note(&self.log, Self::NAME);
                Ok(Self::NAME.to_owned())
            }
        }
    };
}

impl_matrix_tool!(Ping, "ping", EmptyArgs);
impl_matrix_tool!(RecordPayload, "record_payload", PayloadArgs);
impl_matrix_tool!(Alpha, "alpha", ValueArgs);
impl_matrix_tool!(Beta, "beta", ValueArgs);

async fn run_cell(client: openai::Client, cell: Cell, observed: SharedObservation) -> Result<()> {
    assert_eq!(cell.surface, Surface::Agent);
    let invocations = InvocationLog::default();
    let mut ecs = EcsAgent::new(
        client
            .completions_api()
            .completion_model(model_name(cell.model)),
        PREAMBLE,
        1,
    );
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        MaxTokens(Some(128)),
        AdditionalParams(Some(
            json!({"tool_choice":"required", "parallel_tool_calls":cell.shape == Shape::Parallel}),
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
    let streamed = cell.transport == Transport::Streaming;
    let mut request = Prompt::new(ecs.agent, prompt(cell.shape));
    if streamed {
        request = request.streaming().max_turns(1);
    }
    let run = request.spawn(ecs.app.world_mut())?;
    let failure = ecs
        .wait_for_outcome(run)
        .await
        .expect_err("required tool turn exhausts one-turn budget");
    assert_eq!(failure, Failure::MaxTurns { limit: 1 });
    assert!(
        ecs.app
            .world_mut()
            .query::<&Streamed>()
            .iter(ecs.app.world())
            .all(|stream| stream.errors.is_empty()),
        "native provider streams have no item errors"
    );
    // Supplemental native graph observation: the original agent assertion only
    // checks invocation order, while its id/argument checks inspect wire fixtures.
    let world = ecs.app.world_mut();
    let mut query = world.query::<(&ToolCallSlot, &PendingEffect)>();
    let mut calls: Vec<_> = query.iter(world).collect();
    calls.sort_by_key(|(slot, _)| slot.index);
    let names: Vec<_> = calls.iter().map(|(slot, _)| slot.name.as_str()).collect();
    assert_eq!(names, expected_names(cell.shape));
    let ids: Vec<_> = calls.iter().map(|(slot, _)| slot.id.to_string()).collect();
    assert_nonempty_distinct_ids("native dispatched calls", &ids);
    let arguments: Vec<Value> = calls
        .iter()
        .map(|(_, pending)| match &pending.kind {
            rig::effect::EffectKind::ToolCall { args, .. } => {
                serde_json::from_str(args).expect("actual tool args JSON")
            }
            _ => panic!("tool slot owns tool effect"),
        })
        .collect();
    assert_eq!(arguments, expected_arguments(cell.shape));
    *observed.lock().expect("observation mutex poisoned") = Some(Observation {
        errors: vec![format!("{failure:?}")],
        invocations: invocations.lock().expect("invocations").clone(),
        ..Default::default()
    });
    Ok(())
}

#[tokio::test]
async fn blocking_gpt4o_zero_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt4o_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt4o_zero_agent",
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
async fn blocking_gpt4o_nested_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt4o_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt4o_nested_agent",
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
async fn blocking_gpt4o_parallel_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt4o_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt4o_parallel_agent",
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
async fn blocking_gpt41_zero_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt41_zero_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt41_zero_agent",
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
async fn blocking_gpt41_nested_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt41_nested_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt41_nested_agent",
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
async fn blocking_gpt41_parallel_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/blocking_gpt41_parallel_agent";
    let c = cell(
        Transport::Blocking,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/blocking_gpt41_parallel_agent",
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
async fn streaming_gpt4o_zero_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt4o_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt4o_zero_agent",
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
async fn streaming_gpt4o_nested_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt4o_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt4o_nested_agent",
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
async fn streaming_gpt4o_parallel_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt4o_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt4oMini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt4o_parallel_agent",
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
async fn streaming_gpt41_zero_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt41_zero_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Zero,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt41_zero_agent",
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
async fn streaming_gpt41_nested_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt41_nested_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Nested,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt41_nested_agent",
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
async fn streaming_gpt41_parallel_agent() -> Result<()> {
    const S: &str = "chat_tool_lifecycle_matrix/streaming_gpt41_parallel_agent";
    let c = cell(
        Transport::Streaming,
        Model::Gpt41Mini,
        Shape::Parallel,
        Surface::Agent,
    );
    let o = SharedObservation::default();
    with_openai_tool_lifecycle_cassette_result(
        "chat_tool_lifecycle_matrix/streaming_gpt41_parallel_agent",
        {
            let o = Arc::clone(&o);
            move |x| run_cell(x, c, o)
        },
    )
    .await?;
    execute(S, c, o).await;
    Ok(())
}
