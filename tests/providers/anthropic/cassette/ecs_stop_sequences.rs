//! Stop metadata and empty results from native agent execution.
use super::{empty_stop_sequence_matrix as empty, stop_sequence_terminal_matrix as terminal};
use crate::{ecs_agent::EcsAgent, ecs_lifecycle};
use rig::{
    completion::{CompletionModel, FinishReason},
    prelude::*,
    providers::anthropic,
};
use rig_ecs::agent::{AdditionalParams, MaxTokens};
use serde_json::json;

fn agent(model: impl CompletionModel + 'static, tokens: u64, stop: &str) -> EcsAgent {
    let mut ecs = ecs_lifecycle::agent(model, "");
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        MaxTokens(Some(tokens)),
        AdditionalParams(Some(json!({"stop_sequences": [stop]}))),
    ));
    ecs
}

#[tokio::test]
async fn agent_stream_single_sequence() {
    super::super::support::with_anthropic_stop_sequence_cassette(
        "stop_sequence_terminal_matrix/agent_stream_single_sequence",
        |client| async move {
            let mut ecs = agent(
                client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5),
                64,
                "charlie",
            );
            ecs.prompt(terminal::LIST_PROMPT, true).await;
            assert_eq!(
                ecs_lifecycle::provider_final(&mut ecs).finish_reason,
                Some(FinishReason::Stop)
            );
        },
    )
    .await;
    terminal::assert_recorded_terminal_stop_sequence(
        "stop_sequence_terminal_matrix/agent_stream_single_sequence",
        Some("charlie"),
    );
}

#[tokio::test]
async fn agent_prompt_empty_stop_sequence() {
    super::super::support::with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/agent_prompt_empty_stop_sequence",
        |client| async move {
            let mut ecs = agent(
                client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5),
                32,
                "alpha",
            );
            let response = ecs.prompt(empty::IMMEDIATE_PROMPT, false).await;
            assert!(
                response.trim().is_empty(),
                "the turn produced no text: {response:?}"
            );
        },
    )
    .await;
    empty::assert_recorded_empty_stop(
        "empty_stop_sequence_matrix/agent_prompt_empty_stop_sequence",
    );
}

#[tokio::test]
async fn agent_stream_empty_stop_sequence() {
    super::super::support::with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/agent_stream_empty_stop_sequence",
        |client| async move {
            let mut ecs = agent(
                client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5),
                32,
                "alpha",
            );
            let output = ecs.prompt(empty::IMMEDIATE_PROMPT, true).await;
            assert_eq!(
                output, "",
                "the run finishes with empty output rather than an error"
            );
            assert_eq!(
                ecs_lifecycle::provider_final(&mut ecs).finish_reason,
                Some(FinishReason::Stop)
            );
        },
    )
    .await;
    empty::assert_recorded_streamed_empty_stop(
        "empty_stop_sequence_matrix/agent_stream_empty_stop_sequence",
    );
}
