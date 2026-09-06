//! Provider code execution and retained stream text through native agent runs.
use super::{code_execution_matrix as code, stream_terminal_matrix as terminal};
use crate::{ecs_agent::EcsAgent, ecs_lifecycle, ecs_observation};
use rig::{completion::CompletionModel, message::Message, prelude::*, providers::gemini};
use rig_ecs::{
    agent::{AdditionalParams, MaxTokens, Temperature},
    systems::spawn_run,
};

fn agent(model: impl CompletionModel + 'static, params: serde_json::Value) -> EcsAgent {
    let mut ecs = ecs_lifecycle::agent(model, "");
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Temperature(Some(0.0)),
        MaxTokens(Some(2000)),
        AdditionalParams(Some(params)),
    ));
    ecs
}

#[tokio::test]
async fn blocking_agent_prompt_answers_after_code_execution() {
    super::super::support::with_gemini_code_execution_cassette("code_execution_matrix/blocking_agent_prompt_answers_after_code_execution", |client| async move {
        let mut ecs = agent(client.completion_model(gemini::completion::GEMINI_2_5_FLASH), code::code_execution_params());
        let answer = ecs.prompt("Use the code execution tool to compute 2 to the power of 20. State the number in your answer.", false).await;
        assert!(code::states(&answer, "1048576"), "agent answer should carry the computed value, got {answer:?}");
    }).await;
    super::super::support::assert_recorded_response_contains(
        "code_execution_matrix/blocking_agent_prompt_answers_after_code_execution",
        code::CODE_PART_MARKERS,
    );
}

#[tokio::test]
async fn streaming_agent_prompt_answers_after_code_execution() {
    super::super::support::with_gemini_code_execution_cassette("code_execution_matrix/streaming_agent_prompt_answers_after_code_execution", |client| async move {
        let mut ecs = agent(client.completion_model(gemini::completion::GEMINI_2_5_FLASH), code::code_execution_params());
        ecs_observation::install_observers(&mut ecs);
        ecs.prompt("Use the code execution tool to compute 2 to the power of 20. State the number in your answer.", true).await;
        let answer = &ecs_observation::observation(&ecs).all_streamed_text;
        assert!(code::states(answer, "1048576"), "streamed agent answer should carry the computed value, got {answer:?}");
    }).await;
    super::super::support::assert_recorded_response_contains(
        "code_execution_matrix/streaming_agent_prompt_answers_after_code_execution",
        code::CODE_PART_MARKERS,
    );
}

#[tokio::test]
async fn blocking_code_execution_replayed_in_chat_history() {
    super::super::support::with_gemini_code_execution_cassette(
        "code_execution_matrix/blocking_code_execution_replayed_in_chat_history",
        |client| async move {
            let mut ecs = agent(
                client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
                code::code_execution_params(),
            );
            let prompt = "Use the code execution tool to compute 13 times 13. State the number.";
            let first = ecs.prompt(prompt, false).await;
            assert!(
                code::states(&first, "169"),
                "first answer should carry 169, got {first:?}"
            );
            let history = [Message::user(prompt), Message::assistant(first)].map(|message| {
                rig_ecs::agent::MessageParts::from_message(&message).expect("conversation message")
            });
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                "Now double that number and state the result.",
                false,
                None,
            );
            let second = ecs.wait_for_success(run).await;
            assert!(
                code::states(&second, "338"),
                "second answer should carry the doubled value, got {second:?}"
            );
        },
    )
    .await;
    super::super::support::assert_recorded_response_contains(
        "code_execution_matrix/blocking_code_execution_replayed_in_chat_history",
        code::CODE_PART_MARKERS,
    );
}

#[tokio::test]
async fn two_terminal_stream_agent_prompt_keeps_the_answer() {
    super::super::support::with_gemini_stream_terminal_cassette("stream_terminal_matrix/two_terminal_stream_agent_prompt_keeps_the_answer", |client| async move {
        let mut ecs = agent(client.completion_model(gemini::completion::GEMINI_2_5_FLASH), terminal::code_execution_params());
        ecs_observation::install_observers(&mut ecs);
        ecs.prompt(terminal::TWO_ROUND_PROMPT, true).await;
        let answer = &ecs_observation::observation(&ecs).all_streamed_text;
        assert!(terminal::states(answer, terminal::FIRST_ROUND_VALUE), "the agent's streamed answer must survive the intermediate finishReason, got {answer:?}");
    }).await;
    super::super::support::assert_recorded_stream_finishes_early(
        "stream_terminal_matrix/two_terminal_stream_agent_prompt_keeps_the_answer",
        true,
    );
}
