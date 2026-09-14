//! Native ECS provider completions preserving the original cassette assertions.
use crate::ecs_agent::EcsAgent;
use crate::ollama::support::with_ollama_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::prelude::*;
use rig_ecs::agent::DefaultMaxTurns;
const MODEL: &str = "qwen3:4b";
#[tokio::test]
async fn completion_smoke() {
    with_ollama_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(client.completion_model(MODEL), BASIC_PREAMBLE, 1);
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(rig_ecs::agent::AdditionalParams(Some(
                serde_json::json!({ "think" : false }),
            )));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
/// Guards the native token-limit mapping on the wire.
///
/// `max_tokens` has no top-level field in Ollama's native `/api/chat`; the
/// equivalent is the `num_predict` model parameter inside `options`. The
/// recorded request body carries `"options":{"num_predict":24}`, and the
/// cassette matcher compares request bodies, so a regression that dropped
/// `num_predict` or moved the limit back to the top level would stop matching
/// and fail here. The serialization unit tests in `providers::ollama` cover the
/// conversion; this covers that Ollama is actually sent it.
///
/// The recorded response has `done_reason: "length"` rather than `"stop"`,
/// which is the server confirming it honored the budget.
#[tokio::test]
async fn completion_respects_max_tokens() {
    with_ollama_cassette("agent/max_tokens", |client| async move {
        let mut ecs = EcsAgent::new(client.completion_model(MODEL), BASIC_PREAMBLE, 1);
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(rig_ecs::agent::MaxTokens(Some(24)));
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(rig_ecs::agent::AdditionalParams(Some(
                serde_json::json!({ "think" : false }),
            )));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
