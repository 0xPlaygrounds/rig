//! Native agent cache-growth loops with the original neutral assertions.
use super::prompt_caching::{CACHE_MODEL, OPENAI_CACHE_SUPPORT, probe};
use crate::cache_conformance::assert_breakpoints_match_support;
use crate::{
    cache_conformance::assert_prefix_stable, ecs_agent::EcsAgent, ecs_cache::assert_cache_growth,
};
use rig::prelude::*;

#[tokio::test]
async fn chat_completions_agent_loop_keeps_hitting_across_tool_turns() {
    super::super::support::with_openai_completions_prompt_caching_cassette(
        "prompt_caching/chat_completions_agent_loop",
        |client| async move {
            let ecs = EcsAgent::new(client.completion_model(CACHE_MODEL), &probe().preamble, 1);

            assert_cache_growth(ecs, &OPENAI_CACHE_SUPPORT, "chat completions agent loop").await;
        },
    )
    .await;
    assert_prefix_stable("openai", "prompt_caching/chat_completions_agent_loop");
    assert_breakpoints_match_support(
        "openai",
        "prompt_caching/chat_completions_agent_loop",
        &OPENAI_CACHE_SUPPORT,
    );
}

#[tokio::test]
async fn responses_agent_loop_keeps_hitting_across_tool_turns() {
    super::super::support::with_openai_prompt_caching_cassette(
        "prompt_caching/responses_agent_loop",
        |client| async move {
            let mut ecs = EcsAgent::new(client.completion_model(CACHE_MODEL), &probe().preamble, 1);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(rig_ecs::agent::AdditionalParams(Some(
                    serde_json::json!({"prompt_cache_key": "rig-cache-conformance-openai-agent"}),
                )));
            assert_cache_growth(ecs, &OPENAI_CACHE_SUPPORT, "responses agent loop").await;
        },
    )
    .await;
    assert_prefix_stable("openai", "prompt_caching/responses_agent_loop");
    assert_breakpoints_match_support(
        "openai",
        "prompt_caching/responses_agent_loop",
        &OPENAI_CACHE_SUPPORT,
    );
}
