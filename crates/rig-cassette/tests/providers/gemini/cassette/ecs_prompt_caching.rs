//! Native agent cache-growth loops with the original neutral assertions.
use super::prompt_caching::{CACHE_MODEL, GEMINI_CACHE_SUPPORT, probe};
use crate::cache_conformance::assert_breakpoints_match_support;
use crate::{
    cache_conformance::assert_prefix_stable, ecs_agent::EcsAgent, ecs_cache::assert_cache_growth,
};

#[tokio::test]
async fn agent_loop_keeps_hitting_across_tool_turns() {
    rig_test_support::goldens::world_golden_test(
        async {
            super::super::support::with_gemini_prompt_caching_cassette(
                "prompt_caching/agent_loop",
                |client| async move {
                    let ecs = EcsAgent::new(client.completion(CACHE_MODEL), &probe().preamble, 1);

                    assert_cache_growth(ecs, &GEMINI_CACHE_SUPPORT, "agent loop").await;
                },
            )
            .await;
            assert_prefix_stable("gemini", "prompt_caching/agent_loop");
            assert_breakpoints_match_support(
                "gemini",
                "prompt_caching/agent_loop",
                &GEMINI_CACHE_SUPPORT,
            );
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_prompt_caching_agent_loop_keeps_hitting_across_tool_turns",
                log,
            )
        },
    )
    .await
}
