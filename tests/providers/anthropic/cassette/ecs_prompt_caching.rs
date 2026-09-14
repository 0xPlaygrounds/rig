//! Native agent cache-growth loops with the original neutral assertions.
use super::prompt_caching::{ANTHROPIC_CACHE_SUPPORT, conformance_probe};
use crate::cache_conformance::assert_breakpoints_match_support;
use crate::{
    cache_conformance::assert_prefix_stable, ecs_agent::EcsAgent, ecs_cache::assert_cache_growth,
};
use rig::prelude::*;
use rig::providers::anthropic;

#[tokio::test]
async fn conformance_agent_loop_keeps_hitting_across_tool_turns() {
    super::super::support::with_anthropic_cassette(
        "prompt_caching/conformance_agent_loop",
        |client| async move {
            let ecs = EcsAgent::new(
                client
                    .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                    .with_prompt_caching(),
                &conformance_probe().preamble,
                1,
            );

            assert_cache_growth(ecs, &ANTHROPIC_CACHE_SUPPORT, "agent loop").await;
        },
    )
    .await;
    assert_prefix_stable("anthropic", "prompt_caching/conformance_agent_loop");
    assert_breakpoints_match_support(
        "anthropic",
        "prompt_caching/conformance_agent_loop",
        &ANTHROPIC_CACHE_SUPPORT,
    );
}
