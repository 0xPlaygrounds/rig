//! xAI reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test xai xai::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use super::support::with_xai_cassette;
use crate::reasoning::{self, WeatherTool};
use rig::prelude::*;
use rig::providers::xai;

#[tokio::test]
async fn streaming() {
    with_xai_cassette("reasoning_tool_roundtrip/streaming", |env| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = env
            .agent(xai::GROK_3_MINI)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(4096)
            .tool(WeatherTool::new(call_count.clone()))
            .build();

        let stream = agent
            .runner(reasoning::TOOL_USER_PROMPT)
            .history(Vec::<Message>::new())
            .max_turns(3)
            .stream_run();

        let stats = reasoning::collect_stream_stats(stream, "xai").await;
        reasoning::assert_universal(&stats, &call_count, "xai");
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_xai_cassette("reasoning_tool_roundtrip/nonstreaming", |env| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = env
            .agent(xai::GROK_3_MINI)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(4096)
            .tool(WeatherTool::new(call_count.clone()))
            .default_max_turns(2)
            .build();

        let result = agent
            .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
            .await
            .expect("[xai] Non-streaming chat failed - likely 400 from dropped reasoning");

        reasoning::assert_nonstreaming_universal(&result, &call_count, "xai");
    })
    .await;
}
