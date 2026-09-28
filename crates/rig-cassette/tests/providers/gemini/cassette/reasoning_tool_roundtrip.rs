//! The Gemini source of the cross-provider portability matrix: signed thoughts
//! beside a `get_weather` call, then the answer after the tool result. Other
//! wires continue this recording's first reply
//! (`rig_test_support::history_survival::portability`).

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;
use rig::providers::gemini::api;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    super::super::support::with_gemini_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let model = client
                .completion("gemini-2.5-flash")
                .settings(api::RequestSettings {
                    generation_config: api::GenerationSettings {
                        thinking_config: Some(api::ThinkingConfig {
                            thinking_budget: Some(4096),
                            include_thoughts: Some(true),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                });
            let agent = rig::AgentBuilder::new(model)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[gemini] Non-streaming chat failed - likely 400 from dropped reasoning");

            reasoning::assert_nonstreaming_universal(&result.output, &call_count, "gemini");
        },
    )
    .await;
}
