//! Gemini reasoning tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::{GenerationOptions, Message, ProviderOptions, Reasoning};
use rig::providers::gemini::extension::{
    GeminiExt, GeminiOptions, GenerateContentOptions, GenerationConfig,
};

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    super::super::support::with_gemini_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion("gemini-2.5-flash"))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .options(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 4096 }))
                .provider_options(include_thoughts())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[gemini] Non-streaming chat failed - likely 400 from dropped reasoning");

            reasoning::assert_nonstreaming_universal(&result.output(), &call_count, "gemini");
        },
    )
    .await;
}

/// `includeThoughts`, beside the typed thinking budget.
fn include_thoughts() -> ProviderOptions {
    ProviderOptions::new()
        .with::<GeminiExt>(
            &GeminiOptions::new().generate_content(
                GenerateContentOptions::new()
                    .generation_config(GenerationConfig::new().include_thoughts(true)),
            ),
        )
        .expect("Gemini options serialize")
}
