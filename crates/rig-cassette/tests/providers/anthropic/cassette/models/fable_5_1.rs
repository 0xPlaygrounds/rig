//! Claude Fable 5.1's recorded session. It covers only what a fix changed: the extractor and structured output
//! with tools, which now use native structured output, and the documented
//! refusal of a forced tool choice. `main` already covers the rest.

use rig::providers::anthropic::completion::CLAUDE_FABLE_5_1;
use rig_test_support::model_session::{self, AnthropicProfile};

use super::super::super::support::with_anthropic_model_session_cassette;

#[tokio::test]
async fn session() {
    let session = with_anthropic_model_session_cassette(
        "models/fable_5_1/session",
        false,
        |models, files, clock| async move {
            model_session::anthropic(
                models,
                files,
                clock,
                &AnthropicProfile {
                    model: CLAUDE_FABLE_5_1,
                    rejects_forced_tool_choice: true,
                    mid_conversation_system: true,
                    fixes_only: true,
                },
            )
            .await
        },
    )
    .await;
    assert!(!session.phases.is_empty());
}
