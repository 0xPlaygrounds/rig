//! Anthropic agent completion smoke test.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::prelude::AgentClientExt;
use rig::providers::anthropic;
use rig_bevy::{LocalRuntime, TenantId};

use super::super::support::with_anthropic_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn bevy_local_blocking_smoke_preserves_raw_final() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let request = model
            .completion_request(BASIC_PROMPT)
            .preamble(BASIC_PREAMBLE.to_string())
            .build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime.run(request, 1).await.expect("Bevy completion");
        assert!(!result.snapshot.output.is_empty());
        let _: anthropic::completion::CompletionResponse = result.raw_response;
    })
    .await;
}
