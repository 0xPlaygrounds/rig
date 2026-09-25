//! Copilot route-specific completion smoke tests.

use crate::copilot::{LIVE_MODEL, live_client, live_responses_model, with_copilot_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn chat_models_route_through_chat_completions() {
    let response = rig::AgentBuilder::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
    )
    .preamble(BASIC_PREAMBLE)
    .build()
    .prompt(BASIC_PROMPT)
    .await
    .expect("chat-completions route should succeed");

    assert_nonempty_response(&response.output);
}

#[tokio::test]
async fn codex_models_route_through_responses() {
    with_copilot_cassette(
        "routing/codex_models_route_through_responses",
        |client| async move {
            let response = rig::AgentBuilder::new(
                client
                    .completion(live_responses_model())
                    .on(rig::transport()),
            )
            .preamble(BASIC_PREAMBLE)
            .build()
            .prompt("In one short sentence, explain what refactoring is.")
            .await
            .expect("responses route should succeed");

            assert_nonempty_response(&response.output);
        },
    )
    .await;
}
