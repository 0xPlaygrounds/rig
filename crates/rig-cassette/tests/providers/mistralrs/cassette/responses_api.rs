//! Cassette coverage for mistral.rs through Rig's OpenAI Responses wire.

use rig::agent::AgentBuilder;
use rig::providers::openai::responses_api::wire::Responses;
use rig_test_support::cassette_models::MapWire;

use crate::support::assert_contains_all_case_insensitive;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_cassette};

#[tokio::test]
async fn responses_api_multi_turn_replays_history() {
    with_mistralrs_cassette(
        "responses_api/responses_api_multi_turn_replays_history",
        |client| async move {
            let model = client
                .responses(model_name())
                .map_wire(Responses::with_system_instructions_as_messages);
            let agent = AgentBuilder::new(model)
                .preamble(SYSTEM_PROMPT)
                .max_tokens(256)
                .build();
            let mut history = Vec::new();

            let _first = agent
                .chat(
                    "Think briefly, then answer in one sentence why usage accounting matters.",
                    &mut history,
                )
                .await
                .expect("first multi-turn Responses API chat should succeed");
            let first_history_len = history.len();
            let second = agent
                .chat("/no_think Reply with exactly: OK", &mut history)
                .await
                .expect("second multi-turn Responses API chat should succeed")
                .output();

            assert!(
                first_history_len > 0 && history.len() > first_history_len,
                "multi-turn history should be updated; first_len={first_history_len}, final_len={}",
                history.len()
            );
            assert_contains_all_case_insensitive(&second, &["OK"]);
        },
    )
    .await;
}
