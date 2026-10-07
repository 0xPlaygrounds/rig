//! OpenAI Responses API behavior regression tests.
//!
//! Locks down strict-tool opt-in, incomplete-response surfacing, and
//! system-instruction placement as input items.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::agent::AgentBuilder;
use rig::completion::{FinishReason, Message};
use rig::message::AssistantContent;
use rig::providers::openai;
use rig_test_support::cassette_models::MapWire;

use rig::providers::openai::extension::OpenAiExt;

use super::super::support::{stateless, with_openai_cassette};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn incomplete_response_surfaces_partial_output() {
    with_openai_cassette(
        "responses_behaviors/incomplete_response_surfaces_partial_output",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let request = CompletionRequest::new(
                "Write a story of at least 150 words about a lighthouse keeper.",
            )
            .preamble("You are a storyteller.")
            .max_tokens(16);

            // The cassette records a single interaction, and one call yields
            // both views of it: the normalized response, and the provider's
            // own reply in `raw`. `status` has no typed extra, so it is read
            // off the latter; the incomplete reason is a typed extra.
            let response = model
                .call(request)
                .await
                .expect("an incomplete response should still convert, not error");
            let reply = &response.raw;

            assert_eq!(
                reply["status"], "incomplete",
                "hitting max_output_tokens should mark the response incomplete"
            );
            let extras = response
                .extras::<OpenAiExt>()
                .expect("the reply is OpenAI's")
                .expect("the reply holds the extras");
            assert_eq!(
                extras.incomplete_reason.as_deref(),
                Some("max_output_tokens"),
                "incomplete_details should carry the truncation reason"
            );

            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Length),
                "the incomplete/max_output_tokens pair should normalize to a length stop"
            );
            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.trim().is_empty(),
                "partial output text should still be surfaced"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn system_messages_as_input_items_mid_conversation() {
    with_openai_cassette(
        "responses_behaviors/system_messages_as_input_items_mid_conversation",
        |client| async move {
            // The recorded request body locks the placement contract: with
            // `with_system_instructions_as_messages`, the preamble and the
            // mid-conversation system message are sent as `system` input
            // items instead of the top-level `instructions` field.
            let model = client
                .openai
                .responses(openai::GPT_4O)
                .map_wire(|wire| wire.with_system_instructions_as_messages());
            let agent = AgentBuilder::new(model)
                .preamble("You are a concise assistant.")
                .provider_options(stateless())
                .build();
            let mut history = vec![
                Message::user("Hello!"),
                Message::assistant("Hi! How can I help you today?"),
                Message::system(
                    "The user's codename is FALCON-9. Always refer to the user by codename.",
                ),
            ];

            let result = agent
                .chat("What is my codename?", &mut history)
                .await
                .expect("chat with a mid-conversation system message should succeed")
                .output();

            assert!(
                result.contains("FALCON-9"),
                "the mid-conversation system message must reach the model, got {result:?}"
            );
        },
    )
    .await;
}
