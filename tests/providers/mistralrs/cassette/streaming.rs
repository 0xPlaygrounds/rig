//! Cassette coverage for mistral.rs chat-completions streaming reasoning chunks.

use crate::support::collect_stream_observation;
use rig::prelude::*;

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_cassette};

#[tokio::test]
async fn chat_completions_stream_emits_reasoning_and_text_incrementally() {
    with_mistralrs_cassette(
        "streaming/chat_completions_stream_emits_reasoning_and_text_incrementally",
        |env| async move {
            let agent = AgentBuilder::new(env.chat_provider(model_name()))
                .preamble(SYSTEM_PROMPT)
                .max_tokens(512)
                .build();
            let mut stream =Box::pin( agent
                .runner(
                    "Think briefly, then answer with three short bullet points about token usage reporting.",
                )
                .stream_run());
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation.events.contains(&"reasoning_delta"),
                "stream should emit reasoning deltas; events={:?}",
                observation.events
            );
            assert!(
                observation.events.contains(&"text"),
                "stream should emit text chunks; events={:?}",
                observation.events
            );
            assert!(
                observation.got_final_response,
                "stream should emit final response; events={:?}",
                observation.events
            );
        },
    )
    .await;
}
