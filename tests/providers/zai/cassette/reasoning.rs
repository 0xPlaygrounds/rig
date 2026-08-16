//! GLM's thinking output on the OpenAI dialect, round-tripped into a second
//! turn.
//!
//! GLM emits reasoning by default, and the OpenAI dialect carries it as
//! `reasoning_content` on the assistant message. Two things can go wrong and
//! neither is visible from a single turn: rig can drop or mislabel the field on
//! the way in, and rig can fail to echo it back on the way out — which is what
//! a provider that requires reasoning continuity rejects. Feeding the parsed
//! turn back as history is what exercises both.

use rig::completion::{Chat, Message, Prompt};
use rig::prelude::*;

use super::super::THINKING_MODEL;
use super::super::support::{recorded_response_body, with_zai_general_cassette};
use crate::support::assert_nonempty_response;

/// A prompt small enough to be cheap but arithmetical enough that GLM thinks
/// before answering.
const THINKING_PROMPT: &str = "What is 17 * 23? Reply with just the number.";

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_thinking_roundtrip_blocking() {
    with_zai_general_cassette("general/thinking_roundtrip_blocking", |client| async move {
        let agent = client
            .agent(THINKING_MODEL)
            .preamble("You are a concise assistant.")
            .max_tokens(256)
            .build();

        let first = agent
            .prompt(THINKING_PROMPT)
            .await
            .expect("Z.AI thinking turn should succeed");
        assert_nonempty_response(&first);

        // Turn two replays turn one as history, so whatever rig parsed out of
        // `reasoning_content` has to serialize back into a request Z.AI
        // accepts.
        let mut history = vec![
            Message::user(THINKING_PROMPT),
            Message::assistant(first.clone()),
        ];
        let second = agent
            .chat("Now double it.", &mut history)
            .await
            .expect("replaying a thinking turn as history should succeed");
        assert_nonempty_response(&second);
    })
    .await;

    // Premise: a cell about reasoning that stopped receiving reasoning would
    // keep passing while covering nothing.
    let body = recorded_response_body("general/thinking_roundtrip_blocking");
    let reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .expect("the recorded turn must actually carry reasoning_content");
    assert!(
        !reasoning.trim().is_empty(),
        "an empty reasoning_content covers nothing"
    );
}
