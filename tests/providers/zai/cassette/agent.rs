//! One completion smoke per Z.AI client surface.
//!
//! Z.AI reaches rig through three different code paths — the shared OpenAI
//! Chat Completions layer on two base URLs, and the shared Anthropic Messages
//! layer on a third — so "does a plain prompt work" is three separate claims.
//! Each cell also re-derives the endpoint it hit from its own fixture, because
//! base-URL composition (a doubled `/v4`, a dropped `/api`) is invisible from
//! the response alone.

use rig::completion::Prompt;
use rig::prelude::*;

use super::super::support::{
    recorded_request_body, recorded_request_path, with_zai_anthropic_cassette,
    with_zai_coding_cassette, with_zai_general_cassette,
};
use super::super::{ANTHROPIC_MODEL, CHEAP_GENERAL_MODEL, CODING_MODEL};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn general_completion_blocking_smoke() {
    with_zai_general_cassette("general/completion_blocking_smoke", |client| async move {
        let agent = client
            .agent(CHEAP_GENERAL_MODEL)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("Z.AI general completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;

    // Read back after the wrapper returns: in record mode the fixture is
    // written on the way out, so an in-body read would assert against the
    // previous recording.
    assert_eq!(
        recorded_request_path("general/completion_blocking_smoke"),
        "/api/paas/v4/chat/completions",
        "the general base URL must compose with the endpoint suffix exactly once"
    );
    let request = recorded_request_body("general/completion_blocking_smoke");
    assert_eq!(
        request["model"], CHEAP_GENERAL_MODEL,
        "the general endpoint must be asked for the model the cell named"
    );
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn coding_completion_blocking_smoke() {
    with_zai_coding_cassette("coding/completion_blocking_smoke", |client| async move {
        let agent = client
            .agent(CODING_MODEL)
            .preamble("You are a concise coding assistant.")
            .build();

        let response = agent
            .prompt("In one short sentence, explain what a unit test is.")
            .await
            .expect("Z.AI coding completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;

    assert_eq!(
        recorded_request_path("coding/completion_blocking_smoke"),
        "/api/coding/paas/v4/chat/completions",
        "the coding base URL must compose with the endpoint suffix exactly once"
    );
    let request = recorded_request_body("coding/completion_blocking_smoke");
    assert_eq!(
        request["model"], CODING_MODEL,
        "the coding endpoint must be asked for the model the cell named"
    );
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn anthropic_completion_blocking_smoke() {
    with_zai_anthropic_cassette("anthropic/completion_blocking_smoke", |client| async move {
        let agent = client.agent(CODING_MODEL).preamble(BASIC_PREAMBLE).build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("Z.AI Anthropic-compatible completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;

    assert_eq!(
        recorded_request_path("anthropic/completion_blocking_smoke"),
        "/api/anthropic/v1/messages",
        "the Anthropic base URL must compose with the Messages suffix exactly once"
    );

    // The premise worth pinning is that rig sent Messages-API bytes and not
    // Chat Completions bytes. `max_tokens` cannot show that — both dialects
    // have the field, and this client injects a default for it — but two
    // things can: Anthropic hoists the preamble out of `messages` into a
    // top-level `system`, and its user content is an array of typed blocks
    // rather than a bare string.
    let request = recorded_request_body("anthropic/completion_blocking_smoke");
    assert_eq!(request["model"], ANTHROPIC_MODEL);
    assert_eq!(
        request["system"][0]["text"], BASIC_PREAMBLE,
        "the Messages API hoists the preamble into a top-level `system` block array; \
         request was {request}"
    );
    assert!(
        request["messages"][0]["content"].is_array(),
        "the Messages API carries user content as typed blocks; request was {request}"
    );
}
