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
    recorded_request_body, with_zai_anthropic_cassette, with_zai_coding_cassette,
    with_zai_general_cassette,
};
use super::super::{CHEAP_GENERAL_MODEL, CODING_MODEL};
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

    // The Anthropic dialect speaks a different request shape entirely, so the
    // premise worth pinning is that rig sent Messages-API bytes and not Chat
    // Completions bytes to `/api/anthropic`.
    let request = recorded_request_body("anthropic/completion_blocking_smoke");
    assert_eq!(request["model"], CODING_MODEL);
    assert!(
        request["max_tokens"].is_number(),
        "the Anthropic dialect requires max_tokens; request was {request}"
    );
}
