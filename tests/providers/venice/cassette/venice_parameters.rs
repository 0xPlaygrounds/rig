//! Cassette-backed coverage of Venice's `venice_parameters` request block.
//!
//! These are the tests that pin Venice's own dialect: that the block reaches
//! the wire in the shape [`VeniceParameters`] serializes (the cassette matches
//! outbound request bodies, so a serialization regression fails as a mock
//! miss), and that what Venice sends back — the resolved echo, including web
//! search citations — survives into `raw_completion` instead of being dropped
//! by the OpenAI-shaped decode.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::venice::{VeniceParameters, WebSearchMode};

use super::super::{DEFAULT_MODEL, support::with_venice_cassette};

#[tokio::test]
async fn web_search_on_returns_citations() {
    with_venice_cassette("venice_parameters/web_search_on", |client| async move {
        let model = client.completion_model(DEFAULT_MODEL);
        let request = model
            .completion_request("In one sentence, what is the Rust programming language?")
            .max_tokens(64)
            .additional_params(
                VeniceParameters::new()
                    .enable_web_search(WebSearchMode::On)
                    .enable_web_citations(true)
                    .disable_thinking(true)
                    .into_additional_params(),
            )
            .build();

        let response = model
            .raw_completion(request)
            .await
            .expect("web-search completion should succeed");

        let parameters = response
            .venice_parameters
            .as_ref()
            .expect("Venice echoes its resolved parameters");
        assert_eq!(
            parameters.parameters.enable_web_search,
            Some(WebSearchMode::On),
            "Venice should report the web-search mode it applied"
        );
        assert!(
            !response.web_search_citations().is_empty(),
            "web search with citations enabled should return sources"
        );
        let citation = &response.web_search_citations()[0];
        assert!(
            citation.url.starts_with("http"),
            "citation should carry a source URL, got {:?}",
            citation.url
        );
    })
    .await;
}

#[tokio::test]
async fn web_search_auto_is_echoed() {
    with_venice_cassette("venice_parameters/web_search_auto", |client| async move {
        let model = client.completion_model(DEFAULT_MODEL);
        let request = model
            .completion_request("What is 2 + 2? Answer with the number only.")
            .max_tokens(16)
            .additional_params(
                VeniceParameters::new()
                    .enable_web_search(WebSearchMode::Auto)
                    .disable_thinking(true)
                    .into_additional_params(),
            )
            .build();

        let response = model
            .raw_completion(request)
            .await
            .expect("auto web-search completion should succeed");

        assert_eq!(
            response
                .venice_parameters
                .expect("venice parameters echo")
                .parameters
                .enable_web_search,
            Some(WebSearchMode::Auto)
        );
    })
    .await;
}

/// `disable_thinking` is how callers turn a reasoning model into a plain one;
/// the echo is the only place Venice confirms it took effect.
#[tokio::test]
async fn disable_thinking_is_applied() {
    with_venice_cassette("venice_parameters/disable_thinking", |client| async move {
        let model = client.completion_model(DEFAULT_MODEL);
        let request = model
            .completion_request("Name one primary color. Answer with one word.")
            .max_tokens(16)
            .additional_params(
                VeniceParameters::new()
                    .disable_thinking(true)
                    .strip_thinking_response(true)
                    .into_additional_params(),
            )
            .build();

        let response = model
            .raw_completion(request)
            .await
            .expect("completion should succeed");

        let echo = response
            .venice_parameters
            .expect("venice parameters echo")
            .parameters;
        assert_eq!(echo.disable_thinking, Some(true));
        assert_eq!(echo.strip_thinking_response, Some(true));
    })
    .await;
}

/// Venice injects its own system prompt by default; opting out is visible in
/// the echo and is what callers use to control the model's persona.
#[tokio::test]
async fn venice_system_prompt_can_be_disabled() {
    with_venice_cassette(
        "venice_parameters/include_venice_system_prompt_false",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let request = model
                .completion_request("Say hi in three words.")
                .max_tokens(24)
                .additional_params(
                    VeniceParameters::new()
                        .include_venice_system_prompt(false)
                        .disable_thinking(true)
                        .into_additional_params(),
                )
                .build();

            let response = model
                .raw_completion(request)
                .await
                .expect("completion should succeed");

            assert_eq!(
                response
                    .venice_parameters
                    .expect("venice parameters echo")
                    .parameters
                    .include_venice_system_prompt,
                Some(false)
            );
            assert!(
                response.cost.is_some(),
                "Venice reports per-request cost alongside usage"
            );
        },
    )
    .await;
}

/// Characters are Venice-hosted personas selected by slug; the request must
/// carry the slug and the response must echo it back.
#[tokio::test]
async fn character_slug_selects_a_persona() {
    with_venice_cassette("venice_parameters/character_slug", |client| async move {
        let model = client.completion_model(DEFAULT_MODEL);
        let request = model
            .completion_request("Introduce yourself in one sentence.")
            .max_tokens(64)
            .additional_params(
                VeniceParameters::new()
                    .character_slug("alan-watts")
                    .disable_thinking(true)
                    .into_additional_params(),
            )
            .build();

        let response = model
            .raw_completion(request)
            .await
            .expect("character completion should succeed");

        assert_eq!(
            response
                .venice_parameters
                .expect("venice parameters echo")
                .parameters
                .character_slug
                .as_deref(),
            Some("alan-watts")
        );
    })
    .await;
}
