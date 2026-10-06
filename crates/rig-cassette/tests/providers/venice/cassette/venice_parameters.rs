//! Cassette-backed coverage of Venice's `venice_parameters` request block.
//!
//! These are the tests that pin Venice's own dialect: that the block reaches
//! the wire as the caller spells it (the cassette matches
//! outbound request bodies, so a serialization regression fails as a mock
//! miss), and that what Venice sends back — the resolved echo, including web
//! search citations — survives onto
//! [`rig::completion::CompletionResponse::raw`] instead of being dropped by
//! the OpenAI-shaped decode. `raw` is the provider's verbatim reply document
//! and the only route to the echo; the web search citations also cite the
//! answer text and the per-request `cost` is the usage's cost.

use serde_json::{Value, json};

use super::super::{DEFAULT_MODEL, support::with_venice_cassette};
use rig::completion::CompletionRequest;

/// The parameters Venice echoes as resolved, read out of the captured
/// document.
fn venice_echo(raw: &Value) -> &Value {
    raw.get("venice_parameters")
        .expect("Venice echoes its resolved parameters")
}

#[tokio::test]
async fn web_search_on_returns_citations() {
    with_venice_cassette("venice_parameters/web_search_on", |client| async move {
        let model = client.completion(DEFAULT_MODEL);
        let request =
            CompletionRequest::new("In one sentence, what is the Rust programming language?")
                .max_tokens(64)
                .additional_params(json!({"venice_parameters": {
                    "enable_web_search": "on",
                    "enable_web_citations": true,
                    "disable_thinking": true,
                }}));

        let response = model
            .call(request)
            .await
            .expect("web-search completion should succeed");
        let parameters = venice_echo(&response.raw);
        assert_eq!(
            parameters["enable_web_search"], "on",
            "Venice should report the web-search mode it applied"
        );
        let citations = response.raw["venice_parameters"]["web_search_citations"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(
            !citations.is_empty(),
            "web search with citations enabled should return sources"
        );
        let url = citations[0]["url"].as_str().unwrap_or_default();
        assert!(
            url.starts_with("http"),
            "citation should carry a source URL, got {url:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn web_search_auto_is_echoed() {
    with_venice_cassette("venice_parameters/web_search_auto", |client| async move {
        let model = client.completion(DEFAULT_MODEL);
        let request = CompletionRequest::new("What is 2 + 2? Answer with the number only.")
            .max_tokens(16)
            .additional_params(json!({"venice_parameters": {
                "enable_web_search": "auto",
                "disable_thinking": true,
            }}));

        let response = model
            .call(request)
            .await
            .expect("auto web-search completion should succeed");

        assert_eq!(venice_echo(&response.raw)["enable_web_search"], "auto");
    })
    .await;
}

/// `disable_thinking` is how callers turn a reasoning model into a plain one;
/// the echo is the only place Venice confirms it took effect.
#[tokio::test]
async fn disable_thinking_is_applied() {
    with_venice_cassette("venice_parameters/disable_thinking", |client| async move {
        let model = client.completion(DEFAULT_MODEL);
        let request = CompletionRequest::new("Name one primary color. Answer with one word.")
            .max_tokens(16)
            .additional_params(json!({"venice_parameters": {
                "disable_thinking": true,
                "strip_thinking_response": true,
            }}));

        let response = model
            .call(request)
            .await
            .expect("completion should succeed");

        let echo = venice_echo(&response.raw);
        assert_eq!(echo["disable_thinking"], true);
        assert_eq!(echo["strip_thinking_response"], true);
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
            let model = client.completion(DEFAULT_MODEL);
            let request = CompletionRequest::new("Say hi in three words.")
                .max_tokens(24)
                .additional_params(json!({"venice_parameters": {
                    "include_venice_system_prompt": false,
                    "disable_thinking": true,
                }}));

            let response = model
                .call(request)
                .await
                .expect("completion should succeed");
            assert_eq!(
                venice_echo(&response.raw)["include_venice_system_prompt"],
                false
            );
            assert!(
                response.raw["cost"].is_object(),
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
        let model = client.completion(DEFAULT_MODEL);
        let request = CompletionRequest::new("Introduce yourself in one sentence.")
            .max_tokens(64)
            .additional_params(json!({"venice_parameters": {
                "character_slug": "alan-watts",
                "disable_thinking": true,
            }}));

        let response = model
            .call(request)
            .await
            .expect("character completion should succeed");

        assert_eq!(venice_echo(&response.raw)["character_slug"], "alan-watts");
    })
    .await;
}
