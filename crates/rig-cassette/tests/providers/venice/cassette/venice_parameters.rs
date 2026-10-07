//! Cassette-backed coverage of Venice's `venice_parameters` request block.
//!
//! These are the tests that pin Venice's own dialect: that the block reaches
//! the wire as the caller spells it (the cassette matches
//! outbound request bodies, so a serialization regression fails as a mock
//! miss), and that what Venice sends back — the resolved echo, including web
//! search citations — survives into the typed `VeniceExtras` instead of
//! being dropped by the OpenAI-shaped decode. The web search citations also
//! cite the answer text and the per-request `cost` is the usage's cost.
//! `disable_thinking` has no typed field, so it rides `additional_params`
//! and deep-merges into the typed block.

use rig::completion::{CompletionRequest, CompletionResponse, ProviderOptions};
use rig::providers::venice::extension::{
    VeniceExt, VeniceExtras, VeniceOptions, VeniceParameters, VeniceParametersEcho, WebSearchMode,
};
use serde_json::json;

use super::super::{DEFAULT_MODEL, support::with_venice_cassette};

/// A request carrying `parameters` with thinking disabled.
fn request(prompt: &str, max_tokens: u64, parameters: VeniceParameters) -> CompletionRequest {
    let options = ProviderOptions::new().set(VeniceOptions::new().venice_parameters(parameters));
    CompletionRequest::new(prompt)
        .max_tokens(max_tokens)
        .provider_options(options)
        .additional_params(json!({"venice_parameters": {"disable_thinking": true}}))
}

fn extras(response: &CompletionResponse) -> VeniceExtras {
    response
        .extras::<VeniceExt>()
        .expect("a Venice reply")
        .expect("Venice extras decode")
}

/// The parameters Venice echoes as resolved.
fn venice_echo(response: &CompletionResponse) -> VeniceParametersEcho {
    extras(response)
        .venice_parameters
        .expect("Venice echoes its resolved parameters")
}

#[tokio::test]
async fn web_search_on_returns_citations() {
    with_venice_cassette("venice_parameters/web_search_on", |client| async move {
        let model = client.completion(DEFAULT_MODEL);
        let request = request(
            "In one sentence, what is the Rust programming language?",
            64,
            VeniceParameters::new()
                .enable_web_search(WebSearchMode::On)
                .enable_web_citations(true),
        );

        let response = model
            .call(request)
            .await
            .expect("web-search completion should succeed");
        let parameters = venice_echo(&response);
        assert_eq!(
            parameters.enable_web_search,
            Some(WebSearchMode::On),
            "Venice should report the web-search mode it applied"
        );
        let citations = parameters.web_search_citations.unwrap_or_default();
        assert!(
            !citations.is_empty(),
            "web search with citations enabled should return sources"
        );
        let url = citations[0].url.as_deref().unwrap_or_default();
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
        let request = request(
            "What is 2 + 2? Answer with the number only.",
            16,
            VeniceParameters::new().enable_web_search(WebSearchMode::Auto),
        );

        let response = model
            .call(request)
            .await
            .expect("auto web-search completion should succeed");

        assert_eq!(
            venice_echo(&response).enable_web_search,
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
        let model = client.completion(DEFAULT_MODEL);
        let request = request(
            "Name one primary color. Answer with one word.",
            16,
            VeniceParameters::new().strip_thinking_response(true),
        );

        let response = model
            .call(request)
            .await
            .expect("completion should succeed");

        let echo = venice_echo(&response);
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
            let model = client.completion(DEFAULT_MODEL);
            let request = request(
                "Say hi in three words.",
                24,
                VeniceParameters::new().include_venice_system_prompt(false),
            );

            let response = model
                .call(request)
                .await
                .expect("completion should succeed");
            assert_eq!(
                venice_echo(&response).include_venice_system_prompt,
                Some(false)
            );
            assert!(
                extras(&response).cost.is_some(),
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
        let request = request(
            "Introduce yourself in one sentence.",
            64,
            VeniceParameters::new().character_slug("alan-watts"),
        );

        let response = model
            .call(request)
            .await
            .expect("character completion should succeed");

        assert_eq!(
            venice_echo(&response).character_slug.as_deref(),
            Some("alan-watts")
        );
    })
    .await;
}
