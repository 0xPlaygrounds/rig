//! Venice's options as the bodies they encode to, and its extras read from
//! a recorded reply. The encoder tests are unit tests because no recording
//! sends typed provider options.

use serde_json::{Value, json};

use super::*;
use crate::providers::openai::wire::{Chat, OpenAIConfig, VENICE};
use crate::test_utils::provider_extensions::{
    assert_no_reserved_leaf, body_with, recorded_reply, reply_of,
};

fn chat_wire() -> Chat {
    OpenAIConfig::with_key(&VENICE, "key").chat("qwen3-5-9b")
}

fn parameters(parameters: VeniceParameters) -> Value {
    body_with::<Venice, _>(
        &chat_wire(),
        &VeniceOptions::new().venice_parameters(parameters),
    )["venice_parameters"]
        .clone()
}

#[test]
fn character_slug_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().character_slug("alan-watts"));
    assert_eq!(sent, json!({"character_slug": "alan-watts"}));
}

#[test]
fn strip_thinking_response_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().strip_thinking_response(true));
    assert_eq!(sent, json!({"strip_thinking_response": true}));
}

#[test]
fn enable_web_search_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().enable_web_search(WebSearchMode::On));
    assert_eq!(sent, json!({"enable_web_search": "on"}));
}

#[test]
fn enable_web_scraping_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().enable_web_scraping(true));
    assert_eq!(sent, json!({"enable_web_scraping": true}));
}

#[test]
fn enable_x_search_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().enable_x_search(true));
    assert_eq!(sent, json!({"enable_x_search": true}));
}

#[test]
fn enable_web_citations_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().enable_web_citations(true));
    assert_eq!(sent, json!({"enable_web_citations": true}));
}

#[test]
fn include_search_results_in_stream_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().include_search_results_in_stream(true));
    assert_eq!(sent, json!({"include_search_results_in_stream": true}));
}

#[test]
fn return_search_results_as_documents_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().return_search_results_as_documents(true));
    assert_eq!(sent, json!({"return_search_results_as_documents": true}));
}

#[test]
fn include_venice_system_prompt_lands_under_venice_parameters() {
    let sent = parameters(VeniceParameters::new().include_venice_system_prompt(false));
    assert_eq!(sent, json!({"include_venice_system_prompt": false}));
}

#[test]
fn prompt_cache_key_lands_at_top_level() {
    let body = body_with::<Venice, _>(&chat_wire(), &VeniceOptions::new().prompt_cache_key("k1"));
    assert_eq!(body["prompt_cache_key"], "k1");
}

#[test]
fn no_option_writes_a_leaf_the_request_or_a_mapped_option_owns() {
    let options = VeniceOptions::new()
        .prompt_cache_key("k")
        .venice_parameters(
            VeniceParameters::new()
                .character_slug("c")
                .strip_thinking_response(true)
                .enable_web_search(WebSearchMode::Auto)
                .enable_web_scraping(true)
                .enable_x_search(true)
                .enable_web_citations(true)
                .include_search_results_in_stream(true)
                .return_search_results_as_documents(true)
                .include_venice_system_prompt(true),
        );
    assert_no_reserved_leaf::<Venice, _>(&[chat_wire()], &options);
}

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        chat_wire(),
        recorded_reply("venice", "agent/completion_smoke", 0),
    )
    .await;
    let extras = reply
        .extras::<Venice>()
        .unwrap_or_else(|| panic!("a Venice reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let cost = extras.cost.unwrap_or_default();
    assert_eq!(cost.usd, Some(0.0002562));
    assert_eq!(cost.diem, Some(0.0));
    let echo = extras.venice_parameters.unwrap_or_default();
    assert_eq!(echo.enable_e2ee, Some(true));
    assert_eq!(echo.include_venice_system_prompt, Some(true));
    assert_eq!(echo.enable_web_search, Some(WebSearchMode::Off));
    assert_eq!(echo.disable_thinking, Some(false));
    assert_eq!(echo.web_search_citations, Some(Vec::new()));
}
