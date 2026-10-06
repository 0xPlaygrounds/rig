//! DeepSeek's extras read from a recorded unary reply, and its empty
//! options. The options test is a unit test because DeepSeek has no typed
//! option to record.

use super::*;
use crate::completion::{CompletionRequest, ProviderOptions};
use crate::providers::openai::wire::{DEEPSEEK, OpenAIConfig};
use crate::providers::openrouter::extension::OpenRouter;
use crate::test_utils::provider_extensions::{encoded_body, recorded_reply, reply_of};
use crate::wire::Mode;

const MODEL: &str = crate::providers::deepseek::DEEPSEEK_FLASH;

#[test]
fn empty_options_change_nothing() {
    let wire = OpenAIConfig::with_key(&DEEPSEEK, "key").chat(MODEL);
    let options = ProviderOptions::new()
        .with::<DeepSeek>(&DeepSeekOptions::new())
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(options.is_empty());
    let with = encoded_body(
        &wire,
        CompletionRequest::new("hi").provider_options(options),
        Mode::Unary,
    );
    let without = encoded_body(&wire, CompletionRequest::new("hi"), Mode::Unary);
    assert_eq!(
        with.unwrap_or_else(|error| panic!("{error}")),
        without.unwrap_or_else(|error| panic!("{error}"))
    );
}

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        OpenAIConfig::with_key(&DEEPSEEK, "key").chat(MODEL),
        recorded_reply("deepseek", "portability_matrix/from_openai_responses", 0),
    )
    .await;
    let extras = reply
        .extras::<DeepSeek>()
        .unwrap_or_else(|| panic!("a DeepSeek reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(extras.prompt_cache_hit_tokens, Some(256));
    assert_eq!(extras.prompt_cache_miss_tokens, Some(247));
    assert_eq!(extras.reasoning_tokens, Some(25));
    assert_eq!(
        extras.system_fingerprint.as_deref(),
        Some("aeb56401ca74e127821c4f9126dcb669")
    );
    assert!(reply.extras::<OpenRouter>().is_none());
}
