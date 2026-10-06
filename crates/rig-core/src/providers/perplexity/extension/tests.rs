//! Perplexity's extras read from a recorded reply.

use super::*;
use crate::providers::openai::wire::{OpenAIConfig, PERPLEXITY};
use crate::providers::openrouter::extension::OpenRouter;
use crate::test_utils::provider_extensions::{recorded_reply, reply_of};

#[tokio::test]
async fn extras_from_a_unary_recording() {
    let reply = reply_of(
        OpenAIConfig::with_key(&PERPLEXITY, "key").chat(crate::providers::perplexity::SONAR),
        recorded_reply("perplexity", "agent/completion_with_perplexity_options", 0),
    )
    .await;
    let extras = reply
        .extras::<Perplexity>()
        .unwrap_or_else(|| panic!("a Perplexity reply"))
        .unwrap_or_else(|error| panic!("{error}"));
    let citations = extras.citations.unwrap_or_default();
    assert_eq!(citations.len(), 15);
    assert_eq!(
        citations.first().map(String::as_str),
        Some("https://this-week-in-rust.org/blog/2026/07/22/this-week-in-rust-661/")
    );
    let results = extras.search_results.unwrap_or_default();
    assert_eq!(results.len(), 15);
    let first = results.first().cloned().unwrap_or_default();
    assert_eq!(first.title.as_deref(), Some("This Week in Rust 661"));
    assert_eq!(first.date.as_deref(), Some("2026-07-22"));
    assert_eq!(first.last_updated.as_deref(), Some("2026-08-10"));
    assert_eq!(first.source.as_deref(), Some("web"));
    let questions = extras.related_questions.unwrap_or_default();
    assert_eq!(questions.len(), 5);
    assert_eq!(
        questions.first().map(String::as_str),
        Some("What recent Rust tooling development was highlighted in This Week in Rust?")
    );
    let cost = extras.cost.unwrap_or_default();
    assert_eq!(cost.total_cost, Some(0.00508));
    assert_eq!(cost.request_cost, Some(0.005));
    assert_eq!(cost.input_tokens_cost, Some(2e-5));
    assert_eq!(cost.output_tokens_cost, Some(5e-5));
    assert_eq!(extras.search_context_size.as_deref(), Some("low"));
    assert_eq!(extras.images, None);
    assert!(reply.extras::<OpenRouter>().is_none());
}

/// A stream's `raw` is the unary document, so the extras of the recorded
/// stream equal those of the recorded unary answer of the same prompt, its
/// citations and search results included.
#[tokio::test]
async fn extras_read_alike_from_a_recorded_stream() {
    use crate::test_utils::provider_extensions::{recorded_stream, streamed_reply_of};

    let wire =
        || OpenAIConfig::with_key(&PERPLEXITY, "key").chat(crate::providers::perplexity::SONAR);
    let read = |reply: crate::completion::CompletionResponse| {
        reply
            .extras::<Perplexity>()
            .unwrap_or_else(|| panic!("a Perplexity reply"))
            .unwrap_or_else(|error| panic!("{error}"))
    };
    let streamed = read(
        streamed_reply_of(
            wire(),
            recorded_stream(
                "perplexity",
                "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object",
                0,
            ),
        )
        .await,
    );
    let unary = read(
        reply_of(
            wire(),
            recorded_reply(
                "perplexity",
                "raw_capture_matrix/normalized_fields_match_raw_renormalized",
                0,
            ),
        )
        .await,
    );
    assert_eq!(streamed, unary);
    assert_eq!(streamed.citations.unwrap_or_default().len(), 18);
}
