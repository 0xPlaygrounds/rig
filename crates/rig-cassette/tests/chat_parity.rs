//! Response parity: every recorded 200-status chat-completions reply, decoded
//! through the chat wire, against the committed snapshot in
//! `fixtures/parity/`. Run with `RIG_REGENERATE_PARITY=1` to rewrite it.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

#[path = "common/chat_parity.rs"]
mod chat_parity;

macro_rules! parity {
    ($($provider:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $provider() {
                chat_parity::check(stringify!($provider)).await;
            }
        )*
    };
}

parity!(
    copilot, deepseek, doubleword, groq, llamacpp, mistral, mistralrs, openai, openrouter,
    perplexity, venice,
);

/// The same recorded frames fold to the same response through `call` and
/// through `stream().finish()`, on every recorded reply of every provider.
#[tokio::test]
async fn call_and_stream_finish_agree_on_every_recorded_reply() {
    let mut compared = 0;
    for provider in chat_parity::PROVIDERS {
        for interaction in chat_parity::interactions(provider) {
            let (called, streamed) = chat_parity::both_paths(provider, &interaction).await;
            assert_eq!(
                chat_parity::project(interaction.streaming, &called),
                chat_parity::project(interaction.streaming, &streamed),
                "{provider} {}: call and stream().finish() differ",
                interaction.key
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "no recorded replies");
}

/// Perplexity never sends `[DONE]`: a stream that ends after its finish
/// reason is whole, as it was before the decoder was rewritten.
#[tokio::test]
async fn perplexity_streams_without_done_match_the_snapshot() {
    let expected = chat_parity::snapshot("perplexity");
    let without_done: Vec<_> = chat_parity::interactions("perplexity")
        .into_iter()
        .filter(|interaction| interaction.streaming && !interaction.body.contains("[DONE]"))
        .collect();
    assert!(
        !without_done.is_empty(),
        "no Perplexity stream without [DONE]"
    );
    for interaction in &without_done {
        let outcome = chat_parity::decode("perplexity", interaction).await;
        assert!(outcome.is_ok(), "{}: {outcome:?}", interaction.key);
        assert_eq!(
            Some(&chat_parity::project(true, &outcome)),
            expected.get(&interaction.key),
            "{}",
            interaction.key
        );
    }
}
