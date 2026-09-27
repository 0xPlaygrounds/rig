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
