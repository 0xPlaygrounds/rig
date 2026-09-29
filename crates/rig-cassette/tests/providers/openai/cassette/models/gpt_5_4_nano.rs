//! `gpt-5.4-nano`'s recorded session, on Responses and Chat Completions in one
//! recording: see `rig_test_support::model_session`. Chat Completions takes its
//! function tools at the default effort.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_5_4_NANO;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-5.4-nano` standard prices, USD per 1M tokens: input, cached input; no cache-write charge, so writes are the input price (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 0.2,
    cached_read: 0.02,
    cache_write: 0.2,
    storage_per_hour: 0.0,
};
/// `gpt-5.4-nano` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 1.25;

#[tokio::test]
async fn session() {
    let session = with_openai_model_session_cassette(
        "models/gpt_5_4_nano/session",
        |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_5_4_NANO,
                    takes_temperature: true,
                    chat: ChatSupport::Tools,
                    pro: false,
                    structured_outputs: true,
                    web_search: true,
                },
            )
            .await
        },
    )
    .await;
    // OpenAI's cache misses byte-identical prefixes on this model: a raw
    // probe (no rig code) of five calls sharing a 1,763-token prefix and one
    // `prompt_cache_key`, 2 s and then 8 s apart, read 0 cached tokens on
    // every call, where gpt-6-luna and gpt-5.4-mini read from the second call
    // (2026-09-29). So the run asserts the usage and reports the figures, but
    // not that every call after the first read reads again.
    model_session::check_main(
        CacheWire::OpenAiResponses,
        "models/gpt_5_4_nano/session",
        GPT_5_4_NANO,
        RATES,
        OUTPUT_PRICE,
        &session,
        false,
    );
}
