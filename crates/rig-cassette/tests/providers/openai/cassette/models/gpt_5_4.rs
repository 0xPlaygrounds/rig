//! `gpt-5.4`'s recorded session, on Responses and Chat Completions in one
//! recording: see `rig_test_support::model_session`. Chat Completions takes its
//! function tools at the default effort.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_5_4;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-5.4` standard prices, USD per 1M tokens: input, cached input; no cache-write charge, so writes are the input price (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 2.5,
    cached_read: 0.25,
    cache_write: 2.5,
    storage_per_hour: 0.0,
};
/// `gpt-5.4` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 15.0;

#[tokio::test]
async fn session() {
    let session =
        with_openai_model_session_cassette("models/gpt_5_4/session", |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_5_4,
                    takes_temperature: true,
                    chat: ChatSupport::Tools,
                    pro: false,
                    structured_outputs: true,
                    web_search: true,
                },
            )
            .await
        })
        .await;
    // OpenAI's cache sometimes misses a byte-identical prefix on this model:
    // the rehearsal (2026-09-29) sent every main-conversation request as an
    // exact extension of the one before, other keys equal but `stream`, and
    // calls 4 and 12 (unary after streamed) read 0 while the calls around
    // them read 1,536 to 8,960, as on gpt-5.4-mini and gpt-5.4-nano; the
    // GPT-6 models, same session, never missed. So the run asserts the usage
    // and reports the figures, but not that every call after the first read
    // reads again.
    model_session::check_main(
        CacheWire::OpenAiResponses,
        "models/gpt_5_4/session",
        GPT_5_4,
        RATES,
        OUTPUT_PRICE,
        &session,
        false,
    );
}
