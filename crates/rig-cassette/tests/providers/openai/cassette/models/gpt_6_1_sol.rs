//! `gpt-6.1-sol`'s recorded session, on Responses and Chat Completions in one
//! recording: see `rig_test_support::model_session`. Chat Completions takes it
//! without function tools only, so tools and the extractor there are asserted
//! refusals.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_6_1_SOL;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-6.1-sol` standard prices, USD per 1M tokens: input, cached input and cache writes (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 2.0,
    cached_read: 0.1,
    cache_write: 2.5,
    storage_per_hour: 0.0,
};
/// `gpt-6.1-sol` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 10.0;

#[tokio::test]
async fn session() {
    let session = with_openai_model_session_cassette(
        "models/gpt_6_1_sol/session",
        |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_6_1_SOL,
                    takes_temperature: false,
                    chat: ChatSupport::TextOnly,
                    pro: false,
                    structured_outputs: true,
                    web_search: true,
                },
            )
            .await
        },
    )
    .await;
    model_session::check_main(
        CacheWire::OpenAiResponses,
        "models/gpt_6_1_sol/session",
        GPT_6_1_SOL,
        RATES,
        OUTPUT_PRICE,
        &session,
        true,
    );
}
