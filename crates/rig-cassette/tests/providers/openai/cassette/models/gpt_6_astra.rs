//! `gpt-6-astra`'s recorded session, on Responses and Chat Completions in one
//! recording: see `rig_test_support::model_session`. Chat Completions takes it
//! without function tools only, so tools and the extractor there are asserted
//! refusals.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_6_ASTRA;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-6-astra` standard prices, USD per 1M tokens: input, cached input and cache writes (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 10.0,
    cached_read: 1.0,
    cache_write: 12.5,
    storage_per_hour: 0.0,
};
/// `gpt-6-astra` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 50.0;

#[tokio::test]
async fn session() {
    let session = with_openai_model_session_cassette(
        "models/gpt_6_astra/session",
        |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_6_ASTRA,
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
        "models/gpt_6_astra/session",
        GPT_6_ASTRA,
        RATES,
        OUTPUT_PRICE,
        &session,
        true,
    );
}
