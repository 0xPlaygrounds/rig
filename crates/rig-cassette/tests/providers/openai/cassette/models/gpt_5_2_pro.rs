//! `gpt-5.2-pro`'s recorded session: see `rig_test_support::model_session`. It is served on Responses
//! only, and its page lists no structured outputs and no hosted tools.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_5_2_PRO;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-5.2-pro` standard prices, USD per 1M tokens: input, no cached or write price is listed, so both are the input price (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 21.0,
    cached_read: 21.0,
    cache_write: 21.0,
    storage_per_hour: 0.0,
};
/// `gpt-5.2-pro` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 168.0;

#[tokio::test]
async fn session() {
    let session = with_openai_model_session_cassette(
        "models/gpt_5_2_pro/session",
        |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_5_2_PRO,
                    takes_temperature: false,
                    chat: ChatSupport::None,
                    pro: true,
                    structured_outputs: false,
                    web_search: false,
                },
            )
            .await
        },
    )
    .await;
    // The pricing page lists no cached-input price for the pro models, and
    // the gpt-5.2-pro rehearsal (2026-09-29) read no cached token on any of
    // its 15 main-conversation calls: the provider does not cache them. So
    // the run asserts the usage and reports the figures (all uncached).
    model_session::check_main(
        CacheWire::OpenAiResponses,
        "models/gpt_5_2_pro/session",
        GPT_5_2_PRO,
        RATES,
        OUTPUT_PRICE,
        &session,
        false,
    );
}
