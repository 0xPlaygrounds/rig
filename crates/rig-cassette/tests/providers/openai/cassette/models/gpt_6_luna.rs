//! `gpt-6-luna`'s recorded session, on Responses and Chat Completions in one
//! recording: see `rig_test_support::model_session`. Chat Completions takes its
//! function tools only at the caller's `reasoning_effort: "none"`.

use rig::completion::CacheRates;
use rig::providers::openai::GPT_6_LUNA;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, ChatSupport, OpenAiProfile};

use super::super::super::support::with_openai_model_session_cassette;

/// `gpt-6-luna` standard prices, USD per 1M tokens: input, cached input and cache writes (OpenAI pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 0.1,
    cached_read: 0.01,
    cache_write: 0.125,
    storage_per_hour: 0.0,
};
/// `gpt-6-luna` output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 0.5;

#[tokio::test]
async fn session() {
    let session = with_openai_model_session_cassette(
        "models/gpt_6_luna/session",
        |client, clock| async move {
            model_session::openai(
                client.openai,
                client.chat,
                clock,
                &OpenAiProfile {
                    model: GPT_6_LUNA,
                    takes_temperature: false,
                    chat: ChatSupport::ToolsAtEffortNone,
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
        "models/gpt_6_luna/session",
        GPT_6_LUNA,
        RATES,
        OUTPUT_PRICE,
        &session,
        true,
    );
}
