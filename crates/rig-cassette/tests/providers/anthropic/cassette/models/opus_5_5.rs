//! Claude Opus 5.5's recorded session. Everything the model documents and rig supports without a
//! model-specific setting, in one recording: see `rig_test_support::model_session`.

use rig::completion::CacheRates;
use rig::providers::anthropic::completion::CLAUDE_OPUS_5_5;
use rig_test_support::cache_longrun::CacheWire;
use rig_test_support::model_session::{self, AnthropicProfile};

use super::super::super::support::with_anthropic_model_session_cassette;

/// Claude Opus 5.5 prices, USD per 1M tokens: input, cache hit and 5-minute cache
/// write (Anthropic pricing page, 2026-09-29).
const RATES: CacheRates = CacheRates {
    input: 4.0,
    cached_read: 0.2,
    cache_write: 5.0,
    storage_per_hour: 0.0,
};
/// Claude Opus 5.5 output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 20.0;
#[tokio::test]
async fn session() {
    let session = with_anthropic_model_session_cassette(
        "models/opus_5_5/session",
        true,
        |models, files, clock| async move {
            model_session::anthropic(
                models,
                files,
                clock,
                &AnthropicProfile {
                    model: CLAUDE_OPUS_5_5,
                    rejects_forced_tool_choice: true,
                    mid_conversation_system: true,
                    fixes_only: false,
                },
            )
            .await
        },
    )
    .await;
    assert!(!session.phases.is_empty());
    model_session::check_main(
        CacheWire::Anthropic,
        "models/opus_5_5/session",
        CLAUDE_OPUS_5_5,
        RATES,
        OUTPUT_PRICE,
        &session,
        true,
    );
}
