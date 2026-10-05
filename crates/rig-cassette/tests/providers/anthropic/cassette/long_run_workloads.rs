//! Prompt caching over long live runs of realistic workloads on Claude Opus
//! 5.5, recorded and replayed: unary and streamed turns in one conversation, a
//! tool-heavy agent loop, and a document reused every turn.
//!
//! The workloads, figures and shared checks live in
//! `rig_test_support::cache_longrun`. Every run uses automatic caching, so
//! the cache marker moves forward each call.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `ANTHROPIC_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::CacheRates;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::CLAUDE_OPUS_5_5;
use rig_test_support::cache_longrun::workloads::{self, DOCUMENT_PREAMBLE};
use rig_test_support::cache_longrun::{
    self, CacheWire, Figures, Limits, LongRun, LookupOrder, Recording, RunLog, SUPPORT_PREAMBLE,
};
use rig_test_support::cassette_models::{AnthropicModels, MapWire};

use super::super::support::with_anthropic_long_run_cassette;

/// Claude Opus 5.5 prices, USD per 1M tokens: input, cache hit and a write
/// to the 5-minute cache (pricing page, 2026-09-29).
pub(super) const OPUS_5_5_RATES: CacheRates = CacheRates {
    input: 4.0,
    cached_read: 0.20,
    cache_write: 5.0,
    storage_per_hour: 0.0,
};
/// Claude Opus 5.5 output price.
pub(super) const OPUS_5_5_OUTPUT: f64 = 20.0;

/// `model` with automatic caching.
pub(super) fn cached(
    models: &AnthropicModels,
    model: &str,
) -> rig::DynModel<rig::operation::Completion> {
    models
        .completion(model)
        .map_wire(anthropic::Messages::with_automatic_caching)
        .into()
}

pub(super) fn support_agent(model: rig::DynModel<rig::operation::Completion>) -> Agent {
    AgentBuilder::new(model)
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .max_tokens(800)
        .default_max_turns(4)
        .build()
}

pub(super) fn check(
    scenario: &str,
    model: &str,
    rates: CacheRates,
    output_price: f64,
    limits: Option<Limits>,
    log: &RunLog,
) -> (Recording, Figures) {
    cache_longrun::check(
        &LongRun {
            wire: CacheWire::Anthropic,
            scenario,
            model,
            rates,
            output_price,
            limits,
            drops_signatures: false,
            conversation: None,
        },
        log,
        None,
    )
}

/// Sixty questions about one store policy attached, with citations, on the
/// first turn.
#[tokio::test]
async fn document_60() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/document_60",
        |models, clock| async move {
            let agent = AgentBuilder::new(cached(&models, CLAUDE_OPUS_5_5))
                .preamble(DOCUMENT_PREAMBLE)
                .max_tokens(800)
                .build();
            workloads::document_session(
                &agent,
                &clock,
                60,
                "DOCUMENT SESSION",
                workloads::policy_document(true),
            )
            .await
        },
    )
    .await;
    check(
        "long_run_caching/document_60",
        CLAUDE_OPUS_5_5,
        OPUS_5_5_RATES,
        OPUS_5_5_OUTPUT,
        Some(Limits {
            min_saving: Some(0.60),
            min_call_share: Some(0.80),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
}
