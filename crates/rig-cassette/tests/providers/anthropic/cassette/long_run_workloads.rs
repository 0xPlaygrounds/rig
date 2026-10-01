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
use rig_test_support::cache_longrun::workloads::{
    self, DOCUMENT_PREAMBLE, OrderHistory, ShippingLog,
};
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

/// Unary and streamed turns alternating in one conversation.
#[tokio::test]
async fn mixed_delivery_100() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/mixed_delivery_100",
        |models, clock| async move {
            let agent = support_agent(cached(&models, CLAUDE_OPUS_5_5));
            workloads::mixed_delivery(&agent, &clock, 100, "MIXED DELIVERY", "A").await
        },
    )
    .await;
    check(
        "long_run_caching/mixed_delivery_100",
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

/// Twenty account investigations: sixty calls, each turn two histories at
/// once, then a carrier log, then the answer, with about a thousand tokens
/// per tool result.
#[tokio::test]
async fn tool_loop_60() {
    let (log, turns) = with_anthropic_long_run_cassette(
        "long_run_caching/tool_loop_60",
        |models, clock| async move {
            let agent = AgentBuilder::new(cached(&models, CLAUDE_OPUS_5_5))
                .preamble(workloads::tool_loop_preamble())
                .tool(OrderHistory)
                .tool(ShippingLog)
                .max_tokens(1500)
                .default_max_turns(6)
                .build();
            workloads::tool_loop(&agent, &clock, 20, "TOOL LOOP").await
        },
    )
    .await;
    assert_eq!(turns.len(), 20);
    check(
        "long_run_caching/tool_loop_60",
        CLAUDE_OPUS_5_5,
        OPUS_5_5_RATES,
        OPUS_5_5_OUTPUT,
        Some(Limits {
            min_saving: Some(0.60),
            // The second call adds two histories of about a thousand
            // tokens each to a prompt of under three thousand, so it can read
            // well under half of its prompt; the first recording read 38%.
            min_call_share: Some(0.30),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
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
