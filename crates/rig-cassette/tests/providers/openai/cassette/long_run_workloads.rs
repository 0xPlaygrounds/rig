//! Prompt caching over long live runs of realistic workloads on the GPT-6
//! models, over Responses, recorded and replayed: unary and streamed turns in
//! one conversation, a tool-heavy agent loop, and a document reused every
//! turn.
//!
//! The workloads, figures and shared checks live in
//! `rig_test_support::cache_longrun`. Every run is stateless (`store:
//! false`), reasons at low effort so encrypted reasoning rides the history,
//! and sends one fixed `prompt_cache_key`. The runs are Responses-only: the
//! GPT-6 models take function tools on Chat Completions only at
//! `reasoning_effort: "none"`, and gpt-6-astra and gpt-6.1-sol not at all.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `OPENAI_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::CacheRates;
use rig::providers::openai::{GPT_6_ASTRA, GPT_6_LUNA, GPT_6_SOL};
use rig_test_support::cache_longrun::workloads::{
    self, DOCUMENT_PREAMBLE, OrderHistory, ShippingLog,
};
use rig_test_support::cache_longrun::{
    self, CacheWire, Figures, Limits, LongRun, LookupOrder, Recording, RunLog, SUPPORT_PREAMBLE,
};
use serde_json::{Value, json};

use super::super::support::with_openai_long_run_cassette;

/// The one key every run sends.
const CACHE_KEY: &str = "rig-long-run-workloads";

/// Standard prices, USD per 1M tokens: input, cached input and cache writes
/// (pricing page, 2026-09-29), and output.
const ASTRA: (CacheRates, f64) = (
    CacheRates {
        input: 10.0,
        cached_read: 1.0,
        cache_write: 12.5,
        storage_per_hour: 0.0,
    },
    50.0,
);
const SOL: (CacheRates, f64) = (
    CacheRates {
        input: 2.0,
        cached_read: 0.20,
        cache_write: 2.5,
        storage_per_hour: 0.0,
    },
    10.0,
);
pub(super) const LUNA: (CacheRates, f64) = (
    CacheRates {
        input: 0.1,
        cached_read: 0.01,
        cache_write: 0.125,
        storage_per_hour: 0.0,
    },
    0.5,
);

/// Stateless, low effort, one cache key.
fn params() -> Value {
    json!({
        "prompt_cache_key": CACHE_KEY,
        "reasoning": { "effort": "low" },
        "store": false,
    })
}

pub(super) fn support_agent(model: rig::DynModel<rig::operation::Completion>) -> Agent {
    AgentBuilder::new(model)
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .max_tokens(800)
        .additional_params(params())
        .default_max_turns(4)
        .build()
}

pub(super) fn check(
    scenario: &str,
    model: &str,
    (rates, output_price): (CacheRates, f64),
    limits: Option<Limits>,
    log: &RunLog,
) -> (Recording, Figures) {
    cache_longrun::check(
        &LongRun {
            wire: CacheWire::OpenAiResponses,
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

/// The same on gpt-6-astra, kept to thirty turns for its price.
#[tokio::test]
async fn mixed_delivery_30_responses() {
    let log = with_openai_long_run_cassette(
        "long_run_caching/mixed_delivery_30_responses",
        |client, clock| async move {
            let agent = support_agent(client.openai.completion(GPT_6_ASTRA).into());
            workloads::mixed_delivery(&agent, &clock, 30, "MIXED DELIVERY", "A").await
        },
    )
    .await;
    check(
        "long_run_caching/mixed_delivery_30_responses",
        GPT_6_ASTRA,
        ASTRA,
        Some(Limits {
            min_saving: Some(0.50),
            min_call_share: Some(0.70),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
}

/// Twenty account investigations: sixty calls, each turn two histories at
/// once, then a carrier log, then the answer, with about a thousand tokens
/// per tool result.
#[tokio::test]
async fn tool_loop_60_responses() {
    let (log, turns) = with_openai_long_run_cassette(
        "long_run_caching/tool_loop_60_responses",
        |client, clock| async move {
            let agent = AgentBuilder::new(client.openai.completion(GPT_6_LUNA))
                .preamble(workloads::tool_loop_preamble())
                .tool(OrderHistory)
                .tool(ShippingLog)
                .max_tokens(1500)
                .additional_params(params())
                .default_max_turns(6)
                .build();
            workloads::tool_loop(&agent, &clock, 20, "TOOL LOOP").await
        },
    )
    .await;
    assert_eq!(turns.len(), 20);
    check(
        "long_run_caching/tool_loop_60_responses",
        GPT_6_LUNA,
        LUNA,
        Some(Limits {
            min_saving: Some(0.50),
            // The second call adds two histories of about a thousand
            // tokens each to a prompt of under three thousand, so it can read
            // well under half of its prompt; the first recording read 38%.
            min_call_share: Some(0.30),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
}

/// Sixty questions about one store policy attached on the first turn.
#[tokio::test]
async fn document_60_responses() {
    let log = with_openai_long_run_cassette(
        "long_run_caching/document_60_responses",
        |client, clock| async move {
            let agent = AgentBuilder::new(client.openai.completion(GPT_6_SOL))
                .preamble(DOCUMENT_PREAMBLE)
                .max_tokens(800)
                .additional_params(params())
                .build();
            workloads::document_session(
                &agent,
                &clock,
                60,
                "DOCUMENT SESSION",
                workloads::policy_document(false),
            )
            .await
        },
    )
    .await;
    check(
        "long_run_caching/document_60_responses",
        GPT_6_SOL,
        SOL,
        Some(Limits {
            min_saving: Some(0.60),
            min_call_share: Some(0.80),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
}
