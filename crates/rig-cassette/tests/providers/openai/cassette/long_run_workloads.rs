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
use rig::completion::CacheRates;
use rig::providers::openai::GPT_6_SOL;
use rig_test_support::cache_longrun::workloads::{self, DOCUMENT_PREAMBLE};
use rig_test_support::cache_longrun::{
    self, CacheWire, Figures, Limits, LongRun, Recording, RunLog,
};
use serde_json::{Value, json};

use super::super::support::with_openai_long_run_cassette;

/// The one key every run sends.
const CACHE_KEY: &str = "rig-long-run-workloads";

const SOL: (CacheRates, f64) = (
    CacheRates {
        input: 2.0,
        cached_read: 0.20,
        cache_write: 2.5,
        storage_per_hour: 0.0,
    },
    10.0,
);

/// Stateless, low effort, one cache key.
fn params() -> Value {
    json!({
        "prompt_cache_key": CACHE_KEY,
        "reasoning": { "effort": "low" },
        "store": false,
    })
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
