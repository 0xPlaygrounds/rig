//! Prompt caching over long live runs on gpt-6-sol, over Chat Completions
//! and Responses, recorded and replayed.
//!
//! The figures and the shared checks live in
//! `rig_test_support::cache_longrun`. OpenAI caches automatically: gpt-6-sol
//! writes the prefix up to the latest eligible message at 1.25 times the
//! input price and reads it at a tenth. Every cached run sends one fixed
//! `prompt_cache_key`, asserts the key never changes, caps cache writes
//! relative to prompt tokens, keeps every call from the first hit on above a
//! cached-share floor, and must save a share of input cost at OpenAI's
//! rates, writes included. Rig's OpenAI path reads no time, so these runs
//! have no clock sidecar.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `OPENAI_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::{CacheRates, Message};
use rig_test_support::cache_longrun::{
    self, CacheWire, Limits, LongRun, LookupOrder, RunLog, SUPPORT_PREAMBLE, chat, question,
};
use serde_json::{Value, json};

use crate::cassettes::CassetteClock;

use super::super::support::with_openai_long_run_cassette;

const MODEL: &str = "gpt-6-sol";
/// The one key every cached run sends.
const CACHE_KEY: &str = "rig-long-run-caching";
/// gpt-6-sol standard prices, USD per 1M tokens: input, cached input and
/// cache writes. OpenAI bills no storage.
const RATES: CacheRates = CacheRates {
    input: 2.0,
    cached_read: 0.20,
    cache_write: 2.50,
    storage_per_hour: 0.0,
};
/// gpt-6-sol standard output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 10.0;
/// What every cached run asserts.
const LIMITS: Limits = Limits {
    min_saving: Some(0.60),
    min_call_share: Some(0.70),
    max_writes_share: Some(0.25),
};

fn support_agent(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    params: Value,
) -> Agent {
    AgentBuilder::new(model)
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .max_tokens(800)
        .additional_params(params)
        .default_max_turns(4)
        .build()
}

async fn support_chat(agent: &Agent, clock: &CassetteClock, turns: usize) -> RunLog {
    let mut history: Vec<Message> = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=turns {
        chat(agent, clock, question(turn, "A"), &mut history, &mut log).await;
    }
    log
}

fn check(wire: CacheWire, scenario: &str, limits: Option<Limits>, log: &RunLog) {
    let run = LongRun {
        wire,
        scenario,
        model: MODEL,
        rates: RATES,
        output_price: OUTPUT_PRICE,
        limits,
        drops_signatures: false,
        conversation: None,
    };
    cache_longrun::check(&run, log, None);
}

/// Chat Completions on gpt-6-sol accepts function tools only without
/// reasoning.
#[tokio::test]
async fn support_chat_100_chat() {
    let log = with_openai_long_run_cassette(
        "long_run_caching/support_chat_100_chat",
        |client, clock| async move {
            let agent = support_agent(
                client.chat.completion(MODEL),
                json!({ "prompt_cache_key": CACHE_KEY, "reasoning_effort": "none" }),
            );
            support_chat(&agent, &clock, 100).await
        },
    )
    .await;
    check(
        CacheWire::OpenAiChat,
        "long_run_caching/support_chat_100_chat",
        Some(LIMITS),
        &log,
    );
}

/// The same chat over Chat Completions with caching off (explicit-only
/// mode with no breakpoints), reported only.
#[tokio::test]
async fn support_chat_30_baseline() {
    let log = with_openai_long_run_cassette(
        "long_run_caching/support_chat_30_baseline",
        |client, clock| async move {
            let agent = support_agent(
                client.chat.completion(MODEL),
                json!({
                    "prompt_cache_options": { "mode": "explicit" },
                    "reasoning_effort": "none",
                }),
            );
            support_chat(&agent, &clock, 30).await
        },
    )
    .await;
    check(
        CacheWire::OpenAiChat,
        "long_run_caching/support_chat_30_baseline",
        None,
        &log,
    );
}
