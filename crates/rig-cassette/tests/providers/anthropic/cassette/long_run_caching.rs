//! Automatic prompt caching over long live runs on claude-opus-5-5, recorded
//! and replayed.
//!
//! The figures and the shared checks live in
//! `rig_test_support::cache_longrun`. Anthropic reports cache reads and
//! writes beside `input_tokens` and bills a write above the input price, so a
//! marker that stops moving forward shows as the same prefix written again
//! and again. Every cached run therefore caps cache writes relative to prompt
//! tokens, keeps every call from the first read on above a cached-share
//! floor, and must save a share of input cost at Anthropic's rates, writes
//! included. Rig's Anthropic path reads no time, so these runs have no clock
//! sidecar.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `ANTHROPIC_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::completion::{CacheRates, Message};
use rig::providers::anthropic;
use rig_test_support::cache_longrun::{
    self, CacheWire, Limits, LongRun, LookupOrder, RunLog, SUPPORT_PREAMBLE, chat, chat_streamed,
    question,
};
use rig_test_support::cassette_models::MapWire;

use crate::cassettes::CassetteClock;

use super::super::support::with_anthropic_long_run_cassette;

const MODEL: &str = "claude-opus-5-5";
/// Claude Opus 5.5 prices, USD per 1M tokens: input, cache hit, and a write
/// to the 5-minute cache. Anthropic bills no storage.
const RATES_5M: CacheRates = CacheRates {
    input: 4.0,
    cached_read: 0.20,
    cache_write: 5.0,
    storage_per_hour: 0.0,
};
/// [`RATES_5M`] with a write to the 1-hour cache.
const RATES_1H: CacheRates = CacheRates {
    cache_write: 8.0,
    ..RATES_5M
};
/// Claude Opus 5.5 output price, USD per 1M tokens.
const OUTPUT_PRICE: f64 = 20.0;
/// What every cached run asserts.
const LIMITS: Limits = Limits {
    min_saving: Some(0.60),
    min_call_share: Some(0.80),
    max_writes_share: Some(0.25),
};

fn support_agent(model: impl Into<rig::DynModel<rig::operation::Completion>>) -> Agent {
    AgentBuilder::new(model)
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .max_tokens(800)
        .default_max_turns(4)
        .build()
}

/// `turns` turns of the support chat, over the streaming endpoint when
/// `streamed`.
async fn support_chat(
    agent: &Agent,
    clock: &CassetteClock,
    turns: usize,
    streamed: bool,
) -> RunLog {
    let mut history: Vec<Message> = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=turns {
        let prompt = question(turn, "A");
        if streamed {
            chat_streamed(agent, clock, prompt, &mut history, &mut log).await;
        } else {
            chat(agent, clock, prompt, &mut history, &mut log).await;
        }
    }
    log
}

fn check(scenario: &str, rates: CacheRates, limits: Option<Limits>, log: &RunLog) {
    let run = LongRun {
        wire: CacheWire::Anthropic,
        scenario,
        model: MODEL,
        rates,
        output_price: OUTPUT_PRICE,
        limits,
        drops_signatures: false,
        conversation: None,
    };
    cache_longrun::check(&run, log, None);
}

#[tokio::test]
async fn support_chat_100_automatic() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/support_chat_100_automatic",
        |models, clock| async move {
            let model = models
                .completion(MODEL)
                .map_wire(anthropic::Messages::with_automatic_caching);
            support_chat(&support_agent(model), &clock, 100, false).await
        },
    )
    .await;
    check(
        "long_run_caching/support_chat_100_automatic",
        RATES_5M,
        Some(LIMITS),
        &log,
    );
}

#[tokio::test]
async fn support_chat_100_automatic_1h() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/support_chat_100_automatic_1h",
        |models, clock| async move {
            let model = models
                .completion(MODEL)
                .map_wire(anthropic::Messages::with_automatic_caching_1h);
            support_chat(&support_agent(model), &clock, 100, false).await
        },
    )
    .await;
    check(
        "long_run_caching/support_chat_100_automatic_1h",
        RATES_1H,
        Some(LIMITS),
        &log,
    );
}

#[tokio::test]
async fn support_chat_100_streamed() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/support_chat_100_streamed",
        |models, clock| async move {
            let model = models
                .completion(MODEL)
                .map_wire(anthropic::Messages::with_automatic_caching);
            support_chat(&support_agent(model), &clock, 100, true).await
        },
    )
    .await;
    check(
        "long_run_caching/support_chat_100_streamed",
        RATES_5M,
        Some(LIMITS),
        &log,
    );
}

/// Today's behavior: the same chat with caching off, reported only.
#[tokio::test]
async fn support_chat_30_baseline() {
    let log = with_anthropic_long_run_cassette(
        "long_run_caching/support_chat_30_baseline",
        |models, clock| async move {
            support_chat(&support_agent(models.completion(MODEL)), &clock, 30, false).await
        },
    )
    .await;
    check(
        "long_run_caching/support_chat_30_baseline",
        RATES_5M,
        None,
        &log,
    );
}
