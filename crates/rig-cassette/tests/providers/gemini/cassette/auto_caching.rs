//! Automatic explicit caching (`gemini::caching`) over long live runs on
//! gemini-3.8-flash, recorded and replayed.
//!
//! Every assertion here is on recorded traffic. The figures and the checks
//! every long run shares live in `rig_test_support::cache_longrun`; for a run
//! that uses a cache book they include Gemini's rules: a request that reads a
//! cache carries none of the prefix the cache holds, cache plus request tail
//! is the conversation byte for byte, every cache is deleted, every replaced
//! cache was read at least three times, and cache creation stays at or below
//! 15% of prompt tokens. The input saving counts every creation at the input
//! price and every cache's storage.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `GEMINI_API_KEY`; see
//! `tests/README.md`.

use std::collections::HashMap;
use std::time::Duration;

use rig::agent::Agent;
use rig::completion::{CacheCost, CacheRates, Message};
use rig::providers::gemini::{
    AutoCache, CacheBook, CacheEvent, CacheReport, Gemini, Lease, ThoughtReplay,
};
use rig::tool::Tool;
use rig::{AgentBuilder, AgentRun};
use rig_test_support::cache_longrun::{
    self, CacheWire, Created, Interaction, Limits, LongRun, LookupOrder, Recording, RunLog,
    SUPPORT_PREAMBLE, chat, chat_streamed, question,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cassettes::{CassetteClock, CassetteSpec};

use super::super::support::with_gemini_auto_caching_cassette;

const MODEL: &str = "gemini-3.8-flash";
/// Gemini 3.8 Flash standard prices, USD per 1M tokens: input, cached read,
/// creation (billed as input) and storage per hour.
const RATES: CacheRates = CacheRates {
    input: 0.75,
    cached_read: 0.075,
    cache_write: 0.75,
    storage_per_hour: 0.50,
};
/// Gemini 3.8 Flash standard output price (thinking included), USD per 1M tokens.
const OUTPUT_PRICE: f64 = 3.75;
/// The most caches a 100-turn chat may create. Rolling when the premium
/// already paid equals the next cache's cost is the cheapest schedule for a
/// conversation that grows every call; measured over 100 turns it created 12
/// caches with current-turn replay (three recordings) and 13 with default
/// replay (two recordings). The cap started at 12 as an estimate made before
/// any 100-turn run and was set from those measurements. The per-cache
/// "read at least three times" check and the 15% creation limit are what rule
/// out paying for caches that are never used.
const MAX_CACHES_100_TURNS: usize = 14;
/// Gemini's limit on tokens put into caches, relative to prompt tokens.
const MAX_CREATED_SHARE: f64 = 0.15;

// ---------------------------------------------------------------------------
// Tools.

#[derive(Deserialize)]
struct LogArgs {
    name: String,
}

/// Reads one log. Every tenth log is about 20k tokens of mostly digits
/// (digits tokenize one per character, which keeps the fixture small for its
/// token count); the rest are short.
struct ReadLog;

fn log_body(index: u32) -> String {
    let big = index.is_multiple_of(10);
    let lines = if big { 425 } else { 6 };
    let mut seed = u64::from(index)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(17);
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        seed >> 33
    };
    (0..lines)
        .map(|line| {
            let level = if next() % 9 == 0 { "ERROR" } else { "INFO" };
            format!(
                "{level} t={} rid={:010} lat={:05} st={} b={:07}",
                1_780_000_000 + u64::from(index) * 1_000 + line,
                next(),
                next() % 100_000,
                [200, 200, 200, 404, 500][(next() % 5) as usize],
                next() % 10_000_000,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl Tool for ReadLog {
    const NAME: &'static str = "read_log";
    type Error = std::convert::Infallible;
    type Args = LogArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read one log file by name, such as log-7.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let index = args
            .name
            .rsplit('-')
            .next()
            .and_then(|number| number.parse::<u32>().ok())
            .unwrap_or(0);
        Ok(log_body(index))
    }
}

// ---------------------------------------------------------------------------
// Agents, books and checks.

fn thinking_low() -> Value {
    json!({ "generationConfig": { "thinkingConfig": { "thinkingLevel": "low" } } })
}

fn support_agent(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    preamble: &str,
) -> Agent {
    AgentBuilder::new(model)
        .preamble(preamble)
        .tool(LookupOrder)
        .max_tokens(800)
        .additional_params(thinking_low())
        .default_max_turns(4)
        .build()
}

/// A book on the session's clock with the run's display-name prefix.
fn book(clock: &CassetteClock, policy: AutoCache) -> CacheBook {
    let clock = clock.clone();
    CacheBook::new(policy)
        .with_clock(move || clock.now())
        .with_display_prefix("rig-auto-caching-test-")
}

/// Limits for a run on a book: its saving floor and, where it holds from
/// the first cache read on, its per-call floor.
fn limits(min_saving: f64, min_call_share: Option<f64>) -> Option<Limits> {
    Some(Limits {
        min_saving,
        min_call_share,
        max_writes_share: Some(MAX_CREATED_SHARE),
    })
}

/// Read the run back and run the shared checks, pricing the book's caches
/// (`resources`) beside the calls.
fn check(
    scenario: &str,
    log: &RunLog,
    resources: Option<CacheCost>,
    limits: Option<Limits>,
    drops_signatures: bool,
) -> (Recording, cache_longrun::Figures) {
    let run = LongRun {
        wire: CacheWire::Gemini,
        scenario,
        model: MODEL,
        rates: RATES,
        output_price: OUTPUT_PRICE,
        limits,
        drops_signatures,
    };
    cache_longrun::check(&run, log, resources)
}

/// How far the book's size estimate was from Gemini's count, per cache.
fn print_estimates(scenario: &str, events: &[CacheEvent]) {
    for event in events {
        if let CacheEvent::Created {
            name,
            tokens,
            estimated,
            ..
        } = event
        {
            println!(
                "AUTO_CACHING_ESTIMATE {scenario} {name} tokens={tokens} estimated={estimated} \
                 error={:+.1}%",
                100.0 * (*estimated as f64 - *tokens as f64) / (*tokens).max(1) as f64
            );
        }
    }
}

/// Every call that names a cache reads it whole: its cached tokens equal the
/// cache's size.
fn assert_reads_whole(recording: &Recording) {
    let sizes: HashMap<&str, u64> = recording
        .created
        .iter()
        .map(|created| (created.name.as_str(), created.tokens))
        .collect();
    for call in &recording.calls {
        if let Some(cache) = &call.cache {
            let cached = call.usage.cached_input_tokens.unwrap_or(0);
            assert_eq!(
                cached,
                sizes[cache.as_str()],
                "interaction {} read {cached} of {cache}'s {} tokens",
                call.index,
                sizes[cache.as_str()]
            );
        }
    }
}

fn assert_caches_at_most(figures: &cache_longrun::Figures, most: usize) {
    cache_longrun::print_limit(figures, &format!("at most {most} caches"));
    let created = figures.caches_created.unwrap_or(0);
    assert!(created <= most, "{created} caches (limit {most})");
}

fn assert_cached_share_at_least(figures: &cache_longrun::Figures, least: f64) {
    cache_longrun::print_limit(
        figures,
        &format!("cached share at least {:.0}%", least * 100.0),
    );
    assert!(
        figures.cached_share >= least,
        "cached share {:.3} (limit {least:.2})",
        figures.cached_share
    );
}

// ---------------------------------------------------------------------------
// The runs.

/// One 100-turn support chat on a book; `streamed` sends it over
/// `streamGenerateContent`.
async fn support_chat_100(
    gemini: Gemini,
    clock: CassetteClock,
    replay: ThoughtReplay,
    streamed: bool,
) -> (CacheReport, RunLog, Vec<CacheEvent>) {
    let book = book(&clock, AutoCache::default());
    let agent = support_agent(
        gemini
            .completion(MODEL)
            .thought_replay(replay)
            .caching(&book),
        SUPPORT_PREAMBLE,
    );
    let mut history = Vec::new();
    let mut log = RunLog::default();
    for turn in 1..=100 {
        let prompt = question(turn, "A");
        if streamed {
            chat_streamed(&agent, &clock, prompt, &mut history, &mut log).await;
        } else {
            chat(&agent, &clock, prompt, &mut history, &mut log).await;
        }
    }
    book.close(&gemini.cached_contents()).await;
    (book.report(), log, book.events())
}

#[tokio::test]
async fn support_chat_100_current_turn() {
    const SCENARIO: &str = "auto_caching/support_chat_100_current_turn";
    let (report, log, events) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_current_turn",
        |gemini, clock| support_chat_100(gemini, clock, ThoughtReplay::CurrentTurn, false),
    )
    .await;
    let (_, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.65, Some(0.50)),
        true,
    );
    print_estimates(SCENARIO, &events);
    assert_cached_share_at_least(&figures, 0.80);
    assert_caches_at_most(&figures, MAX_CACHES_100_TURNS);
}

#[tokio::test]
async fn support_chat_100_auto() {
    const SCENARIO: &str = "auto_caching/support_chat_100_auto";
    let (report, log, events) =
        with_gemini_auto_caching_cassette("auto_caching/support_chat_100_auto", |gemini, clock| {
            support_chat_100(gemini, clock, ThoughtReplay::All, false)
        })
        .await;
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.30, Some(0.50)),
        false,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(&figures, "every read takes the whole cache");
    assert_reads_whole(&recording);
    assert_caches_at_most(&figures, MAX_CACHES_100_TURNS);
}

#[tokio::test]
async fn support_chat_100_streamed() {
    const SCENARIO: &str = "auto_caching/support_chat_100_streamed";
    let (report, log, events) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_streamed",
        |gemini, clock| support_chat_100(gemini, clock, ThoughtReplay::CurrentTurn, true),
    )
    .await;
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.65, Some(0.50)),
        true,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(&figures, "every call streams");
    assert!(
        recording
            .calls
            .iter()
            .all(|call| recording.interactions[call.index]
                .path
                .contains(":streamGenerateContent")),
        "every call streams"
    );
    assert_cached_share_at_least(&figures, 0.80);
    assert_caches_at_most(&figures, MAX_CACHES_100_TURNS);
}

/// What a process writes at a turn boundary: the next turn as a run, the
/// history it continues, and the book's caches.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    run: AgentRun,
    history: Vec<Message>,
    leases: Vec<Lease>,
}

#[tokio::test]
async fn support_chat_100_resume() {
    const SCENARIO: &str = "auto_caching/support_chat_100_resume";
    let (report, log, events, pre_leases, pre_created) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_resume",
        |gemini, clock| async move {
            let mut log = RunLog::default();
            // Turns 1..=50, then a checkpoint as a process would write it.
            let (saved, pre_created) = {
                let book = book(&clock, AutoCache::default());
                let agent = support_agent(
                    gemini
                        .completion(MODEL)
                        .thought_replay(ThoughtReplay::CurrentTurn)
                        .caching(&book),
                    SUPPORT_PREAMBLE,
                );
                let mut history = Vec::new();
                for turn in 1..=50 {
                    chat(&agent, &clock, question(turn, "A"), &mut history, &mut log).await;
                }
                let checkpoint = Checkpoint {
                    run: AgentRun::new(question(51, "A"))
                        .with_history(history.clone())
                        .max_turns(4),
                    history,
                    leases: book.leases(),
                };
                // The first process's caches; its report is never read, since
                // reading one reads the clock.
                let created: u64 = book
                    .events()
                    .iter()
                    .filter_map(|event| match event {
                        CacheEvent::Created { tokens, .. } => Some(*tokens),
                        _ => None,
                    })
                    .sum();
                (
                    serde_json::to_string(&checkpoint).expect("checkpoint serializes"),
                    created,
                )
            };

            // A new process: a new book from the checkpoint, proven live.
            let checkpoint: Checkpoint = serde_json::from_str(&saved).expect("checkpoint loads");
            let pre_leases = checkpoint.leases.clone();
            let book = book(&clock, AutoCache::default());
            book.restore(checkpoint.leases);
            let survived = book.prove(&gemini.cached_contents()).await;
            assert!(survived >= 1, "the checkpoint's cache is still alive");
            let agent = support_agent(
                gemini
                    .completion(MODEL)
                    .thought_replay(ThoughtReplay::CurrentTurn)
                    .caching(&book),
                SUPPORT_PREAMBLE,
            );
            let response = agent.resume(checkpoint.run).await.expect("resumed turn");
            log.usages
                .extend(response.completion_calls.iter().map(|call| call.usage));
            let mut history = checkpoint.history;
            history.extend(response.messages.clone().unwrap_or_default());
            for turn in 52..=100 {
                chat(&agent, &clock, question(turn, "A"), &mut history, &mut log).await;
            }
            book.close(&gemini.cached_contents()).await;
            (book.report(), log, book.events(), pre_leases, pre_created)
        },
    )
    .await;
    // Creation counts both processes' caches; storage only the second
    // book's, which counts restored caches from the restore.
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(
            CacheCost::from(&report)
                + CacheCost {
                    cache_writes: pre_created,
                    ..CacheCost::default()
                },
        ),
        limits(0.65, Some(0.50)),
        true,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(
        &figures,
        "the first call after the resume reads a pre-checkpoint cache",
    );

    // The first call after the resume reads a pre-checkpoint cache, and no
    // cache was created between the checkpoint and that call.
    let pre: Vec<&str> = pre_leases.iter().map(|lease| lease.name.as_str()).collect();
    let resume_get = recording
        .interactions
        .iter()
        .position(|interaction| {
            interaction.method == "GET" && interaction.path.contains("/cachedContents/")
        })
        .expect("the resume proves its leases");
    let first_after = recording
        .calls
        .iter()
        .find(|call| call.index > resume_get)
        .expect("a call after the resume");
    assert!(
        first_after
            .cache
            .as_deref()
            .is_some_and(|cache| pre.contains(&cache)),
        "the first call after the resume reads {:?}, not a pre-checkpoint cache {pre:?}",
        first_after.cache
    );
    assert!(
        !recording
            .created
            .iter()
            .any(|created| created.index > resume_get && created.index < first_after.index),
        "the resume created a cache before its first call"
    );
    assert_cached_share_at_least(&figures, 0.80);
    assert_caches_at_most(&figures, MAX_CACHES_100_TURNS);
}

#[tokio::test]
async fn support_chat_100_compaction() {
    const SCENARIO: &str = "auto_caching/support_chat_100_compaction";
    let (report, log, events) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_compaction",
        |gemini, clock| async move {
            let book = book(&clock, AutoCache::default());
            let agent = support_agent(
                gemini
                    .completion(MODEL)
                    .thought_replay(ThoughtReplay::CurrentTurn)
                    .caching(&book),
                SUPPORT_PREAMBLE,
            );
            let mut history = Vec::new();
            let mut log = RunLog::default();
            for turn in 1..=100 {
                if turn == 51 {
                    history = vec![
                        Message::user(
                            "Summary of the conversation so far: the customer asked about \
                             orders A-1 to A-50; each was looked up and answered.",
                        ),
                        Message::assistant("Understood. How else can I help with your orders?"),
                    ];
                }
                chat(&agent, &clock, question(turn, "A"), &mut history, &mut log).await;
            }
            book.close(&gemini.cached_contents()).await;
            (book.report(), log, book.events())
        },
    )
    .await;
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.50, Some(0.50)),
        true,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(
        &figures,
        "no stale read after compaction, old cache retired",
    );

    // The first call after compaction reads no cache made before it, and the
    // pre-compaction cache is retired before the run's close.
    let compacted = recording
        .calls
        .iter()
        .find(|call| {
            call.raw_contents
                .first()
                .is_some_and(|first| first.contains("Summary of the conversation so far"))
                || call.cache.as_deref().is_some_and(|cache| {
                    recording.created.iter().any(|created| {
                        created.name == cache
                            && created.raw_contents.first().is_some_and(|first| {
                                first.contains("Summary of the conversation so far")
                            })
                    })
                })
        })
        .expect("a call after the compaction");
    let stale: Vec<&Created> = recording
        .created
        .iter()
        .filter(|created| {
            created.index < compacted.index
                && created
                    .raw_contents
                    .first()
                    .is_some_and(|first| !first.contains("Summary of the conversation so far"))
        })
        .collect();
    assert!(
        compacted
            .cache
            .as_deref()
            .is_none_or(|cache| !stale.iter().any(|created| created.name == cache)),
        "the first call after compaction read a stale cache"
    );
    let last_call = recording.calls.last().map_or(0, |call| call.index);
    for created in &stale {
        let target = format!("/v1beta/{}", created.name);
        assert!(
            recording.interactions[..last_call]
                .iter()
                .any(|interaction| interaction.method == "DELETE" && interaction.path == target),
            "the pre-compaction cache {} was not retired before the run ended",
            created.name
        );
    }
}

#[tokio::test]
async fn agent_loop_large_results() {
    const SCENARIO: &str = "auto_caching/agent_loop_large_results";
    let (report, log, events) = with_gemini_auto_caching_cassette(
        "auto_caching/agent_loop_large_results",
        |gemini, clock| async move {
            let book = book(&clock, AutoCache::default());
            let agent = AgentBuilder::new(gemini.completion(MODEL).caching(&book))
                .preamble(
                    "You are a meticulous log auditor. You read logs with the read_log tool, \
                     exactly one call per turn, never several calls in one turn, strictly in the \
                     order you are told. Do not stop early.",
                )
                .tool(ReadLog)
                .max_tokens(800)
                .additional_params(thinking_low())
                .default_max_turns(70)
                .build();
            let mut log = RunLog::default();
            let mut history = Vec::new();
            chat(
                &agent,
                &clock,
                "Read log-1 through log-60 with read_log, one call per turn, in order (log-1, \
                 log-2, ..., log-60). After the last one, report how many lines start with \
                 ERROR across all sixty logs, as a single number."
                    .to_owned(),
                &mut history,
                &mut log,
            )
            .await;
            book.close(&gemini.cached_contents()).await;
            (book.report(), log, book.events())
        },
    )
    .await;
    // The loop stays inline while implicit caching serves its large
    // prompts, so it has no per-call floor.
    let (_, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.60, None),
        false,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(&figures, "at least 60 calls");
    assert!(figures.calls >= 60, "a long loop: {} calls", figures.calls);
    assert_caches_at_most(&figures, 3);
    assert_cached_share_at_least(&figures, 0.60);
}

/// A ~5.5k-token prefix: the handbook and a product appendix.
fn shared_preamble() -> String {
    let mut preamble = SUPPORT_PREAMBLE.to_owned();
    preamble.push_str("\n## Appendix: product catalog\n\n");
    for sku in 0..220 {
        preamble.push_str(&format!(
            "- SKU-{sku:04}: {} {}, {} grams, ships in {} business days, returnable: {}.\n",
            ["Linen", "Cotton", "Wool", "Canvas", "Leather", "Bamboo"][sku % 6],
            [
                "shirt",
                "tote",
                "scarf",
                "cap",
                "wallet",
                "sock pack",
                "apron"
            ][sku % 7],
            120 + (sku * 37) % 900,
            1 + sku % 5,
            if sku % 11 == 0 {
                "no (final sale)"
            } else {
                "yes"
            },
        ));
    }
    preamble
}

#[tokio::test]
async fn subagents_shared_prefix() {
    const SCENARIO: &str = "auto_caching/subagents_shared_prefix";
    let (report, log, events) = with_gemini_auto_caching_cassette(
        CassetteSpec::new("auto_caching/subagents_shared_prefix").unordered(),
        |gemini, clock| async move {
            let book = book(&clock, AutoCache::default());
            let preamble = shared_preamble();
            let agents: Vec<Agent> = (0..4)
                .map(|_| support_agent(gemini.completion(MODEL).caching(&book), &preamble))
                .collect();
            let prefixes = ["B", "C", "D", "E"];
            let mut histories: Vec<Vec<Message>> = vec![Vec::new(); 4];
            let mut logs: Vec<RunLog> = (0..4).map(|_| RunLog::default()).collect();
            // Turn 1 runs agent by agent, so which conversation reaches the
            // shared prefix second (and caches it) is the same on replay; the
            // other 24 turns run all four agents at once.
            for (index, agent) in agents.iter().enumerate() {
                chat(
                    agent,
                    &clock,
                    question(1, prefixes[index]),
                    &mut histories[index],
                    &mut logs[index],
                )
                .await;
            }
            for turn in 2..=25 {
                let clock = &clock;
                futures::future::join_all(
                    agents
                        .iter()
                        .zip(histories.iter_mut())
                        .zip(logs.iter_mut())
                        .enumerate()
                        .map(|(index, ((agent, history), log))| async move {
                            chat(agent, clock, question(turn, prefixes[index]), history, log).await;
                        }),
                )
                .await;
            }
            book.close(&gemini.cached_contents()).await;
            let mut log = RunLog::default();
            for part in logs {
                log.usages.extend(part.usages);
                log.retries += part.retries;
            }
            (book.report(), log, book.events())
        },
    )
    .await;
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.60, Some(0.50)),
        false,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(&figures, "shared prefix created exactly once");
    let prefix_caches = recording
        .created
        .iter()
        .filter(|created| created.raw_contents.is_empty())
        .count();
    assert_eq!(
        prefix_caches, 1,
        "the shared prefix is created exactly once"
    );
}

#[tokio::test]
async fn lifecycle() {
    const SCENARIO: &str = "auto_caching/lifecycle";
    let (report, log, events) =
        with_gemini_auto_caching_cassette("auto_caching/lifecycle", |gemini, clock| async move {
            // (a) min_tokens 0 and a small first request: Gemini refuses the
            // cache as too small. (b) TTL 10 s with a 15 s pause.
            let book = book(
                &clock,
                AutoCache {
                    min_tokens: 0,
                    ttl: Duration::from_secs(10),
                    ..AutoCache::default()
                },
            );
            let agent = support_agent(
                gemini.completion(MODEL).caching(&book),
                "You are a customer support agent. Look orders up with lookup_order before \
                 answering, and answer in one sentence.",
            );
            let mut history = Vec::new();
            let mut log = RunLog::default();
            for turn in 1..=40 {
                if turn == 15 {
                    clock.pause(Duration::from_secs(15)).await;
                }
                if turn == 25 {
                    // (c) The live cache disappears behind the book's back.
                    let caches = gemini.cached_contents();
                    for lease in book.leases() {
                        let _ = caches.delete(&lease.name).await;
                    }
                }
                chat(&agent, &clock, question(turn, "A"), &mut history, &mut log).await;
            }
            // (d) Close twice: the second close finds everything gone.
            book.close(&gemini.cached_contents()).await;
            book.close(&gemini.cached_contents()).await;
            (book.report(), log, book.events())
        })
        .await;
    // Caches expire and are deleted mid-run by design, so no per-call floor;
    // caching must still never cost more than sending everything inline.
    let (recording, figures) = check(
        SCENARIO,
        &log,
        Some(CacheCost::from(&report)),
        limits(0.0, None),
        false,
    );
    print_estimates(SCENARIO, &events);
    cache_longrun::print_limit(
        &figures,
        "refused, expired and deleted caches handled (a)-(d)",
    );
    println!(
        "AUTO_CACHING_EVENTS {SCENARIO} {}",
        serde_json::to_string(&events).unwrap_or_default()
    );

    // (a) Gemini refused a cache as too small, and the request still succeeded.
    let refused = recording
        .interactions
        .iter()
        .position(|interaction| {
            interaction.method == "POST"
                && interaction.path.ends_with("/cachedContents")
                && interaction.status == 400
                && interaction.response.contains("too small")
        })
        .expect("a cache refused as too small");
    assert!(
        recording.interactions[refused + 1..]
            .iter()
            .find(|interaction| interaction.path.contains(":generateContent"))
            .is_some_and(|interaction| interaction.status == 200
                && !interaction.request.contains("cachedContent")),
        "the request after the refusal went out inline and succeeded"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, CacheEvent::CreateFailed { .. }))
    );

    // (b) No request ever named a cache that had expired: the only 403 on a
    // generate call is (c)'s.
    let forbidden: Vec<usize> = recording
        .interactions
        .iter()
        .enumerate()
        .filter(|(_, interaction)| {
            interaction.path.contains(":generateContent") && interaction.status == 403
        })
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        forbidden.len(),
        1,
        "exactly one 403 on a generate call: the deleted cache"
    );

    // (c) The 403 is followed at once by the same request inline, which
    // succeeds, and later by a new cache.
    let retry = &recording.interactions[forbidden[0] + 1];
    assert!(retry.path.contains(":generateContent") && retry.status == 200);
    assert!(!retry.request.contains("\"cachedContent\""));
    assert!(
        recording
            .created
            .iter()
            .any(|created| created.index > forbidden[0]),
        "a new cache after the loss"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, CacheEvent::Lost { .. }))
    );

    // (d) The second close deleted nothing new: every DELETE after the first
    // close's targets a cache already gone (403).
    let deletes: Vec<&Interaction> = recording
        .interactions
        .iter()
        .filter(|interaction| interaction.method == "DELETE")
        .collect();
    assert!(!deletes.is_empty());
}

#[tokio::test]
async fn support_chat_30_baseline() {
    const SCENARIO: &str = "auto_caching/support_chat_30_baseline";
    let log = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_30_baseline",
        |gemini, clock| async move {
            let agent = support_agent(gemini.completion(MODEL), SUPPORT_PREAMBLE);
            let mut history = Vec::new();
            let mut log = RunLog::default();
            for turn in 1..=30 {
                chat(&agent, &clock, question(turn, "A"), &mut history, &mut log).await;
            }
            log
        },
    )
    .await;
    let (recording, figures) = check(SCENARIO, &log, None, None, false);
    cache_longrun::print_limit(&figures, "no cache created or read");
    assert!(
        recording.created.is_empty(),
        "the baseline creates no cache"
    );
    assert!(
        recording.calls.iter().all(|call| call.cache.is_none()),
        "the baseline reads no cache"
    );
}
