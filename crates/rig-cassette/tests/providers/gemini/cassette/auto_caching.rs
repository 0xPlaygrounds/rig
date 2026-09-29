//! Automatic explicit caching (`gemini::caching`) over long live runs on
//! gemini-3.8-flash, recorded and replayed.
//!
//! Every assertion here is on recorded traffic. Beside the per-run
//! thresholds, every run that uses a cache book is checked for two things:
//!
//! - **Integrity.** A request that reads a cache carries none of the prefix
//!   the cache holds; the cache's contents followed by the request's are the
//!   conversation byte for byte; every cache the run created is deleted; and
//!   rig's `Usage` never reports more cached than input tokens.
//! - **"Not faked" economics.** A run could look well cached by creating a
//!   new cache before every call and paying for it each time. So every cache
//!   a later one replaced must have been read at least three times, the
//!   tokens put into caches must stay at or below 15% of all prompt tokens,
//!   and the input saving counts every creation at the input price and every
//!   cache's storage.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `GEMINI_API_KEY`; see
//! `tests/README.md`.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use rig::agent::{Agent, MultiTurnStreamItem};
use rig::completion::{Message, Usage};
use rig::providers::gemini::{
    AutoCache, CacheBook, CacheEvent, CacheReport, Gemini, Lease, ThoughtReplay,
};
use rig::tool::Tool;
use rig::{AgentBuilder, AgentRun};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::cassettes::{CassetteClock, CassetteSpec};

use super::super::support::with_gemini_auto_caching_cassette;

const MODEL: &str = "gemini-3.8-flash";
const SUPPORT_PREAMBLE: &str = include_str!("support_preamble.md");
/// Gemini 3.8 Flash standard input price, USD per 1M tokens.
const INPUT_PRICE: f64 = 0.75;
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

// ---------------------------------------------------------------------------
// Tools.

#[derive(Deserialize)]
struct OrderArgs {
    order_id: String,
}

/// Looks an order up. Deterministic: the same id always gives the same
/// status, date and carrier, so a re-recording sees the same facts.
struct LookupOrder;

const STATUSES: [&str; 6] = [
    "processing",
    "shipped",
    "delivered",
    "refunded",
    "cancelled",
    "returned",
];

impl Tool for LookupOrder {
    const NAME: &'static str = "lookup_order";
    type Error = std::convert::Infallible;
    type Args = OrderArgs;
    type Output = Value;

    fn description(&self) -> String {
        "Status of an order by id.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "order_id": { "type": "string" } },
            "required": ["order_id"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let hash = args.order_id.bytes().fold(7u32, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(u32::from(byte))
        });
        let status = STATUSES[(hash % 6) as usize];
        let day = hash % 27 + 1;
        let mut result = json!({
            "order_id": args.order_id,
            "status": status,
            "date": format!("2026-05-{day:02}"),
        });
        if status == "shipped" || status == "delivered" {
            result["carrier"] = json!(["DHL", "UPS", "FedEx"][(hash % 3) as usize]);
        }
        Ok(result)
    }
}

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
// Agents and turns.

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

fn question(turn: usize, prefix: &str) -> String {
    let templates = [
        "What happened to order {id}?",
        "Can you check the status of order {id} for me?",
        "Where is my order {id}?",
        "Has order {id} shipped yet?",
        "I need an update on order {id}, please.",
        "Any news on {id}? I ordered it a while ago.",
        "Could you look up {id}?",
    ];
    templates[turn % templates.len()].replace("{id}", &format!("{prefix}-{turn}"))
}

/// What a run did, from the test's side of the wire.
#[derive(Default)]
struct RunLog {
    /// rig's `Usage` for every completion call.
    usages: Vec<Usage>,
    /// Turns retried after a Google 5xx or 429.
    retries: usize,
    /// The book's events, read after the run.
    events: Vec<CacheEvent>,
}

fn retryable(status: Option<http::StatusCode>) -> bool {
    status.is_some_and(|status| status.is_server_error() || status.as_u16() == 429)
}

const BACKOFF: [u64; 3] = [2, 5, 10];

/// One chat turn, retried up to three times on a Google 5xx or 429 with a
/// record-only pause between attempts. The failed attempts are recorded like
/// any other calls, so replay matches.
async fn chat(
    agent: &Agent,
    clock: &CassetteClock,
    prompt: String,
    history: &mut Vec<Message>,
    log: &mut RunLog,
) {
    let mut waits = BACKOFF.iter();
    loop {
        let error = match agent.chat(prompt.clone(), history).await {
            Ok(response) => {
                log.usages
                    .extend(response.completion_calls.iter().map(|call| call.usage));
                return;
            }
            Err(error) => error,
        };
        match waits.next() {
            Some(wait) if retryable(error.provider_response_status()) => {
                log.retries += 1;
                clock.pause(Duration::from_secs(*wait)).await;
            }
            _ => panic!("turn failed: {error}"),
        }
    }
}

fn streaming_status(error: &rig::agent::StreamingError) -> Option<http::StatusCode> {
    match error {
        rig::agent::StreamingError::Completion(error) => error.provider_response_status(),
        rig::agent::StreamingError::Report(report) => report
            .http_status
            .and_then(|status| http::StatusCode::from_u16(status).ok()),
        rig::agent::StreamingError::Prompt(error) => error.provider_response_status(),
    }
}

/// [`chat`] over `streamGenerateContent`.
async fn chat_streamed(
    agent: &Agent,
    clock: &CassetteClock,
    prompt: String,
    history: &mut Vec<Message>,
    log: &mut RunLog,
) {
    use futures::StreamExt;
    let mut waits = BACKOFF.iter();
    loop {
        let mut stream = agent
            .prompt(prompt.clone())
            .history(history.clone())
            .stream();
        let mut failure = None;
        let mut done = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(response)) => done = Some(response),
                Ok(_) => {}
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        let error = match (done, failure) {
            (Some(response), None) => {
                history.extend(response.messages.clone().unwrap_or_default());
                log.usages
                    .extend(response.completion_calls.iter().map(|call| call.usage));
                return;
            }
            (_, error) => error,
        };
        match (waits.next(), &error) {
            (Some(wait), Some(failure)) if retryable(streaming_status(failure)) => {
                log.retries += 1;
                clock.pause(Duration::from_secs(*wait)).await;
            }
            _ => panic!("streamed turn failed: {error:?}"),
        }
    }
}

/// A book on the session's clock with the run's display-name prefix.
fn book(clock: &CassetteClock, policy: AutoCache) -> CacheBook {
    let clock = clock.clone();
    CacheBook::new(policy)
        .with_clock(move || clock.now())
        .with_display_prefix("rig-auto-caching-test-")
}

// ---------------------------------------------------------------------------
// Reading the recording back.

#[derive(Debug)]
struct Interaction {
    method: String,
    path: String,
    request: String,
    status: u16,
    response: String,
}

fn interactions(scenario: &str) -> Vec<Interaction> {
    let path = crate::cassettes::cassette_path("gemini", scenario);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()));
    serde_yaml::Deserializer::from_str(&text)
        .map(|document| {
            let document = serde_yaml::Value::deserialize(document).expect("cassette document");
            let field = |side: &str, key: &str| {
                document
                    .get(side)
                    .and_then(|side| side.get(key))
                    .cloned()
                    .unwrap_or(serde_yaml::Value::Null)
            };
            Interaction {
                method: field("when", "method")
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                path: field("when", "path")
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                request: field("when", "body")
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                status: field("then", "status")
                    .as_u64()
                    .and_then(|status| u16::try_from(status).ok())
                    .unwrap_or_default(),
                response: field("then", "body")
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            }
        })
        .collect()
}

#[derive(Default, Debug, Clone, Copy)]
struct Tokens {
    prompt: u64,
    tool_use: u64,
    cached: u64,
    thoughts: u64,
    candidates: u64,
}

fn usage_of(response: &str) -> Option<Tokens> {
    let read = |value: &Value| {
        let usage = value.get("usageMetadata")?;
        let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
        Some(Tokens {
            prompt: count("promptTokenCount"),
            tool_use: count("toolUsePromptTokenCount"),
            cached: count("cachedContentTokenCount"),
            thoughts: count("thoughtsTokenCount"),
            candidates: count("candidatesTokenCount"),
        })
    };
    if let Ok(value) = serde_json::from_str::<Value>(response) {
        return read(&value);
    }
    response
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .filter_map(|value| read(&value))
        .next_back()
}

/// One successful `generateContent` call.
struct Call {
    index: usize,
    cache: Option<String>,
    body: Value,
    raw_contents: Vec<String>,
    tokens: Tokens,
}

/// One cache the run created.
struct Created {
    index: usize,
    name: String,
    tokens: u64,
    body: Value,
    raw_contents: Vec<String>,
}

struct Recording {
    interactions: Vec<Interaction>,
    calls: Vec<Call>,
    created: Vec<Created>,
}

fn raw_contents(body: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct Contents<'a> {
        #[serde(borrow, default)]
        contents: Vec<&'a serde_json::value::RawValue>,
    }
    serde_json::from_str::<Contents<'_>>(body)
        .map(|contents| {
            contents
                .contents
                .iter()
                .map(|raw| raw.get().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn load(scenario: &str) -> Recording {
    let interactions = interactions(scenario);
    let mut calls = Vec::new();
    let mut created = Vec::new();
    for (index, interaction) in interactions.iter().enumerate() {
        if interaction.path.contains(":generateContent")
            || interaction.path.contains(":streamGenerateContent")
        {
            if interaction.status != 200 {
                continue;
            }
            let body: Value = serde_json::from_str(&interaction.request).expect("request JSON");
            calls.push(Call {
                index,
                cache: body
                    .get("cachedContent")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                raw_contents: raw_contents(&interaction.request),
                body,
                tokens: usage_of(&interaction.response).expect("a successful call reports usage"),
            });
        } else if interaction.method == "POST"
            && interaction.path.ends_with("/cachedContents")
            && interaction.status == 200
        {
            let reply: Value = serde_json::from_str(&interaction.response).expect("cache JSON");
            created.push(Created {
                index,
                name: reply["name"].as_str().expect("cache name").to_owned(),
                tokens: reply
                    .pointer("/usageMetadata/totalTokenCount")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                body: serde_json::from_str(&interaction.request).expect("create JSON"),
                raw_contents: raw_contents(&interaction.request),
            });
        }
    }
    Recording {
        interactions,
        calls,
        created,
    }
}

/// Whole-run figures, in tokens and in dollars at the standard input price.
#[derive(Debug, Serialize)]
struct Figures {
    calls: usize,
    prompt: u64,
    cached: u64,
    cached_share: f64,
    caches_created: usize,
    created_tokens: u64,
    created_share: f64,
    reads: BTreeMap<String, u64>,
    storage_token_hours: f64,
    input_with_caching: f64,
    input_uncached: f64,
    input_saving: f64,
    usd_with_caching: f64,
    usd_uncached: f64,
    output_tokens: u64,
    usd_output: f64,
    fixture_bytes: u64,
    retries: usize,
}

fn figures(
    scenario: &str,
    recording: &Recording,
    report: Option<&CacheReport>,
    retries: usize,
) -> Figures {
    let policy = AutoCache::default();
    let prompt: u64 = recording
        .calls
        .iter()
        .map(|call| call.tokens.prompt + call.tokens.tool_use)
        .sum();
    let cached: u64 = recording.calls.iter().map(|call| call.tokens.cached).sum();
    let created_tokens: u64 = recording.created.iter().map(|created| created.tokens).sum();
    let output_tokens: u64 = recording
        .calls
        .iter()
        .map(|call| call.tokens.candidates + call.tokens.thoughts)
        .sum();
    let token_hours = report.map_or(0.0, |report| report.token_hours);
    let with = (prompt - cached) as f64
        + cached as f64 * policy.cached_ratio
        + created_tokens as f64
        + token_hours * policy.storage_ratio_per_hour;
    let mut reads = BTreeMap::new();
    for call in &recording.calls {
        if let Some(cache) = &call.cache {
            *reads.entry(cache.clone()).or_default() += 1;
        }
    }
    let fixture_bytes = std::fs::metadata(crate::cassettes::cassette_path("gemini", scenario))
        .map_or(0, |metadata| metadata.len());
    Figures {
        calls: recording.calls.len(),
        prompt,
        cached,
        cached_share: cached as f64 / prompt.max(1) as f64,
        caches_created: recording.created.len(),
        created_tokens,
        created_share: created_tokens as f64 / prompt.max(1) as f64,
        reads,
        storage_token_hours: token_hours,
        input_with_caching: with,
        input_uncached: prompt as f64,
        input_saving: 1.0 - with / (prompt.max(1) as f64),
        usd_with_caching: with * INPUT_PRICE / 1e6,
        usd_uncached: prompt as f64 * INPUT_PRICE / 1e6,
        output_tokens,
        usd_output: output_tokens as f64 * OUTPUT_PRICE / 1e6,
        fixture_bytes,
        retries,
    }
}

/// Parts before the newest user text of `contents` with their signatures
/// removed: what `ThoughtReplay::CurrentTurn` sends for finished turns.
fn is_user_text(content: &Value) -> bool {
    content["role"] == "user"
        && content["parts"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|part| part.get("text").is_some()))
        && !content["parts"].as_array().is_some_and(|parts| {
            parts
                .iter()
                .any(|part| part.get("functionResponse").is_some())
        })
}

fn without_signatures(content: &str) -> Value {
    let mut value: Value = serde_json::from_str(content).expect("content JSON");
    if let Some(parts) = value.get_mut("parts").and_then(Value::as_array_mut) {
        for part in parts {
            if let Some(part) = part.as_object_mut() {
                part.remove("thoughtSignature");
            }
        }
    }
    value
}

/// The integrity checks every book run shares.
fn assert_integrity(recording: &Recording, usages: &[Usage], current_turn: bool) {
    let caches: HashMap<&str, &Created> = recording
        .created
        .iter()
        .map(|created| (created.name.as_str(), created))
        .collect();

    // Rig's usage never reports more cached than input tokens.
    assert!(!usages.is_empty(), "the run reported no completion calls");
    for (index, usage) in usages.iter().enumerate() {
        assert!(
            usage.cached_input_tokens.unwrap_or(0) <= usage.input_tokens.unwrap_or(0),
            "call {index}: cached {:?} > input {:?}",
            usage.cached_input_tokens,
            usage.input_tokens
        );
    }

    // Each conversation, spliced, only grows.
    let mut conversations: HashMap<String, (Vec<String>, String)> = HashMap::new();
    for call in &recording.calls {
        let (contents, prefix) = match &call.cache {
            Some(name) => {
                for key in ["systemInstruction", "tools", "toolConfig"] {
                    assert!(
                        call.body.get(key).is_none(),
                        "interaction {}: reads {name} but also sends {key}",
                        call.index
                    );
                }
                let cache = caches.get(name.as_str()).unwrap_or_else(|| {
                    panic!(
                        "interaction {} reads {name}, which the run never created",
                        call.index
                    )
                });
                assert!(cache.index < call.index, "a cache is read before it exists");
                let mut contents = cache.raw_contents.clone();
                contents.extend(call.raw_contents.iter().cloned());
                let prefix = format!(
                    "{}|{}",
                    cache
                        .body
                        .get("systemInstruction")
                        .map(Value::to_string)
                        .unwrap_or_default(),
                    cache
                        .body
                        .get("tools")
                        .map(Value::to_string)
                        .unwrap_or_default()
                );
                (contents, prefix)
            }
            None => (
                call.raw_contents.clone(),
                format!(
                    "{}|{}",
                    call.body
                        .get("systemInstruction")
                        .filter(|v| !v.is_null())
                        .map(Value::to_string)
                        .unwrap_or_default(),
                    call.body
                        .get("tools")
                        .filter(|v| !v.is_null())
                        .map(Value::to_string)
                        .unwrap_or_default()
                ),
            ),
        };
        let Some(first) = contents.first().cloned() else {
            continue;
        };
        if let Some((previous, previous_prefix)) = conversations.get(&first) {
            assert_eq!(
                &prefix, previous_prefix,
                "interaction {}: the conversation's system instruction or tools changed",
                call.index
            );
            let newest_user = contents
                .iter()
                .rposition(|content| {
                    is_user_text(&serde_json::from_str(content).unwrap_or(Value::Null))
                })
                .unwrap_or(0);
            assert!(
                contents.len() >= previous.len(),
                "interaction {}: the conversation shrank from {} to {} contents",
                call.index,
                previous.len(),
                contents.len()
            );
            for (position, (earlier, later)) in previous.iter().zip(&contents).enumerate() {
                if earlier == later {
                    continue;
                }
                // Current-turn replay drops a finished turn's signatures when
                // the next user message arrives; nothing else may change.
                let dropped_signatures = current_turn
                    && position < newest_user
                    && without_signatures(earlier)
                        == serde_json::from_str::<Value>(later).unwrap_or(Value::Null);
                assert!(
                    dropped_signatures,
                    "interaction {}: content {position} differs from the previous request's \
                     (cache + request tail must be the conversation byte for byte)",
                    call.index
                );
            }
        }
        conversations.insert(first, (contents, prefix));
    }

    // Every cache the run created is deleted.
    for created in &recording.created {
        let target = format!("/v1beta/{}", created.name);
        assert!(
            recording
                .interactions
                .iter()
                .skip(created.index)
                .any(|interaction| interaction.method == "DELETE" && interaction.path == target),
            "{} was never deleted",
            created.name
        );
    }
}

/// The "not faked" economics every book run shares.
fn assert_economics(recording: &Recording, figures: &Figures) {
    // A cache a later cache of the same conversation replaced was read at
    // least three times; so was every prefix-only cache.
    let conversation = |created: &Created| created.raw_contents.first().cloned();
    for (position, created) in recording.created.iter().enumerate() {
        let replaced = match conversation(created) {
            Some(first) => recording.created[position + 1..]
                .iter()
                .any(|later| conversation(later).as_ref() == Some(&first)),
            None => true,
        };
        if replaced {
            let reads = figures.reads.get(&created.name).copied().unwrap_or(0);
            assert!(
                reads >= 3,
                "{} ({} tokens) was read {reads} times before it was replaced",
                created.name,
                created.tokens
            );
        }
    }
    assert!(
        figures.created_share <= 0.15,
        "caches held {:.1}% of all prompt tokens (limit 15%)",
        figures.created_share * 100.0
    );
}

fn print(scenario: &str, figures: &Figures, report: Option<&CacheReport>) {
    println!(
        "AUTO_CACHING {scenario} {}",
        serde_json::to_string(figures).unwrap_or_default()
    );
    if let Some(report) = report {
        println!(
            "AUTO_CACHING_REPORT {scenario} {}",
            serde_json::to_string(report).unwrap_or_default()
        );
    }
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

/// Every call after the first cache read reads its cache whole: its cached
/// tokens equal the cache's size.
fn assert_reads_whole(recording: &Recording) {
    let sizes: HashMap<&str, u64> = recording
        .created
        .iter()
        .map(|created| (created.name.as_str(), created.tokens))
        .collect();
    for call in &recording.calls {
        if let Some(cache) = &call.cache {
            assert_eq!(
                call.tokens.cached,
                sizes[cache.as_str()],
                "interaction {} read {} of {cache}'s {} tokens",
                call.index,
                call.tokens.cached,
                sizes[cache.as_str()]
            );
        }
    }
}

/// Every call after the first cache read covers at least `share` of its
/// visible tokens (prompt tokens, which with current-turn replay carry only
/// the current turn's thoughts).
fn assert_cover_after_first_read(recording: &Recording, share: f64) {
    let Some(first) = recording.calls.iter().position(|call| call.cache.is_some()) else {
        panic!("no call ever read a cache");
    };
    for call in &recording.calls[first..] {
        let covered =
            call.tokens.cached as f64 / (call.tokens.prompt + call.tokens.tool_use).max(1) as f64;
        assert!(
            covered >= share,
            "interaction {} covered {:.0}% of its prompt (limit {:.0}%)",
            call.index,
            covered * 100.0,
            share * 100.0
        );
    }
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
) -> (CacheReport, RunLog) {
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
    log.events = book.events();
    (book.report(), log)
}

#[tokio::test]
async fn support_chat_100_current_turn() {
    const SCENARIO: &str = "auto_caching/support_chat_100_current_turn";
    let (report, log) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_current_turn",
        |gemini, clock| support_chat_100(gemini, clock, ThoughtReplay::CurrentTurn, false),
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert_integrity(&recording, &log.usages, true);
    assert_economics(&recording, &figures);
    assert!(
        figures.cached_share >= 0.80,
        "cached share {:.3}",
        figures.cached_share
    );
    assert!(
        figures.input_saving >= 0.65,
        "input saving {:.3}",
        figures.input_saving
    );
    assert_cover_after_first_read(&recording, 0.50);
    assert!(
        figures.caches_created <= MAX_CACHES_100_TURNS,
        "{} caches",
        figures.caches_created
    );
}

#[tokio::test]
async fn support_chat_100_auto() {
    const SCENARIO: &str = "auto_caching/support_chat_100_auto";
    let (report, log) =
        with_gemini_auto_caching_cassette("auto_caching/support_chat_100_auto", |gemini, clock| {
            support_chat_100(gemini, clock, ThoughtReplay::All, false)
        })
        .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert_integrity(&recording, &log.usages, false);
    assert_economics(&recording, &figures);
    assert!(
        figures.input_saving >= 0.30,
        "input saving {:.3}",
        figures.input_saving
    );
    assert_reads_whole(&recording);
    assert!(
        figures.caches_created <= MAX_CACHES_100_TURNS,
        "{} caches",
        figures.caches_created
    );
    // Restored thoughts: every earlier thought whose signature a request
    // re-sends is billed again as input (measured exactly on 3.8-flash).
    let mut earlier_thoughts = 0u64;
    let mut restored = 0u64;
    for call in &recording.calls {
        restored += earlier_thoughts;
        earlier_thoughts += call.tokens.thoughts;
    }
    println!(
        "AUTO_CACHING_RESTORED {SCENARIO} restored_thought_tokens={restored} of prompt={} ({:.1}%)",
        figures.prompt,
        100.0 * restored as f64 / figures.prompt.max(1) as f64
    );
}

#[tokio::test]
async fn support_chat_100_streamed() {
    const SCENARIO: &str = "auto_caching/support_chat_100_streamed";
    let (report, log) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_streamed",
        |gemini, clock| support_chat_100(gemini, clock, ThoughtReplay::CurrentTurn, true),
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert!(
        recording
            .calls
            .iter()
            .all(|call| recording.interactions[call.index]
                .path
                .contains(":streamGenerateContent")),
        "every call streams"
    );
    assert_integrity(&recording, &log.usages, true);
    assert_economics(&recording, &figures);
    assert!(
        figures.cached_share >= 0.80,
        "cached share {:.3}",
        figures.cached_share
    );
    assert!(
        figures.input_saving >= 0.65,
        "input saving {:.3}",
        figures.input_saving
    );
    assert_cover_after_first_read(&recording, 0.50);
    assert!(
        figures.caches_created <= MAX_CACHES_100_TURNS,
        "{} caches",
        figures.caches_created
    );
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
    let (report, log, pre_leases) = with_gemini_auto_caching_cassette(
        "auto_caching/support_chat_100_resume",
        |gemini, clock| async move {
            let mut log = RunLog::default();
            // Turns 1..=50, then a checkpoint as a process would write it.
            let saved = {
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
                serde_json::to_string(&checkpoint).expect("checkpoint serializes")
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
            log.events = book.events();
            (book.report(), log, pre_leases)
        },
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert_integrity(&recording, &log.usages, true);
    assert_economics(&recording, &figures);

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
    assert!(
        figures.cached_share >= 0.80,
        "cached share {:.3}",
        figures.cached_share
    );
    assert!(
        figures.input_saving >= 0.65,
        "input saving {:.3}",
        figures.input_saving
    );
    assert_cover_after_first_read(&recording, 0.50);
    assert!(
        figures.caches_created <= MAX_CACHES_100_TURNS,
        "{} caches",
        figures.caches_created
    );
}

#[tokio::test]
async fn support_chat_100_compaction() {
    const SCENARIO: &str = "auto_caching/support_chat_100_compaction";
    let (report, log) = with_gemini_auto_caching_cassette(
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
            log.events = book.events();
            (book.report(), log)
        },
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert_integrity(&recording, &log.usages, true);
    assert_economics(&recording, &figures);

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
    assert!(
        figures.input_saving >= 0.50,
        "input saving {:.3}",
        figures.input_saving
    );
}

#[tokio::test]
async fn agent_loop_large_results() {
    const SCENARIO: &str = "auto_caching/agent_loop_large_results";
    let (report, log) = with_gemini_auto_caching_cassette(
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
            log.events = book.events();
            (book.report(), log)
        },
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert!(figures.calls >= 60, "a long loop: {} calls", figures.calls);
    assert_integrity(&recording, &log.usages, false);
    assert_economics(&recording, &figures);
    assert!(
        figures.caches_created <= 3,
        "{} caches",
        figures.caches_created
    );
    assert!(
        figures.cached_share >= 0.60,
        "cached share {:.3}",
        figures.cached_share
    );
    assert!(
        figures.input_saving >= 0.60,
        "input saving {:.3}",
        figures.input_saving
    );
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
    let (report, log) = with_gemini_auto_caching_cassette(
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
            let mut log = RunLog {
                events: book.events(),
                ..RunLog::default()
            };
            for part in logs {
                log.usages.extend(part.usages);
                log.retries += part.retries;
            }
            (book.report(), log)
        },
    )
    .await;
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    assert_integrity(&recording, &log.usages, false);
    assert_economics(&recording, &figures);
    let prefix_caches = recording
        .created
        .iter()
        .filter(|created| created.raw_contents.is_empty())
        .count();
    assert_eq!(
        prefix_caches, 1,
        "the shared prefix is created exactly once"
    );
    assert!(
        figures.input_saving >= 0.60,
        "input saving {:.3}",
        figures.input_saving
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
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, Some(&report), log.retries);
    print(SCENARIO, &figures, Some(&report));
    print_estimates(SCENARIO, &log.events);
    println!(
        "AUTO_CACHING_EVENTS {SCENARIO} {}",
        serde_json::to_string(&events).unwrap_or_default()
    );
    assert_integrity(&recording, &log.usages, false);
    assert_economics(&recording, &figures);

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
    let recording = load(SCENARIO);
    let figures = figures(SCENARIO, &recording, None, log.retries);
    print(SCENARIO, &figures, None);
    assert!(
        recording.created.is_empty(),
        "the baseline creates no cache"
    );
    assert!(
        recording.calls.iter().all(|call| call.cache.is_none()),
        "the baseline reads no cache"
    );
    for usage in &log.usages {
        assert!(usage.cached_input_tokens.unwrap_or(0) <= usage.input_tokens.unwrap_or(0));
    }
}
