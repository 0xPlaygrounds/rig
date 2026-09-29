//! Long-run prompt-cache checks shared by every provider: drive a recorded
//! run, read its traffic back, compute its figures and assert that caching
//! held up economically over the whole run.
//!
//! A short cache probe proves a hit on turn 2. A long run can still degrade:
//! a marker that stops moving, a prefix that drifts after turn 40, a cache key
//! that changes, or cache writes that cost more than the reads save. Each
//! provider reports its cache traffic differently, so one small reader per
//! wire ([`CacheWire`]) turns every recorded call into the same counters, and
//! everything after that is shared: the figures a run prints as one
//! `CACHE_LONGRUN <provider>/<scenario> {json}` line, and the checks in
//! [`check`].

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use rig_agent::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig_cassette::http::CassetteClock;
use rig_core::completion::{CacheCost, CacheRates, Message, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The support handbook every long-run support chat uses as its preamble.
pub const SUPPORT_PREAMBLE: &str = include_str!("cache_longrun/support_preamble.md");

// ---------------------------------------------------------------------------
// The workload.

/// [`LookupOrder`]'s arguments.
#[derive(Deserialize)]
pub struct OrderArgs {
    order_id: String,
}

/// Looks an order up. Deterministic: the same id always gives the same
/// status, date and carrier, so a re-recording sees the same facts.
pub struct LookupOrder;

const STATUSES: [&str; 6] = [
    "processing",
    "shipped",
    "delivered",
    "refunded",
    "cancelled",
    "returned",
];

impl rig_core::tool::Tool for LookupOrder {
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
        _context: &mut rig_core::tool::ToolContext,
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

/// The customer's message for `turn`, about order `<prefix>-<turn>`.
pub fn question(turn: usize, prefix: &str) -> String {
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

// ---------------------------------------------------------------------------
// Driving a run.

/// What a run saw from the test's side of the wire.
#[derive(Default)]
pub struct RunLog {
    /// Rig's `Usage` for every successful completion call.
    pub usages: Vec<Usage>,
    /// Turns retried after a 5xx or 429.
    pub retries: usize,
}

fn retryable(status: Option<http::StatusCode>) -> bool {
    status.is_some_and(|status| status.is_server_error() || status.as_u16() == 429)
}

/// Record-only pauses between the attempts of one turn.
const BACKOFF: [u64; 3] = [2, 5, 10];

/// One chat turn, retried up to three times on a 5xx or 429 with a
/// record-only pause between attempts. The failed attempts are recorded like
/// any other calls, so replay matches.
///
/// # Panics
/// When the turn still fails, or fails with anything else.
pub async fn chat(
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

fn streaming_status(error: &StreamingError) -> Option<http::StatusCode> {
    match error {
        StreamingError::Completion(error) => error.provider_response_status(),
        StreamingError::Report(report) => report
            .http_status
            .and_then(|status| http::StatusCode::from_u16(status).ok()),
        StreamingError::Prompt(error) => error.provider_response_status(),
    }
}

/// [`chat`] over the provider's streaming endpoint.
///
/// # Panics
/// When the turn still fails, or fails with anything else.
pub async fn chat_streamed(
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

// ---------------------------------------------------------------------------
// Reading a recording back.

/// The wire a run was recorded on, and so how its traffic reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheWire {
    /// Gemini `generateContent` / `streamGenerateContent`, with
    /// `cachedContents` resources.
    Gemini,
    /// Anthropic Messages.
    Anthropic,
    /// OpenAI Chat Completions.
    OpenAiChat,
    /// OpenAI Responses.
    OpenAiResponses,
}

impl CacheWire {
    /// The fixture directory under `fixtures/cassettes/`.
    pub fn provider(self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Anthropic => "anthropic",
            Self::OpenAiChat | Self::OpenAiResponses => "openai",
        }
    }

    /// Whether `path` is a completion call on this wire.
    fn is_call(self, path: &str) -> bool {
        match self {
            Self::Gemini => {
                path.contains(":generateContent") || path.contains(":streamGenerateContent")
            }
            Self::Anthropic => path.ends_with("/messages"),
            Self::OpenAiChat => path.ends_with("/chat/completions"),
            Self::OpenAiResponses => path.ends_with("/responses"),
        }
    }

    /// The call's usage from its response body (JSON, or SSE frames), as rig
    /// reports it: input (cache reads and writes included), cached reads,
    /// cache writes, output (reasoning included) and their total, plus the
    /// thoughts Gemini counts apart from output.
    fn usage(self, response: &str) -> Option<(Usage, u64)> {
        let frames: Vec<Value> = match serde_json::from_str::<Value>(response) {
            Ok(value) => vec![value],
            Err(_) => response
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
                .collect(),
        };
        let count = |value: &Value, pointer: &str| value.pointer(pointer).and_then(Value::as_u64);
        let read = match self {
            Self::Gemini => frames
                .iter()
                .filter_map(|frame| frame.get("usageMetadata"))
                .next_back()
                .map(|usage| {
                    let thoughts = count(usage, "/thoughtsTokenCount").unwrap_or(0);
                    let usage = Usage {
                        input_tokens: Some(
                            count(usage, "/promptTokenCount").unwrap_or(0)
                                + count(usage, "/toolUsePromptTokenCount").unwrap_or(0),
                        ),
                        cached_input_tokens: Some(
                            count(usage, "/cachedContentTokenCount").unwrap_or(0),
                        ),
                        output_tokens: Some(
                            count(usage, "/candidatesTokenCount").unwrap_or(0) + thoughts,
                        ),
                        ..Usage::default()
                    };
                    (usage, thoughts)
                }),
            // A stream's `message_start` carries the input counters and its
            // `message_delta` the final output (and repeats the cache
            // counters); later frames win. Anthropic counts cache reads and
            // writes beside `input_tokens`; rig's input includes them.
            Self::Anthropic => {
                let mut found = false;
                let mut usage = Usage::default();
                for frame in &frames {
                    let Some(counters) = frame
                        .get("usage")
                        .or_else(|| frame.pointer("/message/usage"))
                    else {
                        continue;
                    };
                    found = true;
                    let pick = |key: &str, into: &mut Option<u64>| {
                        if let Some(value) = counters.get(key).and_then(Value::as_u64) {
                            *into = Some(value);
                        }
                    };
                    pick("input_tokens", &mut usage.input_tokens);
                    pick("cache_read_input_tokens", &mut usage.cached_input_tokens);
                    pick(
                        "cache_creation_input_tokens",
                        &mut usage.cache_creation_input_tokens,
                    );
                    pick("output_tokens", &mut usage.output_tokens);
                }
                usage.input_tokens = usage.input_tokens.map(|uncached| {
                    uncached
                        + usage.cached_input_tokens.unwrap_or(0)
                        + usage.cache_creation_input_tokens.unwrap_or(0)
                });
                found.then_some((usage, 0))
            }
            Self::OpenAiChat => frames
                .iter()
                .filter_map(|frame| frame.get("usage").filter(|usage| !usage.is_null()))
                .next_back()
                .map(|usage| {
                    let usage = Usage {
                        input_tokens: count(usage, "/prompt_tokens"),
                        cached_input_tokens: count(usage, "/prompt_tokens_details/cached_tokens"),
                        cache_creation_input_tokens: count(
                            usage,
                            "/prompt_tokens_details/cache_write_tokens",
                        ),
                        output_tokens: count(usage, "/completion_tokens"),
                        ..Usage::default()
                    };
                    (usage, 0)
                }),
            Self::OpenAiResponses => frames
                .iter()
                .filter_map(|frame| {
                    frame
                        .pointer("/response/usage")
                        .or_else(|| frame.get("usage"))
                        .filter(|usage| !usage.is_null())
                })
                .next_back()
                .map(|usage| {
                    let usage = Usage {
                        input_tokens: count(usage, "/input_tokens"),
                        cached_input_tokens: count(usage, "/input_tokens_details/cached_tokens"),
                        cache_creation_input_tokens: count(
                            usage,
                            "/input_tokens_details/cache_write_tokens",
                        ),
                        output_tokens: count(usage, "/output_tokens"),
                        ..Usage::default()
                    };
                    (usage, 0)
                }),
        };
        read.map(|(mut usage, thoughts)| {
            usage.total_tokens = usage
                .input_tokens
                .zip(usage.output_tokens)
                .map(|(input, output)| input + output);
            (usage, thoughts)
        })
    }

    /// The request's (or a Gemini cache's) messages.
    fn messages(self, body: &Value) -> &[Value] {
        let messages = match self {
            Self::Gemini => body.get("contents"),
            Self::Anthropic | Self::OpenAiChat => body.get("messages"),
            Self::OpenAiResponses => body.get("input"),
        };
        messages
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    /// The request's first user message, which names its conversation.
    fn first_user_message(self, body: &Value) -> Option<String> {
        self.messages(body)
            .iter()
            .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
            .map(Value::to_string)
    }

    /// How many customer messages the request carries: its turn number.
    fn user_turns(self, body: &Value) -> usize {
        self.messages(body)
            .iter()
            .filter(|message| match self {
                Self::Gemini => is_user_text(message),
                Self::Anthropic => {
                    message["role"] == "user"
                        && (message["content"].is_string()
                            || message["content"].as_array().is_some_and(|blocks| {
                                blocks.iter().any(|block| block["type"] == "text")
                            }))
                }
                Self::OpenAiChat | Self::OpenAiResponses => message["role"] == "user",
            })
            .count()
    }
}

/// One recorded exchange.
#[derive(Debug)]
pub struct Interaction {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Request body.
    pub request: String,
    /// Response status.
    pub status: u16,
    /// Response body.
    pub response: String,
}

fn interactions(provider: &str, scenario: &str) -> Vec<Interaction> {
    let path = crate::cassettes::cassette_path(provider, scenario);
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

/// One successful completion call.
pub struct Call {
    /// Its position among the recording's interactions.
    pub index: usize,
    /// Its usage as rig reports it, read from the wire.
    pub usage: Usage,
    /// Gemini thoughts generated (also inside `usage.output_tokens`).
    pub thoughts: u64,
    /// The conversation it continues: its first user message, or its cache's.
    pub conversation: String,
    /// The customer messages its conversation holds so far: its turn.
    pub turn: usize,
    /// The Gemini cache it reads (`cachedContent`).
    pub cache: Option<String>,
    /// The OpenAI `prompt_cache_key` it sends.
    pub cache_key: Option<String>,
    /// The request body.
    pub body: Value,
    /// Gemini: the request's `contents`, each as sent.
    pub raw_contents: Vec<String>,
}

/// One Gemini cache the run created.
pub struct Created {
    /// Its position among the recording's interactions.
    pub index: usize,
    /// `cachedContents/<id>`.
    pub name: String,
    /// Its size, as Gemini counted it.
    pub tokens: u64,
    /// The create request's body.
    pub body: Value,
    /// The create request's `contents`, each as sent.
    pub raw_contents: Vec<String>,
}

/// A recording read back.
pub struct Recording {
    /// The wire it was recorded on.
    pub wire: CacheWire,
    /// Every exchange, in order.
    pub interactions: Vec<Interaction>,
    /// The successful completion calls, in order.
    pub calls: Vec<Call>,
    /// The Gemini caches created, in order.
    pub created: Vec<Created>,
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

/// Read `<wire's provider>/<scenario>` back.
///
/// # Panics
/// When the fixture is unreadable, or a successful call reports no usage.
pub fn load(wire: CacheWire, scenario: &str) -> Recording {
    let interactions = interactions(wire.provider(), scenario);
    let mut calls = Vec::new();
    let mut created: Vec<Created> = Vec::new();
    for (index, interaction) in interactions.iter().enumerate() {
        if wire.is_call(&interaction.path) {
            if interaction.status != 200 {
                continue;
            }
            let body: Value = serde_json::from_str(&interaction.request).expect("request JSON");
            let cache = body
                .get("cachedContent")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let read = cache
                .as_ref()
                .and_then(|name| created.iter().find(|created| &created.name == name));
            let conversation = read
                .and_then(|created| wire.first_user_message(&created.body))
                .or_else(|| wire.first_user_message(&body))
                .unwrap_or_default();
            let turn =
                read.map_or(0, |created| wire.user_turns(&created.body)) + wire.user_turns(&body);
            let (usage, thoughts) = wire
                .usage(&interaction.response)
                .unwrap_or_else(|| panic!("interaction {index}: a successful call reports usage"));
            calls.push(Call {
                index,
                usage,
                thoughts,
                conversation,
                turn,
                cache,
                cache_key: body
                    .get("prompt_cache_key")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                raw_contents: raw_contents(&interaction.request),
                body,
            });
        } else if wire == CacheWire::Gemini
            && interaction.method == "POST"
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
        wire,
        interactions,
        calls,
        created,
    }
}

// ---------------------------------------------------------------------------
// Figures and checks.

/// A run's asserted limits.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Limits {
    /// The least input saving over the run, counting writes, reads and
    /// storage at the provider's rates.
    pub min_saving: f64,
    /// The least cached share of every call from its conversation's first
    /// cache read on; also asserts no call after that read reads nothing.
    /// `None` for a run that deliberately goes uncached mid-run.
    pub min_call_share: Option<f64>,
    /// The most cache-write (or creation) tokens relative to prompt tokens.
    pub max_writes_share: Option<f64>,
}

/// One long run: where it was recorded, how it is priced and what it asserts.
pub struct LongRun<'a> {
    /// The wire it was recorded on.
    pub wire: CacheWire,
    /// Its fixture, relative to the provider's directory.
    pub scenario: &'a str,
    /// The model it ran on.
    pub model: &'a str,
    /// Input-side prices.
    pub rates: CacheRates,
    /// Output price (thinking included), USD per 1M tokens.
    pub output_price: f64,
    /// The asserted limits; `None` for a reporting-only baseline.
    pub limits: Option<Limits>,
    /// Gemini: finished turns may lose their thought signatures
    /// (current-turn replay), so a conversation may change there.
    pub drops_signatures: bool,
}

/// Whole-run figures, printed as the run's `CACHE_LONGRUN` line.
#[derive(Debug, Serialize)]
pub struct Figures {
    /// `<provider>/<scenario>`.
    pub run: String,
    /// The model.
    pub model: String,
    /// Successful completion calls.
    pub calls: usize,
    /// The fixture's size.
    pub fixture_bytes: u64,
    /// Turns retried after a 5xx or 429.
    pub retries: usize,
    /// First to last reading of the session clock, when it has readings.
    pub recording_seconds: Option<u64>,
    /// Prompt tokens: the calls' `input_tokens`.
    pub prompt: u64,
    /// Cached-read tokens.
    pub cached: u64,
    /// Cache-write tokens (Anthropic, OpenAI) or tokens put into caches (Gemini).
    pub writes: u64,
    /// Cached reads over prompt tokens.
    pub cached_share: f64,
    /// Calls after the first that read any cached tokens, over calls after the first.
    pub hit_rate: f64,
    /// Calls before the first that read cached tokens.
    pub warm_up_calls: usize,
    /// The smallest cached share of any call from its conversation's first read on.
    pub min_call_share: f64,
    /// What a perfect cache could have served: each call's prompt tokens up
    /// to its conversation's previous prompt (a conversation's first call:
    /// what it read).
    pub cacheable: u64,
    /// Cached reads over cacheable tokens.
    pub cache_accuracy: f64,
    /// Where the reads fell short of cacheable tokens.
    pub shortfall: Shortfall,
    /// Writes over prompt tokens.
    pub writes_share: f64,
    /// Gemini: caches created.
    pub caches_created: Option<usize>,
    /// Gemini: reads of each cache, by name.
    pub reads: Option<BTreeMap<String, u64>>,
    /// Gemini: Σ tokens × hours the caches lived.
    pub storage_token_hours: f64,
    /// Gemini with every signature replayed: earlier thoughts re-billed as
    /// input, over prompt tokens.
    pub restored_thought_share: Option<f64>,
    /// 1 - input $ with caching / input $ uncached.
    pub input_saving: f64,
    /// Input $ as billed.
    pub usd_with_caching: f64,
    /// Input $ had every prompt token been uncached.
    pub usd_uncached: f64,
    /// Output tokens, thinking included.
    pub output_tokens: u64,
    /// Output $.
    pub usd_output: f64,
    /// The rates the dollars use.
    pub rates: CacheRates,
    /// The asserted limits.
    pub limits: Option<Limits>,
    /// The first [`BASELINE_TURNS`] turns alone, for a run longer than that.
    pub first_turns: Option<Window>,
}

/// Where a run's cache reads fell short of what a perfect cache could have
/// served, call by call within each conversation.
#[derive(Debug, Default, Serialize)]
pub struct Shortfall {
    /// Cacheable tokens not read before the conversation's first read.
    pub warm_up: u64,
    /// Cacheable tokens not read from the conversation's first read on.
    pub after_first_read: u64,
    /// Calls from the first read on that read exactly what the
    /// conversation's previous call read, more than [`CLOSING_TOKENS`] short
    /// of their cacheable tokens: the cache stayed at an older prefix.
    pub stalled_calls: usize,
    /// Calls from the first read on that read less than the conversation's
    /// previous call: a prefix that moved, or a cache that expired or was
    /// evicted.
    pub dropped_calls: usize,
}

/// Tokens after a request's last block that its cache write does not hold
/// (measured: 2 to 4 on Anthropic and OpenAI), so a call reading its
/// predecessor's prefix that much short has not stalled.
pub const CLOSING_TOKENS: u64 = 8;

fn shortfall(recording: &Recording, billed: &[CacheCost], cacheable: &[u64]) -> Shortfall {
    let mut short = Shortfall::default();
    // Per conversation: the previous call's reads, once it has read.
    let mut previous: HashMap<&str, u64> = HashMap::new();
    for ((call, billed), cacheable) in recording.calls.iter().zip(billed).zip(cacheable) {
        let reads = billed.cache_reads;
        let missed = cacheable.saturating_sub(reads);
        match previous.get(call.conversation.as_str()).copied() {
            None if reads == 0 => short.warm_up += missed,
            None => short.after_first_read += missed,
            Some(before) => {
                short.after_first_read += missed;
                if missed > CLOSING_TOKENS && reads == before {
                    short.stalled_calls += 1;
                } else if reads < before {
                    short.dropped_calls += 1;
                }
            }
        }
        if reads > 0 || previous.contains_key(call.conversation.as_str()) {
            previous.insert(call.conversation.as_str(), reads);
        }
    }
    short
}

/// How many turns a baseline runs, and so the window of a longer run set
/// beside it.
pub const BASELINE_TURNS: usize = 30;

/// A run's first turns, priced like the whole run. Gemini storage is left
/// out: the book reports it for the run, not per turn.
#[derive(Debug, Serialize)]
pub struct Window {
    /// Turns covered.
    pub turns: usize,
    /// Successful calls in them.
    pub calls: usize,
    /// Prompt tokens.
    pub prompt: u64,
    /// Cached reads over prompt tokens.
    pub cached_share: f64,
    /// 1 - input $ with caching / input $ uncached.
    pub input_saving: f64,
    /// Input $ as billed.
    pub usd_with_caching: f64,
    /// Input $ had every prompt token been uncached.
    pub usd_uncached: f64,
}

fn clock_span(provider: &str, scenario: &str) -> Option<u64> {
    #[derive(Deserialize)]
    struct Readings {
        readings: Vec<u64>,
    }
    let sidecar =
        rig_cassette::http::clock_sidecar(&crate::cassettes::cassette_path(provider, scenario));
    let text = std::fs::read_to_string(sidecar).ok()?;
    let readings = serde_json::from_str::<Readings>(&text).ok()?.readings;
    Some(readings.last()? - readings.first()?)
}

/// Per call: the prompt tokens a perfect cache could have served, which is
/// the conversation's previous prompt, capped at this one. A conversation's
/// first call counts what it read: only a cache written before the run can
/// serve it, and its newest content cannot be told from its prefix without
/// a tokenizer.
fn cacheable(recording: &Recording, billed: &[CacheCost]) -> Vec<u64> {
    let mut previous: HashMap<&str, u64> = HashMap::new();
    recording
        .calls
        .iter()
        .zip(billed)
        .map(|(call, billed)| {
            let prompt = billed.prompt_tokens();
            previous
                .insert(call.conversation.as_str(), prompt)
                .map_or(billed.cache_reads, |earlier| earlier.min(prompt))
        })
        .collect()
}

/// The calls of turns `1..=turns`: what the run's first turns cost, to set
/// beside a baseline of that many turns.
fn window(
    run: &LongRun<'_>,
    recording: &Recording,
    billed: &[CacheCost],
    turns: usize,
) -> Option<Window> {
    if recording.calls.iter().all(|call| call.turn <= turns) {
        return None;
    }
    // Up to the first call past the window: a compacted conversation counts
    // its turns again from one.
    let inside: Vec<(&Call, &CacheCost)> = recording
        .calls
        .iter()
        .zip(billed)
        .take_while(|(call, _)| call.turn <= turns)
        .collect();
    let last = inside.last().map_or(0, |(call, _)| call.index);
    let calls: CacheCost = inside.iter().map(|(_, billed)| **billed).sum();
    // Gemini caches created by then; their storage is not split by time.
    let created: u64 = recording
        .created
        .iter()
        .filter(|created| created.index < last)
        .map(|created| created.tokens)
        .sum();
    let total = calls
        + CacheCost {
            cache_writes: created,
            ..CacheCost::default()
        };
    let usd_with_caching = total.usd(&run.rates);
    let usd_uncached = calls.uncached_usd(&run.rates);
    Some(Window {
        turns,
        calls: inside.len(),
        prompt: calls.prompt_tokens(),
        cached_share: total.cache_reads as f64 / calls.prompt_tokens().max(1) as f64,
        input_saving: 1.0 - usd_with_caching / usd_uncached.max(f64::MIN_POSITIVE),
        usd_with_caching,
        usd_uncached,
    })
}

/// Compute the run's figures. `resources` is what the provider's cache
/// resources cost beside the calls (Gemini: creation and storage).
fn figures(
    run: &LongRun<'_>,
    recording: &Recording,
    resources: CacheCost,
    retries: usize,
) -> Figures {
    let billed: Vec<CacheCost> = recording
        .calls
        .iter()
        .map(|call| CacheCost::from_usage(&call.usage))
        .collect();
    let calls: CacheCost = billed.iter().copied().sum();
    let total = calls + resources;
    let prompt = calls.prompt_tokens();
    let share = |part: u64, whole: u64| part as f64 / whole.max(1) as f64;

    let hits_after_first = billed
        .iter()
        .skip(1)
        .filter(|call| call.cache_reads > 0)
        .count();
    let warm_up_calls = billed
        .iter()
        .position(|call| call.cache_reads > 0)
        .unwrap_or(billed.len());
    let min_call_share = after_first_read(recording, &billed)
        .map(|(_, call)| share(call.cache_reads, call.prompt_tokens()))
        .fold(None, |least: Option<f64>, share| {
            Some(least.map_or(share, |least| least.min(share)))
        })
        .unwrap_or(0.0);
    let per_call = cacheable(recording, &billed);
    let cacheable: u64 = per_call.iter().sum();

    let (caches_created, reads) = if run.wire == CacheWire::Gemini {
        let mut reads = BTreeMap::new();
        for call in &recording.calls {
            if let Some(cache) = &call.cache {
                *reads.entry(cache.clone()).or_default() += 1;
            }
        }
        (Some(recording.created.len()), Some(reads))
    } else {
        (None, None)
    };
    let restored_thought_share =
        (run.wire == CacheWire::Gemini && !run.drops_signatures).then(|| {
            let mut earlier: HashMap<&str, u64> = HashMap::new();
            let mut restored = 0u64;
            for call in &recording.calls {
                let thoughts = earlier.entry(call.conversation.as_str()).or_default();
                restored += *thoughts;
                *thoughts += call.thoughts;
            }
            share(restored, prompt)
        });

    let output_tokens: u64 = recording
        .calls
        .iter()
        .map(|call| call.usage.output_tokens.unwrap_or(0))
        .sum();
    // Uncached, the run would have sent its calls' prompts and nothing else.
    let usd_with_caching = total.usd(&run.rates);
    let usd_uncached = calls.uncached_usd(&run.rates);
    let provider = run.wire.provider();
    Figures {
        run: format!("{provider}/{}", run.scenario),
        model: run.model.to_owned(),
        calls: recording.calls.len(),
        fixture_bytes: std::fs::metadata(crate::cassettes::cassette_path(provider, run.scenario))
            .map_or(0, |metadata| metadata.len()),
        retries,
        recording_seconds: clock_span(provider, run.scenario),
        prompt,
        cached: total.cache_reads,
        writes: total.cache_writes,
        cached_share: share(total.cache_reads, prompt),
        hit_rate: share(
            hits_after_first as u64,
            billed.len().saturating_sub(1) as u64,
        ),
        warm_up_calls,
        min_call_share,
        cacheable,
        cache_accuracy: share(total.cache_reads, cacheable),
        shortfall: shortfall(recording, &billed, &per_call),
        writes_share: share(total.cache_writes, prompt),
        caches_created,
        reads,
        storage_token_hours: total.storage_token_hours,
        restored_thought_share,
        input_saving: 1.0 - usd_with_caching / usd_uncached.max(f64::MIN_POSITIVE),
        usd_with_caching,
        usd_uncached,
        output_tokens,
        usd_output: output_tokens as f64 * run.output_price / 1e6,
        rates: run.rates,
        limits: run.limits,
        first_turns: window(run, recording, &billed, BASELINE_TURNS),
    }
}

/// Every call from its conversation's first cache read on, with its billing.
fn after_first_read<'a>(
    recording: &'a Recording,
    billed: &'a [CacheCost],
) -> impl Iterator<Item = (&'a Call, &'a CacheCost)> {
    let mut reading: Vec<&str> = Vec::new();
    recording
        .calls
        .iter()
        .zip(billed)
        .filter(move |(call, billed)| {
            if reading.contains(&call.conversation.as_str()) {
                return true;
            }
            if billed.cache_reads > 0 {
                reading.push(call.conversation.as_str());
                return true;
            }
            false
        })
}

/// Print one limit a run asserts beyond its [`Limits`], as a
/// `CACHE_LONGRUN_LIMIT <provider>/<scenario> <limit>` line, so a table of
/// thresholds is generated from the tests.
pub fn print_limit(figures: &Figures, limit: &str) {
    println!("CACHE_LONGRUN_LIMIT {} {limit}", figures.run);
}

/// Read the run back, compute and print its figures, and run every shared
/// check that applies to its wire. `resources` is the cost of the provider's
/// cache resources beside the calls (Gemini's `CacheReport`: creation and
/// storage).
///
/// # Panics
/// When a check fails.
pub fn check(
    run: &LongRun<'_>,
    log: &RunLog,
    resources: Option<CacheCost>,
) -> (Recording, Figures) {
    let recording = load(run.wire, run.scenario);
    let resources = resources.unwrap_or_default();
    let figures = figures(run, &recording, resources, log.retries);
    println!(
        "CACHE_LONGRUN {} {}",
        figures.run,
        serde_json::to_string(&figures).unwrap_or_default()
    );

    assert_usage_matches_wire(&recording, log);
    match run.wire {
        CacheWire::Gemini => {
            assert_gemini_integrity(&recording, run.drops_signatures);
            let created: u64 = recording.created.iter().map(|created| created.tokens).sum();
            assert_eq!(
                created, resources.cache_writes,
                "the caches the recording created and the tokens priced as creation differ"
            );
            assert_gemini_economics(&recording, &figures);
        }
        CacheWire::OpenAiChat | CacheWire::OpenAiResponses => {
            if run.limits.is_some() {
                assert_constant_cache_key(&recording);
            }
        }
        CacheWire::Anthropic => {}
    }
    if let Some(limits) = run.limits {
        assert!(
            figures.input_saving >= limits.min_saving,
            "{}: input saving {:.3} (limit {:.3})",
            figures.run,
            figures.input_saving,
            limits.min_saving
        );
        if let Some(floor) = limits.min_call_share {
            assert_no_collapse(&recording);
            assert_call_share(&recording, floor);
        }
        if let Some(cap) = limits.max_writes_share {
            assert!(
                figures.writes_share <= cap,
                "{}: cache writes are {:.1}% of prompt tokens (limit {:.1}%)",
                figures.run,
                figures.writes_share * 100.0,
                cap * 100.0
            );
        }
    }
    (recording, figures)
}

/// Rig's `Usage` for every call is what the wire reported: input, cached
/// reads, cache writes, output and total. The recording may hold more
/// successful calls than rig reported, but only in a run that retried a
/// turn: a turn that failed after a successful call reports none of its
/// calls. Reads and writes never exceed input.
fn assert_usage_matches_wire(recording: &Recording, log: &RunLog) {
    let key = |usage: &Usage| {
        (
            usage.input_tokens.unwrap_or(0),
            usage.cached_input_tokens.unwrap_or(0),
            usage.cache_creation_input_tokens.unwrap_or(0),
            usage.output_tokens.unwrap_or(0),
            usage.total_tokens.unwrap_or(0),
        )
    };
    assert!(
        !log.usages.is_empty(),
        "the run reported no completion calls"
    );
    // Concurrent conversations finish in any order, so match as multisets.
    let mut wire: Vec<_> = recording
        .calls
        .iter()
        .map(|call| key(&call.usage))
        .collect();
    for usage in &log.usages {
        let usage = key(usage);
        let position = wire.iter().position(|call| *call == usage);
        assert!(
            position.is_some(),
            "rig reported usage (input, cached, written, output, total) {usage:?}, which no \
             recorded call has"
        );
        if let Some(position) = position {
            wire.swap_remove(position);
        }
    }
    assert!(
        wire.is_empty() || log.retries > 0,
        "{} recorded calls have no usage in rig's report, and no turn was retried: {wire:?}",
        wire.len()
    );
    for call in &recording.calls {
        let (input, cached, written, _, _) = key(&call.usage);
        assert!(
            cached + written <= input,
            "interaction {}: cached {cached} + written {written} > input {input}",
            call.index
        );
    }
}

/// After a conversation's first cache read, no call of it reads nothing: a
/// zero there is a prefix that moved.
fn assert_no_collapse(recording: &Recording) {
    let billed: Vec<CacheCost> = recording
        .calls
        .iter()
        .map(|call| CacheCost::from_usage(&call.usage))
        .collect();
    for (call, billed) in after_first_read(recording, &billed) {
        assert!(
            billed.cache_reads > 0,
            "interaction {}: read no cached tokens after its conversation's first read \
             (the prefix moved)",
            call.index
        );
    }
}

/// Every call from its conversation's first cache read on covers at least
/// `floor` of its prompt tokens.
fn assert_call_share(recording: &Recording, floor: f64) {
    let billed: Vec<CacheCost> = recording
        .calls
        .iter()
        .map(|call| CacheCost::from_usage(&call.usage))
        .collect();
    assert!(
        billed.iter().any(|call| call.cache_reads > 0),
        "no call ever read a cache"
    );
    for (call, billed) in after_first_read(recording, &billed) {
        let covered = billed.cache_reads as f64 / billed.prompt_tokens().max(1) as f64;
        assert!(
            covered >= floor,
            "interaction {} covered {:.0}% of its prompt (limit {:.0}%)",
            call.index,
            covered * 100.0,
            floor * 100.0
        );
    }
}

/// Every request carries the same `prompt_cache_key`.
fn assert_constant_cache_key(recording: &Recording) {
    let first = recording
        .calls
        .first()
        .and_then(|call| call.cache_key.clone())
        .expect("the run sends a prompt_cache_key");
    for call in &recording.calls {
        assert_eq!(
            call.cache_key.as_deref(),
            Some(first.as_str()),
            "interaction {}: the prompt_cache_key changed",
            call.index
        );
    }
}

// ---------------------------------------------------------------------------
// Gemini's cache resources.

/// A user message that carries text rather than a function response.
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

/// `content` with its thought signatures removed: what
/// `ThoughtReplay::CurrentTurn` sends for a finished turn.
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

/// A request that reads a cache carries none of what the cache holds; the
/// cache's contents followed by the request's are the conversation byte for
/// byte; every cache the run created is deleted.
fn assert_gemini_integrity(recording: &Recording, drops_signatures: bool) {
    let caches: HashMap<&str, &Created> = recording
        .created
        .iter()
        .map(|created| (created.name.as_str(), created))
        .collect();

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
                let dropped_signatures = drops_signatures
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

/// A run could look well cached by creating a new cache before every call
/// and paying for it each time. So every cache a later one of the same
/// conversation replaced (and every prefix-only cache) was read at least
/// three times, and creation stays at or below 15% of prompt tokens.
fn assert_gemini_economics(recording: &Recording, figures: &Figures) {
    let conversation = |created: &Created| created.raw_contents.first().cloned();
    let reads = figures.reads.clone().unwrap_or_default();
    for (position, created) in recording.created.iter().enumerate() {
        let replaced = match conversation(created) {
            Some(first) => recording.created[position + 1..]
                .iter()
                .any(|later| conversation(later).as_ref() == Some(&first)),
            None => true,
        };
        if replaced {
            let reads = reads.get(&created.name).copied().unwrap_or(0);
            assert!(
                reads >= 3,
                "{} ({} tokens) was read {reads} times before it was replaced",
                created.name,
                created.tokens
            );
        }
    }
    assert!(
        figures.writes_share <= 0.15,
        "caches held {:.1}% of all prompt tokens (limit 15%)",
        figures.writes_share * 100.0
    );
}
