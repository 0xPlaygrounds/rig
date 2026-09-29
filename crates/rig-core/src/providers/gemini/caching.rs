//! Automatic explicit caching for long Gemini runs.
//!
//! [`Caching`] is a transport that wraps another. Every `generateContent`
//! and `streamGenerateContent` request passes through it with its encoded
//! body, and a shared [`CacheBook`] decides what the request reads from a
//! cache and when a new cache is worth creating:
//!
//! ```no_run
//! use rig_core::providers::gemini::{AutoCache, CacheBook, Gemini};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::from_env()?;
//! let book = CacheBook::new(AutoCache::default());
//! let model = gemini.completion("gemini-3.8-flash").caching(&book);
//! // ... run an agent or a chat on `model` ...
//! book.close(&gemini.cached_contents()).await;
//! println!("{:?}", book.report());
//! # Ok(())
//! # }
//! ```
//!
//! # What it does
//!
//! Gemini's implicit caching reuses an earlier request's prefix at no cost,
//! but on gemini-3.8-flash a chat-sized request that extends the previous one
//! almost never hits (measured: 1 hit in 138). An explicit cache
//! (`cachedContents`) can hold the whole conversation so far: system
//! instruction, tools, tool config, every model turn with its signatures, and
//! function calls and responses. A request that reads it sends only what
//! came after. Caches cannot be extended or chained, so a growing
//! conversation needs a new cache from time to time ("rolling"), and each new
//! cache is billed again at the input price.
//!
//! The `Auto` policy rolls when rolling has already paid for itself: it
//! counts the input premium the conversation paid for tokens a cache could
//! have held, and creates a new cache once that premium covers the new
//! cache's creation and storage (ski rental). So:
//!
//! - a cache read once never pays, and is never made;
//! - rolling beats one fixed prefix cache from about 20 calls, and the saving
//!   grows with the run;
//! - where implicit caching already serves a conversation (large contexts,
//!   repeated documents), the book stays inline instead of paying to cache
//!   what is already cheap. It goes inline above `implicit_ceiling` tokens
//!   until implicit caching is seen failing.
//!
//! Cost savings top out below 90%, because a cached read costs 10% of the
//! input price. How close a run gets depends on its shape: a large stable
//! prefix with little new content per call approaches 99% cached (a batch of
//! short questions over a 5.5k-token prefix measured 99.6%); a chat on a small
//! preamble cannot, because each call's new turn is never cached, and on
//! Gemini 3 each earlier thought a request replays through its signature is
//! billed again as input that no explicit cache holds (see
//! [`ThoughtReplay`](super::completion::ThoughtReplay)).
//!
//! # Lifecycle
//!
//! - The book tracks each cache's expiry and never sends one it knows has
//!   expired. A new cache's TTL follows the gaps between calls (at least
//!   twice the longest gap seen, never less than [`AutoCache::ttl`], never more
//!   than 81 minutes, past which keeping a cache costs more than it saves at
//!   the default prices), and a cache read close to expiry is extended.
//! - A replaced cache is deleted when the roll replaces it, and a
//!   conversation cache nobody has read for a while is deleted too.
//! - A cache that answers 403 (deleted elsewhere, or expired early) is
//!   forgotten and the request is sent again inline, once.
//! - [`CacheBook::close`] deletes everything still live.
//!
//! # Resume
//!
//! A checkpoint stores [`CacheBook::leases`] beside the run. A new book in a
//! new process calls [`CacheBook::restore`] with them, then
//! [`CacheBook::prove`], which keeps a lease only if Google still has it
//! under its name. The resumed run then reads the surviving cache instead of
//! creating a new one.
//!
//! # Scope
//!
//! This covers `generateContent` and `streamGenerateContent` on the Gemini
//! Developer API. The Interactions API has no explicit caching. Vertex AI
//! and `rig-gemini-grpc` are not covered, nor are hosted-tool parts, which
//! this crate's Gemini decoder does not keep in history.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};

/// Unix seconds.
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Seconds since the Unix epoch, from the system clock where the target has
/// one. On `wasm32-unknown-unknown` there is none: pass a clock with
/// [`CacheBook::with_clock`] there, or every cache looks forever young and
/// expiry is found by Gemini's 403 instead.
fn system_now() -> u64 {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs())
    }
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        0
    }
}

/// The longest TTL the book gives a cache: past about 81 minutes between
/// reads, storage costs more than a cached read saves at the default prices.
const MAX_TTL_SECS: u64 = 81 * 60;

/// A cache is used only with more than this many seconds left.
const EXPIRY_MARGIN_SECS: u64 = 5;

/// A conversation cache unread for this long, or three times its line's
/// longest gap if longer, belongs to a conversation that moved on.
const MIN_IDLE_SECS: u64 = 60;

/// The `Auto` policy's parameters. The price ratios default to
/// gemini-3.8-flash standard prices relative to its input price (cached read
/// $0.075 and storage $0.50 per 1M tokens per hour, against $0.75 input);
/// set them for other models or tiers. Nothing here reads a model name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoCache {
    /// The shortest TTL a new cache gets.
    pub ttl: Duration,
    /// Smallest request implicit caching serves (4,096 tokens on Gemini 3).
    pub implicit_floor: u64,
    /// Above this many cacheable tokens the book sends requests inline and
    /// lets implicit caching serve them, until implicit caching is seen
    /// failing.
    pub implicit_ceiling: u64,
    /// Gemini's minimum cache size.
    pub min_tokens: u64,
    /// The fewest tokens a new cache must add over the one it replaces.
    pub min_gain: u64,
    /// Cached-read price over input price.
    pub cached_ratio: f64,
    /// Storage price per hour over input price.
    pub storage_ratio_per_hour: f64,
}

impl Default for AutoCache {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(60 * 60),
            implicit_floor: 4_096,
            implicit_ceiling: 16_000,
            min_tokens: 1_024,
            min_gain: 256,
            cached_ratio: 0.1,
            storage_ratio_per_hour: 0.5 / 0.75,
        }
    }
}

/// One cache the book holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    /// `cachedContents/<id>`.
    pub name: String,
    /// Hex SHA-256 of the prefix the cache holds. Its first 40 characters
    /// end the cache's display name, which is how [`CacheBook::prove`]
    /// recognizes it.
    pub digest: String,
    /// How many contents the cache holds.
    pub covers: usize,
    /// Its size, as Gemini counted it when it was created.
    pub tokens: u64,
    /// Unix seconds.
    pub expires_at: u64,
    /// The model the cache belongs to.
    pub model: String,
}

/// Something that happened to a cache.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CacheEvent {
    /// A cache was created.
    Created {
        /// `cachedContents/<id>`.
        name: String,
        /// Its size, as Gemini counted it.
        tokens: u64,
        /// The book's estimate of its size before creating it.
        estimated: u64,
        /// How many contents it holds.
        covers: usize,
        /// Its TTL.
        ttl_secs: u64,
        /// Unix seconds.
        at: u64,
    },
    /// The book deleted a cache it no longer needs.
    Retired {
        /// `cachedContents/<id>`.
        name: String,
        /// Unix seconds.
        at: u64,
    },
    /// A cache answered 403 when a request named it, and was forgotten.
    Lost {
        /// `cachedContents/<id>`.
        name: String,
        /// Unix seconds.
        at: u64,
    },
    /// A cache's expiry was pushed back.
    Extended {
        /// `cachedContents/<id>`.
        name: String,
        /// The new expiry, Unix seconds.
        expires_at: u64,
        /// Unix seconds.
        at: u64,
    },
    /// Gemini refused to create a cache; the request went out without it.
    CreateFailed {
        /// The HTTP status, when there was one.
        status: Option<u16>,
        /// Gemini's message.
        message: String,
        /// Unix seconds.
        at: u64,
    },
}

/// One cache the book created.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreatedCache {
    /// `cachedContents/<id>`.
    pub name: String,
    /// Its size.
    pub tokens: u64,
    /// Unix seconds.
    pub at: u64,
}

/// What the book's caches did and cost, for pricing a run. Creation is
/// billed like input (`tokens` × input price); storage is `token_hours` ×
/// the storage price.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheReport {
    /// Every cache created, in order.
    pub created: Vec<CreatedCache>,
    /// How many requests read each cache, by name.
    pub reads: BTreeMap<String, u64>,
    /// Caches the book deleted because a roll replaced them, their
    /// conversation moved on, or [`CacheBook::close`] ran.
    pub retired: Vec<String>,
    /// Caches that answered 403 and were forgotten.
    pub lost: Vec<String>,
    /// Σ tokens × hours each cache lived, until it was deleted, lost or
    /// expired, or until now for a live one.
    pub token_hours: f64,
    /// The caches still live.
    pub live: Vec<Lease>,
}

#[derive(Clone, Debug, Default)]
struct Line {
    /// Premium tokens: cacheable input paid at full price, weighted by
    /// `1 - cached_ratio`, since the line last rolled.
    premium: f64,
    /// EWMA of implicit coverage on inline calls big enough for it.
    implicit: Option<f64>,
    rolled_at: u64,
    last_call: Option<u64>,
    longest_gap: u64,
    /// Coverable size at the last refused creation: no retry until the line
    /// grows past it.
    failed_at: Option<u64>,
}

#[derive(Clone, Debug)]
struct Life {
    tokens: u64,
    created: u64,
    expires_at: u64,
    ended: Option<u64>,
    last_read: u64,
    line_gap: u64,
}

#[derive(Default)]
pub(crate) struct Book {
    pub(crate) leases: HashMap<String, Lease>,
    lives: BTreeMap<String, Life>,
    /// For each prefix, the conversations seen on it, by their first content.
    lineages: HashMap<String, HashSet<String>>,
    lines: HashMap<String, Line>,
    events: Vec<CacheEvent>,
    report: CacheReport,
    /// Gemini's tokens over this book's estimate, learned from each cache it
    /// creates.
    calibration: Option<f64>,
}

/// The shared, content-addressed record of the caches a set of models
/// reads. Clone it into every model that should share caches: sub-agents on
/// one prefix, a resumed run and its successor.
#[derive(Clone)]
pub struct CacheBook {
    inner: Arc<Mutex<Book>>,
    /// Serializes cache creation, so a prefix is created once.
    pub(crate) create: Arc<futures::lock::Mutex<()>>,
    policy: AutoCache,
    clock: Clock,
    pub(crate) display_prefix: Arc<str>,
}

impl std::fmt::Debug for CacheBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheBook")
            .field("policy", &self.policy)
            .field("display_prefix", &self.display_prefix)
            .finish_non_exhaustive()
    }
}

impl CacheBook {
    /// A book with `policy`, on the system clock.
    pub fn new(policy: AutoCache) -> Self {
        Self {
            inner: Arc::default(),
            create: Arc::default(),
            policy,
            clock: Arc::new(system_now),
            display_prefix: Arc::from("rig-cache-"),
        }
    }

    /// The same book reading time from `clock` (Unix seconds).
    pub fn with_clock(mut self, clock: impl Fn() -> u64 + Send + Sync + 'static) -> Self {
        self.clock = Arc::new(clock);
        self
    }

    /// The same book naming its caches `<prefix><digest>`, so they can be
    /// listed and swept. Defaults to `rig-cache-`.
    pub fn with_display_prefix(mut self, prefix: &str) -> Self {
        self.display_prefix = Arc::from(prefix);
        self
    }

    /// The book's policy.
    pub fn policy(&self) -> AutoCache {
        self.policy
    }

    pub(crate) fn book(&self) -> std::sync::MutexGuard<'_, Book> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn now(&self) -> u64 {
        (self.clock)()
    }

    /// Everything that happened to the book's caches, in order.
    pub fn events(&self) -> Vec<CacheEvent> {
        self.book().events.clone()
    }

    /// The caches the book holds, for a checkpoint.
    pub fn leases(&self) -> Vec<Lease> {
        let mut leases: Vec<Lease> = self.book().leases.values().cloned().collect();
        leases.sort_by(|a, b| a.name.cmp(&b.name));
        leases
    }

    /// Put checkpointed leases back. Call [`Self::prove`] next.
    pub fn restore(&self, leases: Vec<Lease>) {
        let now = self.now();
        let mut book = self.book();
        for lease in leases {
            book.lives.entry(lease.name.clone()).or_insert(Life {
                tokens: lease.tokens,
                created: now,
                expires_at: lease.expires_at,
                ended: None,
                last_read: now,
                line_gap: 0,
            });
            book.leases.insert(lease.digest.clone(), lease);
        }
    }

    /// What the caches did and cost so far.
    pub fn report(&self) -> CacheReport {
        let now = self.now();
        let book = self.book();
        let mut report = book.report.clone();
        report.token_hours = book
            .lives
            .values()
            .map(|life| {
                let end = life.ended.unwrap_or(now).min(life.expires_at);
                life.tokens as f64 * end.saturating_sub(life.created) as f64 / 3600.0
            })
            .sum();
        report.live = book.leases.values().cloned().collect();
        report.live.sort_by(|a, b| a.name.cmp(&b.name));
        report
    }

    pub(crate) fn retired(&self, lease: &Lease) {
        let now = self.now();
        let mut book = self.book();
        book.leases.remove(&lease.digest);
        end_life(&mut book, &lease.name, now);
        book.events.push(CacheEvent::Retired {
            name: lease.name.clone(),
            at: now,
        });
        book.report.retired.push(lease.name.clone());
        tracing::info!(target: "gemini.cache.retired", name = %lease.name);
    }

    pub(crate) fn lost(&self, lease: &Lease) {
        let now = self.now();
        let mut book = self.book();
        book.leases.remove(&lease.digest);
        end_life(&mut book, &lease.name, now);
        book.events.push(CacheEvent::Lost {
            name: lease.name.clone(),
            at: now,
        });
        book.report.lost.push(lease.name.clone());
        tracing::info!(target: "gemini.cache.lost", name = %lease.name);
    }
}

pub(crate) fn end_life(book: &mut Book, name: &str, now: u64) {
    if let Some(life) = book.lives.get_mut(name)
        && life.ended.is_none()
    {
        life.ended = Some(now);
    }
}

pub(crate) fn short_digest(digest: &str) -> &str {
    digest.get(..40).unwrap_or(digest)
}

// ---------------------------------------------------------------------------
// Request bodies: parsing, digests, stripping and cache bodies.

const PREFIX_KEYS: [&str; 3] = ["systemInstruction", "tools", "toolConfig"];

/// A `generateContent` body, its contents kept as their exact bytes.
pub(crate) struct Parsed {
    /// Every other top-level field, in order, as exact bytes.
    pub(crate) rest: Vec<(String, Box<RawValue>)>,
    /// `systemInstruction`, `tools` and `toolConfig`, when present and not null.
    pub(crate) prefix: [Option<Box<RawValue>>; 3],
    pub(crate) contents: Vec<Box<RawValue>>,
    pub(crate) has_cached_content: bool,
}

pub(crate) fn parse(bytes: &[u8]) -> Option<Parsed> {
    let fields: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(bytes).ok()?;
    // Parse again as raw values so every part keeps its bytes; the first
    // pass only fixes the key order, which `RawValue` maps do not keep.
    let raw: HashMap<String, Box<RawValue>> = serde_json::from_slice(bytes).ok()?;
    let mut parsed = Parsed {
        rest: Vec::new(),
        prefix: [None, None, None],
        contents: Vec::new(),
        has_cached_content: false,
    };
    for key in fields.keys() {
        let value = raw.get(key)?.clone();
        if key == "contents" {
            parsed.contents = serde_json::from_str(value.get()).ok()?;
        } else if let Some(slot) = PREFIX_KEYS.iter().position(|prefix| prefix == key) {
            if value.get() != "null"
                && let Some(field) = parsed.prefix.get_mut(slot)
            {
                *field = Some(value);
            }
        } else {
            if key == "cachedContent" && value.get() != "null" {
                parsed.has_cached_content = true;
            }
            parsed.rest.push((key.clone(), value));
        }
    }
    Some(parsed)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// `d[k]`: the digest of the model, the prefix and `contents[..k]`.
pub(crate) fn digests(model: &str, parsed: &Parsed) -> Vec<String> {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    for field in &parsed.prefix {
        hasher.update([0u8]);
        if let Some(raw) = field {
            hasher.update(raw.get().as_bytes());
        }
    }
    let mut state = hasher.finalize().to_vec();
    let mut out = vec![hex(&state)];
    for content in &parsed.contents {
        let mut hasher = Sha256::new();
        hasher.update(&state);
        hasher.update(content.get().as_bytes());
        state = hasher.finalize().to_vec();
        out.push(hex(&state));
    }
    out
}

/// A rough token count of a JSON fragment: its bytes over four, without
/// thought signatures, which Gemini expands into restored thoughts that no
/// explicit cache holds.
fn estimate(json: &str) -> u64 {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(json) else {
        return (json.len() / 4) as u64;
    };
    fn scrub(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.remove("thoughtSignature");
                map.values_mut().for_each(scrub);
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(scrub),
            _ => {}
        }
    }
    scrub(&mut value);
    (value.to_string().len() / 4) as u64
}

/// Whether a content is the user's text: a new user turn, not a tool result.
pub(crate) fn is_user_text(raw: &RawValue) -> bool {
    #[derive(Deserialize)]
    struct Content {
        role: Option<String>,
        #[serde(default)]
        parts: Vec<serde_json::Map<String, serde_json::Value>>,
    }
    serde_json::from_str::<Content>(raw.get()).is_ok_and(|content| {
        content.role.as_deref() == Some("user")
            && content.parts.iter().any(|part| part.contains_key("text"))
            && !content
                .parts
                .iter()
                .any(|part| part.contains_key("functionResponse"))
    })
}

/// The body that reads `lease` instead of what it holds.
pub(crate) fn stripped(parsed: &Parsed, lease: &Lease) -> Option<Vec<u8>> {
    let mut out: Vec<(String, &RawValue)> = Vec::new();
    let name = serde_json::value::to_raw_value(&lease.name).ok()?;
    let contents = serde_json::value::to_raw_value(&parsed.contents.get(lease.covers..)?).ok()?;
    out.push(("cachedContent".to_owned(), &name));
    out.push(("contents".to_owned(), &contents));
    for (key, value) in &parsed.rest {
        if key != "cachedContent" {
            out.push((key.clone(), value));
        }
    }
    let mut bytes = b"{".to_vec();
    for (index, (key, value)) in out.iter().enumerate() {
        if index > 0 {
            bytes.push(b',');
        }
        bytes.extend(serde_json::to_vec(key).ok()?);
        bytes.push(b':');
        bytes.extend(value.get().as_bytes());
    }
    bytes.push(b'}');
    Some(bytes)
}

/// The `cachedContents` create body for the first `covers` contents, from
/// the request's own bytes.
pub(crate) fn cache_body(
    model: &str,
    parsed: &Parsed,
    covers: usize,
    display_name: &str,
    ttl_secs: u64,
) -> Option<Vec<u8>> {
    let mut bytes = b"{".to_vec();
    let mut field = |key: &str, value: &str| {
        if bytes.len() > 1 {
            bytes.push(b',');
        }
        bytes.extend(format!("\"{key}\":").as_bytes());
        bytes.extend(value.as_bytes());
    };
    field(
        "model",
        &serde_json::to_string(&format!("models/{model}")).ok()?,
    );
    field("displayName", &serde_json::to_string(display_name).ok()?);
    field("ttl", &format!("\"{ttl_secs}s\""));
    for (key, value) in PREFIX_KEYS.iter().zip(&parsed.prefix) {
        if let Some(value) = value {
            field(key, value.get());
        }
    }
    if covers > 0 {
        let contents = serde_json::value::to_raw_value(&parsed.contents.get(..covers)?).ok()?;
        field("contents", contents.get());
    }
    bytes.push(b'}');
    Some(bytes)
}

// ---------------------------------------------------------------------------
// The plan.

/// What one request reads, creates, retires and extends.
pub(crate) struct Plan {
    pub(crate) read: Option<Lease>,
    pub(crate) create: Option<Create>,
    pub(crate) retire: Vec<Lease>,
    pub(crate) extend: Option<(Lease, u64)>,
    pub(crate) line: String,
    pub(crate) coverable: u64,
}

pub(crate) struct Create {
    pub(crate) covers: usize,
    pub(crate) digest: String,
    pub(crate) ttl_secs: u64,
    /// Uncalibrated bytes/4 estimate of what the cache holds.
    estimate: u64,
    /// The calibrated estimate the plan used.
    expected: u64,
    pub(crate) replaces: Option<Lease>,
}

impl CacheBook {
    pub(crate) fn plan(
        &self,
        model: &str,
        d: &[String],
        parsed: &Parsed,
        roll_allowed: bool,
    ) -> Plan {
        let p = self.policy;
        let now = self.now();
        let n = parsed.contents.len();
        // `d` holds one digest per content boundary, `0..=n`.
        let at = |k: usize| d.get(k).cloned().unwrap_or_default();
        let mut book = self.book();
        let calibration = book.calibration.unwrap_or(1.0);
        let prefix_estimate: u64 = parsed
            .prefix
            .iter()
            .flatten()
            .map(|raw| estimate(raw.get()))
            .sum();
        let content_estimates: Vec<u64> = parsed
            .contents
            .iter()
            .map(|raw| estimate(raw.get()))
            .collect();

        // Who shares this prefix: a second conversation on it (sub-agents,
        // concurrent users, or the conversation a compaction started).
        if n >= 1 {
            book.lineages.entry(at(0)).or_default().insert(at(1));
        }
        let shared = book
            .lineages
            .get(&at(0))
            .is_some_and(|seen| seen.len() >= 2);

        // Retire conversation caches nobody reads any more.
        let idle: Vec<Lease> = book
            .leases
            .values()
            .filter(|lease| lease.covers > 0 && lease.model == model)
            .filter(|lease| {
                book.lives.get(&lease.name).is_some_and(|life| {
                    now.saturating_sub(life.last_read)
                        > MIN_IDLE_SECS.max(life.line_gap.saturating_mul(3))
                })
            })
            .cloned()
            .collect();

        // The longest live cache this request starts with; never the newest content.
        let read = (0..n.max(1))
            .rev()
            .find_map(|k| {
                book.leases
                    .get(&at(k))
                    .filter(|lease| lease.model == model)
                    .filter(|lease| lease.expires_at > now + EXPIRY_MARGIN_SECS)
                    .cloned()
            })
            .filter(|lease| !idle.iter().any(|gone| gone.name == lease.name));
        let covered = read.as_ref().map_or(0, |lease| lease.covers);
        let tail: u64 = content_estimates
            .get(covered..n.saturating_sub(1).max(covered))
            .unwrap_or_default()
            .iter()
            .sum();
        let coverable = match &read {
            Some(lease) => lease.tokens + (tail as f64 * calibration) as u64,
            None => ((prefix_estimate + tail) as f64 * calibration) as u64,
        };

        // The conversation's line: its cache, or its first content when inline.
        let line_key = read
            .as_ref()
            .filter(|lease| lease.covers > 0)
            .map_or_else(|| at(1.min(n)), |lease| lease.digest.clone());
        let prefix_leased = book.leases.contains_key(&at(0));
        let line = book.lines.entry(line_key.clone()).or_insert_with(|| Line {
            rolled_at: now,
            ..Line::default()
        });
        if let Some(last) = line.last_call {
            line.longest_gap = line.longest_gap.max(now.saturating_sub(last));
        }
        line.last_call = Some(now);
        let longest_gap = line.longest_gap;
        let ttl_secs = p
            .ttl
            .as_secs()
            .max(longest_gap.saturating_mul(2).min(MAX_TTL_SECS))
            .max(1);

        let mut plan = Plan {
            read: read.clone(),
            create: None,
            retire: idle,
            extend: None,
            line: line_key,
            coverable,
        };

        // Past the ceiling, implicit caching serves: go inline until it fails.
        let implicit_serving = line.implicit.is_none_or(|ratio| ratio >= 0.5);
        if coverable >= p.implicit_ceiling && implicit_serving {
            plan.read = None;
            return plan;
        }

        let failed_below = line.failed_at.is_some_and(|at| coverable < at + p.min_gain);
        if read.is_none()
            && shared
            && !prefix_leased
            && (prefix_estimate as f64 * calibration) as u64 >= p.min_tokens
            && !failed_below
        {
            plan.create = Some(Create {
                covers: 0,
                digest: at(0),
                ttl_secs,
                estimate: prefix_estimate,
                expected: (prefix_estimate as f64 * calibration) as u64,
                replaces: None,
            });
        } else if n >= 2 && roll_allowed && !failed_below {
            let cached = read.as_ref().map_or(0, |lease| lease.tokens);
            let gain = coverable.saturating_sub(cached);
            let held_hours = now.saturating_sub(line.rolled_at).max(60) as f64 / 3600.0;
            let cost = coverable as f64 * (1.0 + p.storage_ratio_per_hour * held_hours);
            if gain >= p.min_gain && coverable >= p.min_tokens && line.premium >= cost {
                plan.create = Some(Create {
                    covers: n - 1,
                    digest: at(n - 1),
                    ttl_secs,
                    estimate: prefix_estimate
                        + content_estimates
                            .get(..n - 1)
                            .unwrap_or_default()
                            .iter()
                            .sum::<u64>(),
                    expected: coverable,
                    replaces: read.clone().filter(|lease| lease.covers > 0),
                });
            }
        }

        // Extend a cache that would expire before its line's next call.
        if plan.create.is_none()
            && let Some(lease) = &read
            && longest_gap > 0
            && lease.expires_at.saturating_sub(now) < longest_gap.saturating_mul(3) / 2
        {
            plan.extend = Some((lease.clone(), ttl_secs));
        }
        plan
    }

    /// Account for one reply's usage on `line`.
    pub(crate) fn observe(&self, line: &str, read: Option<&str>, coverable: u64, cached: u64) {
        let p = self.policy;
        let mut book = self.book();
        if let Some(name) = read {
            *book.report.reads.entry(name.to_owned()).or_default() += 1;
        }
        let entry = book.lines.entry(line.to_owned()).or_default();
        let mut missed = coverable.saturating_sub(cached) as f64;
        if read.is_none() && coverable >= p.implicit_floor {
            let ratio = cached as f64 / coverable.max(1) as f64;
            let implicit = entry
                .implicit
                .map_or(ratio, |ewma| 0.5 * ewma + 0.5 * ratio);
            entry.implicit = Some(implicit);
            if implicit >= 0.5 {
                missed = 0.0;
            }
        }
        entry.premium += missed * (1.0 - p.cached_ratio);
    }

    pub(crate) fn created(
        &self,
        from_line: &str,
        create: &Create,
        name: String,
        tokens: u64,
        model: &str,
    ) -> Lease {
        let now = self.now();
        let lease = Lease {
            name: name.clone(),
            digest: create.digest.clone(),
            covers: create.covers,
            tokens,
            expires_at: now + create.ttl_secs,
            model: model.to_owned(),
        };
        let mut book = self.book();
        if create.estimate > 0 && tokens > 0 {
            let ratio = tokens as f64 / create.estimate as f64;
            book.calibration = Some(book.calibration.map_or(ratio, |c| 0.5 * c + 0.5 * ratio));
        }
        let previous = book.lines.get(from_line).cloned().unwrap_or_default();
        if create.covers > 0 {
            book.lines.insert(
                lease.digest.clone(),
                Line {
                    premium: 0.0,
                    implicit: previous.implicit,
                    rolled_at: now,
                    last_call: previous.last_call,
                    longest_gap: previous.longest_gap,
                    failed_at: None,
                },
            );
        }
        book.lives.insert(
            name.clone(),
            Life {
                tokens,
                created: now,
                expires_at: lease.expires_at,
                ended: None,
                last_read: now,
                line_gap: previous.longest_gap,
            },
        );
        book.events.push(CacheEvent::Created {
            name: name.clone(),
            tokens,
            estimated: create.expected,
            covers: create.covers,
            ttl_secs: create.ttl_secs,
            at: now,
        });
        book.report.created.push(CreatedCache {
            name,
            tokens,
            at: now,
        });
        book.leases.insert(lease.digest.clone(), lease.clone());
        tracing::info!(target: "gemini.cache.created", name = %lease.name, tokens, covers = create.covers);
        lease
    }

    pub(crate) fn create_failed(
        &self,
        line: &str,
        coverable: u64,
        status: Option<u16>,
        message: String,
    ) {
        let now = self.now();
        let mut book = self.book();
        if let Some(entry) = book.lines.get_mut(line) {
            entry.failed_at = Some(coverable);
        }
        book.events.push(CacheEvent::CreateFailed {
            status,
            message,
            at: now,
        });
    }

    pub(crate) fn extended(&self, lease: &Lease, ttl_secs: u64) {
        let now = self.now();
        let expires_at = now + ttl_secs;
        let mut book = self.book();
        if let Some(held) = book.leases.get_mut(&lease.digest) {
            held.expires_at = expires_at;
        }
        if let Some(life) = book.lives.get_mut(&lease.name) {
            life.expires_at = expires_at;
        }
        book.events.push(CacheEvent::Extended {
            name: lease.name.clone(),
            expires_at,
            at: now,
        });
        tracing::info!(target: "gemini.cache.extended", name = %lease.name, expires_at);
    }

    pub(crate) fn touched(&self, lease: &Lease) {
        let now = self.now();
        let mut book = self.book();
        let gap = book
            .lines
            .get(&lease.digest)
            .map_or(0, |line| line.longest_gap);
        if let Some(life) = book.lives.get_mut(&lease.name) {
            life.last_read = now;
            life.line_gap = life.line_gap.max(gap);
        }
    }
}
