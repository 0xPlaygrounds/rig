//! OC-20: the provider quota status item.
//!
//! The plugin wraps the HTTP client every model is reached through
//! ([`Connector::wrap_http`]) and reads the rate-limit headers of each reply
//! to a model call (OpenAI's and Groq's `x-ratelimit-*-requests|tokens`,
//! Anthropic's `anthropic-ratelimit-*`, OpenRouter's `x-ratelimit-*` and the
//! IETF draft's `ratelimit-*`), keyed by the call's effect id from its
//! [`Serving`]. When the call's [`Outcome`] arrives, its reading becomes the
//! latest [`Quota`] of the provider it was [`SentTo`] and the key it was
//! sent with; a 429 marks the quota limited until the wait the provider
//! asked for (rig-core's `ErrorReport::retry_after`). The plugin never sends
//! a request of its own.
//!
//! Each agent's status item shows its model's provider's latest quota,
//! `req 42/50 · tok 18k/40k · resets 34s`, in the tone of the tighter
//! allowance (yellow under 25 % left, red under 5 % or after a 429), or
//! `limited · retry in 12s` counting down; it is hidden for a provider
//! that sent no such headers. `/quota` lists every provider and key seen
//! this session with their last values and when they came.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::Connector;
use rig_harness::rig_core::effect::EffectId;
use rig_harness::rig_core::http_client::{
    self, Bytes, DynHttpClient, HeaderMap, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, StreamingResponse,
};
use rig_harness::rig_core::serve::Serving;
use rig_harness::rig_core::wasm_compat::WasmCompatSend;

// The test drives the status item, which only the terminal view has.
#[cfg(all(test, feature = "tui"))]
mod tests;

/// The `(limit, remaining, reset)` headers of a request allowance, in the
/// formats of the main providers; the first pair present wins.
const REQUESTS: [[&str; 3]; 4] = [
    [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    ],
    [
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
    ],
    [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    ],
    ["ratelimit-limit", "ratelimit-remaining", "ratelimit-reset"],
];

/// The `(limit, remaining, reset)` headers of a token allowance.
const TOKENS: [[&str; 3]; 3] = [
    [
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ],
    [
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
    ],
    [
        "anthropic-ratelimit-input-tokens-limit",
        "anthropic-ratelimit-input-tokens-remaining",
        "anthropic-ratelimit-input-tokens-reset",
    ],
];

/// Shows the focused agent's provider quota in the status line and lists
/// every provider's with `/quota`.
#[derive(Default)]
pub struct RateLimitStatusPlugin;

impl Plugin for RateLimitStatusPlugin {
    fn build(&self, app: &mut App) {
        let seen = Seen::default();
        let models = &mut app.world_mut().get_resource_or_init::<Models>().0;
        observe_replies(models, seen.clone());
        app.insert_resource(seen)
            .init_resource::<Quotas>()
            .add_observer(read_reply)
            .add_command(
                "quota",
                "Each provider's last rate limits this session",
                quota,
            );
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// One allowance: what is left of it, of how much, and when it resets.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct Allowance {
    /// Requests or tokens left.
    pub remaining: u64,
    /// Requests or tokens per window.
    pub limit: u64,
    /// When it resets, on [`Time<Real>`]'s elapsed time.
    pub resets: Option<Duration>,
}

impl Allowance {
    /// The share left, from 0 to 1.
    pub fn left(&self) -> f64 {
        if self.limit == 0 {
            return 0.0;
        }
        self.remaining as f64 / self.limit as f64
    }
}

/// A provider key's latest rate limits.
#[derive(Reflect, Clone, Debug, PartialEq)]
pub struct Quota {
    /// The provider's vendor, such as `openai`.
    pub provider: String,
    /// The key's last four characters, or empty when none was seen.
    pub key: String,
    /// Its request allowance.
    pub requests: Option<Allowance>,
    /// Its token allowance.
    pub tokens: Option<Allowance>,
    /// When the last reply came, on [`Time<Real>`]'s elapsed time.
    pub received: Duration,
    /// Set by a 429, until the provider's next good reply: when it may be
    /// called again, if it said.
    pub limited: Option<Option<Duration>>,
}

impl Quota {
    /// The tighter of the two allowances.
    pub fn tighter(&self) -> Option<Allowance> {
        match (self.requests, self.tokens) {
            (Some(requests), Some(tokens)) if tokens.left() < requests.left() => Some(tokens),
            (Some(requests), _) => Some(requests),
            (None, tokens) => tokens,
        }
    }

    /// `req 42/50 · tok 18k/40k · resets 34s`, or `limited · retry in
    /// 12s`, as of `now`; `None` with nothing to say.
    pub fn says(&self, now: Duration) -> Option<String> {
        if let Some(Some(until)) = self.limited
            && until > now
        {
            return Some(format!("limited · retry in {}", wait(until - now)));
        }
        let mut parts = Vec::new();
        for (name, allowance) in [("req", self.requests), ("tok", self.tokens)] {
            if let Some(allowance) = allowance {
                let (left, of) = (count(allowance.remaining), count(allowance.limit));
                parts.push(format!("{name} {left}/{of}"));
            }
        }
        let tighter = self.tighter();
        let resets = tighter.and_then(|tighter| tighter.resets);
        let resets = resets.or_else(|| {
            [self.requests, self.tokens]
                .iter()
                .flatten()
                .find_map(|a| a.resets)
        });
        if let Some(at) = resets.filter(|at| *at > now) {
            parts.push(format!("resets {}", wait(at - now)));
        }
        if parts.is_empty() {
            return self.limited.map(|_| "limited".to_owned());
        }
        Some(parts.join(" · "))
    }

    /// Whether what it says counts down, as of `now`: a reset or a
    /// retry still ahead.
    pub fn ticking(&self, now: Duration) -> bool {
        let limited = self.limited.flatten();
        let resets = [self.requests, self.tokens]
            .into_iter()
            .flatten()
            .filter_map(|a| a.resets);
        limited.into_iter().chain(resets).any(|at| at > now)
    }
}

/// Every provider key's latest rate limits this session, in the order
/// first seen. Not saved: they are stale in a later run.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Default)]
pub struct Quotas(pub Vec<Quota>);

impl Quotas {
    /// The quota whose reply came last of `provider`'s keys.
    pub fn latest(&self, provider: &str) -> Option<&Quota> {
        let of = self.0.iter().filter(|quota| quota.provider == provider);
        of.max_by_key(|quota| quota.received)
    }

    /// `provider`'s quota for `key`, added when new.
    fn entry(&mut self, provider: &str, key: String, now: Duration) -> Option<&mut Quota> {
        let found = self
            .0
            .iter()
            .position(|q| q.provider == provider && q.key == key);
        let at = found.unwrap_or_else(|| {
            self.0.push(Quota {
                provider: provider.to_owned(),
                key,
                requests: None,
                tokens: None,
                received: now,
                limited: None,
            });
            self.0.len().saturating_sub(1)
        });
        self.0.get_mut(at)
    }
}

/// An allowance as its headers say it, its reset counted from the reply.
#[derive(Clone, Copy, Debug)]
struct Read {
    remaining: u64,
    limit: u64,
    resets_in: Option<Duration>,
}

impl Read {
    /// The allowance, its reset counted from `now`.
    fn at(self, now: Duration) -> Allowance {
        Allowance {
            remaining: self.remaining,
            limit: self.limit,
            resets: self.resets_in.map(|wait| now + wait),
        }
    }
}

/// What the rate-limit headers of one reply said.
#[derive(Clone, Debug)]
struct Reading {
    key: String,
    requests: Option<Read>,
    tokens: Option<Read>,
}

/// The readings of model calls whose outcome has not arrived, by effect
/// id, shared with the HTTP client wrapper.
#[derive(Resource, Clone, Default)]
struct Seen(Arc<Mutex<HashMap<EffectId, Reading>>>);

impl Seen {
    fn lock(&self) -> MutexGuard<'_, HashMap<EffectId, Reading>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Wraps every HTTP client `models` reaches a model through in one that
/// reads its replies' rate limits.
fn observe_replies(models: &mut Connector, seen: Seen) {
    models.wrap_http(move |inner| {
        DynHttpClient::new(Observing {
            inner,
            seen: seen.clone(),
        })
    });
}

/// An HTTP client that keeps the rate limits of each reply to a model
/// call, by the call's effect id; it changes nothing.
#[derive(Clone)]
struct Observing {
    inner: DynHttpClient,
    seen: Seen,
}

impl Observing {
    /// Sends `request` with `send` and reads the reply's headers, or a
    /// failed reply's, when the request serves a model call.
    async fn observe<B, F>(
        self,
        request: Request<Bytes>,
        send: impl FnOnce(DynHttpClient, Request<Bytes>) -> F,
    ) -> http_client::Result<Response<B>>
    where
        F: Future<Output = http_client::Result<Response<B>>>,
    {
        let effect = request.extensions().get::<Serving>().map(|s| s.effect);
        let key = key_of(request.headers());
        let reply = send(self.inner, request).await;
        let headers = match &reply {
            Ok(reply) => Some(reply.headers()),
            Err(error) => error.non_success_headers(),
        };
        if let (Some(effect), Some(headers)) = (effect, headers) {
            let now = SystemTime::now();
            let reading = Reading {
                key,
                requests: allowance(headers, &REQUESTS, now),
                tokens: allowance(headers, &TOKENS, now),
            };
            self.seen.lock().insert(effect, reading);
        }
        reply
    }
}

impl HttpClientExt for Observing {
    fn send<T, U>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        let request = request.map(Into::into);
        self.clone().observe(request, |inner, request| async move {
            inner.send::<Bytes, U>(request).await
        })
    }

    /// No model call sends a multipart body.
    fn send_multipart<U>(
        &self,
        request: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.inner.send_multipart(request)
    }

    fn send_streaming<T>(
        &self,
        request: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let request = request.map(Into::into);
        self.clone().observe(request, |inner, request| async move {
            inner.send_streaming(request).await
        })
    }
}

/// The last four characters of the key a request carries, so `/quota`
/// tells two keys of a provider apart without showing either.
fn key_of(headers: &HeaderMap) -> String {
    let names = ["authorization", "x-api-key", "x-goog-api-key", "api-key"];
    let value = names
        .iter()
        .find_map(|name| headers.get(*name)?.to_str().ok());
    let key = value.map(|value| value.trim_start_matches("Bearer ").trim());
    let chars: Vec<char> = key.unwrap_or_default().chars().collect();
    let tail = chars.iter().rev().take(4).rev().collect::<String>();
    if chars.len() > 8 {
        format!("…{tail}")
    } else {
        String::new()
    }
}

/// The first allowance of `names` the headers give both a limit and a
/// remainder for.
fn allowance(headers: &HeaderMap, names: &[[&str; 3]], now: SystemTime) -> Option<Read> {
    let text = |name: &str| headers.get(name)?.to_str().ok().map(str::trim);
    names.iter().find_map(|[limit, remaining, reset]| {
        Some(Read {
            limit: leading_number(text(limit)?)?,
            remaining: leading_number(text(remaining)?)?,
            resets_in: text(reset).and_then(|reset| reset_in(reset, now)),
        })
    })
}

/// The whole number a header value starts with: `100` of `100;w=60`.
fn leading_number(text: &str) -> Option<u64> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    text.get(..end)?.parse().ok()
}

/// How long until a reset a header names, as of `now`: seconds (`34`),
/// a Unix time in seconds or milliseconds (OpenRouter), a Go duration
/// (`6m0s`, `20ms`: OpenAI) or a date (`2026-01-01T00:00:00Z`: Anthropic).
fn reset_in(text: &str, now: SystemTime) -> Option<Duration> {
    if let Ok(number) = text.parse::<f64>() {
        let epoch = now.duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
        let seconds = if number > 1e12 {
            number / 1000.0 - epoch
        } else if number > 1e9 {
            number - epoch
        } else {
            number
        };
        return Duration::try_from_secs_f64(seconds.max(0.0)).ok();
    }
    if let Some(wait) = go_duration(text) {
        return Some(wait);
    }
    let at = chrono::DateTime::parse_from_rfc3339(text)
        .or_else(|_| chrono::DateTime::parse_from_rfc2822(text))
        .ok()?;
    Some(SystemTime::from(at).duration_since(now).unwrap_or_default())
}

/// A Go duration such as `1h2m3.5s` or `20ms`.
fn go_duration(text: &str) -> Option<Duration> {
    let mut rest = text;
    let mut seconds = 0.0;
    while !rest.is_empty() {
        let number = rest.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
        let (value, after) = rest.split_at_checked(number)?;
        let unit = after
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(after.len());
        let (unit, next) = after.split_at_checked(unit)?;
        let scale = match unit {
            "h" => 3600.0,
            "m" => 60.0,
            "s" => 1.0,
            "ms" => 1e-3,
            "us" | "µs" => 1e-6,
            "ns" => 1e-9,
            _ => return None,
        };
        seconds += value.parse::<f64>().ok()? * scale;
        rest = next;
    }
    (!text.is_empty()).then(|| Duration::try_from_secs_f64(seconds).ok())?
}

/// `18k`, `1.5k`, `2M`, `42`.
fn count(n: u64) -> String {
    let scaled = |unit: f64, suffix: &str| {
        let value = n as f64 / unit;
        if value >= 100.0 || value.fract() < 0.05 {
            format!("{value:.0}{suffix}")
        } else {
            format!("{value:.1}{suffix}")
        }
    };
    match n {
        1_000_000.. => scaled(1e6, "M"),
        1_000.. => scaled(1e3, "k"),
        _ => n.to_string(),
    }
}

/// `34s`, `6m`, `1m30s`, `2h5m`, rounded up to the second.
fn wait(left: Duration) -> String {
    let seconds = left.as_secs() + u64::from(left.subsec_nanos() > 0);
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    match (hours, minutes, seconds) {
        (0, 0, s) => format!("{s}s"),
        (0, m, 0) => format!("{m}m"),
        (0, m, s) => format!("{m}m{s}s"),
        (h, 0, _) => format!("{h}h"),
        (h, m, _) => format!("{h}h{m}m"),
    }
}

/// When a model call's outcome arrives: what its reply's headers said
/// becomes its provider key's latest quota, and a 429 limits it for the
/// wait the provider asked for. A good reply ends a provider's limit.
fn read_reply(
    answered: On<Add<Outcome>>,
    calls: Query<(&Effect, &Outcome, &SentTo)>,
    seen: Res<Seen>,
    time: Res<Time<Real>>,
    mut quotas: ResMut<Quotas>,
) {
    let Ok((effect, Outcome(outcome), SentTo(connection))) = calls.get(answered.entity) else {
        return;
    };
    let reading = seen.lock().remove(&effect.id());
    let provider = connection.spec.provider.vendor();
    let now = time.elapsed();
    let limited = outcome
        .as_ref()
        .err()
        .filter(|report| report.http_status == Some(429))
        .map(|report| report.retry_after(SystemTime::now()).map(|wait| now + wait));
    let key = reading.as_ref().map(|reading| reading.key.clone());
    let reading = reading.filter(|reading| reading.requests.is_some() || reading.tokens.is_some());
    if reading.is_none() && limited.is_none() {
        if outcome.is_ok() {
            for quota in quotas.0.iter_mut().filter(|q| q.provider == provider) {
                quota.limited = None;
            }
        }
        return;
    }
    let Some(quota) = quotas.entry(provider, key.unwrap_or_default(), now) else {
        return;
    };
    quota.received = now;
    if let Some(reading) = reading {
        quota.requests = reading.requests.map(|read| read.at(now)).or(quota.requests);
        quota.tokens = reading.tokens.map(|read| read.at(now)).or(quota.tokens);
    }
    quota.limited = limited;
}

/// `/quota`: every provider key seen this session, its last values and
/// how long ago they came.
fn quota(
    In(args): In<CommandArgs>,
    quotas: Res<Quotas>,
    time: Res<Time<Real>>,
    mut notices: MessageWriter<Notice>,
) {
    if quotas.0.is_empty() {
        let text = "No provider has sent rate limits this session.";
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let now = time.elapsed();
    let lines = quotas.0.iter().map(|quota| {
        let name = [quota.provider.as_str(), quota.key.as_str()].join(" ");
        let says = quota.says(now).unwrap_or_default();
        let ago = wait(now.saturating_sub(quota.received));
        format!("{}: {says} (received {ago} ago)", name.trim_end())
    });
    notices.write(Notice::info(
        args.agent,
        lines.collect::<Vec<_>>().join("\n"),
    ));
}

/// The status line item, in the terminal view.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

    use super::{Quota, Quotas};

    /// After the agent's status; gone early when the line is too narrow.
    pub(crate) const QUOTA: StatusItem = StatusItem::new("rate-limit-status", Side::Left, 75, 8);

    /// Keeps the loop running while an item counts down.
    #[derive(Component)]
    struct Countdown;

    pub(super) fn add(app: &mut App) {
        app.world_mut()
            .spawn((Name::new("quota countdown"), Countdown));
        app.add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    }

    /// The tone of `quota`: red after a 429 or under 5 % left, yellow
    /// under 25 %.
    pub(crate) fn tone(quota: &Quota) -> Tone {
        let left = quota.tighter().map_or(1.0, |tighter| tighter.left());
        if quota.limited.is_some() || left < 0.05 {
            Tone::Red
        } else if left < 0.25 {
            Tone::Yellow
        } else {
            Tone::Plain
        }
    }

    /// Each agent's item, of its model's provider, changed only when what
    /// it says does: on a new reply, a change of model, and each second of
    /// a countdown, which keeps the loop awake while it lasts.
    fn show(
        quotas: Res<Quotas>,
        time: Res<Time<Real>>,
        mut agents: Query<(Option<&Connection>, &mut StatusItems), With<Agent>>,
        countdown: Query<(Entity, Has<KeepAwake>), With<Countdown>>,
        mut commands: Commands,
    ) {
        let now = time.elapsed();
        let mut ticking = false;
        for (connection, mut items) in &mut agents {
            let quota = connection.and_then(|c| quotas.latest(c.spec.provider.vendor()));
            ticking |= quota.is_some_and(|quota| quota.ticking(now));
            let text = quota.and_then(|quota| quota.says(now)).unwrap_or_default();
            let item = QUOTA.says(text, quota.map_or(Tone::Plain, tone));
            let shown = items.iter().find(|shown| shown.name == QUOTA.name);
            if shown != Some(&item) && !(shown.is_none() && item.text.is_empty()) {
                items.show(item);
            }
        }
        for (entity, awake) in &countdown {
            if ticking && !awake {
                commands
                    .entity(entity)
                    .insert(KeepAwake(Duration::from_secs(1)));
            } else if !ticking && awake {
                commands.entity(entity).remove::<KeepAwake>();
            }
        }
    }
}
