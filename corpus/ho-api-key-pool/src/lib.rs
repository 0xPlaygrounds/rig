//! Plugin corpus (held-out): H05, an API-key pool with failover.
//!
//! `RIG_HOME/key-pool.toml` lists, per provider, keys by reference only:
//!
//! ```toml
//! [deepseek]
//! keys = [{ env = "DEEPSEEK_KEY_1" }, { env = "DEEPSEEK_KEY_2" }, { file = "keys/spare" }]
//! # base_url = "http://127.0.0.1:8080/v1"; format = "openai"
//! ```
//!
//! The provider's models connect with a stand-in key, and the pool wraps
//! the HTTP client every model is reached through
//! ([`Connector::wrap_http`]). Each request carrying a pool's stand-in goes
//! out on that pool's current key, which it stays on for the prompt cache.
//! A rate limit or a spent quota cools the key (`Retry-After`, else 60 s
//! doubling to 15 minutes) and a refused key is disabled for the session;
//! either way the same request is sent again at once on the next key, so
//! no model call, a turn's, a subagent's or a plugin's own, sees the
//! failure. With every key cooling, the failure goes up asking to wait for
//! the key that cools down first, and the kernel's retry waits for it.
//! `/keys` shows each key's state, requests and tokens; the usage is saved
//! with the session. Key values live only in the pool, never in a request
//! an effect sees, a notice or a record.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime};

use rig_harness::prelude::*;
use rig_harness::rig_core::ProviderError;
use rig_harness::rig_core::catalog::Connector;
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::http_client::{
    self, DynHttpClient, HeaderValue, HttpClientExt, LazyBody, LazyBytes, MultipartForm, Request,
    Response, StreamingResponse,
};
use rig_harness::rig_core::providers::registry::{ConnectOptions, Format};
use rig_harness::rig_core::wasm_compat::WasmCompatSend;
use rig_harness::rig_core::wire::Secret;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How long a rate-limited key cools when the provider names no wait.
const FIRST_COOLDOWN: Duration = Duration::from_secs(60);
/// The longest cooldown the pool picks itself.
const MAX_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// `bytes::Bytes`, the body type of rig-core's HTTP client trait, named
/// through a public alias: neither rig-core nor rig-http re-exports it.
type Bytes = <<LazyBytes as Future>::Output as OkOf>::Ok;

/// The success type of a `Result`, for [`Bytes`].
trait OkOf {
    type Ok;
}

impl<T, E> OkOf for Result<T, E> {
    type Ok = T;
}

/// One provider's table of `key-pool.toml`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct PoolConfig {
    keys: Vec<KeyRef>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    format: Option<String>,
}

/// Where a key is read from: `{ env = "NAME" }` or `{ file = "path" }`, a
/// path relative to `RIG_HOME`.
#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum KeyRef {
    Env(String),
    File(PathBuf),
}

impl KeyRef {
    /// What the key is called everywhere: the variable's name or the path.
    fn name(&self) -> String {
        match self {
            Self::Env(name) => name.clone(),
            Self::File(path) => path.display().to_string(),
        }
    }

    /// The key's value, if set.
    fn read(&self, home: &Home) -> Option<Secret> {
        let value = match self {
            Self::Env(name) => std::env::var(name).ok()?,
            Self::File(path) => std::fs::read_to_string(home.root().join(path)).ok()?,
        };
        let value = value.trim();
        (!value.is_empty()).then(|| Secret::from(value))
    }
}

/// What a key can do now.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Health {
    Ok,
    /// Rate-limited until then.
    Cooling(Instant),
    /// Refused by the provider: out for the session.
    Disabled,
    /// Its variable or file is not set.
    Missing,
}

/// One key of a pool.
struct Key {
    reference: String,
    value: Option<Secret>,
    health: Health,
    /// Rate limits in a row, which double the next cooldown.
    strikes: u32,
    /// Requests sent on it and not yet counted in [`KeyUsage`].
    sent: u64,
}

/// One provider's keys and the one its requests go to.
struct Pool {
    vendor: String,
    /// The provider's models connect with this in place of a key; the
    /// pool puts the key it picks wherever a request carries it.
    stand_in: String,
    /// Where the provider is reached, from `key-pool.toml`.
    options: ConnectOptions,
    current: usize,
    keys: Vec<Key>,
    /// The references of keys refused since the last notice.
    refused: Vec<String>,
}

/// What became of a request sent on a key.
enum Failed {
    /// Try the next key.
    Next,
    /// Give the failure up, as it came.
    Up,
    /// Give it up, asking to wait this long: every key cools.
    Wait(Duration),
}

impl Pool {
    fn usable(&self, index: usize, now: Instant) -> bool {
        self.keys.get(index).is_some_and(|key| match key.health {
            Health::Ok => true,
            Health::Cooling(until) => until <= now,
            Health::Disabled | Health::Missing => false,
        })
    }

    /// The key a request goes to now, which becomes current, and its
    /// value: the current one while it is usable, else the next usable
    /// one not `tried`, else, for a request not tried on any, the one that
    /// cools down first; `None` when no key is left.
    fn pick(&mut self, tried: &[usize], now: Instant) -> Option<(usize, Secret)> {
        let count = self.keys.len();
        let usable = (0..count)
            .map(|step| (self.current + step) % count)
            .find(|index| !tried.contains(index) && self.usable(*index, now));
        let first = || tried.is_empty().then(|| self.first_cooled()).flatten();
        let index = usable.or_else(|| first().map(|(index, _)| index))?;
        let key = self.keys.get_mut(index)?;
        key.sent += 1;
        self.current = index;
        Some((index, key.value.clone()?))
    }

    /// The cooling key whose cooldown ends first, and when.
    fn first_cooled(&self) -> Option<(usize, Instant)> {
        let cooling = self.keys.iter().enumerate();
        cooling
            .filter_map(|(index, key)| match key.health {
                Health::Cooling(until) => Some((index, until)),
                _ => None,
            })
            .min_by_key(|&(_, until)| until)
    }

    /// Reads a request's failure on the key `index`, after it `tried`
    /// those: a rate limit or a spent quota cools the key, a refusal
    /// disables it, and either sends the request on to the next key, if
    /// one it did not try is usable.
    fn failed(
        &mut self,
        index: usize,
        tried: &[usize],
        error: &http_client::Error,
        now: Instant,
    ) -> Failed {
        let (Some(status), Some(headers), Some(body)) = (
            error.non_success_status(),
            error.non_success_headers(),
            error.non_success_body(),
        ) else {
            return Failed::Up;
        };
        let reply = ProviderError::from_http_response(status, body);
        let report = ErrorReport::from(reply.with_response_headers(Some(headers.clone())));
        let Some(key) = self.keys.get_mut(index) else {
            return Failed::Up;
        };
        let rate_limited = matches!(report.http_status, Some(429 | 402))
            || report
                .code
                .as_deref()
                .is_some_and(|code| code.contains("quota"));
        if rate_limited {
            let doubled = FIRST_COOLDOWN.saturating_mul(1 << key.strikes.min(8));
            let wait = report.retry_after(SystemTime::now());
            key.health = Health::Cooling(now + wait.unwrap_or(doubled.min(MAX_COOLDOWN)));
            key.strikes += 1;
        } else if matches!(report.http_status, Some(401 | 403)) {
            if key.health != Health::Disabled {
                key.health = Health::Disabled;
                self.refused.push(key.reference.clone());
            }
        } else {
            return Failed::Up;
        }
        let mut untried = (0..self.keys.len()).filter(|index| !tried.contains(index));
        if untried.any(|index| self.usable(index, now)) {
            Failed::Next
        } else if let Some((_, until)) = self.first_cooled() {
            Failed::Wait(until.saturating_duration_since(now))
        } else {
            Failed::Up
        }
    }

    /// A reply came on the key `index`: its rate limits are over.
    fn answered(&mut self, index: usize) {
        if let Some(key) = self.keys.get_mut(index) {
            key.strikes = 0;
        }
    }
}

/// Every pool, by provider, in the order of `key-pool.toml`, shared with
/// the HTTP client wrapper. Never reflected, logged or saved.
#[derive(Resource, Clone, Default)]
struct Pools(Arc<Mutex<Vec<Pool>>>);

impl Pools {
    fn lock(&self) -> MutexGuard<'_, Vec<Pool>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The requests and tokens each key served, by `vendor reference`. Saved
/// with the session.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource, Saved, Default)]
pub struct KeyUsage(pub HashMap<String, KeyCount>);

/// One key's usage.
#[derive(Reflect, Default, Debug, Clone, Copy)]
pub struct KeyCount {
    pub requests: u64,
    pub tokens: u64,
}

/// What was wrong with `key-pool.toml`, said at startup.
#[derive(Resource)]
struct PoolProblem(Option<String>);

#[derive(Default)]
pub struct KeyPoolPlugin;

impl Plugin for KeyPoolPlugin {
    fn build(&self, app: &mut App) {
        let (pools, problem) = load(&RigHome::of(app.world()));
        let models = &mut app.world_mut().get_resource_or_init::<Models>().0;
        wrap(models, pools.clone());
        app.insert_resource(pools)
            .insert_resource(PoolProblem(problem))
            .insert_resource(KeyUsage::default())
            .add_systems(Startup, provide)
            .add_observer(count)
            .add_command(
                "keys",
                "Each pooled API key's state and usage",
                keys_command,
            );
    }
}

/// The pools of `home`'s `key-pool.toml`, and what was wrong with it.
fn load(home: &Home) -> (Pools, Option<String>) {
    let path = home.root().join("key-pool.toml");
    let mut problem = None;
    let config: Vec<(String, PoolConfig)> = match std::fs::read_to_string(&path) {
        Ok(text) => match toml::from_str::<toml::Table>(&text) {
            Ok(table) => table
                .into_iter()
                .filter_map(|(vendor, value)| match value.try_into::<PoolConfig>() {
                    Ok(config) => Some((vendor, config)),
                    Err(why) => {
                        problem = Some(format!("{}: [{vendor}]: {why}", path.display()));
                        None
                    }
                })
                .collect(),
            Err(why) => {
                problem = Some(format!("{}: {why}", path.display()));
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    };
    let pools = config.into_iter().map(|(vendor, config)| {
        let mut options = ConnectOptions::new();
        if let Some(url) = config.base_url {
            options = options.base_url(url);
        }
        let format = config.format.as_deref().and_then(|name| {
            Format::ALL
                .into_iter()
                .find(|format| format.as_str() == name)
        });
        if let Some(format) = format {
            options = options.format(format);
        }
        let keys = config.keys.iter().map(|reference| {
            let value = reference.read(home);
            Key {
                reference: reference.name(),
                health: if value.is_some() {
                    Health::Ok
                } else {
                    Health::Missing
                },
                value,
                strikes: 0,
                sent: 0,
            }
        });
        Pool {
            stand_in: format!("rig-key-pool.{vendor}"),
            vendor,
            options,
            current: 0,
            keys: keys.collect(),
            refused: Vec::new(),
        }
    });
    (Pools(Arc::new(Mutex::new(pools.collect()))), problem)
}

/// Each pooled provider's models connect with the pool's stand-in for a
/// key, so an agent can choose them with no key in the usual variables.
fn provide(
    mut models: ResMut<Models>,
    pools: Res<Pools>,
    problem: Res<PoolProblem>,
    mut notices: MessageWriter<Notice>,
) {
    if let Some(problem) = &problem.0 {
        notices.write(Notice::error(None, problem.clone()));
    }
    for pool in pools.lock().iter() {
        let catalog = models.0.catalog();
        let found = catalog
            .iter()
            .find(|spec| spec.provider.vendor() == pool.vendor);
        if let Some(provider) = found.map(|spec| spec.provider) {
            let options = pool.options.clone().api_key(pool.stand_in.as_str());
            models.0.provide(provider, options);
        }
    }
}

/// Wraps every HTTP client `models` reaches a model through in the pool.
fn wrap(models: &mut Connector, pools: Pools) {
    models.wrap_http(move |inner| {
        DynHttpClient::new(Rotating {
            inner,
            pools: pools.clone(),
        })
    });
}

/// An HTTP client that sends each request carrying a pool's stand-in on
/// that pool's keys, the next one after a rate limit or a refusal.
#[derive(Clone)]
struct Rotating {
    inner: DynHttpClient,
    pools: Pools,
}

impl Rotating {
    /// Sends `request` with `send`, on the keys of the pool whose stand-in
    /// it carries until one answers or the failure is given up; a request
    /// for no pool goes as it is.
    async fn send_keyed<R, F>(
        self,
        request: Request<Bytes>,
        send: impl Fn(DynHttpClient, Request<Bytes>) -> F,
    ) -> http_client::Result<R>
    where
        F: Future<Output = http_client::Result<R>>,
    {
        let (head, body) = request.into_parts();
        let carries = |stand_in: &str| {
            head.uri.to_string().contains(stand_in)
                || head.headers.values().any(|value| {
                    let text = value.to_str().unwrap_or_default();
                    text.contains(stand_in)
                })
        };
        let pool = self
            .pools
            .lock()
            .iter()
            .position(|pool| carries(&pool.stand_in));
        let Some(pool) = pool else {
            return send(self.inner, Request::from_parts(head, body)).await;
        };
        let mut tried = Vec::new();
        loop {
            let picked = self.pools.lock().get_mut(pool).and_then(|pool| {
                let (index, key) = pool.pick(&tried, Instant::now())?;
                Some((index, key, pool.stand_in.clone()))
            });
            let Some((index, key, stand_in)) = picked else {
                let why = "no key of the pool can be used";
                return Err(http_client::Error::instance(std::io::Error::other(why)));
            };
            tried.push(index);
            let keyed = keyed(&head, body.clone(), &stand_in, &key)?;
            let error = match send(self.inner.clone(), keyed).await {
                Ok(reply) => {
                    if let Some(pool) = self.pools.lock().get_mut(pool) {
                        pool.answered(index);
                    }
                    return Ok(reply);
                }
                Err(error) => error,
            };
            let failed = self
                .pools
                .lock()
                .get_mut(pool)
                .map(|pool| pool.failed(index, &tried, &error, Instant::now()));
            match failed {
                Some(Failed::Next) => {}
                Some(Failed::Wait(wait)) => return Err(waiting(error, wait)),
                Some(Failed::Up) | None => return Err(error),
            }
        }
    }
}

/// The request `head` with `body`, with `key` wherever it carries
/// `stand_in`, marked sensitive.
fn keyed(
    head: &http::request::Parts,
    body: Bytes,
    stand_in: &str,
    key: &Secret,
) -> http_client::Result<Request<Bytes>> {
    let mut request = Request::new(body);
    *request.method_mut() = head.method.clone();
    *request.version_mut() = head.version;
    let uri = head.uri.to_string();
    *request.uri_mut() = match uri.contains(stand_in) {
        true => uri
            .replace(stand_in, key.expose())
            .parse()
            .map_err(http_client::Error::instance)?,
        false => head.uri.clone(),
    };
    let mut headers = head.headers.clone();
    for value in headers.values_mut() {
        let text = value.to_str().unwrap_or_default();
        if text.contains(stand_in) {
            let mut with_key = HeaderValue::from_str(&text.replace(stand_in, key.expose()))
                .map_err(http_client::Error::instance)?;
            with_key.set_sensitive(true);
            *value = with_key;
        }
    }
    *request.headers_mut() = headers;
    Ok(request)
}

/// `error`, a rate limit, asking to wait `wait` instead, as the key that
/// cools down first does.
fn waiting(error: http_client::Error, wait: Duration) -> http_client::Error {
    let (Some(status), Some(headers), Some(body)) = (
        error.non_success_status(),
        error.non_success_headers(),
        error.non_success_body(),
    ) else {
        return error;
    };
    let mut headers = headers.clone();
    headers.remove("retry-after-ms");
    let seconds = wait.as_secs_f64().ceil() as u64;
    headers.insert("retry-after", HeaderValue::from(seconds));
    http_client::Error::non_success_with_details(status, headers, body.to_owned())
}

impl HttpClientExt for Rotating {
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
        self.clone()
            .send_keyed(request, |inner, request| async move {
                inner.send::<Bytes, U>(request).await
            })
    }

    /// A multipart body is sent once, on the current key: it cannot be
    /// sent again. No model call sends one.
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
        self.clone()
            .send_keyed(request, |inner, request| async move {
                inner.send_streaming(request).await
            })
    }
}

/// When a model call settles: counts the requests each key sent and the
/// call's tokens, on the key its provider's pool is on, and says which
/// keys were refused since.
fn count(
    settled: On<Add<Settled>>,
    calls: Query<(&SentTo, &Outcome, Option<&ForAgent>)>,
    pools: Res<Pools>,
    mut usage: ResMut<KeyUsage>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((sent, outcome, agent)) = calls.get(settled.entity) else {
        return;
    };
    let agent = agent.map(|agent| agent.0);
    let vendor = sent.0.spec.provider.vendor();
    for pool in pools.lock().iter_mut() {
        for key in &mut pool.keys {
            let entry = format!("{} {}", pool.vendor, key.reference);
            usage.0.entry(entry).or_default().requests += std::mem::take(&mut key.sent);
        }
        for reference in pool.refused.drain(..) {
            notices.write(Notice::warn(
                agent,
                format!(
                    "The {} key {reference} was refused; it is left out for the rest \
                     of the session.",
                    pool.vendor
                ),
            ));
        }
        let current = pool.keys.get(pool.current);
        if let (true, Some(key), Some(response)) =
            (pool.vendor == vendor, current, outcome.completion())
        {
            let used = &response.usage;
            let sum = used.input_tokens.unwrap_or(0) + used.output_tokens.unwrap_or(0);
            let entry = format!("{vendor} {}", key.reference);
            usage.0.entry(entry).or_default().tokens += used.total_tokens.unwrap_or(sum);
        }
    }
}

/// `/keys`: each pooled key's reference, state, and requests and tokens
/// this session.
fn keys_command(
    In(args): In<CommandArgs>,
    pools: Res<Pools>,
    usage: Res<KeyUsage>,
    mut notices: MessageWriter<Notice>,
) {
    let pools = pools.lock();
    if pools.is_empty() {
        let text = "No key pool: list keys in RIG_HOME/key-pool.toml.";
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let now = Instant::now();
    let mut lines = Vec::new();
    for pool in pools.iter() {
        for (index, key) in pool.keys.iter().enumerate() {
            let state = match key.health {
                Health::Cooling(until) if until > now => {
                    format!("cooling {} s", (until - now).as_secs_f64().ceil())
                }
                Health::Ok | Health::Cooling(_) => "ok".to_owned(),
                Health::Disabled => "disabled".to_owned(),
                Health::Missing => "not set".to_owned(),
            };
            let current = if index == pool.current {
                " (current)"
            } else {
                ""
            };
            let used = usage.0.get(&format!("{} {}", pool.vendor, key.reference));
            let used = used.copied().unwrap_or_default();
            lines.push(format!(
                "{} {}{current}: {state}, {} requests, {} tokens",
                pool.vendor, key.reference, used.requests, used.tokens
            ));
        }
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
