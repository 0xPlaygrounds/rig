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
//! The provider's models connect through the pool ([`Provide`]), and every
//! model call, a turn's, a subagent's or a plugin's own, gets an [`Adapter`]
//! in Prepare that sends it on the pool's current key, which it stays on
//! for the prompt cache. A Review system reads a failed call: a rate limit
//! or a spent quota cools its key (`Retry-After`, else 60 s doubling to 15
//! minutes), a refused key is disabled for the session with one notice
//! naming its reference, and a turn's call is sent again at once on the
//! next key that does not cool down, superseding the failed one. With every
//! key cooling, the kernel's retry waits for the key that cools down first.
//! `/keys` shows each key's state, requests and tokens; the usage is saved
//! with the session. Key values live only in the handlers, never in a
//! component, notice or record.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rig_compaction::CompactionSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{Catalog, ModelSpec, Provide, Transport};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::providers::registry::{ConnectError, ConnectOptions, Format};
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::wire::Secret;
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The name of the pool's adapter on a model call.
const ADAPTER: &str = "key-pool";
/// How long a rate-limited key cools when the provider names no wait.
const FIRST_COOLDOWN: Duration = Duration::from_secs(60);
/// The longest cooldown the pool picks itself.
const MAX_COOLDOWN: Duration = Duration::from_secs(15 * 60);

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

/// How a model of a pooled provider is reached with one key: the model's
/// spec, the catalog, the transport, the key, and the provider's address
/// and protocol family from `key-pool.toml`. By default, as the registry connects it
/// with [`ConnectOptions`]; a test inserts its own before adding the plugin.
#[derive(Resource, Clone)]
pub struct KeyConnect(
    pub  Arc<
        dyn Fn(&ModelSpec, &Catalog, &Transport, &Secret, ConnectOptions) -> ConnectResult
            + Send
            + Sync,
    >,
);

/// A handler, or why the model does not connect.
pub type ConnectResult = Result<ErasedHandler, ConnectError>;

impl Default for KeyConnect {
    fn default() -> Self {
        Self(Arc::new(|spec, catalog, transport, key, options| {
            options
                .api_key(key.clone())
                .handler(spec, catalog, transport)
        }))
    }
}

/// The key values and how a key connects, shared with the adapters and
/// providers that build handlers. Never reflected, logged or saved.
struct Keyring {
    connect: KeyConnect,
    /// Per provider: the options its keys connect with, and each key's
    /// value, `None` when it is not set.
    vendors: HashMap<String, (ConnectOptions, Vec<Option<Secret>>)>,
}

impl Keyring {
    /// `spec`'s model on the key `index` of its provider's pool.
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
        index: usize,
    ) -> ConnectResult {
        let vendor = spec.provider.vendor();
        let key = self
            .vendors
            .get(vendor)
            .and_then(|(options, keys)| Some((options, keys.get(index)?.as_ref()?)));
        let Some((options, key)) = key else {
            return Err(ConnectError::Provider(format!(
                "key {index} of the {vendor} pool is not set"
            )));
        };
        (self.connect.0)(spec, catalog, transport, key, options.clone())
    }
}

/// The keyring, shared.
#[derive(Resource, Clone)]
struct Keys(Arc<Keyring>);

/// What a key can do now.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Health {
    Ok,
    /// Rate-limited until then, on Bevy's real clock.
    Cooling(Duration),
    /// Refused by the provider: out for the session.
    Disabled,
    /// Its variable or file is not set.
    Missing,
}

/// One key of a pool, by reference.
#[derive(Debug)]
struct KeyState {
    reference: String,
    health: Health,
    /// Rate limits in a row, which double the next cooldown.
    strikes: u32,
}

/// One provider's keys and the one its requests go to.
#[derive(Debug)]
struct Pool {
    vendor: String,
    current: usize,
    keys: Vec<KeyState>,
}

impl Pool {
    fn usable(&self, index: usize, now: Duration) -> bool {
        self.keys.get(index).is_some_and(|key| match key.health {
            Health::Ok => true,
            Health::Cooling(until) => until <= now,
            Health::Disabled | Health::Missing => false,
        })
    }

    /// The key a request goes to now: the current one while it is usable,
    /// else the next usable one, which becomes current, else the one that
    /// cools down first; `None` when every key is out.
    fn pick(&mut self, now: Duration) -> Option<usize> {
        let count = self.keys.len();
        let next = (0..count)
            .map(|step| (self.current + step) % count.max(1))
            .find(|&index| self.usable(index, now))
            .or_else(|| self.first_cooled().map(|(index, _)| index))?;
        self.current = next;
        Some(next)
    }

    /// The cooling key whose cooldown ends first, and when.
    fn first_cooled(&self) -> Option<(usize, Duration)> {
        let cooling = self.keys.iter().enumerate();
        cooling
            .filter_map(|(index, key)| match key.health {
                Health::Cooling(until) => Some((index, until)),
                _ => None,
            })
            .min_by_key(|&(_, until)| until)
    }

    /// Cools the key `index` for `asked`, else for its doubling cooldown.
    fn cool(&mut self, index: usize, asked: Option<Duration>, now: Duration) {
        if let Some(key) = self.keys.get_mut(index)
            && key.health != Health::Disabled
        {
            let doubled = FIRST_COOLDOWN.saturating_mul(1 << key.strikes.min(8));
            key.health = Health::Cooling(now + asked.unwrap_or(doubled.min(MAX_COOLDOWN)));
            key.strikes += 1;
        }
    }

    /// Disables the key `index`; its reference when it was not disabled.
    fn disable(&mut self, index: usize) -> Option<&str> {
        let key = self.keys.get_mut(index)?;
        let was = std::mem::replace(&mut key.health, Health::Disabled);
        (was != Health::Disabled).then_some(key.reference.as_str())
    }
}

/// Every pool, by provider, in the order of `key-pool.toml`.
#[derive(Resource, Default, Debug)]
struct Pools(Vec<Pool>);

impl Pools {
    fn get_mut(&mut self, vendor: &str) -> Option<&mut Pool> {
        self.0.iter_mut().find(|pool| pool.vendor == vendor)
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

/// On a model call: the key of each pool its adapter sends it on, should
/// it go to that provider.
#[derive(Component, Debug)]
struct OnKey(Vec<(String, usize)>);

impl OnKey {
    fn index(&self, vendor: &str) -> Option<usize> {
        let found = self.0.iter().find(|(pooled, _)| pooled == vendor);
        found.map(|&(_, index)| index)
    }
}

/// On a failed call the pool has read.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Judged;

/// The pool's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyPoolSystems {
    /// Prepare: puts the key adapter on each model call.
    Route,
    /// Review: cools or disables a failed call's key and sends a turn's
    /// call again on the next key.
    FailOver,
}

#[derive(Default)]
pub struct KeyPoolPlugin;

impl Plugin for KeyPoolPlugin {
    fn build(&self, app: &mut App) {
        let home = RigHome::of(app.world());
        let connect = app
            .world()
            .get_resource::<KeyConnect>()
            .cloned()
            .unwrap_or_default();
        let (pools, keys, problem) = load(&home, connect);
        app.insert_resource(pools)
            .insert_resource(PoolProblem(problem))
            .insert_resource(keys)
            .insert_resource(KeyUsage::default())
            .configure_sets(
                AgentPass,
                (
                    KeyPoolSystems::Route
                        .in_set(Stage::Prepare)
                        .before(CompactionSystems::Prepare),
                    KeyPoolSystems::FailOver
                        .in_set(Stage::Review)
                        .before(CompactionSystems::Recover),
                ),
            )
            .add_systems(Startup, provide)
            .add_systems(
                AgentPass,
                (
                    route.in_set(KeyPoolSystems::Route),
                    fail_over.in_set(KeyPoolSystems::FailOver),
                ),
            )
            .add_observer(count)
            .add_command(
                "keys",
                "Each pooled API key's state and usage",
                keys_command,
            );
    }
}

/// The pools of `home`'s `key-pool.toml`, their keys, and what was wrong
/// with the file.
fn load(home: &Home, connect: KeyConnect) -> (Pools, Keys, Option<String>) {
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
    let mut pools = Pools::default();
    let mut vendors = HashMap::new();
    for (vendor, config) in config {
        let mut options = ConnectOptions::new();
        if let Some(url) = &config.base_url {
            options = options.base_url(url.clone());
        }
        let format = config.format.as_deref().and_then(|name| {
            Format::ALL
                .into_iter()
                .find(|format| format.as_str() == name)
        });
        if let Some(format) = format {
            options = options.format(format);
        }
        let secrets: Vec<Option<Secret>> = config.keys.iter().map(|key| key.read(home)).collect();
        let keys = config.keys.iter().zip(&secrets);
        let keys = keys.map(|(key, secret)| KeyState {
            reference: key.name(),
            health: if secret.is_some() {
                Health::Ok
            } else {
                Health::Missing
            },
            strikes: 0,
        });
        pools.0.push(Pool {
            vendor: vendor.clone(),
            current: 0,
            keys: keys.collect(),
        });
        vendors.insert(vendor, (options, secrets));
    }
    let keys = Keys(Arc::new(Keyring { connect, vendors }));
    (pools, keys, problem)
}

/// What was wrong with `key-pool.toml`, said at startup.
#[derive(Resource)]
struct PoolProblem(Option<String>);

/// Each pooled provider's models connect through the pool, on its first
/// key that is set, so an agent can choose them with no key in the usual
/// variables; each call's adapter then picks the key it goes on.
fn provide(
    mut models: ResMut<Models>,
    keys: Res<Keys>,
    problem: Res<PoolProblem>,
    mut notices: MessageWriter<Notice>,
) {
    if let Some(problem) = &problem.0 {
        notices.write(Notice::error(None, problem.clone()));
    }
    let vendors: Vec<_> = keys.0.vendors.keys().cloned().collect();
    for vendor in vendors {
        let catalog = models.0.catalog();
        let found = catalog.iter().find(|spec| spec.provider.vendor() == vendor);
        let Some(provider) = found.map(|spec| spec.provider) else {
            continue;
        };
        let keyring = keys.0.clone();
        let first = move |spec: &ModelSpec, catalog: &Catalog, transport: &Transport| {
            let set = keyring.vendors.get(spec.provider.vendor());
            let first = set.and_then(|(_, keys)| keys.iter().position(Option::is_some));
            keyring.handler(spec, catalog, transport, first.unwrap_or(usize::MAX))
        };
        models.0.provide(provider, first);
    }
}

/// Puts the key adapter first on each model call as it is admitted, so the
/// other adapters wrap the keyed model: whichever model the call finally
/// goes to, it goes on the current key of that model's pool, if pooled.
fn route(
    mut calls: Query<(Entity, &mut ModelRequest), Added<Admitted>>,
    mut pools: ResMut<Pools>,
    keys: Res<Keys>,
    models: Res<Models>,
    time: Res<Time<Real>>,
    mut commands: Commands,
) {
    if pools.0.is_empty() {
        return;
    }
    let now = time.elapsed();
    for (call, mut request) in &mut calls {
        let picked: Vec<(String, usize)> = pools
            .0
            .iter_mut()
            .filter_map(|pool| Some((pool.vendor.clone(), pool.pick(now)?)))
            .collect();
        let (keyring, connector) = (keys.0.clone(), models.0.clone());
        let on: Arc<[(String, usize)]> = picked.clone().into();
        let adapter = Adapter::new(ADAPTER, move |connection: Connection| {
            let vendor = connection.spec.provider.vendor();
            let index = on.iter().find(|(pooled, _)| pooled == vendor);
            let keyed = index.map(|&(_, index)| {
                let (catalog, transport) = (connector.catalog(), connector.transport());
                keyring.handler(&connection.spec, catalog, transport, index)
            });
            match keyed {
                Some(Ok(handler)) => Connection {
                    handler: Handler(handler),
                    ..connection
                },
                _ => connection,
            }
        });
        request.adapters.retain(|adapter| adapter.name != ADAPTER);
        request.adapters.insert(0, adapter);
        commands.entity(call).insert(OnKey(picked));
    }
}

/// Whether `report` says the key is rate-limited or out of quota.
fn rate_limited(report: &ErrorReport) -> bool {
    matches!(report.http_status, Some(429 | 402))
        || report
            .code
            .as_deref()
            .is_some_and(|code| code.contains("quota"))
}

/// A failed call sent on a pooled key: a rate limit cools the key, a
/// refusal disables it with a notice. A turn's call is then sent again at
/// once, built anew, on the next key not cooling, superseding the failed
/// one; when every key cools, the kernel's retry waits for the first to
/// cool down. A plugin's own call keeps its failure: its plugin waits for
/// that entity's outcome, which a successor would not give it.
#[allow(clippy::type_complexity)]
fn fail_over(
    mut failed: Query<
        (
            Entity,
            &Outcome,
            &OnKey,
            &SentTo,
            &Effect,
            Option<&CallOf>,
            Option<&ForAgent>,
            Option<&mut RetryVerdict>,
        ),
        (
            Unreviewed,
            With<ModelRequest>,
            Without<SupersededBy>,
            Without<Judged>,
        ),
    >,
    turn_calls: Query<(), With<TurnCall>>,
    mut pools: ResMut<Pools>,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let now = time.elapsed();
    for (call, outcome, on, sent, effect, of, agent, verdict) in &mut failed {
        commands.entity(call).insert(Judged);
        let Err(report) = &outcome.0 else {
            continue;
        };
        let vendor = sent.0.spec.provider.vendor();
        let (Some(index), Some(pool)) = (on.index(vendor), pools.get_mut(vendor)) else {
            continue;
        };
        let agent = agent.map(|agent| agent.0);
        if rate_limited(report) {
            let wall = UNIX_EPOCH + Duration::from_millis(wall_ms());
            pool.cool(index, report.retry_after(wall), now);
        } else if matches!(report.http_status, Some(401 | 403)) {
            let status = report.http_status.unwrap_or_default();
            if let Some(reference) = pool.disable(index) {
                notices.write(Notice::error(
                    agent,
                    format!(
                        "The {vendor} key {reference} was refused ({status}); \
                         it is left out for the rest of the session."
                    ),
                ));
            }
        } else {
            continue;
        }
        let (Some(&CallOf(turn)), Some(agent)) = (of, agent) else {
            continue;
        };
        if !turn_calls.contains(call) {
            continue;
        }
        if (0..pool.keys.len()).any(|index| pool.usable(index, now)) {
            commands.spawn((
                TurnCall,
                Supersedes(call),
                CallOf(turn),
                ForAgent(agent),
                Effect::child_of(effect.parent),
            ));
        } else if let (Some(mut verdict), Some((_, until))) = (verdict, pool.first_cooled())
            && matches!(verdict.0, Verdict::Retry(_))
        {
            verdict.0 = Verdict::Retry(until.saturating_sub(now));
        }
    }
}

/// Milliseconds since the epoch, which a `Retry-After` date is read
/// against.
fn wall_ms() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |since| since.as_millis() as u64)
}

/// Counts each model call that went out on a pooled key, and its tokens;
/// a reply resets its key's rate-limit strikes.
fn count(
    settled: On<Add<Settled>>,
    calls: Query<(&OnKey, &SentTo, &Outcome)>,
    mut pools: ResMut<Pools>,
    mut usage: ResMut<KeyUsage>,
) {
    let Ok((on, sent, outcome)) = calls.get(settled.entity) else {
        return;
    };
    let vendor = sent.0.spec.provider.vendor();
    let (Some(index), Some(pool)) = (on.index(vendor), pools.get_mut(vendor)) else {
        return;
    };
    let Some(key) = pool.keys.get_mut(index) else {
        return;
    };
    let count = usage
        .0
        .entry(format!("{vendor} {}", key.reference))
        .or_default();
    count.requests += 1;
    if let Some(response) = outcome.completion() {
        let usage = &response.usage;
        let sum = usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0);
        count.tokens += usage.total_tokens.unwrap_or(sum);
        key.strikes = 0;
    }
}

/// `/keys`: each pooled key's reference, state, and requests and tokens
/// this session.
fn keys_command(
    In(args): In<CommandArgs>,
    pools: Res<Pools>,
    usage: Res<KeyUsage>,
    time: Res<Time<Real>>,
    mut notices: MessageWriter<Notice>,
) {
    if pools.0.is_empty() {
        let text = "No key pool: list keys in RIG_HOME/key-pool.toml.";
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let now = time.elapsed();
    let mut lines = Vec::new();
    for pool in &pools.0 {
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
