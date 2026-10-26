//! CT-21: Prometheus metrics endpoint (conversation-over-time fresh
//! held-out spec `21-metrics-endpoint.md`).
//!
//! `rig --metrics=127.0.0.1:9464`, or `addr` in the `[metrics]` settings,
//! serves `/metrics` in the Prometheus text format from threads of its own,
//! which read a registry the app updates and never wait on the app's loop.
//! Off by default; an address that is not loopback needs
//! `metrics_public = true` in `[metrics]`, else the endpoint is refused
//! with an error notice.
//!
//! Counters: model requests, input, output and cached tokens and cost (by
//! model), tool calls and tool errors (by tool), errors the process logged
//! (by plugin, from `rig_basics::telemetry`'s `Diagnostics`) and retries (by model).
//! Histograms: turn duration, model time to first token (by model), tool
//! call duration (by tool). Gauges: agents busy, subagents running,
//! approvals waiting, each agent's context fill ratio and seconds since its
//! last activity (by agent id). Every series has a `session` label: the
//! session's `/name`, else its id. No label holds prompt text, paths or
//! arguments, and each label takes at most
//! [`MAX_LABEL_VALUES`](registry::MAX_LABEL_VALUES) values, later ones
//! folded into `other`. Counters live as long as the process: the agent
//! runs one session per process, so `/new` and `/resume` start them again.

use std::collections::BTreeSet;
use std::time::{Instant, SystemTime};

use rig_basics::approval::NeedsApproval;
use rig_basics::sessions::SessionTitle;
use rig_basics::telemetry::diagnostics::Diagnostics;
use rig_basics::telemetry::diagnostics::MAX_DIAGNOSTICS;
use rig_harness::prelude::*;
use serde::Deserialize;

pub mod registry;
mod server;

use registry::{Labels, SharedRegistry};

/// The option naming the address to serve on, such as `127.0.0.1:9464`.
pub const METRICS: &str = "metrics";

/// The `[metrics]` settings.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsSettings {
    /// The address to serve on; `--metrics=ADDR` overrides it. Port 0 picks
    /// a free one.
    pub addr: Option<String>,
    /// Whether an address that is not loopback may be bound.
    pub metrics_public: bool,
}

/// Where the endpoint serves, once it does.
#[derive(Resource, Clone, Copy, Debug)]
pub struct MetricsEndpoint {
    /// The bound address.
    pub addr: std::net::SocketAddr,
}

/// The Prometheus endpoint. Serves nothing without `--metrics` or
/// `[metrics] addr`.
#[derive(Default)]
pub struct MetricsEndpointPlugin;

impl Plugin for MetricsEndpointPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(
            Flag::new(METRICS, "Serve Prometheus metrics at /metrics on ADDR").value("ADDR"),
        )
        .add_setting::<MetricsSettings>(METRICS);
        let world = app.world();
        let flag = world
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.flag_value(METRICS).map(str::to_owned));
        let settings = world.resource::<MetricsSettings>();
        let Some(addr) = flag.or_else(|| settings.addr.clone()) else {
            return;
        };
        let public = settings.metrics_public;
        app.init_resource::<SharedRegistry>()
            .add_systems(
                Startup,
                move |registry: Res<SharedRegistry>,
                      mut commands: Commands,
                      mut notices: MessageWriter<Notice>| {
                    match server::start(&addr, public, registry.clone()) {
                        Ok(listener) => {
                            let addr = listener.addr();
                            notices.write(Notice::info(
                                None,
                                format!("Metrics on http://{addr}/metrics"),
                            ));
                            commands.insert_resource(MetricsEndpoint { addr });
                            commands.insert_resource(listener);
                        }
                        Err(why) => {
                            notices.write(Notice::error(None, why));
                        }
                    }
                },
            )
            .add_systems(
                Last,
                (
                    gauges,
                    plugin_errors.run_if(resource_exists_and_changed::<Diagnostics>),
                ),
            )
            .add_observer(model_settled)
            .add_observer(tool_settled)
            .add_observer(turn_started)
            .add_observer(turn_settled)
            .add_observer(first_token)
            .add_observer(retried)
            .add_observer(started_activity)
            .add_observer(settled_activity);
    }
}

/// The model a call went to: where it was sent, else its own connection.
fn model_of(request: &ModelRequest, sent: Option<&SentTo>) -> String {
    sent.map(|SentTo(connection)| connection)
        .or(request.connection.as_ref())
        .map_or_else(
            || "unknown".to_owned(),
            |connection| connection.spec.reference(),
        )
}

/// Counts a settled model call that was sent, with its reply's tokens and
/// cost.
fn model_settled(
    settled: On<Add<Settled>>,
    calls: Query<(&ModelRequest, Option<&SentTo>, Option<&Outcome>)>,
    registry: Res<SharedRegistry>,
) {
    let Ok((request, sent @ Some(_), outcome)) = calls.get(settled.entity) else {
        return;
    };
    let mut registry = registry.lock();
    let model = vec![registry.label("model", &model_of(request, sent))];
    registry.add(registry::MODEL_REQUESTS, model.clone(), 1.0);
    let Some(usage) = outcome
        .and_then(Outcome::completion)
        .map(|reply| reply.usage)
    else {
        return;
    };
    let counts = [
        (registry::INPUT_TOKENS, usage.input_tokens),
        (registry::OUTPUT_TOKENS, usage.output_tokens),
        (registry::CACHED_TOKENS, usage.cached_input_tokens),
    ];
    for (name, tokens) in counts {
        registry.add(name, model.clone(), tokens.unwrap_or(0) as f64);
    }
    registry.add(
        registry::COST,
        model,
        usage.cost.map_or(0.0, |cost| cost.total),
    );
}

/// Counts a settled tool call, its failure, and how long it ran.
fn tool_settled(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome, Option<&Started>)>,
    registry: Res<SharedRegistry>,
) {
    let Ok((request, outcome, started)) = calls.get(settled.entity) else {
        return;
    };
    let mut registry = registry.lock();
    let tool = vec![registry.label("tool", request.call.function.name.as_str())];
    registry.add(registry::TOOL_CALLS, tool.clone(), 1.0);
    if outcome.tool_result().is_error() {
        registry.add(registry::TOOL_ERRORS, tool.clone(), 1.0);
    }
    if let Some(started) = started {
        let ran = SystemTime::now()
            .duration_since(started.at)
            .unwrap_or_default();
        registry.observe(registry::TOOL_DURATION, tool, ran.as_secs_f64());
    }
}

/// On a turn: when it began, in this process.
#[derive(Component)]
struct TurnClock(Instant);

fn turn_started(started: On<Add<TurnOf>>, mut commands: Commands) {
    commands
        .entity(started.entity)
        .try_insert(TurnClock(Instant::now()));
}

/// Observes a turn's duration as it ends.
fn turn_settled(
    settled: On<Add<Settled>>,
    turns: Query<&TurnClock>,
    registry: Res<SharedRegistry>,
) {
    if let Ok(TurnClock(began)) = turns.get(settled.entity) {
        let took = began.elapsed().as_secs_f64();
        registry
            .lock()
            .observe(registry::TURN_DURATION, Labels::new(), took);
    }
}

/// Observes how long a model call waited for its first streamed item.
fn first_token(
    streamed: On<Add<StreamTiming>>,
    calls: Query<(&StreamTiming, &Started, &ModelRequest, Option<&SentTo>)>,
    registry: Res<SharedRegistry>,
) {
    if let Ok((streamed, started, request, sent)) = calls.get(streamed.entity) {
        let waited = streamed
            .first
            .duration_since(started.at)
            .unwrap_or_default();
        let mut registry = registry.lock();
        let model = vec![registry.label("model", &model_of(request, sent))];
        registry.observe(registry::FIRST_TOKEN, model, waited.as_secs_f64());
    }
}

/// Counts each retry the kernel waits for, on the hold it puts on the
/// failed call.
fn retried(
    backoff: On<Add<Backoff>>,
    holds: Query<&HoldOf>,
    calls: Query<(&ModelRequest, Option<&SentTo>)>,
    registry: Res<SharedRegistry>,
) {
    let call = holds
        .get(backoff.entity)
        .ok()
        .and_then(|&HoldOf(call)| calls.get(call).ok());
    let model = call.map_or_else(
        || "unknown".to_owned(),
        |(request, sent)| model_of(request, sent),
    );
    let mut registry = registry.lock();
    let model = vec![registry.label("model", &model)];
    registry.add(registry::RETRIES, model, 1.0);
}

/// Notes the activity of the agent an effect works for.
fn note_activity(
    entity: Entity,
    effects: &Query<&ForAgent>,
    ids: &Query<&AgentId>,
    registry: &SharedRegistry,
) {
    let Some(id) = effects
        .get(entity)
        .ok()
        .and_then(|&ForAgent(agent)| ids.get(agent).ok())
    else {
        return;
    };
    let mut registry = registry.lock();
    let (_, agent) = registry.label("agent", &id.0);
    registry.active(agent, SystemTime::now());
}

fn started_activity(
    started: On<Add<Started>>,
    effects: Query<&ForAgent>,
    ids: Query<&AgentId>,
    registry: Res<SharedRegistry>,
) {
    note_activity(started.entity, &effects, &ids, &registry);
}

fn settled_activity(
    settled: On<Add<Settled>>,
    effects: Query<&ForAgent>,
    ids: Query<&AgentId>,
    registry: Res<SharedRegistry>,
) {
    note_activity(settled.entity, &effects, &ids, &registry);
}

/// An agent as the gauges read it.
type AgentGauge<'a> = (
    &'a AgentId,
    Has<ActiveTurn>,
    Has<SpawnedBy>,
    &'a LastUsage,
    Option<&'a Connection>,
    Has<ContextStale>,
);

/// Sets every gauge from the world as the frame left it, the session label,
/// and each agent's activity from what streamed.
fn gauges(
    agents: Query<AgentGauge, With<Agent>>,
    asks: Query<&HoldOf, (With<Ask>, Without<Outcome>, Without<Settled>)>,
    approvals: Query<(), With<NeedsApproval>>,
    mut deltas: MessageReader<StreamDelta>,
    ids: Query<&AgentId>,
    (title, paths): (Option<Res<SessionTitle>>, Option<Res<SessionPaths>>),
    registry: Res<SharedRegistry>,
) {
    let mut registry = registry.lock();
    let named = title.and_then(|title| title.name.clone());
    let id = paths.and_then(|paths| paths.id()).map(|id| id.to_string());
    registry.set_session(named.or(id).as_deref().unwrap_or("none"));
    let streamed: BTreeSet<Entity> = deltas.read().filter_map(|delta| delta.agent).collect();
    for agent in streamed {
        if let Ok(id) = ids.get(agent) {
            let (_, agent) = registry.label("agent", &id.0);
            registry.active(agent, SystemTime::now());
        }
    }
    let busy = agents.iter().filter(|(_, busy, ..)| *busy).count();
    let subagents = agents
        .iter()
        .filter(|(_, busy, spawned, ..)| *busy && *spawned)
        .count();
    let waiting = asks
        .iter()
        .filter(|&&HoldOf(held)| approvals.contains(held))
        .count();
    let mut gauges = vec![
        ((registry::AGENTS_BUSY, Labels::new()), busy as f64),
        (
            (registry::SUBAGENTS_RUNNING, Labels::new()),
            subagents as f64,
        ),
        ((registry::APPROVALS_WAITING, Labels::new()), waiting as f64),
    ];
    let mut present = BTreeSet::new();
    for (id, _, _, last, connection, stale) in &agents {
        let label = registry.label("agent", &id.0);
        present.insert(label.1.clone());
        if stale {
            continue;
        }
        let limit = connection.and_then(|connection| connection.spec.input_limit());
        if let (Some(tokens), Some(limit)) = (last.context(), limit.filter(|limit| *limit > 0)) {
            gauges.push((
                (registry::CONTEXT_FILL, vec![label]),
                tokens as f64 / f64::from(limit),
            ));
        }
    }
    registry.keep_activity(&present);
    registry.set_gauges(gauges);
}

/// What tells a logged warning or error from another.
type Fingerprint = (NoticeLevel, String, String, Option<String>);

fn fingerprint(logged: &Logged) -> Fingerprint {
    (
        logged.level,
        logged.target.clone(),
        logged.message.clone(),
        logged.plugin.clone(),
    )
}

/// Counts the errors added to `rig_basics::telemetry`'s [`Diagnostics`] since the
/// last change, by the plugin that logged them. It keeps the newest
/// [`MAX_DIAGNOSTICS`] and has no event per entry, so the new ones are those after the
/// last one counted.
fn plugin_errors(
    diagnostics: Res<Diagnostics>,
    mut last: Local<Option<Fingerprint>>,
    registry: Res<SharedRegistry>,
) {
    let kept = &diagnostics.0;
    let seen = last
        .as_ref()
        .and_then(|last| kept.iter().rposition(|logged| fingerprint(logged) == *last))
        .map_or(0, |at| at + 1);
    let fresh = kept.iter().skip(seen.min(MAX_DIAGNOSTICS));
    let mut registry = registry.lock();
    for logged in fresh.filter(|logged| logged.level == NoticeLevel::Error) {
        let plugin = vec![registry.label("plugin", logged.plugin.as_deref().unwrap_or("unknown"))];
        registry.add(registry::PLUGIN_ERRORS, plugin, 1.0);
    }
    *last = kept.back().map(fingerprint);
}

#[cfg(test)]
mod tests;
