//! Plugin corpus: 15. Model fallback chain (spec `15-model-fallback.md`).
//!
//! When a model call fails for good with a rate limit, an overload, a
//! server error or a refused key, the turn goes on with the next model of
//! the chain that is not cooling down, and the model that failed is skipped
//! for the cooldown; after it, new turns go back to it. A context overflow
//! is left to compaction, and a failure another plugin took over is left
//! to it. The chain is `RIG_HOME/fallback.json`:
//!
//! ```json
//! { "chain": ["anthropic/claude-opus-4-5", "openai/gpt-5"], "cooldown": "5m" }
//! ```

use std::collections::HashMap;

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_ecs::turn::ModelCall;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// How long a model that failed is skipped when the config does not say.
pub const COOLDOWN: Duration = Duration::from_secs(5 * 60);

/// The models a turn falls back to, in order, as catalog references, and
/// how long a model that failed is skipped.
#[derive(Resource, Clone, Debug)]
pub struct FallbackChain {
    /// The models, the preferred first.
    pub chain: Vec<String>,
    /// How long a model that failed is skipped.
    pub cooldown: Duration,
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self {
            chain: Vec::new(),
            cooldown: COOLDOWN,
        }
    }
}

/// The models that failed, by catalog reference, and until when they are
/// skipped, on the real clock.
#[derive(Resource, Default, Debug)]
pub struct Cooling(pub HashMap<String, Duration>);

/// The chain's models as connected, by catalog reference: each connects
/// once.
#[derive(Resource, Default, Clone)]
pub struct Connected(pub HashMap<String, Connection>);

/// Each agent's last failed model call of its turn, and the model it was
/// sent to; a reply clears it.
#[derive(Resource, Default)]
struct LastFailure(HashMap<Entity, (String, ErrorReport)>);

#[derive(Default)]
pub struct ModelFallbackPlugin;

impl Plugin for ModelFallbackPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FallbackChain>()
            .init_resource::<Cooling>()
            .init_resource::<Connected>()
            .init_resource::<LastFailure>()
            .add_systems(Startup, read_config)
            .add_observer(route)
            .add_observer(remember)
            .add_observer(fall_back)
            .add_observer(fall_back_after_retries);
    }
}

/// `RIG_HOME/fallback.json`.
#[derive(Deserialize)]
struct Config {
    chain: Vec<String>,
    cooldown: Option<String>,
}

/// Reads the chain of an agent the launcher started, so a test never reads
/// the user's home.
fn read_config(invoked: Option<Res<Invoked>>, mut chain: ResMut<FallbackChain>) {
    if invoked.is_none() || !chain.chain.is_empty() {
        return;
    }
    let path = Home::from_env().root().join("fallback.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(config) = serde_json::from_str::<Config>(&text) else {
        return;
    };
    chain.chain = config.chain;
    if let Some(cooldown) = config.cooldown.as_deref().and_then(parse_duration) {
        chain.cooldown = cooldown;
    }
}

/// `90`, `90s`, `5m` or `1h`.
fn parse_duration(text: &str) -> Option<Duration> {
    let text = text.trim();
    let split = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let number: u64 = number.parse().ok()?;
    let seconds = match unit.trim() {
        "" | "s" => 1,
        "m" => 60,
        "h" => 3600,
        _ => return None,
    };
    Some(Duration::from_secs(number.checked_mul(seconds)?))
}

/// The chain as it stands now.
#[derive(SystemParam)]
struct Chain<'w> {
    config: Res<'w, FallbackChain>,
    cooling: ResMut<'w, Cooling>,
    connected: ResMut<'w, Connected>,
    models: Res<'w, Models>,
    time: Res<'w, Time<Real>>,
}

impl Chain<'_> {
    fn is_cooling(&self, reference: &str) -> bool {
        let now = self.time.elapsed();
        self.cooling
            .0
            .get(reference)
            .is_some_and(|until| now < *until)
    }

    /// Skips the model `reference` for the cooldown.
    fn cool(&mut self, reference: &str) {
        let until = self.time.elapsed() + self.config.cooldown;
        self.cooling.0.insert(reference.to_owned(), until);
    }

    /// The first model of the chain that is not cooling down and connects.
    fn next(&mut self) -> Option<Connection> {
        let catalog = self.models.0.catalog();
        for entry in &self.config.chain {
            let reference = catalog
                .resolve(entry)
                .map_or_else(|_| entry.clone(), |found| found.spec.reference());
            if self.is_cooling(&reference) {
                continue;
            }
            if let Some(connection) = self.connected.0.get(&reference) {
                return Some(connection.clone());
            }
            if let Ok(connection) = self.models.connect(&reference) {
                self.connected.0.insert(reference, connection.clone());
                return Some(connection);
            }
        }
        None
    }
}

/// Whether a turn falls back after `report`: a rate limit, an overload, a
/// server error or a refused key; not a request too long for the model.
fn falls_back(report: &ErrorReport) -> bool {
    !report.is_context_overflow()
        && report.kind != ErrorKind::Cancelled
        && (report.is_retryable() || matches!(report.http_status, Some(401 | 403)))
}

/// "opus overloaded; continuing on gpt-x".
fn switched(failed: &str, report: &ErrorReport, next: &Connection) -> String {
    let short = |reference: &str| {
        let name = reference.rsplit('/').next().unwrap_or(reference);
        name.to_owned()
    };
    let why = match report.http_status {
        Some(429) => "rate-limited",
        Some(401 | 403) => "refused the credentials",
        Some(503 | 529) => "overloaded",
        _ if report.message.to_ascii_lowercase().contains("overloaded") => "overloaded",
        _ => "failing",
    };
    let next = short(&next.spec.reference());
    format!("{} {why}; continuing on {next}", short(failed))
}

/// A new turn of an agent whose model cools down goes to the chain's first
/// model that does not, unless the turn has a model of its own.
fn route(
    added: On<Add<TurnOf>>,
    turns: Query<(&TurnOf, Has<Connection>)>,
    agents: Query<&Connection, With<Agent>>,
    mut chain: Chain,
    mut commands: Commands,
) {
    let turn = added.entity;
    let Ok((&TurnOf(agent), routed)) = turns.get(turn) else {
        return;
    };
    let Ok(own) = agents.get(agent) else {
        return;
    };
    if routed || !chain.is_cooling(&own.spec.reference()) {
        return;
    }
    if let Some(next) = chain.next() {
        commands.entity(turn).insert(next);
    }
}

/// Remembers each turn's last failed model call and its model, for the
/// failures the kernel ends a turn for without [`ModelFailed`].
fn remember(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Done<ModelReply>), With<ModelCall>>,
    turns: Query<(&TurnOf, Option<&Connection>)>,
    agents: Query<&Connection, With<Agent>>,
    mut last: ResMut<LastFailure>,
) {
    let Ok((&CallOf(turn), Done(reply))) = calls.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), routed)) = turns.get(turn) else {
        return;
    };
    match reply {
        Ok(_) => {
            last.0.remove(&agent);
        }
        Err(report) => {
            if let Some(model) = routed.or(agents.get(agent).ok()) {
                last.0
                    .insert(agent, (model.spec.reference(), report.clone()));
            }
        }
    }
}

/// A call that failed in a way a retry does not fix, such as a refused key,
/// is sent again at once on the next model of the chain, in the same turn.
fn fall_back(
    mut failed: On<ModelFailed>,
    turns: Query<Option<&Connection>, With<TurnOf>>,
    agents: Query<&Connection, With<Agent>>,
    mut chain: Chain,
    mut last: ResMut<LastFailure>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (turn, agent) = (failed.entity, failed.agent);
    if failed.handled || !falls_back(&failed.report) {
        return;
    }
    let routed = turns.get(turn).ok().flatten();
    let Some(model) = routed.or(agents.get(agent).ok()) else {
        return;
    };
    let model = model.spec.reference();
    chain.cool(&model);
    let Some(next) = chain.next() else {
        return;
    };
    last.0.remove(&agent);
    failed.handled = true;
    notices.write(Notice::info(agent, switched(&model, &failed.report, &next)));
    commands.entity(turn).insert(next);
    commands.trigger(CallModel { entity: turn });
}

/// A call that failed as long as the retries lasted (a rate limit, an
/// overload, a server error) ends the turn without [`ModelFailed`]; the
/// user's message is kept, so the agent retries it, and [`route`] sends
/// the new turn to the next model of the chain.
fn fall_back_after_retries(
    ended: On<TurnEnded>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    mut chain: Chain,
    mut last: ResMut<LastFailure>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = ended.entity;
    let failure = last.0.remove(&agent);
    let (TurnOutcome::Failed(_), Some((model, report))) = (&ended.outcome, failure) else {
        return;
    };
    // A failure a retry does not fix went to `ModelFailed`.
    if !report.is_retryable() || !falls_back(&report) || agents.get(agent).unwrap_or(true) {
        return;
    }
    chain.cool(&model);
    let Some(next) = chain.next() else {
        return;
    };
    notices.write(Notice::info(agent, switched(&model, &report, &next)));
    commands.trigger(Retry { entity: agent });
}
