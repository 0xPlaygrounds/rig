//! Plugin corpus: 15. Model fallback chain (spec `15-model-fallback.md`).
//!
//! When a model call fails for good with a rate limit, an overload, a
//! server error or a refused key, the turn goes on with the next model of
//! the chain that is not cooling down, and the model that failed is skipped
//! for the cooldown; after it, new turns go back to it. A Review system
//! reads the kernel's [`RetryVerdict`] (the retries gave up, or a retry
//! would not help) and supersedes the call with the same request on the
//! next model; a Prepare system sends a call bound for a cooling model to
//! the chain instead. A context overflow is left to compaction, and a
//! failure another plugin took over is left to it. The chain is the
//! `[model-fallback]` table of the user's `settings.toml`:
//!
//! ```toml
//! [model-fallback]
//! chain = ["anthropic/claude-opus-4-5", "openai/gpt-5"]
//! cooldown = "5m"
//! ```

use std::collections::HashMap;

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
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

#[derive(Default)]
pub struct ModelFallbackPlugin;

impl Plugin for ModelFallbackPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<FallbackChain>() {
            app.add_setting::<FallbackSettings>("model-fallback");
            let settings = app.world().resource::<FallbackSettings>();
            let chain = FallbackChain {
                chain: settings.chain.clone(),
                cooldown: settings
                    .cooldown
                    .as_deref()
                    .and_then(parse_duration)
                    .unwrap_or(COOLDOWN),
            };
            app.insert_resource(chain);
        }
        app.init_resource::<Cooling>()
            .init_resource::<Connected>()
            .add_systems(
                AgentPass,
                (
                    route.in_set(Stage::Prepare),
                    fall_back.in_set(Stage::Review),
                ),
            );
    }
}

/// The `[model-fallback]` table of `settings.toml`.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FallbackSettings {
    chain: Vec<String>,
    cooldown: Option<String>,
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

/// A turn's call bound for a model that cools down goes to the chain's
/// first model that does not.
fn route(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    agents: Query<&Connection, With<Agent>>,
    mut chain: Chain,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let bound = call.connection.as_ref().or(agents.get(agent).ok());
        let Some(bound) = bound.map(|connection| connection.spec.reference()) else {
            continue;
        };
        if chain.is_cooling(&bound)
            && let Some(next) = chain.next()
        {
            call.connection = Some(next);
        }
    }
}

/// A turn's call that failed for good, its retries spent or not worth
/// trying, is sent again at once on the next model of the chain, in the
/// same turn: a new call with the same request supersedes it.
fn fall_back(
    failed: Query<
        (
            Entity,
            &RetryVerdict,
            &Outcome,
            &ModelRequest,
            &Effect,
            &CallOf,
            &ForAgent,
        ),
        (Unreviewed, With<TurnCall>, Without<SupersededBy>),
    >,
    agents: Query<&Connection, With<Agent>>,
    mut chain: Chain,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, verdict, outcome, request, effect, of, &ForAgent(agent)) in &failed {
        let Err(report) = &outcome.0 else {
            continue;
        };
        if !matches!(verdict.0, Verdict::GaveUp(_) | Verdict::Final) || !falls_back(report) {
            continue;
        }
        let Some(model) = request.connection.as_ref().or(agents.get(agent).ok()) else {
            continue;
        };
        let model = model.spec.reference();
        chain.cool(&model);
        let Some(next) = chain.next() else {
            continue;
        };
        notices.write(Notice::info(agent, switched(&model, report, &next)));
        commands.spawn((
            request.clone().on(next),
            TurnCall,
            Supersedes(call),
            CallOf(of.0),
            ForAgent(agent),
            Effect::child_of(effect.parent),
        ));
    }
}
