//! Plugin corpus: 15. Model fallback chain (spec `15-model-fallback.md`).
//!
//! When a turn's model call fails for good with a rate limit, an overload,
//! a server error or a refused key, the same request goes on, in the same
//! turn, to the next model of the chain that is not cooling down, and the
//! model that failed is skipped for the cooldown; after it, new turns go
//! back to it. A Review system reads the kernel's [`RetryVerdict`] (the
//! retries gave up, or a retry would not help) and supersedes the call; a
//! Prepare system sends a call bound for a cooling model to the chain
//! instead. A context overflow is left to compaction, and a failure another
//! plugin holds or superseded is left to it. Both systems are in the public
//! [`FallbackSystems`] sets, in the stages' phases: a call is routed in
//! [`PrepareSystems::Edit`], before compaction measures it, and a failure
//! is taken over in [`ReviewSystems::Act`], once reviewers edited its
//! verdict. The `[model-fallback]` table of `settings.toml`:
//!
//! ```toml
//! [model-fallback]
//! chain = ["anthropic/claude-opus-4-5", "openai/gpt-5"]
//! cooldown = "5m"
//! ```

use std::collections::HashMap;

use crate::model_router::RouteSystems;
use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::error::ErrorReport;
use serde::{Deserialize, Deserializer};

#[cfg(test)]
mod tests;

/// The `[model-fallback]` settings: the models a turn falls back to, in
/// order, as catalog references, and how long a model that failed is
/// skipped (`90s`, `5m`, `1h`; 5 minutes unless set).
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct FallbackChain {
    pub chain: Vec<String>,
    #[serde(deserialize_with = "duration")]
    pub cooldown: Duration,
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self {
            chain: Vec::new(),
            cooldown: Duration::from_secs(5 * 60),
        }
    }
}

/// The models that failed, by catalog reference, and until when they are
/// skipped, on Bevy's real clock.
#[derive(Resource, Default)]
struct Cooling(HashMap<String, Duration>);

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackSystems {
    /// Prepare: sends a turn's call bound for a cooling model to the chain,
    /// after model-router chose it.
    Route,
    /// Review: takes over a turn's call that failed for good.
    TakeOver,
}

#[derive(Default)]
pub struct ModelFallbackPlugin;

impl Plugin for ModelFallbackPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FallbackChain>("model-fallback")
            .init_resource::<Cooling>()
            .configure_sets(
                AgentPass,
                (
                    FallbackSystems::Route
                        .in_set(PrepareSystems::Edit)
                        .after(RouteSystems),
                    FallbackSystems::TakeOver.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    route.in_set(FallbackSystems::Route),
                    fall_back.in_set(FallbackSystems::TakeOver),
                ),
            );
    }
}

/// `90`, `90s`, `5m` or `1h`.
fn duration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    let text = String::deserialize(deserializer)?;
    let digits = text.trim_end_matches(|c: char| !c.is_ascii_digit());
    let unit = match text.get(digits.len()..).unwrap_or_default().trim() {
        "" | "s" => 1,
        "m" => 60,
        "h" => 3600,
        unit => return Err(serde::de::Error::custom(format!("unknown unit {unit:?}"))),
    };
    let number: u64 = digits.trim().parse().map_err(serde::de::Error::custom)?;
    Ok(Duration::from_secs(number.saturating_mul(unit)))
}

/// The chain as it stands now.
#[derive(SystemParam)]
struct Chain<'w> {
    config: Res<'w, FallbackChain>,
    cooling: ResMut<'w, Cooling>,
    models: Res<'w, Models>,
    time: Res<'w, Time<Real>>,
}

impl Chain<'_> {
    /// Whether the model `reference` is skipped now.
    fn is_cooling(&self, reference: &str) -> bool {
        let until = self.cooling.0.get(reference);
        until.is_some_and(|until| self.time.elapsed() < *until)
    }

    /// Skips `model` for the cooldown.
    fn cool(&mut self, model: &Connection) {
        let until = self.time.elapsed() + self.config.cooldown;
        self.cooling.0.insert(model.spec.reference(), until);
    }

    /// The first model of the chain that is not cooling down and connects.
    fn next(&self) -> Option<Connection> {
        let catalog = self.models.0.catalog();
        let warm = |model: &&String| {
            let found = catalog.resolve(model);
            found.is_ok_and(|found| !self.is_cooling(&found.spec.reference()))
        };
        let mut chain = self.config.chain.iter().filter(warm);
        chain.find_map(|model| self.models.connect(model).ok())
    }
}

/// Whether a turn falls back after `report`: a rate limit, an overload, a
/// server error or a refused key; not a request too long for the model.
fn falls_back(report: &ErrorReport) -> bool {
    !report.is_context_overflow()
        && (report.is_retryable() || matches!(report.http_status, Some(401 | 403)))
}

/// "claude-opus-4-5 overloaded; continuing on gpt-5".
fn switched(failed: &Connection, report: &ErrorReport, next: &Connection) -> String {
    let why = match report.http_status {
        Some(429) => "rate-limited",
        Some(401 | 403) => "refused the credentials",
        Some(503 | 529) => "overloaded",
        _ => "failing",
    };
    format!("{} {why}; continuing on {}", failed.spec.id, next.spec.id)
}

/// A turn's call bound for a model that cools down, a new turn's or a
/// retry's the kernel built from its agent, goes to the chain's first
/// model that does not.
fn route(
    mut calls: Query<&mut ModelRequest, (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    chain: Chain,
) {
    for mut call in &mut calls {
        if call
            .connection
            .as_ref()
            .is_some_and(|to| chain.is_cooling(&to.spec.reference()))
            && let Some(next) = chain.next()
        {
            call.connection = Some(next);
        }
    }
}

/// A turn's call that failed for good, its retries spent or not worth
/// trying, is sent again at once on the next model of the chain, which
/// gets the usual retries of its own: a new call with the same request
/// supersedes it in the same turn.
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
        (
            ReadyToSettle,
            ModelCalls,
            With<TurnCall>,
            Without<SupersededBy>,
        ),
    >,
    mut turns: Query<&mut Recovery>,
    mut chain: Chain,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, verdict, outcome, request, effect, &CallOf(turn), &ForAgent(agent)) in &failed {
        let (Err(report), Some(model)) = (&outcome.0, &request.connection) else {
            continue;
        };
        if !matches!(verdict.0, Verdict::GaveUp(_) | Verdict::Final) || !falls_back(report) {
            continue;
        }
        chain.cool(model);
        let Some(next) = chain.next() else {
            continue;
        };
        notices.write(Notice::info(agent, switched(model, report, &next)));
        if let Ok(mut recovery) = turns.get_mut(turn) {
            recovery.retries = 0;
        }
        commands
            .spawn((
                request.clone().on(next),
                TurnCall,
                CallOf(turn),
                ForAgent(agent),
                Effect::with_parent_id(effect.parent),
            ))
            .add_related::<SupersededBy>(&[call]);
    }
}
