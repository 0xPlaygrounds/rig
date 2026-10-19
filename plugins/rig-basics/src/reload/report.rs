//! A model-asked reload reports back ([`ReloadAsked`]): after the restart,
//! a failed build or a rollback, the agent whose model asked gets one
//! message that starts a turn, and a run nobody sits at waits for the
//! reload instead of ending ([`KeepRunning`]).

use std::mem;

use rig_harness::harness_protocol::env;
use rig_harness::prelude::*;

use crate::print::KeepRunning;
use crate::reload::{ORIGIN, ReloadAsked, ReloadStatus};

pub(crate) fn add(app: &mut App) {
    app.keeps_saved::<ReloadAsked>()
        .add_observer(reset_on_user_message)
        .add_observer(clear_on_stop)
        .add_systems(
            Update,
            (
                report_after_restart.run_if(resource_exists::<SessionRestored>),
                hold_headless_run,
            ),
        );
}

/// Sends `agent`'s model `text` as one message that starts a turn, if the
/// agent is owed the result of a reload it asked for, and marks it paid in
/// the same frame, so the message arrives once. Whether it was owed.
pub(crate) fn report(
    commands: &mut Commands,
    asked: &mut Query<&mut ReloadAsked>,
    agent: Entity,
    text: String,
) -> bool {
    let Ok(mut asked) = asked.get_mut(agent) else {
        return false;
    };
    if !mem::take(&mut asked.owed) {
        return false;
    }
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    commands.trigger(message.with_author(Author::plugin(ORIGIN)));
    true
}

/// Clears what every agent is owed, as when the user cancels the reload.
pub(crate) fn clear(asked: &mut Query<&mut ReloadAsked>) {
    for mut asked in asked.iter_mut().filter(|asked| asked.owed) {
        asked.owed = false;
    }
}

/// Keeps a run nobody sits at going until the reload is done: it restarts
/// on the new build, or a failed build's message carries the agent on.
pub(crate) fn keep_run(commands: &mut Commands) {
    commands.spawn((Reloading, KeepRunning, Name::new("reload")));
}

/// The [`KeepRunning`] a reload holds a run nobody sits at with.
#[derive(Component)]
struct Reloading;

/// Holds a run nobody sits at while a reload is queued, builds or waits to
/// restart, and lets it go once none does.
fn hold_headless_run(
    status: Res<ReloadStatus>,
    held: Query<Entity, With<Reloading>>,
    mut commands: Commands,
) {
    match *status {
        ReloadStatus::Queued { .. } | ReloadStatus::Building { .. } | ReloadStatus::Ready => {
            if held.is_empty() {
                keep_run(&mut commands);
            }
        }
        ReloadStatus::Idle | ReloadStatus::Failed => {
            for entity in &held {
                commands.entity(entity).despawn();
            }
        }
    }
}

/// Once the session is restored, tells each agent owed a reload's result
/// how it went: after the restart on the new build, that it started;
/// after the launcher rolled the new build back, why. Any other start, such
/// as a resume, drops what was owed: nobody knows how that reload went.
fn report_after_restart(
    mut done: Local<bool>,
    start: Option<Res<AppStart>>,
    mut asked: Query<(Entity, &mut ReloadAsked)>,
    mut commands: Commands,
) {
    if mem::replace(&mut *done, true) {
        return;
    }
    let reason = start.map(|start| start.0);
    for (agent, mut asked) in asked.iter_mut().filter(|(_, asked)| asked.owed) {
        asked.owed = false;
        let text = match reason {
            Some(StartReason::Reloaded) => started(&asked.then),
            Some(StartReason::Restarted) => rolled_back(),
            _ => continue,
        };
        let message = Deliver::new(agent, text, DeliveryMode::Queue);
        commands.trigger(message.with_author(Author::plugin(ORIGIN)));
    }
}

/// The message after the restart on the new build.
fn started(then: &str) -> String {
    let mut text = "The reload you asked for is done: the new build started, with your \
                    plugin changes."
        .to_owned();
    if !then.trim().is_empty() {
        text.push_str(&format!(" You said you would then: {}", then.trim()));
    }
    text
}

/// The message after the launcher rolled the new build back.
fn rolled_back() -> String {
    let notice = std::env::var(env::NOTICE).unwrap_or_default();
    format!(
        "The reload you asked for did not take: the new build stopped during startup, so the \
         previous build runs, without your latest plugin changes. {}",
        notice.trim()
    )
    .trim_end()
    .to_owned()
}

/// A message the user sent an agent starts its count of reloads again.
fn reset_on_user_message(
    accepted: On<Add<Started>>,
    messages: Query<(&Incoming, &ForAgent)>,
    mut asked: Query<&mut ReloadAsked>,
) {
    let Ok((message, &ForAgent(agent))) = messages.get(accepted.entity) else {
        return;
    };
    if message.author.kind == AuthorKind::User
        && let Ok(mut asked) = asked.get_mut(agent)
        && asked.attempts > 0
    {
        asked.attempts = 0;
    }
}

/// The user stopping an agent, or an agent above it, clears what it is
/// owed.
fn clear_on_stop(stop: On<StopTurn>, spawned: Query<&Spawned>, mut asked: Query<&mut ReloadAsked>) {
    if stop.reason.is_some() {
        return;
    }
    let below = spawned.iter_descendants::<Spawned>(stop.entity);
    for agent in std::iter::once(stop.entity).chain(below) {
        if let Ok(mut asked) = asked.get_mut(agent)
            && asked.owed
        {
            asked.owed = false;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
