//! Why the app exits. A plugin that ends the run triggers [`Exit`] with
//! its reason: the harness keeps the first reason as the [`ExitReason`]
//! resource at once, tells the launcher what to run next (another session,
//! or the staged build), and spawns the exit as an app-level effect, an
//! [`ExitRequest`]. The app exits when that effect settles, whatever
//! answered it: a plugin delays the exit by holding it (`HoldOf`), and
//! nothing cancels it. An exit written as a plain `AppExit`, such as a
//! window closing, gets its reason from its code and cannot be held. Code
//! that runs on the way out reads the reason in Bevy's
//! `OnAppExitSystems`, such as a plugin that leaves its background jobs
//! running across a reload and stops them on a quit.

use std::fs;

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::effect::{Decider, EffectKind};
use rig_ecs::agent::Notice;
use rig_ecs::stage::{Decision, EffectRequest, Outcome, Settled, Started};
use rig_ecs::tasks::Wake;
use rig_ecs::turn::Exiting;
use rig_harness_protocol::{RELOAD_EXIT_CODE, SIGNAL_EXIT_CODE, SessionId};

use super::session::SessionPaths;

/// Why the app exits; a resource from the frame the exit was asked for.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// The user quit.
    Quit,
    /// The launcher restarts this session on its staged build (`/reload`).
    Reload,
    /// The launcher runs another session next: the one named, else a new
    /// one.
    Switch(Option<SessionId>),
    /// A signal (SIGINT, SIGTERM, SIGHUP) ended the run, or the launcher
    /// that started it went away.
    Signal,
    /// The run failed with this exit code.
    Failed(u8),
}

impl ExitReason {
    /// The reason an `AppExit` written without one stands for.
    pub fn of(exit: &AppExit) -> Self {
        match exit {
            AppExit::Success => Self::Quit,
            AppExit::Error(code) if code.get() == RELOAD_EXIT_CODE => Self::Reload,
            AppExit::Error(code) if code.get() == SIGNAL_EXIT_CODE => Self::Signal,
            AppExit::Error(code) => Self::Failed(code.get()),
        }
    }

    /// Why this exit stops what still runs, as the turns it stops and their
    /// calls say it: a signal interrupted the run, anything else ended it.
    pub fn stops_because(&self) -> &'static str {
        match self {
            Self::Signal => "the run was interrupted",
            _ => "the run ended",
        }
    }

    /// The exit code the launcher reads.
    pub fn exit(&self) -> AppExit {
        match self {
            Self::Quit => AppExit::Success,
            Self::Reload | Self::Switch(_) => AppExit::from_code(RELOAD_EXIT_CODE),
            Self::Signal => AppExit::from_code(SIGNAL_EXIT_CODE),
            Self::Failed(code) => AppExit::from_code(*code),
        }
    }
}

/// Asks the app to exit for a reason. The first reason of a run stands;
/// each exit asked for is an [`ExitRequest`] effect. A
/// [`ExitReason::Switch`] needs the launcher: without one, or when the
/// session to run cannot be named to it, the app keeps running and an
/// error notice says why.
#[derive(Event, Clone, Debug)]
pub struct Exit(pub ExitReason);

/// An exit asked for, on an app-level effect (no `ForAgent`) of the kind
/// `"exit"`. The app exits when this effect settles, however it was
/// answered: a plugin delays the exit only by holding it (`HoldOf`, with a
/// deadline of its own), and nothing cancels it. Work keeps running while
/// it is held. A second exit asked for meanwhile is a second effect, which
/// a holder may take as "stop waiting".
#[derive(Component, Clone, Debug)]
pub struct ExitRequest(pub ExitReason);

/// Handles [`Exit`] and gives a plain `AppExit` its [`ExitReason`].
pub struct ExitPlugin;

impl Plugin for ExitPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_exit)
            .add_observer(answer_started)
            .add_observer(exit_settled)
            .add_systems(
                Last,
                (reason_from_exit, name_exit_stop)
                    .chain()
                    .before(OnAppExitSystems)
                    .distributive_run_if(on_message::<AppExit>),
            );
    }
}

fn on_exit(exit: On<Exit>, mut commands: Commands) {
    let reason = exit.0.clone();
    // Checked as the command is applied, in order with every other exit
    // asked for: a second one triggered before the first one's commands
    // were applied sees the first reason.
    commands.queue(move |world: &mut World| {
        if !world.contains_resource::<ExitReason>() {
            if let ExitReason::Switch(target) = &reason
                && let Err(why) = name_switch(world, target.as_ref())
            {
                world.write_message(Notice::error(
                    None,
                    format!("Could not switch sessions: {why}."),
                ));
                return;
            }
            world.insert_resource(reason.clone());
        }
        let kind = EffectKind::Custom {
            kind: "exit".into(),
            payload: format!("{reason:?}").into(),
        };
        world.spawn((
            Name::new("exit"),
            ExitRequest(reason),
            EffectRequest {
                kind,
                handler: None,
            },
        ));
        if let Some(wake) = world.get_resource::<Wake>() {
            wake.wake();
        }
    });
}

/// Tells the launcher which session to run next.
fn name_switch(world: &World, target: Option<&SessionId>) -> Result<(), String> {
    match (
        world.get_resource::<SessionPaths>(),
        crate::harness_protocol::launcher(),
    ) {
        (Some(paths), Some(_)) => {
            let target = target.map(SessionId::as_str).unwrap_or_default();
            fs::write(paths.switch(), target).map_err(|failure| failure.to_string())
        }
        _ => {
            Err("switching sessions needs the rig launcher: start the agent with `rig`".to_owned())
        }
    }
}

/// Answers an exit effect once it starts: nothing held it back.
fn answer_started(started: On<Add<Started>>, mut exits: Query<&mut Decision, With<ExitRequest>>) {
    if let Ok(mut decision) = exits.get_mut(started.entity) {
        decision.decide(Decider::plugin("rig-harness"), Outcome::custom("exited"));
    }
}

/// Exits once the first exit effect settled, with the run's first reason.
fn exit_settled(
    settled: On<Add<Settled>>,
    requests: Query<&ExitRequest>,
    reason: Option<Res<ExitReason>>,
    mut exits: MessageWriter<AppExit>,
    mut exited: Local<bool>,
) {
    if let Ok(request) = requests.get(settled.entity)
        && !*exited
    {
        *exited = true;
        exits.write(reason.as_deref().unwrap_or(&request.0).exit());
    }
}

/// Gives an `AppExit` that came without [`Exit`] its reason.
pub(super) fn reason_from_exit(
    mut exits: MessageReader<AppExit>,
    reason: Option<Res<ExitReason>>,
    mut commands: Commands,
) {
    let first = exits.read().next();
    if let Some(exit) = first.filter(|_| reason.is_none()) {
        commands.insert_resource(ExitReason::of(exit));
    }
}

/// Tells the kernel why the exit stops the running turns
/// ([`ExitReason::stops_because`]).
fn name_exit_stop(reason: Option<Res<ExitReason>>, mut commands: Commands) {
    let why = reason.map_or(ExitReason::Quit.stops_because(), |reason| {
        reason.stops_because()
    });
    commands.insert_resource(Exiting(why.to_owned()));
}

#[cfg(test)]
mod tests;
