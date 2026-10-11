//! Why the app exits. A plugin that ends the run triggers [`Exit`] with
//! its reason: the harness keeps the reason as the [`ExitReason`]
//! resource, tells the launcher what to run next (another session, or the
//! staged build), and writes the `AppExit` that ends the run. An exit
//! written as a plain `AppExit`, such as a window closing, gets its
//! reason from its code. Code that runs on the way out reads the reason in
//! Bevy's `OnAppExitSystems`, such as a plugin that leaves its background
//! jobs running across a reload and stops them on a quit.

use std::fs;

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{RELOAD_EXIT_CODE, SessionId};
use rig_ecs::agent::Notice;

use super::launcher;
use super::session::SessionPaths;

/// The exit code of a signalled exit, as a shell reports Ctrl+C.
pub const SIGNAL_EXIT_CODE: u8 = 130;

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
    /// A signal (SIGINT, SIGTERM, SIGHUP) ended the run.
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

/// Asks the app to exit for a reason. The first reason of a run stands.
/// A [`ExitReason::Switch`] needs the launcher: without one, or when the
/// session to run cannot be named to it, the app keeps running and an
/// error notice says why.
#[derive(Event, Clone, Debug)]
pub struct Exit(pub ExitReason);

/// Handles [`Exit`] and gives a plain `AppExit` its [`ExitReason`].
pub struct ExitPlugin;

impl Plugin for ExitPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(on_exit).add_systems(
            Last,
            reason_from_exit
                .before(OnAppExitSystems)
                .run_if(on_message::<AppExit>),
        );
    }
}

fn on_exit(exit: On<Exit>, mut commands: Commands) {
    let reason = exit.0.clone();
    // In order with every other exit asked for before it.
    commands.queue(move |world: &mut World| {
        if world.contains_resource::<ExitReason>() {
            return;
        }
        if let ExitReason::Switch(target) = &reason {
            let paths = world.get_resource::<SessionPaths>();
            let named = match (paths, launcher::executable()) {
                (Some(paths), Some(_)) => {
                    let target = target.as_ref().map(SessionId::as_str).unwrap_or_default();
                    fs::write(paths.switch(), target).map_err(|failure| failure.to_string())
                }
                _ => Err(
                    "switching sessions needs the rig launcher: start the agent with `rig`"
                        .to_owned(),
                ),
            };
            if let Err(why) = named {
                world.write_message(Notice::error(
                    None,
                    format!("Could not switch sessions: {why}."),
                ));
                return;
            }
        }
        world.write_message(reason.exit());
        world.insert_resource(reason);
    });
}

/// Gives an `AppExit` that came without [`Exit`] its reason.
fn reason_from_exit(
    mut exits: MessageReader<AppExit>,
    reason: Option<Res<ExitReason>>,
    mut commands: Commands,
) {
    let first = exits.read().next();
    if let Some(exit) = first.filter(|_| reason.is_none()) {
        commands.insert_resource(ExitReason::of(exit));
    }
}

#[cfg(test)]
mod tests;
