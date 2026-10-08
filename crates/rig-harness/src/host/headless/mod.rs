//! The mode without a terminal view, chosen by the process's arguments
//! ([`rig::harness_protocol::Invocation`], which the `rig` launcher passes
//! on): `--print` answers one prompt. It is a view like the terminal one:
//! it reads agent components and messages and sends the agents the same
//! requests, and it never owns the loop.
//!
//! In this mode stdout carries only the answer; notices go to stderr, and
//! the log stays in the session's `agent.log`.

mod print;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::Invocation;

/// How this process runs: its [`Invocation`], read from its arguments
/// unless the app inserted one before adding [`ModePlugin`]. Views check
/// it: the terminal view stays out of the headless mode.
#[derive(Resource, Clone, Debug, Default)]
pub struct RunMode(pub Invocation);

impl RunMode {
    /// Whether nobody sits at a terminal view.
    pub fn is_headless(&self) -> bool {
        self.0.is_headless()
    }
}

/// Reads the [`RunMode`] and adds what its mode needs: the print loop.
pub struct ModePlugin;

impl Plugin for ModePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<RunMode>() {
            let invocation = Invocation::from_env().unwrap_or_else(|failure| {
                // The launcher checked them; the binary run alone says why
                // it runs as usual.
                eprintln!("rig: {failure}; starting the terminal view");
                Invocation::default()
            });
            app.insert_resource(RunMode(invocation));
        }
        let print = app
            .world()
            .get_resource::<RunMode>()
            .and_then(|mode| mode.0.print.clone());
        if let Some(prompt) = print {
            app.add_plugins(print::PrintPlugin { prompt });
        }
    }
}
