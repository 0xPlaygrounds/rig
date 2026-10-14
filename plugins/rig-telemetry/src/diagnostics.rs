//! Warnings and errors the process logs, Bevy's own included, as data (the
//! [`Diagnostics`] resource, which the optional `rig-inspect` plugin's
//! `inspect` tool reads), not as notes in turns. Each names the
//! plugin whose module logged it when exactly one plugin's module holds
//! that module, so Bevy's own name none. They follow a rewrite of the
//! session, as the text log does.

use std::collections::VecDeque;

use rig_harness::prelude::*;
use rig_harness::rig_memory::{Place, rewrite_value};

/// How many warnings and errors [`Diagnostics`] keeps.
pub const KEPT: usize = 100;

/// Keeps the warnings and errors the log passes on in [`Diagnostics`].
#[derive(Default)]
pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Diagnostics>()
            .add_message::<SessionRewritten>()
            .add_systems(Last, collect.run_if(resource_exists::<LoggedQueue>))
            .add_systems(
                Update,
                follow_rewrites.run_if(on_message::<SessionRewritten>),
            );
    }
}

/// The newest warnings and errors this process logged, oldest first.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource, Debug)]
pub struct Diagnostics(pub VecDeque<Logged>);

fn collect(
    events: Res<LoggedQueue>,
    plugins: Query<&Name, With<PluginSource>>,
    mut diagnostics: ResMut<Diagnostics>,
) {
    for mut logged in events.0.try_iter() {
        let target = format!("{}::", logged.target);
        let mut owners = plugins.iter().filter(|name| {
            let module = name.rsplit_once("::").map_or("", |(module, _)| module);
            target.starts_with(&format!("{module}::"))
        });
        logged.plugin = owners
            .next()
            .filter(|_| owners.next().is_none())
            .map(Name::to_string);
        if diagnostics.0.len() == KEPT {
            diagnostics.0.pop_front();
        }
        diagnostics.0.push_back(logged);
    }
}

/// Maps the kept warnings and errors through each rewrite of the session,
/// as the text log they were written to was mapped.
fn follow_rewrites(
    mut rewritten: MessageReader<SessionRewritten>,
    mut diagnostics: ResMut<Diagnostics>,
) {
    let at = Place::File { name: "agent.log" };
    for done in rewritten.read().filter(|done| done.result.is_ok()) {
        for logged in &mut diagnostics.0 {
            let message = &mut logged.message;
            if rewrite_value(&*done.with, &at, "agent.log", message).is_err() {
                message.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests;
