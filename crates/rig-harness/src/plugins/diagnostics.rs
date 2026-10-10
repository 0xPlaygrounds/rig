//! Warnings and errors the process logs, Bevy's own included, as data the
//! agent can read with `inspect`, not as notes in its turns. Each names the
//! plugin whose module logged it when exactly one plugin's module holds
//! that module, so Bevy's own name none.

use std::collections::VecDeque;

use crate::prelude::*;

/// How many warnings and errors [`Diagnostics`] keeps.
pub const KEPT: usize = 100;

/// Keeps the warnings and errors the log passes on in [`Diagnostics`].
#[derive(Default)]
pub struct DiagnosticsPlugin;

impl Plugin for DiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Diagnostics>()
            .add_systems(Last, collect.run_if(resource_exists::<LogEvents>));
    }
}

/// The newest warnings and errors this process logged, oldest first.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource, Debug)]
pub struct Diagnostics(pub VecDeque<Logged>);

fn collect(
    events: Res<LogEvents>,
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

#[cfg(test)]
mod tests;
