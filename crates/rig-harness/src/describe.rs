//! How the built agent tells the `rig` launcher what its plugins add to
//! the command line: their commands, their options and the secrets they
//! ask for ([`Described`]). The launcher asks after each build, by running
//! the binary with [`env::DESCRIBE`] naming a file; [`run`], which the
//! generated `main.rs` ends with, then builds the app (every plugin's
//! `build`, `finish` and `cleanup`), writes the report to that file and
//! exits without running a frame. While it describes, no front takes the
//! run, so no front sets itself up, and the session has no directory.
//!
//! The report is checked as it is made: two plugins adding one command,
//! or a command named as one of the launcher's own (`build`, `plugin`,
//! `help`), fail the build, naming the plugins. A plugin's `build` runs on
//! every describe, so it declares what it adds there and does its work in
//! `finish` or in systems.

use std::path::PathBuf;

use bevy_app::PluginsState;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{CommandInfo, Described, FlagInfo, SecretInfo, env};

use crate::host::cli::{Front, command_refusal, declared, flags};
use crate::load::{PluginSource, ProvidedBy};
use crate::secrets::Secret;

/// The build describes itself: while this is a resource, no front takes
/// the run.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct Describing;

/// The file [`env::DESCRIBE`] names, if the launcher asks for a report.
pub(crate) fn describing() -> Option<PathBuf> {
    std::env::var_os(env::DESCRIBE)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

/// Runs the app, or, when the launcher asks ([`env::DESCRIBE`]), builds it,
/// writes its [`report`] to the file the launcher named and exits without
/// running a frame: with code 2 and the reason on stderr when the report
/// fails. The generated `main.rs` ends with it in place of `app.run()`.
pub fn run(mut app: App) -> AppExit {
    let Some(path) = describing() else {
        return app.run();
    };
    app.insert_resource(Describing);
    // As Bevy's own runners do before their first frame
    // (`bevy_app/src/schedule_runner.rs`).
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy_tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }
    match report(app.world_mut()) {
        Ok(described) => match std::fs::write(&path, described.write()) {
            Ok(()) => AppExit::Success,
            Err(failure) => {
                eprintln!("rig: could not write {}: {failure}", path.display());
                AppExit::from_code(1)
            }
        },
        Err(why) => {
            eprintln!("rig: {why}");
            AppExit::from_code(2)
        }
    }
}

/// What the plugins of the built app `world` add to the command line, or
/// why the build cannot be run: two plugins adding one command, or a
/// command the launcher cannot take, such as `build`.
pub fn report(world: &mut World) -> Result<Described, String> {
    let commands: Vec<(Entity, Front)> = declared::<Front>(world)
        .into_iter()
        .filter(|(_, front)| front.command.is_some())
        .collect();
    let mut described = Described::default();
    for (at, (entity, front)) in commands.iter().enumerate() {
        let Some(command) = &front.command else {
            continue;
        };
        if let Some(why) = command_refusal(command) {
            return Err(format!("{}: {why}", plugin_of(world, *entity)));
        }
        let earlier = commands.iter().take(at).find(|(_, other)| {
            other.command.as_ref().map(|other| other.name) == Some(command.name)
        });
        if let Some((other, _)) = earlier {
            return Err(format!(
                "plugins {} and {} both add the command `{}`",
                plugin_of(world, *other),
                plugin_of(world, *entity),
                command.name
            ));
        }
        described.commands.push(CommandInfo {
            name: command.name.to_owned(),
            usage: command.usage.to_owned(),
            terminal: front.terminal,
            sessionless: front.sessionless,
            help: command.help.to_owned(),
        });
    }
    described.flags = flags(world)
        .into_iter()
        .map(|flag| FlagInfo {
            name: flag.name.to_owned(),
            value: flag.value.map(str::to_owned),
            help: flag.help.to_owned(),
        })
        .collect();
    described.secrets = declared::<Secret>(world)
        .into_iter()
        .map(|(_, secret)| SecretInfo {
            name: secret.name.to_owned(),
            help: secret.help.to_owned(),
        })
        .collect();
    Ok(described)
}

/// The plugin that declared `entity`, as an error names it.
fn plugin_of(world: &World, entity: Entity) -> String {
    world
        .get::<ProvidedBy>(entity)
        .and_then(|&ProvidedBy(plugin)| world.get::<PluginSource>(plugin))
        .map_or_else(
            || "a plugin added in main.rs".to_owned(),
            |source| format!("`{}`", source.krate),
        )
}

#[cfg(test)]
mod tests;
