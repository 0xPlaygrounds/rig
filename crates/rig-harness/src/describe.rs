//! How the built agent tells the `rig` launcher what its plugins add to
//! the command line: their commands, their options and the secrets they
//! ask for ([`AgentDescription`]). The launcher asks after each build, by running
//! the binary with [`env::DESCRIBE`] naming a file; [`run`], which the
//! generated `main.rs` ends with, then builds the app (every plugin's
//! `build`, `finish` and `cleanup`), writes the report to that file and
//! exits without running a frame. While it describes, no front takes the
//! run, so no front sets itself up, and the session has no directory.
//!
//! The report is checked as it is made: two plugins adding one command,
//! or a command named as one of the launcher's own (`build`, `plugin`,
//! `help`), fail the build, naming the plugins. Plugins whose agent-loop
//! systems clash are listed in it, for `rig plugin check --build`. A plugin's `build` runs on
//! every describe, so it declares what it adds there and does its work in
//! `finish` or in systems.

use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness_protocol::{
    AgentDescription, DescribedCommand, DescribedFlag, DescribedSecret, REFUSED_EXIT_CODE, env,
};

use crate::host::cli::{Front, command_refusal, declared, flags};
use crate::host::runner::finish_plugins;
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
    finish_plugins(&mut app);
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
            AppExit::from_code(REFUSED_EXIT_CODE)
        }
    }
}

/// What the plugins of the built app `world` add to the command line, and
/// their [`stage_conflicts`](crate::stage_conflicts), or why the build
/// cannot be run: two plugins adding one command, or a command the
/// launcher cannot take, such as `build`.
pub fn report(world: &mut World) -> Result<AgentDescription, String> {
    let commands: Vec<(Entity, Front)> = declared::<Front>(world)
        .into_iter()
        .filter(|(_, front)| front.command.is_some())
        .collect();
    let mut described = AgentDescription::default();
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
        described.commands.push(DescribedCommand {
            name: command.name.to_owned(),
            usage: command.usage.to_owned(),
            terminal: front.terminal,
            sessionless: front.sessionless,
            help: command.help.to_owned(),
        });
    }
    described.flags = flags(world)
        .into_iter()
        .map(|flag| DescribedFlag {
            name: flag.name.to_owned(),
            value: flag.value.map(str::to_owned),
            help: flag.help.to_owned(),
        })
        .collect();
    described.secrets = declared::<Secret>(world)
        .into_iter()
        .map(|(_, secret)| DescribedSecret {
            name: secret.name.to_owned(),
            help: secret.help.to_owned(),
        })
        .collect();
    described.conflicts = crate::load::stage_conflicts(world);
    Ok(described)
}

/// The plugin that declared `entity`, as an error names it.
fn plugin_of(world: &World, entity: Entity) -> String {
    world
        .get::<ProvidedBy>(entity)
        .and_then(|&ProvidedBy(plugin)| world.get::<PluginSource>(plugin))
        .map_or_else(
            || "a plugin added in main.rs".to_owned(),
            |source| format!("`{}`", source.package),
        )
}

#[cfg(test)]
mod tests;
