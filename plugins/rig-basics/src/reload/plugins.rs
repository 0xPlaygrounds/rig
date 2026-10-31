//! `/plugins`: every plugin in `plugins.toml`, enabled and disabled, as
//! `$RIG_LAUNCHER plugin list` lists them, each with its crate, the comment
//! above its entry, and how it differs from what this build runs (its
//! `PluginSource` entities). In the terminal view it is a picker with the
//! enabled plugins ticked: applying it runs `$RIG_LAUNCHER plugin enable`
//! and `disable` for each plugin ticked or unticked, then reloads at once
//! ([`ReloadStatus::ask`]). Without a terminal it shows the list. The
//! launcher owns the file, so the model changes it the same way from
//! `shell`.

use std::process::Command;

use rig_harness::harness_protocol::{self, essential, secret_variables};
use rig_harness::pick::{Picked, Picker};
use rig_harness::prelude::*;
use rig_tools::process::{Process, Spawn};

use crate::reload::ReloadStatus;

/// `/plugins`.
#[derive(Default)]
pub struct PluginsPlugin;

impl Plugin for PluginsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "plugins",
            "List the plugins in plugins.toml, enabled and disabled, and turn them on and off",
            plugins,
        );
    }
}

/// One entry of `plugins.toml`, from a line of `rig plugin list`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Listed {
    /// Whether its entry is active, not commented out.
    enabled: bool,
    /// The plugin type, such as `rig_tui::TuiPlugin`.
    type_path: String,
    /// Its crate.
    krate: String,
    /// The comment above its entry, with the launcher's own warning.
    description: String,
}

/// The `/plugins` picker's entries, in its order, and the agent it is for.
#[derive(Component)]
struct PluginsPicker {
    listed: Vec<Listed>,
    agent: Entity,
}

/// `/plugins`: the picker on an interactive terminal, else the list.
fn plugins(
    In(args): In<NoArgs>,
    invoked: Option<Res<Invoked>>,
    running: Query<&Name, With<PluginSource>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let listed = match launcher(&["list"]) {
        Ok(output) => parse(&output),
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    let running: Vec<String> = running.iter().map(Name::to_string).collect();
    let details: Vec<String> = listed
        .iter()
        .map(|listed| detail(listed, &running))
        .collect();
    if !invoked.is_some_and(|invoked| invoked.interactive()) {
        let lines: Vec<String> = listed
            .iter()
            .zip(&details)
            .map(|(listed, detail)| {
                let state = if listed.enabled { "on " } else { "off" };
                format!("{state} {}: {detail}", listed.type_path)
            })
            .collect();
        notices.write(Notice::info(args.agent, lines.join("\n")));
        return;
    }
    let mut picker = Picker::new("Plugins")
        .for_agent(args.agent)
        .body(
            "Tick the plugins to run and press Enter: plugins.toml is changed and the agent \
             rebuilds at once. A failed build keeps this one running.",
        )
        .multiple();
    for (listed, detail) in listed.iter().zip(details) {
        picker = picker.item(&listed.type_path, "").checked(listed.enabled);
        if let Some(item) = picker.items.last_mut() {
            item.detail = Some(detail);
        }
    }
    commands
        .spawn((
            picker,
            PluginsPicker {
                listed,
                agent: args.agent,
            },
        ))
        .observe(apply);
}

/// Enables and disables what the user ticked and unticked, then reloads.
fn apply(
    picked: On<Picked>,
    pickers: Query<&PluginsPicker>,
    status: Option<ResMut<ReloadStatus>>,
    mut notices: MessageWriter<Notice>,
) {
    let (Some(chosen), Ok(picker)) = (&picked.chosen, pickers.get(picked.entity)) else {
        return;
    };
    let mut changed = Vec::new();
    for (index, listed) in picker.listed.iter().enumerate() {
        let enable = chosen.items.contains(&index);
        if enable == listed.enabled {
            continue;
        }
        let verb = if enable { "enable" } else { "disable" };
        match launcher(&[verb, &listed.type_path]) {
            Ok(_) => changed.push(format!("{verb}d {}", listed.type_path)),
            Err(why) => {
                notices.write(Notice::error(picker.agent, why));
            }
        }
    }
    if changed.is_empty() {
        notices.write(Notice::info(picker.agent, "No plugin changed."));
        return;
    }
    let reload = match status {
        Some(mut status) => status.ask(picker.agent).err(),
        None => Some("Without rig_basics::reload::ReloadPlugin, it applies when rig starts again."),
    };
    let mut text = format!("plugins.toml: {}.", changed.join(", "));
    if let Some(why) = reload {
        text.push(' ');
        text.push_str(why);
    }
    notices.write(Notice::info(picker.agent, text));
}

/// What a front shows beside a plugin: how it differs from what this build
/// runs and what goes without an essential one, first so a cut row keeps
/// them, then its crate and its description.
fn detail(listed: &Listed, running: &[String]) -> String {
    // A plugin's name is its type's path, which is the path its entry gives.
    let runs = running.contains(&listed.type_path);
    let mut parts = Vec::new();
    match (listed.enabled, runs) {
        (true, false) => parts.push("[enabled, takes effect at the next reload]".to_owned()),
        (false, true) => parts.push("[disabled, runs until the next reload]".to_owned()),
        _ => {}
    }
    if listed.enabled
        && let Some(what) = essential(&listed.type_path)
    {
        parts.push(format!("[warning: disabling it takes away {what}]"));
    }
    parts.push(format!("({})", listed.krate));
    if !listed.description.is_empty() {
        parts.push(listed.description.clone());
    }
    parts.join(" ")
}

/// The entries of `rig plugin list`'s output: `on` or `off`, the type, its
/// crate and its description, separated by spaces.
fn parse(output: &str) -> Vec<Listed> {
    output
        .lines()
        .filter_map(|line| {
            let (state, rest) = word(line)?;
            let (type_path, rest) = word(rest)?;
            let (krate, rest) = word(rest).unwrap_or((rest.trim(), ""));
            Some(Listed {
                enabled: match state {
                    "on" => true,
                    "off" => false,
                    _ => return None,
                },
                type_path: type_path.to_owned(),
                krate: krate.to_owned(),
                description: rest.trim().to_owned(),
            })
        })
        .collect()
}

/// The first word of `text` and the text after it.
fn word(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text.find(char::is_whitespace).unwrap_or(text.len());
    let (word, rest) = text.split_at(end);
    (!word.is_empty()).then_some((word, rest))
}

/// Runs `$RIG_LAUNCHER plugin <args>`, without the secrets, and returns
/// what it printed, or why it failed.
fn launcher(args: &[&str]) -> Result<String, String> {
    let launcher = harness_protocol::launcher()
        .ok_or("/plugins needs the rig launcher: start the agent with `rig`.")?;
    let mut command = Command::new(launcher);
    command.arg("plugin").args(args);
    for secret in secret_variables() {
        command.env_remove(secret);
    }
    let output = Process::start(command, Spawn::default())
        .and_then(|launcher| launcher.output(None))
        .map_err(|failure| format!("Could not run the rig launcher: {failure}"))?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stderr);
        return Err(format!("rig plugin {}: {}", args.join(" "), said.trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests;
