//! `/plugins`: rig's plugins and the extras of `RIG_HOME/settings.toml`,
//! read from that file in this process, each with whether it is on and
//! whether this build runs it (its `PluginSource` entity). In the terminal
//! view it is a picker of short names in group order with the plugins on
//! ticked, and the highlighted plugin's detail beside it ([`PickPreview`]).
//! Applying writes every change in one edit of the file ([`config::turn`])
//! and then reloads ([`ReloadStatus::ask`]), which rebuilds the agent with
//! them. Without a terminal it lists them.
//!
//! `rig plugin enable` and `rig plugin disable` change the same file from a
//! shell, with the same rules.

use rig_harness::harness_protocol::Home;
use rig_harness::harness_protocol::config::{self, BUILT_IN, BuiltIn, Extra, Plugins};
use rig_harness::pick::{Highlighted, PickPreview, Picked, Picker};
use rig_harness::prelude::*;
use rig_harness::services::RigHome;

use crate::reload::{Rebuild, ReloadStatus};

/// `/plugins`.
#[derive(Default)]
pub struct PluginsPlugin;

impl Plugin for PluginsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "plugins",
            "List every plugin, on and off, and turn them on and off in settings.toml",
            plugins,
        )
        .add_systems(Update, preview);
    }
}

/// One plugin as `/plugins` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    /// Its type path, which settings.toml names.
    plugin: String,
    /// What the list shows: a built-in's short name, an extra's crate.
    name: String,
    /// Whether settings.toml turns it on.
    on: bool,
    /// Its state, short: `on`, `off`, or why it is off.
    state: String,
    /// Everything about it, for the preview.
    about: String,
}

/// The `/plugins` picker's rows, in its order, and the agent it is for.
#[derive(Component)]
struct PluginsPicker {
    rows: Vec<Row>,
    agent: Entity,
}

/// `/plugins`: the picker on an interactive terminal, else the list.
fn plugins(
    In(args): In<NoArgs>,
    home: Option<Res<RigHome>>,
    invoked: Option<Res<Invoked>>,
    running: Query<&Name, With<PluginSource>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let home = home.map_or_else(Home::from_env, |home| home.0.clone());
    let chosen = match config::chosen(&home) {
        Ok(chosen) => chosen,
        Err(why) => {
            // An unknown plugin's refusal names its own fix.
            let fix = match &why {
                config::Error::Unavailable { plugin, .. } => {
                    format!("`rig plugin disable {plugin}` in a shell turns it off.")
                }
                _ => "Edit the file to fix it.".to_owned(),
            };
            notices.write(Notice::error(
                args.agent,
                format!(
                    "{} does not read, so /plugins cannot tell which plugins are on: {why}\n{fix}",
                    home.settings().display()
                ),
            ));
            return;
        }
    };
    let running: Vec<String> = running.iter().map(Name::to_string).collect();
    let rows = rows(&chosen, &running);
    if !invoked.is_some_and(|invoked| invoked.interactive()) {
        let names = rows.iter().map(|row| row.name.len()).max().unwrap_or(0);
        let paths = rows.iter().map(|row| row.plugin.len()).max().unwrap_or(0);
        let lines: Vec<String> = (rows.iter())
            .map(|row| {
                format!(
                    "{:<names$}  {:<paths$}  {}",
                    row.name, row.plugin, row.state
                )
            })
            .collect();
        notices.write(Notice::info(args.agent, lines.join("\n")));
        return;
    }
    let mut picker = Picker::new("Plugins")
        .for_agent(args.agent)
        .body(
            "Tick the plugins to run and press Enter: RIG_HOME/settings.toml changes in one \
             edit and the agent rebuilds. A failed build keeps this one running.",
        )
        .multiple();
    picker.items = (rows.iter())
        .map(|row| PickItem {
            checked: row.on,
            // Typing a type path finds its plugin; → shows it with the whole state.
            more: Some(format!("{}: {}", row.plugin, row.state)),
            ..PickItem::new(&row.name, "").detail(&row.state)
        })
        .collect();
    // The terminal view gives the preview its half of the screen from the
    // first frame only when the picker comes with one.
    let first = rows
        .first()
        .map(|row| row.about.clone())
        .unwrap_or_default();
    commands
        .spawn((
            picker,
            PickPreview::Text(first),
            PluginsPicker {
                rows,
                agent: args.agent,
            },
        ))
        .observe(apply);
}

/// Every plugin, rig's in group order and then the extras, as `chosen`
/// turns them on, and whether `running` holds each.
fn rows(chosen: &Plugins, running: &[String]) -> Vec<Row> {
    let mut built_ins: Vec<&BuiltIn> = BUILT_IN.iter().collect();
    built_ins.sort_by_key(|built_in| built_in.group);
    let on = |plugin: &str| chosen.on.iter().any(|on| on == plugin);
    let runs = |plugin: &str| running.iter().any(|runs| runs == plugin);
    let built_ins = (built_ins.into_iter())
        .map(|built_in| built_in_row(built_in, on(built_in.plugin), runs(built_in.plugin)));
    let extras = (chosen.extra.iter()).map(|extra| {
        let shared = (chosen.extra.iter())
            .any(|other| other.krate == extra.krate && other.plugin != extra.plugin);
        extra_row(extra, shared, on(&extra.plugin), runs(&extra.plugin))
    });
    built_ins.chain(extras).collect()
}

/// The row of rig's plugin `built_in`.
fn built_in_row(built_in: &BuiltIn, on: bool, runs: bool) -> Row {
    let mut state = if built_in.available() {
        state(on, runs)
    } else {
        format!("off here: {} only", built_in.targets.join("/"))
    };
    if let Some(what) = built_in.essential.filter(|_| !on) {
        state.push_str(&format!(" [warning: takes away {what}]"));
    }
    let mut about = format!(
        "{}\n{} · {}\n{}\n\n{}\n",
        built_in.name,
        built_in.group.title(),
        built_in.krate,
        built_in.plugin,
        states(Some(built_in.on), on, runs),
    );
    if !built_in.targets.is_empty() {
        about.push_str(&format!(
            "Builds on {} only.\n",
            built_in.targets.join(" and ")
        ));
    }
    about.push_str(reload(on, runs));
    match (built_in.essential, on) {
        (Some(what), true) => {
            about.push_str(&format!("Essential: turning it off takes away {what}.\n"));
        }
        (Some(what), false) => {
            about.push_str(&format!("Warning: off, which takes away {what}.\n"));
        }
        (None, _) => {}
    }
    about.push('\n');
    about.push_str(built_in.about);
    Row {
        plugin: built_in.plugin.to_owned(),
        name: built_in.name.to_owned(),
        on,
        state,
        about,
    }
}

/// The row of the extra plugin `extra`, named by its crate, and by its
/// type too when another extra is `shared` from that crate.
fn extra_row(extra: &Extra, shared: bool, on: bool, runs: bool) -> Row {
    let name = match extra.plugin.rsplit("::").next().filter(|_| shared) {
        Some(last) => format!("{} {last}", extra.krate),
        None => extra.krate.clone(),
    };
    let about = format!(
        "{name}\nYour plugins · {}\n{}\n\n{}\n{}\nFrom {}.",
        extra.krate,
        extra.plugin,
        states(None, on, runs),
        reload(on, runs),
        extra.source,
    );
    Row {
        plugin: extra.plugin.clone(),
        name,
        on,
        state: state(on, runs),
        about,
    }
}

/// A plugin's short state: on or off in settings.toml, and whether that
/// waits for the next reload because this build runs it or not.
fn state(on: bool, runs: bool) -> String {
    let word = if on { "on" } else { "off" };
    if on == runs {
        word.to_owned()
    } else {
        format!("{word} · next reload")
    }
}

/// What the next reload changes for a plugin, as a line, or nothing.
fn reload(on: bool, runs: bool) -> &'static str {
    match (on, runs) {
        (true, false) => "Takes effect at the next reload.\n",
        (false, true) => "Runs until the next reload.\n",
        _ => "",
    }
}

/// A plugin's default, when it has one, its state in settings.toml and
/// whether this build runs it.
fn states(default: Option<bool>, on: bool, runs: bool) -> String {
    let word = |on: bool| if on { "on" } else { "off" };
    let mut parts = Vec::new();
    if let Some(default) = default {
        parts.push(format!("default {}", word(default)));
    }
    parts.push(format!("{} in settings", word(on)));
    parts.push(if runs { "running now" } else { "not running" }.to_owned());
    parts.join(" · ")
}

/// Shows the highlighted plugin's detail beside each `/plugins` picker.
fn preview(
    pickers: Query<
        (Entity, &PluginsPicker, &Highlighted, Option<&PickPreview>),
        Changed<Highlighted>,
    >,
    mut commands: Commands,
) {
    for (entity, picker, highlighted, shown) in &pickers {
        let Some(row) = picker.rows.get(highlighted.0) else {
            continue;
        };
        let preview = PickPreview::Text(row.about.clone());
        if shown != Some(&preview) {
            commands.entity(entity).insert(preview);
        }
    }
}

/// Writes what the user ticked and unticked to settings.toml in one edit,
/// then reloads.
fn apply(
    picked: On<Picked>,
    pickers: Query<&PluginsPicker>,
    home: Option<Res<RigHome>>,
    status: Option<ResMut<ReloadStatus>>,
    mut notices: MessageWriter<Notice>,
) {
    let (Some(chosen), Ok(picker)) = (&picked.chosen, pickers.get(picked.entity)) else {
        return;
    };
    let changed: Vec<(&Row, bool)> = (picker.rows.iter().enumerate())
        .map(|(index, row)| (row, chosen.items.contains(&index)))
        .filter(|(row, on)| row.on != *on)
        .collect();
    if changed.is_empty() {
        notices.write(Notice::info(picker.agent, "No plugin changed."));
        return;
    }
    let path = home
        .map_or_else(Home::from_env, |home| home.0.clone())
        .settings();
    let changes: Vec<(&str, bool)> = (changed.iter())
        .map(|(row, on)| (row.plugin.as_str(), *on))
        .collect();
    let edited = config::edit(&path, |file| {
        config::turn(file, &changes).map(|changed| (changed, ()))
    });
    if let Err(why) = edited {
        notices.write(Notice::error(
            picker.agent,
            format!("{} is unchanged: {why}", path.display()),
        ));
        return;
    }
    let turned = |on: bool| -> Vec<&str> {
        (changed.iter())
            .filter(|(_, turned)| *turned == on)
            .map(|(row, _)| row.name.as_str())
            .collect()
    };
    let mut said = Vec::new();
    for (on, names) in [("on", turned(true)), ("off", turned(false))] {
        if !names.is_empty() {
            said.push(format!("turned {on} {}", names.join(", ")));
        }
    }
    let reload = match status {
        Some(mut status) => status.ask(picker.agent, Rebuild::Build).err(),
        None => Some("Without rig_basics::reload::ReloadPlugin, it applies when rig starts again."),
    };
    let mut text = format!("{}: {}.", path.display(), said.join("; "));
    if let Some(why) = reload {
        text.push(' ');
        text.push_str(why);
    }
    notices.write(Notice::info(picker.agent, text));
}

#[cfg(test)]
mod tests;
