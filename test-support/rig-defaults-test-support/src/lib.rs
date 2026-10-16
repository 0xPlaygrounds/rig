//! The rig agent's default plugins, for tests: [`DEFAULTS`], the plugins
//! of the default `plugins.toml`; [`add_defaults`], which adds them all;
//! and [`assert_stages_ordered`], which fails when a plugin's agent-loop
//! systems are left unordered against theirs, or against each other.

use bevy_app::Plugins;
use rig_harness::harness_protocol::Invocation;
use rig_harness::load;
use rig_harness::prelude::*;

/// The default `plugins.toml`'s plugins, in its order, each loaded from
/// its crate as the generated `main.rs` loads it.
macro_rules! defaults {
    ($($plugin:path),* $(,)?) => {
        [$((stringify!($plugin), |app: &mut App| {
            let krate = stringify!($plugin).split("::").next().unwrap_or_default();
            load::<$plugin>(app, &krate.replace('_', "-"), "")
        })),*]
    };
}

/// The default `plugins.toml`'s plugins, in its order: each one's path,
/// and how the generated `main.rs` adds it.
pub const DEFAULTS: [(&str, fn(&mut App)); 19] = defaults![
    rig_basics::DefaultPromptPlugin,
    rig_models::ModelsPlugin,
    rig_login_chatgpt::ChatgptLoginPlugin,
    rig_models::DefaultsPlugin,
    rig_sessions::SessionsPlugin,
    rig_sessions::AgentsPlugin,
    rig_compaction::CompactionPlugin,
    rig_telemetry::UsagePlugin,
    rig_activity::ActivityPlugin,
    rig_coding_tools::ReadTool,
    rig_coding_tools::EditTool,
    rig_coding_tools::WriteTool,
    rig_coding_tools::ShellTool,
    rig_coding_tools::AttachPlugin,
    rig_reload::ReloadPlugin,
    rig_reload::PluginsPlugin,
    rig_basics::BasicCommandsPlugin,
    rig_print::PrintPlugin,
    rig_tui::TuiPlugin,
];

/// Adds every default plugin to `app`, as the default `plugins.toml` does.
pub fn add_defaults(app: &mut App) {
    for (_, add) in DEFAULTS {
        add(app);
    }
}

/// Fails, naming the systems and their plugins
/// ([`rig_harness::stage_conflicts`], which `rig plugin check --build` runs
/// on the user's own list), when two systems of one [`AgentPass`] stage
/// write the same data unordered in a `--print` run of the harness with
/// the default plugins and `plugins`: which one goes first would be
/// chance. A closure `|app: &mut App| …` is a plugin too, for one that
/// needs setting up. The app is built, not run, in a `RIG_HOME` beside the
/// test binary.
///
/// ```ignore
/// #[test]
/// fn stages_are_ordered_with_the_defaults() {
///     rig_defaults_test_support::assert_stages_ordered(PinnedNotesPlugin);
/// }
/// ```
pub fn assert_stages_ordered<M>(plugins: impl Plugins<M>) {
    let home = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("stages-ordered");
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(Invoked {
            args: Invocation {
                print: Some(String::new()),
                ..Invocation::default()
            },
            terminal: false,
        })
        .add_plugins((HeadlessPlugins, RigHarnessPlugins));
    add_defaults(&mut app);
    app.add_plugins(plugins);
    // The session plugin sends panics to the session's log; a test shows
    // them, so its failed assertions print.
    drop(std::panic::take_hook());
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
}

#[cfg(test)]
mod tests;
