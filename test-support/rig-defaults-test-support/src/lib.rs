//! The rig agent's default plugins, for tests: [`DEFAULTS`], the plugins
//! of the default `plugins.toml`; [`add_defaults`], which adds them all;
//! and [`assert_stages_ordered`], which fails when a plugin's agent-loop
//! systems are left unordered against theirs, or against each other.

use bevy_app::Plugins;
use rig_harness::harness_protocol::Invocation;
use rig_harness::prelude::*;

#[doc(hidden)]
pub use rig_harness as __rig_harness;

/// Plugins by their paths, each with how the generated `main.rs` loads it
/// from its crate: `(&str, fn(&mut App))` pairs, as [`DEFAULTS`] holds.
#[macro_export]
macro_rules! plugins {
    ($($plugin:path),* $(,)?) => {
        [$((stringify!($plugin), |app: &mut $crate::__rig_harness::App| {
            let krate = stringify!($plugin).split("::").next().unwrap_or_default();
            $crate::__rig_harness::load::<$plugin>(app, &krate.replace('_', "-"), "")
        })),*]
    };
}

/// The default `plugins.toml`'s plugins, in its order: each one's path,
/// and how the generated `main.rs` adds it.
pub const DEFAULTS: [(&str, fn(&mut App)); 20] = plugins![
    rig_basics::DefaultPromptPlugin,
    rig_basics::models::ModelsPlugin,
    rig_basics::login_chatgpt::ChatgptLoginPlugin,
    rig_basics::models::DefaultsPlugin,
    rig_basics::session_picker::SessionPickerPlugin,
    rig_basics::sessions::SessionsPlugin,
    rig_basics::sessions::AgentsPlugin,
    rig_basics::compaction::CompactionPlugin,
    rig_basics::telemetry::UsagePlugin,
    rig_basics::activity::ActivityPlugin,
    rig_basics::coding_tools::ReadTool,
    rig_basics::coding_tools::EditTool,
    rig_basics::coding_tools::WriteTool,
    rig_basics::coding_tools::ShellTool,
    rig_basics::coding_tools::AttachPlugin,
    rig_basics::reload::ReloadPlugin,
    rig_basics::reload::PluginsPlugin,
    rig_basics::BasicCommandsPlugin,
    rig_basics::print::PrintPlugin,
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
