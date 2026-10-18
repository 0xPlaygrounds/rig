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

/// The default `plugins.toml`'s agent halves (`plugin =`), in its order;
/// the view halves run in the terminal view, not in a session. Each one's path,
/// and how the generated `main.rs` adds it.
pub const DEFAULTS: [(&str, fn(&mut App)); 20] = plugins![
    rig_basics::prompt::DefaultPromptPlugin,
    rig_basics::models::ModelsPlugin,
    rig_basics::login_chatgpt::ChatgptLoginPlugin,
    rig_basics::models::defaults::DefaultsPlugin,
    rig_basics::session_picker::SessionPickerPlugin,
    rig_basics::sessions::SessionsPlugin,
    rig_basics::sessions::agents::AgentsPlugin,
    rig_basics::compaction::CompactionPlugin,
    rig_basics::telemetry::usage::UsagePlugin,
    rig_basics::activity::ActivityPlugin,
    rig_basics::coding_tools::ReadTool,
    rig_basics::coding_tools::EditTool,
    rig_basics::coding_tools::WriteTool,
    rig_basics::coding_tools::ShellTool,
    rig_basics::coding_tools::AttachPlugin,
    rig_basics::coding_tools::remote::RemoteHostsPlugin,
    rig_agents::AgentsPlugin,
    rig_basics::reload::ReloadPlugin,
    rig_basics::reload::plugins::PluginsPlugin,
    rig_basics::BasicCommandsPlugin,
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
    let _app = assert_ordered("", plugins);
}

/// [`assert_stages_ordered`] for plugins that replace the default `R`:
/// the defaults are added without it, as a `plugins.toml` with its entry
/// disabled has them, and none of them may add it back.
///
/// ```ignore
/// rig_defaults_test_support::assert_stages_ordered_replacing::<CompactionPlugin, _>(
///     KeepLastCompactionPlugin,
/// );
/// ```
pub fn assert_stages_ordered_replacing<R: Plugin, M>(plugins: impl Plugins<M>) {
    let replaced = std::any::type_name::<R>();
    let app = assert_ordered(replaced, plugins);
    assert!(
        !app.is_plugin_added::<R>(),
        "a default plugin adds {replaced} back"
    );
}

/// Builds the harness with every default but the one at `left_out`, and
/// `plugins`, and fails on a stage conflict.
fn assert_ordered<M>(left_out: &str, plugins: impl Plugins<M>) -> App {
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
    for (_, add) in DEFAULTS.iter().filter(|(path, _)| *path != left_out) {
        add(&mut app);
    }
    app.add_plugins(plugins);
    // The session plugin sends panics to the session's log; a test shows
    // them, so its failed assertions print.
    drop(std::panic::take_hook());
    app.finish();
    app.cleanup();
    let conflicts = rig_harness::stage_conflicts(app.world_mut());
    assert!(conflicts.is_empty(), "{}", conflicts.join("\n"));
    app
}

#[cfg(test)]
mod tests;
