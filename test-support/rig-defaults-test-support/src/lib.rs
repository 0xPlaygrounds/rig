//! The rig agent's default plugins, for tests: [`DEFAULTS`], the built-in
//! plugins on by default; [`add_defaults`], which adds them all;
//! and [`assert_stages_ordered`], which fails when a plugin's agent-loop
//! systems are left unordered against theirs, or against each other.

use bevy_app::Plugins;
use rig_harness::harness_protocol::Invocation;
use rig_harness::prelude::*;

#[doc(hidden)]
pub use rig_harness as __rig_harness;

/// The `(path, how the agent adds it)` pairs of the plugins
/// [`built_in_plugins!`](rig_harness::harness_protocol::built_in_plugins)
/// lists as `on`, from its entries.
macro_rules! on_by_default {
    (@keep [$($kept:tt)*]) => { &[$($kept)*] };
    (@keep [$($kept:tt)*] on $first:ident $(:: $rest:ident)*, $krate:literal, [$($os:literal),*],
        $essential:expr, $about:literal; $($more:tt)*) => {
        on_by_default!(@keep [$($kept)* (
            concat!(stringify!($first) $(, "::", stringify!($rest))*),
            |app: &mut App| rig_harness::load::<$first $(:: $rest)*>(app, $krate, ""),
        ),] $($more)*)
    };
    (@keep [$($kept:tt)*] off $first:ident $(:: $rest:ident)*, $krate:literal,
        [$($os:literal),*], $essential:expr, $about:literal; $($more:tt)*) => {
        on_by_default!(@keep [$($kept)*] $($more)*)
    };
    ($($entries:tt)*) => { on_by_default!(@keep [] $($entries)*) };
}

/// The built-in plugins on by default, in the order the agent adds them:
/// each one's path, and how the agent adds it.
pub const DEFAULTS: &[(&str, fn(&mut App))] =
    rig_harness::harness_protocol::built_in_plugins!(on_by_default);

/// Adds every default plugin to `app`, as a home without `[plugins]` does.
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
pub fn assert_stages_ordered<M>(plugins: impl Plugins<M>) {
    let _app = assert_ordered("", plugins);
}

/// A test, `stages_are_ordered_with_the_defaults`, that runs
/// [`assert_stages_ordered`] on `plugins`.
///
/// ```ignore
/// rig_defaults_test_support::stages_ordered_test!(PinnedNotesPlugin);
/// ```
#[macro_export]
macro_rules! stages_ordered_test {
    ($plugins:expr) => {
        #[test]
        fn stages_are_ordered_with_the_defaults() {
            $crate::assert_stages_ordered($plugins);
        }
    };
}

/// [`assert_stages_ordered`] for plugins that replace the default `R`:
/// the defaults are added without it, as `[plugins] disable` naming it has
/// them, and none of them may add it back.
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
