use bevy_app::ValidateParentHasComponentPlugin as ParentHas;
use bevy_log::{LogPlugin, error, info, warn};

use super::*;

/// Logs more warnings than are kept, an error about agent `a1` and an
/// info line, and spawns what makes Bevy warn (B0004): a `Name`d child of
/// an entity without a `Name`.
#[derive(Default)]
struct Noisy;

impl Plugin for Noisy {
    fn build(&self, app: &mut App) {
        app.add_plugins(ParentHas::<Name>::in_schedule(Update))
            .add_systems(Startup, |mut commands: Commands| {
                (0..KEPT).for_each(|n| warn!("warning {n}"));
                error!(agent = "a1", "about a1");
                info!("not kept");
                let parent = commands.spawn_empty().id();
                commands.spawn((Name::new("child"), ChildOf(parent)));
            });
    }
}

#[test]
fn warnings_and_errors_are_kept_tagged_and_bounded() {
    let mut app = App::new();
    let custom_layer = rig_harness::host::session::log_events;
    let log = LogPlugin {
        custom_layer,
        ..LogPlugin::default()
    };
    app.add_plugins((TaskPoolPlugin::default(), log, DiagnosticsPlugin));
    rig_harness::load::<Noisy>(&mut app, "rig-harness", "");
    app.update();
    let kept = &app.world().resource::<Diagnostics>().0;
    let newest: Vec<_> = kept
        .iter()
        .rev()
        .take(2)
        .map(|l| {
            (
                l.error,
                l.target.as_str(),
                l.message.split(':').next(),
                l.agent.as_deref(),
                l.plugin.as_deref(),
            )
        })
        .collect();
    let noisy = Some(std::any::type_name::<Noisy>());
    let bevy = (
        false,
        "bevy_app::hierarchy",
        Some("warning[B0004]"),
        None,
        None,
    );
    let own = (true, module_path!(), Some("about a1"), Some("a1"), noisy);
    assert_eq!(newest, [bevy, own]);
    let oldest = kept.front().map(|logged| logged.message.as_str());
    assert_eq!((kept.len(), oldest), (KEPT, Some("warning 2")));
}
