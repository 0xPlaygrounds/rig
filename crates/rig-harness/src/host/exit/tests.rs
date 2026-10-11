use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::RELOAD_EXIT_CODE;
use rig_ecs::agent::Notice;

use super::{Exit, ExitPlugin, ExitReason};

/// The exits and notices of each frame.
#[derive(Resource, Default)]
struct Seen {
    exits: Vec<AppExit>,
    notices: Vec<String>,
}

fn app() -> App {
    let mut app = App::new();
    app.add_message::<AppExit>()
        .add_message::<Notice>()
        .init_resource::<Seen>()
        .add_plugins(ExitPlugin)
        .add_systems(
            Last,
            |mut exits: MessageReader<AppExit>,
             mut notices: MessageReader<Notice>,
             mut seen: ResMut<Seen>| {
                seen.exits.extend(exits.read().cloned());
                seen.notices
                    .extend(notices.read().map(|notice| notice.text.clone()));
            },
        );
    app
}

#[test]
fn the_first_reason_stands_and_writes_the_launchers_exit_code() {
    let mut app = app();
    app.world_mut().trigger(Exit(ExitReason::Reload));
    app.world_mut().trigger(Exit(ExitReason::Quit));
    app.update();
    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Reload)
    );
    assert_eq!(
        app.world().resource::<Seen>().exits,
        [AppExit::from_code(RELOAD_EXIT_CODE)]
    );
}

#[test]
fn a_plain_exit_gets_the_reason_its_code_stands_for() {
    for (exit, reason) in [
        (AppExit::Success, ExitReason::Quit),
        (AppExit::from_code(RELOAD_EXIT_CODE), ExitReason::Reload),
        (AppExit::from_code(130), ExitReason::Signal),
        (AppExit::from_code(2), ExitReason::Failed(2)),
    ] {
        assert_eq!(reason.exit(), exit);
        let mut app = app();
        app.world_mut().write_message(exit);
        app.update();
        assert_eq!(app.world().get_resource::<ExitReason>(), Some(&reason));
    }
    assert_eq!(
        ExitReason::Switch(None).exit(),
        AppExit::from_code(RELOAD_EXIT_CODE)
    );
}

#[test]
fn a_switch_without_the_launcher_keeps_the_app_running() {
    let mut app = app();
    app.world_mut().trigger(Exit(ExitReason::Switch(None)));
    app.update();
    let seen = app.world().resource::<Seen>();
    assert!(seen.exits.is_empty());
    assert!(
        seen.notices
            .iter()
            .any(|notice| notice.contains("needs the rig launcher")),
        "{:?}",
        seen.notices
    );
    assert!(app.world().get_resource::<ExitReason>().is_none());
}
