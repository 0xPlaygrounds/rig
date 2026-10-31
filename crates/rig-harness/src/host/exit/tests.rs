#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Notice;
use rig_ecs::stage::{AgentPass, Decision, HoldOf, Outcome, PluginEffects, ReadyToStart, Stage};
use rig_harness_protocol::RELOAD_EXIT_CODE;

use super::{Exit, ExitPlugin, ExitReason, ExitRequest};

/// The exits and notices of each frame.
#[derive(Resource, Default)]
struct Seen {
    exits: Vec<AppExit>,
    notices: Vec<String>,
}

fn app() -> App {
    let mut app = App::new();
    app.add_message::<AppExit>()
        .init_resource::<Seen>()
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin, ExitPlugin))
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

/// A hold on the first exit asked for.
#[derive(Component)]
struct NotesHold;

fn hold_first_exit(
    exits: Query<(Entity, &Decision), (With<ExitRequest>, ReadyToStart, PluginEffects)>,
    mut held: Local<bool>,
    mut commands: Commands,
) {
    if *held {
        return;
    }
    if let Some((exit, _)) = exits.iter().find(|(_, decision)| !decision.is_decided()) {
        *held = true;
        commands.spawn((NotesHold, HoldOf(exit)));
    }
}

#[test]
fn a_held_exit_waits_for_its_hold_then_exits_once_with_the_first_reason() {
    let mut app = app();
    app.add_systems(AgentPass, hold_first_exit.in_set(Stage::Gate));
    app.world_mut().trigger(Exit(ExitReason::Quit));
    for _ in 0..3 {
        app.update();
    }
    assert!(app.world().resource::<Seen>().exits.is_empty());
    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Quit)
    );
    let world = app.world_mut();
    let holds: Vec<Entity> = world
        .query_filtered::<Entity, With<NotesHold>>()
        .iter(world)
        .collect();
    assert_eq!(holds.len(), 1);
    for hold in holds {
        world.despawn(hold);
    }
    app.update();
    app.update();
    assert_eq!(app.world().resource::<Seen>().exits, [AppExit::Success]);
}

#[test]
fn a_refused_exit_still_exits() {
    let mut app = app();
    app.add_systems(
        AgentPass,
        (|mut exits: Query<&mut Decision, (With<ExitRequest>, ReadyToStart, PluginEffects)>| {
            for mut decision in &mut exits {
                decision.decide(
                    rig_core::effect::Decider::plugin("test"),
                    Outcome::refused("not now"),
                );
            }
        })
        .in_set(Stage::Gate),
    );
    app.world_mut().trigger(Exit(ExitReason::Failed(3)));
    app.update();
    assert_eq!(
        app.world().resource::<Seen>().exits,
        [AppExit::from_code(3)]
    );
}

/// Exiting for good ends what the calls the exit stopped still run, also a
/// process that ignores the request to stop: the exit does not leave it to a
/// later kill that would die with the agent.
#[cfg(target_os = "linux")]
#[test]
fn an_exit_ends_what_the_calls_it_stopped_still_run() {
    use rig_tools::process::{Launcher, Scope, Spawn};

    let call = Scope::default();
    let ready = std::env::temp_dir().join(format!("rig-exit-stubborn-{}", std::process::id()));
    let script = format!(
        "trap '' TERM; echo $$ > '{}'; exec sleep 30",
        ready.display()
    );
    let started = call.enter(|| Launcher::default().spawn("sh", ["-c", &script], Spawn::default()));
    let Ok(mut process) = started else {
        return assert!(started.is_ok());
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !ready.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    let pid = std::fs::read_to_string(&ready).unwrap_or_default();
    std::fs::remove_file(&ready).ok();
    // The call's thread goes on waiting for it, as a stopped call's does.
    let waiting = std::thread::spawn(move || process.run(None, |_| true));
    let mut app = app();
    app.add_systems(Last, (move || call.stop()).in_set(rig_ecs::StopTurnsOnExit));
    app.world_mut().trigger(Exit(ExitReason::Quit));
    app.update();
    let stat = std::fs::read_to_string(format!("/proc/{}/stat", pid.trim())).unwrap_or_default();
    let alive = !stat.is_empty()
        && !stat
            .rsplit(')')
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('Z');
    assert!(waiting.join().is_ok());
    assert!(
        !pid.trim().is_empty() && !alive,
        "{} outlived the exit",
        pid.trim()
    );
}
