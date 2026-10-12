use std::fs;

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// Holds each turn's proposed end while [`Holding`] says so.
#[derive(Resource)]
struct Holding {
    hold: bool,
    held: bool,
}

/// A hold of the test's.
#[derive(Component)]
struct TestHold;

fn hold(
    proposed: Query<Entity, Added<TurnOutcome>>,
    mut holding: ResMut<Holding>,
    mut commands: Commands,
) {
    for turn in &proposed {
        if holding.hold {
            commands.spawn((TestHold, HoldOf(turn)));
            holding.held = true;
        }
    }
}

/// The titles written, in order, and the bells rung.
fn written(buffer: &Mutex<Vec<u8>>) -> (Vec<String>, usize, Vec<u8>) {
    let bytes = buffer.lock().map(|bytes| bytes.clone()).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let titles = text
        .split("\x1b]2;")
        .skip(1)
        .filter_map(|rest| rest.split_once("\x1b\\").map(|(title, _)| title.to_owned()))
        .collect();
    let bells = bytes.iter().filter(|&&byte| byte == 0x07).count();
    (titles, bells, bytes)
}

/// Runs one frame 100 ms later on the app's clock, then a few with it still.
fn tick(app: &mut App) {
    rig_harness_test_support::advance(app, Duration::from_millis(100));
    for _ in 0..3 {
        app.update();
    }
}

#[test]
fn the_title_follows_the_turn_rings_once_and_is_given_back() {
    // A stand-in `tmux` first on the path, which logs its arguments.
    let bin = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("ho-terminal-title-bin");
    fs::remove_dir_all(&bin).ok();
    fs::create_dir_all(&bin).ok();
    let log = bin.join("tmux.log");
    let script = format!(
        "#!/bin/sh\necho \"$@\" >> '{}'\n[ \"$1\" = display-message ] && printf 'old\\t0\\n'\nexit 0\n",
        log.display()
    );
    fs::write(bin.join("tmux"), script).ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(bin.join("tmux"), fs::Permissions::from_mode(0o755)).ok();
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![bin.clone()];
    paths.extend(std::env::split_paths(&path));
    let path = std::env::join_paths(paths).unwrap_or_default();
    // SAFETY: set before the app starts any thread; the test binary runs
    // this one test.
    unsafe {
        std::env::set_var("PATH", path);
        std::env::set_var("TMUX", "/tmp/tmux-test,1,0");
        std::env::set_var("TMUX_PANE", "%1");
    }

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(Settings::from_toml("[title]\nbell_seconds = 0"))
        .insert_resource(TitleOutput::Buffer(buffer.clone()))
        .insert_resource(Holding {
            hold: true,
            held: false,
        })
        .add_plugins((AgentPlugin, JournalPlugin, ExitPlugin, TitlePlugin))
        .add_systems(AgentPass, hold.in_set(Stage::Review));
    let model = MockCompletionModel::from_stream_turns([reply("One."), reply("Two.")]);
    let line = connect(&model);
    assert!(line.is_some(), "the built-in catalog lists the model");
    let Some(line) = line else {
        return;
    };
    let agent = app.world_mut().spawn((Agent, line)).id();
    app.finish();
    app.update();
    assert_eq!(written(&buffer).0, ["rig · idle"]);

    // A turn of 1 s: "working" while it runs, "idle" once it ended, at most
    // three title writes while it ran, and one bell at its end.
    let go = Deliver::new(agent, "Go.", DeliveryMode::Queue).with_origin(Origin::user());
    app.world_mut().trigger(go);
    run_until(&mut app, &wakes, |_| {
        written(&buffer)
            .0
            .iter()
            .any(|title| title.contains("working"))
    });
    run_until(&mut app, &wakes, |world| world.resource::<Holding>().held);
    for _ in 0..10 {
        tick(&mut app);
    }
    app.world_mut().resource_mut::<Holding>().hold = false;
    let holds: Vec<Entity> = app
        .world_mut()
        .query_filtered::<Entity, With<TestHold>>()
        .iter(app.world())
        .collect();
    for held in holds {
        app.world_mut().despawn(held);
    }
    run_until(&mut app, &wakes, |_| {
        written(&buffer)
            .0
            .last()
            .is_some_and(|title| title == "rig · idle")
    });
    let (titles, bells, _) = written(&buffer);
    let during = titles
        .get(1..titles.len().saturating_sub(1))
        .unwrap_or_default();
    assert!(!during.is_empty() && during.len() <= 3, "{titles:?}");
    assert!(
        during
            .iter()
            .all(|title| title.starts_with("rig · working")),
        "{titles:?}"
    );
    assert_eq!(titles.last().map(String::as_str), Some("rig · idle"));
    assert_eq!(bells, 1);

    // With a 10 s threshold, a short turn rings nothing.
    app.world_mut().resource_mut::<TitleSettings>().bell_seconds = 10;
    let again = Deliver::new(agent, "Again.", DeliveryMode::Queue).with_origin(Origin::user());
    app.world_mut().trigger(again);
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && world.get::<ActiveTurn>(agent).is_none()
    });
    app.update();
    let (titles, bells, _) = written(&buffer);
    assert_eq!(titles.last().map(String::as_str), Some("rig · idle"));
    assert_eq!(bells, 1);

    // tmux renamed the window.
    run_until(&mut app, &wakes, |_| {
        fs::read_to_string(&log).is_ok_and(|log| log.contains("rename-window -t %1 rig · working"))
    });

    // On exit the terminal's title and the window's name are given back.
    app.world_mut().trigger(Exit(ExitReason::Quit));
    run_until(&mut app, &wakes, |world| {
        world
            .get_resource::<Messages<AppExit>>()
            .is_some_and(|exits| !exits.is_empty())
    });
    let (_, _, bytes) = written(&buffer);
    assert!(bytes.ends_with(POP_TITLE), "{bytes:?}");
    let logged = fs::read_to_string(&log).unwrap_or_default();
    assert!(
        logged
            .lines()
            .next()
            .is_some_and(|line| line.starts_with("display-message")),
        "{logged}"
    );
    assert_eq!(
        logged.lines().last(),
        Some("rename-window -t %1 old"),
        "{logged}"
    );
}
