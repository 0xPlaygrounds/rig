use std::fs;

use rig_ecs::AgentPlugin;
use rig_harness::harness_protocol::Home;

use super::*;

/// An empty home of this test's own.
fn home(test: &str) -> Home {
    let root = std::env::temp_dir().join(format!("rig-sessions-{test}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    Home::at(root)
}

/// An app running session `id` under `home`, which has an agent log, so
/// it is saved.
fn running(home: &Home, id: &str) -> Option<App> {
    let id: SessionId = id.parse().ok()?;
    let dir = home.session(&id);
    fs::create_dir_all(dir.path()).ok()?;
    fs::write(dir.path().join("a1.jsonl"), "").ok()?;
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(dir))
        .add_plugins((AgentPlugin, SessionsPlugin));
    Some(app)
}

/// The `/resume` lines of the sessions under `home`.
fn lines(home: &Home) -> Vec<String> {
    let listed = list(home, Path::new("/nowhere"));
    listed.iter().map(SessionEntry::label).collect()
}

#[test]
fn a_label_is_kept_with_the_session_and_shown_in_its_line() {
    let home = home("label");
    let Some(mut app) = running(&home, "1-1") else {
        return;
    };
    let label = app
        .world_mut()
        .spawn(ResumeLabel::new("goal", "Ship the\nparser"))
        .id();
    app.update();
    let shown = lines(&home);
    assert_eq!(shown.len(), 1);
    assert!(
        shown
            .iter()
            .all(|line| line.contains("  · Ship the parser")),
        "{shown:?}"
    );

    app.world_mut().despawn(label);
    app.update();
    let shown = lines(&home);
    assert!(
        shown.iter().all(|line| !line.contains("parser")),
        "{shown:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}

#[cfg(feature = "tui")]
#[test]
fn resume_picks_from_the_sessions_of_the_apps_home() {
    let home = home("resume");
    let Some(mut earlier) = running(&home, "1-1") else {
        return;
    };
    earlier
        .world_mut()
        .spawn(ResumeLabel::new("goal", "Ship it"));
    earlier.update();
    let Some(mut app) = running(&home, "2-2") else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    world.trigger(rig_harness::commands::RunCommand::new(
        agent,
        "resume".to_owned(),
    ));
    world.flush();
    let mut pickers = world.query::<&rig_harness::pick::Picker>();
    let items: Vec<_> = pickers
        .iter(world)
        .flat_map(|picker| picker.items.clone())
        .collect();
    let commands: Vec<&str> = items.iter().map(|item| item.command.as_str()).collect();
    assert_eq!(commands, ["resume 1-1"]);
    assert!(
        items.iter().all(|item| item.label.contains("Ship it")),
        "{items:?}"
    );
    fs::remove_dir_all(home.root()).ok();
}
