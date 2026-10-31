#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::os::unix::fs::PermissionsExt;
use std::time::Duration;

use super::*;

/// The secret this test runs a copy of itself with.
const PROBE: &str = "RIG_SECRET_PROBE";

#[test]
fn the_rebuild_gets_no_secret() {
    if std::env::var_os(PROBE).is_none() {
        let name = "reload::rebuild::tests::the_rebuild_gets_no_secret";
        let output = std::env::current_exe().and_then(|test| {
            Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .env(PROBE, "hunter2")
                .output()
        });
        let ran = output.is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
        });
        assert!(ran, "the run with a secret failed");
        return;
    }
    // A launcher that writes the secret it sees, if any.
    let dir = scratch("secret");
    std::fs::create_dir_all(&dir).ok();
    let launcher = dir.join("rig");
    let script = "#!/bin/sh\nprintf %s \"${RIG_SECRET_PROBE-none}\" > \"$0.seen\"\n";
    std::fs::write(&launcher, script).ok();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).ok();
    let build = ReloadBuild::start(launcher.as_os_str(), Entity::PLACEHOLDER, Wake::default());
    let exited = build.map(|mut build| {
        let process = build
            .process
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        process
            .wait_for(Duration::from_secs(10))
            .is_ok_and(|status| status.is_some())
    });
    let seen = std::fs::read_to_string(dir.join("rig.seen")).ok();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!((exited.ok(), seen.as_deref()), (Some(true), Some("none")));
}

/// A directory of the test's own named `name`, beside its binary.
fn scratch(name: &str) -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    exe.with_file_name(format!("rig-rebuild-{name}-{}", std::process::id()))
}

/// A launcher whose `rig build` fails with `error`, in a directory of the
/// test's own named `name`.
fn failing_launcher(name: &str, error: &str) -> std::path::PathBuf {
    let dir = scratch(name);
    std::fs::create_dir_all(&dir).ok();
    let launcher = dir.join("rig");
    let script = format!("#!/bin/sh\necho '{error}' >&2\nexit 1\n");
    std::fs::write(&launcher, script).ok();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).ok();
    launcher
}

/// The notes from the build plugin carrying the test's error.
#[derive(Resource, Default)]
struct Notes(usize);

/// A failed build leaves its errors in the conversation of the agent that
/// asked as a note, which starts no turn.
#[test]
fn a_failed_build_leaves_its_errors_as_a_note() {
    use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
    use rig_harness_test_support::{MemoryStore, app_on, first_agent, run_until, scripted_models};

    use crate::reload::ReloadPlugin;

    let error = "error[E0425]: cannot find value `words` in this scope";
    let launcher = failing_launcher("failed", error);
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReloadPlugin))
        .init_resource::<Notes>()
        .add_observer(
            |sent: On<Add<Incoming>>, messages: Query<&Incoming>, mut notes: ResMut<Notes>| {
                let note = messages.get(sent.entity).is_ok_and(|message| {
                    message.author.kind == AuthorKind::Plugin(launcher::BUILD_ORIGIN.to_owned())
                        && message.mode == DeliveryMode::Inform
                        && message.text().contains("E0425")
                });
                notes.0 += usize::from(note);
            },
        );
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model_name = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
    });
    let wake = app.world().resource::<Wake>().clone();
    if let Ok(build) = ReloadBuild::start(launcher.as_os_str(), agent, wake) {
        app.insert_resource(build);
        *app.world_mut().resource_mut::<ReloadStatus>() = ReloadStatus::Building { latest: None };
    }
    let failed = |world: &World| *world.resource::<ReloadStatus>() == ReloadStatus::Failed;
    run_until(&mut app, &wakes, failed);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        app.world().resource::<Notes>().0,
        1,
        "the failed build is a note"
    );
    assert_eq!(model.request_count(), 0);
    if let Some(dir) = launcher.parent() {
        std::fs::remove_dir_all(dir).ok();
    }
}
