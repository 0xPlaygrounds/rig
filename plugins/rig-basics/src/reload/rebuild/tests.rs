use std::os::unix::fs::PermissionsExt;

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
    let build = ReloadBuild::start(
        launcher.as_os_str(),
        Entity::PLACEHOLDER,
        Wake::default(),
        false,
    );
    let exited = build.map(|mut build| {
        build.exited = build.child.wait().is_ok();
        build.exited
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

/// A failed build the model asked for brings it the build's errors in one
/// message that starts a turn; a later failed `/reload` the user asked for
/// leaves only a note.
#[test]
fn a_failed_build_the_model_asked_for_carries_it_on_with_the_errors_once() {
    use crate::reload::report::tests::{agent_on_model, reports, session};
    use crate::reload::{ReloadAsked, ReloadStatus};
    use rig_core::test_utils::MockCompletionModel;
    use rig_harness_test_support::{MemoryStore, reply, run_until};

    let error = "error[E0425]: cannot find value `words` in this scope";
    let launcher = failing_launcher("failed", error);
    let model = MockCompletionModel::from_stream_turns([reply("I will fix it.")]);
    let (mut app, wakes) = session(&MemoryStore::default(), StartReason::New, &model);
    let agent = agent_on_model(&mut app);
    let asked = ReloadAsked {
        then: String::new(),
        attempts: 1,
        owed: true,
    };
    app.world_mut().entity_mut(agent).insert(asked);
    let fail = |app: &mut App| {
        let wake = app.world().resource::<Wake>().clone();
        if let Ok(build) = ReloadBuild::start(launcher.as_os_str(), agent, wake, false) {
            app.insert_resource(build);
            *app.world_mut().resource_mut::<ReloadStatus>() =
                ReloadStatus::Building { latest: None };
        }
        let failed = |world: &World| *world.resource::<ReloadStatus>() == ReloadStatus::Failed;
        run_until(app, &wakes, failed);
    };
    fail(&mut app);
    run_until(&mut app, &wakes, |_| model.request_count() == 1);
    let sent = reports(&app);
    let once = matches!(sent.as_slice(), [report] if report.contains(error));
    assert!(once, "{sent:?}");
    let read = format!("{:?}", model.requests());
    assert!(read.contains("E0425"), "{read}");

    fail(&mut app);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(reports(&app).len(), 1, "no second report");
    let noted = app
        .world()
        .resource::<crate::reload::report::tests::Sent>()
        .0
        .iter()
        .any(|(author, mode, text)| {
            *author == AuthorKind::Plugin(launcher::BUILD_ORIGIN.to_owned())
                && *mode == DeliveryMode::Inform
                && text.contains("E0425")
        });
    assert!(noted, "the user's failed build is a note");
    assert_eq!(model.request_count(), 1);
    if let Some(dir) = launcher.parent() {
        std::fs::remove_dir_all(dir).ok();
    }
}
