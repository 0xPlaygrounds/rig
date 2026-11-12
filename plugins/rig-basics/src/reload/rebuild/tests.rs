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
    let build = ReloadBuild::start(
        launcher.as_os_str(),
        Rebuild::Build,
        Entity::PLACEHOLDER,
        Wake::default(),
    );
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

/// Whether the process `pid` runs: there and not a zombie.
#[cfg(target_os = "linux")]
fn alive(pid: &str) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        !stat
            .rsplit(')')
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('Z')
    })
}

/// A stopped rebuild asks `rig build` to stop before killing it, so the
/// launcher kills cargo, which runs in a group of its own.
#[cfg(target_os = "linux")]
#[test]
fn a_stopped_rebuild_leaves_no_cargo() {
    let dir = scratch("stopped");
    std::fs::create_dir_all(&dir).ok();
    let launcher = dir.join("rig");
    // Its "cargo" is a sleep in a session of its own, killed on SIGTERM.
    let script = "#!/bin/sh\ntrap 'kill $!; exit 143' TERM\n\
                  setsid sleep 60 & echo $! > \"$0.pid\"\nwait\n";
    std::fs::write(&launcher, script).ok();
    std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755)).ok();
    let build = ReloadBuild::start(
        launcher.as_os_str(),
        Rebuild::Build,
        Entity::PLACEHOLDER,
        Wake::default(),
    );
    let pid_file = dir.join("rig.pid");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !pid_file.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(100));
    let cargo = std::fs::read_to_string(&pid_file).unwrap_or_default();
    assert!(build.is_ok() && alive(cargo.trim()));
    drop(build);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while alive(cargo.trim()) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let left = alive(cargo.trim());
    std::fs::remove_dir_all(&dir).ok();
    assert!(!left, "cargo outlived the stopped rebuild");
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
        kind: ChoiceKind::ForThisRun,
    });
    let wake = app.world().resource::<Wake>().clone();
    if let Ok(build) = ReloadBuild::start(launcher.as_os_str(), Rebuild::Build, agent, wake) {
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

/// The notices said so far, and the notes from the build plugin.
#[derive(Resource, Default)]
struct Said {
    notices: Vec<String>,
    notes: Vec<String>,
}

/// What a `/update` said whose `rig update` is the shell `script`, with
/// this rig's launcher at `$RIG_INSTALLED`, once it failed.
fn failed_update(name: &str, script: &str) -> Said {
    use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
    use rig_harness_test_support::{MemoryStore, app_on, first_agent, run_until, scripted_models};

    use crate::reload::{LauncherFile, ReloadPlugin};

    let launcher = failing_launcher(name, "");
    let dir = launcher
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let installed = dir.join("rig-installed");
    std::fs::write(&installed, "old").ok();
    let script = format!(
        "#!/bin/sh\nRIG_INSTALLED='{}'\n{script}\n",
        installed.display()
    );
    std::fs::write(&launcher, script).ok();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/model", &model)]))
        .insert_resource(LauncherFile::new(
            installed.clone(),
            harness_protocol::Stamp::of(&installed),
        ))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReloadPlugin))
        .init_resource::<Said>()
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut said: ResMut<Said>| {
                let texts = notices.read().map(|notice| notice.text.clone());
                said.notices.extend(texts);
            },
        )
        .add_observer(
            |sent: On<Add<Incoming>>, messages: Query<&Incoming>, mut said: ResMut<Said>| {
                if let Ok(message) = messages.get(sent.entity)
                    && message.author.kind == AuthorKind::Plugin(launcher::BUILD_ORIGIN.to_owned())
                {
                    said.notes.push(message.text().to_owned());
                }
            },
        );
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let wake = app.world().resource::<Wake>().clone();
    if let Ok(build) = ReloadBuild::start(launcher.as_os_str(), Rebuild::Update, agent, wake) {
        app.insert_resource(build);
        *app.world_mut().resource_mut::<ReloadStatus>() = ReloadStatus::Building { latest: None };
    }
    let failed = |world: &World| *world.resource::<ReloadStatus>() == ReloadStatus::Failed;
    run_until(&mut app, &wakes, failed);
    for _ in 0..3 {
        app.update();
    }
    let said = std::mem::take(app.world_mut().resource_mut::<Said>().as_mut());
    std::fs::remove_dir_all(dir).ok();
    assert_eq!(model.request_count(), 0);
    said
}

/// A failed `/update` runs `rig update`, shows the lines it starts with
/// `rig update: `, such as the command it runs, and keeps this build. One
/// that failed before it installed anything says so with cargo's error, not
/// the build log of an older build.
#[test]
fn a_failed_update_shows_what_rig_update_said() {
    let said = "rig update: `cargo install --force rig-cli` failed (exit status: 101); this rig \
                is unchanged.";
    let script = format!(
        "echo \"rig update: running $1\" >&2\n\
         echo 'error: could not find `rig-cli` in registry' >&2\necho '{said}' >&2\nexit 1"
    );
    let Said { notices, notes } = failed_update("update", &script);
    assert_eq!(
        notices.get(..3).map(<[String]>::to_vec),
        Some(vec![
            "rig update: running update".to_owned(),
            said.to_owned(),
            "The update failed: error: could not find `rig-cli` in registry; this rig keeps \
             running."
                .to_owned(),
        ])
    );
    assert_eq!(notes.len(), 1);
    assert!(notes.iter().all(|note| {
        note.starts_with("Updating rig (/update) failed before it was installed")
            && note.contains("could not find `rig-cli`")
            && !note.contains("build.log")
    }));
}

/// An update that installed the new launcher, whose agent build then
/// failed, is told as a failed build: the launcher's file changed, whatever
/// `rig update` said.
#[test]
fn an_update_that_installed_and_failed_to_build_says_the_build_failed() {
    let script = "echo new > \"$RIG_INSTALLED.new\"; mv \"$RIG_INSTALLED.new\" \"$RIG_INSTALLED\"
echo 'error: could not compile `rig-harness`' >&2
exit 1";
    let Said { notes, .. } = failed_update("installed", script);
    assert_eq!(notes.len(), 1);
    assert!(
        notes.iter().all(|note| {
            note.starts_with("Building the agent (/update) failed")
                && note.contains("could not compile `rig-harness`")
        }),
        "{notes:?}"
    );
}

/// The exits asked for, how the headless runs ended, and the notices.
#[derive(Resource, Default)]
struct Ends {
    exits: Vec<ExitReason>,
    runs: Vec<crate::print::HeadlessEnded>,
    notices: Vec<(NoticeLevel, String)>,
}

/// How a `-p` run that reloads (`rebuild`) ends, its launcher replaced by
/// the build when `replace`; with `--json` when `json`. It writes to this
/// process's stdout and stderr, as any run does.
fn headless_reload(name: &str, rebuild: Rebuild, replace: bool, json: bool) -> Ends {
    use rig_harness::host::exit::ExitPlugin;
    use rig_harness::host::launcher::Invoked;
    use rig_harness_test_support::{MemoryStore, app_on, run_until, scripted_models};

    use crate::print::{HeadlessEnded, PrintPlugin};
    use crate::reload::{LauncherFile, ReloadPlugin};

    let build = failing_launcher(name, "");
    let dir = build
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let installed = dir.join("rig-installed");
    std::fs::write(&installed, "old").ok();
    let replacing = if replace {
        format!(
            "echo new > '{0}.new'; mv '{0}.new' '{0}'",
            installed.display()
        )
    } else {
        String::new()
    };
    std::fs::write(&build, format!("#!/bin/sh\n{replacing}\nexit 0\n")).ok();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Invoked {
        args: harness_protocol::Invocation {
            print: Some("/help".to_owned()),
            flags: json
                .then(|| (crate::print::JSON.to_owned(), None))
                .into_iter()
                .collect(),
            ..Default::default()
        },
        terminal: false,
    })
    .insert_resource(LauncherFile::new(
        installed.clone(),
        harness_protocol::Stamp::of(&installed),
    ))
    .insert_resource(scripted_models([]))
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        PrintPlugin,
        ReloadPlugin,
        crate::BasicCommandsPlugin,
    ))
    .init_resource::<Ends>()
    .add_systems(
        Last,
        |mut notices: MessageReader<Notice>, mut ends: ResMut<Ends>| {
            let notices = notices
                .read()
                .map(|notice| (notice.level, notice.text.clone()));
            ends.notices.extend(notices);
        },
    )
    .add_observer(|exit: On<Exit>, mut ends: ResMut<Ends>| {
        ends.exits.push(exit.0.clone());
    })
    .add_observer(|ended: On<HeadlessEnded>, mut ends: ResMut<Ends>| {
        ends.runs.push(ended.event().clone());
    });
    app.finish();
    let wake = app.world().resource::<Wake>().clone();
    if let Ok(build) = ReloadBuild::start(build.as_os_str(), rebuild, Entity::PLACEHOLDER, wake) {
        app.insert_resource(build);
        *app.world_mut().resource_mut::<ReloadStatus>() = ReloadStatus::Building { latest: None };
    }
    let exited = |world: &World| !world.resource::<Ends>().exits.is_empty();
    run_until(&mut app, &wakes, exited);
    // The exit's frames, on which the run writes its last lines.
    for _ in 0..3 {
        app.update();
    }
    std::fs::remove_dir_all(&dir).ok();
    std::mem::take(app.world_mut().resource_mut::<Ends>().as_mut())
}

/// A `-p` run whose launcher was replaced, by its `/update` or from
/// outside, does not reload: it says so, as a notice a `--json` run writes
/// as a line and a warning a `-p` run writes to stderr whatever its
/// prompt, and ends as any run does, with 0: [`HeadlessEnded`] is what its
/// `exit` line says. The next run starts the new launcher. One whose
/// launcher is the same reloads.
///
/// [`HeadlessEnded`]: crate::print::HeadlessEnded
#[test]
fn a_headless_run_whose_launcher_was_replaced_ends_instead_of_reloading() {
    let ended = |ends: &Ends, notice: (NoticeLevel, &str)| {
        ends.exits == [ExitReason::Quit]
            && ends.runs.iter().map(|run| run.code).eq([0])
            && ends
                .notices
                .iter()
                .any(|(level, text)| (*level, text.as_str()) == notice)
    };
    let updated = headless_reload("headless-update", Rebuild::Update, true, false);
    let said = "rig updated; the next run starts the new version.";
    assert!(
        ended(&updated, (NoticeLevel::Info, said)),
        "{:?} {:?} {:?}",
        updated.exits,
        updated.runs,
        updated.notices
    );
    let replaced = headless_reload("headless-replaced", Rebuild::Build, true, false);
    let said = "rig was replaced since this run started, so it does not reload; the next run \
                starts the new version.";
    assert!(
        ended(&replaced, (NoticeLevel::Warn, said)),
        "{:?} {:?} {:?}",
        replaced.exits,
        replaced.runs,
        replaced.notices
    );
    let reloaded = headless_reload("headless-reload", Rebuild::Build, false, false);
    assert_eq!(reloaded.exits, [ExitReason::Reload]);
}

/// Which output the run of the test below writes, in a copy of this test
/// binary: `json` or `plain`.
const OUTPUT_PROBE: &str = "RIG_REPLACED_OUTPUT_PROBE";

/// A `-p` run whose launcher was replaced from outside says so in what it
/// writes: with `--json`, a `warn` notice line, then its `exit` line with
/// code 0; without, the explanation on stderr. Each runs in a copy of this
/// test binary, whose stdout and stderr are the run's own.
#[test]
fn a_headless_run_whose_launcher_was_replaced_says_so_in_its_output() {
    const SAID: &str = "rig was replaced since this run started, so it does not reload; the \
                        next run starts the new version.";
    if let Some(output) = std::env::var_os(OUTPUT_PROBE) {
        headless_reload(
            &format!("output-{}", output.to_string_lossy()),
            Rebuild::Build,
            true,
            output == "json",
        );
        return;
    }
    let name =
        "reload::rebuild::tests::a_headless_run_whose_launcher_was_replaced_says_so_in_its_output";
    let run = |output: &str| {
        std::env::current_exe()
            .and_then(|test| {
                Command::new(test)
                    .args(["--exact", name, "--nocapture", "--test-threads", "1"])
                    .env(OUTPUT_PROBE, output)
                    .output()
            })
            .ok()
            .filter(|run| run.status.success())
    };
    let json = run("json").map(|run| String::from_utf8_lossy(&run.stdout).into_owned());
    let lines: Vec<serde_json::Value> = (json.iter())
        .flat_map(|stdout| stdout.lines())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let field = |line: &serde_json::Value, key: &str| line.get(key).cloned().unwrap_or_default();
    let notice = lines.iter().any(|line| {
        field(line, "type") == "notice"
            && field(line, "level") == "warn"
            && field(line, "text") == SAID
    });
    assert!(notice, "{json:?}");
    let last = lines.last();
    assert!(
        last.is_some_and(|line| field(line, "type") == "exit" && field(line, "code") == 0),
        "{json:?}"
    );
    let plain = run("plain").map(|run| String::from_utf8_lossy(&run.stderr).into_owned());
    assert!(
        plain.as_deref().is_some_and(|stderr| stderr.contains(SAID)),
        "{plain:?}"
    );
}
