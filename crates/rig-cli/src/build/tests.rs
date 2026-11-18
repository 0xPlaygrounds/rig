#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures run programs and touch files of their own"
)]

use std::os::unix::fs::PermissionsExt;

use super::*;

/// A home whose last build is the shell script `body`, and whose good
/// build is an older one.
fn built(name: &str, body: &str) -> Home {
    let root = std::env::temp_dir().join(format!("rig-describe-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let artifact = home
        .target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX));
    for (path, script) in [(&artifact, body), (&home.good(), "exit 0")] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(path, format!("#!/bin/sh\n{script}\n")).ok();
        fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).ok();
    }
    home
}

/// Stages `home`'s last build to `staged` as `rig build` does ([`stage`]).
fn stage_on(home: &Home, staged: &Path, staging: Staging) -> Result<()> {
    stage(
        &crate::test_launch(home, Path::new("/opt/rig/bin/rig"))?,
        staged,
        staging,
    )
}

/// The build at `binary`, as a describing build is run but by itself.
fn bare(binary: &Path) -> Command {
    Command::new(binary)
}

#[test]
fn a_build_is_staged_with_its_report_and_one_that_cannot_describe_itself_is_rejected() {
    let report = "printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' \
                  > \"$RIG_DESCRIBE\"";
    let home = built("staged", report);
    let staged = stage_on(&home, &home.staged(), Staging::Always);
    let kept = good_report(&home, &home.staged()).ok().flatten();
    fs::remove_dir_all(home.root()).ok();
    assert!(staged.is_ok(), "{staged:?}");
    assert!(kept.is_some_and(|report| report.command("stats").is_some()));
    // Two plugins with one command: the describe exits 2, saying why.
    let clash = "echo 'rig: plugins `a` and `b` both add the command `stats`' >&2; exit 2";
    let home = built("clash", clash);
    let refused = stage_on(&home, &home.staged(), Staging::Always).map_err(|why| why.to_string());
    let rejected = fs::read_to_string(home.rejected()).ok();
    let (good, staged) = (home.good().exists(), home.staged().exists());
    let artifact = home.target().join("debug").join(PACKAGE);
    let stamp = stamp(&artifact).ok();
    fs::remove_dir_all(home.root()).ok();
    let refused = refused.err().unwrap_or_default();
    assert!(
        refused.contains("both add the command `stats`"),
        "{refused}"
    );
    assert!(good && !staged);
    assert_eq!(rejected, stamp);
}

/// A build that cannot be staged says which step failed on which file,
/// not only the system's words: here the copy into the bin directory, where
/// a directory is in the way.
#[test]
fn a_build_that_cannot_be_staged_names_the_step_and_the_file() {
    let report = "printf 'rig-describe 1\\n' > \"$RIG_DESCRIBE\"";
    let home = built("unstaged", report);
    let in_the_way = home.bin().join("staged.tmp");
    fs::create_dir_all(&in_the_way).ok();
    let staged = stage_on(&home, &home.staged(), Staging::Always).map_err(|why| why.to_string());
    fs::remove_dir_all(home.root()).ok();
    let refused = staged.err().unwrap_or_default();
    let to = format!(" to {}: ", in_the_way.display());
    assert!(
        refused.starts_with("copy ") && refused.contains(&to),
        "{refused}"
    );
}

/// A new build is checked against the stamps of the builds already known.
/// A staged build whose stamp cannot be read, here a symlink loop, fails
/// the build naming the file and the step. A rejected stamp that cannot be
/// read, here a directory, is named when telling which build runs, and
/// staging goes on: it only keeps a crashed build from being retried.
#[test]
fn a_known_build_whose_stamp_cannot_be_read_is_named() {
    let report = "printf 'rig-describe 1\\n' > \"$RIG_DESCRIBE\"";
    let home = built("unreadable-stamp", report);
    let staged = home.bin().join("staged-for-session");
    let rejected = home.rejected();
    let made = fs::create_dir_all(&rejected);
    let passed = passed_over(&home, &staged).map_err(|why| why.to_string());
    let restaged = stage_on(&home, &staged, Staging::OnlyNew).map_err(|why| why.to_string());
    let restaged = restaged.map(|()| staged.exists());
    let looped = std::os::unix::fs::symlink(home.staged(), home.staged());
    let looping = stage_on(&home, &staged, Staging::OnlyNew).map_err(|why| why.to_string());
    fs::remove_dir_all(home.root()).ok();
    assert!(made.is_ok() && looped.is_ok());
    let read = format!("read {}: ", rejected.display());
    assert!(
        passed.as_ref().is_err_and(|why| why.starts_with(&read)),
        "{passed:?}"
    );
    assert_eq!(restaged, Ok(true));
    let look = format!("look at {}: ", home.staged().display());
    assert!(
        looping.as_ref().is_err_and(|why| why.starts_with(&look)),
        "{looping:?}"
    );
}

/// A start that builds the build a trial was rejected for stages nothing,
/// and is said to run the previous build, until a build replaces it.
#[test]
fn a_rebuilt_rejected_build_is_passed_over_out_loud() {
    let report = "printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' \
                  > \"$RIG_DESCRIBE\"";
    let home = built("passed-over", report);
    let staged = home.bin().join("staged-for-session");
    let before = passed_over(&home, &staged).ok();
    let recorded = stamp(&artifact(&home)).map(|stamp| fs::write(home.rejected(), stamp));
    let restaged = stage_on(&home, &staged, Staging::OnlyNew).map_err(|why| why.to_string());
    let after = passed_over(&home, &staged).ok();
    let retried = stage_on(&home, &staged, Staging::Always).is_ok()
        && passed_over(&home, &staged).is_ok_and(|passed| !passed);
    // The retried build started, so it is the good build that runs.
    let promoted = fs::rename(&staged, home.good()).is_ok()
        && passed_over(&home, &staged).is_ok_and(|passed| !passed);
    fs::remove_dir_all(home.root()).ok();
    assert!(matches!(recorded, Ok(Ok(()))), "{recorded:?}");
    assert_eq!(
        (before, restaged, after, retried, promoted),
        (Some(false), Ok(()), Some(true), true, true)
    );
}

#[test]
fn a_build_that_does_not_describe_itself_in_time_is_stopped() {
    let home = built("slow", "exec sleep 5");
    let artifact = home.target().join("debug").join(PACKAGE);
    let started = Instant::now();
    let waited = wait_described(
        bare(&artifact),
        &home.root().join("report"),
        Duration::from_millis(200),
    );
    fs::remove_dir_all(home.root()).ok();
    let why = waited
        .map_err(|why| why.to_string())
        .err()
        .unwrap_or_default();
    assert!(why.contains("did not describe itself"), "{why}");
    assert!(started.elapsed() < Duration::from_secs(4));
}

/// A build that leaves a process holding its stderr still ends the wait
/// within the timeout, and one that fails still says why.
#[test]
fn a_build_whose_child_holds_its_stderr_is_waited_for_within_the_timeout() {
    let run = |name: &str, body: &str, timeout: Duration| {
        let home = built(name, body);
        let artifact = home.target().join("debug").join(PACKAGE);
        let report = home.root().join("report");
        let started = Instant::now();
        let mut command = bare(&artifact);
        command.env(rig_harness_protocol::env::DESCRIBE, &report);
        let waited = wait_described(command, &report, timeout).map_err(|why| why.to_string());
        fs::remove_dir_all(home.root()).ok();
        (waited, started.elapsed())
    };
    let (described, took) = run(
        "held",
        "sleep 20 & printf 'rig-describe 1\\n' > \"$RIG_DESCRIBE\"; exit 0",
        Duration::from_secs(1),
    );
    assert_eq!(described.as_deref(), Ok("rig-describe 1\n"));
    assert!(took < Duration::from_secs(3), "{took:?}");
    let (failed, took) = run(
        "held-failing",
        "sleep 20 & echo 'rig: a plugin failed' >&2; exit 3",
        Duration::from_secs(10),
    );
    assert_eq!(failed, Err("rig: a plugin failed".to_owned()));
    assert!(took < Duration::from_secs(3), "{took:?}");
}

/// A launcher told to end while a build describes itself, as Ctrl-C ends
/// it, ends the build too.
#[cfg(target_os = "linux")]
#[test]
fn a_launcher_ended_during_a_describe_ends_the_build() {
    use std::os::unix::process::ExitStatusExt;

    const PROBE: &str = "RIG_DESCRIBE_INTERRUPT_PROBE";
    let seen = std::env::temp_dir().join(format!("rig-describe-interrupt-{}", std::process::id()));
    if let Some(seen) = std::env::var_os(PROBE) {
        let seen = std::path::PathBuf::from(seen);
        let home = built(
            "interrupted",
            &format!("echo $$ > '{}'; exec sleep 30", seen.display()),
        );
        let artifact = home.target().join("debug").join(PACKAGE);
        let watched = seen.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !watched.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(Duration::from_millis(50));
            // SAFETY: plain integers; this run of the test binary is its own.
            unsafe {
                libc::kill(libc::getpid(), libc::SIGINT);
            }
        });
        wait_described(
            bare(&artifact),
            &home.root().join("report"),
            Duration::from_secs(10),
        )
        .ok();
        return;
    }
    let name = "build::tests::a_launcher_ended_during_a_describe_ends_the_build";
    let launcher = std::env::current_exe().and_then(|test| {
        Command::new(test)
            .args(["--exact", name, "--test-threads", "1"])
            .env(PROBE, &seen)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    });
    let Ok(mut launcher) = launcher else {
        return assert!(launcher.is_ok());
    };
    let status = launcher.wait();
    let build = fs::read_to_string(&seen).unwrap_or_default();
    fs::remove_file(&seen).ok();
    let root = format!("rig-describe-interrupted-{}", launcher.id());
    fs::remove_dir_all(std::env::temp_dir().join(root)).ok();
    let alive = || {
        fs::read_to_string(format!("/proc/{}/stat", build.trim())).is_ok_and(|stat| {
            !stat
                .rsplit(')')
                .next()
                .unwrap_or_default()
                .trim_start()
                .starts_with('Z')
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let outlived = alive();
    if outlived && let Ok(pid) = build.trim().parse() {
        // SAFETY: plain integers; the build this test started.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
    assert_eq!(
        status.ok().and_then(|status| status.signal()),
        Some(libc::SIGINT)
    );
    assert!(
        !build.trim().is_empty() && !outlived,
        "the build outlived it"
    );
}

/// What a [`BuildLog`] shows, kept for the test to read.
#[derive(Clone, Default)]
struct Shown(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);

impl Write for Shown {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Shown {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.borrow()).into_owned()
    }
}

/// A build as `output` shows it, on a terminal when `erase`, of a fake
/// cargo that writes `stderr`: what it showed, what it logged, and its
/// failure's report.
fn fake_build(
    name: &str,
    (output, erase): (BuildOutput, bool),
    stderr: &str,
) -> (String, String, String) {
    let root = std::env::temp_dir().join(format!("rig-build-log-{name}-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let shown = Shown::default();
    let mut log = BuildLog::create(home.build_log(), output, Box::new(shown.clone()));
    log.erase = erase;
    let mut cargo = Command::new("sh");
    cargo
        .args(["-c", "printf \"$0\" >&2", stderr])
        .stderr(Stdio::piped());
    let ran = log.run(cargo).map(|status| status.success());
    let reason: Box<dyn std::error::Error> = "cargo could not build the agent".into();
    let told = log.fail(FailedAt::Build, &*reason);
    let logged = fs::read_to_string(home.build_log()).unwrap_or_default();
    fs::remove_dir_all(&root).ok();
    assert!(matches!(ran, Ok(true)), "{ran:?}");
    (shown.text(), logged, told.told(output))
}

#[test]
fn a_log_only_build_shows_one_line_while_cargo_works_and_its_errors_once_it_fails() {
    let cargo = "   Compiling a v1.0.0\n   Compiling b v1.0.0\nerror: boom\n";
    let (shown, logged, told) = fake_build("quiet", (BuildOutput::LogOnly, false), cargo);
    let lines: Vec<&str> = shown.lines().collect();
    let [line] = lines.as_slice() else {
        return assert_eq!(lines.len(), 1, "{shown:?}");
    };
    assert!(
        line.starts_with("rig: building the agent (output in "),
        "{line}"
    );
    for line in cargo.lines() {
        assert!(logged.contains(line), "{logged:?}");
    }
    // Nothing showed the errors, so the failure tells them, and the log.
    assert!(
        told.contains("error: boom") && told.contains("build.log"),
        "{told}"
    );
    // A build with nothing to do says nothing at all.
    let finished = "    Finished `dev` profile [unoptimized] target(s) in 0.20s\n";
    let (shown, logged, _) = fake_build("noop", (BuildOutput::LogOnly, false), finished);
    assert_eq!(shown, "");
    assert!(logged.contains("Finished"), "{logged:?}");
    // Nor does one that waited a moment for another cargo's lock.
    let blocked = format!("    Blocking waiting for file lock on package cache\n{finished}");
    let (shown, logged, _) = fake_build("blocked", (BuildOutput::LogOnly, false), &blocked);
    assert_eq!(shown, "");
    assert!(logged.contains("Blocking"), "{logged:?}");
}

/// A failure rig explains, such as a Bevy pin, points to the log but does
/// not repeat cargo's error after rig's words.
#[test]
fn a_failure_rig_explains_does_not_repeat_cargos_error() {
    let root = std::env::temp_dir().join(format!("rig-build-explained-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::Live,
        Box::new(Shown::default()),
    );
    let mut cargo = Command::new("sh");
    cargo
        .args([
            "-c",
            "echo 'error: failed to select a version for `bevy_ecs`.' >&2",
        ])
        .stderr(Stdio::piped());
    assert!(log.run(cargo).is_ok());
    let why = Explained("a plugin asks for a Bevy that is not this agent's.".to_owned());
    let failure = log.fail(FailedAt::Build, &why);
    fs::remove_dir_all(&root).ok();
    for told in [failure.to_string(), failure.told(BuildOutput::LogOnly)] {
        assert!(told.starts_with("a plugin asks for a Bevy"), "{told}");
        assert!(!told.contains("failed to select"), "{told}");
        assert!(told.contains("(whole output: "), "{told}");
    }
}

/// A short wait for cargo's lock in one command says nothing when a later
/// command of the same build is silent for long.
#[test]
fn a_wait_for_cargos_lock_ends_with_its_command() {
    let root = std::env::temp_dir().join(format!("rig-build-log-two-{}", std::process::id()));
    let home = Home::at(&root);
    let shown = Shown::default();
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::LogOnly,
        Box::new(shown.clone()),
    );
    let commands = [
        "printf '    Blocking waiting for file lock on build directory\\n' >&2",
        "sleep 4",
    ];
    for script in commands {
        let mut cargo = Command::new("sh");
        cargo.args(["-c", script]).stderr(Stdio::piped());
        assert!(log.run(cargo).is_ok_and(|status| status.success()));
    }
    fs::remove_dir_all(&root).ok();
    assert_eq!(shown.text(), "");
}

/// A wait for another cargo's lock that lasts is said once it passes its
/// time, while cargo still prints nothing.
#[test]
fn a_long_wait_for_cargos_lock_is_said_while_it_lasts() {
    let root = std::env::temp_dir().join(format!("rig-build-log-wait-{}", std::process::id()));
    let home = Home::at(&root);
    let shown = Shown::default();
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::LogOnly,
        Box::new(shown.clone()),
    );
    let mut cargo = Command::new("sh");
    let script = "printf '    Blocking waiting for file lock on build directory\\n' >&2; \
                  sleep 4; printf '    Finished in 4s\\n' >&2";
    cargo.args(["-c", script]).stderr(Stdio::piped());
    let started = Instant::now();
    let ran = log.run(cargo);
    fs::remove_dir_all(&root).ok();
    assert!(ran.is_ok_and(|status| status.success()));
    assert!(started.elapsed() >= Duration::from_secs(4));
    let text = shown.text();
    assert!(
        text.starts_with("rig: building the agent (output in "),
        "{text:?}"
    );
    assert_eq!(text.lines().count(), 1, "{text:?}");
}

/// cargo starts with no signal blocked, though the launcher blocks the
/// ending ones while it starts it.
#[cfg(target_os = "linux")]
#[test]
fn cargo_starts_with_no_signal_blocked() {
    let root = std::env::temp_dir().join(format!("rig-build-mask-{}", std::process::id()));
    let home = Home::at(&root);
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::LogOnly,
        Box::new(Shown::default()),
    );
    let mut cargo = Command::new("sh");
    cargo
        .args(["-c", "grep SigBlk /proc/self/status >&2"])
        .stderr(Stdio::piped());
    let ran = log.run(cargo);
    let logged = fs::read_to_string(home.build_log()).unwrap_or_default();
    fs::remove_dir_all(&root).ok();
    assert!(ran.is_ok_and(|status| status.success()));
    let blocked = logged
        .lines()
        .find_map(|line| line.trim().strip_prefix("SigBlk:"));
    assert_eq!(blocked.map(str::trim), Some("0000000000000000"), "{logged}");
}

/// A signal the launcher was started ignoring, as `nohup` starts it with
/// SIGHUP, stays ignored by cargo during a build and by the launcher after
/// it. Checked in a run of its own, so no other test shares the change.
#[cfg(target_os = "linux")]
#[test]
fn an_ignored_hangup_stays_ignored_through_a_build() {
    const PROBE: &str = "RIG_IGNORED_HANGUP_PROBE";
    if std::env::var_os(PROBE).is_none() {
        let name = "build::tests::an_ignored_hangup_stays_ignored_through_a_build";
        let output = std::env::current_exe().and_then(|test| {
            Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .env(PROBE, "1")
                .output()
        });
        let ran = output.is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
        });
        assert!(ran, "the run with SIGHUP ignored failed");
        return;
    }
    // SAFETY: plain calls; this run of the test binary is this test's own.
    unsafe {
        libc::signal(libc::SIGHUP, libc::SIG_IGN);
    }
    let root = std::env::temp_dir().join(format!("rig-build-hangup-{}", std::process::id()));
    let home = Home::at(&root);
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::LogOnly,
        Box::new(Shown::default()),
    );
    let mut cargo = Command::new("sh");
    cargo
        .args(["-c", "grep SigIgn /proc/self/status >&2"])
        .stderr(Stdio::piped());
    let ran = log.run(cargo);
    let logged = fs::read_to_string(home.build_log()).unwrap_or_default();
    fs::remove_dir_all(&root).ok();
    assert!(ran.is_ok_and(|status| status.success()));
    let ignored = logged
        .lines()
        .find_map(|line| line.trim().strip_prefix("SigIgn:"))
        .and_then(|mask| u64::from_str_radix(mask.trim(), 16).ok());
    assert!(
        ignored.is_some_and(|mask| mask & 1 << (libc::SIGHUP - 1) != 0),
        "{logged}"
    );
    // SAFETY: reads the action into a zeroed struct this test owns.
    let after = unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGHUP, std::ptr::null(), &raw mut action);
        action.sa_sigaction
    };
    assert_eq!(after, libc::SIG_IGN);
}

/// A launcher that leads a session of its own, as an agent's `/reload`
/// starts it, keeps cargo in its own group, which the agent kills as a
/// whole; told to end mid-build, it and cargo's children end by that
/// signal (143 for SIGTERM).
#[cfg(target_os = "linux")]
#[test]
fn a_launcher_leading_its_session_keeps_cargo_in_its_group() {
    use std::os::unix::process::ExitStatusExt;

    const PROBE: &str = "RIG_SESSION_LEADER_PROBE";
    let seen = std::env::temp_dir().join(format!("rig-build-leader-{}", std::process::id()));
    if let Some(seen) = std::env::var_os(PROBE) {
        let seen = std::path::PathBuf::from(seen);
        let root = seen.with_extension("home");
        let home = Home::at(&root);
        let mut log = BuildLog::create(
            home.build_log(),
            BuildOutput::LogOnly,
            Box::new(Shown::default()),
        );
        let mut cargo = Command::new("sh");
        // "cargo" ignores SIGTERM, from before it says it is ready; what it
        // started does not.
        let script = "trap '' TERM; (trap - TERM; exec sleep 30) & \
                      echo $(ps -o pgid= -p $$) $! $$ > \"$0\"; wait";
        cargo.args(["-c", script]).arg(&seen).stderr(Stdio::piped());
        let watched = seen.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !watched.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            std::thread::sleep(Duration::from_millis(50));
            // SAFETY: plain integers; this run of the test binary is its own.
            unsafe {
                libc::kill(libc::getpid(), libc::SIGTERM);
            }
        });
        log.run(cargo).ok();
        return;
    }
    let name = "build::tests::a_launcher_leading_its_session_keeps_cargo_in_its_group";
    let launcher = std::env::current_exe().and_then(|test| {
        Command::new("setsid")
            .arg(test)
            .args(["--exact", name, "--test-threads", "1"])
            .env(PROBE, &seen)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    });
    let Ok(mut launcher) = launcher else {
        return assert!(launcher.is_ok());
    };
    let status = launcher.wait();
    let ids = fs::read_to_string(&seen).unwrap_or_default();
    fs::remove_file(&seen).ok();
    fs::remove_dir_all(seen.with_extension("home")).ok();
    let mut ids = ids.split_whitespace();
    let (group, sleep, cargo) = (
        ids.next().unwrap_or_default(),
        ids.next().unwrap_or_default(),
        ids.next().unwrap_or_default(),
    );
    let alive = |pid: &str| {
        fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
            !stat
                .rsplit(')')
                .next()
                .unwrap_or_default()
                .trim_start()
                .starts_with('Z')
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while (alive(sleep) || alive(cargo)) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        status.ok().and_then(|status| status.signal()),
        Some(libc::SIGTERM)
    );
    assert_eq!(group, launcher.id().to_string());
    assert!(
        !sleep.is_empty() && !alive(sleep),
        "cargo's child outlived it"
    );
    assert!(!cargo.is_empty() && !alive(cargo), "cargo outlived it");
}

/// Stopping the build, as a signal that ends the launcher does, kills
/// cargo's whole group: what cargo started too, as rustc.
#[cfg(target_os = "linux")]
#[test]
fn a_stopped_build_leaves_nothing_running() {
    let root = std::env::temp_dir().join(format!("rig-build-stop-{}", std::process::id()));
    fs::create_dir_all(&root).ok();
    let home = Home::at(&root);
    let pid = root.join("compiler.pid");
    let mut log = BuildLog::create(
        home.build_log(),
        BuildOutput::LogOnly,
        Box::new(Shown::default()),
    );
    let mut cargo = Command::new("sh");
    let script = format!("sleep 60 & echo $! > '{}'; wait", pid.display());
    cargo.args(["-c", &script]).stderr(Stdio::piped());
    let watched = pid.clone();
    let stopper = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !watched.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(50));
        stop::kill();
    });
    let ran = log.run(cargo);
    assert!(stopper.join().is_ok());
    let compiler = fs::read_to_string(&pid).unwrap_or_default();
    let alive = || {
        fs::read_to_string(format!("/proc/{}/stat", compiler.trim())).is_ok_and(|stat| {
            !stat
                .rsplit(')')
                .next()
                .unwrap_or_default()
                .trim_start()
                .starts_with('Z')
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while alive() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    fs::remove_dir_all(&root).ok();
    assert!(ran.is_ok_and(|status| !status.success()));
    assert!(!compiler.trim().is_empty() && !alive());
}

#[test]
fn a_live_build_shows_cargo_as_it_writes_erasing_each_redrawn_line_on_a_terminal() {
    // The bar's tail outlived a shorter line after it: `… in 4.57s)`. A
    // line ended by `\r\n` stays.
    let cargo = "error: boom\r\n    Building [=> ] 1/2: a(bin)\r    Finished in 1s\n";
    let erased = "error: boom\r\n    Building [=> ] 1/2: a(bin)\r\x1b[K    Finished in 1s\n";
    let (shown, _, told) = fake_build("live", (BuildOutput::Live, true), cargo);
    assert_eq!(shown, erased);
    // The terminal showed the errors; the failure is one line.
    assert_eq!(told.lines().count(), 1, "{told}");
    // Redirected, it is passed on as it came.
    let (shown, _, _) = fake_build("redirected", (BuildOutput::Live, false), cargo);
    assert_eq!(shown, cargo);
}

#[test]
fn a_crlf_split_between_two_reads_is_not_erased() {
    let name = format!("rig-build-log-split-{}", std::process::id());
    let home = Home::at(std::env::temp_dir().join(name));
    let shown = Shown::default();
    let mut log = BuildLog::create(home.build_log(), BuildOutput::Live, Box::new(shown.clone()));
    log.erase = true;
    for chunk in [&b"one\r"[..], b"\ntwo\r", b"three\n"] {
        log.show(chunk);
    }
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(shown.text(), "one\r\ntwo\r\x1b[Kthree\n");
}

/// Bevy's own crates, as the agent project's lock lists them.
fn bevy() -> BTreeSet<String> {
    ["bevy", "bevy_app", "bevy_ecs", "bevy_reflect"]
        .map(str::to_owned)
        .into()
}

/// Writes `manifest` as the `Cargo.toml` of `dir`.
fn manifest_at(dir: &Path, manifest: &str) {
    fs::create_dir_all(dir).ok();
    fs::write(dir.join("Cargo.toml"), manifest).ok();
}

#[test]
fn a_plugins_bevy_requirement_the_agents_bevy_does_not_meet_is_named_with_its_table_and_line() {
    let root = std::env::temp_dir().join(format!("rig-bevy-pins-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let dirs = [
        "old",
        "ranged",
        "table",
        "target",
        "renamed",
        "workspace/member",
    ]
    .map(|name| root.join(name));
    let [old, ranged, table, target, renamed, member] = &dirs;
    manifest_at(
        old,
        "[package]\nname = \"old\"\n\n[dependencies]\nrig-harness = \"0.46.0\"\n\
         bevy_ecs = { version = \"=0.20.0\", default-features = false }\n\
         bevy_reflect = { version = \"=0.20.0\", default-features = false }\n",
    );
    // Met, or not Bevy's own: a crate of Bevy's ecosystem at its own version.
    manifest_at(
        ranged,
        "[package]\nname = \"ranged\"\n\n[dependencies]\n\
         bevy_ecs = { version = \"0.20\", default-features = false }\n\
         bevy_reflect = \"=0.20.1\"\nbevy_app = \"~0.20.0\"\nbevy_egui = \"0.33\"\n\n\
         [dev-dependencies]\nbevy = \"=0.19.0\"\n",
    );
    manifest_at(
        table,
        "[package]\nname = \"table\"\n\n[dependencies.bevy_app]\nversion = \"0.19\"\n",
    );
    // A met requirement first, the pin in a target's table after it.
    manifest_at(
        target,
        "[package]\nname = \"target\"\n\n[dependencies]\nbevy_ecs = \"0.20\"\n\n\
         [target.'cfg(unix)'.dependencies]\nbevy_ecs = \"=0.20.0\"\n",
    );
    manifest_at(
        renamed,
        "[package]\nname = \"renamed\"\n\n[dependencies]\n\
         ecs = { package = \"bevy_ecs\", version = \"=0.20.0\" }\n",
    );
    manifest_at(
        &root.join("workspace"),
        "[workspace]\nmembers = [\"member\"]\n\n[workspace.dependencies]\n\
         bevy_reflect = \"=0.20.0\"\n",
    );
    manifest_at(
        member,
        "[package]\nname = \"member\"\n\n[dependencies]\nbevy_reflect = { workspace = true }\n",
    );
    let pins = bevy_pins(dirs.iter().map(PathBuf::as_path), &bevy());
    let file = |dir: &Path| dir.join("Cargo.toml").display().to_string();
    assert_eq!(
        pins,
        [
            format!(
                "{}:6: [dependencies] bevy_ecs = {{ version = \"=0.20.0\", default-features = false }}",
                file(old)
            ),
            format!(
                "{}:7: [dependencies] bevy_reflect = {{ version = \"=0.20.0\", default-features = false }}",
                file(old)
            ),
            format!(
                "{}:5: [dependencies.bevy_app] version = \"0.19\"",
                file(table)
            ),
            format!(
                "{}:8: [target.'cfg(unix)'.dependencies] bevy_ecs = \"=0.20.0\"",
                file(target)
            ),
            format!(
                "{}:5: [dependencies] ecs = {{ package = \"bevy_ecs\", version = \"=0.20.0\" }}",
                file(renamed)
            ),
            format!(
                "{}:5: [workspace.dependencies] bevy_reflect = \"=0.20.0\"",
                file(&root.join("workspace"))
            ),
        ]
    );
    fs::remove_dir_all(&root).ok();
}

/// Bevy's own crates are the ones a plugin names, and each `bevy_*` crate
/// the lock has at its `bevy_ecs`'s version, whatever that is: a home
/// upgraded from 0.46 still has 0.20.0 there, and a first build none.
#[test]
fn bevys_own_crates_are_named_and_read_from_the_lock_at_any_bevy_version() {
    let lock = "version = 4\n\n[[package]]\nname = \"bevy_ecs\"\nversion = \"0.20.0\"\n\n\
                [[package]]\nname = \"bevy_render\"\nversion = \"0.20.0\"\n\n\
                [[package]]\nname = \"bevy_egui\"\nversion = \"0.33.0\"\n\n\
                [[package]]\nname = \"serde\"\nversion = \"0.20.0\"\n";
    let named = ["bevy", "bevy_app", "bevy_ecs", "bevy_reflect"].map(str::to_owned);
    let mut upgraded = BTreeSet::from(named.clone());
    upgraded.insert("bevy_render".to_owned());
    assert_eq!(bevy_crates(lock), upgraded);
    assert_eq!(bevy_crates(""), BTreeSet::from(named));
}

/// A member names its workspace with `[package] workspace`, a sibling
/// directory here, before any directory above it.
#[test]
fn an_inherited_bevy_pin_is_found_in_the_workspace_the_package_names() {
    let root = std::env::temp_dir().join(format!("rig-bevy-ws-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let (ws, member) = (root.join("ws"), root.join("member"));
    manifest_at(
        &root,
        "[workspace]\n\n[workspace.dependencies]\nbevy_ecs = \"0.20\"\n",
    );
    manifest_at(
        &ws,
        "[workspace]\nmembers = [\"../member\"]\n\n[workspace.dependencies]\nbevy_ecs = \"=0.20.0\"\n",
    );
    manifest_at(
        &member,
        "[package]\nname = \"member\"\nworkspace = \"../ws\"\n\n[dependencies]\n\
         bevy_ecs.workspace = true\n",
    );
    let pins = bevy_pins([member.as_path()], &bevy());
    fs::remove_dir_all(&root).ok();
    assert_eq!(
        pins,
        [format!(
            "{}:5: [workspace.dependencies] bevy_ecs = \"=0.20.0\"",
            ws.join("Cargo.toml").display()
        )]
    );
}

/// A cached safe build runs with no cargo only from crates.io, with this
/// launcher's stamp, and while it still writes a description report: a
/// damaged copy (not executable, truncated, or empty, which runs as a
/// shell script that exits 0), or one built from a checkout, which may
/// have changed since, is built again.
#[test]
fn a_safe_build_is_reused_only_from_crates_io_while_it_describes_itself() {
    let root = std::env::temp_dir().join(format!("rig-safe-cache-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let describing = "#!/bin/sh\n[ -n \"$RIG_DESCRIBE\" ] || exit 0\nprintf 'rig-describe 1\\n' > \"$RIG_DESCRIBE\"\n";
    let current = (|| -> Result<Vec<bool>> {
        let launch = crate::test_launch(&home, Path::new("/opt/rig/bin/rig"))?;
        let checkout = RigSource::Local(root.clone());
        fs::create_dir_all(home.bin())?;
        fs::write(safe(&home), describing)?;
        fs::set_permissions(safe(&home), std::fs::Permissions::from_mode(0o755))?;
        fs::write(safe_version(&home), safe_stamp(&RigSource::Registry))?;
        let mut current = vec![safe_is_current(&launch, &RigSource::Registry)];
        fs::write(safe_version(&home), safe_stamp(&checkout))?;
        current.push(safe_is_current(&launch, &checkout));
        fs::write(safe_version(&home), safe_stamp(&RigSource::Registry))?;
        fs::set_permissions(safe(&home), std::fs::Permissions::from_mode(0o644))?;
        current.push(safe_is_current(&launch, &RigSource::Registry));
        fs::write(safe(&home), "\x7fELF")?;
        fs::set_permissions(safe(&home), std::fs::Permissions::from_mode(0o755))?;
        current.push(safe_is_current(&launch, &RigSource::Registry));
        fs::write(safe(&home), "")?;
        current.push(safe_is_current(&launch, &RigSource::Registry));
        Ok(current)
    })();
    fs::remove_dir_all(&root).ok();
    let current = current.map_err(|why| why.to_string());
    assert_eq!(current, Ok(vec![true, false, false, false, false]));
}

/// The lock files a safe build changed are kept on disk, so after a safe
/// build that a signal stopped, the next build of the user's plugins puts
/// them back as they were before it builds (here it then fails on its
/// settings), removing one that was not there.
#[test]
fn a_stopped_safe_build_leaves_the_projects_lock_files_for_the_next_build() {
    let root = std::env::temp_dir().join(format!("rig-safe-locks-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let after = (|| -> Result<_> {
        fs::create_dir_all(home.project())?;
        fs::write(home.project().join("Cargo.lock"), "user's lock")?;
        fs::write(home.project().join("Cargo.lock.seed"), "commit A's lock")?;
        keep_locks(&home)?;
        for name in LOCKS {
            fs::write(home.project().join(name), "the safe build's")?;
        }
        fs::write(
            home.settings(),
            "[plugins]\nenable = [\"rig_basics::gone::GonePlugin\"]\n",
        )?;
        let launch = crate::test_launch(&home, Path::new("/opt/rig/bin/rig"))?;
        let built = compile(
            &launch,
            &home.staged(),
            Staging::Always,
            BuildOutput::LogOnly,
        );
        let locks = LOCKS.map(|name| fs::read_to_string(home.project().join(name)).ok());
        Ok((built.is_err(), locks, kept_locks(&home).exists()))
    })();
    fs::remove_dir_all(&root).ok();
    let after = after.map_err(|why| why.to_string());
    assert_eq!(
        after,
        Ok((
            true,
            [
                Some("user's lock".to_owned()),
                Some("commit A's lock".to_owned()),
                None
            ],
            false
        ))
    );
}

/// A safe build that fails logs to its own log, and the failed build's
/// `build.log` stays as it was.
#[test]
fn a_safe_build_logs_apart_from_build_log() {
    let root = std::env::temp_dir().join(format!("rig-safe-log-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let logs = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        fs::write(home.build_log(), "the broken build's output\n")?;
        let mut launch = crate::test_launch(&home, Path::new("/opt/rig/bin/rig"))?;
        launch.source = Some(rig_harness_protocol::launch::AbsPath::new(
            root.join("not-a-checkout"),
        )?);
        let built = compile_safe(&launch, BuildOutput::LogOnly).map_err(|failure| failure.log);
        Ok((
            built,
            fs::read_to_string(home.build_log())?,
            fs::read_to_string(safe_log(&home))?,
        ))
    })();
    fs::remove_dir_all(&root).ok();
    let logs = logs.map_err(|why| why.to_string());
    let Ok((built, build_log, safe_build_log)) = logs else {
        return assert!(logs.is_ok(), "{logs:?}");
    };
    assert_eq!(built, Err(safe_log(&home)));
    assert_eq!(build_log, "the broken build's output\n");
    assert!(
        safe_build_log.contains("is not a rig checkout"),
        "{safe_build_log}"
    );
}

/// Keeping the lock files of a home that has no agent project yet makes
/// none, so the build that follows still sees a new home and sets up the
/// plugin-writing skill ([`super::super::plugin::skill::set_up`]).
#[test]
fn keeping_the_lock_files_of_a_new_home_makes_no_project() {
    let root = std::env::temp_dir().join(format!("rig-safe-new-home-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let kept = (|| -> Result<_> {
        fs::create_dir_all(home.root())?;
        keep_locks(&home)?;
        let project = home.project().exists();
        put_back(&home)?;
        Ok((project, home.project().exists()))
    })();
    fs::remove_dir_all(&root).ok();
    assert_eq!(kept.map_err(|why| why.to_string()), Ok((false, false)));
}

/// A kill while a used backup is being removed leaves part of it. That
/// part is never read as a backup: the next build leaves the user's
/// restored lock files alone.
#[test]
fn a_partly_removed_backup_is_not_read_as_one() {
    let root = std::env::temp_dir().join(format!("rig-safe-partial-{}", std::process::id()));
    fs::remove_dir_all(&root).ok();
    let home = Home::at(&root);
    let after = (|| -> Result<_> {
        fs::create_dir_all(home.project())?;
        fs::write(home.project().join("Cargo.lock"), "user's lock")?;
        fs::write(home.project().join("Cargo.lock.seed"), "commit A's lock")?;
        keep_locks(&home)?;
        // As `put_back` leaves it when killed while it removes the backup
        // it restored from: retired, its kept Cargo.lock already gone.
        fs::rename(kept_locks(&home), retired_locks(&home))?;
        fs::remove_file(retired_locks(&home).join("Cargo.lock"))?;
        put_back(&home)?;
        if retired_locks(&home).exists() {
            return Err("the retired backup is still there".into());
        }
        Ok(fs::read_to_string(home.project().join("Cargo.lock")).ok())
    })();
    fs::remove_dir_all(&root).ok();
    assert_eq!(
        after.map_err(|why| why.to_string()),
        Ok(Some("user's lock".to_owned()))
    );
}

#[test]
fn a_plugin_on_another_bevy_is_told_first_which_version_to_write() {
    let message = foreign_bevy("my_plugin", "MyPlugin", "0.19.0");
    let first = message.split(". ").next().unwrap_or_default();
    let range = crate::bevy_range();
    assert!(
        first.contains(&format!("\"{range}\"")) && first.contains("my_plugin"),
        "{message}"
    );
    assert!(message.contains("0.19.0") && message.contains("rig plugin remove MyPlugin"));
}

#[test]
fn a_plugin_pinning_another_bevy_is_told_first_which_version_to_write() {
    let pin = "plugin/Cargo.toml:7: [dependencies] bevy_app = \"=0.20.0\"".to_owned();
    let message = pinned_bevy(&"cargo could not resolve", std::slice::from_ref(&pin));
    let first = message.lines().next().unwrap_or_default();
    let range = crate::bevy_range();
    assert!(first.contains(&format!("\"{range}\"")), "{message}");
    assert!(
        message.contains(&pin)
            && message.contains("cargo could not resolve")
            && message.contains(BEVY_VERSION)
            && message.contains("rig plugin remove <type>"),
        "{message}"
    );
}
