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
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).ok();
    }
    home
}

#[test]
fn a_build_is_staged_with_its_report_and_one_that_cannot_describe_itself_is_rejected() {
    let report = "printf 'rig-describe 1\\ncommand\\tstats\\t-\\theadless\\tsession\\tCount\\n' \
                  > \"$RIG_DESCRIBE\"";
    let home = built("staged", report);
    let staged = stage(&home, &home.staged(), Staging::Always);
    let kept = good_report(&home, &home.staged());
    fs::remove_dir_all(home.root()).ok();
    assert!(staged.is_ok(), "{staged:?}");
    assert!(kept.is_some_and(|report| report.command("stats").is_some()));
    // Two plugins with one command: the describe exits 2, saying why.
    let clash = "echo 'rig: plugins `a` and `b` both add the command `stats`' >&2; exit 2";
    let home = built("clash", clash);
    let refused = stage(&home, &home.staged(), Staging::Always).map_err(|why| why.to_string());
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

#[test]
fn a_build_that_does_not_describe_itself_in_time_is_stopped() {
    let home = built("slow", "exec sleep 5");
    let artifact = home.target().join("debug").join(PACKAGE);
    let started = Instant::now();
    let child = child(&artifact).stderr(Stdio::piped()).spawn();
    let waited = child.map_err(Into::into).and_then(|child| {
        wait_described(
            child,
            &home.root().join("report"),
            Duration::from_millis(200),
        )
    });
    fs::remove_dir_all(home.root()).ok();
    let why = waited
        .map_err(|why| why.to_string())
        .err()
        .unwrap_or_default();
    assert!(why.contains("did not describe itself"), "{why}");
    assert!(started.elapsed() < Duration::from_secs(4));
}

/// The secret this file's tests run a copy of themselves with.
const PROBE: &str = "RIG_SECRET_PROBE";

#[test]
fn no_launcher_child_gets_a_secret() {
    if std::env::var_os(PROBE).is_none() {
        let name = "launcher::build::tests::no_launcher_child_gets_a_secret";
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
    let home = Home::at("home");
    for command in [cargo(&home), child("stty")] {
        let removed = command
            .get_envs()
            .any(|(name, value)| name == PROBE && value.is_none());
        assert!(removed, "{command:?}");
    }
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
    let mut log = BuildLog::create(&home, output, Box::new(shown.clone()));
    log.erase = erase;
    let mut cargo = Command::new("sh");
    cargo
        .args(["-c", "printf \"$0\" >&2", stderr])
        .stderr(Stdio::piped());
    let ran = log.run(cargo).map(|status| status.success());
    let told = log.fail("cargo could not build the agent".to_owned());
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

/// A short wait for cargo's lock in one command says nothing when a later
/// command of the same build is silent for long.
#[test]
fn a_wait_for_cargos_lock_ends_with_its_command() {
    let root = std::env::temp_dir().join(format!("rig-build-log-two-{}", std::process::id()));
    let home = Home::at(&root);
    let shown = Shown::default();
    let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(shown.clone()));
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
    let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(shown.clone()));
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
    let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(Shown::default()));
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
        let name = "launcher::build::tests::an_ignored_hangup_stays_ignored_through_a_build";
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
    let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(Shown::default()));
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
        let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(Shown::default()));
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
    let name = "launcher::build::tests::a_launcher_leading_its_session_keeps_cargo_in_its_group";
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
    let mut log = BuildLog::create(&home, BuildOutput::LogOnly, Box::new(Shown::default()));
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
    let mut log = BuildLog::create(&home, BuildOutput::Live, Box::new(shown.clone()));
    log.erase = true;
    for chunk in [&b"one\r"[..], b"\ntwo\r", b"three\n"] {
        log.show(chunk);
    }
    fs::remove_dir_all(home.root()).ok();
    assert_eq!(shown.text(), "one\r\ntwo\r\x1b[Kthree\n");
}
