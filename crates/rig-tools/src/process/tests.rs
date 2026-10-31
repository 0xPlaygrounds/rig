use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use super::{Ended, LEDGER_VARIABLE, Launcher, OutputLine, Spawn, piece_end, record_groups};

/// What a process reads on stdin reaches it, and its stdout and stderr
/// lines arrive tagged, each with its `\n`.
#[cfg(unix)]
#[test]
fn stdin_reaches_the_child_and_its_lines_arrive_tagged() {
    let how = Spawn {
        stdin: Some(b"one\ntwo".to_vec()),
        merge_stderr: false,
    };
    let process = Launcher::default().spawn("sh", ["-c", "cat; echo; echo err >&2"], how);
    let mut process = process.ok();
    let status = process
        .as_mut()
        .and_then(|process| process.wait_for(Duration::from_secs(5)).ok().flatten());
    assert!(status.is_some_and(|status| status.success()));
    let lines: Vec<OutputLine> = process
        .as_ref()
        .map(|process| process.lines().iter().collect())
        .unwrap_or_default();
    let out: Vec<&OutputLine> = lines
        .iter()
        .filter(|line| matches!(line, OutputLine::Out(_)))
        .collect();
    assert_eq!(
        out,
        [
            &OutputLine::Out(b"one\n".to_vec()),
            &OutputLine::Out(b"two\n".to_vec())
        ]
    );
    assert!(lines.contains(&OutputLine::Err(b"err\n".to_vec())));
}

/// `stop` asks the group to end with SIGTERM, which ends a command that
/// heeds it long before the grace runs out.
#[cfg(unix)]
#[test]
fn stop_ends_a_child_with_sigterm_before_the_grace() {
    use std::os::unix::process::ExitStatusExt;

    let process = Launcher::default().spawn("sleep", ["30"], Spawn::default());
    let mut process = process.ok();
    let started = Instant::now();
    let status = process
        .as_mut()
        .and_then(|process| process.stop(Duration::from_secs(10)).ok());
    assert_eq!(
        status.and_then(|status| status.signal()),
        Some(libc::SIGTERM)
    );
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// Waiting with no deadline at all, `Duration::MAX`, waits until the
/// process ends instead of overflowing the clock.
#[cfg(unix)]
#[test]
fn waiting_for_ever_returns_when_the_process_ends() {
    let mut process = Launcher::default()
        .spawn("true", [""; 0], Spawn::default())
        .ok();
    let status = process
        .as_mut()
        .and_then(|process| process.wait_for(Duration::MAX).ok().flatten());
    assert!(status.is_some_and(|status| status.success()));
}

/// Running a process to its end gives its stdout, its stderr and how it
/// ended, apart.
#[cfg(unix)]
#[test]
fn output_runs_a_process_to_its_end() {
    let process =
        Launcher::default().spawn("sh", ["-c", "echo a; echo b >&2; exit 3"], Spawn::default());
    let output = process
        .ok()
        .and_then(|process| process.output(Some(Duration::from_secs(10))).ok());
    assert_eq!(
        output.map(|output| (output.stdout, output.stderr, output.status.code())),
        Some((b"a\n".to_vec(), b"b\n".to_vec(), Some(3)))
    );
}

/// A sink that says stop ends the wait and kills the group, though the
/// process would run on. The process cannot end by itself first, so every
/// run of this test takes the same path through the wait.
#[cfg(unix)]
#[test]
fn a_sink_that_stops_ends_the_wait_and_kills_the_process() {
    let how = Spawn {
        merge_stderr: true,
        ..Spawn::default()
    };
    let process = Launcher::default().spawn("sh", ["-c", "echo ready; exec sleep 30"], how);
    let mut process = process.ok();
    let started = Instant::now();
    let mut seen = Vec::new();
    let ended = process.as_mut().and_then(|process| {
        process
            .run(None, |line| {
                seen.push(line);
                false
            })
            .ok()
    });
    assert_eq!(ended, Some(Ended::Stopped));
    assert_eq!(seen, [OutputLine::Out(b"ready\n".to_vec())]);
    assert!(started.elapsed() < Duration::from_secs(10));
    let status = process
        .as_mut()
        .and_then(|process| process.try_wait().ok().flatten());
    assert!(status.is_some_and(|status| !status.success()));
}

/// A process that closes its output but runs on is still waited for: the
/// wait ends at the deadline, not when the output closes, and kills it.
/// The test sees the output close before the wait starts, so the wait
/// always meets a closed pipe and a running process.
#[cfg(unix)]
#[test]
fn a_process_that_closes_its_output_is_waited_for_until_the_deadline() {
    let process = Launcher::default().spawn(
        "sh",
        ["-c", "exec >&- 2>&-; exec sleep 30"],
        Spawn::default(),
    );
    let mut process = process.ok();
    let closed = process
        .as_ref()
        .map(|process| process.lines().recv_timeout(Duration::from_secs(10)));
    assert_eq!(closed, Some(Err(RecvTimeoutError::Disconnected)));
    let deadline = Instant::now() + Duration::from_millis(200);
    let ended = process
        .as_mut()
        .and_then(|process| process.run(Some(deadline), |_| true).ok());
    assert_eq!(ended, Some(Ended::TimedOut));
    assert!(Instant::now() >= deadline);
    let status = process
        .as_mut()
        .and_then(|process| process.try_wait().ok().flatten());
    assert!(status.is_some_and(|status| !status.success()));
}

/// Dropping a running process kills its whole group, also a process it
/// started in the background.
#[cfg(target_os = "linux")]
#[test]
fn dropping_a_process_kills_its_group() {
    let how = Spawn {
        merge_stderr: true,
        ..Spawn::default()
    };
    let process = Launcher::default().spawn("sh", ["-c", "sleep 30 & echo $!; wait"], how);
    let grandchild = process.as_ref().ok().and_then(|process| {
        let line = process.lines().recv_timeout(Duration::from_secs(5)).ok()?;
        line.text().trim().parse::<u32>().ok()
    });
    assert!(grandchild.is_some());
    drop(process);
    // Dead once it is gone or a zombie waiting for its new parent.
    let alive = |pid: u32| {
        std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
            !stat
                .rsplit(')')
                .next()
                .unwrap_or_default()
                .trim_start()
                .starts_with('Z')
        })
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while grandchild.is_some_and(alive) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!grandchild.is_some_and(alive));
}

/// The ledger lists a spawned process's group while it runs and drops it
/// once it is reaped, whether it ended or was killed; each process carries
/// the ledger's path.
#[cfg(unix)]
#[test]
fn the_ledger_lists_each_group_until_it_is_reaped() {
    let path = std::env::temp_dir().join(format!("rig-groups-{}", std::process::id()));
    record_groups(&path, []);
    // Other tests of this process may list their own groups too.
    let listed = |id: u32| {
        let ids = std::fs::read_to_string(&path).unwrap_or_default();
        ids.lines().any(|line| line == id.to_string())
    };
    let marked = Launcher::default().command("true");
    let carried = marked.get_envs().find(|(name, _)| *name == LEDGER_VARIABLE);
    assert_eq!(
        carried,
        Some((LEDGER_VARIABLE.as_ref(), Some(path.as_os_str())))
    );
    let killed = Launcher::default().spawn("sleep", ["30"], Spawn::default());
    let ended = Launcher::default().spawn("true", [""; 0], Spawn::default());
    let ids = killed.as_ref().ok().zip(ended.as_ref().ok());
    let ids = ids.map(|(killed, ended)| (killed.id(), ended.id()));
    assert!(ids.is_some());
    let Some((killed_id, ended_id)) = ids else {
        return;
    };
    assert!(listed(killed_id) && listed(ended_id));
    let status = ended.and_then(|mut ended| ended.wait_for(Duration::from_secs(5)));
    assert!(status.is_ok_and(|status| status.is_some()));
    assert!(listed(killed_id) && !listed(ended_id));
    drop(killed);
    assert!(!listed(killed_id));
    std::fs::remove_file(&path).ok();
}

/// A line ends after its `\n`; a line too long for one piece is cut, but
/// never inside a character.
#[test]
fn a_piece_ends_after_a_newline_or_before_a_character() {
    assert_eq!(piece_end(b"one\ntwo"), Some(4));
    assert_eq!(piece_end(b"no newline yet"), None);
    let mut long = vec![b'a'; super::MAX_PIECE - 1];
    long.extend_from_slice("é".as_bytes());
    assert_eq!(piece_end(&long), Some(super::MAX_PIECE - 1));
}

/// Whether the process `pid` runs: there and not a zombie waiting for its
/// parent.
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        !stat
            .rsplit(')')
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('Z')
    })
}

/// Waits at most `limit` for `done`; whether it came.
#[cfg(target_os = "linux")]
fn within(limit: Duration, done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + limit;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    done()
}

/// The first line `process` prints, as a pid.
#[cfg(target_os = "linux")]
fn printed_pid(process: &super::Process) -> Option<u32> {
    let line = process.lines().recv_timeout(Duration::from_secs(5)).ok()?;
    line.text().trim().parse().ok()
}

/// A job a command leaves running in the background with its output
/// elsewhere outlives the command and the call, and its group stays in
/// the ledger, for the launcher to end with rig, until the job ends.
#[cfg(target_os = "linux")]
#[test]
fn a_job_left_in_the_background_stays_listed_until_it_ends() {
    let path = std::env::temp_dir().join(format!("rig-left-{}", std::process::id()));
    record_groups(&path, []);
    let listed = |id: u32| {
        let ids = std::fs::read_to_string(&path).unwrap_or_default();
        ids.lines().any(|line| line == id.to_string())
    };
    let script = "sleep 2 >/dev/null 2>&1 & echo $!";
    let process = Launcher::default().spawn("sh", ["-c", script], Spawn::default());
    let Ok(mut process) = process else {
        return assert!(process.is_ok());
    };
    let (group, job) = (process.id(), printed_pid(&process));
    let ended = process.run(None, |_| true);
    assert!(matches!(ended, Ok(Ended::Exited(_))), "{ended:?}");
    drop(process);
    assert!(job.is_some_and(alive), "the job died with its command");
    assert!(
        listed(group),
        "the job's group left the ledger with its leader"
    );
    assert!(within(Duration::from_secs(6), || !listed(group)));
    assert!(!job.is_some_and(alive));
    std::fs::remove_file(&path).ok();
}

/// A stopped scope stops the processes started in it, also a member of
/// a group that ignores `SIGTERM` after its leader ended on it, and
/// refuses every process started in it after.
#[cfg(target_os = "linux")]
#[test]
fn a_stopped_scope_stops_its_processes_and_starts_no_more() {
    let scope = super::Scope::default();
    let ready = std::env::temp_dir().join(format!("rig-stubborn-{}", std::process::id()));
    // The member says its id once it ignores SIGTERM; its output is
    // elsewhere, so only the scope can end it.
    let script = format!(
        "sh -c 'trap \"\" TERM; echo $$ > \"$0\"; exec sleep 30' '{}' >/dev/null 2>&1 & \
         exec sleep 30",
        ready.display()
    );
    let started =
        scope.enter(|| Launcher::default().spawn("sh", ["-c", &script], Spawn::default()));
    let Ok(mut process) = started else {
        return assert!(started.is_ok());
    };
    assert!(within(Duration::from_secs(5), || ready.exists()));
    std::thread::sleep(Duration::from_millis(50));
    let stubborn = std::fs::read_to_string(&ready).ok();
    let stubborn = stubborn.and_then(|pid| pid.trim().parse::<u32>().ok());
    std::fs::remove_file(&ready).ok();
    assert!(stubborn.is_some_and(alive));
    // The thread that waits for it goes on reaping it, as a call's does.
    let waiting = std::thread::spawn(move || process.run(None, |_| true));
    scope.stop();
    assert!(within(super::STOP_GRACE * 3, || !stubborn.is_some_and(alive)));
    assert!(waiting.join().is_ok_and(|ended| ended.is_ok()));
    let late = scope.enter(|| Launcher::default().spawn("true", [""; 0], Spawn::default()));
    assert!(late.is_err_and(|error| error.kind() == std::io::ErrorKind::Interrupted));
}

/// Once the sink stops taking output, it gets no more, also from output
/// still waiting after the process exited: a producer of 12 MiB that ends
/// at once is cut where the sink said.
#[cfg(unix)]
#[test]
fn a_sink_that_stops_after_the_exit_gets_nothing_more() {
    const LIMIT: usize = 1024 * 1024;
    let script = "head -c 12582912 /dev/zero | tr '\\000' x";
    let process = Launcher::default().spawn("sh", ["-c", script], Spawn::default());
    let Ok(mut process) = process else {
        return assert!(process.is_ok());
    };
    let mut taken = 0;
    let ended = process.run(None, |line| {
        if taken == 0 {
            // The producer ends meanwhile, its output still queued.
            std::thread::sleep(Duration::from_millis(500));
        }
        taken += line.bytes().len();
        taken <= LIMIT
    });
    assert!(matches!(ended, Ok(Ended::Stopped)), "{ended:?}");
    assert!(taken <= LIMIT + super::MAX_PIECE, "{taken} bytes taken");
}

/// Output that comes past the deadline is not taken, also once the
/// process exited with it still queued: the wait ends as timed out.
#[cfg(unix)]
#[test]
fn output_past_the_deadline_after_the_exit_is_a_timeout() {
    let script = "head -c 12582912 /dev/zero | tr '\\000' x";
    let process = Launcher::default().spawn("sh", ["-c", script], Spawn::default());
    let Ok(mut process) = process else {
        return assert!(process.is_ok());
    };
    let mut first = true;
    let deadline = Instant::now() + Duration::from_millis(200);
    let ended = process.run(Some(deadline), |_| {
        if std::mem::take(&mut first) {
            // The producer ends meanwhile, its output still queued.
            std::thread::sleep(Duration::from_millis(500));
        }
        true
    });
    assert!(matches!(ended, Ok(Ended::TimedOut)), "{ended:?}");
}

/// `output` stops at its timeout though output keeps coming.
#[cfg(unix)]
#[test]
fn output_stops_at_its_timeout_while_output_flows() {
    let flowing = "while :; do echo x; done";
    let process = Launcher::default().spawn("sh", ["-c", flowing], Spawn::default());
    let started = Instant::now();
    let output = process.and_then(|process| process.output(Some(Duration::from_millis(200))));
    let kind = output.err().map(|error| error.kind());
    assert_eq!(kind, Some(std::io::ErrorKind::TimedOut));
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// `output` keeps every byte as it was printed, not only text.
#[cfg(unix)]
#[test]
fn output_keeps_every_byte() {
    let process = Launcher::default().spawn("printf", ["a\\000\\377b"], Spawn::default());
    let output = process.and_then(|process| process.output(Some(Duration::from_secs(5))));
    assert!(output.is_ok_and(|output| output.stdout == b"a\0\xffb"));
}

/// A process handed over is off the ledger, and what it left running
/// outlives a stop of its scope.
#[cfg(target_os = "linux")]
#[test]
fn a_released_process_is_not_stopped_with_its_scope() {
    let path = std::env::temp_dir().join(format!("rig-released-{}", std::process::id()));
    record_groups(&path, []);
    let scope = super::Scope::default();
    let script = "sleep 30 >/dev/null 2>&1 & echo $!";
    let started = scope.enter(|| Launcher::default().spawn("sh", ["-c", script], Spawn::default()));
    let Ok(mut process) = started else {
        return assert!(started.is_ok());
    };
    let (group, server) = (process.id(), printed_pid(&process));
    let exited = process.wait_for(Duration::from_secs(5));
    assert!(exited.is_ok_and(|status| status.is_some()));
    process.release();
    drop(process);
    scope.stop();
    super::end_left();
    std::thread::sleep(Duration::from_millis(200));
    let listed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(!listed.lines().any(|line| line == group.to_string()));
    assert!(server.is_some_and(alive));
    if let Some(server) = server.and_then(|pid| i32::try_from(pid).ok()) {
        // SAFETY: plain integers; the test's own `sleep`.
        unsafe { libc::kill(server, libc::SIGKILL) };
    }
    std::fs::remove_file(&path).ok();
}

/// A program a command starts finds no editor and no pager: `git commit
/// --amend` fails at once saying what to pass instead of opening one.
#[cfg(unix)]
#[test]
fn a_command_finds_no_editor() {
    let dir = std::env::temp_dir().join(format!("rig-no-editor-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let script = "git init -q . && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m one \
                  && git -c user.name=t -c user.email=t@t commit --amend --allow-empty";
    let launcher = Launcher::default()
        .in_dir(&dir)
        .with_env("GIT_CONFIG_GLOBAL", "/dev/null")
        .with_env("GIT_CONFIG_NOSYSTEM", "1");
    std::fs::create_dir_all(&dir).ok();
    let started = Instant::now();
    let output = launcher
        .spawn("sh", ["-c", script], Spawn::default())
        .and_then(|process| process.output(Some(Duration::from_secs(20))));
    std::fs::remove_dir_all(&dir).ok();
    let Ok(output) = output else {
        return assert!(output.is_ok());
    };
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        said.contains("no editor here: pass -m or --no-edit"),
        "{said}"
    );
    assert!(started.elapsed() < Duration::from_secs(10));
}
