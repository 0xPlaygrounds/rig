use std::time::{Duration, Instant};

use super::{Launcher, OutputLine, Spawn, piece_end};

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
            &OutputLine::Out("one\n".to_owned()),
            &OutputLine::Out("two\n".to_owned())
        ]
    );
    assert!(lines.contains(&OutputLine::Err("err\n".to_owned())));
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
