use std::time::{Duration, Instant};

/// A SIGTERM to the launcher reaches its agent, and the launcher's wait
/// ends only once the agent ended, with the agent's own status. It raises
/// SIGTERM in its own process, so it needs one of its own, as nextest
/// gives each test.
#[cfg(unix)]
#[test]
fn a_signal_is_passed_on_and_the_agent_is_waited_for() {
    use std::process::Command;

    use super::{Watch, heard};

    let exe = std::env::current_exe().unwrap_or_default();
    let ready = exe.with_file_name(format!("rig-stop-ready-{}", std::process::id()));
    let script = format!(
        "trap 'sleep 0.3; exit 7' TERM; : > '{}'; while :; do sleep 0.05; done",
        ready.display()
    );
    let agent = Command::new("sh").args(["-c", &script]).spawn();
    assert!(agent.is_ok(), "sh starts");
    let Ok(mut agent) = agent else {
        return;
    };
    // The agent says it is ready once its trap is set.
    let since = Instant::now();
    while !ready.exists() && since.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(ready.exists(), "the agent set its trap");
    std::fs::remove_file(&ready).ok();
    let mut watch = Watch::start();
    // SAFETY: the watch's handler only notes the signal.
    unsafe {
        libc::raise(libc::SIGTERM);
    }
    let since = Instant::now();
    let status = loop {
        watch.pass_on(&mut agent);
        if let Ok(Some(status)) = agent.try_wait() {
            break Some(status);
        }
        if since.elapsed() > Duration::from_secs(5) {
            agent.kill().ok();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(heard(), Some(libc::SIGTERM));
    assert_eq!(status.and_then(|status| status.code()), Some(7));
    assert!(since.elapsed() >= Duration::from_millis(250));
}

/// The closing output after a signal is bounded: a stderr whose reader
/// stopped never holds the launcher.
#[test]
fn closing_output_that_blocks_is_bounded() {
    let since = Instant::now();
    super::bounded(|| {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    });
    assert!(since.elapsed() < Duration::from_secs(3));
}

/// Each signal the launcher hears reaches its agent, so a repeat ends the
/// agent's clean exit as one from a terminal would.
#[cfg(unix)]
#[test]
fn a_repeat_signal_is_passed_on_too() {
    use std::process::Command;

    use super::Watch;

    let exe = std::env::current_exe().unwrap_or_default();
    let (ready, seen) = (
        exe.with_file_name(format!("rig-repeat-ready-{}", std::process::id())),
        exe.with_file_name(format!("rig-repeat-seen-{}", std::process::id())),
    );
    let script = format!(
        "trap 'echo >> \"{seen}\"; [ $(wc -l < \"{seen}\") -ge 2 ] && exit 0' TERM; : > '{ready}'; \
         while :; do sleep 0.05; done",
        seen = seen.display(),
        ready = ready.display()
    );
    let agent = Command::new("sh").args(["-c", &script]).spawn();
    assert!(agent.is_ok(), "sh starts");
    let Ok(mut agent) = agent else {
        return;
    };
    let since = Instant::now();
    while !ready.exists() && since.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(ready.exists(), "the agent set its trap");
    let mut watch = Watch::start();
    let mut ended = None;
    let lines = || std::fs::read_to_string(&seen).map_or(0, |seen| seen.lines().count());
    for heard in 1..=2 {
        // SAFETY: the watch's handler only notes the signal.
        unsafe {
            libc::raise(libc::SIGTERM);
        }
        watch.pass_on(&mut agent);
        // The agent took this one before the next comes.
        let since = Instant::now();
        while lines() < heard && since.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let since = Instant::now();
    while ended.is_none() && since.elapsed() < Duration::from_secs(5) {
        ended = agent.try_wait().ok().flatten();
        std::thread::sleep(Duration::from_millis(20));
    }
    agent.kill().ok();
    let seen_lines = std::fs::read_to_string(&seen).map(|seen| seen.lines().count());
    std::fs::remove_file(&ready).ok();
    std::fs::remove_file(&seen).ok();
    assert_eq!(seen_lines.ok(), Some(2));
    assert!(ended.is_some_and(|status| status.success()));
}
