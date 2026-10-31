use std::process::Command;
use std::time::{Duration, Instant};

use super::{Watch, heard};

/// A SIGTERM to the launcher reaches its agent, and the launcher's wait
/// ends only once the agent ended, with the agent's own status.
#[cfg(unix)]
#[test]
fn a_signal_is_passed_on_and_the_agent_is_waited_for() {
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
