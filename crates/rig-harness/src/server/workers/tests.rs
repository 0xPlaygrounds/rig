use std::process::Command;
use std::time::{Duration, Instant};

use rig_protocol::SessionId;

use super::{TOOLS_OF, kill_strays};

/// A process a worker's tool left in its own session, as `setsid` leaves
/// it: only the variable it inherited says whose it was.
fn stray(session: &str) -> Option<std::process::Child> {
    let mut sleep = Command::new("sleep");
    sleep.arg("300").env(TOOLS_OF, session);
    std::os::unix::process::CommandExt::process_group(&mut sleep, 0);
    sleep.spawn().ok()
}

#[cfg(target_os = "linux")]
#[test]
fn a_dead_workers_stray_tools_are_killed_and_no_other_sessions() {
    let mine = format!("strays-{}", std::process::id());
    let theirs = format!("{mine}-other");
    let (Some(mut stray), Some(mut other)) = (stray(&mine), stray(&theirs)) else {
        return;
    };
    kill_strays(&SessionId(mine));
    let started = Instant::now();
    let mut gone = false;
    while started.elapsed() < Duration::from_secs(5) {
        if stray.try_wait().ok().flatten().is_some() {
            gone = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let other_alive = other.try_wait().ok().flatten().is_none();
    other.kill().ok();
    other.wait().ok();
    stray.kill().ok();
    stray.wait().ok();
    assert!(gone, "the dead worker's tool is killed");
    assert!(other_alive, "another session's tool lives on");
}
