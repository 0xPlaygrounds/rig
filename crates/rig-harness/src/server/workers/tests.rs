use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::{TOOLS_OF, kill_home_strays, kill_strays};
use crate::harness_protocol::Home;

/// A process a worker's tool left in its own group, as `setsid` leaves it:
/// only the variable it inherited says whose it was.
fn stray(dir: &Path) -> Option<Child> {
    let mut sleep = Command::new("sleep");
    sleep.arg("300").env(TOOLS_OF, dir);
    std::os::unix::process::CommandExt::process_group(&mut sleep, 0);
    sleep.spawn().ok()
}

/// Whether `child` ended within 5 s.
fn ends(child: &mut Child) -> bool {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(5) {
        if child.try_wait().ok().flatten().is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

fn reap(child: &mut Child) -> bool {
    let alive = child.try_wait().ok().flatten().is_none();
    child.kill().ok();
    child.wait().ok();
    alive
}

#[test]
fn a_dead_workers_stray_tools_are_killed_and_no_other_sessions() {
    let home = std::env::temp_dir().join(format!("strays-{}", std::process::id()));
    let (mine, theirs) = (home.join("sessions/1"), home.join("sessions/2"));
    let (Some(mut stray), Some(mut other)) = (stray(&mine), stray(&theirs)) else {
        return;
    };
    kill_strays(&mine);
    let gone = ends(&mut stray);
    let other_alive = reap(&mut other);
    reap(&mut stray);
    assert!(gone, "the dead worker's tool is killed");
    assert!(other_alive, "another session's tool lives on");
}

#[test]
fn a_starting_server_kills_what_a_dead_servers_workers_left_of_its_home() {
    let root = std::env::temp_dir().join(format!("strays-home-{}", std::process::id()));
    let elsewhere = root.with_extension("other");
    let (Some(mut left), Some(mut other)) = (
        stray(&root.join("sessions/1")),
        stray(&elsewhere.join("sessions/1")),
    ) else {
        return;
    };
    kill_home_strays(&Home::at(&root));
    let gone = ends(&mut left);
    let other_alive = reap(&mut other);
    reap(&mut left);
    assert!(gone, "the tool of this home's dead server is killed");
    assert!(other_alive, "another home's tool lives on");
}
