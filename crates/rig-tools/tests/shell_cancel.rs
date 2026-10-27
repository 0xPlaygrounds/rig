#![cfg(target_os = "linux")]

use rig_tools::{
    Shell, ShellArgs,
    process::{Launcher, STOP_GRACE, kill_group_of},
};
use std::{
    fs,
    future::Future,
    io,
    path::PathBuf,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

struct Cleanup {
    dir: PathBuf,
    group: Option<u32>,
}
impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(group) = self.group {
            kill_group_of(group);
        }
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn wait_until(mut condition: impl FnMut() -> bool, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while !condition() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}
fn run_case(redirect: bool, cancel: bool, leader: bool) -> TestResult {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rig-cancel-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir)?;
    let mut cleanup = Cleanup { dir, group: None };
    fs::write(
        cleanup.dir.join("child.py"),
        r#"
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('ready').write_text(f'{os.getpid()} {os.getpgrp()}')
end = time.monotonic() + 15
while time.monotonic() < end:
    if Path('gate').exists():
        Path('after-cancel').write_text('child still executed after stop grace')
        break
    time.sleep(0.01)
"#,
    )?;
    let shell = Shell {
        launcher: Launcher::default().in_dir(&cleanup.dir),
        ..Shell::default()
    };
    let command = if leader {
        "exec python3 child.py >/dev/null 2>&1"
    } else if redirect {
        "python3 child.py >/dev/null 2>&1 & wait"
    } else {
        "python3 child.py & wait"
    };
    let args = if cancel {
        ShellArgs::new(command)
    } else {
        ShellArgs::new(command).timeout_secs(1)
    };
    let mut future = Box::pin(shell.run(args));
    if !matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ) {
        return Err(io::Error::other("shell did not stay pending").into());
    }
    if !wait_until(
        || {
            fs::read_to_string(cleanup.dir.join("ready"))
                .is_ok_and(|s| s.split_whitespace().count() == 2)
        },
        Duration::from_secs(5),
    ) {
        return Err(io::Error::other("child did not become ready").into());
    }
    let ready = fs::read_to_string(cleanup.dir.join("ready"))?;
    let group: u32 = ready
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| io::Error::other("no group"))?
        .parse()?;
    cleanup.group = Some(group);
    if cancel {
        drop(future);
    } else {
        println!("timeout result: {:?}", futures::executor::block_on(future)?);
    }
    std::thread::sleep(STOP_GRACE + Duration::from_millis(500));
    fs::write(cleanup.dir.join("gate"), b"go")?;
    let survived = wait_until(
        || cleanup.dir.join("after-cancel").exists(),
        Duration::from_millis(700),
    );
    println!(
        "redirect={redirect}, cancel={cancel}, pid/group={ready}, wrote after stop grace={survived}"
    );
    if survived {
        return Err(
            io::Error::other("cancelled shell descendant executed after SIGKILL grace").into(),
        );
    }
    Ok(())
}
#[test]
fn cancelled_shell_kills_redirected_descendant() -> TestResult {
    run_case(true, true, false)
}
#[test]
fn inherited_output_control() -> TestResult {
    run_case(false, true, false)
}
#[test]
fn timeout_control() -> TestResult {
    run_case(true, false, false)
}

#[test]
fn cancellation_escalates_when_the_leader_ignores_sigterm() -> TestResult {
    run_case(true, true, true)
}
