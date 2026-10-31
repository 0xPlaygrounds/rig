#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use super::Signals;

#[test]
fn signals_a_millisecond_apart_ask_for_one_exit_and_force_none() {
    let signals = Signals::new();
    assert!(!signals.take());
    assert!(!signals.signal());
    std::thread::sleep(std::time::Duration::from_millis(1));
    assert!(!signals.signal());
    assert!(signals.take());
    assert!(!signals.take());
}

/// Set on the copy of the test below that runs in a process of its own.
const INHERITED: &str = "RIG_TEST_INHERITED_SIGNALS";

/// A process started with SIGINT and SIGHUP ignored, as a script's `&` job
/// and `nohup` start one, keeps ignoring them, and still hears SIGTERM.
/// The test runs again in a process of its own started that way, so no
/// other test sees the dispositions or the signal.
#[cfg(unix)]
#[test]
fn inherited_ignored_signals_stay_ignored_and_sigterm_is_still_heard() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};

    if std::env::var_os(INHERITED).is_none() {
        use std::os::unix::process::CommandExt;

        let name = "host::signals::tests::inherited_ignored_signals_stay_ignored_and_sigterm_is_still_heard";
        let output = std::env::current_exe().and_then(|test| {
            let mut command = std::process::Command::new(test);
            command
                .args(["--exact", name, "--test-threads", "1"])
                .env(INHERITED, "1");
            // SAFETY: the closure runs in the forked child before `exec`
            // and only calls `signal`, which is async-signal-safe; an
            // ignored disposition survives `exec`.
            unsafe {
                command.pre_exec(|| {
                    libc::signal(SIGINT, libc::SIG_IGN);
                    libc::signal(SIGHUP, libc::SIG_IGN);
                    Ok(())
                });
            }
            command.output()
        });
        let passed = output.as_ref().is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed")
        });
        assert!(passed, "{output:?}");
        return;
    }
    static HEARD: AtomicBool = AtomicBool::new(false);
    let heard = super::listen(|| HEARD.store(true, Ordering::SeqCst));
    assert_eq!(heard.ok(), Some(vec![SIGTERM]));
    assert!(super::ignored(SIGINT) && super::ignored(SIGHUP));
    // SAFETY: plain integers; SIGTERM is handled now, so the process lives.
    unsafe {
        libc::raise(SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while !HEARD.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(HEARD.load(Ordering::SeqCst));
}

/// An app whose output nobody reads any more exits as on a signal, at its
/// next frame, whatever it is doing.
#[test]
fn an_app_whose_output_is_unread_exits_as_on_a_signal() {
    use bevy_app::TaskPoolPlugin;
    use bevy_app::prelude::*;
    use rig_ecs::AgentPlugin;
    use rig_harness_protocol::SIGNAL_EXIT_CODE;

    use super::{ExitOnSignalPlugin, ExitWhenUnread};
    use crate::host::exit::ExitPlugin;

    fn gone() -> bool {
        true
    }
    let mut app = App::new();
    app.add_message::<AppExit>()
        .add_plugins((TaskPoolPlugin::default(), AgentPlugin))
        .add_plugins((ExitPlugin, ExitOnSignalPlugin))
        .insert_resource(ExitWhenUnread(gone));
    app.finish();
    app.cleanup();
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        app.should_exit(),
        Some(AppExit::from_code(SIGNAL_EXIT_CODE))
    );
}

/// A pipe whose reader closed has nobody reading it; one still open does.
#[cfg(unix)]
#[test]
fn a_pipe_whose_reader_closed_is_unread() {
    use std::os::fd::AsRawFd;

    let pipe = std::io::pipe();
    assert!(pipe.is_ok());
    if let Ok((reader, writer)) = pipe {
        assert!(!super::reader_gone(writer.as_raw_fd()));
        drop(reader);
        assert!(super::reader_gone(writer.as_raw_fd()));
    }
}
