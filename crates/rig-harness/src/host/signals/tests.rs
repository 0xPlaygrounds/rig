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

/// A process started with SIGINT ignored, as a script's `&` job is, keeps
/// ignoring it, and still hears SIGTERM and SIGHUP.
#[cfg(unix)]
#[test]
fn an_inherited_ignored_sigint_stays_ignored_and_sigterm_is_still_heard() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};

    static HEARD: AtomicBool = AtomicBool::new(false);
    // SAFETY: plain integers; the test runs in a process of its own.
    unsafe {
        libc::signal(SIGINT, libc::SIG_IGN);
    }
    let heard = super::listen(|| HEARD.store(true, Ordering::SeqCst));
    assert_eq!(heard.ok(), Some(vec![SIGTERM, SIGHUP]));
    assert!(super::ignored(SIGINT));
    // SAFETY: plain integers; SIGTERM is handled now, so the test lives.
    unsafe {
        libc::raise(SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while !HEARD.load(Ordering::SeqCst) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(HEARD.load(Ordering::SeqCst));
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
