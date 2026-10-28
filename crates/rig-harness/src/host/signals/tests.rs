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

/// A process started with SIGINT and SIGHUP ignored, as a script's `&` job
/// and `nohup` start one, keeps ignoring them, and still handles SIGTERM.
/// Both are put back afterwards; no signal is raised, so other tests of
/// the process see none.
#[cfg(unix)]
#[test]
fn inherited_ignored_signals_stay_ignored_and_sigterm_is_still_handled() {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};

    let handled = |signal: i32| {
        // SAFETY: as in `ignored`.
        unsafe {
            let mut current: libc::sigaction = std::mem::zeroed();
            libc::sigaction(signal, std::ptr::null(), &mut current) == 0
                && current.sa_sigaction != libc::SIG_IGN
                && current.sa_sigaction != libc::SIG_DFL
        }
    };
    // SAFETY: plain integers.
    let before = [SIGINT, SIGHUP].map(|signal| unsafe { libc::signal(signal, libc::SIG_IGN) });
    let heard = super::listen(|| {});
    let kept = super::ignored(SIGINT) && super::ignored(SIGHUP);
    let term_handled = handled(SIGTERM);
    for (signal, before) in [SIGINT, SIGHUP].into_iter().zip(before) {
        // SAFETY: plain integers, putting back what was there.
        unsafe {
            libc::signal(signal, before);
        }
    }
    assert_eq!(heard.ok(), Some(vec![SIGTERM]));
    assert!(kept && term_handled);
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
