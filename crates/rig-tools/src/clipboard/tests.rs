use std::io::ErrorKind;
use std::time::{Duration, Instant};

use super::{Backend, Program, Use, output_of, write_with};
use crate::process::Launcher;

const MISSING: Program = ("rig-test-no-such-clipboard", &[]);
const LIMIT: Duration = Duration::from_secs(5);

/// A clipboard program that is not installed is named, for a read and a
/// write, not taken for an empty clipboard; one missing beside another
/// that runs is not an error.
#[test]
fn a_missing_clipboard_program_is_named() {
    let launcher = Launcher::default();
    let read = output_of(&launcher, &[MISSING], LIMIT);
    let failure = read.err();
    assert_eq!(
        failure.as_ref().map(|it| it.kind()),
        Some(ErrorKind::NotFound)
    );
    assert!(failure.is_some_and(|it| it.to_string().contains(MISSING.0)));
    let written = write_with(&launcher, &[MISSING], "text").err();
    assert_eq!(
        written.as_ref().map(|it| it.kind()),
        Some(ErrorKind::NotFound)
    );
    assert!(written.is_some_and(|it| it.to_string().contains(MISSING.0)));

    #[cfg(unix)]
    {
        let empty: Program = ("false", &[]);
        let read = output_of(&launcher, &[MISSING, empty], LIMIT);
        assert!(matches!(read, Ok(None)), "{read:?}");
        let printing: Program = ("sh", &["-c", "printf copied"]);
        let read = output_of(&launcher, &[MISSING, printing], LIMIT);
        assert_eq!(read.ok().flatten().as_deref(), Some(&b"copied"[..]));
    }
}

/// A clipboard program that does not answer is given up on at the limit,
/// with an error naming it.
#[cfg(unix)]
#[test]
fn a_clipboard_read_gives_up_at_its_limit() {
    let hanging: Program = ("sh", &["-c", "sleep 30"]);
    let started = Instant::now();
    let read = output_of(&Launcher::default(), &[hanging], Duration::from_millis(200));
    assert!(started.elapsed() < Duration::from_secs(10));
    let failure = read.err();
    assert_eq!(
        failure.as_ref().map(|it| it.kind()),
        Some(ErrorKind::TimedOut)
    );
    assert!(failure.is_some_and(|it| it.to_string().contains("sh did not answer")));
}

/// A text read asks Wayland for text, as `wl-paste` with no type prints
/// whatever the clipboard offers, such as a JPEG's bytes.
#[test]
fn a_wayland_text_read_asks_for_text() {
    let (_, args) = Backend::Wayland.command(Use::ReadText);
    assert!(
        args.windows(2).any(|pair| pair == ["--type", "text"]),
        "{args:?}"
    );
}
