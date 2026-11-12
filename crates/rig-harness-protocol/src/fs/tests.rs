use std::io::ErrorKind;
use std::path::Path;

use super::{optional, read_to_string, rename};

/// A failure names its step and its path and keeps the kind of the
/// failure under it, so a caller still tells a missing file apart.
#[test]
fn a_failure_names_the_step_and_the_path() {
    let missing = Path::new("/nonexistent/rig/settings.toml");
    let failure = read_to_string(missing).err();
    assert_eq!(
        failure.as_ref().map(std::io::Error::kind),
        Some(ErrorKind::NotFound)
    );
    let text = failure
        .map(|failure| failure.to_string())
        .unwrap_or_default();
    assert!(
        text.starts_with("read /nonexistent/rig/settings.toml: "),
        "{text}"
    );

    let failure = rename(missing, Path::new("/nonexistent/other")).err();
    let text = failure
        .map(|failure| failure.to_string())
        .unwrap_or_default();
    assert!(
        text.starts_with("rename /nonexistent/rig/settings.toml to /nonexistent/other: "),
        "{text}"
    );
}

/// Only a missing file is absent: a path that cannot be read for another
/// reason, here a directory, is an error.
#[test]
fn only_a_missing_file_is_absent() {
    let missing = optional(read_to_string(Path::new("/nonexistent/rig/name")));
    let unreadable = optional(read_to_string(Path::new("/")));
    assert_eq!(missing.ok(), Some(None));
    assert!(unreadable.is_err(), "{unreadable:?}");
}
