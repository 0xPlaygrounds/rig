use std::io::ErrorKind;
use std::path::Path;

use super::{optional, read_to_string, rename, replace};

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

/// Two threads replacing one file each finish, and every read, the last
/// included, sees one whole version.
#[test]
fn concurrent_replaces_each_land_whole() {
    let dir = std::env::temp_dir().join(format!("rig-fs-replace-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let target = dir.join("file");
    let versions = [vec![b'a'; 1 << 18], vec![b'b'; 1 << 18]];
    let whole = |bytes: &[u8]| versions.iter().any(|version| version.as_slice() == bytes);
    let made = std::fs::create_dir_all(&dir)
        .and_then(|()| replace(&target, versions.first().map_or(&[][..], Vec::as_slice)));
    let (failed, torn) = std::thread::scope(|scope| {
        let writers: Vec<_> = versions
            .iter()
            .map(|version| {
                let target = &target;
                scope.spawn(move || {
                    (0..50)
                        .filter(|_| replace(target, version).is_err())
                        .count()
                })
            })
            .collect();
        let mut torn = 0;
        while !writers.iter().all(|writer| writer.is_finished()) {
            torn += usize::from(std::fs::read(&target).is_ok_and(|bytes| !whole(&bytes)));
        }
        let failed: usize = writers
            .into_iter()
            .filter_map(|writer| writer.join().ok())
            .sum();
        (failed, torn)
    });
    let last = std::fs::read(&target).unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    assert!(made.is_ok(), "{made:?}");
    assert_eq!((failed, torn), (0, 0));
    assert!(whole(&last));
}
