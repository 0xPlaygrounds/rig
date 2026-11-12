use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures::StreamExt as _;

use super::{Watch, WatchEvent};
use crate::workspace::tests::Memory;
use crate::workspace::{Listing, Local, Workspace};

/// The events `watch` tells, on a channel to wait on with a timeout.
fn events(watch: Watch) -> Receiver<WatchEvent> {
    let (sender, events) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut watch = watch;
        while let Some(event) = futures::executor::block_on(watch.next()) {
            if sender.send(event).is_err() {
                return;
            }
        }
    });
    events
}

/// The batches of changes `watch` tells after its snapshot, on a channel
/// to wait on with a timeout, once the snapshot came: what changes from
/// then on is told as a change. The system's watch may also tell one made
/// while it took the snapshot, which the snapshot already holds.
fn batches(watch: Watch) -> Receiver<Vec<PathBuf>> {
    let (sender, batches) = std::sync::mpsc::channel();
    let events = events(watch);
    let first = events.recv_timeout(Duration::from_secs(10));
    assert!(matches!(first, Ok(WatchEvent::Snapshot(_))), "{first:?}");
    std::thread::spawn(move || {
        while let Ok(event) = events.recv() {
            if let WatchEvent::Changed(paths) = event
                && sender.send(paths).is_err()
            {
                return;
            }
        }
    });
    batches
}

/// The paths of the batches that come within `wait`.
fn seen(batches: &Receiver<Vec<PathBuf>>, wait: Duration) -> Vec<PathBuf> {
    let mut seen = Vec::new();
    while let Ok(batch) = batches.recv_timeout(wait) {
        seen.extend(batch);
    }
    seen
}

/// A watch of this machine hears of a created file, and not of one that
/// `files` would leave out: under `target/`, hidden, or matched by an
/// ignore file.
#[test]
fn a_local_watch_reports_what_files_would_list() {
    let root = std::env::temp_dir().join(format!("rig-tools-watch-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(root.join("target")).ok();
    std::fs::write(root.join(".ignore"), "*.log\n").ok();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let batches = batches(watch);
    for name in ["target/built.o", ".env", "noise.log", "src.rs"] {
        std::fs::write(root.join(name), "x").ok();
    }
    let seen = seen(&batches, Duration::from_secs(2));
    assert_eq!(seen, [root.join("src.rs")]);
    std::fs::remove_dir_all(root).ok();
}

/// A recursive watch reports the files of a directory moved in from outside
/// it, nested ones too, as it does for a directory created there, and hears
/// of a later edit inside it.
#[test]
fn a_recursive_watch_reports_the_files_of_a_directory_moved_in() {
    let base = std::env::temp_dir().join(format!("rig-tools-watch-moved-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("root")).ok();
    std::fs::create_dir_all(base.join("staging/nested")).ok();
    std::fs::write(base.join("staging/top.rs"), "x").ok();
    std::fs::write(base.join("staging/nested/deep.rs"), "x").ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let root = base.join("root");
    let watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let batches = batches(watch);
    std::fs::rename(base.join("staging"), root.join("moved")).ok();
    let mut moved_in = seen(&batches, Duration::from_secs(2));
    moved_in.sort();
    moved_in.dedup();
    let deep = root.join("moved/nested/deep.rs");
    std::fs::write(&deep, "edited").ok();
    let mut edited = seen(&batches, Duration::from_secs(2));
    edited.dedup();
    std::fs::remove_dir_all(&base).ok();
    assert_eq!(
        (moved_in, edited),
        (vec![deep.clone(), root.join("moved/top.rs")], vec![deep])
    );
}

/// A watch of a relative directory and one of a relative file each hear of a
/// change. The fixture is below the current directory, which the test leaves
/// alone because tests may share a process.
#[test]
fn a_relative_watch_sees_a_change() {
    let root = Path::new("target").join(format!("rig-tools-watch-relative-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    let made = root.join("made.rs");
    let file = root.join("watched.txt");
    std::fs::write(&file, "before").ok();
    let directory_watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    let file_watch = Arc::new(Local).watch(&file, true, Listing::Visible);
    assert!(directory_watch.is_ok(), "{directory_watch:?}");
    assert!(file_watch.is_ok(), "{file_watch:?}");
    let (Ok(directory_watch), Ok(file_watch)) = (directory_watch, file_watch) else {
        return;
    };
    let (directory_batches, file_batches) = (batches(directory_watch), batches(file_watch));
    std::fs::write(&made, "x").ok();
    std::fs::write(&file, "after").ok();
    let directory_saw = seen(&directory_batches, Duration::from_secs(2));
    let file_saw = seen(&file_batches, Duration::from_secs(2));
    std::fs::remove_dir_all(&root).ok();
    assert!(
        directory_saw.iter().any(|path| path.ends_with(&made))
            && file_saw.iter().any(|path| path.ends_with(&file)),
        "a relative directory watch saw {directory_saw:?}, a relative file watch saw {file_saw:?}"
    );
}

/// The default watch looks again every second, and sees an edit that keeps
/// the size even when the workspace cannot tell a file's change time.
#[test]
fn the_default_watch_sees_a_same_size_edit() {
    let memory = Arc::new(Memory::default());
    let note = Path::new("/project/note.txt");
    memory.write(note, b"one").ok();
    let watch = memory
        .clone()
        .watch(Path::new("/project"), true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let batches = batches(watch);
    memory.write(note, b"two").ok();
    assert_eq!(
        batches.recv_timeout(Duration::from_secs(5)).ok(),
        Some(vec![note.to_path_buf()])
    );
}

/// A watch of a directory that an ignore rule above it leaves out still
/// reports its files, as `files` lists them when asked for that directory.
#[test]
fn a_watch_of_an_ignored_directory_reports_what_files_lists_there() {
    let top = std::env::temp_dir().join(format!("rig-tools-watch-ignored-{}", std::process::id()));
    std::fs::remove_dir_all(&top).ok();
    std::fs::create_dir_all(top.join(".git")).ok();
    std::fs::create_dir_all(top.join("generated")).ok();
    std::fs::write(top.join(".gitignore"), "generated/\n").ok();
    let top = std::fs::canonicalize(&top).unwrap_or(top);
    let root = top.join("generated");
    let made = root.join("made.txt");
    let watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let batches = batches(watch);
    std::fs::write(&made, "x").ok();
    let seen = seen(&batches, Duration::from_secs(2));
    let listed = Local
        .files(&root, None, Listing::Visible)
        .map(|listed| listed.files)
        .unwrap_or_default();
    std::fs::remove_dir_all(&top).ok();
    assert_eq!((listed, seen), (vec![made.clone()], vec![made]));
}

/// [`super::send_batches`] of `tree`, with no system watcher to arm again.
fn send_batches(
    raw: &Receiver<super::Heard>,
    sender: &futures::channel::mpsc::UnboundedSender<WatchEvent>,
    tree: &super::Tree,
) {
    let target = tree.root.clone();
    let armed = super::Armed::new(Weak::new(), target, notify::RecursiveMode::Recursive);
    if let Ok(mut armed) = armed {
        super::send_batches(raw, sender, tree, &mut armed);
    }
}

/// The thread that gathers a watch's events into batches ends once the
/// watch is gone, at its next batch, though the watcher still sends.
#[test]
fn the_batching_thread_ends_with_its_watch() {
    let root = std::env::temp_dir();
    let file = root.join("rig-tools-watch-gone.txt");
    let (raw_sender, raw) = std::sync::mpsc::channel();
    let (sender, watch) = futures::channel::mpsc::unbounded();
    drop(watch);
    raw_sender
        .send(super::Heard::Paths(vec![file.clone()]))
        .ok();
    let tree = super::Tree {
        root: file,
        recursive: false,
        listing: Listing::All,
    };
    let (done, ended) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        send_batches(&raw, &sender, &tree);
        done.send(()).ok();
    });
    assert!(ended.recv_timeout(Duration::from_secs(10)).is_ok());
    drop(raw_sender);
}

/// A watch follows the ignore rules as they change: an edited `.gitignore`
/// counts from the next event, a rule's exception is reported, and a new
/// directory is not; a root that does not exist yet is looked at every
/// second until it does.
#[test]
fn a_local_watch_follows_edited_rules_and_waits_for_a_missing_root() {
    let top = std::env::temp_dir().join(format!("rig-tools-watch-rules-{}", std::process::id()));
    std::fs::remove_dir_all(&top).ok();
    std::fs::create_dir_all(top.join(".git")).ok();
    std::fs::write(top.join(".gitignore"), "*.log\n!keep.log\n").ok();
    let top = std::fs::canonicalize(&top).unwrap_or(top);
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    std::fs::create_dir_all(top.join("sub")).ok();
    std::fs::write(top.join("keep.log"), "x").ok();
    std::fs::write(top.join(".gitignore"), "*.log\n!keep.log\n*.tmp\n").ok();
    std::fs::write(top.join("scratch.tmp"), "x").ok();
    std::fs::write(top.join("sub/new.rs"), "x").ok();
    let mut reported = seen(&events, Duration::from_secs(2));
    reported.sort();
    assert_eq!(reported, [top.join("keep.log"), top.join("sub/new.rs")]);

    let later = top.join("later");
    let watch = Arc::new(Local).watch(&later, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    std::fs::create_dir_all(&later).ok();
    std::fs::write(later.join("made.rs"), "x").ok();
    let reported = seen(&events, Duration::from_secs(3));
    std::fs::remove_dir_all(&top).ok();
    assert_eq!(reported, [later.join("made.rs")]);
}

/// A directory moved out takes its ignore rules with it, its
/// subdirectories' too, even after its own ignore file was edited: a
/// replacement moved in at the same path is filtered by its own rules, with
/// or without ignore files of its own.
#[test]
fn a_directory_moved_in_where_another_was_follows_its_own_rules() {
    let base =
        std::env::temp_dir().join(format!("rig-tools-watch-replaced-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("top/.git")).ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let top = base.join("top");
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    let reused = top.join("reused");
    let (mut edited, mut replaced) = (Vec::new(), Vec::new());
    for (name, ignore) in [("bare", None), ("own", Some(""))] {
        let old = base.join(format!("{name}-old"));
        std::fs::create_dir_all(old.join("inner")).ok();
        std::fs::write(old.join(".gitignore"), "*.log\n").ok();
        std::fs::write(old.join("inner/.gitignore"), "*.rs\n").ok();
        std::fs::write(old.join("inner/visible.txt"), "x").ok();
        std::fs::rename(&old, &reused).ok();
        seen(&events, Duration::from_secs(1));
        // An edit inside reads the moved-in directories' rules.
        std::fs::write(reused.join("inner/visible.txt"), "edited").ok();
        edited.push(seen(&events, Duration::from_secs(1)));
        std::fs::write(reused.join(".gitignore"), "*.log\n*.tmp\n").ok();
        seen(&events, Duration::from_secs(1));
        std::fs::rename(&reused, base.join(format!("{name}-parked"))).ok();
        seen(&events, Duration::from_secs(1));
        let new = base.join(name);
        std::fs::create_dir_all(new.join("inner")).ok();
        std::fs::write(new.join("inner/new.rs"), "x").ok();
        if let Some(rules) = ignore {
            std::fs::write(new.join("inner/.gitignore"), rules).ok();
        }
        std::fs::rename(&new, &reused).ok();
        replaced.push(seen(&events, Duration::from_secs(1)));
        std::fs::rename(&reused, base.join(format!("{name}-out"))).ok();
        seen(&events, Duration::from_secs(1));
    }
    std::fs::remove_dir_all(&base).ok();
    let visible = vec![reused.join("inner/visible.txt")];
    assert_eq!(edited, [visible.clone(), visible]);
    let new = vec![reused.join("inner/new.rs")];
    assert_eq!(replaced, [new.clone(), new]);
}

/// Editing only an ignore file reports the files whose listing it changed,
/// as a look that lists them again would: those it now leaves out and
/// those it lets back in, in the root and in a directory below it, and
/// not the files it leaves alone. Edits in one batch count together: a file
/// one hides and another shows again is not reported, a file created just
/// before its rule goes is, and so is one removed, or removed and made
/// again, or in a directory removed and made again, just before a rule
/// hides it.
#[test]
fn an_edited_ignore_file_reports_the_files_it_hides_or_shows() {
    let top = std::env::temp_dir().join(format!("rig-tools-watch-edited-{}", std::process::id()));
    std::fs::remove_dir_all(&top).ok();
    std::fs::create_dir_all(top.join(".git")).ok();
    std::fs::create_dir_all(top.join("nested")).ok();
    std::fs::write(top.join(".gitignore"), "").ok();
    std::fs::write(top.join("existing.rs"), "x").ok();
    std::fs::write(top.join("notes.txt"), "x").ok();
    std::fs::write(top.join("nested/.gitignore"), "").ok();
    std::fs::write(top.join("nested/deep.rs"), "x").ok();
    let top = std::fs::canonicalize(&top).unwrap_or(top);
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    let mut reported = Vec::new();
    // A file with no text is removed; a directory is removed and made again.
    let steps: [&[(&str, Option<&str>)]; 11] = [
        &[(".gitignore", Some("*.rs\n"))],
        &[(".gitignore", Some(""))],
        &[("nested/.gitignore", Some("*.rs\n"))],
        &[("nested/.gitignore", Some(""))],
        &[
            (".gitignore", Some("*.rs\n")),
            ("nested/.gitignore", Some("!deep.rs\n")),
        ],
        &[("fresh.rs", Some("x")), (".gitignore", Some(""))],
        &[("fresh.rs", None), (".gitignore", Some("*.rs\n"))],
        &[(".gitignore", Some(""))],
        &[
            ("existing.rs", None),
            ("existing.rs", Some("x")),
            (".gitignore", Some("*.rs\n")),
        ],
        &[(".gitignore", Some(""))],
        &[
            ("nested", None),
            ("nested/new.txt", Some("x")),
            (".gitignore", Some("nested/*.rs\n")),
        ],
    ];
    for writes in steps {
        for (file, text) in writes {
            let path = top.join(file);
            match text {
                Some(text) => std::fs::write(path, text).ok(),
                None if path.is_dir() => std::fs::remove_dir_all(&path)
                    .and_then(|()| std::fs::create_dir(&path))
                    .ok(),
                None => std::fs::remove_file(path).ok(),
            };
        }
        let mut seen = seen(&events, Duration::from_secs(1));
        seen.sort();
        reported.push(seen);
    }
    std::fs::remove_dir_all(&top).ok();
    let [existing, deep, fresh] =
        ["existing.rs", "nested/deep.rs", "fresh.rs"].map(|f| top.join(f));
    let both = vec![existing.clone(), deep.clone()];
    assert_eq!(
        reported,
        [
            both.clone(),
            both,
            vec![deep.clone()],
            vec![deep.clone()],
            vec![existing.clone()],
            vec![existing.clone(), fresh.clone()],
            vec![existing.clone(), fresh],
            vec![existing.clone()],
            vec![existing.clone()],
            vec![existing],
            vec![deep, top.join("nested/new.txt")],
        ]
    );
}

/// Rules read for an edited ignore file below an ignored directory are
/// dropped with that directory when it is moved away: its replacement is
/// filtered by its own rules once the parent's rule goes.
#[test]
fn rules_read_for_an_edit_go_with_their_moved_directory() {
    let base = std::env::temp_dir().join(format!("rig-tools-watch-closure-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("top/.git")).ok();
    std::fs::create_dir_all(base.join("top/a/b")).ok();
    std::fs::create_dir_all(base.join("new/b")).ok();
    std::fs::write(base.join("top/.gitignore"), "a/\n").ok();
    std::fs::write(base.join("top/a/b/.gitignore"), "").ok();
    std::fs::write(base.join("new/b/new.rs"), "x").ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let top = base.join("top");
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    std::fs::write(top.join("a/b/.gitignore"), "*.rs\n").ok();
    seen(&events, Duration::from_secs(1));
    std::fs::rename(top.join("a"), base.join("parked")).ok();
    std::fs::rename(base.join("new"), top.join("a")).ok();
    std::fs::write(top.join(".gitignore"), "").ok();
    let reported = seen(&events, Duration::from_secs(1));
    std::fs::remove_dir_all(&base).ok();
    assert!(reported.contains(&top.join("a/b/new.rs")), "{reported:?}");
}

/// A line taken out of `.git/info/exclude` once the watch started shows
/// the file it hid, as a look that lists the files again would.
#[test]
fn an_edited_exclude_file_shows_what_it_hid() {
    let top = std::env::temp_dir().join(format!("rig-tools-watch-exclude-{}", std::process::id()));
    std::fs::remove_dir_all(&top).ok();
    std::fs::create_dir_all(top.join(".git/info")).ok();
    std::fs::write(top.join(".git/info/exclude"), "secret.txt\n").ok();
    std::fs::write(top.join("secret.txt"), "x").ok();
    let top = std::fs::canonicalize(&top).unwrap_or(top);
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    seen(&events, Duration::from_millis(500));
    std::fs::write(top.join(".git/info/exclude"), "").ok();
    let reported = seen(&events, Duration::from_secs(1));
    std::fs::remove_dir_all(&top).ok();
    assert_eq!(reported, [top.join("secret.txt")]);
}

/// Each event `events` tells within `wait` of the one before, by kind,
/// with its paths.
fn told(events: &Receiver<WatchEvent>, wait: Duration) -> Vec<(&'static str, Vec<PathBuf>)> {
    let mut told = Vec::new();
    while let Ok(event) = events.recv_timeout(wait) {
        let kind = match event {
            WatchEvent::Snapshot(_) => "snapshot",
            WatchEvent::Changed(_) => "changed",
            WatchEvent::Unavailable(_) => "unavailable",
            WatchEvent::Closed => "closed",
        };
        told.push((kind, event.paths().to_vec()));
    }
    told
}

/// A polling watch whose workspace fails to list once reports no file
/// removed, and none made again when it lists again: it says the listing
/// failed, once, and then only what really changed.
#[test]
fn a_failed_look_reports_no_change() {
    let memory = Arc::new(Memory::default());
    let [a, b] = ["a.txt", "b.txt"].map(|name| Path::new("/project").join(name));
    memory.write(&a, b"x").ok();
    memory.write(&b, b"x").ok();
    let watch = memory
        .clone()
        .watch(Path::new("/project"), true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = events(watch);
    let started = told(&events, Duration::from_millis(1500));
    memory.1.store(true, std::sync::atomic::Ordering::SeqCst);
    let failing = told(&events, Duration::from_millis(2500));
    memory.write(&b, b"edited").ok();
    memory.1.store(false, std::sync::atomic::Ordering::SeqCst);
    let recovered = told(&events, Duration::from_millis(1500));
    assert_eq!(
        (started, failing, recovered),
        (
            vec![("snapshot", vec![a, b.clone()])],
            vec![("unavailable", vec![])],
            vec![("changed", vec![b])]
        )
    );
}

/// A polling watch whose first listing fails says so and ends: it is no
/// empty directory, nor a watch of nothing kept open. A root that is not
/// there is one with no files, whose files are told as they come.
#[test]
fn a_failed_start_is_not_an_empty_directory() {
    let memory = Arc::new(Memory::default());
    memory.write(Path::new("/project/note.txt"), b"x").ok();
    memory.1.store(true, std::sync::atomic::Ordering::SeqCst);
    let watch = memory
        .clone()
        .watch(Path::new("/project"), true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let memory_events = events(watch);
    let failed = told(&memory_events, Duration::from_millis(1500));
    assert_eq!(failed, [("unavailable", vec![]), ("closed", vec![])]);

    let base = std::env::temp_dir().join(format!("rig-tools-watch-later-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(&base).ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let later = base.join("later");
    let watch = Arc::new(Local).watch(&later, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let later_events = events(watch);
    let before = told(&later_events, Duration::from_millis(1500));
    std::fs::create_dir_all(&later).ok();
    std::fs::write(later.join("made.rs"), "x").ok();
    let after = told(&later_events, Duration::from_millis(2500));
    std::fs::remove_dir_all(&base).ok();
    assert_eq!(
        (before, after),
        (
            vec![("snapshot", vec![])],
            vec![("changed", vec![later.join("made.rs")])]
        )
    );
}

/// A local watch of a root it may not look into, here behind a directory
/// it may not enter, cannot start: only a root that is not there may come
/// later.
#[cfg(unix)]
#[test]
fn a_local_watch_of_a_root_behind_a_locked_directory_cannot_start() {
    use std::os::unix::fs::PermissionsExt as _;
    let base = std::env::temp_dir().join(format!("rig-tools-watch-behind-{}", std::process::id()));
    let root = base.join("locked/project");
    std::fs::create_dir_all(&root).ok();
    let mode =
        |mode| std::fs::set_permissions(base.join("locked"), std::fs::Permissions::from_mode(mode));
    mode(0o000).ok();
    // Whoever may enter it anyway (root) can watch it.
    let locked = std::fs::metadata(&root).is_err();
    let watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    mode(0o755).ok();
    std::fs::remove_dir_all(&base).ok();
    if locked {
        assert!(watch.is_err(), "{watch:?}");
    }
}

/// A local watch whose listing fails, of a directory below the root or of
/// the root itself, reports a file edited meanwhile once a listing works,
/// as the default watch does: within a second, though no event says so.
#[cfg(unix)]
#[test]
fn an_edit_made_while_a_listing_fails_is_reported_once_it_works() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = std::env::temp_dir().join(format!("rig-tools-watch-failing-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(root.join("sub")).ok();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let [a, sub] = ["a.txt", "sub"].map(|name| root.join(name));
    let f = sub.join("f.txt");
    std::fs::write(&a, "one").ok();
    std::fs::write(&f, "one").ok();
    let mode =
        |dir: &Path, mode| std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode));
    let (raw_sender, raw) = std::sync::mpsc::channel();
    let (sender, watch) = futures::channel::mpsc::unbounded();
    let tree = super::Tree {
        root: root.clone(),
        recursive: true,
        listing: Listing::Visible,
    };
    std::thread::spawn(move || send_batches(&raw, &sender, &tree));
    let events = events(Watch::new(watch, ()));
    let wait = Duration::from_millis(1500);
    let started = told(&events, wait);
    // Written to, not listed: the edit's event names the file.
    mode(&sub, 0o300).ok();
    let unreadable = std::fs::read_dir(&sub).is_err();
    std::fs::write(&f, "two").ok();
    raw_sender.send(super::Heard::Paths(vec![f.clone()])).ok();
    let sub_failing = told(&events, wait);
    mode(&sub, 0o755).ok();
    let sub_back = told(&events, wait);
    mode(&root, 0o300).ok();
    std::fs::write(&a, "two").ok();
    raw_sender.send(super::Heard::Paths(vec![a.clone()])).ok();
    let root_failing = told(&events, wait);
    mode(&root, 0o755).ok();
    let root_back = told(&events, Duration::from_millis(2500));
    std::fs::remove_dir_all(&root).ok();
    if !unreadable {
        return;
    }
    assert_eq!(
        (started, sub_failing, sub_back, root_failing, root_back),
        (
            vec![("snapshot", vec![a.clone(), f.clone()])],
            vec![],
            vec![("changed", vec![f])],
            vec![("unavailable", vec![])],
            vec![("changed", vec![a])],
        )
    );
}

/// When the system's watcher says it may have missed events, the watch
/// lists everything again and reports a file edited meanwhile, though no
/// event named it.
#[test]
fn a_rescan_reports_a_file_edited_unheard() {
    let root = std::env::temp_dir().join(format!("rig-tools-watch-rescan-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let [a, b] = ["a.txt", "b.txt"].map(|name| root.join(name));
    std::fs::write(&a, "one").ok();
    std::fs::write(&b, "one").ok();
    let (raw_sender, raw) = std::sync::mpsc::channel();
    let (sender, watch) = futures::channel::mpsc::unbounded();
    let tree = super::Tree {
        root: root.clone(),
        recursive: true,
        listing: Listing::Visible,
    };
    std::thread::spawn(move || send_batches(&raw, &sender, &tree));
    let events = events(Watch::new(watch, ()));
    let started = told(&events, Duration::from_millis(1000));
    std::fs::write(&a, "two").ok();
    raw_sender.send(super::Heard::Everything).ok();
    let rescanned = told(&events, Duration::from_millis(1000));
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(
        (started, rescanned),
        (
            vec![("snapshot", vec![a.clone(), b])],
            vec![("changed", vec![a])]
        )
    );
}

/// A directory swapped in for another under the same name, holding files
/// of the same names and sizes with other content, reports them: they are
/// other files.
#[test]
fn a_directory_swapped_in_with_the_same_names_reports_its_files() {
    let base = std::env::temp_dir().join(format!("rig-tools-watch-swap-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("top/dir")).ok();
    std::fs::write(base.join("top/dir/x.rs"), "one").ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let top = base.join("top");
    let watch = Arc::new(Local).watch(&top, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    std::fs::create_dir_all(base.join("new")).ok();
    std::fs::write(base.join("new/x.rs"), "two").ok();
    // Written within one tick of the clock, both would have the same time.
    let later = std::time::SystemTime::now() + Duration::from_secs(5);
    let new = std::fs::File::options()
        .write(true)
        .open(base.join("new/x.rs"));
    assert!(new.and_then(|file| file.set_modified(later)).is_ok());
    std::fs::rename(top.join("dir"), base.join("old")).ok();
    std::fs::rename(base.join("new"), top.join("dir")).ok();
    let mut reported = seen(&events, Duration::from_secs(1));
    reported.dedup();
    std::fs::remove_dir_all(&base).ok();
    assert_eq!(reported, [top.join("dir/x.rs")]);
}

/// A local watch of a directory that is moved away and replaced, or
/// removed and made again, watches the one now there: what it held is
/// removed, and a later edit in the new one is told.
#[test]
fn a_replaced_root_is_watched_again() {
    let base = std::env::temp_dir().join(format!(
        "rig-tools-watch-replaced-root-{}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("root")).ok();
    std::fs::write(base.join("root/old.txt"), "old").ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    let root = base.join("root");
    let watch = Arc::new(Local).watch(&root, true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    let [old, new] = ["old.txt", "new.txt"].map(|name| root.join(name));
    std::fs::rename(&root, base.join("root.old")).ok();
    std::fs::create_dir_all(&root).ok();
    std::fs::write(&new, "new").ok();
    let replaced = seen(&events, Duration::from_millis(2500));
    std::fs::write(&new, "edited").ok();
    let edited = seen(&events, Duration::from_millis(1500));
    std::fs::remove_dir_all(&root).ok();
    let removed = seen(&events, Duration::from_millis(1500));
    std::fs::create_dir_all(&root).ok();
    std::fs::write(&old, "again").ok();
    let remade = seen(&events, Duration::from_millis(2500));
    std::fs::write(&old, "edited again").ok();
    let edited_again = seen(&events, Duration::from_millis(1500));
    std::fs::remove_dir_all(&base).ok();
    let set = |paths: Vec<PathBuf>| paths.into_iter().collect::<BTreeSet<_>>();
    assert_eq!(
        [replaced, edited, removed, remade, edited_again].map(set),
        [
            BTreeSet::from([old.clone(), new.clone()]),
            BTreeSet::from([new.clone()]),
            BTreeSet::from([new]),
            BTreeSet::from([old.clone()]),
            BTreeSet::from([old]),
        ]
    );
}

/// One change to a tree, its paths relative to the directory that holds
/// it: text written to a file (its directories made), a file or directory
/// removed, or one renamed.
#[derive(Clone, Debug)]
enum Op {
    Write(&'static str, &'static str),
    Remove(&'static str),
    Rename(&'static str, &'static str),
}

impl Op {
    fn apply(&self, base: &Path) {
        match self {
            Self::Write(file, text) => {
                let path = base.join(file);
                path.parent().map(std::fs::create_dir_all);
                std::fs::write(path, text).ok();
            }
            Self::Remove(path) => {
                let path = base.join(path);
                match path.is_dir() {
                    true => std::fs::remove_dir_all(path).ok(),
                    false => std::fs::remove_file(path).ok(),
                };
            }
            Self::Rename(from, to) => {
                std::fs::rename(base.join(from), base.join(to)).ok();
            }
        }
    }
}

/// The steps every watch of the tree goes through: each is applied at
/// once, then the watches settle.
const SCRIPT: &[&[Op]] = {
    use Op::{Remove, Rename, Write};
    &[
        &[Write("top/a.rs", "a")],
        &[Write("top/sub/b.rs", "b")],
        &[Write("top/sub/deep/c.rs", "c")],
        &[Write("top/a.rs", "edited")],
        &[
            Write("top/.gitignore", "*.log\n"),
            Write("top/sub/x.log", "x"),
        ],
        &[Write("top/.gitignore", "")],
        &[Write("top/sub/.ignore", "*.rs\n")],
        &[Remove("top/sub/.ignore")],
        &[Write("top/.git/info/exclude", "secret.txt\n")],
        &[
            Write("top/secret.txt", "s"),
            Write("top/sub/secret.txt", "s"),
            Write("tree/secret.txt", "s"),
        ],
        &[Write("top/.git/info/exclude", "")],
        &[Write("top/secret.txt", "edited")],
        &[
            Write("staging/n/m.rs", "m"),
            Rename("staging", "top/sub/moved"),
        ],
        &[Rename("top/sub/moved", "parked")],
        &[
            Write("new/inner/r.rs", "r"),
            Rename("top/sub/deep", "old-deep"),
            Rename("new", "top/sub/deep"),
        ],
        &[Write("top/sub/swap/x.rs", "one")],
        &[
            Write("swapped/x.rs", "two"),
            Rename("top/sub/swap", "unswapped"),
            Rename("swapped", "top/sub/swap"),
        ],
        &[Remove("top/a.rs")],
        &[
            Write("top/target/t.rs", "t"),
            Write("top/.hidden/h.rs", "h"),
        ],
        &[Write("top/.gitignore", "sub/deep/\n")],
        &[
            Write("top/sub/deep/inner/r.rs", "edited"),
            Write("top/sub/deep/d.rs", "d"),
        ],
        &[Write("top/.gitignore", "")],
        &[
            Write("newsub/z.rs", "z"),
            Rename("top/sub", "oldsub"),
            Rename("newsub", "top/sub"),
        ],
        &[Write("top/sub/z.rs", "edited")],
        &[Remove("top/sub")],
        &[Write("top/sub/y.rs", "y")],
        &[Write("top/sub/y.rs", "edited")],
    ]
};

/// What a seeded walk over `count` steps does: writes, edits and removes
/// files, and edits the rules that hide some of them.
fn generated(seed: u64, count: usize) -> Vec<Op> {
    const FILES: [&str; 6] = [
        "top/p.rs",
        "top/q.txt",
        "top/sub/p.rs",
        "top/sub/q.txt",
        "top/sub/deep/p.rs",
        "top/sub/deep/q.log",
    ];
    const RULES: [Op; 6] = [
        Op::Write("top/sub/.gitignore", "*.txt\n"),
        Op::Write("top/sub/.gitignore", ""),
        Op::Write("top/.git/info/exclude", "*.log\n"),
        Op::Write("top/.git/info/exclude", ""),
        Op::Write("top/.ignore", "deep/\n"),
        Op::Write("top/.ignore", ""),
    ];
    const TEXTS: [&str; 3] = ["one", "two two", "three three three"];
    let mut state = seed;
    let mut next = move |below: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize % below
    };
    (0..count)
        .map(|_| {
            let file = FILES.get(next(FILES.len())).copied().unwrap_or("top/p.rs");
            match next(4) {
                0 => Op::Remove(file),
                1 => RULES
                    .get(next(RULES.len()))
                    .cloned()
                    .unwrap_or(Op::Remove(file)),
                _ => Op::Write(file, TEXTS.get(next(TEXTS.len())).copied().unwrap_or("x")),
            }
        })
        .collect()
}

/// The events of each watch since the last call, until none came for a
/// while and each pair of watches (the system's and the polling one of a
/// root) told the same, or for at most five seconds: each watch's paths, a
/// failed listing as the path `!unavailable`. A loaded machine may take
/// longer than the quiet while to tell what a pair agrees on.
fn settled(watches: &[Receiver<WatchEvent>]) -> Vec<BTreeSet<PathBuf>> {
    let mut told = vec![BTreeSet::new(); watches.len()];
    let started = std::time::Instant::now();
    let mut last = started;
    let agree = |told: &[BTreeSet<PathBuf>]| told.chunks(2).all(|pair| pair.first() == pair.last());
    while (last.elapsed() < Duration::from_millis(400) || !agree(&told))
        && started.elapsed() < Duration::from_secs(5)
    {
        for (events, told) in watches.iter().zip(&mut told) {
            while let Ok(event) = events.try_recv() {
                last = std::time::Instant::now();
                match event {
                    WatchEvent::Unavailable(_) => {
                        told.insert(PathBuf::from("!unavailable"));
                    }
                    event => told.extend(event.paths().iter().cloned()),
                }
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    told
}

/// The system's watch tells what listing again would: run beside a watch
/// that lists every 100 ms, on the same steps (files made, edited, removed
/// and moved, a directory moved in, out and replaced, also by one with
/// the same names, `.gitignore`, `.ignore` and `.git/info/exclude`
/// edited, a watched directory replaced, removed and made again), of the
/// whole work tree, a directory in it, the tree's top
/// alone, its hidden files too, and a linked work tree, each pair tells
/// the same snapshot and the same files changed at each step.
#[test]
fn the_system_watch_tells_what_listing_again_would() {
    let base = std::env::temp_dir().join(format!("rig-tools-watch-same-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    std::fs::create_dir_all(base.join("top/.git/info")).ok();
    std::fs::create_dir_all(base.join("top/sub/deep")).ok();
    std::fs::write(base.join("top/start.rs"), "x").ok();
    std::fs::write(base.join("top/sub/deep/start.rs"), "x").ok();
    let base = std::fs::canonicalize(&base).unwrap_or(base);
    // A linked work tree of `top`'s repository, laid out as `git worktree
    // add` makes it: its `.git` is a file naming its git directory, whose
    // `commondir` leads back to `top/.git`, where its excludes are.
    let linked = base.join("top/.git/worktrees/tree");
    std::fs::create_dir_all(&linked).ok();
    std::fs::write(linked.join("commondir"), "../..\n").ok();
    std::fs::create_dir_all(base.join("tree")).ok();
    let gitdir = format!("gitdir: {}\n", linked.display());
    std::fs::write(base.join("tree/.git"), gitdir).ok();
    let roots = [
        (base.join("top"), true, Listing::Visible),
        (base.join("top/sub"), true, Listing::Visible),
        (base.join("top"), false, Listing::Visible),
        (base.join("top"), true, Listing::Hidden),
        (base.join("tree"), true, Listing::Visible),
    ];
    let mut watches = Vec::new();
    for (root, recursive, listing) in &roots {
        let native = super::notified(root, *recursive, *listing);
        let every = Duration::from_millis(100);
        let polled = super::poll_every(Arc::new(Local), root, *recursive, *listing, every);
        assert!(native.is_ok() && polled.is_ok(), "{native:?} {polled:?}");
        let (Ok(native), Ok(polled)) = (native, polled) else {
            return;
        };
        watches.extend([events(native), events(polled)]);
    }
    let mut steps: Vec<Vec<Op>> = SCRIPT.iter().map(|step| step.to_vec()).collect();
    steps.extend(generated(0x5eed, 16).into_iter().map(|op| vec![op]));
    let mut told = vec![(String::from("snapshot"), settled(&watches))];
    for step in steps {
        step.iter().for_each(|op| op.apply(&base));
        told.push((format!("{step:?}"), settled(&watches)));
    }
    std::fs::remove_dir_all(&base).ok();
    let pairs = |told: &[BTreeSet<PathBuf>], at: usize| {
        let pick = |index| told.get(index).cloned().unwrap_or_default();
        (pick(at), pick(at + 1))
    };
    let differ: Vec<_> = told
        .iter()
        .flat_map(|(step, told)| {
            (0..roots.len()).filter_map(move |root| {
                let (native, polled) = pairs(told, 2 * root);
                (native != polled).then(|| (step.clone(), root, native, polled))
            })
        })
        .collect();
    // A line taken out of the excludes shows its files, and an edit of one
    // is told after.
    let at = |name: &str| {
        told.iter()
            .find(|(step, _)| step == name)
            .map(|(_, told)| pairs(told, 0).0)
    };
    let shown = at(r#"[Write("top/.git/info/exclude", "")]"#);
    let edited = at(r#"[Write("top/secret.txt", "edited")]"#);
    let shown_linked = told
        .iter()
        .find(|(step, _)| step == r#"[Write("top/.git/info/exclude", "")]"#)
        .map(|(_, told)| pairs(told, 8).0);
    assert_eq!(differ, []);
    let secret = base.join("top/secret.txt");
    assert_eq!(
        (shown, edited, shown_linked),
        (
            Some(BTreeSet::from([
                secret.clone(),
                base.join("top/sub/secret.txt")
            ])),
            Some(BTreeSet::from([secret])),
            Some(BTreeSet::from([base.join("tree/secret.txt")]))
        )
    );
}
