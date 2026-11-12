use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use futures::StreamExt as _;

use super::Watch;
use crate::workspace::tests::Memory;
use crate::workspace::{Listing, Local, Workspace};

/// The batches `watch` sends, on a channel to wait on with a timeout.
fn batches(watch: Watch) -> Receiver<Vec<PathBuf>> {
    let (sender, batches) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut watch = watch;
        while let Some(batch) = futures::executor::block_on(watch.next()) {
            if sender.send(batch).is_err() {
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
        .send(super::Raw {
            path: file.clone(),
            created: false,
            made: false,
        })
        .ok();
    let filter = super::Filter {
        root: root.clone(),
        file: Some(file),
        recursive: false,
        listing: Listing::All,
        ignores: super::Ignores::new(&root),
    };
    let (done, ended) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        super::send_batches(&raw, &sender, filter);
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
/// the file it hid: an edit of that file is reported, as a look that lists
/// the files again would.
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
    seen(&events, Duration::from_millis(500));
    std::fs::write(top.join("secret.txt"), "edited").ok();
    let reported = seen(&events, Duration::from_secs(1));
    std::fs::remove_dir_all(&top).ok();
    assert!(reported.contains(&top.join("secret.txt")), "{reported:?}");
}

/// A polling watch whose workspace fails to list once reports no file
/// removed, and none made again when it lists again: nothing changed.
#[test]
fn a_failed_look_reports_no_change() {
    let memory = Arc::new(Memory::default());
    for name in ["a.txt", "b.txt"] {
        memory.write(&Path::new("/project").join(name), b"x").ok();
    }
    let watch = memory
        .clone()
        .watch(Path::new("/project"), true, Listing::Visible);
    assert!(watch.is_ok(), "{watch:?}");
    let Ok(watch) = watch else { return };
    let events = batches(watch);
    seen(&events, Duration::from_millis(1500));
    memory.1.store(true, std::sync::atomic::Ordering::SeqCst);
    let failing = seen(&events, Duration::from_millis(1500));
    memory.1.store(false, std::sync::atomic::Ordering::SeqCst);
    let recovered = seen(&events, Duration::from_millis(1500));
    assert_eq!((failing, recovered), (vec![], vec![]));
}
