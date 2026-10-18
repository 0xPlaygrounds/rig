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
