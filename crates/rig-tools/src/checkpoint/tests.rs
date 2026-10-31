#![allow(clippy::disallowed_methods, reason = "test fixtures")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use rig_core::effect::{EffectId, EffectKind, HandlerKey};
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{Dispatch, DispatchContext, ErasedHandler, RecordBuffer, Recorder};
use rig_core::tool::{ChangeKind, FilesChanged, PublishedContext, content_hash};

use super::{Checkpointed, Taking, forget_kept, snapshot};
use crate::Shell;
use crate::process::Launcher;
use crate::workspace::{Local, Workspace};

/// A fresh scratch directory for `name`, resolved.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-tools-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

/// Every file under `dir`, with its bytes.
fn tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut files: Vec<_> = ignore::WalkBuilder::new(dir)
        .standard_filters(false)
        .build()
        .flatten()
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .map(|entry| {
            (
                entry.path().to_path_buf(),
                std::fs::read(entry.path()).unwrap_or_default(),
            )
        })
        .collect();
    files.sort();
    files
}

/// Whether `git args` succeeded in `repo`, with no user configuration.
fn git_in(repo: &Path, args: &[&str]) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            // No background `git maintenance` after the commit: its
            // `objects/maintenance.lock` would come and go while the
            // test compares `.git`.
            "-c",
            "maintenance.auto=false",
        ])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// In a git work tree, a write between two snapshots is one modified file
/// with its content before and after and its hunks, the earlier snapshot
/// still has the old bytes, a scan is never compared with it, and nothing
/// in the repository's `.git` changed.
#[test]
fn a_git_snapshot_sees_a_write_and_leaves_the_repository_alone() {
    let repo = scratch("checkpoint-git");
    let keep = scratch("checkpoint-keep");
    let git = |args: &[&str]| git_in(&repo, args);
    assert!(git(&["init", "-q"]));
    std::fs::write(repo.join("a.txt"), "one\n").ok();
    assert!(git(&["add", "a.txt"]) && git(&["commit", "-qm", "first"]));
    let objects = tree(&repo.join(".git"));

    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &repo, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(repo.join("a.txt"), "one\ntwo\n").ok();
    let second = snapshot(&Local, &repo, &taking, Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let changes: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| {
            let after = change.after_text();
            (
                change.path,
                change.kind,
                change.added,
                change.before.and_then(|it| it.hash),
                change.after.and_then(|it| it.hash),
                after,
            )
        })
        .collect();
    assert_eq!(
        changes,
        [(
            repo.join("a.txt").to_string_lossy().into_owned(),
            ChangeKind::Modified,
            1,
            Some(content_hash(b"one\n")),
            Some(content_hash(b"one\ntwo\n")),
            Some("one\ntwo\n".to_owned())
        )]
    );
    assert_eq!(
        first.content(&repo.join("a.txt")).and_then(Result::ok),
        Some(b"one\n".to_vec())
    );
    // A scan of the same files, taken with nowhere to keep content, is
    // never compared with a git snapshot.
    let scan = snapshot(&Local, &repo, &Taking::default(), None);
    let mixed = scan.and_then(|scan| scan.changes_since(&first));
    assert_eq!(
        mixed.map_err(|error| error.kind()).err(),
        Some(std::io::ErrorKind::InvalidInput)
    );
    assert_eq!(tree(&repo.join(".git")), objects);
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// A shell call wrapped in a checkpoint publishes the files it changed,
/// created and deleted, found by a scan, each with the hash of the content
/// it had before, and its record keeps them, as a resumed session reads
/// them.
#[cfg(unix)]
#[test]
fn a_checkpointed_shell_call_publishes_what_it_changed() {
    let dir = scratch("checkpoint-shell");
    std::fs::write(dir.join("kept.txt"), "same\n").ok();
    std::fs::write(dir.join("a.txt"), "old\n").ok();
    std::fs::write(dir.join("gone.txt"), "bye\n").ok();
    let shell = Shell {
        launcher: Launcher::default().in_dir(&dir),
        ..Shell::default()
    };
    let workspace: Arc<dyn Workspace> = Arc::new(Local);
    let newest = Arc::new(Mutex::new(None));
    let first = Arc::new(Mutex::new(None));
    let before = Arc::new(Mutex::new(None));
    let handler = ErasedHandler::new(Checkpointed {
        inner: ErasedHandler::new(ToolAdapter::new(shell)),
        workspace,
        root: dir.clone(),
        taking: Taking::default(),
        newest: newest.clone(),
        first: first.clone(),
        before: before.clone(),
    });
    let published = PublishedContext::new();
    let kind = EffectKind::ToolCall {
        name: "shell".into(),
        args: r#"{"command":"printf new > a.txt; printf b > b.txt; rm gone.txt"}"#.into(),
    };
    let buffer = RecordBuffer::new();
    let recorder: Arc<dyn Recorder + Send + Sync> = Arc::new(buffer.clone());
    let id = EffectId::from_raw(1);
    recorder.begin(
        id,
        HandlerKey::from("shell"),
        kind.clone(),
        DispatchContext::default(),
    );
    let dispatch = Dispatch::new(id, false)
        .with_scope(published.clone())
        .recorded_by(recorder);
    let outcome = futures::executor::block_on(async {
        handler.handle(kind, dispatch).await.into_outcome().await
    });
    assert!(outcome.is_ok(), "{outcome:?}");
    let changed = published
        .take()
        .and_then(|context| context.result::<FilesChanged>().ok().flatten())
        .unwrap_or_default();
    let recorded = buffer
        .take(id)
        .and_then(|record| record.published)
        .and_then(|output| output.get::<FilesChanged>().ok().flatten());
    assert_eq!(
        recorded.as_ref(),
        Some(&changed),
        "the record lacks the changes"
    );
    let changed: Vec<_> = changed
        .0
        .into_iter()
        .map(|change| {
            (
                change.path,
                change.kind,
                change.before.and_then(|it| it.hash),
            )
        })
        .collect();
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    assert_eq!(
        changed,
        [
            (
                path("a.txt"),
                ChangeKind::Modified,
                Some(content_hash(b"old\n"))
            ),
            (path("b.txt"), ChangeKind::Created, None),
            (
                path("gone.txt"),
                ChangeKind::Deleted,
                Some(content_hash(b"bye\n"))
            ),
        ]
    );
    // The call's own "before" and the first one are kept, and against the
    // last "after" each shows the call's three changes.
    let kept = |snapshot: &Mutex<Option<_>>| snapshot.lock().ok().and_then(|it| it.clone());
    for before in [&first, &before] {
        let since = kept(&newest)
            .zip(kept(before))
            .and_then(|(last, before)| last.changes_since(&before).ok());
        assert_eq!(since.map(|changes| changes.len()), Some(3));
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A root reached through a symbolic link into a git work tree names its
/// changes as the root was given, as a scan and the Footprint do, and
/// reads a file's content back by that spelling.
#[cfg(unix)]
#[test]
fn a_git_snapshot_names_files_under_the_root_as_given() {
    let repo = scratch("checkpoint-link-repo");
    let keep = scratch("checkpoint-link-keep");
    let link = scratch("checkpoint-link").join("project");
    std::fs::create_dir_all(repo.join("src")).ok();
    std::fs::write(repo.join("src/a.txt"), "one\n").ok();
    assert!(git_in(&repo, &["init", "-q"]));
    assert!(std::os::unix::fs::symlink(repo.join("src"), &link).is_ok());

    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &link, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(link.join("a.txt"), "two\n").ok();
    let second = snapshot(&Local, &link, &taking, Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let paths: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| change.path)
        .collect();
    assert_eq!(paths, [link.join("a.txt").to_string_lossy().into_owned()]);
    assert_eq!(
        first.content(&link.join("a.txt")).and_then(Result::ok),
        Some(b"one\n".to_vec())
    );
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
    link.parent().map(std::fs::remove_dir_all);
}

/// Outside git, a snapshot taken with somewhere to keep content reads a
/// file back as it was, and diffs it against a later one; once the kept
/// content is forgotten, it cannot.
#[test]
fn a_scan_keeps_content_when_it_may() {
    let dir = scratch("checkpoint-scan-kept");
    let keep = scratch("checkpoint-scan-keep");
    std::fs::write(dir.join("a.txt"), "one\n").ok();
    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &dir, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(dir.join("a.txt"), "one\ntwo\n").ok();
    let second = snapshot(&Local, &dir, &taking, Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let diffs: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| (change.kind, change.added, change.after_text()))
        .collect();
    assert_eq!(
        diffs,
        [(ChangeKind::Modified, 1, Some("one\ntwo\n".to_owned()))]
    );
    let a = dir.join("a.txt");
    assert_eq!(
        first.content(&a).and_then(Result::ok),
        Some(b"one\n".to_vec())
    );
    assert!(forget_kept(&keep).is_ok());
    assert!(first.content(&a).is_some_and(|read| read.is_err()));
    let unkept = snapshot(&Local, &dir, &Taking::default(), None);
    assert!(unkept.is_ok_and(|unkept| unkept.content(&a).is_none()));
    std::fs::remove_dir_all(dir).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// A scan reuses an earlier scan's hash only for a file that had not just
/// changed when that scan began: one rewritten at once to the same size,
/// its change time the same (as a coarse clock leaves it), is still a
/// change.
#[test]
fn a_scan_reads_again_a_file_that_had_just_changed() {
    let dir = scratch("checkpoint-scan-racy");
    let a = dir.join("a.txt");
    std::fs::write(&a, "one\n").ok();
    let first = snapshot(&Local, &dir, &Taking::default(), None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    let modified = std::fs::metadata(&a).and_then(|meta| meta.modified());
    assert!(modified.is_ok(), "{modified:?}");
    let Ok(modified) = modified else { return };
    std::fs::write(&a, "two\n").ok();
    let file = std::fs::File::options().write(true).open(&a);
    assert!(file.and_then(|file| file.set_modified(modified)).is_ok());
    let second = snapshot(&Local, &dir, &Taking::default(), Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let kinds: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| change.kind)
        .collect();
    assert_eq!(kinds, [ChangeKind::Modified]);
    std::fs::remove_dir_all(dir).ok();
}

/// A path named in `also` counts although `.gitignore` leaves it out: a
/// file created there is a change, and is none without it.
#[test]
fn a_git_snapshot_counts_ignored_paths_it_is_told_to() {
    let repo = scratch("checkpoint-also");
    let keep = scratch("checkpoint-also-keep");
    std::fs::write(repo.join(".gitignore"), "dist/\n").ok();
    std::fs::create_dir_all(repo.join("dist")).ok();
    assert!(git_in(&repo, &["init", "-q"]));
    let created = |also: Vec<PathBuf>, name: &str| {
        let taking = Taking {
            keep: Some(keep.join(name)),
            also,
            ..Taking::default()
        };
        let first = snapshot(&Local, &repo, &taking, None).ok()?;
        std::fs::write(repo.join("dist").join(name), "built\n").ok()?;
        let second = snapshot(&Local, &repo, &taking, Some(&first)).ok()?;
        let changes = second.changes_since(&first).ok()?;
        Some(
            changes
                .into_iter()
                .map(|change| change.kind)
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(created(Vec::new(), "plain.js"), Some(Vec::new()));
    assert_eq!(
        created(vec![PathBuf::from("dist")], "also.js"),
        Some(vec![ChangeKind::Created])
    );
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// A project in a directory an enclosing git repository ignores, such as
/// its `target`, is in no work tree of that repository: it is scanned,
/// and a created and a modified file are both changes.
#[test]
fn a_root_an_enclosing_repository_ignores_is_scanned() {
    let repo = scratch("checkpoint-ignored-root");
    let keep = scratch("checkpoint-ignored-root-keep");
    std::fs::write(repo.join(".gitignore"), "target/\n").ok();
    assert!(git_in(&repo, &["init", "-q"]));
    let root = repo.join("target").join("project");
    std::fs::create_dir_all(&root).ok();
    std::fs::write(root.join("a.txt"), "one\n").ok();
    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &root, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(root.join("a.txt"), "one\ntwo\n").ok();
    std::fs::write(root.join("b.txt"), "new\n").ok();
    let second = snapshot(&Local, &root, &taking, Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    assert!(changes.is_ok(), "{changes:?}");
    let mut changes: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| (change.path, change.kind))
        .collect();
    changes.sort_by(|a, b| a.0.cmp(&b.0));
    let path = |name: &str| root.join(name).to_string_lossy().into_owned();
    assert_eq!(
        changes,
        [
            (path("a.txt"), ChangeKind::Modified),
            (path("b.txt"), ChangeKind::Created),
        ]
    );
    assert_eq!(
        first.content(&root.join("a.txt")).and_then(Result::ok),
        Some(b"one\n".to_vec())
    );
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// Calls checkpointed at once share the root's private index, and each
/// still gets its snapshot: none fails on the index's lock.
#[test]
fn git_snapshots_taken_at_once_all_succeed() {
    let repo = scratch("checkpoint-concurrent");
    let keep = scratch("checkpoint-concurrent-keep");
    assert!(git_in(&repo, &["init", "-q"]));
    for file in 0..50 {
        std::fs::write(repo.join(format!("{file}.txt")), format!("{file}\n")).ok();
    }
    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let failed: Vec<String> = std::thread::scope(|scope| {
        let calls: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    (0..5)
                        .filter_map(|_| snapshot(&Local, &repo, &taking, None).err())
                        .map(|error| error.to_string())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        calls
            .into_iter()
            .flat_map(|call| call.join().unwrap_or_default())
            .collect()
    });
    assert_eq!(failed, Vec::<String>::new());
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// Git snapshot changes created, modified and deleted at once each carry
/// their own content: the text before and after of every file.
#[test]
fn git_changes_carry_each_files_own_content() {
    let repo = scratch("checkpoint-sides");
    let keep = scratch("checkpoint-sides-keep");
    assert!(git_in(&repo, &["init", "-q"]));
    for (name, text) in [("a", "a1\n"), ("b", "b1\n"), ("d", "")] {
        std::fs::write(repo.join(name), text).ok();
    }
    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &repo, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(repo.join("a"), "a2\n").ok();
    std::fs::remove_file(repo.join("b")).ok();
    std::fs::write(repo.join("c"), "c2\n").ok();
    std::fs::write(repo.join("d"), "d2\n").ok();
    let second = snapshot(&Local, &repo, &taking, Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let sides: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| {
            let after = change.after_text();
            (change.kind, change.before_text, after)
        })
        .collect();
    let text = |text: &str| Some(text.to_owned());
    assert_eq!(
        sides,
        [
            (ChangeKind::Modified, text("a1\n"), text("a2\n")),
            (ChangeKind::Deleted, text("b1\n"), None),
            (ChangeKind::Created, None, text("c2\n")),
            (ChangeKind::Modified, text(""), text("d2\n")),
        ]
    );
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// `git` starts as the launcher starts every child: a wrapper that refuses
/// to run anything leaves no git snapshot, so the snapshot is a scan.
#[cfg(unix)]
#[test]
fn git_starts_through_the_launcher() {
    let repo = scratch("checkpoint-launcher");
    let keep = scratch("checkpoint-launcher-keep");
    assert!(git_in(&repo, &["init", "-q"]));
    std::fs::write(repo.join("a.txt"), "one\n").ok();
    let taking = |launcher: Launcher| Taking {
        keep: Some(keep.clone()),
        launcher,
        ..Taking::default()
    };
    let is_git = |taking: Taking| {
        snapshot(&Local, &repo, &taking, None)
            .is_ok_and(|snapshot| matches!(snapshot.0, super::Taken::Git { .. }))
    };
    assert!(is_git(taking(Launcher::default())));
    assert!(!is_git(taking(Launcher::default().wrapped(["false"]))));
}

/// A scan that could not `stat` a file, in a directory it may list but not
/// search, does not take it for deleted, and neither a file in a directory
/// it could not list at all; its content is not known rather than missing.
/// A file it found but could not read is there: created.
#[cfg(unix)]
#[test]
fn a_file_a_scan_could_not_look_at_is_not_deleted() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = scratch("checkpoint-scan-unknown");
    let keep = scratch("checkpoint-scan-unknown-keep");
    let (listed, closed) = (dir.join("listed"), dir.join("closed"));
    for sub in [&listed, &closed] {
        std::fs::create_dir_all(sub).ok();
        std::fs::write(sub.join("a.txt"), "one\n").ok();
    }
    let taking = Taking {
        keep: Some(keep.clone()),
        ..Taking::default()
    };
    let first = snapshot(&Local, &dir, &taking, None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    let mode = |path: &Path, mode| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).ok()
    };
    let made = dir.join("made.txt");
    std::fs::write(&made, "new\n").ok();
    mode(&made, 0o000);
    mode(&listed, 0o600);
    mode(&closed, 0o000);
    let second = snapshot(&Local, &dir, &taking, Some(&first));
    mode(&listed, 0o755);
    mode(&closed, 0o755);
    let changes = second.as_ref().map_err(|error| error.kind());
    let changes = changes.map(|second| {
        let changes = second.changes_since(&first).unwrap_or_default();
        let kinds = changes.into_iter().map(|change| (change.kind, change.path));
        kinds.collect::<Vec<_>>()
    });
    let unknown = second
        .ok()
        .and_then(|second| second.content(&closed.join("a.txt")));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&keep).ok();
    let created = (ChangeKind::Created, made.to_string_lossy().into_owned());
    assert_eq!(changes, Ok(vec![created]));
    // Not known, or, to one who may read anything, as it was.
    let content = unknown.map(|read| read.map_err(|error| error.kind()));
    assert!(
        matches!(&content, Some(Err(kind)) if *kind != std::io::ErrorKind::NotFound)
            || content == Some(Ok(b"one\n".to_vec())),
        "{content:?}"
    );
}
