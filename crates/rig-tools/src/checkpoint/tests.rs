use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use rig_core::effect::EffectKind;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{ErasedHandler, serve_inline_with};
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
/// with its content before and after and its diff, the earlier snapshot
/// still has the old bytes, and nothing in the repository's `.git` changed.
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
        also: Vec::new(),
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
            (
                change.path,
                change.kind,
                change.added,
                change.before.and_then(|it| it.hash),
                change.after.and_then(|it| it.hash),
                change.diff.is_some(),
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
            true
        )]
    );
    assert_eq!(
        first.content(&repo.join("a.txt")).and_then(Result::ok),
        Some(b"one\n".to_vec())
    );
    assert_eq!(tree(&repo.join(".git")), objects);
    std::fs::remove_dir_all(repo).ok();
    std::fs::remove_dir_all(keep).ok();
}

/// A shell call wrapped in a checkpoint publishes the files it changed,
/// found by a scan, each with the content it had before.
#[cfg(unix)]
#[test]
fn a_checkpointed_shell_call_publishes_what_it_changed() {
    let dir = scratch("checkpoint-shell");
    std::fs::write(dir.join("kept.txt"), "same\n").ok();
    std::fs::write(dir.join("a.txt"), "old\n").ok();
    let shell = Shell {
        launcher: Launcher::default().in_dir(&dir),
        ..Shell::default()
    };
    let workspace: Arc<dyn Workspace> = Arc::new(Local);
    let last = Arc::new(Mutex::new(None));
    let first = Arc::new(Mutex::new(None));
    let handler = ErasedHandler::new(Checkpointed {
        inner: ErasedHandler::new(ToolAdapter::new(shell)),
        workspace,
        root: dir.clone(),
        taking: Taking::default(),
        last: last.clone(),
        first: first.clone(),
    });
    let published = PublishedContext::new();
    let kind = EffectKind::ToolCall {
        name: "shell".into(),
        args: r#"{"command":"printf new > a.txt; printf b > b.txt"}"#.into(),
    };
    let scopes = vec![published.clone() as Arc<dyn std::any::Any + Send + Sync>];
    let outcome = futures::executor::block_on(serve_inline_with(&handler, kind, scopes));
    assert!(outcome.is_ok(), "{outcome:?}");
    let changed = published
        .take()
        .and_then(|context| context.result::<FilesChanged>().ok().flatten())
        .unwrap_or_default();
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
        ]
    );
    // The first "before" is kept, and against the last "after" it shows
    // the call's two changes.
    let kept = |snapshot: &Mutex<Option<_>>| snapshot.lock().ok().and_then(|it| it.clone());
    let since = kept(&last)
        .zip(kept(&first))
        .and_then(|(last, first)| last.changes_since(&first).ok());
    assert_eq!(since.map(|changes| changes.len()), Some(2));
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
        also: Vec::new(),
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
        also: Vec::new(),
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
        .map(|change| (change.kind, change.added, change.diff.is_some()))
        .collect();
    assert_eq!(diffs, [(ChangeKind::Modified, 1, true)]);
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
