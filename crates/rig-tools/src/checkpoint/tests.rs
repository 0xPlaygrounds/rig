use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use rig_core::effect::EffectKind;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::serve::{ErasedHandler, serve_inline_with};
use rig_core::tool::{ChangeKind, FilesChanged, PublishedContext, content_hash};

use super::{Checkpointed, snapshot};
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

/// In a git work tree, a write between two snapshots is one modified file
/// with its content before and after and its diff, the earlier snapshot
/// still has the old bytes, and nothing in the repository's `.git` changed.
#[test]
fn a_git_snapshot_sees_a_write_and_leaves_the_repository_alone() {
    let repo = scratch("checkpoint-git");
    let keep = scratch("checkpoint-keep");
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .output()
            .is_ok_and(|output| output.status.success())
    };
    assert!(git(&["init", "-q"]));
    std::fs::write(repo.join("a.txt"), "one\n").ok();
    assert!(git(&["add", "a.txt"]) && git(&["commit", "-qm", "first"]));
    let objects = tree(&repo.join(".git"));

    let first = snapshot(&Local, &repo, Some(&keep), None);
    assert!(first.is_ok(), "{first:?}");
    let Ok(first) = first else { return };
    std::fs::write(repo.join("a.txt"), "one\ntwo\n").ok();
    let second = snapshot(&Local, &repo, Some(&keep), Some(&first));
    let changes = second.and_then(|second| second.changes_since(&first));
    let changes: Vec<_> = changes
        .unwrap_or_default()
        .into_iter()
        .map(|change| {
            (
                change.path,
                change.kind,
                change.added,
                change.before,
                change.after,
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
    let handler = ErasedHandler::new(Checkpointed {
        inner: ErasedHandler::new(ToolAdapter::new(shell)),
        workspace,
        root: dir.clone(),
        keep: None,
        last: last.clone(),
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
        .map(|change| (change.path, change.kind, change.before))
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
    assert!(last.lock().is_ok_and(|last| last.is_some()));
    std::fs::remove_dir_all(dir).ok();
}
