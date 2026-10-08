//! Snapshots of the working tree for rewinds, kept in a git object store
//! outside the project (`RIG_HOME/snapshots/<hash of the work tree>`), with
//! an index of the session's own. Taking one stages the whole tree, which
//! git does from file stats for files it has seen, and writes a tree object:
//! the tree's hash is the snapshot's id. The project's own repository, its
//! index and its history are never touched; files its `.gitignore`s and
//! `info/exclude` leave out are left out here too. As opencode's
//! snapshots (`references/opencode/packages/opencode/src/snapshot/index.ts:318-347`,
//! restoring at `:382-405`).
//!
//! Restoring takes a snapshot of the files as they are, diffs the two trees
//! and puts back only the paths that differ: changed and deleted files are
//! checked out of the snapshot, and files it does not have are removed.
//!
//! Only a git work tree gets snapshots: without one, a stray `$HOME` would
//! be staged whole.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use bevy_app::prelude::*;
use bevy_log::{info, warn};
use rig::code_protocol::{Home, Mode};

use crate::core::rewind::{FileSnapshots, Restored, Snapshots};
use crate::core::save::SessionPaths;
use crate::host::headless::RunMode;

/// Keeps snapshots of the working tree for rewinds, when the agent runs in
/// a git work tree and git is installed. Added by
/// [`RigCodePlugins`](crate::RigCodePlugins) after the session plugin;
/// disabling it leaves rewinds to the conversation alone.
pub struct SnapshotPlugin;

impl Plugin for SnapshotPlugin {
    fn build(&self, app: &mut App) {
        let Some(paths) = app.world().get_resource::<SessionPaths>().cloned() else {
            return;
        };
        // An eval's trials work in directories of their own, not this one.
        if app
            .world()
            .get_resource::<RunMode>()
            .is_some_and(|mode| matches!(mode.mode(), Mode::Eval { .. }))
        {
            return;
        }
        let Some(tree) = std::env::current_dir().ok().and_then(|cwd| work_tree(&cwd)) else {
            info!("not in a git work tree: rewinds leave the files alone");
            return;
        };
        let home = Home::from_env();
        app.insert_resource(Snapshots::new(GitSnapshots {
            store: home.snapshots(&tree),
            index: paths.snapshot_index(),
            home: home.root().to_owned(),
            tree,
            ready: Mutex::new(false),
        }));
    }
}

/// The top of the git work tree holding `directory`, if any.
fn work_tree(directory: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(directory)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let top = String::from_utf8(output.stdout).ok()?;
    let top = PathBuf::from(top.trim_end_matches(['\n', '\r']));
    top.is_dir().then_some(top)
}

/// Snapshots of one work tree in a store of their own.
struct GitSnapshots {
    /// The work tree's top.
    tree: PathBuf,
    /// The git directory holding the snapshots' objects.
    store: PathBuf,
    /// This session's index into the store.
    index: PathBuf,
    /// `RIG_HOME`, left out when it is inside the work tree.
    home: PathBuf,
    /// Whether this process set the store up; held while git runs, so the
    /// agents of a session take turns.
    ready: Mutex<bool>,
}

/// Options for every snapshot command: files as they are on disk, no
/// quoting of odd paths, and no filesystem monitor of the project's.
const CONFIG: &[&str] = &[
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.symlinks=true",
    "-c",
    "core.longpaths=true",
    "-c",
    "core.quotepath=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "gc.auto=0",
];

impl GitSnapshots {
    /// A git command on the store, the work tree and the session's index.
    fn git(&self, args: &[&str]) -> Command {
        let mut command = Command::new("git");
        command
            .current_dir(&self.tree)
            .env("GIT_INDEX_FILE", &self.index)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .args(CONFIG)
            .arg("--git-dir")
            .arg(&self.store)
            .arg("--work-tree")
            .arg(&self.tree)
            .args(args)
            .stderr(Stdio::piped())
            .stdout(Stdio::piped());
        command
    }

    /// Runs git with `args`, writing `input` to its standard input, and
    /// returns what it printed.
    fn run(&self, args: &[&str], input: Option<&[u8]>) -> Result<Vec<u8>, String> {
        let mut command = self.git(args);
        command.stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        let mut child = command
            .spawn()
            .map_err(|error| format!("could not run git: {error}"))?;
        if let (Some(input), Some(mut stdin)) = (input, child.stdin.take()) {
            stdin
                .write_all(input)
                .map_err(|error| format!("could not write to git: {error}"))?;
        }
        let output = child
            .wait_with_output()
            .map_err(|error| format!("git failed: {error}"))?;
        if output.status.success() {
            return Ok(output.stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "`git {}` failed: {}",
            args.first().copied().unwrap_or_default(),
            stderr.trim()
        ))
    }

    /// Creates the store the first time, and writes what it leaves out.
    fn set_up(&self, ready: &mut bool) -> Result<(), String> {
        if *ready {
            return Ok(());
        }
        if !self.store.join("HEAD").exists() {
            fs::create_dir_all(&self.store)
                .map_err(|error| format!("could not create the snapshot store: {error}"))?;
            self.run(&["init", "--quiet"], None)?;
        }
        let info = self.store.join("info");
        fs::create_dir_all(&info)
            .and_then(|()| fs::write(info.join("exclude"), self.excluded()))
            .map_err(|error| format!("could not write the snapshot excludes: {error}"))?;
        *ready = true;
        Ok(())
    }

    /// What the store leaves out beyond the work tree's `.gitignore`s: the
    /// project repository's `info/exclude`, and `RIG_HOME` when it is
    /// inside the work tree (the store itself is there).
    fn excluded(&self) -> String {
        let mut lines = Vec::new();
        let exclude = Command::new("git")
            .args([
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "info/exclude",
            ])
            .current_dir(&self.tree)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|path| PathBuf::from(path.trim_end_matches(['\n', '\r'])));
        if let Some(text) = exclude.and_then(|path| fs::read_to_string(path).ok()) {
            lines.push(text.trim_end().to_owned());
        }
        if let Ok(inside) = self.home.strip_prefix(&self.tree) {
            let inside = inside.to_string_lossy().replace('\\', "/");
            if !inside.is_empty() {
                lines.push(format!("/{inside}/"));
            }
        }
        let mut text = lines.join("\n");
        text.push('\n');
        text
    }

    /// Stages the whole tree into the session's index and writes it as a
    /// tree object; its hash.
    fn write_tree(&self) -> Result<String, String> {
        self.run(&["add", "--all", "--", "."], None)?;
        let hash = self.run(&["write-tree"], None)?;
        let hash = String::from_utf8_lossy(&hash).trim().to_owned();
        if hash.is_empty() {
            return Err("git wrote no tree".to_owned());
        }
        Ok(hash)
    }
}

impl FileSnapshots for GitSnapshots {
    fn take(&self) -> Result<String, String> {
        let mut ready = self
            .ready
            .lock()
            .map_err(|_| "the snapshot store is poisoned".to_owned())?;
        self.set_up(&mut ready)?;
        self.write_tree()
    }

    fn restore(&self, id: &str) -> Result<Restored, String> {
        let mut ready = self
            .ready
            .lock()
            .map_err(|_| "the snapshot store is poisoned".to_owned())?;
        self.set_up(&mut ready)?;
        let before = self.write_tree()?;
        if before == id {
            return Ok(Restored {
                before,
                written: 0,
                removed: 0,
            });
        }
        let changes = self.run(
            &[
                "diff-tree",
                "-r",
                "-z",
                "--no-renames",
                "--name-status",
                id,
                &before,
            ],
            None,
        )?;
        let mut back = Vec::new();
        let mut gone = Vec::new();
        let mut fields = changes
            .split(|byte| *byte == 0)
            .filter(|field| !field.is_empty());
        while let (Some(status), Some(path)) = (fields.next(), fields.next()) {
            // `A`dded since the snapshot: it goes; anything else is put back.
            if status.first() == Some(&b'A') {
                gone.push(path);
            } else {
                back.push(path);
            }
        }
        // The index becomes the snapshot, and the paths that differ are
        // written from it.
        self.run(&["read-tree", id], None)?;
        if !back.is_empty() {
            let mut input = Vec::new();
            for path in &back {
                input.extend_from_slice(path);
                input.push(0);
            }
            self.run(&["checkout-index", "-f", "-z", "--stdin"], Some(&input))?;
        }
        let mut removed = 0;
        for path in gone {
            let path = self.tree.join(relative(path));
            match fs::remove_file(&path) {
                Ok(()) => {
                    removed += 1;
                    remove_empty_parents(&path, &self.tree);
                }
                Err(error) => warn!("could not remove {}: {error}", path.display()),
            }
        }
        Ok(Restored {
            before,
            written: back.len(),
            removed,
        })
    }
}

/// A path git printed, relative to the work tree.
fn relative(path: &[u8]) -> PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        PathBuf::from(std::ffi::OsStr::from_bytes(path))
    }
    #[cfg(not(unix))]
    {
        PathBuf::from(String::from_utf8_lossy(path).into_owned())
    }
}

/// Removes the directories above `path` that its removal left empty, up
/// to `top`.
fn remove_empty_parents(path: &Path, top: &Path) {
    let mut parent = path.parent();
    while let Some(directory) = parent {
        if directory == top || !directory.starts_with(top) || fs::remove_dir(directory).is_err() {
            return;
        }
        parent = directory.parent();
    }
}
