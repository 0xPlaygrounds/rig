//! Checkpoints: the files under a root at one moment ([`snapshot`]), to
//! learn what changed by a later one ([`Snapshot::changes_since`]), and
//! [`Checkpointed`], a handler that reports what any call changed, such as
//! a `shell` call's, as the file tools report their own.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use rig_core::completion::UnsupportedOption;
use rig_core::effect::{EffectKind, HandlerDescriptor};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::tool::{
    ContentHash, FileChange, FilesChanged, PublishedContext, Touches, content_hash,
};

use crate::fs::Previous;
use crate::workspace::{Stat, Workspace};
use crate::{MAX_FILE_BYTES, blocking};

/// The files under a root at one moment, to diff against a later one.
///
/// Taken one of two ways, and compared only with one taken the same way:
/// - in a git work tree on this machine, as a git tree written with a
///   private index and object directory, so the user's repository is
///   never written to, and the content of each file is kept;
/// - elsewhere, as a scan of what [`Workspace::files`] lists, with each
///   file's size, change time and hash, and no content.
///
/// Either way, the files git or `files` leaves out (ignored, hidden,
/// `target`) are not in it.
#[derive(Clone, Debug)]
pub struct Snapshot(Taken);

/// How a [`Snapshot`] was taken.
#[derive(Clone, Debug)]
enum Taken {
    /// A git tree in a private object directory.
    Git { git: Git, tree: String },
    /// Each file's [`Seen`], by path.
    Scan(BTreeMap<PathBuf, Seen>),
}

/// What a scan saw of one file.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    /// Its size.
    size: u64,
    /// When it last changed, when the workspace can tell.
    modified: Option<SystemTime>,
    /// The hash of its bytes; `None` when it is too large to read.
    hash: Option<ContentHash>,
    /// Its lines.
    lines: usize,
}

/// Takes a [`Snapshot`] of `root` in `workspace` as it is now.
///
/// With `keep`, a directory on this machine for the private index and
/// objects, a local `workspace` ([`host`](Workspace::host) is `None`) and
/// `root` in a git work tree, it is a git tree: the private index makes
/// each later snapshot cost what changed. Otherwise, or when `earlier` was
/// a scan, it scans, hashing only the files whose size or change time
/// differ from `earlier`'s (every file when there is none, or the workspace
/// cannot tell the time).
///
/// Blocking: run it off a thread that must not wait ([`blocking`]).
///
/// # Errors
///
/// When git fails after `earlier` was taken with git, or `root` cannot be
/// listed.
pub fn snapshot(
    workspace: &dyn Workspace,
    root: &Path,
    keep: Option<&Path>,
    earlier: Option<&Snapshot>,
) -> io::Result<Snapshot> {
    let git = match earlier.map(|it| &it.0) {
        Some(Taken::Git { git, .. }) => Some(git.clone()),
        Some(Taken::Scan(_)) => None,
        None => keep
            .filter(|_| workspace.host().is_none())
            .and_then(|keep| Git::find(root, keep)),
    };
    match git {
        Some(git) => {
            let tree = git.write_tree()?;
            Ok(Snapshot(Taken::Git { git, tree }))
        }
        None => {
            let before = match earlier.map(|it| &it.0) {
                Some(Taken::Scan(files)) => Some(files),
                _ => None,
            };
            scan(workspace, root, before).map(|files| Snapshot(Taken::Scan(files)))
        }
    }
}

impl Snapshot {
    /// What changed from `earlier` to this snapshot, in path order: files
    /// created, modified and deleted, with the hashes of their bytes before
    /// and after. A git snapshot also gives each change's diff and line
    /// counts; a scan, which keeps no content, gives a modified file's
    /// hashes only.
    ///
    /// # Errors
    ///
    /// When the two were taken different ways, or git fails.
    pub fn changes_since(&self, earlier: &Snapshot) -> io::Result<Vec<FileChange>> {
        match (&earlier.0, &self.0) {
            (Taken::Git { tree: before, .. }, Taken::Git { git, tree }) => {
                git.changes(before, tree)
            }
            (Taken::Scan(before), Taken::Scan(after)) => Ok(scan_changes(before, after)),
            _ => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "the two snapshots were taken different ways",
            )),
        }
    }

    /// The bytes the file `path` held when this snapshot was taken, when it
    /// kept them (a git snapshot); `None` for a scan.
    pub fn content(&self, path: &Path) -> Option<io::Result<Vec<u8>>> {
        match &self.0 {
            Taken::Git { git, tree } => Some(git.content(tree, path)),
            Taken::Scan(_) => None,
        }
    }
}

/// The files under `root` with what a scan sees of them, reusing
/// `earlier`'s hash of a file whose size and change time are the same.
fn scan(
    workspace: &dyn Workspace,
    root: &Path,
    earlier: Option<&BTreeMap<PathBuf, Seen>>,
) -> io::Result<BTreeMap<PathBuf, Seen>> {
    let mut seen = BTreeMap::new();
    for path in workspace.files(root, None)? {
        let Ok(Stat::File { size, modified, .. }) = workspace.stat(&path) else {
            continue;
        };
        let same = earlier
            .and_then(|earlier| earlier.get(&path))
            .filter(|was| modified.is_some() && was.size == size && was.modified == modified);
        let file = match same {
            Some(was) => was.clone(),
            None => {
                let bytes = (size <= MAX_FILE_BYTES)
                    .then(|| workspace.read(&path, MAX_FILE_BYTES).ok())
                    .flatten()
                    .map(|contents| contents.bytes);
                Seen {
                    size,
                    modified,
                    hash: bytes.as_deref().map(content_hash),
                    lines: bytes.map_or(0, |bytes| String::from_utf8_lossy(&bytes).lines().count()),
                }
            }
        };
        seen.insert(path, file);
    }
    Ok(seen)
}

/// What changed from the scan `before` to the scan `after`.
fn scan_changes(
    before: &BTreeMap<PathBuf, Seen>,
    after: &BTreeMap<PathBuf, Seen>,
) -> Vec<FileChange> {
    let paths: BTreeSet<&PathBuf> = before.keys().chain(after.keys()).collect();
    paths
        .into_iter()
        .filter_map(|path| {
            let shown = path.to_string_lossy();
            let change = match (before.get(path), after.get(path)) {
                (None, Some(now)) => FileChange::created(shown, now.lines),
                (Some(was), None) => FileChange::deleted(shown, was.lines),
                (Some(was), Some(now)) if was.hash.is_some() && was.hash == now.hash => {
                    return None;
                }
                (Some(was), Some(now)) if was.hash.is_none() && was == now => return None,
                (Some(_), Some(_)) => FileChange::modified(shown, 0, 0),
                (None, None) => return None,
            };
            let was = before.get(path).and_then(|was| was.hash.clone());
            let now = after.get(path).and_then(|now| now.hash.clone());
            Some(with_hashes(change, was, now))
        })
        .collect()
}

/// `change` with the hashes of the bytes before and after it, those known.
fn with_hashes(
    change: FileChange,
    before: Option<ContentHash>,
    after: Option<ContentHash>,
) -> FileChange {
    let change = match before {
        Some(hash) => change.with_before(hash),
        None => change,
    };
    match after {
        Some(hash) => change.with_after(hash),
        None => change,
    }
}

/// A git work tree, reached with a private index and object directory: the
/// repository's objects are read as an alternate and never written.
#[derive(Clone, Debug)]
struct Git {
    /// The work tree's top directory.
    top: PathBuf,
    /// The repository's git directory.
    dir: PathBuf,
    /// The repository's object directory, read as an alternate.
    objects: PathBuf,
    /// The snapshotted root, below `top` (empty for all of it).
    below: PathBuf,
    /// The private directory: `index` and `objects` in it.
    keep: PathBuf,
}

impl Git {
    /// The git work tree `root` is in, with a private directory under
    /// `keep`; `None` when it is in none or git cannot be run.
    fn find(root: &Path, keep: &Path) -> Option<Self> {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args([
                "rev-parse",
                "--show-toplevel",
                "--absolute-git-dir",
                "--git-path",
                "objects",
            ])
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        let text = String::from_utf8(output.stdout).ok()?;
        let mut lines = text.lines();
        let (top, dir, objects) = (lines.next()?, lines.next()?, lines.next()?);
        let top = PathBuf::from(top);
        let root = std::fs::canonicalize(root).ok()?;
        let below = root.strip_prefix(&top).ok()?.to_path_buf();
        let name = content_hash(top.as_os_str().as_encoded_bytes()).0;
        let keep = keep
            .join("checkpoints")
            .join(name.get(..16).unwrap_or(&name));
        Some(Self {
            objects: root.join(objects),
            top,
            dir: PathBuf::from(dir),
            below,
            keep,
        })
    }

    /// A git command on the work tree, with the private index and objects.
    fn command(&self, args: &[&OsStr]) -> Command {
        let mut command = Command::new("git");
        command
            .arg("--git-dir")
            .arg(&self.dir)
            .arg("--work-tree")
            .arg(&self.top)
            .args([
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
            ])
            .args(args)
            .current_dir(&self.top)
            .env("GIT_INDEX_FILE", self.keep.join("index"))
            .env("GIT_OBJECT_DIRECTORY", self.keep.join("objects"))
            .env("GIT_ALTERNATE_OBJECT_DIRECTORIES", &self.objects)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .stdin(Stdio::null());
        command
    }

    /// What `args` prints, or why it failed.
    fn run(&self, args: &[&OsStr]) -> io::Result<Vec<u8>> {
        let output = self.command(args).output()?;
        match output.status.success() {
            true => Ok(output.stdout),
            false => Err(io::Error::other(format!(
                "git {} failed: {}",
                args.first()
                    .map_or(std::borrow::Cow::Borrowed(""), |it| it.to_string_lossy()),
                String::from_utf8_lossy(&output.stderr).trim()
            ))),
        }
    }

    /// Adds the root's files to the private index and writes its tree.
    fn write_tree(&self) -> io::Result<String> {
        std::fs::create_dir_all(self.keep.join("objects"))?;
        let below = match self.below.as_os_str().is_empty() {
            true => OsStr::new("."),
            false => self.below.as_os_str(),
        };
        let add = ["add", "--all", "--ignore-errors", "--"].map(OsStr::new);
        self.run(&[add.as_slice(), &[below]].concat())?;
        let tree = self.run(&[OsStr::new("write-tree")])?;
        Ok(String::from_utf8_lossy(&tree).trim().to_owned())
    }

    /// The bytes of the object `id`.
    fn blob(&self, id: &str) -> io::Result<Vec<u8>> {
        self.run(&["cat-file", "blob", id].map(OsStr::new))
    }

    /// The bytes `path` held in `tree`.
    fn content(&self, tree: &str, path: &Path) -> io::Result<Vec<u8>> {
        let below = path.strip_prefix(&self.top).map_err(|_| {
            io::Error::new(ErrorKind::InvalidInput, "the path is outside the work tree")
        })?;
        self.blob(&format!("{tree}:{}", below.to_string_lossy()))
    }

    /// What changed from the tree `before` to the tree `after`.
    fn changes(&self, before: &str, after: &str) -> io::Result<Vec<FileChange>> {
        let raw = self.run(
            &[
                "diff-tree",
                "-r",
                "-z",
                "--no-renames",
                "--raw",
                before,
                after,
            ]
            .map(OsStr::new),
        )?;
        let text = String::from_utf8_lossy(&raw);
        let mut fields = text.split('\0');
        let mut changes = Vec::new();
        while let (Some(header), Some(path)) = (fields.next(), fields.next()) {
            // `:old-mode new-mode old-id new-id status`
            let mut parts = header.split(' ').skip(2);
            let (Some(old), Some(new)) = (parts.next(), parts.next()) else {
                continue;
            };
            if old == new {
                continue;
            }
            let missing = |id: &str| id.bytes().all(|byte| byte == b'0');
            let was = (!missing(old)).then(|| self.blob(old)).transpose()?;
            let now = (!missing(new)).then(|| self.blob(new)).transpose()?;
            let shown = self.top.join(path).to_string_lossy().into_owned();
            changes.push(change(&shown, was.as_deref(), now.as_deref()));
        }
        Ok(changes)
    }
}

/// The change from `before` to `after` (`None`: no file) of the file
/// `path`, with its diff when both are small enough to diff.
fn change(path: &str, before: Option<&[u8]>, after: Option<&[u8]>) -> FileChange {
    let small =
        |bytes: Option<&[u8]>| bytes.is_none_or(|bytes| bytes.len() as u64 <= MAX_FILE_BYTES);
    let previous = before.map(|bytes| Previous::of(Some(bytes)));
    match after {
        None => {
            let lines = previous.as_ref().map_or(0, |it| it.text.lines().count());
            with_hashes(
                FileChange::deleted(path, lines),
                before.map(content_hash),
                None,
            )
        }
        Some(after) if small(before) && small(Some(after)) => {
            Previous::change(previous.as_ref(), path, &String::from_utf8_lossy(after))
                .with_after(content_hash(after))
        }
        Some(after) => with_hashes(
            FileChange::modified(path, 0, 0),
            before.map(content_hash),
            Some(content_hash(after)),
        ),
    }
}

/// A handler that takes a [`Snapshot`] before and after its inner handler
/// serves a call, and publishes what changed as [`FilesChanged`] beside the
/// call's answer, unless the call published its own: so a call of any tool
/// that may write anything, such as `shell`, reports what it changed.
///
/// `last` is the snapshot after the previous call, shared between the
/// calls one host checkpoints, and taken as this call's "before" when it is
/// there: a host clears it when something it does not see may have
/// changed the files since. Snapshots run on a thread of their own. A call
/// that streams its answer is served as it is, with nothing published.
#[derive(Clone, Debug)]
pub struct Checkpointed {
    /// The handler that serves the call.
    pub inner: ErasedHandler,
    /// Where the files are.
    pub workspace: Arc<dyn Workspace>,
    /// The directory whose files are compared.
    pub root: PathBuf,
    /// A directory on this machine for a git snapshot's private index and
    /// objects; `None` always scans.
    pub keep: Option<PathBuf>,
    /// The snapshot after the last call, if it still holds.
    pub last: Arc<Mutex<Option<Snapshot>>>,
}

impl Checkpointed {
    /// A snapshot now, against `earlier`, off this thread.
    async fn take(&self, earlier: Option<Snapshot>) -> Option<(Option<Snapshot>, Snapshot)> {
        let (workspace, root, keep) =
            (self.workspace.clone(), self.root.clone(), self.keep.clone());
        blocking(move || {
            let now = snapshot(&*workspace, &root, keep.as_deref(), earlier.as_ref())
                .map_err(|error| rig_core::tool::ToolExecutionError::other(error.to_string()))?;
            Ok((earlier, now))
        })
        .await
        .ok()
    }
}

impl Serve for Checkpointed {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let published = dispatch.scope::<PublishedContext>();
        let last = self
            .last
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let before = match last {
            Some(last) => Some(last),
            None => self.take(None).await.map(|(_, now)| now),
        };
        let reply = self.inner.handle(kind, dispatch).await;
        if !matches!(reply, Reply::Outcome(_)) {
            return reply;
        }
        let Some((Some(before), after)) = self.take(before).await else {
            return reply;
        };
        let changes = after.changes_since(&before).unwrap_or_default();
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(after);
        if let Some(published) = published {
            let own = published
                .result_context()
                .and_then(|context| context.get::<FilesChanged>().ok().flatten());
            if own.is_none() && !changes.is_empty() {
                published.insert(FilesChanged(changes)).ok();
            }
        }
        reply
    }

    fn footprint(&self, kind: &EffectKind) -> Option<Touches> {
        self.inner.footprint(kind)
    }

    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        self.inner.check(kind)
    }
}

#[cfg(test)]
mod tests;
