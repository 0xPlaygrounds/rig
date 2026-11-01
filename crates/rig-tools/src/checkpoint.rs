//! Checkpoints: the files under a root at one moment ([`snapshot`]), to
//! learn what changed by a later one ([`Snapshot::changes_since`]), and
//! [`Checkpointed`], a handler that reports what any call changed, such as
//! a `shell` call's, as the file tools report their own.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io::{self, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use rig_core::completion::UnsupportedOption;
use rig_core::effect::{EffectKind, HandlerDescriptor, HandlerKey};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::tool::{
    ContentHash, FileChange, FileState, FilesChanged, PublishedContext, Touches, content_hash,
};

use crate::fs::Previous;
use crate::process::{Launcher, Process, Spawn};
use crate::workspace::{Listed, Listing, Stat, Workspace};
use crate::{MAX_FILE_BYTES, blocking};

/// The files under a root at one moment, to diff against a later one.
///
/// Taken one of two ways, and compared only with one taken the same way:
/// - in a git work tree on this machine that does not ignore the root, as
///   a git tree written with a private index and object directory, so the
///   user's repository is never written to;
/// - elsewhere, as a scan of what [`Workspace::files`] lists, with each
///   file's size, change time and hash.
///
/// Either way, the content of each file up to [`MAX_FILE_BYTES`] is kept
/// when the snapshot was taken with somewhere to keep it ([`Taking::keep`]).
/// The files git or `files` leaves out (ignored, hidden, `target`) are not
/// in it, unless named in [`Taking::also`].
#[derive(Clone, Debug)]
pub struct Snapshot(Taken);

/// How a [`Snapshot`] was taken.
#[derive(Clone, Debug)]
enum Taken {
    /// A git tree in a private object directory.
    Git { git: Git, tree: String },
    /// Each file's [`Seen`], by path, when the scan began, the paths it
    /// could not look at, what is at or under which is not known, and the
    /// directory its content is kept in, by hash, when it is.
    Scan {
        files: Arc<BTreeMap<PathBuf, Seen>>,
        unknown: Arc<Vec<PathBuf>>,
        began: SystemTime,
        blobs: Option<PathBuf>,
    },
}

/// How to take a [`Snapshot`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Taking {
    /// A directory on this machine for a git snapshot's private index and
    /// objects, and for a scan's kept content, under `checkpoints/`;
    /// `None`: scan, and keep no content.
    pub keep: Option<PathBuf>,
    /// Paths counted whole although ignore rules, hidden names or `target`
    /// would leave them out, such as `dist` or `.env`: relative to the root,
    /// or absolute under it.
    pub also: Vec<PathBuf>,
    /// How a git snapshot starts `git`: as every other child process
    /// starts, under the host's wrapper, in its environment.
    pub launcher: Launcher,
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

impl Seen {
    /// The file as a change's side.
    fn state(&self) -> FileState {
        FileState {
            bytes: self.size,
            lines: self.hash.is_some().then_some(self.lines),
            hash: self.hash.clone(),
        }
    }
}

/// Takes a [`Snapshot`] of `root` in `workspace` as it is now.
///
/// With [`Taking::keep`], a local `workspace` ([`host`](Workspace::host) is
/// `None`) and `root` in a git work tree that does not ignore it, it is a
/// git tree of the working files' unconverted bytes. Otherwise, or when `earlier` was a scan, it scans, hashing
/// only the files whose size or change time differ from `earlier`'s (every
/// file when there is none, or the workspace cannot tell the time), and
/// with `keep` writes the content of each file it hashed to `keep`, once
/// per hash. A file that changed just before `earlier` began is hashed
/// again too: a write within its change time's precision after `earlier`
/// read it would leave its size and time as they were (what git calls
/// "racily clean"). The kept content is scratch, not forced to disk: no
/// run after this one reads it ([`forget_kept`]).
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
    taking: &Taking,
    earlier: Option<&Snapshot>,
) -> io::Result<Snapshot> {
    let keep = taking.keep.as_deref();
    let git = match earlier.map(|it| &it.0) {
        Some(Taken::Git { git, .. }) => Some(git.clone()),
        Some(Taken::Scan { .. }) => None,
        None => keep
            .filter(|_| workspace.host().is_none())
            .and_then(|keep| Git::find(root, keep, &taking.launcher)),
    };
    let also: Vec<PathBuf> = taking
        .also
        .iter()
        .map(|path| root.join(path))
        .filter(|path| {
            path.starts_with(root) && !path.components().any(|part| part == Component::ParentDir)
        })
        .collect();
    match git {
        Some(git) => {
            let tree = git.write_tree(&also)?;
            Ok(Snapshot(Taken::Git { git, tree }))
        }
        None => {
            let before = match earlier.map(|it| &it.0) {
                Some(Taken::Scan { files, began, .. }) => Some((&**files, *began)),
                _ => None,
            };
            let blobs = keep.map(|keep| private(keep, root).join("blobs"));
            let began = SystemTime::now();
            let (files, unknown) = scan(workspace, root, &also, before, blobs.as_deref())?;
            Ok(Snapshot(Taken::Scan {
                files: Arc::new(files),
                unknown: Arc::new(unknown),
                began,
                blobs,
            }))
        }
    }
}

/// The private directory under `keep` for the snapshots of `dir`.
fn private(keep: &Path, dir: &Path) -> PathBuf {
    let name = content_hash(dir.as_os_str().as_encoded_bytes()).0;
    keep.join("checkpoints")
        .join(name.get(..16).unwrap_or(&name))
}

/// Removes the content scans kept under `keep` ([`Taking::keep`]): no
/// [`Snapshot`] outlives the process that took it, so a host calls this
/// when it starts, before taking any. Git's private objects are left, as
/// they hold only what changed.
///
/// # Errors
///
/// When a directory there cannot be removed.
pub fn forget_kept(keep: &Path) -> io::Result<()> {
    let Ok(dirs) = std::fs::read_dir(keep.join("checkpoints")) else {
        return Ok(());
    };
    for dir in dirs.flatten() {
        match std::fs::remove_dir_all(dir.path().join("blobs")) {
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

impl Snapshot {
    /// What changed from `earlier` to this snapshot, in path order: files
    /// created, modified and deleted, with the [`FileState`] of each side.
    /// A snapshot that kept content also gives each change's diff and line
    /// counts; a scan that kept none gives a modified file's states only.
    /// A path a scan of either could not look at is left out, never taken
    /// for created or deleted.
    ///
    /// # Errors
    ///
    /// When the two were taken different ways, or git fails.
    pub fn changes_since(&self, earlier: &Snapshot) -> io::Result<Vec<FileChange>> {
        match (&earlier.0, &self.0) {
            (Taken::Git { tree: before, .. }, Taken::Git { git, tree }) => {
                git.changes(before, tree)
            }
            (
                Taken::Scan {
                    files: before,
                    unknown: was_unknown,
                    ..
                },
                Taken::Scan {
                    files,
                    unknown,
                    blobs,
                    ..
                },
            ) => {
                let known = |path: &Path| !under(was_unknown, path) && !under(unknown, path);
                Ok(scan_changes(before, files, blobs.as_deref(), known))
            }
            _ => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "the two snapshots were taken different ways",
            )),
        }
    }

    /// The bytes the file `path` held when this snapshot was taken, when it
    /// kept content ([`Taking::keep`]); `None` for a scan that kept none.
    pub fn content(&self, path: &Path) -> Option<io::Result<Vec<u8>>> {
        match &self.0 {
            Taken::Git { git, tree } => Some(git.content(tree, path)),
            Taken::Scan {
                files,
                unknown,
                blobs,
                ..
            } => {
                let blobs = blobs.as_deref()?;
                Some(match files.get(path).map(|seen| &seen.hash) {
                    Some(Some(hash)) => std::fs::read(blobs.join(&hash.0)),
                    Some(None) => Err(io::Error::new(
                        ErrorKind::FileTooLarge,
                        "the file was too large to keep",
                    )),
                    None if under(unknown, path) => Err(io::Error::other(
                        "the scan could not look at the file, so whether it was there is not known",
                    )),
                    None => Err(ErrorKind::NotFound.into()),
                })
            }
        }
    }
}

/// The files under `root`, and every file under each of `also`, with what
/// a scan sees of them, reusing the hash of the scan `earlier` (its files
/// and when it began) for a file whose size and change time are the same
/// and that had not just changed when it began ([`snapshot`]); with
/// `blobs`, the content of each file it read is kept there, by hash, once.
/// Also the paths it could not look at: a directory it could not list, a
/// file it could not `stat`.
fn scan(
    workspace: &dyn Workspace,
    root: &Path,
    also: &[PathBuf],
    earlier: Option<(&BTreeMap<PathBuf, Seen>, SystemTime)>,
    blobs: Option<&Path>,
) -> io::Result<(BTreeMap<PathBuf, Seen>, Vec<PathBuf>)> {
    if let Some(blobs) = blobs {
        std::fs::create_dir_all(blobs)?;
    }
    let Listed {
        files: mut listed,
        failed,
    } = workspace.files(root, None, Listing::Visible)?;
    let mut unknown: Vec<PathBuf> = failed.into_iter().map(|(path, _)| path).collect();
    for path in also {
        match workspace.files(path, None, Listing::All) {
            Ok(also) => {
                listed.extend(also.files);
                unknown.extend(also.failed.into_iter().map(|(path, _)| path));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(_) => unknown.push(path.clone()),
        }
    }
    let mut seen = BTreeMap::new();
    for path in listed {
        let (size, modified) = match workspace.stat(&path) {
            Ok(Stat::File { size, modified, .. }) => (size, modified),
            // Gone, or no file any more: what it was is gone.
            Ok(Stat::Missing | Stat::Dir | Stat::Other) => continue,
            Err(_) => {
                unknown.push(path);
                continue;
            }
        };
        let same = earlier.and_then(|(files, began)| {
            let was = files.get(&path)?;
            let settled = modified.is_some_and(|modified| settled_before(modified, began));
            (settled && was.size == size && was.modified == modified).then_some(was)
        });
        let file = match same {
            Some(was) => was.clone(),
            None => {
                // A file `stat` found but that could not be read is there, by
                // its size and time.
                let bytes = (size <= MAX_FILE_BYTES)
                    .then(|| workspace.read(&path, MAX_FILE_BYTES).ok())
                    .flatten()
                    .map(|contents| contents.bytes);
                let hash = bytes.as_deref().map(content_hash);
                if let (Some(blobs), Some(bytes), Some(hash)) = (blobs, &bytes, &hash) {
                    let blob = blobs.join(&hash.0);
                    if !blob.exists() {
                        crate::fs::write_scratch(&blob, bytes).ok();
                    }
                }
                Seen {
                    size,
                    modified,
                    hash,
                    lines: bytes.map_or(0, |bytes| String::from_utf8_lossy(&bytes).lines().count()),
                }
            }
        };
        seen.insert(path, file);
    }
    Ok((seen, unknown))
}

/// Whether `path` is at or under one of the paths a scan could not look
/// at.
fn under(unknown: &[PathBuf], path: &Path) -> bool {
    unknown.iter().any(|unknown| path.starts_with(unknown))
}

/// Whether a file last changed at `modified` had not just changed when a
/// scan began at `began`, so a write after the scan read it changes its
/// time: older than `began` by more than its time's precision, two seconds
/// for a time in whole seconds (FAT keeps even seconds, some file systems
/// whole ones), else a tenth of a second (the kernel's clock for file
/// times ticks every few milliseconds). Times come from this machine's
/// clock and the workspace's file system; for a workspace on another
/// machine, a clock ahead of this one only hashes more.
fn settled_before(modified: SystemTime, began: SystemTime) -> bool {
    let whole_seconds = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .is_ok_and(|since| since.subsec_nanos() == 0);
    let precision = if whole_seconds {
        Duration::from_secs(2)
    } else {
        Duration::from_millis(100)
    };
    began
        .checked_sub(precision)
        .is_some_and(|settled| modified < settled)
}

/// What changed from the scan `before` to the scan `after` at the paths
/// both scans could look at (`known`); with `blobs`, where both kept their
/// content, each change with its hunks and text before.
fn scan_changes(
    before: &BTreeMap<PathBuf, Seen>,
    after: &BTreeMap<PathBuf, Seen>,
    blobs: Option<&Path>,
    known: impl Fn(&Path) -> bool,
) -> Vec<FileChange> {
    let paths: BTreeSet<&PathBuf> = before.keys().chain(after.keys()).collect();
    paths
        .into_iter()
        .filter(|path| known(path))
        .filter_map(|path| {
            let shown = path.to_string_lossy();
            let (was, now) = (before.get(path), after.get(path));
            match (was, now) {
                (Some(was), Some(now)) if was.hash.is_some() && was.hash == now.hash => {
                    return None;
                }
                (Some(was), Some(now)) if was.hash.is_none() && was == now => return None,
                (None, None) => return None,
                _ => {}
            }
            let kept = |seen: Option<&Seen>| match (blobs, seen) {
                (_, None) => Some(None),
                (
                    Some(blobs),
                    Some(Seen {
                        hash: Some(hash), ..
                    }),
                ) => std::fs::read(blobs.join(&hash.0)).ok().map(Some),
                _ => None,
            };
            if let (Some(was_bytes), Some(now_bytes)) = (kept(was), kept(now)) {
                return Some(change(&shown, was_bytes, now_bytes.as_deref()));
            }
            let mut change = match (was, now) {
                (None, Some(now)) => FileChange::created(shown, now.lines),
                (Some(was), None) => FileChange::deleted(shown, was.lines),
                _ => FileChange::modified(shown, 0, 0),
            };
            change.before = was.map(Seen::state);
            change.after = now.map(Seen::state);
            Some(change)
        })
        .collect()
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
    /// The snapshotted root as the caller gave it, which names changes.
    root: PathBuf,
    /// The snapshotted root, below `top` (empty for all of it).
    below: PathBuf,
    /// The private directory: `index` and `objects` in it.
    keep: PathBuf,
    /// How `git` starts.
    launcher: Launcher,
}

impl Git {
    /// The git work tree `root` is in, with a private directory under
    /// `keep`; `None` when it is in none, the work tree ignores it (such as
    /// a project under an enclosing repository's `target`), or git cannot
    /// be run. `git` starts as `launcher` starts processes.
    fn find(root: &Path, keep: &Path, launcher: &Launcher) -> Option<Self> {
        let git = |args: &[&OsStr]| {
            let mut git = launcher.command("git");
            git.arg("-C")
                .arg(root)
                .args(args)
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .env_remove("GIT_INDEX_FILE")
                .env_remove("GIT_OBJECT_DIRECTORY");
            Process::start(git, Spawn::default())
                .and_then(|git| git.output(None))
                .ok()
        };
        let output = git(&[
            "rev-parse",
            "--show-toplevel",
            "--absolute-git-dir",
            "--git-path",
            "objects",
        ]
        .map(OsStr::new))
        .filter(|output| output.status.success())?;
        let text = String::from_utf8(output.stdout).ok()?;
        let mut lines = text.lines();
        let (top, dir, objects) = (lines.next()?, lines.next()?, lines.next()?);
        let top = PathBuf::from(top);
        let canonical = std::fs::canonicalize(root).ok()?;
        let below = canonical.strip_prefix(&top).ok()?.to_path_buf();
        // `git add` refuses an ignored root, even with `--ignore-errors`.
        let check = ["check-ignore", "-q", "--"].map(OsStr::new);
        if !below.as_os_str().is_empty()
            && git(&[check.as_slice(), &[canonical.as_os_str()]].concat())
                .is_some_and(|output| output.status.success())
        {
            return None;
        }
        let keep = private(keep, &canonical);
        Some(Self {
            objects: canonical.join(objects),
            root: root.to_path_buf(),
            top,
            dir: PathBuf::from(dir),
            below,
            keep,
            launcher: launcher.clone(),
        })
    }

    /// A git command on the work tree, with the private index and objects,
    /// started as the launcher starts processes.
    fn command(&self, args: &[&OsStr]) -> Command {
        let mut command = self.launcher.command("git");
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
            .env_remove("GIT_WORK_TREE");
        command
    }

    /// What `args` prints, or why it failed.
    fn run(&self, args: &[&OsStr]) -> io::Result<Vec<u8>> {
        self.run_with(args, None)
    }

    fn run_with(&self, args: &[&OsStr], stdin: Option<Vec<u8>>) -> io::Result<Vec<u8>> {
        let how = Spawn {
            stdin,
            merge_stderr: false,
        };
        let output = Process::start(self.command(args), how)?.output(None)?;
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

    /// Adds the root's files, and every file under each of `also` that is
    /// there, to the private index and writes its tree.
    fn write_tree(&self, also: &[PathBuf]) -> io::Result<String> {
        // Listing, replacing the index and writing its tree must belong to
        // one snapshot when calls checkpoint the same root at once.
        static INDEX: Mutex<()> = Mutex::new(());
        let _index = INDEX.lock().unwrap_or_else(PoisonError::into_inner);
        std::fs::create_dir_all(self.keep.join("objects"))?;
        let below = match self.below.as_os_str().is_empty() {
            true => OsStr::new("."),
            false => self.below.as_os_str(),
        };
        let list = [
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
        ]
        .map(OsStr::new);
        let mut listed = self.run(&[list.as_slice(), &[below]].concat())?;
        for path in also
            .iter()
            .filter_map(|path| path.strip_prefix(&self.root).ok())
        {
            let path = self.below.join(path);
            let force = ["ls-files", "--cached", "--others", "-z", "--"].map(OsStr::new);
            listed.extend(self.run(&[force.as_slice(), &[path.as_os_str()]].concat())?);
        }
        let mut files = BTreeMap::new();
        for bytes in listed
            .split(|byte| *byte == 0)
            .filter(|bytes| !bytes.is_empty())
        {
            let bytes = bytes.strip_suffix(b"/").unwrap_or(bytes);
            #[cfg(unix)]
            let path = {
                use std::os::unix::ffi::OsStrExt;
                PathBuf::from(OsStr::from_bytes(bytes))
            };
            #[cfg(not(unix))]
            let path = PathBuf::from(std::str::from_utf8(bytes).map_err(io::Error::other)?);
            let meta = match std::fs::symlink_metadata(self.top.join(&path)) {
                Ok(meta) => meta,
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if meta.is_file()
                || meta.is_symlink()
                || meta.is_dir() && self.top.join(&path).join(".git").exists()
            {
                files.insert(path, meta);
            }
        }
        // Hash the actual bytes, without attributes, clean filters or stat-cache
        // shortcuts: a line-ending-only edit must change the tree too.
        let mut entries = Vec::new();
        let mut entry = |path: &Path, mode: &str, hash: &[u8]| {
            entries.extend_from_slice(mode.as_bytes());
            entries.push(b' ');
            entries.extend_from_slice(hash);
            entries.push(b'\t');
            entries.extend_from_slice(path.as_os_str().as_encoded_bytes());
            entries.push(0);
        };
        let regular: Vec<_> = files.iter().filter(|(_, meta)| meta.is_file()).collect();
        for chunk in regular.chunks(100) {
            let args = ["hash-object", "-w", "--no-filters", "--"].map(OsStr::new);
            let paths = chunk.iter().map(|(path, _)| path.as_os_str());
            let hashes = self.run(&args.into_iter().chain(paths).collect::<Vec<_>>())?;
            let hashes: Vec<_> = hashes
                .split(|byte| *byte == b'\n')
                .filter(|hash| !hash.is_empty())
                .collect();
            if hashes.len() != chunk.len() {
                return Err(io::Error::other("git did not hash every checkpoint file"));
            }
            for ((path, _meta), hash) in chunk.iter().zip(hashes) {
                #[cfg(unix)]
                let executable = {
                    use std::os::unix::fs::PermissionsExt;
                    _meta.permissions().mode() & 0o111 != 0
                };
                #[cfg(not(unix))]
                let executable = false;
                entry(path, if executable { "100755" } else { "100644" }, hash);
            }
        }
        for (path, _) in files.iter().filter(|(_, meta)| meta.is_symlink()) {
            let target = std::fs::read_link(self.top.join(path))?;
            let hash = self.run_with(
                &["hash-object", "-w", "--no-filters", "--stdin"].map(OsStr::new),
                Some(target.as_os_str().as_encoded_bytes().to_vec()),
            )?;
            entry(path, "120000", hash.trim_ascii());
        }
        for (path, _) in files.iter().filter(|(_, meta)| meta.is_dir()) {
            let mut command = self.launcher.command("git");
            command
                .arg("-C")
                .arg(self.top.join(path))
                .args(["rev-parse", "--verify", "HEAD"])
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .env_remove("GIT_INDEX_FILE")
                .env_remove("GIT_OBJECT_DIRECTORY")
                .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES");
            let head = Process::start(command, Spawn::default())?.output(None)?;
            if !head.status.success() {
                return Err(io::Error::other(
                    "a nested repository has no checkpointable HEAD",
                ));
            }
            entry(path, "160000", head.stdout.trim_ascii());
        }
        self.run(&["read-tree", "--empty"].map(OsStr::new))?;
        self.run_with(
            &["update-index", "-z", "--index-info"].map(OsStr::new),
            Some(entries),
        )?;
        let tree = self.run(&[OsStr::new("write-tree")])?;
        Ok(String::from_utf8_lossy(&tree).trim().to_owned())
    }

    /// The bytes of each object in `ids` (an id, or `tree:path`), in order,
    /// read by one `git cat-file --batch`.
    fn blobs(&self, ids: &[&str]) -> io::Result<Vec<Vec<u8>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let input: String = ids.iter().map(|id| format!("{id}\n")).collect();
        let how = Spawn {
            stdin: Some(input.into_bytes()),
            merge_stderr: false,
        };
        let cat = self.command(&["cat-file", "--batch=%(objectsize)"].map(OsStr::new));
        let output = Process::start(cat, how)?.output(None)?;
        let mut rest = output.stdout.as_slice();
        let mut blobs = Vec::with_capacity(ids.len());
        for id in ids {
            // `<size>\n<bytes>\n`, or `<id> missing\n`.
            let missing = || io::Error::new(ErrorKind::NotFound, format!("git has no {id}"));
            let end = rest.iter().position(|byte| *byte == b'\n');
            let header = rest.get(..end.ok_or_else(missing)?).unwrap_or_default();
            let size: usize = String::from_utf8_lossy(header)
                .parse()
                .map_err(|_| missing())?;
            let body = rest.get(header.len() + 1..).unwrap_or_default();
            blobs.push(body.get(..size).ok_or_else(missing)?.to_vec());
            rest = body.get(size + 1..).unwrap_or_default();
        }
        Ok(blobs)
    }

    /// The bytes `path`, under the root as given or the work tree's top,
    /// held in `tree`.
    fn content(&self, tree: &str, path: &Path) -> io::Result<Vec<u8>> {
        let below = match path.strip_prefix(&self.root) {
            Ok(rest) => self.below.join(rest),
            Err(_) => path
                .strip_prefix(&self.top)
                .map_err(|_| {
                    io::Error::new(ErrorKind::InvalidInput, "the path is outside the work tree")
                })?
                .to_path_buf(),
        };
        let spec = format!("{tree}:{}", below.to_string_lossy());
        self.run(&["cat-file", "blob", &spec].map(OsStr::new))
    }

    /// The file git names `path` (below `top`), spelled under the root as
    /// given.
    fn shown(&self, path: &str) -> String {
        match Path::new(path).strip_prefix(&self.below) {
            Ok(rest) => self.root.join(rest),
            Err(_) => self.top.join(path),
        }
        .to_string_lossy()
        .into_owned()
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
        let mut changed = Vec::new();
        while let (Some(header), Some(path)) = (fields.next(), fields.next()) {
            // `:old-mode new-mode old-id new-id status`
            let mut parts = header.split(' ').skip(2);
            if let (Some(old), Some(new)) = (parts.next(), parts.next())
                && old != new
            {
                changed.push((path, old, new));
            }
        }
        // An id of zeros is no file on that side.
        let present = |id: &&str| id.bytes().any(|byte| byte != b'0');
        let sides = changed.iter().flat_map(|(_, old, new)| [*old, *new]);
        let mut blobs = self
            .blobs(&sides.filter(present).collect::<Vec<_>>())?
            .into_iter();
        let mut side = |id: &str| present(&id).then(|| blobs.next()).flatten();
        let changes = changed.into_iter().map(|(path, old, new)| {
            let (was, now) = (side(old), side(new));
            change(&self.shown(path), was, now.as_deref())
        });
        Ok(changes.collect())
    }
}

/// The change from `before` to `after` (`None`: no file) of the file
/// `path`, with its hunks and text before when both are small enough to
/// diff.
fn change(path: &str, before: Option<Vec<u8>>, after: Option<&[u8]>) -> FileChange {
    let small =
        |bytes: Option<&[u8]>| bytes.is_none_or(|bytes| bytes.len() as u64 <= MAX_FILE_BYTES);
    if small(before.as_deref()) && small(after) {
        return Previous::change(before.map(Previous::of), path, after);
    }
    let was = before.as_deref().map(FileState::of);
    let now = after.map(FileState::of);
    let lines = |state: &Option<FileState>| state.as_ref().and_then(|it| it.lines).unwrap_or(0);
    let mut change = match (&was, &now) {
        (_, None) => FileChange::deleted(path, lines(&was)),
        (None, Some(_)) => FileChange::created(path, lines(&now)),
        (Some(_), Some(_)) => FileChange::modified(path, 0, 0),
    };
    change.before = was;
    change.after = now;
    change
}

/// A handler that takes a [`Snapshot`] just before and just after its
/// inner handler serves a call, and publishes what changed as
/// [`FilesChanged`] beside the call's answer, unless the call published its
/// own: so a call of any tool that may write anything, such as `shell`,
/// reports what it changed.
///
/// What it reports is what changed while the call ran, by whatever wrote
/// it: a process outside the call that wrote in that time, such as the
/// user's editor, is in it. A change made between two calls is in neither.
///
/// `newest`, shared between the calls one host checkpoints with the same
/// `root`, is the newest snapshot any of them took: each call hands it to [`snapshot`] as
/// `earlier`, so a scan hashes again only what changed and git reuses its
/// private objects, but never takes it as its "before". `first`, shared
/// the same way, keeps the first "before" taken, the files as they were
/// before the first of those calls, so the host can tell what all of them
/// changed or restore a file. `before`, this call's own, keeps the
/// snapshot taken just before it ran, so the host can tell what it alone
/// changed later on or read a file as it found it. Snapshots run on a
/// thread of their own; one that fails is logged, and the call publishes
/// nothing. A call that streams its answer is served as it is, with
/// nothing published.
#[derive(Clone, Debug)]
pub struct Checkpointed {
    /// The handler that serves the call.
    pub inner: ErasedHandler,
    /// Where the files are.
    pub workspace: Arc<dyn Workspace>,
    /// The directory whose files are compared.
    pub root: PathBuf,
    /// How each snapshot is taken.
    pub taking: Taking,
    /// The newest snapshot taken by any call it shares this with: only
    /// reused as `earlier`, never as a "before".
    pub newest: Arc<Mutex<Option<Snapshot>>>,
    /// The snapshot before the first call, once one was taken.
    pub first: Arc<Mutex<Option<Snapshot>>>,
    /// The snapshot taken just before this call ran, once taken.
    pub before: Arc<Mutex<Option<Snapshot>>>,
}

impl Checkpointed {
    /// A snapshot now, against `earlier`, off this thread.
    async fn take(&self, earlier: Option<Snapshot>) -> Option<(Option<Snapshot>, Snapshot)> {
        let (workspace, root, taking) = (
            self.workspace.clone(),
            self.root.clone(),
            self.taking.clone(),
        );
        let taken = blocking(move || {
            let now = snapshot(&*workspace, &root, &taking, earlier.as_ref())
                .map_err(|error| rig_core::tool::ToolExecutionError::other(error.to_string()))?;
            Ok((earlier, now))
        })
        .await;
        taken
            .inspect_err(|error| {
                tracing::warn!(root = %self.root.display(), "no checkpoint of the call: {error}");
            })
            .ok()
    }
}

impl Serve for Checkpointed {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    fn key(&self) -> HandlerKey {
        self.inner.key().clone()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let published = dispatch.scope::<PublishedContext>();
        let newest = self
            .newest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let before = self.take(newest).await.map(|(_, now)| now);
        if let Some(before) = &before {
            self.first
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert_with(|| before.clone());
            *self.newest.lock().unwrap_or_else(PoisonError::into_inner) = Some(before.clone());
            *self.before.lock().unwrap_or_else(PoisonError::into_inner) = Some(before.clone());
        }
        let reply = self.inner.handle(kind, dispatch).await;
        if !matches!(reply, Reply::Outcome(_)) {
            return reply;
        }
        let Some((Some(before), after)) = self.take(before).await else {
            return reply;
        };
        let changes = after
            .changes_since(&before)
            .inspect_err(|error| {
                tracing::warn!(root = %self.root.display(), "no changes of the call: {error}");
            })
            .unwrap_or_default();
        *self.newest.lock().unwrap_or_else(PoisonError::into_inner) = Some(after);
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
