//! File helpers the tools share: an atomic write, capped text reads, JSON
//! files, and the [`FileChange`] a tool that writes publishes. The reads and
//! the path rules here are this machine's ([`Local`]); a tool reads through
//! its [`Workspace`].

use std::fs::{self, File};
use std::io::{self, ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rig_core::tool::{FileChange, FileState, Hunk, HunkLine, Lines, ToolExecutionError};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::workspace::{Contents, Listing, Local, Stat, Workspace};

/// Reads `path` as text after checking, without opening it, that it is a
/// regular file of at most [`MAX_FILE_BYTES`](crate::MAX_FILE_BYTES):
/// opening a FIFO or a device can block forever. [`Local`]'s
/// `read_text`.
pub fn read_text(path: &str) -> Result<String, ToolExecutionError> {
    local().read_text(Path::new(path))
}

/// The start of a file read by [`read_prefix`].
pub struct Prefix {
    /// At most the asked number of bytes, as text, without a byte-order
    /// mark.
    pub text: String,
    /// The file's size, more than `text` holds when it was cut.
    pub size: u64,
}

impl Prefix {
    /// Whether the file is longer than [`text`](Self::text).
    pub fn cut(&self) -> bool {
        self.size > self.text.len() as u64
    }

    /// `contents`, read with a limit of `cap` bytes, as text without a
    /// byte-order mark. A character cut at the end is dropped; bad bytes
    /// inside are replaced.
    pub fn of(contents: Contents, cap: usize) -> Self {
        let Contents { bytes, size } = contents;
        let mut text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(error) => {
                let valid = error.utf8_error().valid_up_to();
                let mut bytes = error.into_bytes();
                if size > cap as u64 && bytes.len().saturating_sub(valid) < 4 {
                    bytes.truncate(valid);
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }
        };
        if let Some(rest) = text.strip_prefix('\u{feff}') {
            text = rest.to_owned();
        }
        Self { text, size }
    }
}

/// Reads at most `cap` bytes of the regular file at `path` on this machine
/// as a [`Prefix`].
pub fn read_prefix(path: &Path, cap: usize) -> io::Result<Prefix> {
    Ok(Prefix::of(Local.read(path, cap as u64)?, cap))
}

/// Replaces the file at `path` with `contents` in one step, creating its
/// directory: a temporary file beside it is written, synced and renamed
/// over it, so a crash or a full disk never leaves it half written. A
/// symlink's target is replaced and the link kept, and a dangling link's
/// target is created; an existing file keeps its permissions. A read-only
/// file is refused, as is anything but a regular file, because opening a
/// FIFO or a device for writing can block forever.
///
/// Unlike writing in place, the rename needs write access to the directory,
/// gives the file a new inode (a hard link to the old one keeps the old
/// text) and makes this process the file's owner. A crash between create
/// and rename can leave a `.name.<pid>-<n>.tmp` beside the file.
pub fn write_atomic(path: &Path, contents: &[u8]) -> io::Result<()> {
    let (target, permissions) = match fs::metadata(path) {
        Ok(meta) if !meta.is_file() => {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        Ok(meta) if meta.permissions().readonly() => {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "the file is read-only; leave it unchanged or ask the user",
            ));
        }
        Ok(meta) => (fs::canonicalize(path)?, Some(meta.permissions())),
        Err(error) if error.kind() == ErrorKind::NotFound => (link_target(path), None),
        Err(error) => return Err(error),
    };
    if let Some(parent) = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let name = target
        .file_name()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "names no file"))?;
    let temporary = temporary_beside(&target, name);
    let written = File::create_new(&temporary).and_then(|mut file| {
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)?;
        }
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(&temporary, &target)
    });
    written.inspect_err(|_| {
        fs::remove_file(&temporary).ok();
    })
}

/// Writes `contents` to the file `path` names in one step, as
/// [`write_atomic`] does, but without forcing it to disk, and with none of
/// its checks: for scratch that no run after a crash reads, such as a
/// checkpoint's kept content. A reader never sees it half written.
pub(crate) fn write_scratch(path: &Path, contents: &[u8]) -> io::Result<()> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "names no file"))?;
    let temporary = temporary_beside(path, name);
    let written = File::create_new(&temporary)
        .and_then(|mut file| file.write_all(contents))
        .and_then(|()| fs::rename(&temporary, path));
    written.inspect_err(|_| {
        fs::remove_file(&temporary).ok();
    })
}

/// A temporary file's path beside `target`, whose file name is `name`:
/// `.name.<pid>-<n>.tmp`, unique among this process's writes.
fn temporary_beside(target: &Path, name: &std::ffi::OsStr) -> PathBuf {
    /// Tells apart the temporary files of writes running at once.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    target.with_file_name(format!(
        ".{}.{}-{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The JSON file at `path` as a `T`, or `None` when it is missing or is
/// not one.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Replaces the file at `path` with `value` as pretty JSON, by
/// [`write_atomic`].
pub fn write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    write_atomic(path, &serde_json::to_vec_pretty(value)?)
}

/// Where a missing `path` is created: the end of its chain of dangling
/// symlinks, or `path` itself when it is no link.
fn link_target(path: &Path) -> PathBuf {
    /// Linux's own limit on links followed in one lookup.
    const MAX_HOPS: usize = 40;
    let mut target = path.to_path_buf();
    for _ in 0..MAX_HOPS {
        let Ok(link) = fs::read_link(&target) else {
            break;
        };
        target = target.parent().unwrap_or(Path::new("")).join(link);
    }
    target
}

/// A model-visible error for an I/O failure on `path`.
pub fn io_error(path: &str, error: io::Error) -> ToolExecutionError {
    let message = format!("{path}: {error}");
    match error.kind() {
        ErrorKind::NotFound => ToolExecutionError::not_found(message),
        ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => ToolExecutionError::other(message),
    }
}

/// The files `workspace` lists under `root` ([`Workspace::files`]), for a
/// tool to show: a missing `root` is a not-found error, so the model can tell
/// a wrong path from an empty one, and a malformed glob is an
/// invalid-arguments error.
pub(crate) fn listed(
    workspace: &dyn Workspace,
    root: &Path,
    glob: Option<&str>,
    listing: Listing,
) -> Result<Vec<PathBuf>, ToolExecutionError> {
    let shown = root.to_string_lossy();
    let missing = || {
        let missing = io::Error::new(ErrorKind::NotFound, "no such file or directory");
        io_error(&shown, missing)
    };
    let files = workspace
        .files(root, glob, listing)
        .map_err(|error| match error.kind() {
            ErrorKind::InvalidInput => ToolExecutionError::invalid_args(error.to_string()),
            ErrorKind::NotFound => missing(),
            _ => io_error(&shown, error),
        })?;
    // Only an empty listing asks whether `root` is there at all.
    if files.is_empty()
        && workspace
            .stat(root)
            .map_err(|error| io_error(&shown, error))?
            == Stat::Missing
    {
        return Err(missing());
    }
    Ok(files)
}

/// `path` as the file tools open it: made absolute against `base` (`None`:
/// this process's working directory) or, when its first component is `~`,
/// against the home directory, with every symbolic link of its
/// longest existing part followed (the file's own too, when it exists) and
/// `..` folded. Two spellings of one file resolve alike, so this is how the
/// facts the tools publish ([`FilesRead`](rig_core::tool::FilesRead),
/// [`FileChange::path`]) and their footprints name files. A path that
/// cannot be made absolute (no working directory) stays relative.
///
/// ```
/// use std::path::Path;
/// use rig_tools::fs::resolve;
///
/// let here = std::fs::canonicalize(".")?;
/// assert_eq!(resolve(None, "src/../Cargo.toml"), here.join("Cargo.toml"));
/// assert_eq!(resolve(Some(&here), "gone/../new.rs"), here.join("new.rs"));
/// assert_eq!(resolve(Some(Path::new("/nowhere")), "/nowhere/a/../b"), Path::new("/nowhere/b"));
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn resolve(base: Option<&Path>, path: &str) -> PathBuf {
    local().resolve(base, path)
}

/// `path` [`resolve`]d against `base`, as a file tool opens it; with
/// `confine`, refused before any I/O when it lies outside `base` (`None`:
/// the working directory). The rule the file tools follow, for code that
/// must decide as they would, such as a preview of a write.
///
/// ```
/// use std::path::Path;
/// use rig_tools::fs::place;
///
/// let base = std::fs::canonicalize(".")?;
/// assert_eq!(place(Some(&base), true, "src/../a.rs")?, base.join("a.rs"));
/// assert!(place(Some(&base), true, "../outside.rs").is_err());
/// assert!(place(Some(&base), false, "../outside.rs").is_ok());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn place(
    base: Option<&Path>,
    confine: bool,
    path: &str,
) -> Result<PathBuf, ToolExecutionError> {
    local().place(base, confine, path)
}

/// This machine, as a [`Workspace`].
fn local() -> &'static dyn Workspace {
    &Local
}

/// What turning `before` into `after` did to the file at `path` (`None`:
/// there was no file before, or there is none after): its lines added and
/// removed, counted by a line diff, that diff as hunks with 3 lines of
/// context, the text before, and the [`FileState`] of each text: the fact a
/// tool that writes files publishes
/// ([`FilesChanged`](rig_core::tool::FilesChanged)).
///
/// ```
/// use rig_core::tool::ChangeKind;
/// use rig_tools::fs::file_change;
///
/// let created = file_change("a.rs", None, Some("one\ntwo\n"));
/// assert_eq!((created.kind, created.added, created.removed), (ChangeKind::Created, 2, 0));
/// let changed = file_change("a.rs", Some("one\ntwo\n"), Some("one\n2\nthree\n"));
/// assert_eq!((changed.kind, changed.added, changed.removed), (ChangeKind::Modified, 2, 1));
/// assert_eq!(
///     changed.unified(),
///     "--- a.rs\n+++ a.rs\n@@ -1,2 +1,3 @@\n one\n-two\n+2\n+three\n"
/// );
/// assert_eq!(changed.after_text().as_deref(), Some("one\n2\nthree\n"));
/// assert_eq!(changed.before.and_then(|before| before.lines), Some(2));
/// let deleted = file_change("a.rs", Some("one\n"), None);
/// assert_eq!((deleted.kind, deleted.removed), (ChangeKind::Deleted, 1));
/// assert_eq!(deleted.unified(), "--- a.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-one\n");
/// ```
pub fn file_change(path: &str, before: Option<&str>, after: Option<&str>) -> FileChange {
    let mut change = diff(path, before, after);
    change.before_text = before.map(str::to_owned);
    change.before = before.map(|text| FileState::of(text.as_bytes()));
    change.after = after.map(|text| FileState::of(text.as_bytes()));
    change
}

/// [`file_change`] without the text before and the states, for a caller
/// that has them.
pub(crate) fn diff(path: &str, before: Option<&str>, after: Option<&str>) -> FileChange {
    // Past the deadline the diff is still exact, only less minimal.
    let diff = similar::TextDiff::configure()
        .timeout(Duration::from_millis(100))
        .diff_lines(before.unwrap_or_default(), after.unwrap_or_default());
    let hunks: Vec<Hunk> = diff
        .grouped_ops(3)
        .iter()
        .filter_map(|ops| hunk(&diff, ops))
        .collect();
    let lines = hunks.iter().flat_map(|hunk| &hunk.lines);
    let counted = |sign| lines.clone().filter(|line| line.sign() == sign).count();
    let (added, removed) = (counted('+'), counted('-'));
    let mut change = match (before, after) {
        (_, None) => FileChange::deleted(path, removed),
        (None, Some(_)) => FileChange::created(path, added),
        (Some(_), Some(_)) => FileChange::modified(path, added, removed),
    };
    change.hunks = hunks;
    change
}

/// The hunk `ops`, a group of [`similar::TextDiff::grouped_ops`], spans.
fn hunk(diff: &similar::TextDiff<'_, '_, '_, str>, ops: &[similar::DiffOp]) -> Option<Hunk> {
    let (first, last) = (ops.first()?, ops.last()?);
    let span = |start: usize, end: usize| Lines {
        start: start + 1,
        count: end.saturating_sub(start),
    };
    let lines = ops
        .iter()
        .flat_map(|op| diff.iter_changes(op))
        .map(|change| {
            let text = change.value().to_owned();
            match change.tag() {
                similar::ChangeTag::Equal => HunkLine::Same(text),
                similar::ChangeTag::Delete => HunkLine::Removed(text),
                similar::ChangeTag::Insert => HunkLine::Added(text),
            }
        });
    Some(Hunk {
        old: span(first.old_range().start, last.old_range().end),
        new: span(first.new_range().start, last.new_range().end),
        lines: lines.collect(),
    })
}

/// What a file held before a tool changed it.
#[derive(Default)]
pub(crate) struct Previous {
    /// Its text, any bytes that are not UTF-8 replaced, to diff; empty when
    /// it was not read.
    pub(crate) text: String,
    /// Whether [`text`](Self::text) is the file exactly: it was read and is
    /// UTF-8.
    pub(crate) exact: bool,
    /// The file, when it is a regular file.
    pub(crate) state: Option<FileState>,
}

impl Previous {
    /// A file that held `bytes`.
    pub(crate) fn of(bytes: Vec<u8>) -> Self {
        let state = Some(FileState::of(&bytes));
        let (text, exact) = match String::from_utf8(bytes) {
            Ok(text) => (text, true),
            Err(error) => (String::from_utf8_lossy(error.as_bytes()).into(), false),
        };
        Self { text, exact, state }
    }

    /// What a [`read`](Workspace::read) of a file with a limit of
    /// [`MAX_FILE_BYTES`](crate::MAX_FILE_BYTES) found: `None` when there
    /// is no file; one with only its size when it is larger; one with no
    /// state when it is no regular file or could not be read.
    pub(crate) fn read(read: io::Result<Contents>) -> Option<Self> {
        match read {
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Ok(contents) if contents.size > crate::MAX_FILE_BYTES => {
                Some(Self::unread(contents.size))
            }
            Ok(contents) => Some(Self::of(contents.bytes)),
            Err(_) => Some(Self::default()),
        }
    }

    /// A file of `size` bytes that was not read.
    pub(crate) fn unread(size: u64) -> Self {
        Self {
            text: String::new(),
            exact: false,
            state: Some(FileState::unread(size)),
        }
    }

    /// The change from this file (`None`: none) to `after` (`None`: removed):
    /// a [`file_change`] with this file's own state and text (none unless
    /// [`exact`](Self::exact)), and the state of the bytes `after` holds.
    pub(crate) fn change(this: Option<Self>, path: &str, after: Option<&[u8]>) -> FileChange {
        let text = after.map(String::from_utf8_lossy);
        let before = this.as_ref().map(|it| it.text.as_str());
        let mut change = diff(path, before, text.as_deref());
        change.after = after.map(FileState::of);
        if let Some(this) = this {
            change.before = this.state;
            change.before_text = this.exact.then_some(this.text);
        }
        change
    }
}

#[cfg(test)]
mod tests;
