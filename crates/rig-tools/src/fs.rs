//! File helpers the tools share: an atomic write, capped text reads, JSON
//! files, and the [`FileChange`] a tool that writes publishes.

use std::fs::{self, File};
use std::io::{self, ErrorKind, Read as _, Write as _};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rig_core::tool::{FileChange, ToolExecutionError};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::MAX_FILE_BYTES;

/// Reads `path` as text after checking, without opening it, that it is a
/// regular file of at most [`MAX_FILE_BYTES`]: opening a FIFO or a device
/// can block forever.
pub fn read_text(path: &str) -> Result<String, ToolExecutionError> {
    let meta = fs::metadata(path).map_err(|error| io_error(path, error))?;
    if !meta.is_file() {
        return Err(ToolExecutionError::invalid_args(format!(
            "{path} is not a regular file"
        )));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(ToolExecutionError::invalid_args(format!(
            "{path} is {} MB; the tools open files up to {} MB",
            meta.len() / (1024 * 1024),
            MAX_FILE_BYTES / (1024 * 1024)
        )));
    }
    fs::read_to_string(path).map_err(|error| io_error(path, error))
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
}

/// Reads at most `cap` bytes of the file at `path` as text, without a
/// byte-order mark. A character cut at the end is dropped; bad bytes inside
/// are replaced.
pub fn read_prefix(path: &Path, cap: usize) -> io::Result<Prefix> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut bytes = Vec::new();
    file.take(cap as u64).read_to_end(&mut bytes)?;
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
    Ok(Prefix { text, size })
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
    /// Tells apart the temporary files of writes running at once.
    static NEXT: AtomicU64 = AtomicU64::new(0);
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
    let temporary = target.with_file_name(format!(
        ".{}.{}-{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
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

/// `path` as the file tools open it: made absolute against `base` (`None`:
/// this process's working directory), with every symbolic link of its
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
    let joined = base.map_or_else(|| PathBuf::from(path), |base| base.join(path));
    let joined = std::path::absolute(&joined).unwrap_or(joined);
    let real = joined.ancestors().find_map(|ancestor| {
        let real = fs::canonicalize(ancestor).ok()?;
        Some((real, joined.strip_prefix(ancestor).ok()?))
    });
    let (mut out, rest) = real.unwrap_or((PathBuf::new(), &joined));
    for component in rest.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
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
    let resolved = resolve(base, path);
    if confine {
        let root = resolve(base, ".");
        if !resolved.starts_with(&root) {
            return Err(ToolExecutionError::permission_denied(format!(
                "{path} is outside {}, the only directory this tool works in",
                root.display()
            )));
        }
    }
    Ok(resolved)
}

/// What turning `before` (`None`: there was no file) into `after` did to the
/// file at `path`: its lines added and removed, counted by a line diff, and
/// that diff, unified with 3 lines of context: the fact a tool that writes
/// files publishes ([`FilesChanged`](rig_core::tool::FilesChanged)).
///
/// ```
/// use rig_core::tool::ChangeKind;
/// use rig_tools::fs::file_change;
///
/// let created = file_change("a.rs", None, "one\ntwo\n");
/// assert_eq!((created.kind, created.added, created.removed), (ChangeKind::Created, 2, 0));
/// let changed = file_change("a.rs", Some("one\ntwo\n"), "one\n2\nthree\n");
/// assert_eq!((changed.kind, changed.added, changed.removed), (ChangeKind::Modified, 2, 1));
/// assert_eq!(
///     changed.diff.as_deref(),
///     Some("--- a.rs\n+++ a.rs\n@@ -1,2 +1,3 @@\n one\n-two\n+2\n+three\n")
/// );
/// ```
pub fn file_change(path: &str, before: Option<&str>, after: &str) -> FileChange {
    let diff = similar::TextDiff::from_lines(before.unwrap_or_default(), after);
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(before.map_or("/dev/null", |_| path), path)
        .to_string();
    let (mut added, mut removed) = (0, 0);
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => added += 1,
            similar::ChangeTag::Delete => removed += 1,
            similar::ChangeTag::Equal => {}
        }
    }
    match before {
        None => FileChange::created(path, added),
        Some(_) => FileChange::modified(path, added, removed),
    }
    .with_diff(unified)
}

#[cfg(test)]
mod tests;
