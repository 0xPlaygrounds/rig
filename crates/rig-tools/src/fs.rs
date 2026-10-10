//! File helpers the tools share: an atomic write, capped text reads and
//! JSON files.

use std::fs::{self, File};
use std::io::{self, ErrorKind, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rig_core::tool::ToolExecutionError;
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
