//! File access for the launcher and the protocol. Each function is the
//! [`std::fs`] one of its name, and its error names the step and the path,
//! as `write /home/u/.rig/bin/rig-agent: No space left on device`, keeping
//! the [`io::ErrorKind`] of the failure. [`optional`] reads only a missing
//! file as absent, so a file that cannot be read is an error, never nothing.
//! [`replace`] replaces a file whole.
//!
//! ```
//! use std::path::Path;
//!
//! use rig_harness_protocol::fs;
//!
//! let missing = fs::optional(fs::read_to_string(Path::new("/nonexistent/rig")))?;
//! assert_eq!(missing, None);
//! # Ok::<(), std::io::Error>(())
//! ```

#![allow(
    clippy::disallowed_methods,
    reason = "the one place that calls std::fs, to name the path in every error"
)]

use std::fs::{DirBuilder, DirEntry, File, Metadata, OpenOptions, Permissions};
use std::io::{self, Write as _};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// `Ok(None)` for a [`io::ErrorKind::NotFound`] failure, the value for
/// success, and every other failure as it is.
///
/// # Errors
///
/// The failure of `result` when it is not NotFound.
pub fn optional<T>(result: io::Result<T>) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(failure) => Err(failure),
    }
}

/// The failure of step `op` on `path`, as `"{op} {path}: {source}"`, for a
/// step on a file this module opened, such as a write to it.
pub fn named(op: &str, path: &Path) -> impl FnOnce(io::Error) -> io::Error {
    move |source| io::Error::new(source.kind(), format!("{op} {}: {source}", path.display()))
}

/// The failure of step `op` from `from` to `to`.
fn named_two(op: &str, from: &Path, to: &Path) -> impl FnOnce(io::Error) -> io::Error {
    move |source| {
        let (from, to) = (from.display(), to.display());
        io::Error::new(source.kind(), format!("{op} {from} to {to}: {source}"))
    }
}

/// [`std::fs::read`].
///
/// # Errors
///
/// When the file cannot be read, naming it.
pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let path = path.as_ref();
    std::fs::read(path).map_err(named("read", path))
}

/// [`std::fs::read_to_string`].
///
/// # Errors
///
/// When the file cannot be read as UTF-8, naming it.
pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
    let path = path.as_ref();
    std::fs::read_to_string(path).map_err(named("read", path))
}

/// [`std::fs::write`].
///
/// # Errors
///
/// When the file cannot be written, naming it.
pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::write(path, contents).map_err(named("write", path))
}

/// [`std::fs::rename`].
///
/// # Errors
///
/// When `from` cannot be renamed to `to`, naming both.
pub fn rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let (from, to) = (from.as_ref(), to.as_ref());
    std::fs::rename(from, to).map_err(named_two("rename", from, to))
}

/// [`std::fs::copy`].
///
/// # Errors
///
/// When `from` cannot be copied to `to`, naming both.
pub fn copy(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<u64> {
    let (from, to) = (from.as_ref(), to.as_ref());
    std::fs::copy(from, to).map_err(named_two("copy", from, to))
}

/// [`std::fs::hard_link`].
///
/// # Errors
///
/// When `to` cannot be made a link to `from`, naming both.
pub fn hard_link(from: impl AsRef<Path>, to: impl AsRef<Path>) -> io::Result<()> {
    let (from, to) = (from.as_ref(), to.as_ref());
    std::fs::hard_link(from, to).map_err(named_two("link", from, to))
}

/// [`std::fs::remove_file`].
///
/// # Errors
///
/// When the file cannot be removed, naming it.
pub fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::remove_file(path).map_err(named("remove", path))
}

/// [`std::fs::remove_dir`].
///
/// # Errors
///
/// When the empty directory cannot be removed, naming it.
pub fn remove_dir(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::remove_dir(path).map_err(named("remove", path))
}

/// [`std::fs::remove_dir_all`].
///
/// # Errors
///
/// When the directory cannot be removed, naming it.
pub fn remove_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::remove_dir_all(path).map_err(named("remove", path))
}

/// [`std::fs::create_dir_all`].
///
/// # Errors
///
/// When the directory cannot be created, naming it.
pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::create_dir_all(path).map_err(named("create", path))
}

/// [`DirBuilder::create`] with `builder`.
///
/// # Errors
///
/// When the directory cannot be created, naming it.
pub fn create_dir(builder: &DirBuilder, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    builder.create(path).map_err(named("create", path))
}

/// [`std::fs::read_dir`], whose entries' errors name the directory too.
///
/// # Errors
///
/// When the directory cannot be listed, naming it.
pub fn read_dir(path: impl AsRef<Path>) -> io::Result<impl Iterator<Item = io::Result<DirEntry>>> {
    let path = path.as_ref().to_path_buf();
    let entries = std::fs::read_dir(&path).map_err(named("list", &path))?;
    Ok(entries.map(move |entry| entry.map_err(named("list", &path))))
}

/// [`std::fs::metadata`].
///
/// # Errors
///
/// When the file cannot be looked at, naming it.
pub fn metadata(path: impl AsRef<Path>) -> io::Result<Metadata> {
    let path = path.as_ref();
    std::fs::metadata(path).map_err(named("look at", path))
}

/// [`std::fs::symlink_metadata`].
///
/// # Errors
///
/// When the file cannot be looked at, naming it.
pub fn symlink_metadata(path: impl AsRef<Path>) -> io::Result<Metadata> {
    let path = path.as_ref();
    std::fs::symlink_metadata(path).map_err(named("look at", path))
}

/// [`std::fs::read_link`].
///
/// # Errors
///
/// When the link cannot be read, naming it.
pub fn read_link(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let path = path.as_ref();
    std::fs::read_link(path).map_err(named("read link", path))
}

/// [`std::fs::canonicalize`].
///
/// # Errors
///
/// When the path cannot be resolved, naming it.
pub fn canonicalize(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let path = path.as_ref();
    std::fs::canonicalize(path).map_err(named("resolve", path))
}

/// [`std::fs::set_permissions`].
///
/// # Errors
///
/// When the permissions cannot be set, naming the path.
pub fn set_permissions(path: impl AsRef<Path>, permissions: Permissions) -> io::Result<()> {
    let path = path.as_ref();
    std::fs::set_permissions(path, permissions).map_err(named("set permissions of", path))
}

/// [`OpenOptions::open`] with `options`.
///
/// # Errors
///
/// When the file cannot be opened, naming it.
pub fn open(options: &OpenOptions, path: impl AsRef<Path>) -> io::Result<File> {
    let path = path.as_ref();
    options.open(path).map_err(named("open", path))
}

/// Replaces the file `target` with `bytes` whole, so a reader sees the old
/// file or the new one. The bytes go to a new file beside it, named for this
/// call alone and never one that was there (such as a planted link), which
/// is private from creation, takes the old file's permissions and group, and
/// is synced before it is renamed over `target`; the folder is synced after
/// where the system allows. A new `target` stays private.
///
/// # Errors
///
/// When `target` names no file, or a step fails, naming it and its path,
/// the old file's group that cannot be kept included. The staged file is
/// removed.
pub fn replace(target: &Path, bytes: &[u8]) -> io::Result<()> {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let (Some(folder), Some(name)) = (target.parent(), target.file_name()) else {
        let why = format!("replace {}: not a file's path", target.display());
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    };
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    let name = name.to_string_lossy();
    let staged = folder.join(format!(".{name}.{}.{call}", std::process::id()));
    let kept = optional(symlink_metadata(target))?.filter(Metadata::is_file);
    // A leftover of a crashed process with the same id, or a link planted
    // here, must not become this copy.
    optional(remove_file(&staged))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let replaced = open(&options, &staged)
        .and_then(|mut file| {
            if let Some(kept) = &kept {
                // A folder may give a new file another group.
                #[cfg(unix)]
                std::os::unix::fs::fchown(&file, None, Some(MetadataExt::gid(kept)))
                    .map_err(named("keep the group of", &staged))?;
                (file.set_permissions(kept.permissions()))
                    .map_err(named("set permissions of", &staged))?;
            }
            file.write_all(bytes).map_err(named("write", &staged))?;
            file.sync_all().map_err(named("sync", &staged))
        })
        .and_then(|()| rename(&staged, target));
    if replaced.is_err() {
        std::fs::remove_file(&staged).ok();
    }
    replaced?;
    // The rename lasts a power loss once the folder is synced. The `.` turns
    // a bare file name's empty folder into the current one.
    open(OpenOptions::new().read(true), folder.join("."))
        .and_then(|folder| folder.sync_all())
        .ok();
    Ok(())
}

#[cfg(test)]
mod tests;
