//! The built-in tools: read, edit, write, shell and search. Each runs its
//! blocking work on a thread of its own, so no task pool thread blocks.

mod edit;
mod read;
mod search;
mod shell;
mod write;

use std::fs::{self, File};
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_app::prelude::*;
use rig_core::tool::ToolExecutionError;

use crate::core::tools::AppToolsExt;

pub use edit::Edit;
pub use read::Read;
pub use search::Search;
pub use shell::Shell;
pub use write::Write;

/// Most lines a tool returns.
const MAX_LINES: usize = 2000;
/// Most bytes a tool returns.
const MAX_BYTES: usize = 50 * 1024;
/// The largest file `read` and `edit` load whole, and `search` scans.
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Registers the built-in tools with [`AppToolsExt::add_tool`].
#[derive(Default)]
pub struct BuiltinToolsPlugin;

impl Plugin for BuiltinToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(Read)
            .add_tool(Edit)
            .add_tool(Write)
            .add_tool(Shell)
            .add_tool(Search);
    }
}

/// `line` cut to its first `limit` characters.
pub(crate) fn clip(line: &str, limit: usize) -> &str {
    match line.char_indices().nth(limit) {
        Some((end, _)) => line.get(..end).unwrap_or(line),
        None => line,
    }
}

/// Reads `path` as text after checking, without opening it, that it is a
/// regular file of at most [`MAX_FILE_BYTES`]: opening a FIFO or a device
/// can block forever.
fn read_text(path: &str) -> Result<String, ToolExecutionError> {
    let meta = std::fs::metadata(path).map_err(|error| io_error(path, error))?;
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
    std::fs::read_to_string(path).map_err(|error| io_error(path, error))
}

/// Replaces the file at `path` with `contents` in one step: a temporary
/// file beside it is written, synced and renamed over it, so a crash or a
/// full disk never leaves it half written. A symlink's target is replaced
/// and the link kept, and a dangling link's target is created; an existing
/// file keeps its permissions. A read-only file is refused, as is anything
/// but a regular file, because opening a FIFO or a device for writing can
/// block forever.
///
/// Unlike writing in place, the rename needs write access to the directory,
/// gives the file a new inode (a hard link to the old one keeps the old
/// text) and makes this process the file's owner. A crash between create
/// and rename can leave a `.name.<pid>-<n>.tmp` beside the file.
fn write_atomic(path: &str, contents: &[u8]) -> Result<(), ToolExecutionError> {
    /// Tells apart the temporary files of calls running at once.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (target, permissions) = match fs::metadata(path) {
        Ok(meta) if !meta.is_file() => {
            return Err(ToolExecutionError::invalid_args(format!(
                "{path} is not a regular file"
            )));
        }
        Ok(meta) if meta.permissions().readonly() => {
            return Err(ToolExecutionError::invalid_args(format!(
                "{path} is read-only; leave it unchanged or ask the user"
            )));
        }
        Ok(meta) => (
            fs::canonicalize(path).map_err(|error| io_error(path, error))?,
            Some(meta.permissions()),
        ),
        Err(error) if error.kind() == ErrorKind::NotFound => (link_target(path), None),
        Err(error) => return Err(io_error(path, error)),
    };
    let name = target
        .file_name()
        .ok_or_else(|| ToolExecutionError::invalid_args(format!("{path} names no file")))?;
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
    written.map_err(|error| {
        fs::remove_file(&temporary).ok();
        io_error(path, error)
    })
}

/// Where a missing `path` is created: the end of its chain of dangling
/// symlinks, or `path` itself when it is no link.
fn link_target(path: &str) -> PathBuf {
    /// Linux's own limit on links followed in one lookup.
    const MAX_HOPS: usize = 40;
    let mut target = PathBuf::from(path);
    for _ in 0..MAX_HOPS {
        let Ok(link) = fs::read_link(&target) else {
            break;
        };
        target = target.parent().unwrap_or(Path::new("")).join(link);
    }
    target
}

/// A model-visible error for an I/O failure on `path`.
fn io_error(path: &str, error: std::io::Error) -> ToolExecutionError {
    let message = format!("{path}: {error}");
    match error.kind() {
        std::io::ErrorKind::NotFound => ToolExecutionError::not_found(message),
        std::io::ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => ToolExecutionError::other(message),
    }
}
