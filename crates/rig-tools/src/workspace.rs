//! The [`Workspace`] the tools act on: the disk they read and write and
//! where their commands run. [`Local`] is this machine; another, such as
//! one reached over SSH, is a type of its own that every tool and the shell
//! then go through.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read as _};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use rig_core::tool::ToolExecutionError;

use crate::MAX_FILE_BYTES;
use crate::fs::{Previous, io_error, write_atomic};
use crate::process::{Launcher, Process, Spawn};
pub use crate::watch::Watch;

/// The disk and the commands the tools act on: this machine's ([`Local`]),
/// or another, such as one reached over SSH. Paths are the workspace's own:
/// a tool's base, the paths it resolves and the facts it publishes
/// ([`FilesRead`](rig_core::tool::FilesRead),
/// [`FilesChanged`](rig_core::tool::FilesChanged)) name files in it.
///
/// Synchronous, as the rest of rig-tools: the tools call it inside
/// [`blocking`](crate::blocking). Code on a thread that must not wait, such
/// as an app's main thread, calls it inline only when [`host`](Self::host)
/// is `None`, and otherwise off that thread.
///
/// A workspace of your own implements every method; [`Local`] is the
/// reference for what each does.
pub trait Workspace: Send + Sync + fmt::Debug + 'static {
    /// Where it is when it is not this machine, such as `ssh://dev@box`;
    /// `None` for this machine.
    fn host(&self) -> Option<&str> {
        None
    }

    /// What `path` is, its links followed: a file (its size, whether it is
    /// read-only, when it was last changed), a directory, something else, or
    /// nothing.
    fn stat(&self, path: &Path) -> io::Result<Stat>;

    /// Up to `limit` bytes of the regular file `path`, and its whole size.
    /// Anything but a regular file is refused without being opened: opening
    /// a FIFO or a device can block forever.
    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents>;

    /// Replaces the file `path` with `bytes` in one step, creating its
    /// directory, keeping its permissions, and refusing a read-only file or
    /// anything but a regular file (see [`write_atomic`]).
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// Removes the file `path`; a directory is refused.
    fn remove(&self, path: &Path) -> io::Result<()>;

    /// Moves the file `from` to `to`, creating `to`'s directory and
    /// replacing a file there.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// `path` made absolute, with the links of its longest existing part
    /// followed (the file's own too, when it exists) and `..` folded, so two
    /// spellings of one file come out alike. A footprint is made with it,
    /// which may be on a thread that must not wait: a remote workspace may
    /// answer from a cache, or fold `..` only.
    fn canonical(&self, path: &Path) -> PathBuf;

    /// The regular files under `root` (or `root` itself when it is a file),
    /// in walk order, as `root` joined with their path below it: without the
    /// files `.gitignore` leaves out, hidden ones (named with a leading `.`)
    /// and `target` directories, and only those matching `glob`, such as
    /// `*.rs`, when it is given. A malformed glob is an
    /// [`InvalidInput`](ErrorKind::InvalidInput) error that says so.
    fn files(&self, root: &Path, glob: Option<&str>) -> io::Result<Vec<PathBuf>>;

    /// Starts `program` with `args` in `dir` (`None`: the launcher's own
    /// directory) under `launcher`'s policy: its wrapper and variables.
    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        how: Spawn,
    ) -> io::Result<Process>;

    /// Watches `root`, a file or a directory (with `recursive`, every
    /// directory under it too), from now on: a batch of the paths created,
    /// changed or removed each time something changed, the files
    /// [`files`](Self::files) would list, or every file with `all`.
    /// Dropping the [`Watch`] stops it. To miss nothing, list the files
    /// once after the watch has started.
    ///
    /// By default it looks at what [`files`](Self::files) lists every
    /// second, on a thread of its own, comparing each file's size and
    /// change time (its hash when [`stat`](Self::stat) cannot tell the
    /// time); `all` adds nothing then, as `files` lists what it lists.
    /// [`Local`] hears of changes from the system instead.
    fn watch(self: Arc<Self>, root: &Path, recursive: bool, all: bool) -> io::Result<Watch> {
        let _ = all;
        crate::watch::poll(self, root, recursive)
    }
}

/// What [`Workspace::read`] read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Contents {
    /// At most the asked number of bytes, from the file's start.
    pub bytes: Vec<u8>,
    /// The file's whole size, more than `bytes` holds when it was cut.
    pub size: u64,
}

/// What a path is, as [`Workspace::stat`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stat {
    /// A regular file.
    File {
        /// Its size in bytes.
        size: u64,
        /// Whether it may not be written.
        read_only: bool,
        /// When its content last changed, when the workspace can tell:
        /// how a scan or a polling watch sees an edit that kept the size.
        modified: Option<SystemTime>,
    },
    /// A directory.
    Dir,
    /// Something else, such as a FIFO or a device: the tools neither read
    /// nor write it.
    Other,
    /// Nothing is there.
    Missing,
}

impl dyn Workspace {
    /// `path` as the file tools name it: joined onto `base` when relative,
    /// then made [`canonical`](Workspace::canonical).
    pub fn resolve(&self, base: Option<&Path>, path: &str) -> PathBuf {
        self.canonical(&base.map_or_else(|| PathBuf::from(path), |base| base.join(path)))
    }

    /// `path` `resolve`d against `base`, as a file tool
    /// opens it; with `confine`, refused before any other I/O when it lies
    /// outside `base` (`None`: the working directory).
    ///
    /// ```
    /// use rig_tools::workspace::{Local, Workspace};
    ///
    /// let local: &dyn Workspace = &Local;
    /// let base = std::fs::canonicalize(".")?;
    /// assert_eq!(local.place(Some(&base), true, "src/../a.rs")?, base.join("a.rs"));
    /// assert!(local.place(Some(&base), true, "../outside.rs").is_err());
    /// assert!(local.place(Some(&base), false, "../outside.rs").is_ok());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn place(
        &self,
        base: Option<&Path>,
        confine: bool,
        path: &str,
    ) -> Result<PathBuf, ToolExecutionError> {
        let resolved = self.resolve(base, path);
        if confine {
            let root = self.resolve(base, ".");
            if !resolved.starts_with(&root) {
                return Err(ToolExecutionError::permission_denied(format!(
                    "{path} is outside {}, the only directory this tool works in",
                    root.display()
                )));
            }
        }
        Ok(resolved)
    }

    /// The regular file `path` as text, after checking, without opening it,
    /// that it is a regular file of at most [`MAX_FILE_BYTES`].
    pub fn read_text(&self, path: &Path) -> Result<String, ToolExecutionError> {
        let shown = path.to_string_lossy();
        match self.stat(path).map_err(|error| io_error(&shown, error))? {
            Stat::File { size, .. } if size > MAX_FILE_BYTES => {
                return Err(ToolExecutionError::invalid_args(format!(
                    "{shown} is {} MB; the tools open files up to {} MB",
                    size / (1024 * 1024),
                    MAX_FILE_BYTES / (1024 * 1024)
                )));
            }
            Stat::File { .. } => {}
            Stat::Missing => {
                return Err(ToolExecutionError::not_found(format!(
                    "{shown}: no such file"
                )));
            }
            Stat::Dir | Stat::Other => {
                return Err(ToolExecutionError::invalid_args(format!(
                    "{shown} is not a regular file"
                )));
            }
        }
        let contents = self
            .read(path, MAX_FILE_BYTES)
            .map_err(|error| io_error(&shown, error))?;
        String::from_utf8(contents.bytes).map_err(|_| {
            ToolExecutionError::other(format!("{shown}: stream did not contain valid UTF-8"))
        })
    }

    /// What the file `path` holds before a tool changes it: `None` when
    /// there is none; empty, with no hash, when it is too large, or no
    /// regular file, to diff.
    pub(crate) fn previous(&self, path: &Path) -> Option<Previous> {
        match self.stat(path) {
            Ok(Stat::Missing) => None,
            Ok(Stat::File { size, .. }) if size <= MAX_FILE_BYTES => {
                let bytes = self.read(path, MAX_FILE_BYTES).ok().map(|it| it.bytes);
                Some(Previous::of(bytes.as_deref()))
            }
            _ => Some(Previous::default()),
        }
    }
}

/// This machine: its file system and its processes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Local;

impl Workspace for Local {
    fn stat(&self, path: &Path) -> io::Result<Stat> {
        match fs::metadata(path) {
            Ok(meta) if meta.is_file() => Ok(Stat::File {
                size: meta.len(),
                read_only: meta.permissions().readonly(),
                modified: meta.modified().ok(),
            }),
            Ok(meta) if meta.is_dir() => Ok(Stat::Dir),
            Ok(_) => Ok(Stat::Other),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Stat::Missing),
            Err(error) => Err(error),
        }
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        if !fs::metadata(path)?.is_file() {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut bytes = Vec::new();
        file.take(limit).read_to_end(&mut bytes)?;
        Ok(Contents { bytes, size })
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        write_atomic(path, bytes)
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        if let Some(parent) = to.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        match fs::rename(from, to) {
            Err(error) if error.kind() == ErrorKind::CrossesDevices => {
                fs::copy(from, to)?;
                fs::remove_file(from)
            }
            moved => moved,
        }
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        let joined = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
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

    fn files(&self, root: &Path, glob: Option<&str>) -> io::Result<Vec<PathBuf>> {
        let mut walk = WalkBuilder::new(root);
        walk.hidden(false).filter_entry(|entry| {
            entry.depth() == 0 || (entry.file_name() != "target" && !is_hidden(entry.path()))
        });
        if let Some(glob) = glob {
            let overrides = OverrideBuilder::new(root)
                .add(glob)
                .and_then(|builder| builder.build())
                .map_err(|error| {
                    io::Error::new(ErrorKind::InvalidInput, format!("bad glob: {error}"))
                })?;
            walk.overrides(overrides);
        }
        // Only regular files: symbolic links are not followed.
        Ok(walk
            .build()
            .flatten()
            .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
            .map(ignore::DirEntry::into_path)
            .collect())
    }

    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        how: Spawn,
    ) -> io::Result<Process> {
        match dir {
            Some(dir) => launcher.clone().in_dir(dir).spawn(program, args, how),
            None => launcher.spawn(program, args, how),
        }
    }

    /// The system's file events under `root`, filtered as
    /// [`files`](Workspace::files) lists unless `all`, gathered into a
    /// batch until no event came for 50 ms. A `root` that does not exist
    /// yet is looked at every second, as the default does.
    fn watch(self: Arc<Self>, root: &Path, recursive: bool, all: bool) -> io::Result<Watch> {
        crate::watch::notified(root, recursive, all)
    }
}

/// Whether `path` is hidden: its name starts with a `.`, such as `.env` or
/// `.github/`.
pub(crate) fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."))
}

#[cfg(test)]
pub(crate) mod tests;
