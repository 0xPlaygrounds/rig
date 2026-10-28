//! The [`Workspace`] the tools act on: the disk they read and write and
//! where their commands run. [`Local`] is this machine; another, such as
//! a [`Remote`](crate::remote::Remote) reached over SSH, is a type of its
//! own that every tool and the shell then go through.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read as _};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use rig_core::tool::{ContentHash, ToolExecutionError, content_hash};

use crate::fs::{Previous, io_error, write_atomic};
use crate::process::{Launcher, Process, Spawn};
pub use crate::watch::Watch;
use crate::{MAX_BYTES, MAX_FILE_BYTES, shell_path};

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

    /// The whole content of each of `paths` that is a regular file of at
    /// most `limit` bytes, in order: `None` for any other path, and for one
    /// that could not be read. What a search reads; a workspace where each
    /// call costs a round trip, such as a remote one, reads them in one.
    fn read_files(&self, paths: &[PathBuf], limit: u64) -> Vec<Option<Vec<u8>>> {
        paths
            .iter()
            .map(|path| read_file(self, path, limit))
            .collect()
    }

    /// Lines `first` (from 1) on, at most `count` of them, of the regular
    /// text file `path`, with how many lines it has and the hash of all of
    /// it. Refused, as `read_text` refuses,
    /// when it is no regular file, larger than [`MAX_FILE_BYTES`] (a
    /// [`FileTooLarge`](ErrorKind::FileTooLarge) error) or not UTF-8 (an
    /// [`InvalidData`](ErrorKind::InvalidData) error); a workspace that
    /// sends only the lines asked of a file larger than
    /// [`MAX_BYTES`], such as a remote one, checks only those for UTF-8.
    ///
    /// By default the whole file is [`read`](Self::read) and cut here.
    fn excerpt(&self, path: &Path, first: usize, count: usize) -> io::Result<Excerpt> {
        excerpt_of(self, path, first, count)
    }

    /// Replaces the file `path` with `bytes` in one step, creating its
    /// directory, keeping its permissions, and refusing a read-only file or
    /// anything but a regular file (see [`write_atomic`]).
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// [`write`](Self::write)s `bytes` to `path` and says what the file
    /// held before, as [`read`](Self::read) with `limit` reads it: a
    /// [`NotFound`](ErrorKind::NotFound) error when there was no file, the
    /// file's own when it could not be read. What a tool that writes a
    /// file diffs against; a workspace where each call costs a round trip,
    /// such as a remote one, does both in one.
    ///
    /// By default it reads, then writes.
    fn replace(&self, path: &Path, bytes: &[u8], limit: u64) -> io::Result<io::Result<Contents>> {
        let before = self.read(path, limit);
        self.write(path, bytes)?;
        Ok(before)
    }

    /// [`write`](Self::write)s `bytes` to the regular file `path` only
    /// while it holds the text whose [`content_hash`] is `held`: `None`
    /// when it did. Otherwise nothing is written and it says what the file
    /// holds instead, as [`read`](Self::read) with a limit of
    /// [`MAX_FILE_BYTES`] reads it. What an edit made from text seen
    /// before writes with; a workspace where each call costs a round trip,
    /// such as a remote one, checks and writes in one.
    ///
    /// By default it reads, then writes.
    fn write_if(
        &self,
        path: &Path,
        bytes: &[u8],
        held: &ContentHash,
    ) -> io::Result<Option<Contents>> {
        let now = self.read(path, MAX_FILE_BYTES)?;
        if now.size > MAX_FILE_BYTES || content_hash(&now.bytes) != *held {
            return Ok(Some(now));
        }
        self.write(path, bytes)?;
        Ok(None)
    }

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

    /// `path` relative to `base`, as `resolve` joins them:
    /// `base.join(path)`, so an absolute `path` stays as it is. A workspace
    /// whose paths can name another machine, such as
    /// [`Hosts`](crate::remote::Hosts), keeps such a path whole too, and one
    /// with a home directory reads a first component `~` as that home
    /// ([`Local`]: `~`, `~/notes` and, on Windows, `~\notes`; not `~bob` or
    /// `./~`).
    fn join(&self, base: &Path, path: &Path) -> PathBuf {
        base.join(path)
    }

    /// The regular files under `root` (or `root` itself when it is a file),
    /// in walk order, as `root` joined with their path below it: those
    /// `listing` keeps ([`Listing`]), and only those matching `glob`, such
    /// as `*.rs`, when it is given. A malformed glob is an
    /// [`InvalidInput`](ErrorKind::InvalidInput) error that says so.
    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Vec<PathBuf>>;

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
    /// changed or removed each time something changed, of the files
    /// [`files`](Self::files) would list with `listing`.
    /// Dropping the [`Watch`] stops it. To miss nothing, list the files
    /// once after the watch has started.
    ///
    /// By default it looks at what [`files`](Self::files) lists every
    /// second, on a thread of its own, comparing each file's
    /// size and change time (its hash when [`stat`](Self::stat) cannot tell
    /// the time). [`Local`] hears of changes from the system instead.
    fn watch(self: Arc<Self>, root: &Path, recursive: bool, listing: Listing) -> io::Result<Watch> {
        crate::watch::poll(self, root, recursive, listing)
    }
}

/// Which files [`Workspace::files`] lists, and a watch reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Listing {
    /// What a person browsing the project sees, as ripgrep lists it: no
    /// file the ignore rules (`.gitignore`, `.ignore`, the git excludes)
    /// leave out, no hidden file or directory (named with a leading `.`),
    /// no `target` directory.
    #[default]
    Visible,
    /// Every file the ignore rules keep, hidden ones such as `.github/` and
    /// `.env.example` included; `.git` left out.
    Hidden,
    /// Every regular file.
    All,
}

impl Listing {
    /// The listing of a tool call's `hidden` and `glob` arguments.
    pub(crate) fn from_options(hidden: bool, glob: Option<&str>) -> Self {
        match hidden {
            true => Self::Hidden,
            false => Self::Visible,
        }
        .for_glob(glob)
    }

    /// The listing a search or listing for `glob` uses: `Visible` becomes
    /// `Hidden` when the glob names a hidden file or directory, such as
    /// `.github/**` or `**/.env*`, so asking for one finds it.
    pub fn for_glob(self, glob: Option<&str>) -> Self {
        let names_hidden = glob.is_some_and(|glob| {
            glob.split(['/', '\\'])
                .any(|part| part.starts_with('.') && part != "." && part != "..")
        });
        match self {
            Self::Visible if names_hidden => Self::Hidden,
            other => other,
        }
    }

    /// Whether this listing keeps `below`, a path below the listed root, by
    /// its names alone (the ignore rules aside): `Visible` none of whose
    /// parts is hidden or `target`, `Hidden` none of whose parts is `.git`.
    pub fn keeps(self, below: &Path) -> bool {
        let mut parts = below.iter();
        match self {
            Self::Visible => !parts.any(|part| part == "target" || is_hidden(Path::new(part))),
            Self::Hidden => !parts.any(|part| part == ".git"),
            Self::All => true,
        }
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

/// What [`Workspace::excerpt`] read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Excerpt {
    /// The lines asked, each with its line end as in the file.
    pub text: String,
    /// How many lines the file has, as [`str::lines`] counts them.
    pub total: usize,
    /// The hash of the whole file.
    pub hash: ContentHash,
    /// The whole file's text, when it is at most [`MAX_BYTES`]: sending
    /// that much costs about what a call does, and an edit after the read
    /// starts from it instead of fetching the file.
    pub whole: Option<String>,
}

impl Excerpt {
    /// Lines `first` (from 1) on, at most `count` of them, of `text`, the
    /// whole file, kept whole too when it is small.
    pub fn of(text: &str, first: usize, count: usize) -> Self {
        let start = text
            .split_inclusive('\n')
            .take(first.saturating_sub(1))
            .map(str::len)
            .sum();
        let rest = text.get(start..).unwrap_or_default();
        let len = rest.split_inclusive('\n').take(count).map(str::len).sum();
        Self {
            text: rest.get(..len).unwrap_or_default().to_owned(),
            total: text.lines().count(),
            hash: content_hash(text.as_bytes()),
            whole: (text.len() <= MAX_BYTES).then(|| text.to_owned()),
        }
    }
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
    /// `path` as the file tools name it: [`join`](Workspace::join)ed onto
    /// `base` (`None`: the working directory), then made
    /// [`canonical`](Workspace::canonical).
    pub fn resolve(&self, base: Option<&Path>, path: &str) -> PathBuf {
        self.canonical(&self.join(base.unwrap_or(Path::new("")), path.as_ref()))
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

    /// The regular file `path` as text, in one [`read`](Workspace::read):
    /// refused when it is no regular file or larger than
    /// [`MAX_FILE_BYTES`].
    pub fn read_text(&self, path: &Path) -> Result<String, ToolExecutionError> {
        let shown = path.to_string_lossy();
        self.read(path, MAX_FILE_BYTES)
            .and_then(text_of)
            .map_err(|error| text_error(&shown, error))
    }

    /// What the file `path` holds before a tool changes it, in one
    /// [`read`](Workspace::read): `None` when there is none; empty, with
    /// only its size, when it is too large to read; empty, with no state,
    /// when it is no regular file or could not be read.
    pub(crate) fn previous(&self, path: &Path) -> Option<Previous> {
        Previous::read(self.read(path, MAX_FILE_BYTES))
    }
}

/// The whole text of a file read with a limit of [`MAX_FILE_BYTES`]:
/// refused when it is larger or not UTF-8.
pub(crate) fn text_of(contents: Contents) -> io::Result<String> {
    if contents.size > MAX_FILE_BYTES {
        return Err(too_large(contents.size));
    }
    String::from_utf8(contents.bytes).map_err(|_| not_text())
}

/// The error for a file of `size` bytes, more than [`MAX_FILE_BYTES`].
pub(crate) fn too_large(size: u64) -> io::Error {
    io::Error::new(
        ErrorKind::FileTooLarge,
        format!(
            "{} MB; the tools open files up to {} MB",
            size / (1024 * 1024),
            MAX_FILE_BYTES / (1024 * 1024)
        ),
    )
}

/// The error for bytes that are not UTF-8 text.
pub(crate) fn not_text() -> io::Error {
    io::Error::new(ErrorKind::InvalidData, "stream did not contain valid UTF-8")
}

/// A model-visible error for a failed text read of `shown`
/// (`read_text`, [`Workspace::excerpt`]).
pub(crate) fn text_error(shown: &str, error: io::Error) -> ToolExecutionError {
    match error.kind() {
        ErrorKind::NotFound => ToolExecutionError::not_found(format!("{shown}: no such file")),
        ErrorKind::InvalidInput | ErrorKind::IsADirectory => {
            ToolExecutionError::invalid_args(format!("{shown} is not a regular file"))
        }
        ErrorKind::FileTooLarge => ToolExecutionError::invalid_args(format!("{shown} is {error}")),
        ErrorKind::InvalidData => ToolExecutionError::invalid_args(match shell_path(shown) {
            Some(path) => format!(
                "{shown} is not a text file (not UTF-8); `file {path}` in `shell` shows what it is"
            ),
            None => format!("{shown} is not a text file (not UTF-8)"),
        }),
        _ => io_error(shown, error),
    }
}

/// [`Workspace::excerpt`] as it is by default: the whole file read and cut
/// here.
pub(crate) fn excerpt_of<W: Workspace + ?Sized>(
    workspace: &W,
    path: &Path,
    first: usize,
    count: usize,
) -> io::Result<Excerpt> {
    let text = text_of(workspace.read(path, MAX_FILE_BYTES)?)?;
    Ok(Excerpt::of(&text, first, count))
}

/// `path`'s whole content when it is a regular file of at most `limit`
/// bytes, as [`Workspace::read_files`] reads each by default.
pub(crate) fn read_file<W: Workspace + ?Sized>(
    workspace: &W,
    path: &Path,
    limit: u64,
) -> Option<Vec<u8>> {
    let small = matches!(workspace.stat(path), Ok(Stat::File { size, .. }) if size <= limit);
    small
        .then(|| workspace.read(path, limit).ok())
        .flatten()
        .map(|contents| contents.bytes)
}

/// `path` in this machine's home directory when its first component is
/// `~`: `None` for any other path, and when the home is unknown.
pub(crate) fn home_path(path: &Path) -> Option<PathBuf> {
    let rest = path.strip_prefix("~").ok()?;
    let home = std::env::home_dir()?;
    Some(match rest.as_os_str().is_empty() {
        true => home,
        false => home.join(rest),
    })
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

    /// `~` is this machine's home directory, when it is known.
    fn join(&self, base: &Path, path: &Path) -> PathBuf {
        home_path(path).unwrap_or_else(|| base.join(path))
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

    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Vec<PathBuf>> {
        let mut walk = WalkBuilder::new(root);
        match listing {
            Listing::All => walk.standard_filters(false),
            Listing::Visible | Listing::Hidden => walk.hidden(false).filter_entry(move |entry| {
                entry.depth() == 0 || listing.keeps(Path::new(entry.file_name()))
            }),
        };
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
    /// [`files`](Workspace::files) lists with `listing`, gathered into a
    /// batch until no event came for 50 ms. A `root` that does not exist
    /// yet is looked at every second, as the default does.
    fn watch(self: Arc<Self>, root: &Path, recursive: bool, listing: Listing) -> io::Result<Watch> {
        crate::watch::notified(root, recursive, listing)
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
