//! [`Overlay`]: a workspace that keeps its writes to itself, to learn what
//! a tool would leave on disk without changing it.

use std::ffi::{OsStr, OsString};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ignore::overrides::OverrideBuilder;
use rig_core::tool::{FileChange, content_hash};

use crate::fs::Previous;
use crate::process::{Launcher, Process, Spawn};
use crate::workspace::{Contents, Listing, Stat, Workspace};

/// A [`Workspace`] over another that reads through to it and keeps every
/// write, removal and rename to itself: run a file tool on it to learn what
/// the tool would leave on disk ([`changes`](Self::changes)) while the disk
/// stays as it is. It runs no commands, since what a command changes cannot
/// be kept from the disk.
///
/// ```
/// use std::sync::Arc;
/// use rig_core::tool::ChangeKind;
/// use rig_tools::overlay::Overlay;
/// use rig_tools::workspace::{Local, Workspace};
///
/// let path = std::env::temp_dir().join(format!("overlay-doc-{}.txt", std::process::id()));
/// let overlay = Overlay::new(Arc::new(Local));
/// overlay.write(&path, b"would be written\n")?;
/// assert!(!path.exists());
/// let changes = overlay.changes();
/// assert_eq!(changes.iter().map(|change| change.kind).collect::<Vec<_>>(), [ChangeKind::Created]);
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Debug)]
pub struct Overlay {
    /// The workspace read through to.
    inner: Arc<dyn Workspace>,
    /// Each path written or removed here, in the order first touched: its
    /// bytes now, or `None` once removed.
    kept: Mutex<Vec<(PathBuf, Option<Arc<[u8]>>)>>,
}

impl Overlay {
    /// An overlay over `inner`, with nothing written yet.
    pub fn new(inner: Arc<dyn Workspace>) -> Self {
        Self {
            inner,
            kept: Mutex::new(Vec::new()),
        }
    }

    /// The paths written or removed here.
    fn kept(&self) -> MutexGuard<'_, Vec<(PathBuf, Option<Arc<[u8]>>)>> {
        self.kept.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// What this overlay holds for `path`: `Some(Some(bytes))` written,
    /// `Some(None)` removed, `None` untouched.
    fn held(&self, path: &Path) -> Option<Option<Arc<[u8]>>> {
        self.kept()
            .iter()
            .find(|(kept, _)| kept == path)
            .map(|(_, bytes)| bytes.clone())
    }

    /// Keeps `bytes` (`None`: removed) for `path`.
    fn keep(&self, path: &Path, bytes: Option<Arc<[u8]>>) {
        let mut kept = self.kept();
        match kept.iter_mut().find(|(kept, _)| kept == path) {
            Some((_, held)) => *held = bytes,
            None => kept.push((path.to_path_buf(), bytes)),
        }
    }

    /// What the writes kept here would do to the files of the workspace
    /// under it, in the order each file was first touched: created,
    /// modified or deleted, with its hunks, its text before and the
    /// [`FileState`] of each side (the [`FileState::unread`] size of a file
    /// too large to read). A file written back as it was, or created and
    /// removed again, is no change.
    ///
    /// [`FileState`]: rig_core::tool::FileState
    /// [`FileState::unread`]: rig_core::tool::FileState::unread
    pub fn changes(&self) -> Vec<FileChange> {
        let kept = self.kept().clone();
        kept.into_iter()
            .filter_map(|(path, after)| {
                let before = match self.inner.stat(&path) {
                    Ok(Stat::File { .. }) => self.inner.previous(&path),
                    _ => None,
                };
                let unchanged = |before: &Previous, after: &[u8]| {
                    let hash = before.state.as_ref().and_then(|it| it.hash.as_ref());
                    hash.is_some_and(|hash| *hash == content_hash(after))
                };
                match (&before, &after) {
                    (None, None) => None,
                    (Some(before), Some(after)) if unchanged(before, after) => None,
                    _ => Some(Previous::change(
                        before,
                        &path.to_string_lossy(),
                        after.as_deref(),
                    )),
                }
            })
            .collect()
    }

    /// The bytes of the file `path` as this overlay sees it, whole.
    fn bytes(&self, path: &Path) -> io::Result<Arc<[u8]>> {
        match self.held(path) {
            Some(Some(bytes)) => Ok(bytes),
            Some(None) => Err(ErrorKind::NotFound.into()),
            None => self
                .inner
                .read(path, u64::MAX)
                .map(|contents| contents.bytes.into()),
        }
    }
}

impl Workspace for Overlay {
    fn host(&self) -> Option<&str> {
        self.inner.host()
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        match self.held(path) {
            Some(Some(bytes)) => Ok(Stat::File {
                size: bytes.len() as u64,
                read_only: false,
                modified: None,
            }),
            Some(None) => Ok(Stat::Missing),
            None => match self.inner.stat(path)? {
                Stat::Missing
                    if self
                        .kept()
                        .iter()
                        .any(|(kept, bytes)| bytes.is_some() && kept.starts_with(path)) =>
                {
                    Ok(Stat::Dir)
                }
                stat => Ok(stat),
            },
        }
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        match self.held(path) {
            Some(Some(bytes)) => Ok(Contents {
                size: bytes.len() as u64,
                bytes: bytes
                    .get(..limit.try_into().unwrap_or(usize::MAX))
                    .unwrap_or(&bytes)
                    .to_vec(),
            }),
            Some(None) => Err(ErrorKind::NotFound.into()),
            None => self.inner.read(path, limit),
        }
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        match self.stat(path)? {
            Stat::Dir | Stat::Other => Err(io::Error::new(
                ErrorKind::InvalidInput,
                "not a regular file",
            )),
            Stat::File {
                read_only: true, ..
            } => Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "the file is read-only; leave it unchanged or ask the user",
            )),
            Stat::File { .. } | Stat::Missing => {
                self.keep(path, Some(bytes.into()));
                Ok(())
            }
        }
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        match self.stat(path)? {
            Stat::Missing => Err(ErrorKind::NotFound.into()),
            Stat::Dir => Err(io::Error::new(ErrorKind::IsADirectory, "is a directory")),
            Stat::File { .. } | Stat::Other => {
                self.keep(path, None);
                Ok(())
            }
        }
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let bytes = self.bytes(from)?;
        self.keep(to, Some(bytes));
        self.keep(from, None);
        Ok(())
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        self.inner.canonical(path)
    }

    fn join(&self, base: &Path, path: &Path) -> PathBuf {
        self.inner.join(base, path)
    }

    /// The files under it, as listed there, without those removed here,
    /// then the files created here that `listing` keeps by their names
    /// ([`Listing::keeps`]; ignore files are not read for them).
    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Vec<PathBuf>> {
        let listed = self.inner.files(root, glob, listing)?;
        let matcher = glob
            .map(|glob| OverrideBuilder::new(root).add(glob)?.build())
            .transpose()
            .map_err(|error| {
                io::Error::new(ErrorKind::InvalidInput, format!("bad glob: {error}"))
            })?;
        let kept = self.kept().clone();
        let removed = |path: &PathBuf| {
            kept.iter()
                .any(|(kept, bytes)| kept == path && bytes.is_none())
        };
        let created = kept.iter().filter_map(|(path, bytes)| {
            let visible = listing.keeps(path.strip_prefix(root).ok()?);
            let matches = matcher
                .as_ref()
                .is_none_or(|matcher| matcher.matched(path, false).is_whitelist());
            (bytes.is_some() && visible && matches && !listed.contains(path)).then(|| path.clone())
        });
        let mut files: Vec<PathBuf> = listed
            .iter()
            .filter(|path| !removed(path))
            .cloned()
            .collect();
        files.extend(created);
        Ok(files)
    }

    fn spawn(
        &self,
        _: &Launcher,
        _: Option<&Path>,
        _: &OsStr,
        _: &[OsString],
        _: Spawn,
    ) -> io::Result<Process> {
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "an overlay runs no commands: what a command changes cannot be kept from the disk",
        ))
    }
}

#[cfg(test)]
mod tests;
