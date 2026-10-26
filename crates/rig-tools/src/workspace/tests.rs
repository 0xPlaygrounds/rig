use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::io::{self, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use rig_core::tool::{
    ChangeKind, ContentHash, FilesChanged, Tool, ToolContext, ToolOutput, content_hash,
};

use super::{Contents, Listing, Stat, Workspace};
use crate::process::{Launcher, Process, Spawn};
use crate::{
    Delete, DeleteArgs, Edit, EditArgs, Move, MoveArgs, Read, ReadArgs, Replacement, Search,
    SearchArgs, Write, WriteArgs,
};

/// Files held in memory, by absolute path: a workspace that is not this
/// machine's disk.
#[derive(Debug, Default)]
pub(crate) struct Memory(Mutex<BTreeMap<PathBuf, Vec<u8>>>);

impl Memory {
    pub(crate) fn files(&self) -> std::sync::MutexGuard<'_, BTreeMap<PathBuf, Vec<u8>>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Workspace for Memory {
    fn host(&self) -> Option<&str> {
        Some("memory://")
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        let files = self.files();
        Ok(match files.get(path) {
            Some(bytes) => Stat::File {
                size: bytes.len() as u64,
                read_only: false,
                modified: None,
            },
            None if files.keys().any(|file| file.starts_with(path)) => Stat::Dir,
            None => Stat::Missing,
        })
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        let files = self.files();
        let bytes = files.get(path).ok_or(ErrorKind::NotFound)?;
        Ok(Contents {
            bytes: bytes.iter().copied().take(limit as usize).collect(),
            size: bytes.len() as u64,
        })
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.files().insert(path.to_path_buf(), bytes.to_vec());
        Ok(())
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        self.files()
            .remove(path)
            .map(drop)
            .ok_or(ErrorKind::NotFound.into())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut files = self.files();
        let bytes = files.remove(from).ok_or(ErrorKind::NotFound)?;
        files.insert(to.to_path_buf(), bytes);
        Ok(())
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        let mut out = PathBuf::from("/");
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    out.pop();
                }
                Component::Normal(part) => out.push(part),
                _ => {}
            }
        }
        out
    }

    fn files(&self, root: &Path, glob: Option<&str>, _: Listing) -> io::Result<Vec<PathBuf>> {
        let suffix = glob.map(|glob| glob.trim_start_matches('*'));
        Ok(self
            .files()
            .keys()
            .filter(|path| path.starts_with(root))
            .filter(|path| suffix.is_none_or(|suffix| path.to_string_lossy().ends_with(suffix)))
            .cloned()
            .collect())
    }

    fn spawn(
        &self,
        _: &Launcher,
        _: Option<&Path>,
        _: &OsStr,
        _: &[OsString],
        _: Spawn,
    ) -> io::Result<Process> {
        Err(ErrorKind::Unsupported.into())
    }
}

fn call<T: Tool<Error = rig_core::tool::ToolExecutionError>>(
    tool: &T,
    args: T::Args,
) -> (
    Result<T::Output, rig_core::tool::ToolExecutionError>,
    ToolContext,
) {
    let mut context = ToolContext::new();
    let output = futures::executor::block_on(tool.call(&mut context, args));
    (output, context)
}

fn changes(context: &ToolContext) -> Vec<(String, ChangeKind)> {
    let changed: FilesChanged = context.result().ok().flatten().unwrap_or_default();
    changed
        .0
        .into_iter()
        .map(|change| (change.path, change.kind))
        .collect()
}

/// The content each changed file held before the call, by its hash.
fn befores(context: &ToolContext) -> Vec<Option<ContentHash>> {
    let changed: FilesChanged = context.result().ok().flatten().unwrap_or_default();
    changed
        .0
        .into_iter()
        .map(|change| change.before.and_then(|it| it.hash))
        .collect()
}

/// Every file tool acts on its workspace, here files in memory: what one
/// writes the next reads, edits, finds, moves and deletes, each change
/// naming the content it was made to, and nothing reaches this machine's
/// disk.
#[test]
fn every_file_tool_goes_through_its_workspace() {
    let memory = Arc::new(Memory::default());
    let workspace: Arc<dyn Workspace> = memory.clone();
    let base = Some(PathBuf::from("/project"));
    let (read, write, edit, search, delete, moved) = (
        Read {
            workspace: workspace.clone(),
            base: base.clone(),
            confine: true,
            seen: crate::Seen::default(),
        },
        Write {
            workspace: workspace.clone(),
            base: base.clone(),
            confine: true,
            seen: Default::default(),
        },
        Edit {
            workspace: workspace.clone(),
            base: base.clone(),
            confine: true,
            seen: crate::Seen::default(),
        },
        Search {
            workspace: workspace.clone(),
            base: base.clone(),
            confine: true,
        },
        Delete {
            workspace: workspace.clone(),
            base: base.clone(),
            confine: true,
        },
        Move {
            workspace,
            base,
            confine: true,
        },
    );

    let (written, context) = call(&write, WriteArgs::new("src/a.rs", "fn one() {}\n"));
    assert!(written.is_ok(), "{written:?}");
    assert_eq!(
        changes(&context),
        [("/project/src/a.rs".to_owned(), ChangeKind::Created)]
    );
    assert!(!Path::new("/project/src/a.rs").exists());

    let (edited, context) = call(
        &edit,
        EditArgs::new("src/a.rs", [Replacement::new("one", "two")]),
    );
    assert!(edited.is_ok(), "{edited:?}");
    assert_eq!(befores(&context), [Some(content_hash(b"fn one() {}\n"))]);
    let (shown, _) = call(&read, ReadArgs::new("src/../src/a.rs"));
    assert_eq!(
        shown.ok().as_ref().and_then(ToolOutput::as_text),
        Some("fn two() {}\n")
    );
    let (found, _) = call(&search, SearchArgs::new("two").glob("*.rs"));
    assert_eq!(found.ok().as_deref(), Some("src/a.rs:1: fn two() {}"));

    let (refused, _) = call(&moved, MoveArgs::new("src/a.rs", "../b.rs"));
    assert!(refused.is_err());
    let (renamed, context) = call(&moved, MoveArgs::new("src/a.rs", "src/b.rs"));
    assert!(renamed.is_ok(), "{renamed:?}");
    assert_eq!(
        changes(&context),
        [
            ("/project/src/a.rs".to_owned(), ChangeKind::Deleted),
            ("/project/src/b.rs".to_owned(), ChangeKind::Created)
        ]
    );
    assert_eq!(
        befores(&context),
        [Some(content_hash(b"fn two() {}\n")), None]
    );
    let (deleted, context) = call(&delete, DeleteArgs::new("src/b.rs"));
    assert!(deleted.is_ok(), "{deleted:?}");
    assert_eq!(befores(&context), [Some(content_hash(b"fn two() {}\n"))]);
    assert_eq!(
        changes(&context),
        [("/project/src/b.rs".to_owned(), ChangeKind::Deleted)]
    );
    assert!(memory.files().is_empty());
}
