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

/// Only a first component `~` is the home directory: `~bob`, a file whose
/// name starts with `~` and a folder named `~` reached as `./~` or below
/// another stay where they are.
#[test]
fn only_a_leading_tilde_is_the_home_directory() {
    let base = Path::new("/project");
    for path in ["~bob/x", "~$report.docx", "./~/x", "a/~/x"] {
        assert_eq!(
            super::Local.join(base, Path::new(path)),
            base.join(path),
            "{path}"
        );
    }
}

/// The home directory the copy of [`tilde_is_the_home_directory`] runs
/// with.
#[cfg(unix)]
const TEST_HOME: &str = "RIG_TOOLS_TEST_HOME";

/// A path starting with `~` is in the home directory, for every file tool
/// and the shell's directory; a confined tool refuses it as outside the
/// project. The project's own `~` is reached as `./~`, and listed and found
/// so that reading the shown path reads it. Runs again as a copy of itself
/// with a home of its own, so the real one is never touched. Unix only:
/// the shell part needs `sh`, and Windows spells a canonical path with a
/// `\\?\` prefix that the joins here don't follow.
#[cfg(unix)]
#[test]
fn tilde_is_the_home_directory() {
    let Some(home) = std::env::var_os(TEST_HOME).map(PathBuf::from) else {
        let dir = std::env::temp_dir().join(format!("rig-tools-tilde-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("home")).ok();
        std::fs::create_dir_all(dir.join("project")).ok();
        // Spelled as the tools will name it, temporary directories being
        // links on some systems.
        let home = std::fs::canonicalize(dir.join("home")).unwrap_or(dir.join("home"));
        let name = "workspace::tests::tilde_is_the_home_directory";
        let output = std::env::current_exe().and_then(|test| {
            std::process::Command::new(test)
                .args(["--exact", name, "--test-threads", "1"])
                .current_dir(home.with_file_name("project"))
                .env(TEST_HOME, &home)
                .env("HOME", &home)
                .output()
        });
        std::fs::remove_dir_all(&dir).ok();
        let (ran, said) = output.map_or_else(
            |error| (false, error.to_string()),
            |output| {
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let ran = output.status.success() && stdout.contains("1 passed");
                (ran, stdout + &String::from_utf8_lossy(&output.stderr))
            },
        );
        assert!(ran, "the run with a home of its own failed:\n{said}");
        return;
    };
    let project = home.with_file_name("project");
    let base = Some(project.clone());
    let write = Write {
        base: base.clone(),
        ..Write::default()
    };
    let read = Read {
        base: base.clone(),
        ..Read::default()
    };
    let text = |shown: ToolOutput| shown.as_text().unwrap_or_default().to_owned();

    let (wrote, context) = call(&write, WriteArgs::new("~/notes/a.txt", "home\n"));
    let there = std::fs::read_to_string(home.join("notes/a.txt")).ok();
    let (read_back, _) = call(&read, ReadArgs::new("~/notes/a.txt"));
    assert_eq!(wrote.ok().as_deref(), Some("Created ~/notes/a.txt."));
    let file = home.join("notes/a.txt").to_string_lossy().into_owned();
    assert_eq!(changes(&context), [(file, ChangeKind::Created)]);
    assert_eq!(there.as_deref(), Some("home\n"));
    assert!(read_back.map(text).unwrap_or_default().contains("home"));
    assert!(!project.join("~/notes").exists());

    // Without a base, as from this process's working directory.
    let (wrote, _) = call(&Write::default(), WriteArgs::new("~/notes/b.txt", "home\n"));
    let mut everywhere = SearchArgs::new("home");
    everywhere.path = Some("~/notes".into());
    let (found, _) = call(&Search::default(), everywhere);
    // From a base the home is under, a `~` path is still shown whole.
    let above = Search {
        base: home.parent().map(Path::to_path_buf),
        ..Search::default()
    };
    let mut from_above = SearchArgs::new("home");
    from_above.path = Some("~/notes".into());
    let (above, _) = call(&above, from_above);
    let b = home.join("notes/b.txt").display().to_string();
    assert!(wrote.is_ok() && home.join("notes/b.txt").is_file());
    for found in [found, above] {
        let found = found.unwrap_or_default();
        assert!(found.lines().any(|line| line.starts_with(&b)), "{found}");
    }

    let confined = Write {
        confine: true,
        ..write.clone()
    };
    let (refused, _) = call(&confined, WriteArgs::new("~/b.txt", "no\n"));
    assert!(refused.is_err() && !home.join("b.txt").exists());

    let (own, _) = call(&write, WriteArgs::new("./~/own.txt", "own\n"));
    assert!(own.is_ok() && project.join("~/own.txt").is_file());
    let list = crate::List {
        workspace: Arc::new(super::Local),
        base: base.clone(),
        confine: true,
    };
    let (listed, _) = call(&list, crate::ListArgs::new("."));
    let search = Search {
        base: base.clone(),
        ..Search::default()
    };
    let (found, _) = call(&search, SearchArgs::new("own"));
    // Without a base, from the project as this process's directory.
    let (walked, _) = call(&Search::default(), SearchArgs::new("own"));
    let listed = listed.unwrap_or_default();
    let own = Path::new("./~/own.txt");
    let names_own = |found: &str| {
        let first = |line: &str| line.split(':').next().map(Path::new) == Some(own);
        found.lines().any(first)
    };
    assert!(
        listed.lines().any(|line| Path::new(line) == own),
        "{listed}"
    );
    for found in [found, walked] {
        let found = found.unwrap_or_default();
        assert!(names_own(&found), "{found}");
    }
    let (again, _) = call(&read, ReadArgs::new("./~/own.txt"));
    assert!(again.map(text).unwrap_or_default().contains("own"));

    let shell = crate::Shell {
        launcher: Launcher::default().in_dir(&project),
        ..crate::Shell::default()
    };
    let (ran, _) = call(&shell, crate::ShellArgs::new("pwd -P").in_dir("~"));
    let ran = ran.unwrap_or_default();
    assert_eq!(ran.lines().next(), home.to_str(), "{ran}");
}
