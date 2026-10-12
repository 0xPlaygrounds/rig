//! The project's instruction files: one `AGENTS.md`, or `CLAUDE.md`, per
//! directory, read from the most general directory to the most specific,
//! as text for a system prompt.

use std::path::{Path, PathBuf};

use crate::fs::Prefix;
use crate::workspace::{Local, Stat, Workspace};

/// The names of an instruction file, in order of preference: one per
/// directory is read.
pub const FILE_NAMES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];
/// The most of one instruction file read.
pub const MAX_FILE_BYTES: usize = 32 * 1024;
/// The most of all instruction files together. The most specific files are
/// kept first.
pub const MAX_TOTAL_BYTES: usize = 64 * 1024;

/// An instruction file as read.
pub struct InstructionFile {
    pub path: PathBuf,
    pub contents: Prefix,
}

/// The instruction files found, from the most general to the most specific.
pub struct Instructions {
    pub files: Vec<InstructionFile>,
    /// Files that exist but could not be read, with why.
    pub unreadable: Vec<String>,
}

impl Instructions {
    /// Reads the instruction files of the `general` directories on this
    /// machine, such as the user's own, then of each directory of
    /// `workspace` from the root down to `cwd`. A file reached twice is read
    /// once; empty files are left out.
    ///
    /// Reads `workspace`: on a thread that must not wait, call it only for
    /// a local one ([`Workspace::host`] is `None`).
    pub fn discover(
        general: impl IntoIterator<Item = PathBuf>,
        cwd: &Path,
        workspace: &dyn Workspace,
    ) -> Self {
        let local: &dyn Workspace = &Local;
        let mut dirs: Vec<(&dyn Workspace, PathBuf)> =
            general.into_iter().map(|dir| (local, dir)).collect();
        let above: Vec<&Path> = cwd.ancestors().collect();
        dirs.extend(
            above
                .into_iter()
                .rev()
                .map(|dir| (workspace, dir.to_path_buf())),
        );

        let mut found: Vec<(&dyn Workspace, PathBuf)> = Vec::new();
        let mut seen: Vec<(Option<&str>, PathBuf)> = Vec::new();
        for (place, dir) in dirs {
            let Some(path) = instruction_file_in(place, &dir) else {
                continue;
            };
            let canonical = (place.host(), place.canonical(&path));
            if !seen.contains(&canonical) {
                seen.push(canonical);
                found.push((place, path));
            }
        }

        // The budget goes to the most specific files first.
        let mut budget = MAX_TOTAL_BYTES;
        let mut files = Vec::new();
        let mut unreadable = Vec::new();
        for (place, path) in found.into_iter().rev() {
            let cap = budget.min(MAX_FILE_BYTES);
            match place.read(&path, cap as u64) {
                Ok(contents) => {
                    let contents = Prefix::of(contents, cap);
                    budget = budget.saturating_sub(contents.text.len());
                    if !contents.text.trim().is_empty() {
                        files.push(InstructionFile { path, contents });
                    }
                }
                Err(error) => unreadable.push(format!("{}: {error}", path.display())),
            }
        }
        files.reverse();
        Self { files, unreadable }
    }

    /// The files as a system prompt section: each in a `<file path=…>`
    /// tag, a cut one saying so, after a line on how to follow them. Empty
    /// when there are none.
    pub fn prompt_text(&self) -> String {
        let mut text = String::new();
        if !self.files.is_empty() {
            text.push_str(
                "Instructions from the user and the project, from the most general to the most \
                 specific. Follow them; where two disagree, the later one wins.",
            );
        }
        for file in &self.files {
            let contents = &file.contents;
            text.push_str(&format!(
                "\n\n<file path=\"{}\">\n{}",
                file.path.display(),
                contents.text.trim_end()
            ));
            if contents.cut() {
                text.push_str(&format!(
                    "\n[Cut after {} of its {} bytes; read the file for the rest.]",
                    contents.text.len(),
                    contents.size
                ));
            }
            text.push_str("\n</file>");
        }
        text
    }

    /// The files' paths, comma-separated, or `none`.
    pub fn paths(&self) -> String {
        let names: Vec<String> = self
            .files
            .iter()
            .map(|file| file.path.display().to_string())
            .collect();
        if names.is_empty() {
            "none".to_owned()
        } else {
            names.join(", ")
        }
    }
}

/// The instruction file of `dir` on this machine, if it has one.
pub fn instruction_file(dir: &Path) -> Option<PathBuf> {
    instruction_file_in(&Local, dir)
}

/// The instruction file of `dir` in `workspace`, if it has one: the first
/// of [`FILE_NAMES`] that is a file there.
pub fn instruction_file_in(workspace: &dyn Workspace, dir: &Path) -> Option<PathBuf> {
    FILE_NAMES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| matches!(workspace.stat(path), Ok(Stat::File { .. })))
}
