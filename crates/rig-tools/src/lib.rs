//! Coding tools for Rig agents: [`Read`], [`Write`], [`Edit`], [`Search`],
//! [`Shell`], [`Delete`] and [`Move`]. Each has a description of what it does, its usage
//! `RULES`, the lines a system prompt carries on when to pick it, such as
//! "use `read`, not `cat`", and annotations saying what it does to the
//! world (read-only or destructive). They are [`Tool`](rig_core::tool::Tool)s
//! that publish what they did beside their text: the files they read
//! ([`FilesRead`](rig_core::tool::FilesRead)), what a search found
//! ([`SearchHits`]), the files they changed, with their diffs
//! ([`FilesChanged`](rig_core::tool::FilesChanged)), and how the command
//! ended ([`ExitStatus`](rig_core::tool::ExitStatus)). Each acts on a
//! [`Workspace`](workspace::Workspace): this machine
//! ([`Local`](workspace::Local), the default) or another, such as one
//! reached over SSH. The file tools name files by their resolved paths
//! (`resolve` on [`Workspace`](workspace::Workspace)), from an optional
//! base directory they can be confined to. Each call says what it may read
//! and change ([`footprint`](rig_core::tool::Tool::footprint)): `read` and
//! `search` read the path they name, `write`, `edit`, `delete` and `move`
//! write theirs, and `shell` may change anything, so a host runs calls on
//! different files side by side. `shell` reports its output as it comes to
//! a [`Live`]. Their arguments are plain data with public fields, to build
//! or read a call's.
//!
//! Also the pieces they are made of, for tools of your own:
//! - [`edit::apply`]: what `edit` makes of a file's text, without the file,
//!   and [`ReadArgs::show`]: what `read` shows of it;
//! - [`workspace`]: the disk and the commands the tools act on;
//! - [`attach`]: the files a message names as `@path`, read to go with it;
//! - [`context`]: the project's instruction files (`AGENTS.md`, or
//!   `CLAUDE.md`) from the working directory up;
//! - [`fs`]: an atomic file write and capped text reads on this machine;
//! - [`numbered`]: text as numbered lines, the way `read` shows a file;
//! - [`Spill`]: where output a tool cut is kept whole, and [`Cut`]: how
//!   much of a long output is kept;
//! - [`process`]: child processes in a process group of their own, the
//!   [`Launcher`](process::Launcher) that says how tools start them, and
//!   the [`Process`](process::Process) it spawns: input in, output line by
//!   line, stopped gently or killed with its group;
//! - [`blocking`]: blocking work on a thread of its own, so no async
//!   executor thread blocks.
//!
//! Native only: the tools use the file system and processes.

use unicode_width::{UnicodeWidthChar as _, UnicodeWidthStr as _};

pub mod attach;
mod blocking;
pub mod context;
mod delete;
pub mod edit;
pub mod fs;
mod live;
mod move_file;
pub mod process;
mod read;
mod search;
mod shell;
mod spill;
pub mod workspace;
mod write;

pub use blocking::blocking;
pub use delete::{Delete, DeleteArgs};
pub use edit::{Edit, EditArgs, Replacement};
pub use live::Live;
pub use move_file::{Move, MoveArgs};
pub use read::{Read, ReadArgs, numbered};
pub use search::{Found, Hit, MAX_MATCHES, Search, SearchArgs, SearchHits, find, render};
pub use shell::{Shell, ShellArgs, ShellRun, split_status};
pub use spill::Spill;
pub use write::{Write, WriteArgs};

/// Most lines a tool returns.
pub const MAX_LINES: usize = 2000;
/// Most bytes a tool returns.
pub const MAX_BYTES: usize = 50 * 1024;
/// The largest file `read` and `edit` load whole, and `search` scans.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// How much of a long output a tool keeps: its last lines, at most
/// `lines` of them and `bytes` bytes, starting at a line.
///
/// ```
/// use rig_tools::Cut;
///
/// let cut = Cut { lines: 2, bytes: 1024 };
/// assert_eq!(cut.tail("one\ntwo\nthree\n"), ("two\nthree\n", 1));
/// let narrow = Cut { lines: 10, bytes: 7 };
/// assert_eq!(narrow.tail("one\ntwo\nthree\n"), ("three\n", 2));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    /// Most lines kept.
    pub lines: usize,
    /// Most bytes kept.
    pub bytes: usize,
}

impl Default for Cut {
    /// [`MAX_LINES`] lines or [`MAX_BYTES`] bytes.
    fn default() -> Self {
        Self {
            lines: MAX_LINES,
            bytes: MAX_BYTES,
        }
    }
}

impl Cut {
    /// The end of `text` this keeps, starting at a line, and how many lines
    /// before it were left out. When even the last line is longer than
    /// [`bytes`](Self::bytes), nothing is kept.
    pub fn tail<'t>(&self, text: &'t str) -> (&'t str, usize) {
        let starts: Vec<usize> = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(at, _)| at + 1))
            .filter(|start| *start < text.len())
            .collect();
        let first = starts.len().saturating_sub(self.lines);
        let kept = starts
            .iter()
            .enumerate()
            .skip(first)
            .find(|(_, start)| text.len() - **start <= self.bytes);
        match kept {
            Some((dropped, start)) => (text.get(*start..).unwrap_or_default(), dropped),
            None => ("", starts.len()),
        }
    }
}

/// `line` cut to its first `limit` characters.
pub fn clip(line: &str, limit: usize) -> &str {
    match line.char_indices().nth(limit) {
        Some((end, _)) => line.get(..end).unwrap_or(line),
        None => line,
    }
}

/// `text` cut to fit in `width` terminal columns, ending in `…` when cut:
/// a wide character, such as most CJK ones, takes two columns.
///
/// ```
/// use rig_tools::shorten;
///
/// assert_eq!(shorten("a short label", 20), "a short label");
/// assert_eq!(shorten("a longer label", 9), "a longer…");
/// assert_eq!(shorten("日本語のラベル", 7), "日本語…");
/// ```
pub fn shorten(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_owned();
    }
    let room = width.saturating_sub(1);
    let mut used = 0;
    let end = text
        .char_indices()
        .find(|(_, character)| {
            used += character.width().unwrap_or(0);
            used > room
        })
        .map_or(text.len(), |(at, _)| at);
    match width {
        0 => String::new(),
        _ => format!("{}…", text.get(..end).unwrap_or_default()),
    }
}

/// How long ago `elapsed` was, roughly, for a list a person reads:
/// `just now`, `3m ago`, `5h ago`, `2d ago`.
///
/// ```
/// use std::time::Duration;
/// use rig_tools::ago;
///
/// assert_eq!(ago(Duration::from_secs(30)), "just now");
/// assert_eq!(ago(Duration::from_secs(5 * 3600 + 59)), "5h ago");
/// ```
pub fn ago(elapsed: std::time::Duration) -> String {
    let seconds = elapsed.as_secs();
    match seconds {
        0..60 => "just now".to_owned(),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..86_400 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

#[cfg(test)]
mod tests;
