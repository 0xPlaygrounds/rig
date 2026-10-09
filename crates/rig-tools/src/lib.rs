//! Coding tools for Rig agents, as [`PortableTool`](rig_core::tool::PortableTool)s:
//! [`Read`], [`Write`], [`Edit`], [`Search`] and [`Shell`]. Each has a
//! description of what it does and its usage `RULES`, the lines a system
//! prompt carries on when to pick it, such as "use `read`, not `cat`".
//!
//! Also the pieces they are made of, for tools of your own:
//! - [`context`]: the project's instruction files (`AGENTS.md`, or
//!   `CLAUDE.md`) from the working directory up;
//! - [`fs`]: an atomic file write and capped text reads;
//! - [`numbered`]: text as numbered lines, the way `read` shows a file;
//! - [`process`]: child processes in a process group of their own;
//! - [`blocking`]: blocking work on a thread of its own, so no async
//!   executor thread blocks.
//!
//! Native only: the tools use the file system and processes.

mod blocking;
pub mod context;
mod edit;
pub mod fs;
pub mod process;
mod read;
mod search;
mod shell;
mod write;

pub use blocking::blocking;
pub use edit::{Edit, EditArgs};
pub use read::{Read, ReadArgs, numbered};
pub use search::{Search, SearchArgs};
pub use shell::{Shell, ShellArgs};
pub use write::{Write, WriteArgs};

/// Most lines a tool returns.
pub const MAX_LINES: usize = 2000;
/// Most bytes a tool returns.
pub const MAX_BYTES: usize = 50 * 1024;
/// The largest file `read` and `edit` load whole, and `search` scans.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// `line` cut to its first `limit` characters.
pub fn clip(line: &str, limit: usize) -> &str {
    match line.char_indices().nth(limit) {
        Some((end, _)) => line.get(..end).unwrap_or(line),
        None => line,
    }
}

/// `text` cut to its first `limit` characters, ending in `…` when cut.
pub fn shorten(text: &str, limit: usize) -> String {
    let clipped = clip(text, limit);
    if clipped.len() < text.len() {
        format!("{clipped}…")
    } else {
        text.to_owned()
    }
}
