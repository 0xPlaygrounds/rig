#![deny(missing_docs)]
//! Coding tools for Rig agents: [`Read`], [`Write`], [`Edit`], [`Search`], [`List`],
//! [`Shell`], [`Delete`] and [`Move`].
//!
//! - Each says in its description when to pick it, and what it may touch
//!   ([`footprint`](rig_core::tool::Tool::footprint)): `read`, `search` and `list` read the
//!   path they name, the other file tools write theirs, and `shell` may change anything, so a
//!   host runs calls on different files side by side.
//! - Each publishes what it did beside its text: files read
//!   ([`FilesRead`](rig_core::tool::FilesRead)), [`SearchHits`], [`FilesListed`], files changed
//!   with their diffs ([`FilesChanged`](rig_core::tool::FilesChanged)), how a command ended
//!   ([`ExitStatus`](rig_core::tool::ExitStatus)).
//! - Each acts on a [`Workspace`](workspace::Workspace): this machine
//!   ([`Local`](workspace::Local), the default) or another. The file tools resolve paths from an
//!   optional base directory they can be confined to. A path whose first component is `~` is in
//!   the home directory; `./~` is a folder named `~`.
//! - `shell` reports its output as it comes to a [`Live`]. Arguments are plain data with public
//!   fields.
//!
//! Also the pieces they are made of, for tools of your own:
//! - [`edit::apply`]: what `edit` makes of a file's text, without the file,
//!   and [`ReadArgs::show`]: what `read` shows of it;
//! - [`workspace`]: the disk and the commands the tools act on,
//!   [`overlay`]: one that keeps the tools' writes from the disk, to
//!   preview them, and [`remote`]: another machine reached through a
//!   transport such as `ssh`, and paths written `HOST:PATH` that reach it;
//! - [`attach`]: the files a message names as `@path`, read to go with it;
//! - [`checkpoint`]: the files under a root at one moment, what changed by
//!   a later one, and a handler that reports what any call changed;
//! - [`clipboard`]: the system clipboard's text and images;
//! - [`fs`]: an atomic file write and capped text reads on this machine;
//! - [`show_lines`]: some lines of a text, the way `read` shows a file, and
//!   [`excerpt`]: a line the way the tools cut one;
//! - [`Spill`]: where output a tool cut is kept whole, and [`OutputLimit`]: how
//!   much of a long output is kept;
//! - [`process`]: child processes in a process group of their own, the
//!   [`Launcher`](process::Launcher) that says how tools start them, and
//!   the [`Process`](process::Process) it spawns: input in, output line by
//!   line, stopped gently or killed with its group;
//! - [`blocking`]: blocking work on a thread of its own, so no async
//!   executor thread blocks.
//!
//! Native only: the tools use the file system and processes.

use std::ops::Range;

use unicode_width::{UnicodeWidthChar as _, UnicodeWidthStr as _};

pub mod attach;
mod blocking;
pub mod checkpoint;
pub mod clipboard;
mod delete;
pub mod edit;
pub mod fs;
mod list;
mod live;
mod move_file;
pub mod overlay;
pub mod process;
mod read;
pub mod remote;
mod search;
mod shell;
mod spill;
mod watch;
pub mod workspace;
mod write;

pub use blocking::blocking;
pub use delete::{Delete, DeleteArgs};
pub use edit::{Edit, EditArgs, Replacement};
pub use list::{FilesListed, List, ListArgs};
pub use live::Live;
pub use move_file::{Move, MoveArgs};
pub use read::{Read, ReadArgs, Seen, show_lines};
pub use search::{
    Found, Hit, MAX_FILES, MAX_MATCHES, Search, SearchArgs, SearchHits, Unread, find, render,
};
pub use shell::{Shell, ShellArgs, ShellRun, split_status};
pub use spill::Spill;
pub use write::{Write, WriteArgs};

/// Most lines a tool returns.
pub const MAX_LINES: usize = 2000;
/// Most bytes a tool returns.
pub const MAX_BYTES: usize = 50 * 1024;
/// The largest file `read` and `edit` load whole, and `search` scans.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
/// Characters of one line `read` and `shell` show; the rest of a longer
/// line is cut.
const MAX_LINE_CHARS: usize = 2000;
/// What marks a line that was cut.
const CUT_MARK: &str = " […]";

/// How much of a long output a tool keeps: at most `lines` lines and
/// `bytes` bytes, of whole lines. [`tail`](Self::tail) keeps the last ones;
/// [`Shell`] keeps a quarter for the first lines, a quarter for the errors
/// and failures from between, and the rest for the last lines.
///
/// ```
/// use rig_tools::OutputLimit;
///
/// let cut = OutputLimit { lines: 2, bytes: 1024 };
/// assert_eq!(cut.tail("one\ntwo\nthree\n"), ("two\nthree\n", 1));
/// let narrow = OutputLimit { lines: 10, bytes: 7 };
/// assert_eq!(narrow.tail("one\ntwo\nthree\n"), ("three\n", 2));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputLimit {
    /// Most lines kept.
    pub lines: usize,
    /// Most bytes kept.
    pub bytes: usize,
}

impl Default for OutputLimit {
    /// [`MAX_LINES`] lines or [`MAX_BYTES`] bytes.
    fn default() -> Self {
        Self {
            lines: MAX_LINES,
            bytes: MAX_BYTES,
        }
    }
}

impl OutputLimit {
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

/// The bytes of a cut line a command in a tool's note shows at once: a
/// fifth of what `shell` returns, in lines it keeps whole.
const CHUNK_BYTES: usize = 10_000;

/// What a tool shows of a line it picked for a part of it, the span (a
/// match, a change, a failure word): the line, or its first characters
/// when it is longer than a limit, and how that cut is marked. The start
/// is kept, so lines line up as the file has them; when the cut leaves the
/// span out, the mark says where it is.
///
/// ```
/// use rig_tools::excerpt;
///
/// let line = format!("{}OLD", "0".repeat(4000));
/// let shown = excerpt(&line, Some(4000..4003), 300);
/// assert_eq!(shown.kept(), "0".repeat(300));
/// assert_eq!(
///     shown.mark("the match", None),
///     " [… cut at byte 300 of 4003; the match is at bytes 4001-4003]"
/// );
/// assert_eq!(excerpt("short", Some(0..5), 300).mark("the match", None), "");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Excerpt<'l> {
    kept: &'l str,
    bytes: usize,
    span: Option<Range<usize>>,
    /// Bytes before the line's text in the file, such as a byte-order
    /// mark: the positions a mark names count them.
    before: usize,
}

/// [`Excerpt`] of `line` for `span`, its byte range (`None`: no part in
/// particular), at most `limit` characters of it.
pub fn excerpt(line: &str, span: Option<Range<usize>>, limit: usize) -> Excerpt<'_> {
    Excerpt::within(line, line.len(), span, limit)
}

/// Where a line is, for a command that shows part of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source<'a> {
    /// The file, quoted for `sh`.
    pub path: &'a str,
    /// The line's number, from 1.
    pub line: usize,
}

impl<'l> Excerpt<'l> {
    /// The excerpt of a line of `bytes` bytes whose start is `start` (the
    /// line itself, or the start of it an earlier excerpt kept).
    pub(crate) fn within(
        start: &'l str,
        bytes: usize,
        span: Option<Range<usize>>,
        limit: usize,
    ) -> Self {
        let kept = match start.char_indices().nth(limit) {
            Some((end, _)) => start.get(..end).unwrap_or(start),
            None => start,
        };
        Self {
            kept,
            bytes,
            span,
            before: 0,
        }
    }

    /// The same excerpt of a line that has `before` bytes before its text
    /// in the file, such as a byte-order mark before line 1.
    pub fn after(self, before: usize) -> Self {
        Self { before, ..self }
    }

    /// The start of the line shown: all of it unless it was cut.
    pub fn kept(&self) -> &'l str {
        self.kept
    }

    /// Whether the line was cut.
    pub fn is_cut(&self) -> bool {
        self.kept.len() < self.bytes
    }

    /// The span, when the line was cut and the cut left it out, all of it
    /// or its end.
    pub(crate) fn hidden(&self) -> Option<&Range<usize>> {
        let kept = self.kept.len();
        self.span.as_ref().filter(|span| {
            self.is_cut() && (span.end > kept || (span.is_empty() && span.start >= kept))
        })
    }

    /// The mark shown after [`kept`](Self::kept): none when the line is
    /// whole; ` […]` when what it was picked for shows, or it was picked
    /// for no part; else where the line was cut and where `what` (its
    /// span, named so) is, and, with the line's `source`, a `shell`
    /// command that shows the bytes of it from about 5000 before the span.
    pub fn mark(&self, what: &str, source: Option<Source<'_>>) -> String {
        let kept = self.kept.len();
        let span = match self.hidden() {
            _ if !self.is_cut() => return String::new(),
            Some(span) => span,
            None => return CUT_MARK.to_owned(),
        };
        let shift = self.before;
        let at = match span.is_empty() {
            true => format!("after byte {}", span.start + shift),
            false => format!("at bytes {}-{}", span.start + shift + 1, span.end + shift),
        };
        let mut mark = format!(
            " [… cut at byte {} of {}; {what} is {at}",
            kept + shift,
            self.bytes + shift
        );
        if let Some(source) = source {
            let from = (kept + 1).max((span.start + 1).saturating_sub(CHUNK_BYTES / 2)) + shift;
            let to = (from + CHUNK_BYTES - 1).min(self.bytes + shift);
            mark.push_str(&format!(
                ": in `shell`, {} shows bytes {from}-{to}",
                line_bytes(source.path, source.line, from, to)
            ));
        }
        mark.push(']');
        mark
    }
}

/// A `shell` command, quoted as code, that prints bytes `from` to `to`
/// (from 1) of line `line` of the file `path` (quoted for `sh`) in lines
/// `shell` keeps whole.
pub(crate) fn line_bytes(path: &str, line: usize, from: usize, to: usize) -> String {
    format!(
        "`sed -n '{line}p;{line}q' {path} | cut -b {from}-{to} | LC_ALL=C fold -b -w \
         {MAX_LINE_CHARS}`"
    )
}

/// `text` quoted for `sh`.
pub(crate) fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// `path`, a path a file tool resolved, quoted for a command in `shell`:
/// `None` when it is not absolute for `sh`, such as a remote `~/` path, a
/// `HOST:PATH` or a Windows path, which `shell` would not find as written.
pub(crate) fn shell_path(path: &str) -> Option<String> {
    path.starts_with('/').then(|| quote(path))
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
