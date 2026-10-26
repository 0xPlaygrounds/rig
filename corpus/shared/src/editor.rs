//! The user's editor: the command line in `VISUAL`, then `EDITOR`, and
//! `vi` when neither is set. The line is shell syntax, as git reads it, so
//! it can quote a path with spaces or be a whole shell command; the files
//! follow it as arguments.
//!
//! ```
//! let line = corpus_shared::editor::line();
//! let notes = std::path::Path::new("notes.md");
//! let command = corpus_shared::editor::command(line.as_deref(), &[notes]);
//! ```

use std::path::Path;
use std::process::Command;

/// The editor command line from `VISUAL`, then `EDITOR`: the first that is
/// set and not blank.
pub fn line() -> Option<String> {
    ["VISUAL", "EDITOR"].into_iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

/// The command that opens `files` in the editor command line `line`, or in
/// `vi` when `line` is `None` or blank. On unix it runs
/// `sh -c '<line> "$@"' '<line>' <files>`.
#[cfg(not(windows))]
pub fn command(line: Option<&str>, files: &[&Path]) -> Command {
    let line = chosen(line);
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(format!("{line} \"$@\""))
        .arg(line)
        .args(files);
    command
}

/// The command that opens `files` in the editor command line `line`, or in
/// `vi` when `line` is `None` or blank. On Windows it runs
/// `cmd.exe /d /v:off /s /c "<line> "<file>" ..."`, passed as it is. A
/// Windows path cannot hold a `"`, but `cmd.exe` expands a `%NAME%` in it.
#[cfg(windows)]
pub fn command(line: Option<&str>, files: &[&Path]) -> Command {
    use std::os::windows::process::CommandExt;
    let files = files.iter().map(|file| format!(" \"{}\"", file.display()));
    let run = format!("\"{}{}\"", chosen(line), files.collect::<String>());
    let mut command = Command::new("cmd.exe");
    command.args(["/d", "/v:off", "/s", "/c"]).raw_arg(run);
    command
}

/// `line`, or `vi` when it is `None` or blank.
fn chosen(line: Option<&str>) -> &str {
    line.filter(|line| !line.trim().is_empty()).unwrap_or("vi")
}

#[cfg(test)]
mod tests;
