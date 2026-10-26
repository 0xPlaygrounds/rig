//! The user's editor: the command line in `VISUAL`, then `EDITOR`, and
//! `vi` when neither is set.
//!
//! ```
//! let line = corpus_shared::editor::line();
//! let (program, args) = corpus_shared::editor::words(line.as_deref());
//! let mut command = std::process::Command::new(program);
//! command.args(args).arg("notes.md");
//! ```

use std::str::SplitWhitespace;

/// The editor command line from `VISUAL`, then `EDITOR`: the first that is
/// set and not blank.
pub fn line() -> Option<String> {
    ["VISUAL", "EDITOR"].into_iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

/// The program and arguments of the editor command line `line`; `vi` with
/// no arguments when `line` is `None` or blank.
pub fn words(line: Option<&str>) -> (&str, SplitWhitespace<'_>) {
    let mut words = line.unwrap_or_default().split_whitespace();
    (words.next().unwrap_or("vi"), words)
}

#[cfg(test)]
mod tests;
