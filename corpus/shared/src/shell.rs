//! The corpus's read-only shell rule: the commands a guard lets through,
//! or a dry run still runs, because they change nothing.
//!
//! ```
//! assert!(corpus_shared::shell::reads_only("git diff | head"));
//! assert!(!corpus_shared::shell::reads_only("ls > listing"));
//! ```

/// Commands that only read, by their first words.
const READ_ONLY: &[&[&str]] = &[
    &["ls"],
    &["cat"],
    &["head"],
    &["tail"],
    &["wc"],
    &["pwd"],
    &["rg"],
    &["grep"],
    &["git", "status"],
    &["git", "diff"],
    &["git", "log"],
    &["git", "show"],
];

/// Options of those commands that write or run something: `git diff
/// --output` writes a file, `rg --pre` runs a command.
const UNSAFE_OPTIONS: &[&str] = &["--output", "--pre"];

/// Whether `command` only reads: no redirection, sequencing or
/// substitution, and every part of its pipeline an allowed command with no
/// option that writes or runs something.
pub fn reads_only(command: &str) -> bool {
    let forbidden = ['>', '<', ';', '&', '`', '$', '\n', '('];
    if command.trim().is_empty() || command.contains(forbidden) {
        return false;
    }
    command.split('|').all(|part| {
        let words: Vec<&str> = part.split_whitespace().collect();
        READ_ONLY.iter().any(|allowed| words.starts_with(allowed))
            && !words.iter().any(|word| unsafe_option(word))
    })
}

/// Whether `word` may reach its command as one of [`UNSAFE_OPTIONS`]:
/// the option followed by nothing, a value, or shell syntax such as a glob
/// or brace that can expand into a value. Only a longer option name, such
/// as `--pretty`, is safe. Quotes and backslashes are dropped first, as
/// the shell drops them before the command sees the option.
fn unsafe_option(word: &str) -> bool {
    let word = word.replace(['\'', '"', '\\'], "");
    UNSAFE_OPTIONS.iter().any(|option| {
        word.strip_prefix(option).is_some_and(|rest| {
            !rest
                .chars()
                .next()
                .is_some_and(|next| next.is_ascii_alphanumeric() || next == '-')
        })
    })
}

#[cfg(test)]
mod tests;
