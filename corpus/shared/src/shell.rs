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

/// Shell syntax that expands one word into others: globs and braces.
const EXPANDING: [char; 4] = ['*', '?', '[', '{'];

/// Whether `word` may reach its command as one of [`UNSAFE_OPTIONS`].
/// Quotes and backslashes are dropped first, as the shell drops them.
/// The word is unsafe when it spells the option, alone or followed by
/// anything but more of an option name (`--pretty` is safe), or when it
/// expands and may expand to an option: an expanded word keeps the text
/// before its first glob or brace, so only a word starting with `-` or
/// with that syntax, and holding a `-`, can.
fn unsafe_option(word: &str) -> bool {
    let word = word.replace(['\'', '"', '\\'], "");
    let expands = word.contains(EXPANDING);
    let may_expand_to_option =
        expands && word.contains('-') && (word.starts_with('-') || word.starts_with(EXPANDING));
    may_expand_to_option
        || UNSAFE_OPTIONS.iter().any(|option| {
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
