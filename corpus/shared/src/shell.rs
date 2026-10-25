//! The corpus's read-only shell rule: the commands a guard lets through,
//! or a dry run still runs, because they change nothing.
//!
//! ```
//! assert!(corpus_shared::shell::reads_only("git diff | head"));
//! assert!(!corpus_shared::shell::reads_only("ls > listing"));
//! ```

/// Command prefixes allowed by the corpus read-only shell policy.
const READ_ONLY: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "pwd",
    "rg",
    "grep",
    "git status",
    "git diff",
    "git log",
    "git show",
];

/// Whether `command` only reads: no redirection, sequencing or
/// substitution, and every part of its pipeline an allowed command.
pub fn reads_only(command: &str) -> bool {
    let forbidden = ['>', '<', ';', '&', '`', '$', '\n', '('];
    if command.trim().is_empty() || command.contains(forbidden) {
        return false;
    }
    command.split('|').all(|part| {
        let part = part.trim();
        let allowed = READ_ONLY.iter().any(|allowed| {
            part.strip_prefix(allowed)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
        });
        // `git diff --output=…` writes a file; `rg --pre` runs a command.
        allowed && !part.contains("--output") && !part.contains("--pre")
    })
}

#[cfg(test)]
mod tests;
