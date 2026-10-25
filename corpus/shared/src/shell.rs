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

/// Shell syntax that expands one word into others: globs and braces.
const EXPANDING: [char; 4] = ['*', '?', '[', '{'];

/// Whether `command` only reads: no redirection, sequencing or
/// substitution, and every part of its pipeline an allowed command with no
/// word that may reach it as an option that writes or runs something.
pub fn reads_only(command: &str) -> bool {
    let forbidden = ['>', '<', ';', '&', '`', '$', '\n', '('];
    if command.trim().is_empty() || command.contains(forbidden) {
        return false;
    }
    command.split('|').all(|part| {
        let Some(words) = words(part) else {
            return false;
        };
        let texts: Vec<&str> = words.iter().map(|word| word.text.as_str()).collect();
        READ_ONLY.iter().any(|allowed| texts.starts_with(allowed))
            && !words.iter().any(Word::may_be_unsafe_option)
    })
}

/// A shell word as its command sees it, unless it expands.
#[derive(Default)]
struct Word {
    /// Its text without quotes and backslashes.
    text: String,
    /// Whether it holds a glob or brace outside quotes.
    expands: bool,
}

impl Word {
    /// Whether it may reach its command as one of [`UNSAFE_OPTIONS`]: it
    /// spells the option, alone or followed by anything but more of an
    /// option name (`--pretty` is safe), or it expands and may expand to an
    /// option. An expanded word keeps the text before its first glob or
    /// brace, so only one starting with `-` or with that syntax, and
    /// holding a `-`, can.
    fn may_be_unsafe_option(&self) -> bool {
        let text = self.text.as_str();
        let may_expand_to_option = self.expands
            && text.contains('-')
            && (text.starts_with('-') || text.starts_with(EXPANDING));
        may_expand_to_option
            || UNSAFE_OPTIONS.iter().any(|option| {
                text.strip_prefix(option).is_some_and(|rest| {
                    !rest
                        .chars()
                        .next()
                        .is_some_and(|next| next.is_ascii_alphanumeric() || next == '-')
                })
            })
    }
}

/// The shell words of `part`, split at spaces and tabs outside quotes;
/// `None` when a quote is left open. Inside double quotes a backslash
/// escapes only `"` and `\\`, as in the shell; the other characters it
/// escapes there are forbidden.
fn words(part: &str) -> Option<Vec<Word>> {
    let mut words = Vec::new();
    let mut current: Option<Word> = None;
    let mut quote = None;
    let mut chars = part.chars();
    while let Some(c) = chars.next() {
        if quote.is_none() && matches!(c, ' ' | '\t') {
            words.extend(current.take());
            continue;
        }
        let word = current.get_or_insert_with(Word::default);
        match (quote, c) {
            (None, '\'' | '"') => quote = Some(c),
            (Some(open), _) if c == open => quote = None,
            (None, '\\') => word.text.extend(chars.next()),
            (Some('"'), '\\') => match chars.clone().next() {
                Some(next @ ('"' | '\\')) => {
                    chars.next();
                    word.text.push(next);
                }
                _ => word.text.push(c),
            },
            (None, _) => {
                word.expands |= EXPANDING.contains(&c);
                word.text.push(c);
            }
            (Some(_), _) => word.text.push(c),
        }
    }
    words.extend(current);
    quote.is_none().then_some(words)
}

#[cfg(test)]
mod tests;
