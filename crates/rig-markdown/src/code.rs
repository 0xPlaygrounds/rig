//! Lines of code as drawn: split at each `\n`, each line without its end
//! and with tabs as four spaces. The terminal view's code and `rig -p`'s
//! shell command lines are both laid out by these rules.
//!
//! ```
//! use rig_markdown::code::{lines, shown};
//!
//! let rows: Vec<String> = lines("a\r\n\tb").map(|line| shown(line, 0..line.len())).collect();
//! assert_eq!(rows, ["a", "    b"]);
//! ```

use std::ops::Range;

/// The lines of `code`, each with its end: one per `\n`, then the text
/// after the last, if any. Empty code has none.
pub fn lines(code: &str) -> impl Iterator<Item = &str> {
    code.split_inclusive('\n')
}

/// The bytes `range` of `line` as drawn: without the line's end (`\n` or
/// `\r\n`), tabs as four spaces. A range past the end is cut to it.
pub fn shown(line: &str, range: Range<usize>) -> String {
    let end = line.trim_end_matches(['\n', '\r']).len();
    let text = line.get(range.start.min(end)..range.end.min(end));
    text.unwrap_or_default().replace('\t', "    ")
}
