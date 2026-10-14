//! Text helpers the formats share: a splice and the file's own
//! indentation and line ending.

use std::ops::Range;

/// Replaces the bytes of `range` with `text`; every other byte stays.
#[derive(Debug)]
pub struct Splice {
    range: Range<usize>,
    text: String,
}

impl Splice {
    /// Replace `range` with `text`.
    pub fn new(range: Range<usize>, text: String) -> Self {
        Self { range, text }
    }

    /// `original` with this splice made.
    ///
    /// # Errors
    ///
    /// When the range is not on character boundaries of `original`.
    pub fn apply(&self, original: &str) -> Result<String, String> {
        let before = original.get(..self.range.start);
        let after = original.get(self.range.end..);
        match (before, after) {
            (Some(before), Some(after)) => Ok(format!("{before}{}{after}", self.text)),
            _ => Err("the change fell outside the file".to_owned()),
        }
    }
}

/// The whitespace that starts the line holding byte `at`.
pub fn indent_of(text: &str, at: usize) -> &str {
    let start = text
        .get(..at)
        .and_then(|before| before.rfind('\n'))
        .map_or(0, |newline| newline + 1);
    let line = text.get(start..).unwrap_or_default();
    let width = line.len() - line.trim_start_matches([' ', '\t']).len();
    line.get(..width).unwrap_or_default()
}

/// One step of the file's indentation: the shortest indentation of a line
/// that has one, else `default`.
pub fn indent_unit(text: &str, default: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let width = line.len() - line.trim_start_matches([' ', '\t']).len();
            line.get(..width).unwrap_or_default()
        })
        .filter(|indent| !indent.is_empty())
        .min_by_key(|indent| indent.len())
        .unwrap_or(default)
        .to_owned()
}

/// The file's line ending: CRLF when its lines end so, else LF.
pub fn newline(text: &str) -> &'static str {
    if text.contains("\r\n") { "\r\n" } else { "\n" }
}
