//! The `edit` tool: several replacements in one file per call, all matched
//! against the file as it was before the call and applied together or not
//! at all. A byte-order mark and CRLF line endings are kept. Text that does
//! not match exactly may still match line by line once trailing
//! whitespace and typographic quotes, dashes and spaces are evened out, but
//! only where exactly one place in the file matches. The result is a unified diff of the change; a failure
//! says what to send instead.

use std::fmt::Write as _;
use std::ops::Range;
use std::path::Path;

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;
use similar::TextDiff;

use crate::fs::{io_error, read_text, write_atomic};
use crate::{MAX_BYTES, MAX_LINES, blocking};

/// Lines of unchanged context around each hunk of the returned diff.
const CONTEXT: usize = 3;
/// Most matches an ambiguity error lists by line.
const MAX_LISTED: usize = 8;

/// Replaces text in a file.
pub struct Edit;

/// Arguments of [`Edit`].
#[derive(Deserialize)]
pub struct EditArgs {
    path: String,
    edits: Vec<Replacement>,
}

/// One replacement of an [`Edit`] call.
#[derive(Deserialize)]
struct Replacement {
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

impl Edit {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] = &[
        "Use `edit` to change part of a file. Copy each `old_text` from what `read` \
         returned, without the line numbers, with just enough lines around the \
         change to match once. Make all changes to one file in one `edit` call, \
         one entry of `edits` per change; each matches the file as it was before \
         the call.",
    ];
}

impl PortableTool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace text in a file. Each entry of `edits` replaces `old_text`, which must \
         match the file exactly (whitespace included) and only once unless `replace_all` is \
         true, with `new_text`. All entries match the file as it was before the call, must not \
         overlap, and are applied together or not at all. Returns a unified diff of the change."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to edit."},
                "edits": {
                    "type": "array",
                    "description": "The replacements, matched against the original file.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {"type": "string", "description": "The exact text to replace, copied from the file without line numbers."},
                            "new_text": {"type": "string", "description": "The replacement."},
                            "replace_all": {"type": "boolean", "description": "Replace every match instead of exactly one."}
                        },
                        "required": ["old_text", "new_text"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    async fn call(&self, args: EditArgs) -> Result<String, ToolExecutionError> {
        blocking(move || edit(args)).await
    }
}

fn edit(args: EditArgs) -> Result<String, ToolExecutionError> {
    let path = args.path.as_str();
    if args.edits.is_empty() {
        return Err(ToolExecutionError::invalid_args(format!(
            "`edits` is empty, so {path} was not changed; give at least one \
             {{\"old_text\", \"new_text\"}}"
        )));
    }
    let raw = read_text(path)?;
    let file = Decoded::new(&raw);
    let text = file.text.as_str();
    let count = args.edits.len();
    let many = count > 1;
    let mut splices = Vec::new();
    let mut notes = Vec::new();
    // Every edit is checked, so one reply names all that need fixing.
    let mut failed: Vec<(usize, String)> = Vec::new();
    for (index, replacement) in args.edits.iter().enumerate() {
        let label = Label { index, many };
        let old = lf(&replacement.old_text);
        let new = lf(&replacement.new_text);
        if old.is_empty() {
            failed.push((
                index,
                format!(
                    "{label}`old_text` is empty; to add text, replace a nearby line with itself plus the new text"
                ),
            ));
            continue;
        }
        if old == new {
            failed.push((
                index,
                format!(
                    "{label}`old_text` and `new_text` are the same, so it changes nothing; drop it or fix `new_text`"
                ),
            ));
            continue;
        }
        match locate(text, &old, replacement.replace_all) {
            Err(miss) => failed.push((index, format!("{label}{}", miss.explain(&old)))),
            Ok(found) => {
                if let Some(note) = found.note {
                    notes.push(format!("{label}{note}"));
                }
                splices.extend(found.spans.into_iter().map(|span| Splice {
                    index,
                    range: span,
                    text: new.clone(),
                }));
            }
        }
    }
    if !failed.is_empty() {
        return Err(ToolExecutionError::invalid_args(batch_failure(
            path, count, &failed,
        )));
    }
    splices.sort_by_key(|splice| splice.range.start);
    for pair in splices.windows(2) {
        if let [first, second] = pair
            && first.range.end > second.range.start
        {
            return Err(refused(path, overlap(text, first, second, many)));
        }
    }
    let mut edited = String::with_capacity(text.len());
    let mut at = 0;
    for splice in &splices {
        edited.push_str(text.get(at..splice.range.start).unwrap_or_default());
        edited.push_str(&splice.text);
        at = splice.range.end;
    }
    edited.push_str(text.get(at..).unwrap_or_default());
    if edited == text {
        return Err(refused(
            path,
            "the edits leave the file as it was; check `new_text`".to_owned(),
        ));
    }
    write_atomic(Path::new(path), file.encode(&edited).as_bytes())
        .map_err(|error| io_error(path, error))?;

    let replaced = splices.len();
    let mut out = format!(
        "Edited {path}: {replaced} replacement{}.",
        if replaced == 1 { "" } else { "s" }
    );
    for note in notes {
        out.push_str(&format!("\nNote: {note}"));
    }
    out.push('\n');
    out.push_str(&diff(path, text, &edited));
    Ok(out)
}

/// The file's text with LF line endings and no byte-order mark, and how to
/// put both back.
struct Decoded {
    text: String,
    bom: bool,
    crlf: bool,
}

impl Decoded {
    fn new(raw: &str) -> Self {
        let (bom, body) = match raw.strip_prefix('\u{feff}') {
            Some(body) => (true, body),
            None => (false, raw),
        };
        // Only a file whose every line ends in CRLF is converted; a mixed
        // one is edited as it is, and loose matching ignores its `\r`s.
        let newlines = body.matches('\n').count();
        let crlf = newlines > 0 && body.matches("\r\n").count() == newlines;
        let text = if crlf {
            body.replace("\r\n", "\n")
        } else {
            body.to_owned()
        };
        Self { text, bom, crlf }
    }

    /// `text` with the file's line endings and byte-order mark.
    fn encode(&self, text: &str) -> String {
        let body = if self.crlf {
            text.replace('\n', "\r\n")
        } else {
            text.to_owned()
        };
        if self.bom {
            format!("\u{feff}{body}")
        } else {
            body
        }
    }
}

/// `text` with CRLF line endings made LF, as the file is matched.
fn lf(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// Names an edit in messages, when the call has several.
#[derive(Clone, Copy)]
struct Label {
    index: usize,
    many: bool,
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.many {
            write!(f, "edits[{}]: ", self.index)
        } else {
            Ok(())
        }
    }
}

/// A model-visible refusal: nothing was written.
fn refused(path: &str, why: String) -> ToolExecutionError {
    ToolExecutionError::invalid_args(format!(
        "{why}. {path} was not changed; fix the call and send all of its edits again."
    ))
}

/// The refusal for edits that failed: each one's reason, and, in a batch,
/// that the edits that matched were not applied either and must be sent
/// again with the fixed ones.
fn batch_failure(path: &str, count: usize, failed: &[(usize, String)]) -> String {
    let [(_, why)] = failed else {
        let reasons: Vec<&str> = failed.iter().map(|(_, why)| why.as_str()).collect();
        return format!(
            "{} of the {count} edits failed:\n{}.\nNo edit was applied, so {path} was not \
             changed. Send all {count} edits again, with these fixed{}.",
            failed.len(),
            reasons.join(".\n"),
            matched(count, failed)
        );
    };
    if count == 1 {
        return format!("{why}. {path} was not changed; fix the edit and send it again.");
    }
    format!(
        "{why}. No edit was applied, so {path} was not changed. Send all {count} edits again, \
         with this one fixed{}.",
        matched(count, failed)
    )
}

/// `; edits[0], edits[2] matched and stay as they are`, for the edits of a
/// batch of `count` that did not fail.
fn matched(count: usize, failed: &[(usize, String)]) -> String {
    let matched: Vec<String> = (0..count)
        .filter(|index| !failed.iter().any(|(failed, _)| failed == index))
        .map(|index| format!("edits[{index}]"))
        .collect();
    match matched.as_slice() {
        [] => String::new(),
        [one] => format!("; {one} matched and can be sent as it was"),
        many => format!("; {} matched and can be sent as they were", many.join(", ")),
    }
}

/// One replacement of a byte range of the original text.
struct Splice {
    index: usize,
    range: Range<usize>,
    text: String,
}

/// The error for two edits whose matches overlap.
fn overlap(text: &str, first: &Splice, second: &Splice, many: bool) -> String {
    let lines = |splice: &Splice| {
        let (start, end) = (
            line_of(text, splice.range.start),
            line_of(
                text,
                splice.range.end.saturating_sub(1).max(splice.range.start),
            ),
        );
        if start == end {
            format!("line {start}")
        } else {
            format!("lines {start}-{end}")
        }
    };
    if first.index == second.index || !many {
        format!(
            "two matches of `old_text` overlap ({} and {}); make `old_text` unique instead of \
             using `replace_all`",
            lines(first),
            lines(second)
        )
    } else {
        format!(
            "edits[{}] ({}) and edits[{}] ({}) overlap; merge them into one edit, or make \
             them target separate text",
            first.index,
            lines(first),
            second.index,
            lines(second)
        )
    }
}

/// The 1-based line holding byte `at` of `text`.
fn line_of(text: &str, at: usize) -> usize {
    text.get(..at)
        .map_or(0, |before| before.matches('\n').count())
        + 1
}

/// Where an edit's `old_text` matched.
struct Found {
    spans: Vec<Range<usize>>,
    /// How the match was loose, if it was.
    note: Option<String>,
}

/// Why an edit's `old_text` did not match once.
enum Miss {
    /// It matches nowhere.
    NotFound,
    /// It matches at these 1-based lines, `loose` when only after evening
    /// out whitespace or punctuation.
    Ambiguous { lines: Vec<usize>, loose: bool },
}

impl Miss {
    /// What went wrong and what to send instead.
    fn explain(&self, old: &str) -> String {
        match self {
            Miss::Ambiguous { lines, loose } => {
                let listed: Vec<String> = lines
                    .iter()
                    .take(MAX_LISTED)
                    .map(ToString::to_string)
                    .collect();
                let more = lines.len().saturating_sub(MAX_LISTED);
                let more = if more > 0 {
                    format!(" and {more} more")
                } else {
                    String::new()
                };
                format!(
                    "`old_text` {}matches {} places, starting at lines {}{more}; add the lines \
                     around the one to change so it matches once, or set `replace_all` to \
                     change every one",
                    if *loose {
                        "does not match exactly, and loosely "
                    } else {
                        ""
                    },
                    lines.len(),
                    listed.join(", ")
                )
            }
            Miss::NotFound => not_found(old),
        }
    }
}

/// Finds `old` in `text`: exactly, else line by line with trailing
/// whitespace and typographic punctuation evened out. A loose match must be
/// the only one; `replace_all` takes exact matches only.
fn locate(text: &str, old: &str, replace_all: bool) -> Result<Found, Miss> {
    let exact: Vec<Range<usize>> = text
        .match_indices(old)
        .map(|(start, matched)| start..start + matched.len())
        .collect();
    match exact.len() {
        0 => {}
        1 => {
            return Ok(Found {
                spans: exact,
                note: None,
            });
        }
        _ if replace_all => {
            return Ok(Found {
                spans: exact,
                note: None,
            });
        }
        _ => {
            return Err(Miss::Ambiguous {
                lines: exact.iter().map(|span| line_of(text, span.start)).collect(),
                loose: false,
            });
        }
    }
    if replace_all {
        return Err(Miss::NotFound);
    }
    let lines = Lines::new(text);
    let wanted: Vec<&str> = old.strip_suffix('\n').unwrap_or(old).split('\n').collect();
    let start = match lines.windows(&wanted).as_slice() {
        [] => return Err(Miss::NotFound),
        [start] => *start,
        starts => {
            return Err(Miss::Ambiguous {
                lines: starts.iter().map(|start| start + 1).collect(),
                loose: true,
            });
        }
    };
    Ok(Found {
        spans: vec![lines.span(start, wanted.len(), old.ends_with('\n'))],
        note: Some(format!(
            "`old_text` matched lines {}-{} only after evening out trailing whitespace and \
             quote, dash or space characters; copy text exactly next time",
            start + 1,
            start + wanted.len()
        )),
    })
}

/// `line` as loosely compared: without trailing whitespace, and with
/// typographic quotes, dashes and spaces made ASCII.
fn loose_key(line: &str) -> String {
    line.trim_end().chars().map(plain).collect()
}

/// The ASCII form of a typographic quote, dash or space.
fn plain(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        c => c,
    }
}

/// The lines of a text, with where each starts and ends.
struct Lines<'a> {
    text: &'a str,
    /// For each line, its byte range without and with its `\n`.
    spans: Vec<(Range<usize>, usize)>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        let mut spans = Vec::new();
        let mut start = 0;
        for line in text.split_inclusive('\n') {
            let end = start + line.len();
            let content = end - usize::from(line.ends_with('\n'));
            spans.push((start..content, end));
            start = end;
        }
        Self { text, spans }
    }

    fn line(&self, index: usize) -> &'a str {
        self.spans
            .get(index)
            .and_then(|(content, _)| self.text.get(content.clone()))
            .unwrap_or_default()
    }

    /// The 0-based first lines of each run of lines that loosely equals
    /// `wanted`.
    fn windows(&self, wanted: &[&str]) -> Vec<usize> {
        let keys: Vec<String> = (0..self.spans.len())
            .map(|index| loose_key(self.line(index)))
            .collect();
        let wanted: Vec<String> = wanted.iter().map(|line| loose_key(line)).collect();
        if wanted.iter().all(String::is_empty) {
            return Vec::new();
        }
        keys.windows(wanted.len().max(1))
            .enumerate()
            .filter(|(_, window)| *window == wanted.as_slice())
            .map(|(start, _)| start)
            .collect()
    }

    /// The bytes of `count` lines from `start`, with the last line's `\n`
    /// when `whole_lines`.
    fn span(&self, start: usize, count: usize, whole_lines: bool) -> Range<usize> {
        let first = self
            .spans
            .get(start)
            .map_or(self.text.len(), |(content, _)| content.start);
        let last = self.spans.get(start + count - 1);
        let end = match last {
            Some((_, with_newline)) if whole_lines => *with_newline,
            Some((content, _)) => content.end,
            None => self.text.len(),
        };
        first..end
    }
}

/// Why `old` matched nowhere: line numbers copied from `read`, or the
/// file changed since it was read.
fn not_found(old: &str) -> String {
    let wanted: Vec<&str> = old.strip_suffix('\n').unwrap_or(old).split('\n').collect();
    if wanted.iter().all(|line| numbered(line)) {
        return "`old_text` was not found: it starts with line numbers as `read` shows them; \
                copy the text after the tab only"
            .to_owned();
    }
    "`old_text` was not found, not even ignoring trailing whitespace; the file may have \
     changed since it was read, so read it again and copy `old_text` from what it holds now"
        .to_owned()
}

/// Whether `line` looks like a line `read` returned: spaces, a number and
/// a tab.
fn numbered(line: &str) -> bool {
    line.trim_start()
        .split_once('\t')
        .is_some_and(|(number, _)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()))
}

/// The unified diff from `old` to `new`, cut to what a tool may return.
fn diff(path: &str, old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(CONTEXT)
        .header(path, path)
        .to_string();
    let lines: Vec<&str> = diff.lines().collect();
    let mut out = String::new();
    for (count, line) in lines.iter().enumerate() {
        if count >= MAX_LINES / 4 || out.len() + line.len() > MAX_BYTES / 2 {
            let rest = lines.len() - count;
            let _ = writeln!(out, "[{rest} more diff lines; read the file to see them]");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests;
