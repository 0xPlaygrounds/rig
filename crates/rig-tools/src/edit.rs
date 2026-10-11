//! The `edit` tool: several replacements in one file per call, all matched
//! against the file as it was before the call and applied together or not
//! at all. A byte-order mark and CRLF line endings are kept. Text that does
//! not match exactly may still match line by line once trailing
//! whitespace and typographic quotes, dashes and spaces are evened out, but
//! only where exactly one place in the file matches. The result is a unified diff of the change; a failure
//! says what to send instead.

use std::fmt::Write as _;
use std::ops::Range;
use std::path::PathBuf;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    FileChange, FilesChanged, Tool, ToolContext, ToolExecutionError, args_schema, content_hash,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::{file_change, io_error, place, read_text, write_atomic};
use crate::{MAX_BYTES, MAX_LINES, blocking};

/// Most matches an ambiguity error lists by line.
const MAX_LISTED: usize = 8;

/// Replaces text in a file. Publishes the file it changed
/// ([`FilesChanged`]): its [resolved](crate::fs::resolve) path, and the
/// whole diff.
#[derive(Clone, Debug, Default)]
pub struct Edit {
    /// The directory a relative path is edited in; `None` is this
    /// process's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is read.
    pub confine: bool,
}

/// Arguments of [`Edit`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditArgs {
    /// The file to edit.
    pub path: String,
    /// The replacements, matched against the original file.
    #[schemars(length(min = 1))]
    pub edits: Vec<Replacement>,
}

impl EditArgs {
    /// Make `edits` to `path`.
    pub fn new(path: impl Into<String>, edits: impl IntoIterator<Item = Replacement>) -> Self {
        Self {
            path: path.into(),
            edits: edits.into_iter().collect(),
        }
    }
}

// One replacement of an `edit` call; not a doc comment, which the model would see.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Replacement {
    /// The exact text to replace, copied from the file without line numbers.
    pub old_text: String,
    /// The replacement.
    pub new_text: String,
    /// Replace every match instead of exactly one.
    #[serde(default)]
    pub replace_all: bool,
}

impl Replacement {
    /// Replace the one match of `old_text` with `new_text`.
    pub fn new(old_text: impl Into<String>, new_text: impl Into<String>) -> Self {
        Self {
            old_text: old_text.into(),
            new_text: new_text.into(),
            replace_all: false,
        }
    }

    /// The same replacement, of every match.
    pub fn all(mut self) -> Self {
        self.replace_all = true;
        self
    }
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

impl Tool for Edit {
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
        args_schema::<EditArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(false)
            .with_open_world(false)
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: EditArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = place(self.base.as_deref(), self.confine, &args.path)?;
        let (output, change) = blocking(move || edit(path, args)).await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(output)
    }
}

fn edit(resolved: PathBuf, args: EditArgs) -> Result<(String, FileChange), ToolExecutionError> {
    let path = resolved.to_string_lossy();
    let raw = read_text(&path)?;
    let applied = apply(&path, &raw, &args.edits)?;
    write_atomic(&resolved, applied.text.as_bytes()).map_err(|error| io_error(&path, error))?;
    let replaced = applied.replaced;
    let mut out = format!(
        "Edited {}: {replaced} replacement{}.",
        args.path,
        if replaced == 1 { "" } else { "s" }
    );
    for note in applied.notes {
        out.push_str(&format!("\nNote: {note}"));
    }
    out.push('\n');
    out.push_str(&shown(applied.change.diff.as_deref().unwrap_or_default()));
    Ok((out, applied.change))
}

/// What [`apply`] makes of a file's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    /// The edited text, with the byte-order mark and line endings the file
    /// had: what [`Edit`] writes.
    pub text: String,
    /// What the edit does to the file, with its diff and the hash of
    /// [`text`](Self::text).
    pub change: FileChange,
    /// How many places were replaced.
    pub replaced: usize,
    /// How each edit that matched only loosely was matched, for the model.
    pub notes: Vec<String>,
}

/// Makes `edits` to `text`, the whole text of the file at `path`, exactly
/// as [`Edit`] does, without touching the file: every edit is matched
/// against the text as it is, exactly or loosely, and all are applied or
/// none. A byte-order mark and CRLF line endings are kept. `path` names the
/// file in the errors and the [`FileChange`].
///
/// ```
/// use rig_tools::{Replacement, edit::apply};
///
/// let applied = apply("a.txt", "one\r\ntwo\r\n", &[Replacement::new("two", "2")])?;
/// assert_eq!(applied.text, "one\r\n2\r\n");
/// assert_eq!((applied.change.added, applied.change.removed), (1, 1));
/// assert!(apply("a.txt", "one\n", &[Replacement::new("three", "3")]).is_err());
/// # Ok::<(), rig_core::tool::ToolExecutionError>(())
/// ```
///
/// # Errors
///
/// When an edit matches nowhere or more than once, edits overlap, or they
/// change nothing: the error says what to send instead.
pub fn apply(path: &str, text: &str, edits: &[Replacement]) -> Result<Applied, ToolExecutionError> {
    if edits.is_empty() {
        return Err(ToolExecutionError::invalid_args(format!(
            "`edits` is empty, so {path} was not changed; give at least one \
             {{\"old_text\", \"new_text\"}}"
        )));
    }
    let file = Decoded::new(text);
    let text = file.text.as_str();
    let count = edits.len();
    let many = count > 1;
    let mut splices = Vec::new();
    let mut notes = Vec::new();
    // Every edit is checked, so one reply names all that need fixing.
    let mut failed: Vec<String> = Vec::new();
    for (index, replacement) in edits.iter().enumerate() {
        let label = if many {
            format!("edits[{index}]: ")
        } else {
            String::new()
        };
        let old = lf(&replacement.old_text);
        let new = lf(&replacement.new_text);
        if old.is_empty() {
            failed.push(format!(
                "{label}`old_text` is empty; to add text, replace a nearby line with itself plus the new text"
            ));
            continue;
        }
        if old == new {
            failed.push(format!(
                "{label}`old_text` and `new_text` are the same, so it changes nothing; drop it or fix `new_text`"
            ));
            continue;
        }
        match locate(text, &old, replacement.replace_all) {
            Err(miss) => failed.push(format!("{label}{}", miss.explain())),
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
    let encoded = file.encode(&edited);
    Ok(Applied {
        change: file_change(path, Some(text), &edited).with_after(content_hash(encoded.as_bytes())),
        text: encoded,
        replaced: splices.len(),
        notes,
    })
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

/// A model-visible refusal: nothing was written.
fn refused(path: &str, why: String) -> ToolExecutionError {
    ToolExecutionError::invalid_args(format!(
        "{why}. {path} was not changed; fix the call and send all of its edits again."
    ))
}

/// The refusal for edits that failed: each one's reason, and, in a batch,
/// that the edits that matched were not applied either and must be sent
/// again with the fixed ones.
fn batch_failure(path: &str, count: usize, failed: &[String]) -> String {
    let [why] = failed else {
        return format!(
            "{} of the {count} edits failed:\n{}.\nNo edit was applied, so {path} was not \
             changed. Send all {count} edits again, with these fixed.",
            failed.len(),
            failed.join(".\n"),
        );
    };
    if count == 1 {
        return format!("{why}. {path} was not changed; fix the edit and send it again.");
    }
    format!(
        "{why}. No edit was applied, so {path} was not changed. Send all {count} edits again, \
         with this one fixed."
    )
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
    fn explain(&self) -> String {
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
            Miss::NotFound => "`old_text` was not found, not even ignoring trailing whitespace; \
                the file may have changed since it was read, so read it again and copy \
                `old_text` from what it holds now"
                .to_owned(),
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
        count if count == 1 || replace_all => {
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

/// `diff`, a unified diff, cut to what a tool may return.
fn shown(diff: &str) -> String {
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
