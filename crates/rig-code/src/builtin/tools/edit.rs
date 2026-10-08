//! The `edit` tool: several replacements in one file per call, all matched
//! against the file as it was before the call and applied together or not
//! at all. A byte-order mark and CRLF line endings are kept. Text that does
//! not match exactly may still match line by line once trailing
//! whitespace and typographic quotes, dashes and spaces are evened out, or,
//! failing that, indentation too, but only where exactly one place in the
//! file matches. The result is a unified diff of the change; a failure
//! says what to send instead.

use std::fmt::Write as _;
use std::ops::Range;

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;
use similar::TextDiff;

use super::{MAX_BYTES, MAX_LINES, clip, read_text, write_atomic};
use crate::core::blocking::blocking;
use crate::core::workdir;

/// Lines of unchanged context around each hunk of the returned diff.
const CONTEXT: usize = 3;
/// Most matches an ambiguity error lists by line.
const MAX_LISTED: usize = 8;
/// Most line comparisons spent looking for the closest text when
/// `old_text` is not found, so a huge file does not stall the call.
const MAX_HINT_WORK: usize = 4_000_000;
/// Characters of a line quoted in an error.
const MAX_QUOTE: usize = 160;

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
    let resolved = workdir::resolve(&args.path);
    let path = resolved.as_str();
    if args.edits.is_empty() {
        return Err(ToolExecutionError::invalid_args(format!(
            "`edits` is empty, so {path} was not changed; give at least one \
             {{\"old_text\", \"new_text\"}}"
        )));
    }
    let raw = read_text(path)?;
    let file = Decoded::new(&raw);
    let text = file.text.as_str();
    let many = args.edits.len() > 1;
    let mut splices = Vec::new();
    let mut notes = Vec::new();
    for (index, replacement) in args.edits.iter().enumerate() {
        let label = Label { index, many };
        let old = lf(&replacement.old_text);
        let new = lf(&replacement.new_text);
        if old.is_empty() {
            return Err(refused(
                path,
                format!(
                    "{label}`old_text` is empty; to add text, replace a nearby line with itself plus the new text"
                ),
            ));
        }
        if old == new {
            return Err(refused(
                path,
                format!(
                    "{label}`old_text` and `new_text` are the same, so it changes nothing; drop it or fix `new_text`"
                ),
            ));
        }
        let found = locate(text, &old, &new, replacement.replace_all)
            .map_err(|miss| refused(path, format!("{label}{}", miss.explain(text, &old))))?;
        if let Some(note) = found.note {
            notes.push(format!("{label}{note}"));
        }
        splices.extend(found.spans.into_iter().map(|span| Splice {
            index,
            range: span,
            text: found.text.clone(),
        }));
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
    write_atomic(path, file.encode(&edited).as_bytes())?;

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

/// Where an edit's `old_text` matched, and what replaces it.
struct Found {
    spans: Vec<Range<usize>>,
    /// The replacement: `new_text`, re-indented after a match that ignored
    /// indentation.
    text: String,
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
    /// It matches only once indentation is ignored, at these lines, but the
    /// file's indentation differs from `old_text`'s unevenly.
    Uneven { lines: Range<usize> },
}

impl Miss {
    /// What went wrong and what to send instead.
    fn explain(&self, text: &str, old: &str) -> String {
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
            Miss::Uneven { lines } => format!(
                "`old_text` matches lines {}-{} only when indentation is ignored, and its \
                 indentation differs from the file's unevenly; copy those lines again with the \
                 file's exact indentation",
                lines.start, lines.end
            ),
            Miss::NotFound => not_found(text, old),
        }
    }
}

/// Finds `old` in `text`: exactly, else line by line with whitespace and
/// punctuation evened out, else with indentation ignored too. A loose match
/// must be the only one at its level; `replace_all` takes exact matches
/// only.
fn locate(text: &str, old: &str, new: &str, replace_all: bool) -> Result<Found, Miss> {
    let exact: Vec<Range<usize>> = text
        .match_indices(old)
        .map(|(start, matched)| start..start + matched.len())
        .collect();
    match exact.len() {
        0 => {}
        1 => {
            return Ok(Found {
                spans: exact,
                text: new.to_owned(),
                note: None,
            });
        }
        _ if replace_all => {
            return Ok(Found {
                spans: exact,
                text: new.to_owned(),
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
    let whole_lines = old.ends_with('\n');
    for level in [Looseness::Spacing, Looseness::Indentation] {
        let starts = lines.windows(&wanted, level);
        let start = match starts.as_slice() {
            [] => continue,
            [start] => *start,
            _ => {
                return Err(Miss::Ambiguous {
                    lines: starts.iter().map(|start| start + 1).collect(),
                    loose: true,
                });
            }
        };
        let span = lines.span(start, wanted.len(), whole_lines);
        let numbers = start + 1..start + wanted.len();
        return match level {
            Looseness::Spacing => Ok(Found {
                spans: vec![span],
                text: new.to_owned(),
                note: Some(format!(
                    "`old_text` matched lines {}-{} only after evening out trailing \
                     whitespace and quote, dash or space characters; copy text exactly next time",
                    numbers.start, numbers.end
                )),
            }),
            Looseness::Indentation => {
                let file: Vec<&str> = lines.text(start, wanted.len()).collect();
                let shift = Shift::between(&file, &wanted).ok_or(Miss::Uneven {
                    lines: numbers.clone(),
                })?;
                Ok(Found {
                    spans: vec![span],
                    text: shift.apply(new),
                    note: Some(format!(
                        "`old_text` matched lines {}-{} only with indentation ignored; \
                         `new_text` was {}",
                        numbers.start,
                        numbers.end,
                        shift.describe()
                    )),
                })
            }
        };
    }
    Err(Miss::NotFound)
}

/// How loosely lines are compared.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Looseness {
    /// Trailing whitespace, typographic quotes, dashes and spaces evened
    /// out.
    Spacing,
    /// [`Spacing`](Self::Spacing), and leading whitespace ignored.
    Indentation,
}

impl Looseness {
    /// `line` as compared at this level.
    fn key(self, line: &str) -> String {
        let line = line.trim_end();
        let line = match self {
            Looseness::Spacing => line,
            Looseness::Indentation => line.trim_start(),
        };
        line.chars().map(plain).collect()
    }
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

    /// The `count` lines from `start`.
    fn text(&self, start: usize, count: usize) -> impl Iterator<Item = &'a str> + '_ {
        (start..start + count).map(|index| self.line(index))
    }

    /// The 0-based first lines of each run of lines that equals `wanted`
    /// at `level`.
    fn windows(&self, wanted: &[&str], level: Looseness) -> Vec<usize> {
        let keys: Vec<String> = (0..self.spans.len())
            .map(|index| level.key(self.line(index)))
            .collect();
        let wanted: Vec<String> = wanted.iter().map(|line| level.key(line)).collect();
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

/// The indentation the file has beyond `old_text`'s, or that `old_text` has
/// beyond the file's, the same on every line that is not blank.
enum Shift {
    Add(String),
    Remove(String),
}

impl Shift {
    fn between(file: &[&str], old: &[&str]) -> Option<Self> {
        let indent = |line: &str| line.len() - line.trim_start().len();
        let mut shift: Option<Shift> = None;
        for (file_line, old_line) in file.iter().zip(old) {
            if file_line.trim().is_empty() || old_line.trim().is_empty() {
                continue;
            }
            let file_indent = file_line.get(..indent(file_line)).unwrap_or_default();
            let old_indent = old_line.get(..indent(old_line)).unwrap_or_default();
            let this = if let Some(extra) = file_indent.strip_suffix(old_indent) {
                Shift::Add(extra.to_owned())
            } else if let Some(extra) = old_indent.strip_suffix(file_indent) {
                Shift::Remove(extra.to_owned())
            } else {
                return None;
            };
            match &shift {
                None => shift = Some(this),
                Some(seen) if seen.same(&this) => {}
                Some(_) => return None,
            }
        }
        shift
    }

    fn same(&self, other: &Shift) -> bool {
        match (self, other) {
            (Shift::Add(a), Shift::Add(b)) | (Shift::Remove(a), Shift::Remove(b)) => a == b,
            (Shift::Add(a), Shift::Remove(b)) | (Shift::Remove(a), Shift::Add(b)) => {
                a.is_empty() && b.is_empty()
            }
        }
    }

    /// `new` with the shift applied to every line that is not blank.
    fn apply(&self, new: &str) -> String {
        new.split_inclusive('\n')
            .map(|line| {
                if line.trim().is_empty() {
                    return line.to_owned();
                }
                match self {
                    Shift::Add(extra) => format!("{extra}{line}"),
                    Shift::Remove(extra) => {
                        line.strip_prefix(extra.as_str()).unwrap_or(line).to_owned()
                    }
                }
            })
            .collect()
    }

    fn describe(&self) -> String {
        let what = |extra: &str| {
            let tabs = extra.chars().filter(|c| *c == '\t').count();
            let spaces = extra.chars().count() - tabs;
            match (spaces, tabs) {
                (spaces, 0) => format!("{spaces} space{}", if spaces == 1 { "" } else { "s" }),
                (0, tabs) => format!("{tabs} tab{}", if tabs == 1 { "" } else { "s" }),
                (spaces, tabs) => format!("{spaces} spaces and {tabs} tabs"),
            }
        };
        match self {
            Shift::Add(extra) if extra.is_empty() => "kept as it was".to_owned(),
            Shift::Remove(extra) if extra.is_empty() => "kept as it was".to_owned(),
            Shift::Add(extra) => format!("indented by {} more to fit the file", what(extra)),
            Shift::Remove(extra) => format!("indented by {} less to fit the file", what(extra)),
        }
    }
}

/// Why `old` matched nowhere in `text`, with the likely cause or the
/// closest lines of the file.
fn not_found(text: &str, old: &str) -> String {
    let wanted: Vec<&str> = old.strip_suffix('\n').unwrap_or(old).split('\n').collect();
    if wanted.iter().all(|line| numbered(line)) {
        return "`old_text` was not found: it starts with line numbers as `read` shows them; \
                copy the text after the tab only"
            .to_owned();
    }
    let lines = Lines::new(text);
    let mut out = "`old_text` was not found, not even ignoring whitespace".to_owned();
    match closest(&lines, &wanted) {
        Some(close) => {
            let file_line = lines.line(close.start + close.differs);
            let old_line = wanted.get(close.differs).copied().unwrap_or_default();
            let _ = write!(
                out,
                ". The closest text is lines {}-{} ({} of {} lines agree); the first \
                 difference is line {}, which in the file is\n  {}\nbut in `old_text` is\n  \
                 {}\nRead those lines again and copy them exactly",
                close.start + 1,
                close.start + wanted.len(),
                close.agree,
                wanted.len(),
                close.start + close.differs + 1,
                quote(file_line),
                quote(old_line)
            );
        }
        None => out.push_str(
            "; the file may have changed since it was read, so read it again and copy \
             `old_text` from what it holds now",
        ),
    }
    out
}

/// Whether `line` looks like a line `read` returned: spaces, a number and
/// a tab.
fn numbered(line: &str) -> bool {
    line.trim_start()
        .split_once('\t')
        .is_some_and(|(number, _)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()))
}

/// The run of lines most like `wanted`.
struct Close {
    /// Its 0-based first line.
    start: usize,
    /// How many of its lines agree with `wanted`'s, indentation ignored.
    agree: usize,
    /// The first line, counted from `start`, that does not agree.
    differs: usize,
}

/// The run of lines that agrees with `wanted` on the most lines,
/// indentation ignored, if any agrees on a line that is not blank and
/// looking is cheap enough.
fn closest(lines: &Lines<'_>, wanted: &[&str]) -> Option<Close> {
    let count = lines.spans.len();
    if wanted.len() > count || count.saturating_mul(wanted.len()) > MAX_HINT_WORK {
        return None;
    }
    let level = Looseness::Indentation;
    let keys: Vec<String> = (0..count)
        .map(|index| level.key(lines.line(index)))
        .collect();
    let wanted: Vec<String> = wanted.iter().map(|line| level.key(line)).collect();
    let mut best: Option<Close> = None;
    for (start, window) in keys.windows(wanted.len().max(1)).enumerate() {
        let agree = window
            .iter()
            .zip(&wanted)
            .filter(|(file, old)| file == old && !old.is_empty())
            .count();
        if agree > 0 && best.as_ref().is_none_or(|best| agree > best.agree) {
            let differs = window
                .iter()
                .zip(&wanted)
                .position(|(file, old)| file != old)
                .unwrap_or(0);
            best = Some(Close {
                start,
                agree,
                differs,
            });
        }
    }
    best
}

/// `line` for an error message: cut short, with its whitespace visible
/// enough to compare.
fn quote(line: &str) -> String {
    let shown = clip(line, MAX_QUOTE);
    let cut = if shown.len() < line.len() { "…" } else { "" };
    format!("`{}{cut}`", shown.replace('\t', "\\t"))
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
