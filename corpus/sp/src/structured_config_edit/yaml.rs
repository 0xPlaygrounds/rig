//! YAML files: `serde_yaml` checks the file and reads its values; a
//! locator of block mappings and sequences finds where a value lies, so a
//! change splices only its own lines. Flow collections (`{a: 1}`,
//! `[1, 2]`) are values as a whole: one is set or removed whole, never
//! edited inside.

use std::ops::Range;

use serde_json::Value;

use crate::structured_config_edit::key::Seg;
use crate::structured_config_edit::text::{Splice, indent_unit, newline};

/// Reads `text` as YAML, as JSON.
///
/// # Errors
///
/// Where it stops parsing, by line and column, or a value JSON cannot hold
/// (a key that is not a string).
pub fn parse(text: &str) -> Result<Value, String> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(text).map_err(|error| {
        let at = error
            .location()
            .map(|at| format!(" at line {}, column {}", at.line(), at.column()))
            .unwrap_or_default();
        format!("does not parse as YAML{at}: {error}")
    })?;
    serde_json::to_value(yaml).map_err(|error| format!("cannot be read as JSON: {error}"))
}

/// One line of the file.
struct Line {
    start: usize,
    /// Where its text ends, before `\r\n` or `\n`.
    end: usize,
    indent: usize,
    /// Neither blank, a comment alone, nor a document marker.
    content: bool,
}

/// A located value.
enum Node {
    Map {
        col: usize,
        entries: Vec<Entry>,
    },
    Seq {
        items: Vec<Entry>,
    },
    /// A scalar or flow collection, possibly over several lines.
    Leaf {
        span: Range<usize>,
    },
    /// Nothing after the `:` or `-`.
    Empty,
}

/// A mapping's entry or a sequence's item.
struct Entry {
    /// The key, for a mapping's entry.
    key: String,
    /// Where it starts: its key or its `-`.
    start: usize,
    /// Right after its `:` or `-`.
    after: usize,
    /// Its first and last lines.
    first: usize,
    last: usize,
    value: Node,
}

struct Locator<'t> {
    text: &'t str,
    lines: Vec<Line>,
}

impl<'t> Locator<'t> {
    fn new(text: &'t str) -> Self {
        let mut lines = Vec::new();
        let mut start = 0;
        for raw in text.split_inclusive('\n') {
            let body = raw.trim_end_matches(['\n', '\r']);
            let trimmed = body.trim_start_matches(' ');
            let content = !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !(body.starts_with("---") || body.starts_with("..."));
            lines.push(Line {
                start,
                end: start + body.len(),
                indent: body.len() - trimmed.len(),
                content,
            });
            start += raw.len();
        }
        Self { text, lines }
    }

    /// The text of line `line` from column `col` to its end.
    fn rest(&self, line: usize, col: usize) -> &'t str {
        self.lines
            .get(line)
            .and_then(|it| self.text.get(it.start + col..it.end))
            .unwrap_or_default()
    }

    fn next_content(&self, from: usize) -> Option<usize> {
        (from..self.lines.len()).find(|at| self.lines.get(*at).is_some_and(|line| line.content))
    }

    fn indent(&self, line: usize) -> usize {
        self.lines.get(line).map_or(0, |it| it.indent)
    }

    fn root(&self) -> Option<Node> {
        let first = self.next_content(0)?;
        self.node(first, self.indent(first), None)
            .map(|(node, _)| node)
    }

    /// The value starting at column `col` of line `line`, inside a
    /// collection at column `parent`, and its last line.
    fn node(&self, line: usize, col: usize, parent: Option<usize>) -> Option<(Node, usize)> {
        let rest = self.rest(line, col);
        if is_dash(rest) {
            self.seq(line, col)
        } else if key_of(rest).is_some() {
            self.map(line, col)
        } else {
            Some(self.leaf(line, col, parent))
        }
    }

    fn leaf(&self, line: usize, col: usize, parent: Option<usize>) -> (Node, usize) {
        let start = self.lines.get(line).map_or(0, |it| it.start + col);
        let mut last = line;
        let mut next = line + 1;
        while let Some(at) = self.next_content(next) {
            if parent.is_some_and(|parent| self.indent(at) <= parent) {
                break;
            }
            last = at;
            next = at + 1;
        }
        let end = if last == line {
            let rest = self.rest(line, col);
            start + without_comment(rest).trim_end().len()
        } else {
            self.lines.get(last).map_or(start, |it| it.end)
        };
        (Node::Leaf { span: start..end }, last)
    }

    /// The value after a `:` or `-` that ends at column `after` of `line`,
    /// in a collection at column `col`.
    fn value_after(&self, line: usize, after: usize, col: usize) -> Option<(Node, usize)> {
        let rest = self.rest(line, after);
        let trimmed = rest.trim_start_matches(' ');
        if trimmed.is_empty() || trimmed.starts_with('#') {
            match self.next_content(line + 1) {
                Some(next) if self.indent(next) > col => {
                    self.node(next, self.indent(next), Some(col))
                }
                Some(next) if self.indent(next) == col && is_dash(self.rest(next, col)) => {
                    self.seq(next, col)
                }
                _ => Some((Node::Empty, line)),
            }
        } else {
            let at = after + rest.len() - trimmed.len();
            Some(self.leaf(line, at, Some(col)))
        }
    }

    fn map(&self, line: usize, col: usize) -> Option<(Node, usize)> {
        let mut entries: Vec<Entry> = Vec::new();
        let mut at = line;
        loop {
            let (key, after) = key_of(self.rest(at, col))?;
            let start = self.lines.get(at)?.start + col;
            let (value, last) = self.value_after(at, col + after, col)?;
            entries.push(Entry {
                key,
                start,
                after: start + after,
                first: at,
                last,
                value,
            });
            match self.next_content(last + 1) {
                Some(next)
                    if self.indent(next) == col
                        && !is_dash(self.rest(next, col))
                        && key_of(self.rest(next, col)).is_some() =>
                {
                    at = next;
                }
                _ => break,
            }
        }
        let last = entries.last()?.last;
        Some((Node::Map { col, entries }, last))
    }

    fn seq(&self, line: usize, col: usize) -> Option<(Node, usize)> {
        let mut items: Vec<Entry> = Vec::new();
        let mut at = line;
        loop {
            let start = self.lines.get(at)?.start + col;
            let rest = self.rest(at, col + 1);
            let trimmed = rest.trim_start_matches(' ');
            let (value, last) = if trimmed.is_empty() || trimmed.starts_with('#') {
                self.value_after(at, col + 1, col)?
            } else {
                self.node(at, col + 1 + rest.len() - trimmed.len(), Some(col))?
            };
            items.push(Entry {
                key: String::new(),
                start,
                after: start + 1,
                first: at,
                last,
                value,
            });
            match self.next_content(last + 1) {
                Some(next) if self.indent(next) == col && is_dash(self.rest(next, col)) => {
                    at = next;
                }
                _ => break,
            }
        }
        let last = items.last()?.last;
        Some((Node::Seq { items }, last))
    }

    /// Where line `line` ends and the next starts (the end of the text for
    /// the last).
    fn line_end(&self, line: usize) -> Option<(usize, usize)> {
        let it = self.lines.get(line)?;
        let next = self
            .lines
            .get(line + 1)
            .map_or(self.text.len(), |next| next.start);
        Some((it.end, next))
    }

    /// Whether `entry` starts its line, rather than after a `- `.
    fn starts_line(&self, entry: &Entry) -> bool {
        self.lines
            .get(entry.first)
            .is_some_and(|line| line.start + line.indent == entry.start)
    }
}

/// Whether `rest` starts a block sequence item.
fn is_dash(rest: &str) -> bool {
    rest == "-" || rest.starts_with("- ")
}

/// `rest` without a trailing `# comment` outside quotes.
fn without_comment(rest: &str) -> &str {
    let mut quote = None;
    let mut previous = ' ';
    for (at, c) in rest.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(open), c) if c == open => quote = None,
            (None, '#') if previous == ' ' => return rest.get(..at).unwrap_or(rest),
            _ => {}
        }
        previous = c;
    }
    rest
}

/// The key a mapping entry in `rest` starts with, and where its `:` ends.
fn key_of(rest: &str) -> Option<(String, usize)> {
    let (key, end) = match rest.chars().next()? {
        '"' => {
            let mut escaped = false;
            let close = rest.char_indices().skip(1).find(|(_, c)| {
                let closes = *c == '"' && !escaped;
                escaped = *c == '\\' && !escaped;
                closes
            })?;
            let quoted = rest.get(..=close.0)?;
            (serde_json::from_str::<String>(quoted).ok()?, close.0 + 1)
        }
        '\'' => {
            let body = rest.get(1..)?;
            let mut at = 0;
            loop {
                let found = body.get(at..)?.find('\'')? + at;
                if body.get(found + 1..)?.starts_with('\'') {
                    at = found + 2;
                } else {
                    break (body.get(..found)?.replace("''", "'"), found + 2);
                }
            }
        }
        '[' | '{' | '#' | '&' | '*' | '!' | '|' | '>' | '%' | '@' | '`' | '?' => return None,
        _ => {
            let mut previous = ' ';
            let mut chars = rest.char_indices().peekable();
            loop {
                let (at, c) = chars.next()?;
                let next = chars.peek().map(|(_, c)| *c);
                if c == '#' && previous == ' ' {
                    return None;
                }
                if c == ':' && next.is_none_or(|next| next == ' ' || next == '\t') {
                    break (rest.get(..at)?.trim_end().to_owned(), at);
                }
                previous = c;
            }
        }
    };
    let after = rest.get(end..)?;
    let spaces = after.len() - after.trim_start_matches(' ').len();
    after
        .get(spaces..)?
        .starts_with(':')
        .then_some((key, end + spaces + 1))
}

/// Whether `text` can be written plain (unquoted) and read back as itself.
fn plain_ok(text: &str) -> bool {
    !text.is_empty()
        && !text.contains(['\n', '\r', '#', '\t'])
        && !text.contains(": ")
        && text.trim() == text
        && !text.starts_with([
            '-', '?', ':', ',', '[', ']', '{', '}', '&', '*', '!', '|', '>', '\'', '"', '%', '@',
            '`',
        ])
        && matches!(serde_yaml::from_str(text), Ok(serde_yaml::Value::String(back)) if back == text)
}

/// `value` as a YAML scalar or flow collection, quoted as `old` was.
fn render(value: &Value, old: &str) -> String {
    match value {
        Value::String(text) if old.starts_with('\'') && !text.contains(['\n', '\r']) => {
            format!("'{}'", text.replace('\'', "''"))
        }
        Value::String(text) if !old.starts_with('"') && plain_ok(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// A key as YAML writes it.
fn render_key(key: &str) -> String {
    if plain_ok(key) {
        key.to_owned()
    } else {
        serde_json::to_string(key).unwrap_or_default()
    }
}

/// The entry `path` names and the node holding it.
fn find<'n>(root: &'n Node, path: &[Seg]) -> Option<(&'n Node, &'n Entry)> {
    let (last, parent) = path.split_last()?;
    let holder = parent.iter().try_fold(root, |here, seg| {
        entry_of(here, seg).map(|entry| &entry.value)
    })?;
    Some((holder, entry_of(holder, last)?))
}

fn entry_of<'n>(node: &'n Node, seg: &Seg) -> Option<&'n Entry> {
    match (node, seg) {
        (Node::Map { entries, .. }, Seg::Key(key)) => {
            entries.iter().rev().find(|entry| entry.key == *key)
        }
        (Node::Seq { items }, Seg::Index(index)) => items.get(*index),
        _ => None,
    }
}

const FLOW: &str = "the value is inside a YAML flow collection or a layout this tool does not \
    edit in place; set its enclosing value whole instead";

/// The splice that sets `path` (found) to `value`.
pub fn replace(text: &str, path: &[Seg], value: &Value) -> Result<Splice, String> {
    let locator = Locator::new(text);
    let root = locator.root().ok_or(FLOW)?;
    let (_, entry) = find(&root, path).ok_or(FLOW)?;
    match &entry.value {
        Node::Leaf { span } => {
            let old = text.get(span.clone()).unwrap_or_default();
            if old.starts_with(['|', '>']) || old.starts_with(['&', '!']) {
                let (end, _) = locator.line_end(entry.last).ok_or(FLOW)?;
                return Ok(Splice::new(span.start..end, render(value, "")));
            }
            Ok(Splice::new(span.clone(), render(value, old)))
        }
        Node::Empty => Ok(Splice::new(
            entry.after..entry.after,
            format!(" {}", render(value, "")),
        )),
        Node::Map { .. } | Node::Seq { .. } => {
            let (end, _) = locator.line_end(entry.last).ok_or(FLOW)?;
            Ok(Splice::new(
                entry.after..end,
                format!(" {}", render(value, "")),
            ))
        }
    }
}

/// The splice that adds `key: value` to the mapping at `parent`, after its
/// last entry and indented as its entries are.
pub fn insert(text: &str, parent: &[Seg], key: &str, value: &Value) -> Result<Splice, String> {
    let locator = Locator::new(text);
    let root = locator.root().ok_or(FLOW)?;
    let holder = match parent.is_empty() {
        true => &root,
        false => &find(&root, parent).ok_or(FLOW)?.1.value,
    };
    let Node::Map { col, entries } = holder else {
        return Err(FLOW.to_owned());
    };
    let last = entries.last().ok_or(FLOW)?.last;
    let (end, _) = locator.line_end(last).ok_or(FLOW)?;
    let nl = newline(text);
    let unit = indent_unit(text, "  ");
    let mut lines = String::new();
    let mut indent = " ".repeat(*col);
    let (mut key, mut value) = (key.to_owned(), value);
    // A new object's keys go one per line, nested; anything else inline.
    loop {
        lines.push_str(&format!("{nl}{indent}{}:", render_key(&key)));
        match value {
            Value::Object(map) if map.len() == 1 => {
                let Some((inner_key, inner)) = map.iter().next() else {
                    break;
                };
                indent.push_str(&unit);
                key = inner_key.clone();
                value = inner;
            }
            other => {
                lines.push(' ');
                lines.push_str(&render(other, ""));
                break;
            }
        }
    }
    Ok(Splice::new(end..end, lines))
}

/// The splice that removes the entry or item `path` names, with its lines.
pub fn remove(text: &str, path: &[Seg]) -> Result<Splice, String> {
    let locator = Locator::new(text);
    let root = locator.root().ok_or(FLOW)?;
    let (holder, entry) = find(&root, path).ok_or(FLOW)?;
    let (_, after_last) = locator.line_end(entry.last).ok_or(FLOW)?;
    if locator.starts_line(entry) {
        let first = locator.lines.get(entry.first).ok_or(FLOW)?.start;
        return Ok(Splice::new(first..after_last, String::new()));
    }
    // The first entry of a mapping written after a `- `: up to the next
    // entry, which takes its place after the dash.
    let Node::Map { entries, .. } = holder else {
        return Err(FLOW.to_owned());
    };
    let next = entries
        .iter()
        .find(|other| other.first > entry.last)
        .map(|other| other.start);
    match next {
        Some(next) => Ok(Splice::new(entry.start..next, String::new())),
        None => {
            let (end, _) = locator.line_end(entry.last).ok_or(FLOW)?;
            Ok(Splice::new(entry.start..end, "{}".to_owned()))
        }
    }
}
