//! JSON files: where each value lies in the text, so a change splices only
//! its own bytes.

use std::ops::Range;

use serde_json::Value;

use crate::structured_config_edit::key::Seg;
use crate::structured_config_edit::text::{Splice, indent_of, indent_unit, newline};

/// A value of the file and where it lies.
enum Node {
    Object {
        span: Range<usize>,
        /// Each member's key, the key's span (quotes included) and value.
        members: Vec<(String, Range<usize>, Node)>,
    },
    Array {
        span: Range<usize>,
        items: Vec<Node>,
    },
    Leaf {
        span: Range<usize>,
    },
}

impl Node {
    fn span(&self) -> Range<usize> {
        match self {
            Self::Object { span, .. } | Self::Array { span, .. } | Self::Leaf { span } => {
                span.clone()
            }
        }
    }
}

/// Reads `text` as JSON.
///
/// # Errors
///
/// Where it stops parsing, by line and column.
pub fn parse(text: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|error| {
        format!(
            "does not parse as JSON at line {}, column {}: {error}",
            error.line(),
            error.column()
        )
    })
}

/// The spans of `text`, which [`parse`] accepted.
fn spans(text: &str) -> Result<Node, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
        text,
    };
    parser
        .value()
        .ok_or_else(|| "the JSON could not be located byte by byte".to_owned())
}

struct Parser<'t> {
    text: &'t str,
    bytes: &'t [u8],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_whitespace()) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Option<()> {
        self.skip_space();
        (self.peek()? == byte).then(|| self.at += 1)
    }

    /// A string's span, quotes included.
    fn string(&mut self) -> Option<Range<usize>> {
        let start = self.at;
        self.at += 1;
        loop {
            match self.peek()? {
                b'\\' => self.at += 2,
                b'"' => {
                    self.at += 1;
                    return Some(start..self.at);
                }
                _ => self.at += 1,
            }
        }
    }

    fn value(&mut self) -> Option<Node> {
        self.skip_space();
        let start = self.at;
        match self.peek()? {
            b'{' => {
                self.at += 1;
                let mut members = Vec::new();
                self.skip_space();
                if self.peek()? != b'}' {
                    loop {
                        self.skip_space();
                        let key_span = self.string()?;
                        let key = serde_json::from_str(self.text.get(key_span.clone())?).ok()?;
                        self.eat(b':')?;
                        members.push((key, key_span, self.value()?));
                        self.skip_space();
                        match self.peek()? {
                            b',' => self.at += 1,
                            _ => break,
                        }
                    }
                }
                self.eat(b'}')?;
                Some(Node::Object {
                    span: start..self.at,
                    members,
                })
            }
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.peek()? != b']' {
                    loop {
                        items.push(self.value()?);
                        self.skip_space();
                        match self.peek()? {
                            b',' => self.at += 1,
                            _ => break,
                        }
                    }
                }
                self.eat(b']')?;
                Some(Node::Array {
                    span: start..self.at,
                    items,
                })
            }
            b'"' => self.string().map(|span| Node::Leaf { span }),
            _ => {
                while self
                    .peek()
                    .is_some_and(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
                {
                    self.at += 1;
                }
                (self.at > start).then_some(Node::Leaf {
                    span: start..self.at,
                })
            }
        }
    }
}

/// The node `path` names under `node`, which [`crate::structured_config_edit::key::reach`] found.
fn find<'n>(node: &'n Node, path: &[Seg]) -> Option<&'n Node> {
    path.iter().try_fold(node, |here, seg| match (seg, here) {
        (Seg::Key(key), Node::Object { members, .. }) => members
            .iter()
            .rev()
            .find(|(name, ..)| name == key)
            .map(|(.., value)| value),
        (Seg::Index(index), Node::Array { items, .. }) => items.get(*index),
        _ => None,
    })
}

/// `value` as JSON at a line indented `indent`, pretty with `unit` when
/// `pretty`.
fn render(value: &Value, pretty: bool, unit: &str, indent: &str, nl: &str) -> String {
    if !pretty {
        return serde_json::to_string(value).unwrap_or_default();
    }
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(unit.as_bytes());
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    serde::Serialize::serialize(value, &mut serializer).ok();
    String::from_utf8_lossy(&out).replace('\n', &format!("{nl}{indent}"))
}

/// The splice that sets `path` (found) to `value`.
pub fn replace(text: &str, path: &[Seg], value: &Value) -> Result<Splice, String> {
    let root = spans(text)?;
    let node = find(&root, path).ok_or("the key was not found in the text")?;
    let span = node.span();
    let indent = indent_of(text, span.start);
    // Pretty when the old value spans lines or ends its line, as in a
    // pretty-printed file; compact inside a one-line object or array.
    let rest = text.get(span.end..).unwrap_or_default();
    let rest = rest.split('\n').next().unwrap_or_default().trim();
    let pretty = text.get(span.clone()).is_some_and(|old| old.contains('\n'))
        || rest.is_empty()
        || rest == ",";
    let new = render(
        value,
        pretty,
        &indent_unit(text, "  "),
        indent,
        newline(text),
    );
    Ok(Splice::new(span, new))
}

/// The splice that adds `key: value` to the object at `parent`.
pub fn insert(text: &str, parent: &[Seg], key: &str, value: &Value) -> Result<Splice, String> {
    let root = spans(text)?;
    let Some(Node::Object { span, members }) = find(&root, parent) else {
        return Err("the object to add to was not found in the text".to_owned());
    };
    let nl = newline(text);
    let unit = indent_unit(text, "  ");
    let quoted = serde_json::to_string(key).unwrap_or_default();
    let Some((_, last_key, last)) = members.last() else {
        let outer = indent_of(text, span.start);
        let inner = format!("{outer}{unit}");
        let rendered = render(value, true, &unit, &inner, nl);
        return Ok(Splice::new(
            span.clone(),
            format!("{{{nl}{inner}{quoted}: {rendered}{nl}{outer}}}"),
        ));
    };
    let colon = text.get(last_key.end..last.span().start).unwrap_or(": ");
    let multiline = text
        .get(span.start..last_key.start)
        .is_some_and(|before| before.contains('\n'));
    let at = last.span().end;
    if multiline {
        let indent = indent_of(text, last_key.start);
        let rendered = render(value, true, &unit, indent, nl);
        return Ok(Splice::new(
            at..at,
            format!(",{nl}{indent}{quoted}{colon}{rendered}"),
        ));
    }
    let between = match members.len().checked_sub(2).and_then(|i| members.get(i)) {
        Some((_, _, before)) => text.get(before.span().end..last_key.start).unwrap_or(", "),
        None => ", ",
    };
    let rendered = render(value, false, &unit, "", nl);
    Ok(Splice::new(
        at..at,
        format!("{between}{quoted}{colon}{rendered}"),
    ))
}

/// The splice that removes the member or element `path` names.
pub fn remove(text: &str, path: &[Seg]) -> Result<Splice, String> {
    let root = spans(text)?;
    let (parent, last) = match path.split_last() {
        Some((last, parent)) => (parent, last),
        None => return Err("the key is empty".to_owned()),
    };
    let missing = || "the key was not found in the text".to_owned();
    // Each element's start (a member's key) and end, and which one goes.
    let (span, parts, index): (Range<usize>, Vec<Range<usize>>, usize) =
        match (find(&root, parent), last) {
            (Some(Node::Object { span, members }), Seg::Key(key)) => {
                let index = members
                    .iter()
                    .rposition(|(name, ..)| name == key)
                    .ok_or_else(missing)?;
                let parts = members
                    .iter()
                    .map(|(_, key, value)| key.start..value.span().end)
                    .collect();
                (span.clone(), parts, index)
            }
            (Some(Node::Array { span, items }), Seg::Index(index)) => {
                (span.clone(), items.iter().map(Node::span).collect(), *index)
            }
            _ => return Err(missing()),
        };
    let gone = match (
        index.checked_sub(1).and_then(|i| parts.get(i)),
        parts.get(index + 1),
    ) {
        // Up to the next element, so its indentation stays.
        (_, Some(next)) => parts.get(index).map(|this| this.start..next.start),
        // From the end of the one before: its comma goes.
        (Some(before), None) => parts.get(index).map(|this| before.end..this.end),
        // The only element: the container is left empty.
        (None, None) => Some(span.start + 1..span.end - 1),
    };
    gone.map(|range| Splice::new(range, String::new()))
        .ok_or_else(missing)
}
