//! Key paths (`dependencies.serde.version`, `jobs.test.steps[2].run`,
//! `"a.b".c`) and what they name in a document read as JSON.

use std::fmt;

use serde_json::Value;

/// One step of a key path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seg {
    /// A key of a table, object or mapping.
    Key(String),
    /// An element of an array or sequence, from 0.
    Index(usize),
}

impl fmt::Display for Seg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Key(key) if key.is_empty() || key.contains(['.', '[', ']', '"']) => {
                write!(f, "{}", serde_json::to_string(key).unwrap_or_default())
            }
            Self::Key(key) => f.write_str(key),
            Self::Index(index) => write!(f, "[{index}]"),
        }
    }
}

/// `path` written back as a key path.
pub fn show(path: &[Seg]) -> String {
    let mut out = String::new();
    for seg in path {
        if matches!(seg, Seg::Key(_)) && !out.is_empty() {
            out.push('.');
        }
        out.push_str(&seg.to_string());
    }
    out
}

/// Parses a dotted key path: keys separated by `.`, a key in double quotes
/// when it holds a dot (`"a.b"`, with `\"` for a quote), each followed by
/// any number of `[n]` indices.
///
/// # Errors
///
/// An empty path or key, an unclosed quote or bracket, or an index that is
/// not a number.
pub fn parse(key: &str) -> Result<Vec<Seg>, String> {
    let mut path = Vec::new();
    let mut chars = key.chars().peekable();
    let mut expect_key = true;
    while let Some(&next) = chars.peek() {
        match next {
            '[' => {
                chars.next();
                let digits: String = chars.by_ref().take_while(|c| *c != ']').collect();
                let index = digits
                    .trim()
                    .parse()
                    .map_err(|_| format!("`[{digits}]` in `{key}` is not an array index"))?;
                path.push(Seg::Index(index));
                expect_key = false;
            }
            '.' if !expect_key => {
                chars.next();
                expect_key = true;
            }
            '"' if expect_key => {
                chars.next();
                let mut name = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    match c {
                        '\\' => name.extend(chars.next()),
                        '"' => {
                            closed = true;
                            break;
                        }
                        c => name.push(c),
                    }
                }
                if !closed {
                    return Err(format!("`{key}` has an unclosed quote"));
                }
                path.push(Seg::Key(name));
                expect_key = false;
            }
            _ if expect_key => {
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '.' || c == '[' {
                        break;
                    }
                    name.push(c);
                    chars.next();
                }
                if name.is_empty() {
                    return Err(format!("`{key}` has an empty key"));
                }
                path.push(Seg::Key(name));
                expect_key = false;
            }
            other => return Err(format!("`{key}`: unexpected `{other}` after a key")),
        }
    }
    if expect_key {
        return Err(format!("`{key}` is empty or ends in `.`"));
    }
    Ok(path)
}

/// How far a path reaches into a document.
#[derive(Debug)]
pub enum Reach<'v> {
    /// The whole path names this value.
    Found(&'v Value),
    /// Its first `at` steps exist, and step `at` is a key missing from the
    /// object they name.
    Missing {
        /// How many steps exist.
        at: usize,
    },
}

/// What kind of value `value` is, for an error.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "a table/object",
    }
}

/// Follows `path` through `doc`.
///
/// # Errors
///
/// A step into a scalar or of the wrong kind (a key into an array, an
/// index into an object) is a type error; an index past the end of an
/// array is an error too, since arrays are not grown.
pub fn reach<'v>(doc: &'v Value, path: &[Seg]) -> Result<Reach<'v>, String> {
    let mut here = doc;
    for (at, seg) in path.iter().enumerate() {
        let above = path.get(..at).map(show).unwrap_or_default();
        let named = if above.is_empty() {
            "the document".to_owned()
        } else {
            format!("`{above}`")
        };
        here = match (seg, here) {
            (Seg::Key(key), Value::Object(map)) => match map.get(key) {
                Some(next) => next,
                None => return Ok(Reach::Missing { at }),
            },
            (Seg::Index(index), Value::Array(items)) => match items.get(*index) {
                Some(next) => next,
                None => {
                    return Err(format!(
                        "{named} has {} elements, so there is no [{index}]; set the whole array \
                         to add elements",
                        items.len()
                    ));
                }
            },
            (Seg::Key(key), other) => {
                return Err(format!(
                    "type error: {named} is {}, not a table/object, so it has no key `{key}`",
                    kind(other)
                ));
            }
            (Seg::Index(index), other) => {
                return Err(format!(
                    "type error: {named} is {}, not an array, so it has no [{index}]",
                    kind(other)
                ));
            }
        };
    }
    Ok(Reach::Found(here))
}

/// `value` under the keys of `path` (from the innermost out), for a key
/// created with its missing tables/objects.
pub fn nest(path: &[Seg], value: Value) -> Value {
    path.iter().rev().fold(value, |inner, seg| match seg {
        Seg::Key(key) => Value::Object([(key.clone(), inner)].into_iter().collect()),
        Seg::Index(_) => inner,
    })
}

/// The 1-based line and column of byte `offset` in `text`.
pub fn line_column(text: &str, offset: usize) -> (usize, usize) {
    let before = text.get(..offset).unwrap_or(text);
    let line = before.matches('\n').count() + 1;
    let column = before
        .rsplit('\n')
        .next()
        .unwrap_or_default()
        .chars()
        .count()
        + 1;
    (line, column)
}
