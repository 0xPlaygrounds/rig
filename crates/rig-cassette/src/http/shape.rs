//! Shape-matched replay: a coarse key of a request body, so replay can serve
//! a recording for a request whose values, or the types of its content,
//! differ from what was recorded.
//!
//! `RIG_CASSETTE_MATCHING=shape` (or [`super::CassetteSpec::shape_matched`]) makes a
//! replay session compare request bodies by this key instead of exactly. The
//! method, path, query and recorded headers still match exactly. The key
//! keeps the body's field structure, the order and length of every array of
//! objects, and the values of `model`, `role`, `name` and `stream`. It erases
//! every other scalar and its type, collapses an array of scalars, reduces a
//! content value (`content`, `parts`, `system`, ...) to the tool names it
//! mentions, and erases tool schemas and tool arguments. A multipart body is
//! keyed by its field names, a binary or text body by its kind.
//!
//! Shape matching trusts the request snapshot to pin the bytes: with
//! `RIG_CASSETTE_SNAPSHOTS=check` a request that differs from its recording
//! replays only while the snapshot beside the fixture holds the difference.
//!
//! ```
//! use rig_cassette::http::CassetteSpec;
//! let spec = CassetteSpec::new("chat").shape_matched();
//! assert_eq!(spec.scenario(), "chat");
//! ```
#![deny(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unreachable
)]

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{Map, Value};

use super::CassetteError;

const MATCHING_ENV: &str = "RIG_CASSETTE_MATCHING";

/// Keys whose scalar value stays in the key.
const KEPT: &[&str] = &["model", "name", "role", "stream"];

/// Keys holding message content or a tool result. Their value collapses to
/// the tool names inside it, so a string and a one-part array agree.
const CONTENT: &[&str] = &[
    "content",
    "instructions",
    "output",
    "parts",
    "prompt",
    "response",
    "result",
    "system",
];

/// Keys holding a tool schema or tool arguments. Their value is erased.
const OPAQUE: &[&str] = &[
    "args",
    "arguments",
    "inputSchema",
    "input_schema",
    "parameters",
    "parametersJsonSchema",
    "responseJsonSchema",
    "responseSchema",
    "response_schema",
    "schema",
];

/// How a replay session compares a request body with a recorded one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum BodyMatching {
    /// The scrubbed bodies must be equal, as canonical JSON when both parse.
    #[default]
    Exact,
    /// The bodies' coarse shape keys must be equal.
    Shape,
}

impl BodyMatching {
    /// Read `RIG_CASSETTE_MATCHING`, ignoring case. Unset, empty, non-Unicode
    /// and `exact` are `Exact`, `shape` is `Shape`, and any other value is an
    /// error.
    pub(super) fn current(cassette_path: &Path) -> Result<Self, CassetteError> {
        Self::parse(std::env::var(MATCHING_ENV).ok(), cassette_path)
    }

    pub(super) fn parse(
        value: Option<String>,
        cassette_path: &Path,
    ) -> Result<Self, CassetteError> {
        let Some(value) = value else {
            return Ok(Self::Exact);
        };
        match value.to_ascii_lowercase().as_str() {
            "" | "exact" => Ok(Self::Exact),
            "shape" => Ok(Self::Shape),
            _ => Err(CassetteError::InvalidMatchingMode {
                path: cassette_path.to_path_buf(),
                value,
            }),
        }
    }
}

/// The coarse key of a body as [`super::snapshot::body_view`] reads it.
pub(super) fn shape_key(view: &Value) -> Value {
    if let Some(names) = multipart_names(view) {
        return Value::from(format!("multipart[{}]", names.join(",")));
    }
    match view {
        Value::Null => Value::Null,
        Value::String(_) => Value::from("text"),
        Value::Object(map) if is_opaque_view(map) => Value::from("binary"),
        value => coarse(value),
    }
}

/// The field names of a multipart view, sorted, or `None` for another view.
fn multipart_names(view: &Value) -> Option<Vec<String>> {
    let Value::Object(map) = view else {
        return None;
    };
    let Some(Value::Array(parts)) = map.get("multipart") else {
        return None;
    };
    if map.len() != 1 {
        return None;
    }
    let names: BTreeSet<String> = parts
        .iter()
        .map(|part| {
            let disposition = part.get("headers")?.get("content-disposition")?.as_str()?;
            Some(field_name(disposition).unwrap_or_default().to_owned())
        })
        .collect::<Option<_>>()?;
    Some(names.into_iter().collect())
}

/// The `name` parameter of a `content-disposition` header value.
pub(super) fn field_name(disposition: &str) -> Option<&str> {
    disposition.split(';').find_map(|parameter| {
        let (key, value) = parameter.trim().split_once('=')?;
        (key == "name").then(|| value.trim_matches('"'))
    })
}

/// The view of a body that is not UTF-8: its length and hash only.
fn is_opaque_view(map: &Map<String, Value>) -> bool {
    map.len() == 2 && map.contains_key("bytes") && map.contains_key("fnv1a64")
}

fn coarse(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), coarse_field(key, value)))
                .collect(),
        ),
        Value::Array(items) if items.iter().all(is_scalar) => Value::from("_"),
        Value::Array(items) => Value::Array(items.iter().map(coarse).collect()),
        _ => Value::from("_"),
    }
}

fn coarse_field(key: &str, value: &Value) -> Value {
    if KEPT.contains(&key) && is_scalar(value) {
        return value.clone();
    }
    if CONTENT.contains(&key) {
        let mut names = BTreeSet::new();
        tool_names(value, &mut names);
        return Value::Array(names.into_iter().map(Value::from).collect());
    }
    // `input` is a conversation in the Responses API and tool arguments
    // elsewhere; only the conversation keeps its structure.
    let conversation = matches!(value, Value::Array(items) if items.iter().any(Value::is_object));
    if OPAQUE.contains(&key) || (key == "input" && !conversation) {
        return Value::from("_");
    }
    coarse(value)
}

/// Every string `name` inside `value`: the tools a content value calls or
/// answers.
fn tool_names(value: &Value, names: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                match value {
                    Value::String(name) if key == "name" => {
                        names.insert(name.clone());
                    }
                    value => tool_names(value, names),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| tool_names(item, names)),
        _ => {}
    }
}

fn is_scalar(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}

#[cfg(test)]
mod tests;
