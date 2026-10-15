//! [`Schema`]: a JSON Schema checked to be well formed where it is built, the
//! type of a tool's parameters and of its output.
//!
//! Derive one from a type, so the schema the model sees and the type a value
//! is parsed into cannot disagree, or check one written by hand or read from
//! elsewhere:
//!
//! ```
//! use rig_core::tool::Schema;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct Args {
//!     /// How many lines to read.
//!     limit: Option<u32>,
//! }
//!
//! let schema = Schema::arguments::<Args>();
//! let limit = serde_json::json!({"type": "integer", "minimum": 0,
//!     "description": "How many lines to read."});
//! assert_eq!(schema["properties"]["limit"], limit);
//! assert_eq!(schema["additionalProperties"], false);
//!
//! let typo = Schema::new(serde_json::json!({"type": "objekt"}));
//! assert_eq!(typo.unwrap_err().to_string(),
//!     "not a valid JSON Schema: /type: unknown type `objekt`");
//! ```

use std::fmt;
use std::ops::Deref;

use schemars::generate::SchemaSettings;
use schemars::transform::RecursiveTransform;
use schemars::{JsonSchema, generate::SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::validate::Problem;

/// A JSON Schema known to be well formed: an object or a boolean whose
/// keywords have the shapes JSON Schema gives them, at every depth.
///
/// It reads as the JSON value it is ([`Deref`] to [`Value`]) and serializes
/// as that value; deserializing checks it. Keywords it does not know, such as
/// a provider's extensions, are kept and not checked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(into = "Value", try_from = "Value")]
pub struct Schema(Value);

impl Schema {
    /// Checks `value` as a JSON Schema.
    ///
    /// # Errors
    ///
    /// [`SchemaError`] with every malformed keyword, when `value` is not an
    /// object or a boolean, or a keyword it knows has the wrong shape, such
    /// as an unknown `type` name or a `required` that is not a list of
    /// names.
    pub fn new(value: Value) -> Result<Self, SchemaError> {
        let mut problems = Vec::new();
        check(&value, "", &mut problems);
        if problems.is_empty() {
            Ok(Self(value))
        } else {
            Err(SchemaError { problems })
        }
    }

    /// The schema of an arguments type `A`, in the plain shape tool
    /// parameters use: inline, without `$schema`, `title`, `default`,
    /// `format` (providers support it unevenly) or the type's own doc
    /// comment; with `Option` fields optional, not nullable (no `null` type,
    /// enum value or `anyOf` alternative); with each object refusing keys it
    /// does not declare (`additionalProperties: false`, as
    /// `deny_unknown_fields` says; a map keeps its own); and with `required`
    /// listed on a root object even when empty.
    pub fn arguments<A: JsonSchema>() -> Self {
        let settings = SchemaSettings::draft2020_12().with_transform(RecursiveTransform(plain));
        let mut schema = generate::<A>(settings);
        if let Some(root) = schema.as_object_mut() {
            root.shift_remove("description");
            if root.get("type").is_some_and(|kind| kind == "object") {
                root.entry("required")
                    .or_insert_with(|| Value::Array(Vec::new()));
            }
        }
        Self(schema.to_value())
    }

    /// The schema of a type `T` as it serializes, such as a tool's output:
    /// inline and without `$schema`, keeping `null` for an `Option`.
    pub fn of<T: JsonSchema>() -> Self {
        Self(generate::<T>(SchemaSettings::draft2020_12()).to_value())
    }

    /// The schema of any JSON object: `{"type": "object"}`.
    pub fn object() -> Self {
        let mut object = Map::new();
        object.insert("type".to_owned(), Value::String("object".to_owned()));
        Self(Value::Object(object))
    }

    /// The schema as its JSON value.
    pub fn into_value(self) -> Value {
        self.0
    }
}

/// The root schema `settings` generate for `T`, inline and without
/// `$schema`.
fn generate<T: JsonSchema>(mut settings: SchemaSettings) -> schemars::Schema {
    settings.inline_subschemas = true;
    settings.meta_schema = None;
    SchemaGenerator::new(settings).into_root_schema_for::<T>()
}

impl Deref for Schema {
    type Target = Value;

    fn deref(&self) -> &Value {
        &self.0
    }
}

impl fmt::Display for Schema {
    /// The schema as JSON text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Schema> for Value {
    fn from(schema: Schema) -> Self {
        schema.0
    }
}

impl From<Schema> for schemars::Schema {
    fn from(schema: Schema) -> Self {
        match schema.0 {
            Value::Object(object) => object.into(),
            Value::Bool(boolean) => boolean.into(),
            // A `Schema` is an object or a boolean; this arm is never taken.
            _ => true.into(),
        }
    }
}

impl TryFrom<Value> for Schema {
    type Error = SchemaError;

    fn try_from(value: Value) -> Result<Self, SchemaError> {
        Self::new(value)
    }
}

impl PartialEq<Value> for Schema {
    fn eq(&self, other: &Value) -> bool {
        self.0 == *other
    }
}

/// Why a value is not a well-formed JSON Schema: each malformed keyword, at
/// a JSON pointer into the schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaError {
    problems: Vec<Problem>,
}

impl SchemaError {
    /// Each malformed keyword, in document order.
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a valid JSON Schema: ")?;
        let problems: Vec<String> = self.problems.iter().map(Problem::to_string).collect();
        f.write_str(&problems.join("; "))
    }
}

impl std::error::Error for SchemaError {}

/// Keywords whose value is a schema.
const SCHEMAS: [&str; 11] = [
    "not",
    "additionalProperties",
    "additionalItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
    "contentSchema",
];

/// Keywords whose value is an object of schemas.
const SCHEMA_MAPS: [&str; 5] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

/// Keywords whose value is a non-empty list of schemas.
const SCHEMA_LISTS: [&str; 4] = ["allOf", "anyOf", "oneOf", "prefixItems"];

/// Keywords whose value is a non-negative integer.
const COUNTS: [&str; 8] = [
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
    "minContains",
    "maxContains",
];

/// Keywords whose value is a string.
const STRINGS: [&str; 13] = [
    "$ref",
    "$id",
    "$schema",
    "$anchor",
    "$dynamicRef",
    "$dynamicAnchor",
    "$comment",
    "title",
    "description",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
];

/// Keywords whose value is a boolean.
const BOOLEANS: [&str; 4] = ["uniqueItems", "readOnly", "writeOnly", "deprecated"];

/// The type names JSON Schema defines.
const TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "object", "array", "null",
];

/// Adds the problems of `schema`, at pointer `at` into the root, to
/// `problems`.
fn check(schema: &Value, at: &str, problems: &mut Vec<Problem>) {
    let schema = match schema {
        Value::Object(schema) => schema,
        Value::Bool(_) => return,
        other => {
            problems.push(problem(
                at,
                format!(
                    "expected a schema (an object or a boolean), got {}",
                    kind(other)
                ),
            ));
            return;
        }
    };
    for (key, value) in schema {
        let at = child(at, key);
        let key = key.as_str();
        if SCHEMAS.contains(&key) {
            check(value, &at, problems);
        } else if SCHEMA_MAPS.contains(&key) {
            match value {
                Value::Object(map) => {
                    for (name, subschema) in map {
                        check(subschema, &child(&at, name), problems);
                    }
                }
                other => problems.push(expected(&at, "an object of schemas", other)),
            }
        } else if SCHEMA_LISTS.contains(&key) {
            schema_list(value, &at, problems);
        } else if key == "items" {
            // A list is draft-07's items by position.
            match value {
                Value::Array(_) => schema_list(value, &at, problems),
                single => check(single, &at, problems),
            }
        } else if key == "type" {
            type_names(value, &at, problems);
        } else if key == "required" {
            let names = value
                .as_array()
                .filter(|names| names.iter().all(Value::is_string));
            if names.is_none() {
                problems.push(expected(&at, "a list of property names", value));
            }
        } else if key == "enum" || key == "examples" {
            if !value.is_array() {
                problems.push(expected(&at, "a list", value));
            }
        } else if COUNTS.contains(&key) {
            if value.as_u64().is_none() {
                problems.push(expected(&at, "a non-negative integer", value));
            }
        } else if key == "multipleOf" {
            if !value.as_f64().is_some_and(|n| n > 0.0) {
                problems.push(expected(&at, "a number above zero", value));
            }
        } else if key == "minimum" || key == "maximum" {
            if !value.is_number() {
                problems.push(expected(&at, "a number", value));
            }
        } else if key == "exclusiveMinimum" || key == "exclusiveMaximum" {
            // Draft-04 spells these as booleans beside `minimum`/`maximum`.
            if !value.is_number() && !value.is_boolean() {
                problems.push(expected(&at, "a number", value));
            }
        } else if STRINGS.contains(&key) {
            if !value.is_string() {
                problems.push(expected(&at, "a string", value));
            }
        } else if BOOLEANS.contains(&key) && !value.is_boolean() {
            problems.push(expected(&at, "a boolean", value));
        }
    }
}

/// Adds the problems of a non-empty list of schemas.
fn schema_list(value: &Value, at: &str, problems: &mut Vec<Problem>) {
    match value {
        Value::Array(list) if !list.is_empty() => {
            for (index, subschema) in list.iter().enumerate() {
                check(subschema, &child(at, &index.to_string()), problems);
            }
        }
        other => problems.push(expected(at, "a non-empty list of schemas", other)),
    }
}

/// Adds the problems of a `type`: one name, or a non-empty list of distinct
/// names.
fn type_names(value: &Value, at: &str, problems: &mut Vec<Problem>) {
    let mut name = |name: &Value, at: &str| match name.as_str() {
        Some(known) if TYPES.contains(&known) => {}
        Some(unknown) => problems.push(problem(at, format!("unknown type `{unknown}`"))),
        None => problems.push(expected(at, "a type name", name)),
    };
    match value {
        Value::Array(names) if !names.is_empty() => {
            for (index, each) in names.iter().enumerate() {
                name(each, &child(at, &index.to_string()));
            }
            let distinct = names
                .iter()
                .enumerate()
                .all(|(index, each)| !names.iter().skip(index + 1).any(|other| other == each));
            if !distinct {
                problems.push(problem(at, "a type is listed twice"));
            }
        }
        Value::Array(_) => problems.push(problem(at, "expected at least one type name")),
        single => name(single, at),
    }
}

/// Drops from one subschema what [`Schema::arguments`] leaves out, keeping
/// the other keys' order, closes an object to undeclared keys, joins a doc
/// comment's lines as rustdoc does, and makes an `Option` optional rather
/// than nullable: no `null` type or enum value, and no `{"type": "null"}`
/// alternative (the one left is merged in).
fn plain(schema: &mut schemars::Schema) {
    let Some(schema) = schema.as_object_mut() else {
        return;
    };
    if let Some(Value::Array(alternatives)) = schema.get_mut("anyOf") {
        let null = serde_json::json!({"type": "null"});
        let before = alternatives.len();
        alternatives.retain(|alternative| *alternative != null);
        if let [Value::Object(only)] = alternatives.as_slice()
            && before > 1
        {
            let only = only.clone();
            schema.shift_remove("anyOf");
            for (key, value) in only {
                schema.entry(key).or_insert(value);
            }
        }
    }
    if let Some(Value::String(text)) = schema.get_mut("description") {
        let paragraphs: Vec<String> = text.split("\n\n").map(|p| p.replace('\n', " ")).collect();
        *text = paragraphs.join("\n\n");
    }
    for key in ["title", "default", "format"] {
        schema.shift_remove(key);
    }
    if schema.contains_key("properties") && !schema.contains_key("additionalProperties") {
        schema.insert("additionalProperties".to_owned(), Value::Bool(false));
    }
    if let Some(Value::Array(types)) = schema.get_mut("type")
        && types.iter().any(|kind| kind == "null")
    {
        types.retain(|kind| kind != "null");
        if let [kind] = types.as_slice() {
            let kind = kind.clone();
            schema.insert("type".to_owned(), kind);
        }
        if let Some(Value::Array(values)) = schema.get_mut("enum") {
            values.retain(|value| !value.is_null());
        }
    }
}

fn expected(at: &str, what: &str, got: &Value) -> Problem {
    problem(at, format!("expected {what}, got {}", kind(got)))
}

/// `value`'s type as words.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// The pointer to `key` inside the value at `at`.
fn child(at: &str, key: &str) -> String {
    format!("{at}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn problem(at: &str, what: impl Into<String>) -> Problem {
    Problem {
        at: at.to_owned(),
        what: what.into(),
    }
}

#[cfg(test)]
mod tests;
