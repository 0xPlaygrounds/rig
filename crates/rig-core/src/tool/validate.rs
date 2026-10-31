//! Whether a value fits a JSON schema, and where it does not: the check a
//! tool call's arguments pass before the tool runs
//! ([`arguments_refusal`](crate::tool::feedback::arguments_refusal)).
//!
//! ```
//! use rig_core::tool::validate::{Problem, validate};
//! use serde_json::json;
//!
//! let schema = json!({
//!     "type": "object",
//!     "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
//!     "required": ["path"],
//! });
//! assert_eq!(validate(&schema, &json!({"path": "a.rs", "limit": 3})), Ok(()));
//! let problems = validate(&schema, &json!({"limit": "3"})).unwrap_err();
//! assert_eq!(
//!     problems.iter().map(Problem::to_string).collect::<Vec<_>>(),
//!     ["/limit: expected an integer, got a string", "/path: missing"],
//! );
//! ```
//!
//! It checks a subset of JSON Schema, in plain `serde_json`:
//! - `type` (a name or a list; `integer` is a whole number a 64-bit
//!   integer holds, `30.0` too, as [`decode_args`](crate::tool::decode_args)
//!   decodes it),
//!   `enum` and `const`;
//! - `properties`, `required` and `additionalProperties` (`false` or a
//!   schema);
//! - `items` (a schema, or a list by position as in draft-07),
//!   `prefixItems`, `minItems` and `maxItems`;
//! - `minLength` and `maxLength`, in characters;
//! - `minimum`, `maximum`, `exclusiveMinimum` and `exclusiveMaximum`;
//! - `allOf`, `anyOf` and `oneOf` (exactly one);
//! - `$ref` to any local JSON pointer (`#`, `#/$defs/…`, draft-07's
//!   `#/definitions/…`), resolved against the root schema. A remote or
//!   unresolvable `$ref`, or one that loops back without reaching a value,
//!   **accepts** the value there.
//!
//! A `null` for a declared property the object does not require counts as
//! absent, as serde reads an `Option` field:
//! [`args_schema`](crate::tool::args_schema) declares an `Option` optional,
//! not nullable, and a model may still send `null` for it.
//!
//! Every other keyword is ignored, so it accepts: `pattern`, `format`,
//! `patternProperties` (and so `additionalProperties` beside it, since
//! which keys a pattern allows is not checked), and any keyword not listed.

use std::fmt;

use serde_json::{Map, Number, Value};

use crate::tool::decode::whole;

/// Where a value does not fit a schema, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// A JSON pointer to the value, such as `/edits/0/old_text`; empty for
    /// the whole value.
    pub at: String,
    /// What is wrong there, such as `expected a string, got a number`.
    pub what: String,
}

impl fmt::Display for Problem {
    /// `at: what`, or `what` alone for the whole value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.at.is_empty() {
            f.write_str(&self.what)
        } else {
            write!(f, "{}: {}", self.at, self.what)
        }
    }
}

/// Checks `value` against `schema`: `Ok` when it fits; otherwise every
/// problem, in document order. The [module docs](self) list what is
/// checked.
pub fn validate(schema: &Value, value: &Value) -> Result<(), Vec<Problem>> {
    let problems = Validator { root: schema }.check(schema, value, "", &[]);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}

/// Checks values against subschemas of `root`, which `$ref`s resolve in.
struct Validator<'s> {
    root: &'s Value,
}

impl<'s> Validator<'s> {
    /// The problems of `value`, at pointer `at`, against `schema`. `refs`
    /// are the references already followed at this value, so a loop of
    /// them ends.
    fn check(&self, schema: &'s Value, value: &Value, at: &str, refs: &[&'s str]) -> Vec<Problem> {
        let schema = match schema {
            Value::Object(schema) => schema,
            Value::Bool(false) => return vec![problem(at, "no value is allowed here")],
            _ => return Vec::new(),
        };
        let mut problems = Vec::new();
        if let Some(Value::String(reference)) = schema.get("$ref")
            && !refs.contains(&reference.as_str())
            && let Some(target) = self.resolve(reference)
        {
            let mut followed = refs.to_vec();
            followed.push(reference);
            problems.extend(self.check(target, value, at, &followed));
        }
        if let Some(types) = schema.get("type")
            && !fits_type(types, value)
        {
            problems.push(problem(
                at,
                format!(
                    "expected {}, got {}",
                    expected(types),
                    got(names_integer(types), value)
                ),
            ));
            return problems;
        }
        if let Some(Value::Array(allowed)) = schema.get("enum")
            && !allowed.iter().any(|it| same(it, value))
        {
            let allowed: Vec<String> = allowed.iter().map(Value::to_string).collect();
            let what = format!(
                "expected one of {}, got {}",
                allowed.join(", "),
                shown(value)
            );
            problems.push(problem(at, what));
        }
        if let Some(constant) = schema.get("const")
            && !same(constant, value)
        {
            problems.push(problem(
                at,
                format!("expected {constant}, got {}", shown(value)),
            ));
        }
        match value {
            Value::Object(fields) => problems.extend(self.object(schema, fields, at)),
            Value::Array(items) => problems.extend(self.array(schema, items, at)),
            Value::String(text) => {
                let length = text.chars().count();
                problems.extend(count(
                    schema,
                    "minLength",
                    "maxLength",
                    "character",
                    length,
                    at,
                ));
            }
            Value::Number(number) => problems.extend(bounds(schema, number, at)),
            Value::Null | Value::Bool(_) => {}
        }
        if let Some(Value::Array(all)) = schema.get("allOf") {
            for alternative in all {
                problems.extend(self.check(alternative, value, at, refs));
            }
        }
        if let Some(Value::Array(any)) = schema.get("anyOf") {
            let results: Vec<_> = any
                .iter()
                .map(|it| self.check(it, value, at, refs))
                .collect();
            if !results.iter().any(Vec::is_empty) {
                problems.extend(closest(any, results, value, at, "anyOf"));
            }
        }
        if let Some(Value::Array(one)) = schema.get("oneOf") {
            let results: Vec<_> = one
                .iter()
                .map(|it| self.check(it, value, at, refs))
                .collect();
            match results.iter().filter(|it| it.is_empty()).count() {
                0 => problems.extend(closest(one, results, value, at, "oneOf")),
                1 => {}
                fitting => problems.push(problem(
                    at,
                    format!("fits {fitting} of its `oneOf` alternatives, not exactly one"),
                )),
            }
        }
        problems
    }

    /// The problems of an object's fields: each against its property's
    /// schema or `additionalProperties`, then each required one missing. A
    /// `null` for a declared property it does not require counts as absent,
    /// as serde reads an `Option`.
    fn object(
        &self,
        schema: &'s Map<String, Value>,
        fields: &Map<String, Value>,
        at: &str,
    ) -> Vec<Problem> {
        let mut problems = Vec::new();
        let properties = schema.get("properties").and_then(Value::as_object);
        let additional = schema
            .get("additionalProperties")
            .filter(|_| !schema.contains_key("patternProperties"));
        let required: Vec<&str> = match schema.get("required") {
            Some(Value::Array(required)) => required.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        for (key, field) in fields {
            let at = child(at, key);
            match (
                properties.and_then(|declared| declared.get(key)),
                additional,
            ) {
                (Some(_), _) if field.is_null() && !required.contains(&key.as_str()) => {}
                (Some(property), _) => problems.extend(self.check(property, field, &at, &[])),
                (None, Some(Value::Bool(false))) => {
                    let allowed: Vec<String> = properties
                        .into_iter()
                        .flat_map(Map::keys)
                        .map(|name| format!("`{name}`"))
                        .collect();
                    let what = if allowed.is_empty() {
                        "not allowed: no keys are".to_owned()
                    } else {
                        format!("not allowed; the keys are {}", allowed.join(", "))
                    };
                    problems.push(problem(&at, what));
                }
                (None, Some(additional)) => {
                    problems.extend(self.check(additional, field, &at, &[]))
                }
                (None, None) => {}
            }
        }
        for name in required {
            if !fields.contains_key(name) {
                problems.push(problem(&child(at, name), "missing"));
            }
        }
        problems
    }

    /// The problems of an array's items, by position (`prefixItems`, or a
    /// draft-07 `items` list) and then each against `items`, and of its
    /// length.
    fn array(&self, schema: &'s Map<String, Value>, items: &[Value], at: &str) -> Vec<Problem> {
        let mut problems = count(schema, "minItems", "maxItems", "item", items.len(), at);
        let (prefix, rest) = match (schema.get("prefixItems"), schema.get("items")) {
            (Some(Value::Array(prefix)), rest) => (prefix.as_slice(), rest),
            (_, Some(Value::Array(prefix))) => (prefix.as_slice(), None),
            (_, rest) => (&[][..], rest),
        };
        for (index, item) in items.iter().enumerate() {
            let subschema = prefix
                .get(index)
                .or_else(|| rest.filter(|_| index >= prefix.len()));
            if let Some(subschema) = subschema {
                problems.extend(self.check(subschema, item, &child(at, &index.to_string()), &[]));
            }
        }
        problems
    }

    /// The subschema a local `$ref` (`#` or `#/a/b`) points to.
    fn resolve(&self, reference: &str) -> Option<&'s Value> {
        let pointer = reference.strip_prefix('#')?;
        if pointer.is_empty() {
            Some(self.root)
        } else if pointer.starts_with('/') {
            self.root.pointer(pointer)
        } else {
            None
        }
    }
}

/// When no alternative of `anyOf`/`oneOf` fits: the types they expect, if
/// none is of `value`'s type; else the problems of the alternative that
/// failed only inside the value and had the fewest; else that none fits.
fn closest(
    alternatives: &[Value],
    results: Vec<Vec<Problem>>,
    value: &Value,
    at: &str,
    keyword: &str,
) -> Vec<Problem> {
    let types: Option<Vec<&Value>> = alternatives
        .iter()
        .map(|alternative| {
            alternative
                .get("type")
                .filter(|types| !fits_type(types, value))
        })
        .collect();
    if let Some(types) = types.filter(|types| !types.is_empty()) {
        let integer = types.iter().any(|types| names_integer(types));
        let names: Vec<String> = types.into_iter().map(expected).collect();
        return vec![problem(
            at,
            format!(
                "expected {}, got {}",
                names.join(" or "),
                got(integer, value)
            ),
        )];
    }
    results
        .into_iter()
        .filter(|problems| problems.iter().all(|it| it.at.len() > at.len()))
        .min_by_key(Vec::len)
        .unwrap_or_else(|| {
            vec![problem(
                at,
                format!("fits none of its `{keyword}` alternatives"),
            )]
        })
}

/// The problems of a length, in `unit`s, against `schema`'s `min` and
/// `max` keywords.
fn count(
    schema: &Map<String, Value>,
    min: &str,
    max: &str,
    unit: &str,
    length: usize,
    at: &str,
) -> Vec<Problem> {
    let limit = |key| schema.get(key).and_then(Value::as_u64);
    let plural = |n: u64| {
        if n == 1 {
            format!("{n} {unit}")
        } else {
            format!("{n} {unit}s")
        }
    };
    let length = u64::try_from(length).unwrap_or(u64::MAX);
    let mut problems = Vec::new();
    if let Some(min) = limit(min).filter(|min| length < *min) {
        problems.push(problem(
            at,
            format!("expected at least {}, got {length}", plural(min)),
        ));
    }
    if let Some(max) = limit(max).filter(|max| length > *max) {
        problems.push(problem(
            at,
            format!("expected at most {}, got {length}", plural(max)),
        ));
    }
    problems
}

/// The problems of a number against `schema`'s bounds.
fn bounds(schema: &Map<String, Value>, number: &Number, at: &str) -> Vec<Problem> {
    let Some(n) = number.as_f64() else {
        return Vec::new();
    };
    let checks: [(&str, &str, fn(f64, f64) -> bool); 4] = [
        ("minimum", "at least", |n, bound| n >= bound),
        ("maximum", "at most", |n, bound| n <= bound),
        ("exclusiveMinimum", "more than", |n, bound| n > bound),
        ("exclusiveMaximum", "less than", |n, bound| n < bound),
    ];
    checks
        .into_iter()
        .filter_map(|(key, words, fits)| {
            let bound = schema.get(key).filter(|bound| bound.is_number())?;
            let fit = bound.as_f64().is_none_or(|bound| fits(n, bound));
            (!fit).then(|| problem(at, format!("expected {words} {bound}, got {number}")))
        })
        .collect()
}

/// Whether `value` is of the type `types` names, or one of those it lists.
/// An unknown type name accepts.
fn fits_type(types: &Value, value: &Value) -> bool {
    let is = |name: &str| match name {
        "string" => value.is_string(),
        "integer" => value.as_number().is_some_and(integral),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    };
    match types {
        Value::String(name) => is(name),
        Value::Array(names) => names.iter().filter_map(Value::as_str).any(is),
        _ => true,
    }
}

/// The type `types` names, or those it lists, as words: `a string or null`.
fn expected(types: &Value) -> String {
    let word = |name: &str| match name {
        "string" | "number" | "boolean" => format!("a {name}"),
        "integer" | "array" | "object" => format!("an {name}"),
        _ => name.to_owned(),
    };
    match types {
        Value::String(name) => word(name),
        Value::Array(names) => {
            let words: Vec<String> = names.iter().filter_map(Value::as_str).map(word).collect();
            words.join(" or ")
        }
        other => other.to_string(),
    }
}

/// Whether `types` (a name or a list of them) names `integer`.
fn names_integer(types: &Value) -> bool {
    match types {
        Value::String(name) => name == "integer",
        Value::Array(names) => names.iter().any(|name| name == "integer"),
        _ => false,
    }
}

/// [`kind`] for a value where an `integer` was or was not expected: a
/// whole number out of its range says so only where one was.
fn got(integer: bool, value: &Value) -> &'static str {
    match value {
        Value::Number(number)
            if integer
                && !integral(number)
                && number.as_f64().is_some_and(|n| n.fract() == 0.0) =>
        {
            "a number out of an integer's range"
        }
        value => kind(value),
    }
}

/// `value`'s type as words, for a problem's "got".
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(number) if number.as_f64().is_some_and(|n| n.fract() == 0.0) => "a number",
        Value::Number(_) => "a number with a fraction",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// `value` itself when it is short to quote, else its type.
fn shown(value: &Value) -> String {
    match value {
        Value::Array(_) | Value::Object(_) => kind(value).to_owned(),
        Value::String(text) if text.chars().count() > 40 => kind(value).to_owned(),
        scalar => scalar.to_string(),
    }
}

/// Whether a number is an integer ([`whole`]): `1` and `1.0` are, `1.5`
/// and `1e30` are not.
fn integral(number: &Number) -> bool {
    whole(number).is_some()
}

/// Equality as JSON Schema means it, where `1` and `1.0` are the same
/// number.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            a == b || ((a.is_f64() || b.is_f64()) && a.as_f64() == b.as_f64())
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| same(a, b)))
        }
        (a, b) => a == b,
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
