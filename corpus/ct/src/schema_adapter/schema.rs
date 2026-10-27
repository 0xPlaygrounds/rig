//! The schema rewrites of the profiles, on plain JSON, and the mapping of
//! a call's arguments back to the schema the tool declared.

use std::collections::BTreeMap;

use rig_harness::rig_core::json_utils::remove_preserving_order;
use rig_harness::rig_core::tool::validate::validate;
use serde_json::{Map, Value};

/// How deep `$ref`s are followed into each other before a recursive one is
/// left unconstrained.
const MAX_REF_DEPTH: usize = 6;
/// The most characters a shortened description keeps.
const MAX_DESCRIPTION: usize = 200;
/// What `strict-subset` drops wherever it appears in a schema.
const DROPPED: [&str; 5] = [
    "format",
    "pattern",
    "default",
    "examples",
    "additionalProperties",
];

/// What a rewrite changed, by kind of change, with how often.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Changes(pub BTreeMap<String, usize>);

impl Changes {
    pub(crate) fn note(&mut self, what: impl Into<String>) {
        *self.0.entry(what.into()).or_default() += 1;
    }

    /// One line of what changed, such as `inlined $ref ×2; dropped format`,
    /// or `nothing`.
    pub fn summary(&self) -> String {
        if self.0.is_empty() {
            return "nothing".to_owned();
        }
        let each = self.0.iter().map(|(what, &times)| match times {
            1 => what.clone(),
            _ => format!("{what} ×{times}"),
        });
        each.collect::<Vec<_>>().join("; ")
    }
}

/// The subschemas directly under `schema`, by the keywords that hold them.
fn subschemas_mut(schema: &mut Map<String, Value>) -> Vec<&mut Value> {
    let mut found = Vec::new();
    for (key, value) in schema.iter_mut() {
        match (key.as_str(), value) {
            (
                "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas",
                Value::Object(map),
            ) => found.extend(map.values_mut()),
            ("allOf" | "anyOf" | "oneOf" | "prefixItems" | "items", Value::Array(list)) => {
                found.extend(list.iter_mut());
            }
            (
                "items"
                | "additionalProperties"
                | "not"
                | "contains"
                | "if"
                | "then"
                | "else"
                | "propertyNames"
                | "additionalItems"
                | "unevaluatedItems"
                | "unevaluatedProperties",
                value @ Value::Object(_),
            ) => found.push(value),
            _ => {}
        }
    }
    found
}

/// Calls `edit` on `schema` and then on each subschema under it, as edited.
fn walk(schema: &mut Value, edit: &mut impl FnMut(&mut Map<String, Value>)) {
    let Value::Object(map) = schema else {
        return;
    };
    edit(map);
    for child in subschemas_mut(map) {
        walk(child, edit);
    }
}

/// `root` with every local `$ref` replaced by what it points to, and its
/// `$defs` gone. A ref that cannot be resolved, or recurses too deep,
/// accepts anything there, as rig-core's validator reads it.
pub fn inline_refs(root: &Value, changes: &mut Changes) -> Value {
    let mut inlined = root.clone();
    if let Value::Object(map) = &mut inlined {
        remove_preserving_order(map, "$defs");
        remove_preserving_order(map, "definitions");
    }
    resolve(&mut inlined, root, 0, changes);
    inlined
}

fn resolve(schema: &mut Value, root: &Value, depth: usize, changes: &mut Changes) {
    let Value::Object(map) = schema else {
        return;
    };
    if let Some(Value::String(pointer)) = remove_preserving_order(map, "$ref") {
        changes.note("inlined $ref");
        let target = pointer.strip_prefix('#').and_then(|at| root.pointer(at));
        if let (Some(Value::Object(target)), true) = (target, depth < MAX_REF_DEPTH) {
            for (key, value) in target {
                map.entry(key.clone()).or_insert_with(|| value.clone());
            }
            return resolve(schema, root, depth + 1, changes);
        }
    }
    for child in subschemas_mut(map) {
        resolve(child, root, depth, changes);
    }
}

/// `parameters` in the subset strict providers take: refs inlined, each
/// union of objects one object with every field optional and a note on
/// the shapes in its description, a union of plain types a type list, and
/// no `format`, `pattern`, `default`, `examples` or `additionalProperties`.
pub fn strict_subset(parameters: &Value, changes: &mut Changes) -> Value {
    let mut schema = inline_refs(parameters, changes);
    walk(&mut schema, &mut |map| {
        for union in ["oneOf", "anyOf"] {
            if let Some(Value::Array(alternatives)) = remove_preserving_order(map, union) {
                changes.note(format!("flattened {union}"));
                flatten(map, alternatives);
            }
        }
        for keyword in DROPPED {
            if remove_preserving_order(map, keyword).is_some() {
                changes.note(format!("dropped {keyword}"));
            }
        }
    });
    schema
}

/// Whether `schema` describes an object.
fn is_object(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("object")
        || schema.get("properties").is_some()
}

/// The names `schema` requires.
fn required(schema: &Value) -> Vec<&str> {
    let names = schema.get("required").and_then(Value::as_array);
    names
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

/// Puts the union of `alternatives` into `map` in a form with no union.
fn flatten(map: &mut Map<String, Value>, alternatives: Vec<Value>) {
    let plain_types: Option<Vec<Value>> = alternatives
        .iter()
        .map(|it| match it.as_object() {
            Some(only) if only.keys().all(|key| key == "type" || key == "description") => {
                only.get("type").cloned()
            }
            _ => None,
        })
        .collect();
    if !alternatives.is_empty() && alternatives.iter().all(is_object) {
        let mut properties = match remove_preserving_order(map, "properties") {
            Some(Value::Object(own)) => own,
            _ => Map::new(),
        };
        let mut shapes = Vec::new();
        for alternative in &alternatives {
            let fields = alternative.get("properties").and_then(Value::as_object);
            let names: Vec<&str> = fields
                .into_iter()
                .flatten()
                .map(|(n, _)| n.as_str())
                .collect();
            for (name, field) in fields.into_iter().flatten() {
                merge_field(&mut properties, name, field);
            }
            let needed = required(alternative);
            shapes.push(match needed.is_empty() {
                true => format!("({}) {}", shapes.len() + 1, names.join(", ")),
                false => format!(
                    "({}) {} (required: {})",
                    shapes.len() + 1,
                    names.join(", "),
                    needed.join(", ")
                ),
            });
        }
        map.insert("type".to_owned(), Value::from("object"));
        map.insert("properties".to_owned(), Value::Object(properties));
        add_note(
            map,
            &format!(
                "Exactly one of these shapes; send only its fields: {}.",
                shapes.join("; ")
            ),
        );
    } else if let Some(types) = plain_types.filter(|types| !types.is_empty()) {
        let mut names: Vec<Value> = Vec::new();
        for kind in types.into_iter().flat_map(|kind| match kind {
            Value::Array(kinds) => kinds,
            one => vec![one],
        }) {
            if !names.contains(&kind) {
                names.push(kind);
            }
        }
        map.insert("type".to_owned(), Value::Array(names));
    } else {
        add_note(
            map,
            &format!(
                "One of {} kinds of value; see the tool's description.",
                alternatives.len()
            ),
        );
    }
}

/// Adds the field `name` of one shape to the merged `properties`: the
/// first shape's schema, with the values of each `const`/`enum` joined.
fn merge_field(properties: &mut Map<String, Value>, name: &str, field: &Value) {
    let Some(have) = properties.get_mut(name) else {
        properties.insert(name.to_owned(), field.clone());
        return;
    };
    if have == field {
        return;
    }
    let values = |schema: &Value| -> Option<Vec<Value>> {
        schema
            .get("const")
            .map(|one| vec![one.clone()])
            .or_else(|| schema.get("enum").and_then(Value::as_array).cloned())
    };
    if let (Some(mut joined), Some(more), Value::Object(have)) =
        (values(have), values(field), &mut *have)
    {
        for value in more {
            if !joined.contains(&value) {
                joined.push(value);
            }
        }
        remove_preserving_order(have, "const");
        have.insert("enum".to_owned(), Value::Array(joined));
    }
}

/// Appends `note` to the description of `map`.
fn add_note(map: &mut Map<String, Value>, note: &str) {
    let text = match map.get("description").and_then(Value::as_str) {
        Some(own) if !own.trim().is_empty() => format!("{} {note}", own.trim()),
        _ => note.to_owned(),
    };
    map.insert("description".to_owned(), Value::from(text));
}

/// The first sentence of `text`, at most 200 characters.
pub fn first_sentence(text: &str) -> String {
    let text = text.trim();
    let ends = [". ", ".\n", "! ", "!\n", "? ", "?\n"];
    let end = ends
        .iter()
        .filter_map(|end| text.find(end).map(|at| at + 1))
        .min()
        .unwrap_or(text.len());
    let sentence = text.get(..end).unwrap_or(text);
    match sentence.char_indices().nth(MAX_DESCRIPTION - 1) {
        Some((at, _)) => format!("{}…", sentence.get(..at).unwrap_or(sentence)),
        None => sentence.to_owned(),
    }
}

/// `parameters` with each description cut to its first sentence.
pub fn short_descriptions(parameters: &Value, changes: &mut Changes) -> Value {
    let mut schema = parameters.clone();
    walk(&mut schema, &mut |map| {
        if let Some(Value::String(text)) = map.get_mut("description") {
            let short = first_sentence(text);
            if short != *text {
                changes.note("shortened description");
                *text = short;
            }
        }
    });
    schema
}

/// `arguments` sent for the flattened form of `original`, back in the
/// shape `original` declares: each flattened union's object becomes the
/// alternative its fields fit. `None` when the result still does not fit
/// `original`, so the call cannot be mapped.
pub fn map_back(original: &Value, arguments: &Value) -> Option<Value> {
    if validate(original, arguments).is_ok() {
        return Some(arguments.clone());
    }
    let inlined = inline_refs(original, &mut Changes::default());
    let mapped = back(&inlined, arguments);
    validate(original, &mapped).is_ok().then_some(mapped)
}

fn back(schema: &Value, value: &Value) -> Value {
    match value {
        Value::Object(fields) => back_object(schema, fields),
        Value::Array(items) => match schema.get("items") {
            Some(item @ Value::Object(_)) => {
                Value::Array(items.iter().map(|it| back(item, it)).collect())
            }
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

/// The schema of field `name` in `schema`'s own properties.
fn field<'a>(schema: &'a Value, name: &str) -> Option<&'a Value> {
    schema.get("properties").and_then(|it| it.get(name))
}

fn back_object(schema: &Value, fields: &Map<String, Value>) -> Value {
    let union = ["oneOf", "anyOf"]
        .iter()
        .find_map(|key| schema.get(*key).and_then(Value::as_array))
        .filter(|alternatives| alternatives.iter().all(is_object));
    let Some(alternatives) = union else {
        let mapped = fields.iter().map(|(name, value)| {
            let value = field(schema, name).map_or_else(|| value.clone(), |it| back(it, value));
            (name.clone(), value)
        });
        return Value::Object(mapped.collect());
    };
    let sent: Vec<&String> = fields
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(name, _)| name)
        .collect();
    // The alternative the sent fields fit that names the most of them.
    let mut best: Option<(usize, Value)> = None;
    for alternative in alternatives {
        let own = |name: &str| field(alternative, name).or_else(|| field(schema, name));
        if sent.iter().any(|name| own(name).is_none())
            || required(alternative)
                .iter()
                .any(|name| !sent.iter().any(|it| it == name))
        {
            continue;
        }
        let kept = sent.iter().filter_map(|name| {
            let value = fields.get(*name)?;
            Some(((*name).clone(), back(own(name)?, value)))
        });
        let candidate = Value::Object(kept.collect());
        let named = sent
            .iter()
            .filter(|name| field(alternative, name).is_some())
            .count();
        if validate(alternative, &candidate).is_ok()
            && best.as_ref().is_none_or(|(most, _)| named > *most)
        {
            best = Some((named, candidate));
        }
    }
    best.map_or_else(|| Value::Object(fields.clone()), |(_, chosen)| chosen)
}

/// Whether `glob` (`*` for any run of characters) matches all of `text`.
pub fn glob_matches(glob: &str, text: &str) -> bool {
    let mut parts = glob.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        return rest.is_empty();
    };
    for part in middle {
        let Some(at) = rest.find(part) else {
            return false;
        };
        rest = rest.get(at + part.len()..).unwrap_or_default();
    }
    rest.len() >= last.len() && rest.ends_with(last)
}
