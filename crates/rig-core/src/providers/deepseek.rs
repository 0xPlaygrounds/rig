//! DeepSeek model identifiers. [`from_env`] and [`new`] build a client on the [`DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK) dialect.
//!
//! A chat model with [strict tools](crate::providers::openai::wire::Chat::with_strict_tools)
//! uses DeepSeek's strict function calling: requests go to its `/beta`
//! endpoint, with each tool schema closed, every property required, a
//! `$ref` inside `anyOf` written out, a one-member `allOf` merged, a type
//! list naming `object` or `array` split into an `anyOf`, and a plain
//! string's `format` outside `email`, `hostname`, `ipv4`, `ipv6` and `uuid`
//! dropped. A tool whose schema strict mode cannot express, such as a map,
//! an untyped value or an object with no properties below the top, is sent
//! as written without `strict`.
//!
//! ```no_run
//! use rig_core::providers::deepseek;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut deepseek_chat = deepseek::from_env()?.chat(deepseek::DEEPSEEK_V4_FLASH);
//! deepseek_chat.wire = deepseek_chat.wire.with_strict_tools();
//! # let _ = deepseek_chat;
//! # Ok(())
//! # }
//! ```

/// `deepseek-flash` completion model. DeepSeek points this unversioned name at
/// its latest Flash model, which is DeepSeek-V4.1-Flash as of September 2026.
pub const DEEPSEEK_FLASH: &str = "deepseek-flash";
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

pub mod extension;

use serde_json::{Map, Value};

/// The string formats DeepSeek's strict mode takes on a `"string"` schema.
const STRICT_FORMATS: [&str; 5] = ["email", "hostname", "ipv4", "ipv6", "uuid"];

/// Rewrite `schema` into the one DeepSeek's strict mode takes: every
/// object closed and every property required, as OpenAI's strict mode
/// wants, but with keywords beside a `$ref` kept; a `$ref` inside `anyOf`
/// (schemars' `Option` of a struct) written out as its definition; a
/// one-member `allOf` merged into its parent; a type list naming `object`
/// or `array` (schemars' `Option<Vec<T>>`) split into an `anyOf`, as
/// strict mode takes only primitive types in a list; and a plain string's
/// `format` outside the five it names dropped.
///
/// # Errors
///
/// What strict mode cannot express: an object open to further properties
/// (a map), an object with no properties below the top, an untyped schema
/// (`true`, `{}`), an `allOf` of several schemas or one restating a keyword
/// differently, a `oneOf` or a type list naming `object` or `array` beside
/// an `anyOf`, a tuple, a type that holds itself through an `anyOf`, or a
/// `$ref` that names no definition.
pub(crate) fn strict_schema(schema: &mut Value) -> Result<(), &'static str> {
    // DeepSeek takes keywords beside a `$ref` (a field's description), which
    // OpenAI's strict mode refuses.
    let options = crate::providers::internal::schema::SanitizeOptions {
        strip_ref_siblings: false,
        inject_empty_properties: true,
        strip_numeric_constraints: false,
    };
    crate::providers::internal::schema::sanitize_schema(schema, options);
    // The shared sanitizer reaches no `$def` or `definitions`: close their
    // objects first, so the definitions written out below are closed too.
    close(schema)?;
    let definitions = schema.clone();
    strict_node(schema, &definitions, &mut Vec::new(), true)
}

/// Close every object schema at or below `node`, through every keyword
/// that holds schemas: `additionalProperties: false`, every property
/// required, and `oneOf` read as `anyOf`, as OpenAI's strict mode wants.
///
/// # Errors
///
/// A type list naming `object` or `array` beside an `anyOf` of its own, and
/// a `oneOf` beside one, which joining would loosen.
fn close(node: &mut Value) -> Result<(), &'static str> {
    let Value::Object(object) = node else {
        return Ok(());
    };
    split_type_list(object)?;
    if object.get("type").and_then(Value::as_str) == Some("object")
        || object.contains_key("properties")
    {
        let properties = object
            .entry("properties")
            .or_insert_with(|| Value::Object(Map::new()));
        let required: Vec<Value> = properties
            .as_object()
            .map(|properties| properties.keys().cloned().map(Value::String).collect())
            .unwrap_or_default();
        object.insert("required".to_owned(), Value::Array(required));
        object
            .entry("additionalProperties")
            .or_insert(Value::Bool(false));
    }
    if let Some(one_of) = object.shift_remove("oneOf") {
        if object.contains_key("anyOf") {
            return Err("a `oneOf` beside an `anyOf`");
        }
        object.insert("anyOf".to_owned(), one_of);
    }
    for key in ["properties", "$defs", "$def", "definitions"] {
        if let Some(Value::Object(children)) = object.get_mut(key) {
            children.values_mut().try_for_each(close)?;
        }
    }
    for key in ["anyOf", "allOf"] {
        if let Some(Value::Array(members)) = object.get_mut(key) {
            members.iter_mut().try_for_each(close)?;
        }
    }
    match object.get_mut("items") {
        Some(items) => close(items),
        None => Ok(()),
    }
}

/// The keywords that constrain only one type, by that type.
const TYPE_KEYWORDS: [(&str, &[&str]); 5] = [
    (
        "object",
        &[
            "properties",
            "required",
            "additionalProperties",
            "patternProperties",
            "propertyNames",
        ],
    ),
    (
        "array",
        &[
            "items",
            "prefixItems",
            "minItems",
            "maxItems",
            "uniqueItems",
            "contains",
        ],
    ),
    ("string", &["minLength", "maxLength", "pattern", "format"]),
    (
        "number",
        &[
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "format",
        ],
    ),
    (
        "integer",
        &[
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "format",
        ],
    ),
];

/// Split a type list naming `object` or `array`, which strict mode refuses
/// (it takes `["integer", "null"]` but not `["array", "null"]`), into an
/// `anyOf` of one schema per type, each with the keywords that constrain
/// it. The annotations stay on the parent, which DeepSeek takes beside
/// `anyOf`.
///
/// # Errors
///
/// When the parent already has an `anyOf`, which the split would join.
fn split_type_list(object: &mut Map<String, Value>) -> Result<(), &'static str> {
    let Some(Value::Array(types)) = object.get("type") else {
        return Ok(());
    };
    if !types
        .iter()
        .any(|kind| matches!(kind.as_str(), Some("object" | "array")))
    {
        return Ok(());
    }
    if object.contains_key("anyOf") {
        return Err("a type list naming an object or array beside an `anyOf`");
    }
    let Some(Value::Array(types)) = object.shift_remove("type") else {
        return Ok(());
    };
    let mut members: Vec<Value> = Vec::new();
    for kind in types {
        let mut member = Map::from_iter([("type".to_owned(), kind.clone())]);
        let keywords = TYPE_KEYWORDS
            .iter()
            .find(|(name, _)| kind.as_str() == Some(*name))
            .map_or(&[][..], |(_, keywords)| *keywords);
        for keyword in keywords {
            if let Some(value) = object.get(*keyword) {
                member.insert((*keyword).to_owned(), value.clone());
            }
        }
        members.push(Value::Object(member));
    }
    for (_, keywords) in TYPE_KEYWORDS {
        for keyword in keywords {
            object.shift_remove(*keyword);
        }
    }
    object.insert("anyOf".to_owned(), Value::Array(members));
    Ok(())
}

/// [`strict_schema`] on one node, resolving references in `root`, with
/// `inlining` the definitions being written out above it.
fn strict_node(
    node: &mut Value,
    root: &Value,
    inlining: &mut Vec<String>,
    top: bool,
) -> Result<(), &'static str> {
    let Value::Object(object) = node else {
        return match node {
            Value::Bool(_) => Err("an untyped schema"),
            _ => Ok(()),
        };
    };
    while let Some(all_of) = object.shift_remove("allOf") {
        let Value::Array(members) = all_of else {
            return Err("an `allOf` that is not a list");
        };
        let mut members = members.into_iter();
        match (members.next(), members.next()) {
            (Some(Value::Object(member)), None) => merge(object, member)?,
            _ => return Err("an `allOf` of several schemas"),
        }
    }
    // Live, strict mode refused a schema naming none of these.
    if !["type", "anyOf", "$ref"]
        .iter()
        .any(|key| object.contains_key(*key))
    {
        return Err("an untyped schema");
    }
    // Live, an object with no properties was taken only as the whole tool's
    // parameters.
    if !top
        && object
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(Map::is_empty)
    {
        return Err("an object with no properties below the top");
    }
    if object.contains_key("prefixItems") || object.get("items").is_some_and(Value::is_array) {
        return Err("a tuple");
    }
    if object.contains_key("patternProperties")
        || object
            .get("additionalProperties")
            .is_some_and(|open| open != &Value::Bool(false))
    {
        return Err("an object open to further properties");
    }
    // Only a plain string's format is checked: DeepSeek took `date-time` and
    // `uri` on a `["string", "null"]` type list, and numeric formats, live.
    if object.get("type").and_then(Value::as_str) == Some("string")
        && object
            .get("format")
            .and_then(Value::as_str)
            .is_some_and(|format| !STRICT_FORMATS.contains(&format))
    {
        object.shift_remove("format");
    }
    if let Some(Value::Array(members)) = object.get_mut("anyOf") {
        for member in members {
            let depth = inlining.len();
            // Written out to the end: a merged `allOf` or a definition may
            // itself be a reference.
            while let Value::Object(fields) = member {
                if let Some(all_of) = fields.shift_remove("allOf") {
                    fields.insert("allOf".to_owned(), all_of);
                    strict_node(member, root, inlining, false)?;
                    continue;
                }
                let Some(Value::String(name)) = fields.shift_remove("$ref") else {
                    break;
                };
                if inlining.contains(&name) {
                    return Err("a type that holds itself through `anyOf`");
                }
                let Some(Value::Object(definition)) = definition(root, &name) else {
                    return Err("a `$ref` that names no definition");
                };
                merge(fields, definition.clone())?;
                inlining.push(name);
            }
            strict_node(member, root, inlining, false)?;
            inlining.truncate(depth);
        }
    }
    for key in ["properties", "$defs", "$def", "definitions"] {
        if let Some(Value::Object(children)) = object.get_mut(key) {
            for child in children.values_mut() {
                strict_node(child, root, inlining, false)?;
            }
        }
    }
    if let Some(items) = object.get_mut("items") {
        strict_node(items, root, inlining, false)?;
    }
    Ok(())
}

/// The keywords that annotate a schema without constraining it, where the
/// outer schema's wins when both name one.
const ANNOTATIONS: [&str; 4] = ["description", "title", "default", "examples"];

/// Add `inner`'s keywords to `outer`, which both must hold.
///
/// # Errors
///
/// When both constrain one keyword differently, which a merge would loosen.
fn merge(outer: &mut Map<String, Value>, inner: Map<String, Value>) -> Result<(), &'static str> {
    for (key, value) in inner {
        match outer.get(&key) {
            None => {
                outer.insert(key, value);
            }
            Some(existing) if *existing == value || ANNOTATIONS.contains(&key.as_str()) => {}
            Some(_) => return Err("a reference or `allOf` that restates a keyword differently"),
        }
    }
    Ok(())
}

/// The schema a local `reference` (a JSON Pointer fragment: `#`,
/// `#/$defs/Name`, `#/definitions/Outer/definitions/Inner`) names in
/// `root`.
fn definition<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    root.pointer(reference.strip_prefix('#')?)
}

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "deepseek";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DEEPSEEK, "DeepSeek");

#[cfg(test)]
mod tests;
