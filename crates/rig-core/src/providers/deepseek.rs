//! DeepSeek model identifiers. [`from_env`] and [`new`] build a client on the [`DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK) dialect.
//!
//! A chat model with [strict tools](crate::providers::openai::wire::Chat::with_strict_tools)
//! uses DeepSeek's strict function calling: requests go to its `/beta`
//! endpoint, with each tool schema closed, every property required, a
//! `$ref` inside `anyOf` written out, a one-member `allOf` merged, and a
//! string `format` outside `email`, `hostname`, `ipv4`, `ipv6` and `uuid`
//! dropped. A tool whose schema strict mode cannot express, such as a map,
//! is sent as written without `strict`.
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

/// Where a schema keeps its definitions, and the reference prefix that
/// names one: JSON Schema's `$defs`, the `$def` DeepSeek's guide spells,
/// and draft 7's `definitions`.
const DEFINITIONS: [(&str, &str); 3] = [
    ("$defs", "#/$defs/"),
    ("$def", "#/$def/"),
    ("definitions", "#/definitions/"),
];

/// Rewrite `schema` into the one DeepSeek's strict mode takes: every
/// object closed and every property required, as OpenAI's strict mode
/// wants, but with keywords beside a `$ref` kept; a `$ref` inside `anyOf`
/// (schemars' `Option` of a struct) written out as its definition; a
/// one-member `allOf` merged into its parent; and a plain string's
/// `format` outside the five it names dropped.
///
/// # Errors
///
/// What strict mode cannot express: an object open to further properties
/// (a map), an `allOf` of several schemas or one restating a keyword
/// differently, a tuple, a type that holds itself through an `anyOf`, or a
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
    close(schema);
    let definitions = schema.clone();
    strict_node(schema, &definitions, &mut Vec::new())
}

/// Close every object schema at or below `node`, through every keyword
/// that holds schemas: `additionalProperties: false`, every property
/// required, and `oneOf` read as `anyOf`, as OpenAI's strict mode wants.
fn close(node: &mut Value) {
    let Value::Object(object) = node else {
        return;
    };
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
    if let Some(Value::Array(mut one_of)) = object.shift_remove("oneOf") {
        match object.get_mut("anyOf") {
            Some(Value::Array(any_of)) => any_of.append(&mut one_of),
            _ => {
                object.insert("anyOf".to_owned(), Value::Array(one_of));
            }
        }
    }
    for key in ["properties", "$defs", "$def", "definitions"] {
        if let Some(Value::Object(children)) = object.get_mut(key) {
            children.values_mut().for_each(close);
        }
    }
    for key in ["anyOf", "allOf"] {
        if let Some(Value::Array(members)) = object.get_mut(key) {
            members.iter_mut().for_each(close);
        }
    }
    if let Some(items) = object.get_mut("items") {
        close(items);
    }
}

/// [`strict_schema`] on one node, resolving references in `root`, with
/// `inlining` the definitions being written out above it.
fn strict_node(
    node: &mut Value,
    root: &Value,
    inlining: &mut Vec<String>,
) -> Result<(), &'static str> {
    let Value::Object(object) = node else {
        return Ok(());
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
                    strict_node(member, root, inlining)?;
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
            strict_node(member, root, inlining)?;
            inlining.truncate(depth);
        }
    }
    for key in ["properties", "$defs", "$def", "definitions"] {
        if let Some(Value::Object(children)) = object.get_mut(key) {
            for child in children.values_mut() {
                strict_node(child, root, inlining)?;
            }
        }
    }
    if let Some(items) = object.get_mut("items") {
        strict_node(items, root, inlining)?;
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

/// The definition a local `reference` (`#/$defs/Name`, or `#` for the
/// root) names in `root`.
fn definition<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    if reference == "#" {
        return Some(root);
    }
    DEFINITIONS
        .into_iter()
        .find_map(|(key, prefix)| root.get(key)?.get(reference.strip_prefix(prefix)?))
}

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "deepseek";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DEEPSEEK, "DeepSeek");

#[cfg(test)]
mod tests;
