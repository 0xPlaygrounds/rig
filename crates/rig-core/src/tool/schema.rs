//! A tool's parameters derived from its arguments type, so the schema the
//! model sees and the type its arguments are parsed into cannot disagree:
//!
//! ```
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct Args {
//!     /// How many lines to read.
//!     limit: Option<u32>,
//! }
//!
//! let schema = rig_core::tool::args_schema::<Args>();
//! let limit = serde_json::json!({"type": "integer", "minimum": 0,
//!     "description": "How many lines to read."});
//! assert_eq!(schema["properties"]["limit"], limit);
//! assert_eq!(schema["additionalProperties"], false);
//! ```

use crate::json_utils::remove_preserving_order;
use schemars::generate::SchemaSettings;
use schemars::transform::{RecursiveTransform, transform_subschemas};
use schemars::{JsonSchema, Schema};
use serde_json::Value;

/// The JSON schema of the arguments type `A`, in the plain shape tool
/// definitions use: inline, without `$schema`, `title`, `default`, `format`
/// (providers support it unevenly) or the type's own doc comment, and with
/// `Option` fields optional, not nullable (no `null` type, enum value or
/// `anyOf` alternative), and each standalone object refusing keys
/// it does not declare (`additionalProperties: false`, as
/// `deny_unknown_fields` says; a map keeps its own). Object fragments combined
/// at the same value keep their generated unknown-key policy, so one fragment
/// does not forbid fields declared by another.
pub fn args_schema<A: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft2020_12().with_transform(RecursiveTransform(plain));
    settings.inline_subschemas = true;
    settings.meta_schema = None;
    let mut schema = settings.into_generator().into_root_schema_for::<A>();
    close_objects(&mut schema, false);
    if let Some(root) = schema.as_object_mut() {
        remove_preserving_order(root, "description");
    }
    schema.to_value()
}

/// Drops from one subschema what [`args_schema`] leaves out, keeping the
/// other keys' order, joins a doc
/// comment's lines as rustdoc does, and makes an `Option` optional rather
/// than nullable: no `null` type or enum value, and no `{"type": "null"}`
/// alternative (the one left is merged in).
fn plain(schema: &mut Schema) {
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
            remove_preserving_order(schema, "anyOf");
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
        remove_preserving_order(schema, key);
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

fn close_objects(schema: &mut Schema, fragment: bool) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    let has_properties = object.contains_key("properties");
    let mut compositions = Vec::new();
    for key in ["allOf", "anyOf", "oneOf", "if", "then", "else", "not"] {
        if let Some(value) = object.get_mut(key) {
            compositions.push((key, value.take()));
        }
    }
    if !fragment
        && compositions.is_empty()
        && has_properties
        && !object.contains_key("additionalProperties")
    {
        object.insert("additionalProperties".to_owned(), Value::Bool(false));
    }

    // Properties and array items describe different values; composition branches
    // constrain the same value and may share fields with their parent or peers.
    transform_subschemas(
        &mut |child: &mut Schema| close_objects(child, false),
        schema,
    );
    let shared = fragment || has_properties || compositions.len() > 1;
    for (key, mut value) in compositions {
        let shared = shared || matches!(key, "allOf" | "if" | "then" | "else" | "not");
        if let Some(branches) = value.as_array_mut() {
            for branch in branches {
                if let Ok(branch) = <&mut Schema>::try_from(branch) {
                    close_objects(branch, shared);
                }
            }
        } else if let Ok(branch) = <&mut Schema>::try_from(&mut value) {
            close_objects(branch, shared);
        }
        schema.insert(key.to_owned(), value);
    }
}

#[cfg(test)]
mod tests;
