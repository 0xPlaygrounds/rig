//! A tool's parameters derived from its arguments type, so the schema the
//! model sees and the type its arguments are parsed into cannot disagree:
//!
//! ```
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! #[serde(deny_unknown_fields)]
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

use schemars::generate::SchemaSettings;
use schemars::transform::RecursiveTransform;
use schemars::{JsonSchema, Schema};
use serde_json::Value;

/// The JSON schema of the arguments type `A`, in the plain shape tool
/// definitions use: inline, without `$schema`, `title`, `default`, `format`
/// (providers support it unevenly) or the type's own doc comment, and with
/// `Option` fields optional, not nullable. With `deny_unknown_fields` on
/// every struct, each object refuses keys it does not declare.
pub fn args_schema<A: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft2020_12().with_transform(RecursiveTransform(plain));
    settings.inline_subschemas = true;
    settings.meta_schema = None;
    let mut schema = settings.into_generator().into_root_schema_for::<A>();
    if let Some(root) = schema.as_object_mut() {
        root.shift_remove("description");
    }
    schema.to_value()
}

/// Drops from one subschema what [`args_schema`] leaves out, keeping the
/// other keys' order, and joins a doc comment's lines as rustdoc does.
fn plain(schema: &mut Schema) {
    let Some(schema) = schema.as_object_mut() else {
        return;
    };
    if let Some(Value::String(text)) = schema.get_mut("description") {
        let paragraphs: Vec<String> = text.split("\n\n").map(|p| p.replace('\n', " ")).collect();
        *text = paragraphs.join("\n\n");
    }
    for key in ["title", "default", "format"] {
        schema.shift_remove(key);
    }
    if let Some(Value::Array(types)) = schema.get_mut("type") {
        types.retain(|kind| kind != "null");
        if let [kind] = types.as_slice() {
            let kind = kind.clone();
            schema.insert("type".to_owned(), kind);
        }
    }
}
