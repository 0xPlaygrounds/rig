//! DeepSeek model identifiers. [`from_env`] and [`new`] build a client on the [`DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK) dialect.
//!
//! A chat model with [strict tools](crate::providers::openai::wire::Chat::with_strict_tools)
//! uses DeepSeek's strict function calling: requests go to its `/beta`
//! endpoint, with tool schemas closed and without what strict mode refuses:
//! length and item-count bounds, and any `format` but `email`, `hostname`,
//! `ipv4`, `ipv6` and `uuid` (schemars writes `uint32` for a `u32`).
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

use serde_json::Value;

/// The string formats DeepSeek's strict mode takes on a `"string"` schema.
const STRICT_FORMATS: [&str; 5] = ["email", "hostname", "ipv4", "ipv6", "uuid"];

/// Rewrite `schema`, already sanitized for OpenAI's strict mode, into the
/// one DeepSeek's strict mode takes: a `$ref` inside `anyOf` (schemars'
/// `Option` of a struct) written out as its definition, a one-member
/// `allOf` merged into its parent, and a string `format` outside the five
/// it names dropped.
///
/// # Errors
///
/// What strict mode cannot express: an object open to further properties
/// (a map), an `allOf` of several schemas, a type that holds itself through
/// an `anyOf`, or a `$ref` that names no definition.
pub(crate) fn strict_schema(schema: &mut Value) -> Result<(), &'static str> {
    let definitions = schema.clone();
    strict_node(schema, &definitions, &mut Vec::new())
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
    if let Some(all_of) = object.shift_remove("allOf") {
        let Value::Array(members) = all_of else {
            return Err("an `allOf` that is not a list");
        };
        let mut members = members.into_iter();
        match (members.next(), members.next()) {
            (Some(Value::Object(member)), None) => {
                for (key, value) in member {
                    object.entry(key).or_insert(value);
                }
            }
            _ => return Err("an `allOf` of several schemas"),
        }
    }
    if object
        .get("additionalProperties")
        .is_some_and(|open| open != &Value::Bool(false))
    {
        return Err("an object open to further properties");
    }
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
            let Some(reference) = member.get("$ref").and_then(Value::as_str) else {
                strict_node(member, root, inlining)?;
                continue;
            };
            let name = reference.to_owned();
            if inlining.contains(&name) {
                return Err("a type that holds itself through `anyOf`");
            }
            *member = definition(root, &name)
                .ok_or("a `$ref` that names no definition")?
                .clone();
            inlining.push(name);
            strict_node(member, root, inlining)?;
            inlining.pop();
        }
    }
    for key in ["properties", "$defs", "definitions"] {
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

/// The definition a local `reference` (`#/$defs/Name`) names in `root`.
fn definition<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    [("#/$defs/", "$defs"), ("#/definitions/", "definitions")]
        .into_iter()
        .find_map(|(prefix, key)| root.get(key)?.get(reference.strip_prefix(prefix)?))
}

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "deepseek";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DEEPSEEK, "DeepSeek");

#[cfg(test)]
mod tests;
