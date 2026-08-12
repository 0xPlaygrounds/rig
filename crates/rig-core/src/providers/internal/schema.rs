//! Shared JSON-schema sanitizer for providers with strict structured-output
//! restrictions (OpenAI, Anthropic). The common core enforces
//! `additionalProperties: false`, all-properties-required, recursion into
//! `$defs`/`properties`/`items`/combinators, and the oneOf→anyOf merge;
//! [`SanitizeOptions`] flags carry the per-provider deltas.

/// Per-provider deltas applied by [`sanitize_schema`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SanitizeOptions {
    /// OpenAI does not allow sibling keywords next to `$ref`; strip everything
    /// except `$ref` and stop descending into that node.
    pub(crate) strip_ref_siblings: bool,
    /// OpenAI requires `properties` on all object schemas, even empty ones.
    pub(crate) inject_empty_properties: bool,
    /// Anthropic does not support numerical constraints on integer/number
    /// types (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
    /// `multipleOf`).
    pub(crate) strip_numeric_constraints: bool,
}

/// Recursively rewrites `schema` in place to respect a provider's structured
/// output restrictions. See [`SanitizeOptions`] for the provider-specific
/// behavior.
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value, options: SanitizeOptions) {
    use serde_json::Value;

    if let Value::Object(obj) = schema {
        if options.strip_ref_siblings && obj.contains_key("$ref") {
            obj.retain(|k, _| k == "$ref");
            return;
        }

        let is_object_schema = obj.get("type") == Some(&Value::String("object".to_string()))
            || obj.contains_key("properties");

        if options.inject_empty_properties && is_object_schema && !obj.contains_key("properties") {
            obj.insert("properties".to_string(), Value::Object(Default::default()));
        }

        if is_object_schema && !obj.contains_key("additionalProperties") {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));
        }

        if let Some(Value::Object(properties)) = obj.get("properties") {
            let prop_keys = properties.keys().cloned().map(Value::String).collect();
            obj.insert("required".to_string(), Value::Array(prop_keys));
        }

        if options.strip_numeric_constraints {
            let is_numeric_schema = obj.get("type") == Some(&Value::String("integer".to_string()))
                || obj.get("type") == Some(&Value::String("number".to_string()));

            if is_numeric_schema {
                for key in [
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "multipleOf",
                ] {
                    obj.remove(key);
                }
            }
        }

        if let Some(defs) = obj.get_mut("$defs")
            && let Value::Object(defs_obj) = defs
        {
            for (_, def_schema) in defs_obj.iter_mut() {
                sanitize_schema(def_schema, options);
            }
        }

        if let Some(properties) = obj.get_mut("properties")
            && let Value::Object(props) = properties
        {
            for (_, prop_value) in props.iter_mut() {
                sanitize_schema(prop_value, options);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            sanitize_schema(items, options);
        }

        // Neither provider supports oneOf; convert to anyOf, merging into an
        // existing anyOf array if present.
        if let Some(one_of) = obj.remove("oneOf") {
            match obj.get_mut("anyOf") {
                Some(Value::Array(existing)) => {
                    if let Value::Array(mut incoming) = one_of {
                        existing.append(&mut incoming);
                    }
                }
                _ => {
                    obj.insert("anyOf".to_string(), one_of);
                }
            }
        }

        for key in ["anyOf", "allOf"] {
            if let Some(variants) = obj.get_mut(key)
                && let Value::Array(variants_array) = variants
            {
                for variant in variants_array.iter_mut() {
                    sanitize_schema(variant, options);
                }
            }
        }
    }
}
