//! Anthropic tool-schema normalization and request-wide strict grammar budgets.

use serde_json::{Map, Value, json};

const COMBINATORS: &[&str] = &["anyOf", "oneOf", "allOf"];
const UNIVERSAL: &[&str] = &[
    "$ref",
    "$defs",
    "$schema",
    "definitions",
    "type",
    "anyOf",
    "allOf",
    "enum",
    "const",
    "description",
    "title",
    "default",
    "nullable",
];
const FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Normalize encoded tools before their names are escaped for the OAuth wire format.
pub(crate) fn normalize_tools(body: &mut Value) {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };
    let mut remaining = Budget {
        optional: 24,
        union: 16,
    };
    let mut strict_count = 0;
    for tool in tools {
        let Some(tool) = tool.as_object_mut() else {
            continue;
        };
        let Some(raw) = tool.get("input_schema").and_then(Value::as_object) else {
            continue;
        };
        let mut wire = upgrade(Value::Object(raw.clone()));
        postprocess(&mut wire);
        empty_schemas(&mut wire);
        let Some(raw) = wire.as_object() else {
            continue;
        };
        let candidate = matches!(
            tool.get("name").and_then(Value::as_str),
            Some("python" | "edit" | "find" | "yield")
        ) && tool.get("strict") != Some(&Value::Bool(false))
            && !incompatible(&Value::Object(raw.clone()));
        let mut raw = raw.clone();
        raw.insert("type".into(), json!("object"));
        if !raw.get("properties").is_some_and(Value::is_object) {
            raw.insert("properties".into(), json!({}));
        }
        let required = raw
            .get("required")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| entry.is_string())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        raw.insert("required".into(), Value::Array(required));
        let mut schema = normalize(Value::Object(raw), true);
        tool.shift_remove("strict");
        if candidate && strict_count < 20 {
            let mut budget = remaining;
            if let Some(strict) = strict_schema(schema.clone(), &mut budget) {
                schema = strict;
                remaining = budget;
                strict_count += 1;
                tool.insert("strict".into(), Value::Bool(true));
            }
        }
        tool.insert("input_schema".into(), schema);
    }
}

fn scalar_type(schema: &Map<String, Value>) -> Option<&str> {
    let explicit = match schema.get("type") {
        Some(Value::String(value)) => Some(value.as_str()),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .find(|value| *value != "null"),
        _ => None,
    };
    explicit.or_else(|| {
        if schema.get("properties").is_some_and(Value::is_object) {
            Some("object")
        } else if schema.contains_key("items")
            || schema.get("prefixItems").is_some_and(Value::is_array)
        {
            Some("array")
        } else {
            None
        }
    })
}

fn normalize(value: Value, root: bool) -> Value {
    let mut source = match value {
        Value::Object(source) => source,
        Value::Array(values) => {
            return Value::Array(
                values
                    .into_iter()
                    .map(|value| normalize(value, false))
                    .collect(),
            );
        }
        value => return value,
    };
    let kind = scalar_type(&source).map(str::to_owned);
    let mut result = Map::new();
    let mut spill = Vec::new();
    for (key, value) in std::mem::take(&mut source) {
        let per_type = match kind.as_deref() {
            Some("object") => {
                ["properties", "required", "additionalProperties"].contains(&key.as_str())
            }
            Some("array") => ["items", "prefixItems", "minItems"].contains(&key.as_str()),
            Some("string") => key == "format",
            _ => false,
        };
        if !(root && COMBINATORS.contains(&key.as_str()))
            && (UNIVERSAL.contains(&key.as_str()) || per_type)
        {
            result.insert(key, value);
        } else {
            spill.push((key, value));
        }
    }
    let prune = match kind.as_deref() {
        Some("string")
            if result
                .get("format")
                .and_then(Value::as_str)
                .is_some_and(|value| !FORMATS.contains(&value)) =>
        {
            Some("format")
        }
        Some("array")
            if result
                .get("minItems")
                .is_some_and(|value| !matches!(value.as_f64(), Some(0.0 | 1.0))) =>
        {
            Some("minItems")
        }
        _ => None,
    };
    if let Some(key) = prune
        && let Some(value) = result.shift_remove(key)
    {
        spill.push((key.into(), value));
    }
    if kind.as_deref() == Some("object") && !result.contains_key("additionalProperties") {
        result.insert("additionalProperties".into(), Value::Bool(false));
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(Value::Object(children)) = result.get_mut(key) {
            for child in children.values_mut() {
                *child = normalize(child.take(), false);
            }
        }
    }
    for key in [
        "additionalProperties",
        "items",
        "prefixItems",
        "anyOf",
        "oneOf",
        "allOf",
    ] {
        if let Some(child) = result.get_mut(key)
            && (child.is_object() || (key != "additionalProperties" && child.is_array()))
        {
            *child = normalize(child.take(), false);
            if key == "additionalProperties" && child.as_object().is_some_and(Map::is_empty) {
                *child = Value::Bool(true);
            }
        }
    }
    if !spill.is_empty() {
        let suffix = format!(
            "{{{}}}",
            spill
                .into_iter()
                .map(|(key, value)| format!("{key}: {}", js_json(&value)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let description = result
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let description = if description.is_empty() {
            suffix
        } else {
            format!("{description}\n\n{suffix}")
        };
        result.insert("description".into(), Value::String(description));
    }
    Value::Object(result)
}

fn js_json(value: &Value) -> String {
    match value {
        Value::Number(number) => number
            .as_f64()
            .map(|number| ryu_js::Buffer::new().format(number).to_owned())
            .unwrap_or_else(|| number.to_string()),
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(js_json).collect::<Vec<_>>().join(",")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}:{}", Value::String(key.clone()), js_json(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => value.to_string(),
    }
}

fn incompatible(value: &Value) -> bool {
    match value {
        Value::Object(value) => {
            [
                "oneOf",
                "allOf",
                "$ref",
                "patternProperties",
                "propertyNames",
            ]
            .iter()
            .any(|key| value.contains_key(*key))
                || value.values().any(incompatible)
        }
        Value::Array(values) => values.iter().any(incompatible),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct Budget {
    optional: usize,
    union: usize,
}

fn spend_union(budget: &mut Budget) -> Option<()> {
    budget.union = budget.union.checked_sub(1)?;
    Some(())
}

fn nullable(mut value: Value, budget: &mut Budget) -> Option<Value> {
    if let Some(schema) = value.as_object_mut() {
        if schema
            .get("type")
            .and_then(Value::as_array)
            .is_some_and(|types| types.iter().any(|value| value == "null"))
            || schema
                .get("anyOf")
                .and_then(Value::as_array)
                .is_some_and(|variants| {
                    variants
                        .iter()
                        .any(|value| value.get("type") == Some(&json!("null")))
                })
        {
            return Some(value);
        }
        if let Some(Value::Array(variants)) = schema.get_mut("anyOf") {
            variants.push(json!({"type":"null"}));
            return Some(value);
        }
        if let Some(Value::Array(types)) = schema.get_mut("type") {
            types.push(json!("null"));
            return Some(value);
        }
    }
    spend_union(budget)?;
    Some(json!({"anyOf": [value, {"type":"null"}]}))
}

fn strict_schema(value: Value, budget: &mut Budget) -> Option<Value> {
    let mut schema = match value {
        Value::Object(schema) => schema,
        Value::Array(values) => {
            return values
                .into_iter()
                .map(|value| strict_schema(value, budget))
                .collect::<Option<Vec<_>>>()
                .map(Value::Array);
        }
        value => return Some(value),
    };
    if ![
        "type",
        "properties",
        "additionalProperties",
        "items",
        "prefixItems",
        "enum",
        "const",
        "$ref",
        "anyOf",
        "oneOf",
        "allOf",
        "$defs",
        "definitions",
    ]
    .iter()
    .any(|key| schema.contains_key(*key))
    {
        return None;
    }
    let types = schema.get("type");
    let has_type = |kind: &str| {
        types.is_some_and(|value| {
            value == kind
                || value
                    .as_array()
                    .is_some_and(|types| types.iter().any(|value| value == kind))
        })
    };
    let array = types == Some(&json!("array"))
        || (has_type("array") && !has_type("object"))
        || schema.contains_key("items")
        || schema.get("prefixItems").is_some_and(Value::is_array);
    let object =
        !array && (has_type("object") || schema.get("properties").is_some_and(Value::is_object));
    if object && schema.get("additionalProperties") != Some(&Value::Bool(false)) {
        return None;
    }
    if schema.get("type").is_some_and(Value::is_array)
        || schema.get("anyOf").is_some_and(Value::is_array)
    {
        spend_union(budget)?;
    }
    let original_required = schema
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(Value::Object(properties)) = schema.get_mut("properties") {
        let mut required = Vec::new();
        for (name, child) in properties.iter_mut() {
            let normalized = strict_schema(child.take(), budget)?;
            if original_required
                .iter()
                .any(|value| value.as_str() == Some(name))
            {
                *child = normalized;
                required.push(Value::String(name.clone()));
            } else if budget.optional > 0 {
                budget.optional -= 1;
                *child = normalized;
            } else {
                *child = nullable(normalized, budget)?;
                required.push(Value::String(name.clone()));
            }
        }
        schema.insert("required".into(), Value::Array(required));
    }
    for key in ["items", "prefixItems", "anyOf", "oneOf", "allOf"] {
        if let Some(child) = schema.get_mut(key)
            && (child.is_array() || (key == "items" && child.is_object()))
        {
            *child = strict_schema(child.take(), budget)?;
        }
    }
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(definitions)) = schema.get_mut(key) {
            for child in definitions.values_mut() {
                *child = strict_schema(child.take(), budget)?;
            }
        }
    }
    Some(Value::Object(schema))
}

const DEFINING_SIBLINGS: &[&str] = &[
    "$ref",
    "additionalProperties",
    "allOf",
    "const",
    "contains",
    "enum",
    "if",
    "items",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "properties",
    "propertyNames",
    "then",
    "else",
    "unevaluatedItems",
    "unevaluatedProperties",
];

fn homogeneous(values: &[Value]) -> Option<&'static str> {
    let kind = |value: &Value| match value {
        Value::String(_) => Some("string"),
        Value::Bool(_) => Some("boolean"),
        Value::Number(_) => Some("number"),
        _ => None,
    };
    let first = kind(values.first()?)?;
    values
        .iter()
        .all(|value| kind(value) == Some(first))
        .then_some(first)
}

fn schema_children(schema: &mut Map<String, Value>, mut visit: impl FnMut(&mut Value)) {
    for key in [
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
        "contentSchema",
    ] {
        if let Some(child) = schema.get_mut(key) {
            visit(child);
        }
    }
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ] {
        if let Some(Value::Object(children)) = schema.get_mut(key) {
            for child in children.values_mut() {
                visit(child);
            }
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(children)) = schema.get_mut(key) {
            for child in children {
                visit(child);
            }
        }
    }
}

fn postprocess(value: &mut Value) {
    match value {
        Value::Object(schema) => {
            if !DEFINING_SIBLINGS
                .iter()
                .any(|key| schema.contains_key(*key))
                && let Some(variants) = schema.get("anyOf").and_then(Value::as_array).cloned()
            {
                let scalar = variants.iter().find_map(|value| {
                    value.as_object().filter(|variant| {
                        matches!(
                            variant.get("type").and_then(Value::as_str),
                            Some("string" | "number" | "integer" | "boolean")
                        )
                    })
                });
                if variants.len() == 2
                    && variants
                        .iter()
                        .any(|value| value == &json!({"type":"null"}))
                    && let Some(scalar) = scalar
                {
                    schema.shift_remove("anyOf");
                    for (key, value) in scalar {
                        if !["type", "enum", "const"].contains(&key.as_str())
                            && !schema.contains_key(key)
                        {
                            schema.insert(key.clone(), value.clone());
                        }
                    }
                    if let Some(value) = scalar.get("const") {
                        schema.insert("enum".into(), json!([value, null]));
                    } else if let Some(Value::Array(values)) = scalar.get("enum") {
                        let mut values = values.clone();
                        if !values.contains(&Value::Null) {
                            values.push(Value::Null);
                        }
                        schema.insert("enum".into(), Value::Array(values));
                    }
                    if let Some(kind) = scalar.get("type") {
                        schema.insert("type".into(), json!([kind, "null"]));
                    }
                }
            }
            if !schema.contains_key("type")
                && let Some(kind) = schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .and_then(|values| homogeneous(values))
            {
                schema.insert("type".into(), json!(kind));
            }
            if !schema.contains_key("type")
                && !DEFINING_SIBLINGS
                    .iter()
                    .any(|key| schema.contains_key(*key))
                && let Some(variants) = schema.get("anyOf").and_then(Value::as_array).cloned()
            {
                let values: Option<Vec<_>> = variants
                    .iter()
                    .map(|variant| {
                        let variant = variant.as_object()?;
                        if variant
                            .keys()
                            .any(|key| key != "const" && key != "description")
                        {
                            return None;
                        }
                        variant.get("const").cloned()
                    })
                    .collect();
                if let Some(values) = values.filter(|values| values.len() >= 2) {
                    let descriptions: Vec<_> = variants
                        .iter()
                        .map(|variant| variant.get("description").and_then(Value::as_str))
                        .collect();
                    let shared = descriptions.first().copied().flatten();
                    let consistent = descriptions
                        .iter()
                        .all(|description| *description == shared)
                        && !(shared.is_some()
                            && schema
                                .get("description")
                                .and_then(Value::as_str)
                                .is_some_and(|description| Some(description) != shared));
                    if let Some(kind) = homogeneous(&values).filter(|_| consistent) {
                        schema.shift_remove("anyOf");
                        schema.insert("type".into(), json!(kind));
                        schema.insert("enum".into(), Value::Array(values));
                        if !schema.get("description").is_some_and(Value::is_string)
                            && let Some(description) = shared
                        {
                            schema.insert("description".into(), json!(description));
                        }
                    }
                }
            }
            schema_children(schema, postprocess);
        }
        Value::Array(values) => {
            for child in values {
                postprocess(child);
            }
        }
        _ => {}
    }
}

fn empty_schemas(value: &mut Value) {
    fn replace(value: &mut Value) {
        if value.as_object().is_some_and(Map::is_empty) {
            *value = Value::Bool(true);
        }
    }
    match value {
        Value::Object(schema) => {
            for key in [
                "additionalProperties",
                "unevaluatedProperties",
                "unevaluatedItems",
                "items",
                "contains",
                "propertyNames",
                "if",
                "then",
                "else",
                "not",
            ] {
                if let Some(child) = schema.get_mut(key) {
                    replace(child);
                }
            }
            for key in ["properties", "patternProperties", "$defs", "definitions"] {
                if let Some(Value::Object(children)) = schema.get_mut(key) {
                    for child in children.values_mut() {
                        replace(child);
                    }
                }
            }
            for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
                if let Some(Value::Array(children)) = schema.get_mut(key) {
                    for child in children {
                        replace(child);
                    }
                }
            }
            schema_children(schema, empty_schemas);
        }
        Value::Array(children) => {
            for child in children {
                empty_schemas(child);
            }
        }
        _ => {}
    }
}

fn intersect(left: Value, right: Value) -> Value {
    if left == Value::Bool(true) {
        right
    } else if right == Value::Bool(true) || left == right {
        left
    } else if left == Value::Bool(false) || right == Value::Bool(false) {
        Value::Bool(false)
    } else {
        json!({"allOf":[left,right]})
    }
}

fn upgrade(value: Value) -> Value {
    let source = match value {
        Value::Object(source) => source,
        Value::Array(values) => return Value::Array(values.into_iter().map(upgrade).collect()),
        value => return value,
    };
    let mut result = Map::new();
    for (key, value) in &source {
        match key.as_str() {
            "definitions" | "$defs" => {
                if let Some(definitions) = value.as_object() {
                    let target = result.entry("$defs").or_insert_with(|| json!({}));
                    if let Some(target) = target.as_object_mut() {
                        for (name, value) in definitions {
                            target.insert(name.clone(), upgrade(value.clone()));
                        }
                    }
                }
            }
            "properties" | "patternProperties" | "dependentSchemas" => {
                let value = if let Some(children) = value.as_object() {
                    Value::Object(
                        children
                            .iter()
                            .map(|(name, child)| (name.clone(), upgrade(child.clone())))
                            .collect(),
                    )
                } else {
                    value.clone()
                };
                result.insert(key.clone(), value);
            }
            "const" | "default" | "enum" | "example" | "examples" | "required"
            | "dependentRequired" | "type" => {
                result.insert(key.clone(), value.clone());
            }
            "dependencies" | "additionalItems" | "nullable" => {}
            "$schema" => {
                let value = match value.as_str() {
                    Some(
                        "http://json-schema.org/draft-07/schema#"
                        | "https://json-schema.org/draft-07/schema#"
                        | "http://json-schema.org/draft-07/schema"
                        | "https://json-schema.org/draft-07/schema",
                    ) => json!("https://json-schema.org/draft/2020-12/schema"),
                    _ => value.clone(),
                };
                result.insert(key.clone(), value);
            }
            "$ref" => {
                let value = value
                    .as_str()
                    .and_then(|value| value.strip_prefix("#/definitions/"))
                    .map(|suffix| json!(format!("#/$defs/{suffix}")))
                    .unwrap_or_else(|| value.clone());
                result.insert(key.clone(), value);
            }
            "items" if value.is_array() => {}
            _ => {
                result.insert(key.clone(), upgrade(value.clone()));
            }
        }
    }
    if let Some(Value::Array(items)) = source.get("items") {
        let mut prefix = result
            .get("prefixItems")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (index, item) in items.iter().enumerate() {
            let item = upgrade(item.clone());
            if let Some(existing) = prefix.get_mut(index) {
                *existing = intersect(existing.take(), item);
            } else {
                prefix.push(item);
            }
        }
        result.insert("prefixItems".into(), Value::Array(prefix));
        if let Some(rest) = source
            .get("additionalItems")
            .filter(|value| **value != Value::Bool(true))
        {
            result.insert("items".into(), upgrade(rest.clone()));
        } else {
            result.shift_remove("items");
        }
    }
    if let Some(Value::Object(dependencies)) = source.get("dependencies") {
        for (name, dependency) in dependencies {
            let converted = upgrade(dependency.clone());
            let key = if converted.is_array() {
                "dependentRequired"
            } else {
                "dependentSchemas"
            };
            let target = result.entry(key).or_insert_with(|| json!({}));
            if !target.is_object() {
                *target = json!({});
            }
            if let Some(target) = target.as_object_mut() {
                if let Some(existing) = target.get_mut(name) {
                    if let Value::Array(new) = converted {
                        if let Some(existing) = existing.as_array_mut() {
                            for value in new {
                                if !existing.contains(&value) {
                                    existing.push(value);
                                }
                            }
                        }
                    } else {
                        *existing = intersect(existing.take(), converted);
                    }
                } else {
                    target.insert(name.clone(), converted);
                }
            }
        }
    }
    if source.get("nullable") == Some(&Value::Bool(true)) {
        if let Some(kind) = result.get_mut("type") {
            if let Value::String(name) = kind {
                if name != "null" {
                    *kind = json!([name, "null"]);
                }
                return Value::Object(result);
            }
            if let Value::Array(types) = kind {
                if !types.contains(&json!("null")) {
                    types.push(json!("null"));
                }
                return Value::Object(result);
            }
        }
        if let Some(Value::Array(variants)) = result.get_mut("anyOf") {
            if !variants.iter().any(|variant| {
                variant.get("type").is_some_and(|kind| {
                    kind == "null"
                        || kind
                            .as_array()
                            .is_some_and(|types| types.contains(&json!("null")))
                })
            }) {
                variants.push(json!({"type":"null"}));
            }
        } else {
            return json!({"anyOf":[result, {"type":"null"}]});
        }
    }
    Value::Object(result)
}

#[cfg(test)]
mod differential;
#[cfg(test)]
mod tests;
