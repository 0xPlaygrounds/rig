use aws_smithy_types::{Document, Number};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// The JSON value of a Smithy document.
pub(crate) fn to_value(document: Document) -> Value {
    match document {
        Document::Object(obj) => {
            // Smithy objects are hash maps. Stable insertion order also
            // stabilizes JSON strings used as tool arguments.
            let mut entries: Vec<_> = obj.into_iter().collect();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            let documents = entries
                .into_iter()
                .map(|(k, v)| (k, to_value(v)))
                .collect::<Map<_, _>>();
            Value::Object(documents)
        }
        Document::Array(arr) => Value::Array(arr.into_iter().map(to_value).collect()),
        Document::Number(Number::PosInt(number)) => Value::Number(serde_json::Number::from(number)),
        Document::Number(Number::NegInt(number)) => Value::Number(serde_json::Number::from(number)),
        Document::Number(Number::Float(number)) => match serde_json::Number::from_f64(number) {
            Some(n) => Value::Number(n),
            // JSON cannot represent non-finite numbers.
            None => Value::Null,
        },
        Document::String(s) => Value::String(s),
        Document::Bool(b) => Value::Bool(b),
        Document::Null => Value::Null,
    }
}

/// The Smithy document of a JSON value.
pub(crate) fn to_document(value: Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(b),
        Value::Number(num) => {
            if let Some(unsigned) = num.as_u64() {
                Document::Number(Number::PosInt(unsigned))
            } else if let Some(signed) = num.as_i64() {
                Document::Number(Number::NegInt(signed))
            } else if let Some(f) = num.as_f64() {
                Document::Number(Number::Float(f))
            } else {
                Document::Null
            }
        }
        Value::String(s) => Document::String(s),
        Value::Array(arr) => Document::Array(arr.into_iter().map(to_document).collect()),
        Value::Object(obj) => Document::Object(
            obj.into_iter()
                .map(|(k, v)| (k, to_document(v)))
                .collect::<HashMap<_, _>>(),
        ),
    }
}

#[cfg(test)]
mod tests;
