use aws_smithy_types::{Document, Number};
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug)]
pub struct AwsDocument(pub Document);

impl From<AwsDocument> for Value {
    fn from(value: AwsDocument) -> Self {
        match value.0 {
            Document::Object(obj) => {
                let documents = obj
                    .into_iter()
                    .map(|(k, v)| (k, AwsDocument(v).into()))
                    .collect::<Map<_, _>>();
                Value::Object(documents)
            }
            Document::Array(arr) => {
                let documents = arr.into_iter().map(|v| AwsDocument(v).into()).collect();
                Value::Array(documents)
            }
            Document::Number(Number::PosInt(number)) => {
                Value::Number(serde_json::Number::from(number))
            }
            Document::Number(Number::NegInt(number)) => {
                Value::Number(serde_json::Number::from(number))
            }
            Document::Number(Number::Float(number)) => match serde_json::Number::from_f64(number) {
                Some(n) => Value::Number(n),
                // https://www.rfc-editor.org/rfc/rfc7159
                // Numeric values that cannot be represented in the grammar (such as Infinity and NaN) are not permitted.
                None => Value::Null,
            },
            Document::String(s) => Value::String(s),
            Document::Bool(b) => Value::Bool(b),
            Document::Null => Value::Null,
        }
    }
}

impl From<Value> for AwsDocument {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => AwsDocument(Document::Null),
            Value::Bool(b) => AwsDocument(Document::Bool(b)),
            Value::Number(num) => {
                if let Some(unsigned) = num.as_u64() {
                    AwsDocument(Document::Number(Number::PosInt(unsigned)))
                } else if let Some(signed) = num.as_i64() {
                    AwsDocument(Document::Number(Number::NegInt(signed)))
                } else if let Some(f) = num.as_f64() {
                    AwsDocument(Document::Number(Number::Float(f)))
                } else {
                    AwsDocument(Document::Null)
                }
            }
            Value::String(s) => AwsDocument(Document::String(s)),
            Value::Array(arr) => {
                let documents = arr
                    .into_iter()
                    .map(std::convert::Into::into)
                    .map(|aws: AwsDocument| aws.0)
                    .collect();
                AwsDocument(Document::Array(documents))
            }
            Value::Object(obj) => {
                let documents = obj
                    .into_iter()
                    .map(|(k, v)| {
                        let doc: AwsDocument = v.into();
                        (k, doc.0)
                    })
                    .collect::<HashMap<_, _>>();
                AwsDocument(Document::Object(documents))
            }
        }
    }
}

#[cfg(test)]
mod tests;
