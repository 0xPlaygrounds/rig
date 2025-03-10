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
                if let Some(i) = num.as_i64() {
                    match i > 0 {
                        true => AwsDocument(Document::Number(Number::PosInt(i as u64))),
                        false => AwsDocument(Document::Number(Number::NegInt(i))),
                    }
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
                    .map(|json| json.into())
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
mod tests {
    use std::collections::HashMap;

    use aws_smithy_types::{Document, Number};
    use serde_json::Value;

    use crate::types::json::AwsDocument;

    #[test]
    fn test_json_to_aws_document() {
        let json = r#"
            {
                "type": "object",
                "is_enabled": true,
                "version": 42,
                "fraction": 1.23,
                "negative": -11,
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required":["x", "y", null]
            }
        "#;

        let value: Value = serde_json::from_str(json).unwrap();
        let document: AwsDocument = value.into();
        println!("{:?}", document);
    }

    #[test]
    fn test_aws_document_to_json() {
        let document = AwsDocument(Document::Object(HashMap::from([
            (
                String::from("type"),
                Document::String(String::from("object")),
            ),
            (
                String::from("version"),
                Document::Number(Number::PosInt(42)),
            ),
            (
                String::from("fraction"),
                Document::Number(Number::Float(1.23)),
            ),
            (
                String::from("negative"),
                Document::Number(Number::NegInt(-11)),
            ),
            (String::from("is_enabled"), Document::Bool(true)),
            (
                String::from("properties"),
                Document::Object(HashMap::from([
                    (
                        String::from("x"),
                        Document::Object(HashMap::from([
                            (
                                String::from("type"),
                                Document::String(String::from("number")),
                            ),
                            (
                                String::from("description"),
                                Document::String(String::from("The first number to add")),
                            ),
                        ])),
                    ),
                    (
                        String::from("y"),
                        Document::Object(HashMap::from([
                            (
                                String::from("type"),
                                Document::String(String::from("number")),
                            ),
                            (
                                String::from("description"),
                                Document::String(String::from("The second number to add")),
                            ),
                        ])),
                    ),
                ])),
            ),
            (
                String::from("required"),
                Document::Array(vec![
                    Document::String(String::from("x")),
                    Document::String(String::from("y")),
                    Document::Null,
                ]),
            ),
        ])));

        let json: Value = document.into();
        println!("{:?}", json);
    }
}
