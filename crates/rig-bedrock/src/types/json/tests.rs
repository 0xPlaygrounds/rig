use std::collections::HashMap;

use aws_smithy_types::{Document, Number};
use serde_json::Value;

use crate::types::json::AwsDocument;

#[test]
fn unsigned_json_numbers_round_trip_without_precision_loss() {
    let value = serde_json::json!(u64::MAX);
    let document: AwsDocument = value.clone().into();
    assert!(matches!(
        &document.0,
        Document::Number(Number::PosInt(number)) if *number == u64::MAX
    ));

    let roundtrip: Value = document.into();
    assert_eq!(roundtrip, value);
}

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
    println!("{document:?}");
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
    println!("{json:?}");
}
