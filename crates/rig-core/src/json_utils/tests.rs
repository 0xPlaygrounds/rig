use super::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct SortedMapHolder {
    #[serde(serialize_with = "serialize_map_sorted")]
    map: HashMap<String, u32>,
}

#[derive(Serialize)]
struct OptionalSortedMapHolder {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_map_sorted"
    )]
    map: Option<HashMap<String, u32>>,
}

/// The property this exists to guarantee: identical content serializes to
/// identical bytes, no matter how the map was built.
///
/// Two maps with the same entries inserted in *opposite* orders must produce
/// the same JSON. Without sorting they generally do not, and every request
/// carrying such a map gets a different wire prefix — which makes it a
/// permanent prompt-cache miss.
#[test]
fn sorted_map_serialization_is_insertion_order_independent() {
    let forward = SortedMapHolder {
        map: [("alpha", 1), ("beta", 2), ("gamma", 3), ("delta", 4)]
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    };
    let reverse = SortedMapHolder {
        map: [("delta", 4), ("gamma", 3), ("beta", 2), ("alpha", 1)]
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    };

    let forward = serde_json::to_string(&forward).expect("serialize");
    let reverse = serde_json::to_string(&reverse).expect("serialize");

    assert_eq!(forward, reverse);
    assert_eq!(
        forward, r#"{"map":{"alpha":1,"beta":2,"delta":4,"gamma":3}}"#,
        "keys must come out in sorted order"
    );
}

#[test]
fn optional_sorted_map_serializes_some_sorted_and_skips_none() {
    let some = OptionalSortedMapHolder {
        map: Some(
            [("zulu", 1), ("alpha", 2)]
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        ),
    };
    assert_eq!(
        serde_json::to_string(&some).expect("serialize"),
        r#"{"map":{"alpha":2,"zulu":1}}"#
    );

    let none = OptionalSortedMapHolder { map: None };
    assert_eq!(serde_json::to_string(&none).expect("serialize"), "{}");
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Dummy {
    #[serde(with = "stringified_json")]
    data: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct DummyMaybeStringified {
    #[serde(deserialize_with = "stringified_json::deserialize_maybe_stringified")]
    data: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct ArgWrapper {
    #[serde(default, deserialize_with = "deserialize_json_string_or_value")]
    arguments: Option<String>,
}

/// Spec-compliant case: `arguments` is already a JSON-encoded string, taken verbatim.
#[test]
fn json_string_or_value_string_passthrough() {
    let w: ArgWrapper = serde_json::from_str(r#"{"arguments":"{\"a\":1}"}"#).unwrap();
    assert_eq!(w.arguments.as_deref(), Some(r#"{"a":1}"#));
}

/// Non-compliant gateway: an empty object `{}` must serialize to the string `"{}"`,
/// not be treated as absent (None).
#[test]
fn json_string_or_value_empty_object() {
    let w: ArgWrapper = serde_json::from_str(r#"{"arguments":{}}"#).unwrap();
    assert_eq!(w.arguments.as_deref(), Some("{}"));
}

/// Non-compliant gateway: a nested object is re-serialized to a string.
#[test]
fn json_string_or_value_nested_object() {
    let w: ArgWrapper = serde_json::from_str(r#"{"arguments":{"path":"/tmp","depth":2}}"#).unwrap();
    // `arguments` is re-serialized from a Value; object key order is not guaranteed
    // (depends on serde_json's `preserve_order` feature), so re-parse and compare
    // values rather than the raw string.
    let parsed: serde_json::Value = serde_json::from_str(w.arguments.as_deref().unwrap()).unwrap();
    assert_eq!(parsed["path"], "/tmp");
    assert_eq!(parsed["depth"], 2);
}

/// Non-compliant gateway: an array is also "any other JSON value" and serializes to a
/// string. Array order is meaningful and preserved by serde_json, so compare the string
/// directly.
#[test]
fn json_string_or_value_array() {
    let w: ArgWrapper = serde_json::from_str(r#"{"arguments":[1,2,3]}"#).unwrap();
    assert_eq!(w.arguments.as_deref(), Some("[1,2,3]"));
}

/// Regression test: JSON null must collapse to None (not the string "null").
/// Removing `.filter(|v| !v.is_null())` from the deserializer would fail this test.
#[test]
fn json_string_or_value_null_is_none() {
    let w: ArgWrapper = serde_json::from_str(r#"{"arguments":null}"#).unwrap();
    assert!(w.arguments.is_none());
}

/// A missing field is likewise None (relies on `#[serde(default)]`).
#[test]
fn json_string_or_value_missing_is_none() {
    let w: ArgWrapper = serde_json::from_str(r#"{}"#).unwrap();
    assert!(w.arguments.is_none());
}

#[test]
fn test_merge() {
    let a = serde_json::json!({"key1": "value1"});
    let b = serde_json::json!({"key2": "value2"});
    let result = merge(a, b);
    let expected = serde_json::json!({"key1": "value1", "key2": "value2"});
    assert_eq!(result, expected);
}

#[test]
fn test_merge_inplace() {
    let mut a = serde_json::json!({"key1": "value1"});
    let b = serde_json::json!({"key2": "value2"});
    merge_inplace(&mut a, b);
    let expected = serde_json::json!({"key1": "value1", "key2": "value2"});
    assert_eq!(a, expected);
}

#[test]
fn test_stringified_json_serialize() {
    let dummy = Dummy {
        data: serde_json::json!({"key": "value"}),
    };
    let serialized = serde_json::to_string(&dummy).unwrap();
    let expected = r#"{"data":"{\"key\":\"value\"}"}"#;
    assert_eq!(serialized, expected);
}

#[test]
fn test_stringified_json_deserialize() {
    let json_str = r#"{"data":"{\"key\":\"value\"}"}"#;
    let dummy: Dummy = serde_json::from_str(json_str).unwrap();
    let expected = Dummy {
        data: serde_json::json!({"key": "value"}),
    };
    assert_eq!(dummy, expected);
}

#[test]
fn test_stringified_json_deserialize_empty_string() {
    let json_str = r#"{"data":""}"#;
    let dummy: Dummy = serde_json::from_str(json_str).unwrap();
    assert_eq!(dummy.data, serde_json::json!({}));
}

#[test]
fn test_deserialize_maybe_stringified_value_from_string() {
    let json_str = r#"{"data":"{\"key\":\"value\"}"}"#;
    let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
    assert_eq!(dummy.data, serde_json::json!({"key": "value"}));
}

#[test]
fn test_deserialize_maybe_stringified_value_from_object() {
    let json_str = r#"{"data":{"key":"value"}}"#;
    let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
    assert_eq!(dummy.data, serde_json::json!({"key": "value"}));
}

#[test]
fn test_deserialize_maybe_stringified_value_from_empty_string() {
    let json_str = r#"{"data":""}"#;
    let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
    assert_eq!(dummy.data, serde_json::json!({}));
}

#[test]
fn test_parse_tool_arguments_empty_string() {
    let parsed = parse_tool_arguments("").unwrap();
    assert_eq!(parsed, serde_json::json!({}));
}

#[test]
fn test_parse_tool_arguments_whitespace_string() {
    let parsed = parse_tool_arguments("   ").unwrap();
    assert_eq!(parsed, serde_json::json!({}));
}

#[test]
fn test_parse_tool_arguments_valid_json() {
    let parsed = parse_tool_arguments(r#"{"key":"value"}"#).unwrap();
    assert_eq!(parsed, serde_json::json!({"key": "value"}));
}

mod string_or_vec_shapes {
    use serde::Deserialize;
    use std::convert::Infallible;
    use std::str::FromStr;

    /// A content block that can arrive in every shape the helper accepts.
    ///
    /// The suite deliberately does not use `Vec<String>`: a bare object
    /// cannot deserialize into a `String`, so a string element type makes
    /// the `visit_map` arm structurally untestable — and that is the arm
    /// whose loss would be a silent wire break, since several providers
    /// spell single-block content as a bare object.
    #[derive(Debug, Deserialize, PartialEq)]
    struct Block {
        text: String,
    }

    impl FromStr for Block {
        type Err = Infallible;

        fn from_str(value: &str) -> Result<Self, Self::Err> {
            Ok(Block {
                text: value.to_owned(),
            })
        }
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Holder {
        #[serde(deserialize_with = "super::super::string_or_vec")]
        content: Vec<Block>,
    }

    fn decode(json: serde_json::Value) -> Vec<Block> {
        serde_json::from_value::<Holder>(json)
            .expect("shape should decode")
            .content
    }

    fn block(text: &str) -> Block {
        Block {
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_bare_object_is_one_element() {
        // `visit_map`. Carried over from the removed container's
        // `string_or_one_or_many`, which had this arm where the helper it
        // merged into did not.
        assert_eq!(
            decode(serde_json::json!({"content": {"text": "hi"}})),
            vec![block("hi")]
        );
    }

    #[test]
    fn a_bare_string_becomes_one_element_via_from_str() {
        assert_eq!(
            decode(serde_json::json!({"content": "hi"})),
            vec![block("hi")]
        );
    }

    #[test]
    fn a_sequence_decodes_elementwise() {
        assert_eq!(
            decode(serde_json::json!({"content": [{"text": "a"}, {"text": "b"}]})),
            vec![block("a"), block("b")]
        );
    }

    #[test]
    fn an_empty_sequence_is_an_empty_list() {
        // The non-empty container this helper replaced rejected `[]`
        // outright. It is now a value.
        assert!(decode(serde_json::json!({"content": []})).is_empty());
    }

    #[test]
    fn null_is_an_empty_list() {
        // Load-bearing: OpenAI sends `"content": null` for a message that
        // carries only tool calls, so dropping this arm would turn a normal
        // response into a decode error.
        assert!(decode(serde_json::json!({"content": null})).is_empty());
    }
}
