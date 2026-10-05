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

#[test]
fn json_string_or_value_null_or_missing_is_none() {
    for (case, json) in [("null", r#"{"arguments":null}"#), ("missing", r#"{}"#)] {
        let w: ArgWrapper = serde_json::from_str(json).unwrap();
        assert!(w.arguments.is_none(), "{case}");
    }
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
fn test_deserialize_maybe_stringified_value_from_string_or_object() {
    for (case, json_str) in [
        ("string", r#"{"data":"{\"key\":\"value\"}"}"#),
        ("object", r#"{"data":{"key":"value"}}"#),
    ] {
        let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
        assert_eq!(dummy.data, serde_json::json!({"key": "value"}), "{case}");
    }
}

#[test]
fn test_deserialize_maybe_stringified_value_from_empty_string() {
    let json_str = r#"{"data":""}"#;
    let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
    assert_eq!(dummy.data, serde_json::json!({}));
}

#[test]
fn text_that_states_no_object_has_none() {
    for text in ["not json", "[1, 2", "\"str", "12"] {
        assert_eq!(parse_partial_object(text), None, "{text}");
    }
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

mod partial_arguments {
    use super::super::parse_partial_arguments;
    use serde_json::{Value, json};

    fn parsed(text: &str) -> Value {
        Value::Object(parse_partial_arguments(text))
    }

    #[test]
    fn cut_off_text_keeps_what_it_states() {
        for (text, expected) in [
            ("", json!({})),
            ("   ", json!({})),
            ("{", json!({})),
            (r#"{"path"#, json!({})),
            (r#"{"path":"#, json!({})),
            (r#"{"path": "no"#, json!({"path": "no"})),
            (r#"{"path": "notes.md","#, json!({"path": "notes.md"})),
            (r#"{"path": "notes.md", "con"#, json!({"path": "notes.md"})),
            (r#"{"a": {"b": [1, 2"#, json!({"a": {"b": [1, 2]}})),
            (r#"{"a": [{"b": "c"}, {"d"#, json!({"a": [{"b": "c"}, {}]})),
            (r#"{"a": 1, "b": 2,"#, json!({"a": 1, "b": 2})),
            (r#"{"line": "one\ntw"#, json!({"line": "one\ntw"})),
            (r#"{"quote": "say \"hi"#, json!({"quote": "say \"hi"})),
        ] {
            assert_eq!(parsed(text), expected, "{text}");
        }
    }

    #[test]
    fn escapes_cut_short_keep_the_string_before_them() {
        for (text, expected) in [
            (r#"{"a": "x\"#, json!({"a": "x"})),
            (r#"{"a": "x\u"#, json!({"a": "x"})),
            (r#"{"a": "x\u00"#, json!({"a": "x"})),
            (r#"{"a": "x\u00e"#, json!({"a": "x"})),
            (r#"{"a": "x\u00e9"#, json!({"a": "xé"})),
            (r#"{"a": "x\\u00"#, json!({"a": "x\\u00"})),
        ] {
            assert_eq!(parsed(text), expected, "{text}");
        }
    }

    #[test]
    fn numbers_and_literals_cut_short_are_dropped_until_valid() {
        for (text, expected) in [
            (r#"{"n": 1"#, json!({"n": 1})),
            (r#"{"n": 1."#, json!({})),
            (r#"{"n": -"#, json!({})),
            (r#"{"n": 1e"#, json!({})),
            (r#"{"n": 1.5"#, json!({"n": 1.5})),
            (r#"{"b": tru"#, json!({})),
            (r#"{"b": true"#, json!({"b": true})),
            (r#"{"v": nul"#, json!({})),
            (r#"{"a": 1, "v": nul"#, json!({"a": 1})),
        ] {
            assert_eq!(parsed(text), expected, "{text}");
        }
    }

    #[test]
    fn raw_control_characters_and_stray_backslashes_are_repaired() {
        assert_eq!(
            parsed("{\"text\": \"line one\nline two\ttabbed\"}"),
            json!({"text": "line one\nline two\ttabbed"})
        );
        assert_eq!(
            parsed(r#"{"path": "C:\Users\me"}"#),
            json!({"path": "C:\\Users\\me"})
        );
        assert_eq!(
            parsed("{\"text\": \"cut\nshort"),
            json!({"text": "cut\nshort"})
        );
    }

    #[test]
    fn a_complete_object_parses_as_serde_does() {
        let text = r#"{"path": "a.md", "content": "x\ny \"q\" \u00e9", "n": [1, 2.5, -3e2], "o": {"t": true, "f": null}}"#;
        assert_eq!(
            parsed(text),
            serde_json::from_str::<Value>(text).expect("valid json")
        );
    }

    #[test]
    fn a_top_level_that_is_not_an_object_is_empty() {
        for text in [
            "[1, 2]", "[1, 2", "\"str\"", "\"str", "12", "true", "null", "not json",
        ] {
            assert_eq!(parsed(text), json!({}), "{text}");
        }
    }

    #[test]
    fn every_prefix_parses_and_the_whole_text_parses_exactly() {
        for whole in [
            json!({"path": "notes/today.md", "content": "Line one.\nLine \"two\" \\ é ✓\n\tIndented."}),
            json!({"x": 1, "y": -2.5, "z": 3e10, "flags": [true, false, null], "nested": {"a": [{"b": "c"}]}}),
            json!({"query": "", "limit": 0, "filters": {}, "tags": []}),
        ] {
            for text in [
                whole.to_string(),
                serde_json::to_string_pretty(&whole).expect("json"),
            ] {
                for (cut, _) in text.char_indices() {
                    // Never fails, and always an object.
                    let _ = parse_partial_arguments(&text[..cut]);
                }
                assert_eq!(parsed(&text), whole);
            }
        }
    }
}
