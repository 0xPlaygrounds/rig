//! The fragment merge, one rule at a time.

use serde_json::{Map, Value, json};

use super::merge_fields;

/// `target` after merging each of `deltas` into it in turn.
fn merged(target: Value, deltas: &[Value]) -> Value {
    let Value::Object(mut target) = target else {
        panic!("the target is an object: {target}");
    };
    for delta in deltas {
        let Value::Object(delta) = delta else {
            panic!("a delta is an object: {delta}");
        };
        merge_fields(&mut target, delta);
    }
    Value::Object(target)
}

#[test]
fn fragment_strings_append_and_identifiers_are_restated() {
    let call = merged(
        json!({}),
        &[
            json!({"id": "call_1", "arguments": "{\"a\""}),
            json!({"id": "call_1", "arguments": ":1}"}),
        ],
    );
    assert_eq!(call, json!({"id": "call_1", "arguments": "{\"a\":1}"}));
}

#[test]
fn a_null_argument_placeholder_gives_way_to_the_first_real_fragment() {
    let call = merged(
        json!({"arguments": "null"}),
        &[json!({"arguments": " "}), json!({"arguments": "{}"})],
    );
    // Blank text does not displace the placeholder; the first real
    // fragment does.
    assert_eq!(call, json!({"arguments": "{}"}));
}

#[test]
fn null_and_empty_strings_never_erase_a_value() {
    let item = merged(
        json!({"id": "call_1", "name": "lookup"}),
        &[json!({"id": "", "name": null})],
    );
    assert_eq!(item, json!({"id": "call_1", "name": "lookup"}));
}

#[test]
fn a_non_fragment_string_is_replaced_by_a_later_one() {
    let item = merged(json!({"type": "function"}), &[json!({"type": "custom"})]);
    assert_eq!(item, json!({"type": "custom"}));
}

#[test]
fn arrays_extend_and_objects_merge_key_by_key() {
    let item = merged(
        json!({"annotations": [1], "function": {"name": "f", "arguments": "{"}}),
        &[json!({"annotations": [2], "function": {"arguments": "}"}})],
    );
    assert_eq!(
        item,
        json!({"annotations": [1, 2], "function": {"name": "f", "arguments": "{}"}})
    );
}

#[test]
fn a_value_never_replaces_one_of_another_type_but_fills_a_null() {
    let item = merged(
        json!({"index": 0, "extra": null}),
        &[json!({"index": "zero", "extra": {"k": 1}})],
    );
    assert_eq!(item, json!({"index": 0, "extra": {"k": 1}}));
}

#[test]
fn a_new_key_is_inserted() {
    let mut target = Map::new();
    merge_fields(
        &mut target,
        &Map::from_iter([("refusal".to_owned(), Value::Null)]),
    );
    assert_eq!(Value::Object(target), json!({"refusal": null}));
}
