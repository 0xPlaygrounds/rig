use serde_json::json;

use super::*;

#[test]
fn only_a_deliveries_change_is_churn() {
    let base = r#"{"header":{"deliveries":[1,2],"run_spec":7},"records":[{"a":1}]}"#;
    let reordered = r#"{"header":{"deliveries":[2,1],"run_spec":7},"records":[{"a":1}]}"#;
    let real = r#"{"header":{"deliveries":[2,1],"run_spec":7},"records":[{"a":2}]}"#;
    assert!(delivery_only(base, reordered));
    assert!(!delivery_only(base, real));
    assert!(!delivery_only(base, base), "an unchanged file is not churn");
    assert!(!delivery_only(base, "not json"));
}

fn delta(text: &str) -> Value {
    json!({"event": "block_delta", "id": "b", "delta": {"delta": "text", "text": text}})
}

fn close() -> Value {
    json!({"event": "block_end", "id": "b", "end": {"close": "text"}, "block": {"type": "text", "text": "ab"}})
}

fn last() -> Value {
    json!({"event": "final", "usage": null})
}

fn stream(id: u64, items: usize) -> Value {
    json!({"batch": 0, "id": id, "kind": {"delivery": "stream", "items": items}})
}

fn batch(mut delivery: Value, batch: u64) -> Value {
    delivery["batch"] = json!(batch);
    delivery
}

fn outcome(id: u64, batch: u64) -> Value {
    json!({"batch": batch, "id": id, "kind": {"delivery": "outcome"}})
}

fn log(deliveries: Value, events: Vec<Value>, errors: Option<Value>) -> Value {
    let mut header = json!({"deliveries": deliveries, "required": true});
    if let Some(errors) = errors {
        header["stream_errors"] = json!({"0": errors});
    }
    json!({"header": header, "records": [{"id": 0, "outcome": null, "events": events}]})
}

#[test]
fn an_end_that_gained_its_block_restates_the_base_end() {
    let bare = json!({"event": "block_end", "id": "b", "end": {"close": "text"}, "block": null});
    assert!(restates(&bare, &close()));
    assert!(restates(&close(), &close()));
    assert!(!restates(&close(), &bare), "an end never loses its block");
    assert!(!restates(&delta("a"), &delta("b")));
}

#[test]
fn a_close_before_the_final_counts_toward_the_batch_of_the_final() {
    let base = log(
        json!([
            batch(stream(0, 2), 3),
            batch(stream(0, 1), 4),
            outcome(0, 4)
        ]),
        vec![delta("a"), delta("b"), last()],
        None,
    );
    let head = log(
        json!([batch(stream(0, 4), 2), outcome(0, 2)]),
        vec![delta("a"), delta("b"), close(), last()],
        None,
    );
    assert_eq!(
        rebase_deliveries(&base, &head),
        Ok(Some(json!([
            batch(stream(0, 2), 3),
            batch(stream(0, 2), 4),
            outcome(0, 4)
        ])))
    );
}

#[test]
fn a_close_at_the_end_of_the_stream_counts_toward_the_last_stream_batch() {
    let base = log(
        json!([
            batch(stream(0, 1), 1),
            batch(stream(0, 1), 2),
            outcome(0, 3)
        ]),
        vec![delta("a"), delta("b")],
        None,
    );
    let head = log(json!([]), vec![delta("a"), delta("b"), close()], None);
    assert_eq!(
        rebase_deliveries(&base, &head),
        Ok(Some(json!([
            batch(stream(0, 1), 1),
            batch(stream(0, 2), 2),
            outcome(0, 3)
        ])))
    );
}

#[test]
fn a_close_before_a_stream_error_counts_toward_the_batch_of_the_error() {
    // The error is item 2 in the base and item 3 once the close is inserted.
    let base = log(
        json!([
            batch(stream(0, 2), 1),
            batch(stream(0, 1), 2),
            outcome(0, 2)
        ]),
        vec![delta("a"), delta("b")],
        Some(json!([{"item": 2, "error": {}}])),
    );
    let head = log(
        json!([]),
        vec![delta("a"), delta("b"), close()],
        Some(json!([{"item": 3, "error": {}}])),
    );
    assert_eq!(
        rebase_deliveries(&base, &head),
        Ok(Some(json!([
            batch(stream(0, 2), 1),
            batch(stream(0, 2), 2),
            outcome(0, 2)
        ])))
    );
}

#[test]
fn a_close_before_an_unlisted_trailing_error_counts_toward_the_batch_of_the_error() {
    // The failed outcome's error was delivered as item 2 without a
    // `stream_errors` entry.
    let base = log(
        json!([
            batch(stream(0, 1), 1),
            batch(stream(0, 2), 2),
            outcome(0, 2)
        ]),
        vec![delta("a"), delta("b")],
        None,
    );
    let head = log(json!([]), vec![delta("a"), delta("b"), close()], None);
    assert_eq!(
        rebase_deliveries(&base, &head),
        Ok(Some(json!([
            batch(stream(0, 1), 1),
            batch(stream(0, 3), 2),
            outcome(0, 2)
        ])))
    );
}

#[test]
fn unchanged_streams_keep_the_base_deliveries() {
    let base = log(
        json!([batch(stream(0, 1), 1), outcome(0, 1)]),
        vec![last()],
        None,
    );
    let head = log(
        json!([batch(stream(0, 1), 5), outcome(0, 6)]),
        vec![last()],
        None,
    );
    assert_eq!(
        rebase_deliveries(&base, &head),
        Ok(base.pointer("/header/deliveries").cloned())
    );
}

#[test]
fn a_golden_without_deliveries_has_nothing_to_rebase() {
    let base = json!({"header": {}, "records": []});
    assert_eq!(rebase_deliveries(&base, &base), Ok(None));
}

#[test]
fn inserted_events_that_do_not_fit_the_rule_are_refused() {
    let deliveries = json!([batch(stream(0, 3), 1), outcome(0, 1)]);
    let base = log(
        deliveries.clone(),
        vec![delta("a"), delta("b"), last()],
        None,
    );

    let mid_stream = log(
        json!([]),
        vec![delta("a"), close(), delta("b"), last()],
        None,
    );
    assert!(
        rebase_deliveries(&base, &mid_stream).is_err(),
        "a close before a delta"
    );

    let not_a_close = log(
        json!([]),
        vec![delta("a"), delta("b"), delta("c"), last()],
        None,
    );
    assert!(
        rebase_deliveries(&base, &not_a_close).is_err(),
        "an inserted delta"
    );

    let lost = log(json!([]), vec![delta("a"), last()], None);
    assert!(rebase_deliveries(&base, &lost).is_err(), "a removed event");
}

#[test]
fn a_close_after_an_undelivered_item_is_refused() {
    // A cancelled stream delivered only its first item, so the item the
    // close precedes was never in a batch.
    let base = log(
        json!([batch(stream(0, 1), 1)]),
        vec![delta("a"), last()],
        None,
    );
    let head = log(json!([]), vec![delta("a"), close(), last()], None);
    assert!(rebase_deliveries(&base, &head).is_err());

    // A close at the end of such a stream would make its last batch deliver
    // the undelivered delta instead.
    let base = log(
        json!([batch(stream(0, 1), 1)]),
        vec![delta("a"), delta("b")],
        None,
    );
    let head = log(json!([]), vec![delta("a"), delta("b"), close()], None);
    assert!(rebase_deliveries(&base, &head).is_err());
}

#[test]
fn splicing_keeps_every_other_byte() {
    let text = "{\n  \"header\": {\n    \"deliveries\": [\n      {\n        \"batch\": 1,\n        \"id\": 0,\n        \"kind\": {\n          \"delivery\": \"stream\",\n          \"items\": 2\n        }\n      }\n    ],\n    \"required\": true\n  },\n  \"records\": [\"]\"]\n}\n";
    let spliced = splice_deliveries(text, &json!([batch(stream(0, 3), 1)]));
    assert_eq!(spliced, Ok(text.replace("\"items\": 2", "\"items\": 3")));
}
