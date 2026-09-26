use serde_json::json;

use super::*;

fn delta(id: &str, text: &str) -> Value {
    json!({"event": "block_delta", "id": id, "delta": {"delta": "text", "text": text}})
}

fn text_end(id: &str, block: Option<&str>) -> Value {
    json!({"event": "block_end", "id": id, "end": {"close": "text"},
           "block": block.map(|text| json!({"type": "text", "text": text}))})
}

fn last() -> Value {
    json!({"event": "final", "usage": null})
}

fn log(deliveries: Value, events: Vec<Value>, errors: Option<Value>) -> Value {
    let mut header = json!({"deliveries": deliveries});
    if let Some(errors) = errors {
        header["stream_errors"] = json!({"0": errors});
    }
    json!({"header": header, "records": [{"id": 0, "outcome": null, "events": events}]})
}

fn stream(batch: u64, items: usize) -> Value {
    json!({"batch": batch, "id": 0, "kind": {"delivery": "stream", "items": items}})
}

fn outcome(batch: u64) -> Value {
    json!({"batch": batch, "id": 0, "kind": {"delivery": "outcome"}})
}

fn classified(base: &Value, head: &Value) -> Changes {
    let mut changes = Changes::default();
    classify(base, head, &mut changes);
    changes
}

#[test]
fn a_close_its_block_and_the_counts_it_shifts_are_explained() {
    let base = log(
        json!([stream(1, 2), stream(2, 1), outcome(2)]),
        vec![delta("b", "a"), text_end("b", None)],
        Some(json!([{"item": 2, "error": {"message": "late"}}])),
    );
    let head = log(
        json!([stream(1, 2), stream(2, 1), outcome(2)]),
        vec![delta("b", "a"), text_end("b", Some("a"))],
        Some(json!([{"item": 2, "error": {"message": "late"}}])),
    );
    assert_eq!(
        classified(&base, &head),
        Changes {
            files: 1,
            blocks_added: 1,
            ..Changes::default()
        }
    );

    let base = log(
        json!([stream(1, 2), stream(2, 1), outcome(2)]),
        vec![delta("b", "a"), delta("b", "b")],
        Some(json!([{"item": 2, "error": {"message": "late"}}])),
    );
    let head = log(
        json!([stream(1, 2), stream(2, 2), outcome(2)]),
        vec![delta("b", "a"), delta("b", "b"), text_end("b", Some("ab"))],
        Some(json!([{"item": 3, "error": {"message": "late"}}])),
    );
    assert_eq!(
        classified(&base, &head),
        Changes {
            files: 1,
            closes_inserted: 1,
            count_shifts: 2,
            ..Changes::default()
        }
    );
}

#[test]
fn batches_other_than_the_base_s_grown_are_delivery_churn() {
    let base = log(
        json!([stream(1, 1), stream(2, 1), outcome(2)]),
        vec![delta("b", "a"), last()],
        None,
    );
    let head = log(
        json!([stream(1, 3), outcome(1)]),
        vec![delta("b", "a"), text_end("b", Some("a")), last()],
        None,
    );
    let changes = classified(&base, &head);
    assert_eq!(changes.delivery_churn, 1);
    assert_eq!(changes.closes_inserted, 1);
    assert!(changes.other.is_empty());
}

#[test]
fn anything_else_is_other() {
    let base = log(json!([]), vec![delta("b", "a"), last()], None);

    let edited = log(json!([]), vec![delta("b", "z"), last()], None);
    assert_eq!(classified(&base, &edited).other.len(), 1, "an edited delta");

    let misplaced = log(
        json!([]),
        vec![text_end("b", Some("")), delta("b", "a"), last()],
        None,
    );
    assert_eq!(
        classified(&base, &misplaced).other.len(),
        1,
        "a close before a delta"
    );

    let mut renamed = base.clone();
    renamed["header"]["signature"] = json!("changed");
    assert_eq!(classified(&base, &renamed).other.len(), 1, "a new key");

    let error = log(json!([]), vec![delta("b", "a")], Some(json!([{"item": 1}])));
    let shifted = log(json!([]), vec![delta("b", "a")], Some(json!([{"item": 2}])));
    assert_eq!(
        classified(&error, &shifted).other.len(),
        1,
        "an error position no insertion explains"
    );
}

#[test]
fn a_validated_offset_grows_by_the_closes_inserted_before_it() {
    let program = |validated: u64, events: Vec<Value>| {
        json!({"golden/run#0": [{"entities": [
            {"Outputs": {"stream_validated": validated}},
            {"Streamed": {"events": events}},
        ]}]})
    };
    let base = program(2, vec![delta("b", "a"), last()]);
    let closed = vec![delta("b", "a"), text_end("b", Some("a")), last()];

    let changes = classified(&base, &program(3, closed.clone()));
    assert_eq!((changes.closes_inserted, changes.count_shifts), (1, 1));
    assert!(changes.other.is_empty());

    assert_eq!(
        classified(&base, &program(4, closed)).other.len(),
        1,
        "grown by more than the closes before it"
    );
}

#[test]
fn a_text_block_must_be_its_deltas() {
    let events = [delta("b", "a"), delta("b", "b"), text_end("b", Some("ab"))];
    assert!(block_mismatches(&events).is_empty());
    let events = [delta("b", "a"), text_end("b", Some("b"))];
    assert_eq!(block_mismatches(&events).len(), 1);
}

#[test]
fn a_reasoning_block_is_its_deltas_unless_the_end_restates_it() {
    let reasoning = |id: &str, text: &str| json!({"event": "block_delta", "id": id, "delta": {"delta": "reasoning", "text": text}});
    let end = |id: &str, restated: Option<&str>, block: &str, signature: Option<&str>| {
        json!({"event": "block_end", "id": id,
               "end": {"close": "reasoning", "wire_sent": true, "signature": signature,
                       "reasoning": restated.map(|text| json!({"content": [{"type": "text", "content": {"text": text}}]}))},
               "block": {"type": "reasoning", "content": [{"type": "text", "content": {"text": block, "signature": signature}}]}})
    };
    // A late signature restates the part the synthesized end finished.
    let events = [
        reasoning("r", "th"),
        reasoning("r", "ink"),
        end("r", None, "think", None),
        end("r", None, "think", Some("sig")),
    ];
    assert!(block_mismatches(&events).is_empty());
    let events = [
        reasoning("r", "think"),
        end("r", Some("whole"), "whole", None),
    ];
    assert!(block_mismatches(&events).is_empty(), "a restatement stands");
    let events = [reasoning("r", "think"), end("r", None, "other", None)];
    assert_eq!(block_mismatches(&events).len(), 1);
}

#[test]
fn a_tool_call_is_its_argument_fragments_unless_the_end_restates_them() {
    let arguments = |fragment: &str| json!({"event": "block_delta", "id": "c", "delta": {"delta": "tool_arguments", "arguments": fragment}});
    let end = |restated: Option<Value>, block: Value| {
        let mut end = json!({"close": "tool_call", "on_unparseable": "error"});
        if let Some(restated) = restated {
            end["arguments"] = restated;
        }
        json!({"event": "block_end", "id": "c", "end": end,
               "block": {"type": "toolcall", "function": {"name": "f", "arguments": block}}})
    };
    let events = [
        arguments("{\"q\":"),
        arguments("1}"),
        end(None, json!({"q": 1})),
    ];
    assert!(block_mismatches(&events).is_empty());
    let events = [
        arguments("{\"q\":1}"),
        end(Some(json!({"q": 2})), json!({"q": 2})),
    ];
    assert!(block_mismatches(&events).is_empty(), "a restatement stands");
    let events = [arguments("{\"q\":1}"), end(None, json!({"q": 2}))];
    assert_eq!(block_mismatches(&events).len(), 1);
}
