use serde_json::json;

use super::*;

fn delta(text: &str) -> Value {
    json!({"event": "block_delta", "id": "t", "delta": {"delta": "text", "text": text}})
}

fn end(block: Option<&str>) -> Value {
    let mut end = json!({"event": "block_end", "id": "t", "end": {"close": "text"}});
    if let Some(text) = block {
        end["block"] = json!({"type": "text", "text": text});
    }
    end
}

fn done() -> Value {
    json!({"event": "final", "usage": {}})
}

fn log(events: Vec<Value>, errors: Vec<u64>, deliveries: Vec<(u64, u64)>) -> Value {
    let mut deliveries: Vec<Value> = deliveries
        .into_iter()
        .map(|(batch, items)| {
            json!({"batch": batch, "id": 0, "kind": {"delivery": "stream", "items": items}})
        })
        .collect();
    deliveries.push(json!({"batch": 99, "id": 0, "kind": {"delivery": "outcome"}}));
    json!({
        "header": {
            "deliveries": deliveries,
            "stream_errors": {"0": errors.into_iter().map(|item| json!({"item": item, "error": {}})).collect::<Vec<_>>()},
        },
        "records": [{"id": 0, "events": events}],
    })
}

fn audit(base: Option<&Value>, head: &Value) -> Audit {
    let mut audit = Audit::default();
    audit.file("golden.json", base, head);
    audit
}

#[test]
fn a_block_that_disagrees_with_its_deltas_is_a_mismatch() {
    let head = log(
        vec![delta("hi"), end(Some("ho")), done()],
        vec![],
        vec![(1, 3)],
    );
    let found = audit(None, &head);
    assert_eq!(found.mismatches.len(), 1, "{:?}", found.mismatches);
    let agrees = log(
        vec![delta("h"), delta("i"), end(Some("hi")), done()],
        vec![],
        vec![(1, 4)],
    );
    assert!(audit(None, &agrees).mismatches.is_empty());
}

#[test]
fn a_close_before_the_terminal_and_its_count_shift_are_known_changes() {
    let base = log(vec![delta("hi"), done()], vec![], vec![(1, 1), (2, 1)]);
    let head = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 1), (2, 2)],
    );
    let found = audit(Some(&base), &head);
    assert!(found.other.is_empty(), "{:?}", found.other);
    assert_eq!(found.changes.get(&Change::CloseInserted), Some(&1));
    assert_eq!(found.changes.get(&Change::CountShift), Some(&1));
}

#[test]
fn an_end_that_gained_its_block_is_a_known_change() {
    let base = log(vec![delta("hi"), end(None), done()], vec![], vec![(1, 3)]);
    let head = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let found = audit(Some(&base), &head);
    assert!(found.other.is_empty(), "{:?}", found.other);
    assert_eq!(found.changes.get(&Change::BlockAdded), Some(&1));
}

#[test]
fn a_close_inserted_mid_stream_is_other() {
    let base = log(vec![delta("a"), delta("b"), done()], vec![], vec![(1, 3)]);
    let head = log(
        vec![delta("a"), end(Some("a")), delta("b"), done()],
        vec![],
        vec![(1, 4)],
    );
    assert!(!audit(Some(&base), &head).other.is_empty());
}

#[test]
fn a_stream_error_moves_by_exactly_the_events_inserted_before_it() {
    let base = log(vec![delta("hi")], vec![1], vec![(1, 2)]);
    let head = log(vec![delta("hi"), end(Some("hi"))], vec![2], vec![(1, 3)]);
    let found = audit(Some(&base), &head);
    assert!(found.other.is_empty(), "{:?}", found.other);
    assert_eq!(found.changes.get(&Change::CountShift), Some(&2));
    let moved_too_far = log(vec![delta("hi"), end(Some("hi"))], vec![3], vec![(1, 3)]);
    assert!(!audit(Some(&base), &moved_too_far).other.is_empty());
}

#[test]
fn delivery_batches_that_differ_from_the_base_are_other() {
    let base = log(vec![delta("hi"), done()], vec![], vec![(1, 1), (2, 1)]);
    let churned = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let found = audit(Some(&base), &churned);
    assert!(
        found
            .other
            .iter()
            .any(|problem| problem.contains("delivery batches")),
        "{:?}",
        found.other
    );
}

#[test]
fn any_other_change_is_other() {
    let base = log(vec![delta("hi"), done()], vec![], vec![(1, 2)]);
    let mut head = base.clone();
    head["records"][0]["events"][0]["delta"]["text"] = json!("ho");
    assert!(!audit(Some(&base), &head).other.is_empty());
}

#[test]
fn an_inserted_close_for_a_block_already_closed_is_other() {
    let base = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let head = log(
        vec![delta("hi"), end(Some("hi")), end(Some("hi")), done()],
        vec![],
        vec![(1, 4)],
    );
    let found = audit(Some(&base), &head);
    assert!(
        found
            .other
            .iter()
            .any(|problem| problem.contains("no open block")),
        "{:?}",
        found.other
    );
}

#[test]
fn stream_items_that_grow_past_the_inserted_events_are_other() {
    let base = log(vec![delta("hi"), done()], vec![], vec![(1, 2)]);
    let head = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 4)],
    );
    let found = audit(Some(&base), &head);
    assert!(
        found
            .other
            .iter()
            .any(|problem| problem.contains("grew by 2")),
        "{:?}",
        found.other
    );
}

#[test]
fn an_authoritative_block_is_counted_apart() {
    let restated = json!({
        "event": "block_end",
        "id": "r",
        "end": {"close": "reasoning", "reasoning": {"content": []}, "wire_sent": true},
        "block": {"type": "reasoning", "content": [{"type": "text", "content": {"text": "x"}}]}
    });
    let found = audit(None, &log(vec![restated, done()], vec![], vec![(1, 2)]));
    assert_eq!((found.blocks, found.authoritative), (0, 1));
    assert!(found.mismatches.is_empty());
}

#[test]
fn a_text_end_without_its_block_is_a_mismatch() {
    let head = log(vec![delta("hi"), end(None), done()], vec![], vec![(1, 3)]);
    assert_eq!(audit(None, &head).mismatches.len(), 1);
    let empty = log(vec![end(None), done()], vec![], vec![(1, 2)]);
    assert!(audit(None, &empty).mismatches.is_empty());
}

#[test]
fn a_reasoning_end_without_the_part_its_deltas_opened_is_a_mismatch() {
    let reasoning = json!({"event": "block_delta", "id": "r", "delta": {"delta": "reasoning", "text": "think"}});
    let close = json!({"event": "block_end", "id": "r", "end": {"close": "reasoning"}});
    let head = log(vec![reasoning, close, done()], vec![], vec![(1, 3)]);
    assert_eq!(audit(None, &head).mismatches.len(), 1);
}

/// The checked-in corpus: every golden's ends carry what their deltas
/// assemble.
#[test]
fn every_golden_end_carries_what_its_deltas_assemble() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root");
    let audit = corpus_audit(root).expect("the corpus reads");
    assert!(audit.files > 1000, "only {} goldens found", audit.files);
    assert!(
        audit.mismatches.is_empty(),
        "{}",
        audit.mismatches.join("\n")
    );
}
