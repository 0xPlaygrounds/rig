use serde_json::json;

use super::*;

/// A block-shaped text delta on block `t`.
fn delta(text: &str) -> Value {
    json!({"event": "block_delta", "id": "t", "delta": {"delta": "text", "text": text}})
}

/// A block-shaped text end on block `t`.
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

fn event(value: Value) -> Value {
    json!({"item": "event", "value": value})
}

fn start(part: u64, kind: &str) -> Value {
    event(json!({"event": "start", "part": part, "kind": kind}))
}

fn text(part: u64, text: &str) -> Value {
    event(json!({"event": "text", "part": part, "text": text}))
}

fn text_end(part: u64, text: &str) -> Value {
    event(json!({"event": "end", "part": part, "content": {"type": "text", "text": text}}))
}

fn call(id: &str, arguments: Value) -> Value {
    json!({
        "type": "toolcall",
        "id": {"provider": {"call_id": id}},
        "function": {"name": "add", "arguments": arguments},
    })
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
        "records": [{"id": 0, "outcome": {"Ok": {"choice": []}}, "events": events}],
    })
}

fn audit(base: Option<&Value>, head: &Value) -> Audit {
    let mut audit = Audit::default();
    audit.file("golden.json", base, head);
    audit
}

fn found(audit: &Audit, what: &str) -> bool {
    audit.other.iter().any(|problem| problem.contains(what))
}

#[test]
fn an_ended_text_that_disagrees_with_its_fragments_is_a_mismatch() {
    let head = log(
        vec![start(0, "text"), text(0, "hi"), text_end(0, "ho")],
        vec![],
        vec![(1, 3)],
    );
    let found = audit(None, &head);
    assert_eq!(found.mismatches.len(), 1, "{:?}", found.mismatches);
    let agrees = log(
        vec![
            start(0, "text"),
            text(0, "h"),
            text(0, "i"),
            text_end(0, "hi"),
        ],
        vec![],
        vec![(1, 4)],
    );
    let found = audit(None, &agrees);
    assert!(found.mismatches.is_empty());
    assert_eq!(found.parts, 1);
}

#[test]
fn an_ended_call_whose_arguments_disagree_with_its_json_is_a_mismatch() {
    let arguments = |json: &str| event(json!({"event": "arguments", "part": 0, "json": json}));
    let ended = |content| event(json!({"event": "end", "part": 0, "content": content}));
    let head = log(
        vec![
            start(0, "tool_call"),
            arguments(r#"{"x": 2}"#),
            ended(call("c", json!({"x": 3}))),
        ],
        vec![],
        vec![(1, 3)],
    );
    assert_eq!(audit(None, &head).mismatches.len(), 1);
    let empty = log(
        vec![start(0, "tool_call"), ended(call("c", json!({})))],
        vec![],
        vec![(1, 2)],
    );
    assert!(audit(None, &empty).mismatches.is_empty());
}

#[test]
fn a_block_stream_that_finalizes_the_same_parts_migrated_and_its_counts_may_move() {
    let base = log(
        vec![delta("h"), delta("i"), end(Some("hi")), done()],
        vec![],
        vec![(1, 2), (2, 2)],
    );
    let head = log(
        vec![
            start(0, "text"),
            text(0, "h"),
            text(0, "i"),
            text_end(0, "hi"),
        ],
        vec![],
        vec![(1, 2), (2, 2)],
    );
    let found = audit(Some(&base), &head);
    assert!(found.other.is_empty(), "{:?}", found.other);
    assert_eq!(found.changes.get(&Change::Migrated), Some(&1));
    let shifted = log(
        vec![
            start(0, "text"),
            text(0, "h"),
            text(0, "i"),
            text_end(0, "hi"),
        ],
        vec![],
        vec![(1, 3), (2, 1)],
    );
    let found = audit(Some(&base), &shifted);
    assert!(found.other.is_empty(), "{:?}", found.other);
    assert_eq!(found.changes.get(&Change::CountShift), Some(&2));
}

#[test]
fn a_migrated_stream_that_finalizes_other_content_is_other() {
    let base = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let head = log(
        vec![start(0, "text"), text(0, "ho"), text_end(0, "ho")],
        vec![],
        vec![(1, 3)],
    );
    assert!(found(&audit(Some(&base), &head), "finalizes"));
}

#[test]
fn parts_in_another_order_are_other() {
    let second = |kind: &str| json!({"event": "block_end", "id": "u", "end": {"close": "text"}, "block": {"type": "text", "text": kind}});
    let base = log(
        vec![delta("a"), end(Some("a")), second("b"), done()],
        vec![],
        vec![(1, 4)],
    );
    let head = log(
        vec![
            start(0, "text"),
            text(0, "b"),
            text_end(0, "b"),
            start(1, "text"),
            text(1, "a"),
            text_end(1, "a"),
        ],
        vec![],
        vec![(1, 6)],
    );
    assert!(found(&audit(Some(&base), &head), "finalizes"));
}

#[test]
fn a_dropped_part_of_a_stream_that_ended_well_is_other() {
    let base = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let head = log(vec![], vec![], vec![]);
    assert!(found(&audit(Some(&base), &head), "dropped"));
}

#[test]
fn a_cut_streams_closed_tail_may_stay_open_or_a_call_be_absent() {
    let mut base = log(vec![delta("hi"), end(Some("hi"))], vec![], vec![(1, 2)]);
    let mut head = log(vec![start(0, "text"), text(0, "hi")], vec![], vec![(1, 2)]);
    for log in [&mut base, &mut head] {
        log["records"][0]["outcome"] = json!({"Err": {"message": "cut"}});
    }
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::CutTail), Some(&1));

    let call_end = json!({"event": "block_end", "id": "c", "end": {"close": "tool_call"}, "block": call("c", json!({}))});
    let mut base = log(vec![call_end], vec![], vec![(1, 1)]);
    base["records"][0]["outcome"] = json!({"Err": {"message": "cut"}});
    let mut head = log(vec![], vec![], vec![]);
    head["records"][0]["outcome"] = base["records"][0]["outcome"].clone();
    head["header"]["deliveries"] = json!([{"batch": 99, "id": 0, "kind": {"delivery": "outcome"}}]);
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);

    // Left open with other text, or in a stream that ended well, it is other.
    let mut other_text = log(vec![start(0, "text"), text(0, "ho")], vec![], vec![(1, 2)]);
    other_text["records"][0]["outcome"] = json!({"Err": {"message": "cut"}});
    let mut cut_base = log(vec![delta("hi"), end(Some("hi"))], vec![], vec![(1, 2)]);
    cut_base["records"][0]["outcome"] = json!({"Err": {"message": "cut"}});
    assert!(!audit(Some(&cut_base), &other_text).other.is_empty());
    let whole = log(vec![delta("hi"), end(Some("hi"))], vec![], vec![(1, 2)]);
    let open = log(vec![start(0, "text"), text(0, "hi")], vec![], vec![(1, 2)]);
    assert!(!audit(Some(&whole), &open).other.is_empty());
}

#[test]
fn a_part_the_base_left_open_may_read_on() {
    let base = log(vec![delta("h")], vec![], vec![(1, 1)]);
    let head = log(
        vec![start(0, "text"), text(0, "hi"), text_end(0, "hi")],
        vec![],
        vec![(1, 3)],
    );
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::ReadOn), Some(&1));
    let diverged = log(
        vec![start(0, "text"), text(0, "yo"), text_end(0, "yo")],
        vec![],
        vec![(1, 3)],
    );
    assert!(!audit(Some(&base), &diverged).other.is_empty());
}

#[test]
fn a_typed_stream_that_changes_is_other() {
    let base = log(
        vec![start(0, "text"), text(0, "hi"), text_end(0, "hi")],
        vec![],
        vec![(1, 3)],
    );
    let mut head = base.clone();
    head["records"][0]["events"][1]["value"]["text"] = json!("ho");
    head["records"][0]["events"][2]["value"]["content"]["text"] = json!("ho");
    assert!(found(&audit(Some(&base), &head), "the stream changed"));
}

#[test]
fn counts_move_only_in_a_migrated_golden() {
    let base = log(
        vec![start(0, "text"), text(0, "hi")],
        vec![2],
        vec![(1, 2), (2, 1)],
    );
    let mut head = base.clone();
    head["header"]["stream_errors"]["0"][0]["item"] = json!(1);
    assert!(found(&audit(Some(&base), &head), "moved"));
    let mut head = base.clone();
    head["header"]["deliveries"][0]["kind"]["items"] = json!(1);
    assert!(found(&audit(Some(&base), &head), "delivery batches"));
}

#[test]
fn rebatched_deliveries_keep_the_outcomes_and_stay_within_the_stream() {
    let base = log(
        vec![delta("h"), delta("i"), end(Some("hi")), done()],
        vec![],
        vec![(1, 2), (2, 2)],
    );
    let head = log(
        vec![
            start(0, "text"),
            text(0, "h"),
            text(0, "i"),
            text_end(0, "hi"),
        ],
        vec![],
        vec![(4, 4)],
    );
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::Rebatched), Some(&1));
    let past_the_stream = log(
        vec![
            start(0, "text"),
            text(0, "h"),
            text(0, "i"),
            text_end(0, "hi"),
        ],
        vec![],
        vec![(4, 5)],
    );
    assert!(found(&audit(Some(&base), &past_the_stream), "delivered 5"));
    let mut no_outcome = head.clone();
    no_outcome["header"]["deliveries"] =
        json!([{"batch": 4, "id": 0, "kind": {"delivery": "stream", "items": 4}}]);
    assert!(found(&audit(Some(&base), &no_outcome), "delivery batches"));
}

#[test]
fn a_streamed_outcomes_raw_may_become_the_terminal_document() {
    let base = log(
        vec![delta("hi"), end(Some("hi")), done()],
        vec![],
        vec![(1, 3)],
    );
    let mut head = log(
        vec![start(0, "text"), text(0, "hi"), text_end(0, "hi")],
        vec![],
        vec![(1, 3)],
    );
    let mut base = base;
    base["records"][0]["outcome"]["Ok"]["raw"] = json!({"summary": true});
    head["records"][0]["outcome"]["Ok"]["raw"] = json!({"id": "resp_1"});
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::RawReplaced), Some(&1));
    // An effect that did not stream keeps its raw.
    let unary = |raw: Value| json!({"records": [{"id": 0, "outcome": {"Ok": {"raw": raw}}}]});
    let found_ = audit(Some(&unary(json!({"a": 1}))), &unary(json!({"b": 1})));
    assert!(!found_.other.is_empty());
}

#[test]
fn a_truncated_streams_error_is_reported_as_truncated() {
    let error = |message: &str| json!({"records": [{"id": 0, "outcome": {"Err": {"kind": "response", "message": message}}}]});
    let found_ = audit(
        Some(&error("the stream ended before its terminal record")),
        &error("ResponseError: the reply ended before the provider ended it"),
    );
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::TruncationReported), Some(&1));
    let other = audit(
        Some(&error("the stream ended before its terminal record")),
        &error("ResponseError: something else"),
    );
    assert!(!other.other.is_empty());
}

#[test]
fn a_replayed_rejected_call_may_become_the_call_the_model_made() {
    let local = json!({"local": "00000000-0000-4000-8000-000000000000"});
    let replayed =
        json!({"type": "toolcall", "id": local, "function": {"name": "add", "arguments": null}});
    let result =
        |call: &Value| json!({"type": "toolresult", "call": call, "name": "add", "content": []});
    let made = call("c", json!({"x": 1}));
    let golden = |replayed: &Value, call_id: &Value| {
        json!({"records": [
            {"id": 0, "outcome": {"Ok": {"choice": [made]}}},
            {"id": 1, "kind": {"request": {"chat_history": [
                {"role": "assistant", "content": [replayed]},
                {"role": "user", "content": [result(call_id)]},
            ]}}},
        ]})
    };
    let base = golden(&replayed, &local);
    let head = golden(&made, &made["id"]);
    let found_ = audit(Some(&base), &head);
    assert!(found_.other.is_empty(), "{:?}", found_.other);
    assert_eq!(found_.changes.get(&Change::CallRestored), Some(&1));
    // A call the base never finalized is other.
    let invented = call("z", json!({"x": 9}));
    assert!(
        !audit(Some(&base), &golden(&invented, &invented["id"]))
            .other
            .is_empty()
    );
}

/// The checked-in corpus: every ended part carries what its fragments
/// assemble.
#[test]
fn every_golden_end_carries_what_its_fragments_assemble() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root");
    let mut found = Audit::default();
    for (path, golden) in goldens(root).expect("the corpus reads") {
        found.file(&path, None, &golden);
    }
    assert!(found.files > 1000, "only {} goldens found", found.files);
    assert!(found.parts > 1000, "only {} parts checked", found.parts);
    assert!(
        found.mismatches.is_empty(),
        "{}",
        found.mismatches.join("\n")
    );
}
