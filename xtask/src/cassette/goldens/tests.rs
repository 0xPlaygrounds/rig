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

/// A world golden: one streamed effect whose deliveries split `items`.
fn golden(events: &[&str], errors: &[u64], batches: &[(u64, u64)]) -> String {
    let deliveries: Vec<String> = batches
        .iter()
        .map(|(batch, items)| {
            format!(
                "{{\"batch\": {batch}, \"id\": 0, \"kind\": {{\"delivery\": \"stream\", \"items\": {items}}}}}"
            )
        })
        .chain(std::iter::once(
            "{\"batch\": 99, \"id\": 0, \"kind\": {\"delivery\": \"outcome\"}}".to_owned(),
        ))
        .collect();
    let errors: Vec<String> = errors
        .iter()
        .map(|item| format!("{{\"item\": {item}, \"error\": {{}}}}"))
        .collect();
    format!(
        "{{\"header\": {{\"deliveries\": [{}], \"stream_errors\": {{\"0\": [{}]}}}}, \"records\": [{{\"id\": 0, \"events\": [{}]}}]}}",
        deliveries.join(", "),
        errors.join(", "),
        events.join(", ")
    )
}

const DELTA: &str =
    r#"{"event": "block_delta", "id": "t", "delta": {"delta": "text", "text": "hi"}}"#;
const OPEN_END: &str = r#"{"event": "block_end", "id": "t", "end": {"close": "text"}}"#;
const END: &str = r#"{"event": "block_end", "id": "t", "end": {"close": "text"}, "block": {"type": "text", "text": "hi"}}"#;
const FINAL: &str = r#"{"event": "final", "usage": {}}"#;
const TOOL: &str = r#"{"event": "block_end", "id": "c", "end": {"close": "tool_call"}}"#;

fn batches(text: &str) -> Vec<(u64, Option<u64>)> {
    let value: Value = serde_json::from_str(text).expect("json");
    value["header"]["deliveries"]
        .as_array()
        .expect("deliveries")
        .iter()
        .map(|delivery| {
            (
                delivery["batch"].as_u64().expect("batch"),
                delivery["kind"]["items"].as_u64(),
            )
        })
        .collect()
}

#[test]
fn an_inserted_close_counts_toward_the_batch_of_the_event_after_it() {
    let base = golden(&[DELTA, DELTA, FINAL], &[], &[(1, 2), (2, 1)]);
    // The regenerated run batched differently; the change is one close.
    let head = golden(&[DELTA, DELTA, END, FINAL], &[], &[(1, 1), (2, 3)]);
    let rebased = rebase_deliveries(&base, &head)
        .expect("fits")
        .expect("something to rebase");
    assert_eq!(
        batches(&rebased),
        vec![(1, Some(2)), (2, Some(2)), (99, None)],
        "the base's batches, the one delivering the final grown by the close"
    );
    let value: Value = serde_json::from_str(&rebased).expect("json");
    assert_eq!(
        value["records"][0]["events"].as_array().map(Vec::len),
        Some(4)
    );
}

#[test]
fn a_close_at_the_end_counts_toward_the_last_batch() {
    let base = golden(&[DELTA, DELTA], &[], &[(1, 1), (2, 1)]);
    let head = golden(&[DELTA, DELTA, END], &[], &[(1, 3)]);
    let rebased = rebase_deliveries(&base, &head)
        .expect("fits")
        .expect("rebased");
    assert_eq!(
        batches(&rebased),
        vec![(1, Some(1)), (2, Some(2)), (99, None)]
    );
}

#[test]
fn a_close_before_a_trailing_error_counts_toward_the_errors_batch() {
    // Items: delta, delta, error at 2 | the change closes the text before it.
    let base = golden(&[DELTA, DELTA], &[2], &[(1, 2), (2, 1)]);
    let head = golden(&[DELTA, DELTA, END], &[3], &[(1, 1), (2, 3)]);
    let rebased = rebase_deliveries(&base, &head)
        .expect("fits")
        .expect("rebased");
    assert_eq!(
        batches(&rebased),
        vec![(1, Some(2)), (2, Some(2)), (99, None)]
    );
}

#[test]
fn an_end_that_gained_its_block_is_the_same_event() {
    let base = golden(&[DELTA, OPEN_END, FINAL], &[], &[(1, 3)]);
    let head = golden(&[DELTA, END, FINAL], &[], &[(1, 1), (2, 2)]);
    let rebased = rebase_deliveries(&base, &head)
        .expect("fits")
        .expect("rebased");
    assert_eq!(batches(&rebased), vec![(1, Some(3)), (99, None)]);
}

#[test]
fn a_cancelled_streams_undelivered_tail_takes_no_close() {
    // The base delivered one item of three before the stream was cancelled.
    let base = golden(&[DELTA, DELTA, FINAL], &[], &[(1, 1)]);
    let head = golden(&[DELTA, DELTA, END, FINAL], &[], &[(1, 1)]);
    assert_eq!(
        rebase_deliveries(&base, &head).expect("fits"),
        None,
        "the base's batches already fit: nothing to rewrite"
    );
}

#[test]
fn a_change_that_is_not_an_inserted_close_does_not_fit() {
    let base = golden(&[DELTA, FINAL], &[], &[(1, 2)]);
    let dropped = golden(&[FINAL], &[], &[(1, 1)]);
    assert!(rebase_deliveries(&base, &dropped).is_err());
    let tool = golden(&[DELTA, TOOL, FINAL], &[], &[(1, 3)]);
    assert!(rebase_deliveries(&base, &tool).is_err());
}

#[test]
fn a_golden_without_deliveries_has_nothing_to_rebase() {
    let plain = r#"{"header": {}, "records": []}"#;
    assert_eq!(rebase_deliveries(plain, plain).expect("fits"), None);
}

#[test]
fn the_rebased_text_keeps_the_rest_of_the_regenerated_golden() {
    let base = golden(&[DELTA, FINAL], &[], &[(1, 2)]);
    let head = golden(&[DELTA, END, FINAL], &[], &[(1, 1), (2, 2)]);
    let rebased = rebase_deliveries(&base, &head)
        .expect("fits")
        .expect("rebased");
    let deliveries_end = rebased.find("], \"stream_errors\"").expect("header shape");
    assert_eq!(
        &rebased[deliveries_end..],
        &head[head.find("], \"stream_errors\"").expect("header shape")..],
        "everything after the deliveries is the regenerated text"
    );
}
