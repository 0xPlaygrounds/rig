use super::*;
use serde_json::json;

fn text_end(part: u32, text: &str) -> serde_json::Value {
    json!({"item": "event", "value": {"event": "end", "part": part, "content": {"type": "text", "text": text}}})
}

#[test]
fn a_writer_sequence_round_trips_through_serde() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": "hi"}},
        {"item": "unknown", "value": {"type": "web_search_call"}},
        text_end(0, "hi"),
    ]);
    let transcript = Transcript::parse(value.clone()).expect("a writer sequence");
    assert_eq!(transcript.len(), 4);
    assert_eq!(
        serde_json::to_value(&transcript).expect("serializes"),
        value
    );
    let back: Transcript = serde_json::from_value(value).expect("deserializes");
    assert_eq!(back, transcript);
}

#[test]
fn an_event_for_a_part_that_never_started_is_refused() {
    let value = json!([{"item": "event", "value": {"event": "text", "part": 0, "text": "hi"}}]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::UnknownPart(0)));
}

#[test]
fn a_start_that_reuses_a_position_is_refused() {
    let start = json!({"item": "event", "value": {"event": "start", "part": 1, "kind": "text"}});
    let value = json!([start.clone(), start]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::UnknownPart(1)));
}

#[test]
fn a_part_that_grows_after_its_end_is_refused() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        text_end(0, ""),
        {"item": "event", "value": {"event": "text", "part": 0, "text": "late"}},
    ]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::EndedTwice(2)));
}

#[test]
fn a_fragment_of_another_kind_is_refused() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "reasoning", "part": 0, "text": "no"}},
    ]);
    assert_eq!(Transcript::parse(value), Err(SequenceError::WrongKind(1)));
}

#[test]
fn an_open_part_is_a_prefix_but_not_a_transcript() {
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": "cut"}},
    ]);
    assert_eq!(
        Transcript::parse(value.clone()),
        Err(SequenceError::Unclosed(0))
    );
    let prefix = Transcript::parse_prefix(value).expect("a stream cut short");
    assert_eq!(prefix.events().count(), 2);
}

/// A tool call's start names its tool, and no other part's start names one.
#[test]
fn only_a_tool_call_start_names_a_tool() {
    let start = |kind: &str, name: Option<&str>| {
        let mut value = json!({"event": "start", "part": 0, "kind": kind});
        if let Some(name) = name {
            value["name"] = json!(name);
        }
        json!([{"item": "event", "value": value}])
    };
    assert!(Transcript::parse_prefix(start("tool_call", Some("add"))).is_ok());
    assert_eq!(
        Transcript::parse_prefix(start("tool_call", None)).err(),
        Some(SequenceError::WrongKind(0))
    );
    assert_eq!(
        Transcript::parse_prefix(start("text", Some("add"))).err(),
        Some(SequenceError::WrongKind(0))
    );
}
