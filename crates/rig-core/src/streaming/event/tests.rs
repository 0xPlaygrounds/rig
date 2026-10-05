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

#[test]
fn a_part_reads_back_from_its_serialized_form() {
    let part: Part = serde_json::from_value(json!(7)).expect("a part");
    assert_eq!(part.index(), 7);
    assert_eq!(serde_json::to_value(part).expect("serializes"), json!(7));
}

#[test]
fn a_tool_call_start_names_its_tool_and_no_other_start_does() {
    let call = json!({"event": "start", "part": 0, "kind": "tool_call", "name": "add"});
    let text = json!({"event": "start", "part": 1, "kind": "text"});
    let event: StreamEvent = serde_json::from_value(call.clone()).expect("a call start");
    assert!(matches!(
        &event,
        StreamEvent::Start { kind: PartKind::ToolCall, name: Some(name), .. } if name.as_str() == "add"
    ));
    assert_eq!(serde_json::to_value(&event).expect("serializes"), call);
    let event: StreamEvent = serde_json::from_value(text.clone()).expect("a text start");
    assert!(matches!(event, StreamEvent::Start { name: None, .. }));
    assert_eq!(serde_json::to_value(&event).expect("serializes"), text);
}

#[test]
fn an_item_reads_back_without_its_order_being_checked() {
    let late = json!({"item": "event", "value": {"event": "text", "part": 3, "text": "late"}});
    assert_eq!(
        Transcript::parse(json!([late.clone()])),
        Err(SequenceError::UnknownPart(0))
    );
    let item: Item<StreamEvent> = serde_json::from_value(late.clone()).expect("an item");
    assert_eq!(serde_json::to_value(&item).expect("serializes"), late);
}

#[test]
fn a_misshapen_item_is_refused() {
    let value = json!({"item": "event", "value": {"event": "text", "part": -1, "text": "x"}});
    assert!(serde_json::from_value::<Item<StreamEvent>>(value).is_err());
}

/// An image part reads back, checks against its kind, and hands its items
/// back in order; each event names its variant without its payload.
#[test]
fn an_image_part_reads_back_and_names_its_events() {
    use crate::message::{DocumentSourceKind, Image, ImageMediaType};

    let image = AssistantContent::Image(Image {
        data: DocumentSourceKind::base64("aGk="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    });
    let end = json!({"item": "event", "value": {
        "event": "end",
        "part": 1,
        "content": serde_json::to_value(&image).expect("an image serializes"),
    }});
    let value = json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "reasoning"}},
        {"item": "event", "value": {"event": "reasoning", "part": 0, "text": "hm"}},
        {"item": "event", "value": {"event": "start", "part": 1, "kind": "image"}},
        end,
    ]);
    let transcript = Transcript::parse_prefix(value.clone()).expect("an image part");
    assert_eq!(
        transcript
            .events()
            .map(StreamEvent::name)
            .collect::<Vec<_>>(),
        ["Start", "Reasoning", "Start", "End"]
    );
    assert!(matches!(
        transcript.events().last(),
        Some(StreamEvent::End {
            content: AssistantContent::Image(_),
            ..
        })
    ));
    let items: Vec<Item<StreamEvent>> = transcript.into();
    assert_eq!(serde_json::to_value(&items).expect("serializes"), value);
}
