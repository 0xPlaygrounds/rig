use rig_harness::rig_core::completion::Usage;
use serde_json::json;

use super::{Reader, user_message};
use crate::harness::{Event, Part};

fn read(reader: &mut Reader, line: serde_json::Value) -> Vec<Event> {
    reader.read(&line.to_string())
}

fn stream(event: serde_json::Value) -> serde_json::Value {
    json!({ "type": "stream_event", "event": event, "session_id": "s" })
}

fn usage(input: u64, output: u64, read: u64, written: u64, reasoning: u64) -> Usage {
    let mut usage = Usage::new();
    usage.input_tokens = Some(input);
    usage.output_tokens = Some(output);
    usage.total_tokens = Some(input + output);
    usage.cached_input_tokens = Some(read);
    usage.cache_creation_input_tokens = Some(written);
    usage.reasoning_tokens = Some(reasoning);
    usage
}

#[test]
fn a_streamed_message_is_its_deltas_then_one_usage() {
    let mut reader = Reader::default();
    let lines = [
        json!({ "type": "system", "subtype": "init", "tools": ["mcp__rig__read"] }),
        stream(
            json!({ "type": "message_start", "message": { "id": "msg_1", "usage": {
            "input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
        } } }),
        ),
        stream(
            json!({ "type": "content_block_delta", "delta": { "type": "thinking_delta", "thinking": "hm" } }),
        ),
        stream(
            json!({ "type": "content_block_delta", "delta": { "type": "signature_delta", "signature": "x" } }),
        ),
        stream(
            json!({ "type": "content_block_delta", "delta": { "type": "text_delta", "text": "Hel" } }),
        ),
        stream(
            json!({ "type": "content_block_delta", "delta": { "type": "text_delta", "text": "lo" } }),
        ),
        stream(
            json!({ "type": "content_block_delta", "delta": { "type": "input_json_delta", "partial_json": "{" } }),
        ),
        // The repeat of a streamed message adds nothing.
        json!({ "type": "assistant", "message": { "id": "msg_1", "model": "claude", "content": [
            { "type": "text", "text": "Hello" },
        ] } }),
        stream(
            json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": {
            "output_tokens": 40, "output_tokens_details": { "thinking_tokens": 7 },
        } }),
        ),
        json!({ "type": "rate_limit_event", "rate_limit_info": { "status": "allowed" } }),
    ];

    let events: Vec<Event> = lines
        .into_iter()
        .flat_map(|line| read(&mut reader, line))
        .collect();

    assert_eq!(
        events,
        vec![
            Event::Began,
            Event::Reasoning("hm".to_owned()),
            Event::Text("Hel".to_owned()),
            Event::Text("lo".to_owned()),
            Event::Usage(usage(1102, 40, 1000, 100, 7)),
        ],
    );
}

#[test]
fn a_message_that_was_not_streamed_is_read_whole() {
    let mut reader = Reader::default();
    let whole = json!({ "type": "assistant", "message": { "id": "msg_2", "model": "claude",
        "content": [
            { "type": "thinking", "thinking": "plan" },
            { "type": "text", "text": "Done." },
            { "type": "tool_use", "id": "toolu_1", "name": "mcp__rig__read", "input": {} },
        ],
        "usage": { "input_tokens": 5, "output_tokens": 9 },
    } });

    let first = read(&mut reader, whole.clone());
    let again = read(&mut reader, whole);

    let mut counted = Usage::new();
    counted.input_tokens = Some(5);
    counted.output_tokens = Some(9);
    counted.total_tokens = Some(14);
    assert_eq!(
        first,
        vec![
            Event::Reasoning("plan".to_owned()),
            Event::Text("Done.".to_owned()),
            Event::Usage(counted),
        ],
    );
    assert_eq!(again, Vec::new());
}

#[test]
fn a_result_ends_the_prompt_and_an_error_result_says_why() {
    let mut reader = Reader::default();

    let answered = read(
        &mut reader,
        json!({ "type": "result", "subtype": "success", "is_error": false, "result": "ok" }),
    );
    let limited = read(
        &mut reader,
        json!({ "type": "result", "subtype": "success", "is_error": true, "result": "You are out of usage." }),
    );
    let failed = read(
        &mut reader,
        json!({ "type": "result", "subtype": "error_during_execution", "is_error": true, "errors": ["a", "b"] }),
    );
    // Claude Code's own notice is not the model's text: the result carries it.
    let notice = read(
        &mut reader,
        json!({ "type": "assistant", "message": { "id": "n", "model": "<synthetic>",
        "content": [{ "type": "text", "text": "You are out of usage." }] } }),
    );

    assert_eq!(answered, vec![Event::Done(Ok(()))]);
    assert_eq!(
        limited,
        vec![Event::Done(Err("You are out of usage.".to_owned()))]
    );
    assert_eq!(failed, vec![Event::Done(Err("a; b".to_owned()))]);
    assert_eq!(notice, Vec::new());
}

#[test]
fn a_line_that_is_not_a_known_message_says_nothing() {
    let mut reader = Reader::default();

    assert_eq!(reader.read("not json"), Vec::new());
    assert_eq!(
        read(&mut reader, json!({ "type": "something_new", "x": 1 })),
        Vec::new()
    );
    assert_eq!(
        read(
            &mut reader,
            stream(json!({ "type": "content_block_stop", "index": 0 }))
        ),
        Vec::new()
    );
}

#[test]
fn a_prompt_is_one_user_message_of_text_and_images() {
    let prompt = [
        Part::Text("look".to_owned()),
        Part::Image {
            media_type: "image/png".to_owned(),
            data: "AAAA".to_owned(),
        },
    ];

    assert_eq!(
        user_message(&prompt),
        json!({ "type": "user", "message": { "role": "user", "content": [
            { "type": "text", "text": "look" },
            { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AAAA" } },
        ] } }),
    );
}
