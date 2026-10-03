use serde_json::{Value, json};

use super::{Block, CallFragment, Completion, Finish, Turn, events_of};
use crate::completion::CompletionResponse;
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CallId, Opaque, Reasoning, Text, ToolCall, ToolFunction, ToolName,
};
use crate::streaming::{Item, StreamEvent, Transcript};
use crate::wire::Out;

/// One reply written by `write`, then ended.
fn write(
    write: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    decode_with(Turn::relayed("test"), "test", |reply| {
        let mut out = reply.out();
        write(&mut out)?;
        let _ = out.end(Finish::default());
        Ok(())
    })
}

fn response(decoded: Decoded<Completion>) -> CompletionResponse {
    decoded.outcome.expect("the reply ends")
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn object(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

#[test]
fn blocks_keep_the_order_their_items_opened_in() {
    let decoded = write(|out| {
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.open(1, Block::Text, Value::Null)?;
        out.open(
            2,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: name("lookup"),
            },
            Value::Null,
        )?;
        out.push(2, "{}")?;
        // The call finishes first, the reasoning last: neither moves.
        out.finish(2)?;
        out.push(1, "answer")?;
        out.finish(1)?;
        out.push(0, "think")?;
        out.finish(0)
    });
    let kinds: Vec<_> = response(decoded)
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "call",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["reasoning", "text", "call"]);
}

#[test]
fn deltas_merge_into_the_item_signatures_concatenate_and_unknown_fields_survive() {
    let decoded = write(|out| {
        out.open(
            7,
            Block::Reasoning { redacted: false },
            json!({"type": "thinking", "thinking": "", "signature": ""}),
        )?;
        for delta in [
            json!({"type": "thinking_delta", "thinking": "hm"}),
            json!({"type": "signature_delta", "signature": "abc"}),
            json!({"type": "signature_delta", "signature": "def"}),
            json!({"type": "novel_delta", "novel": "x", "list": [1]}),
            json!({"type": "novel_delta", "novel": "y", "list": [2]}),
        ] {
            out.merge(7, &object(delta))?;
        }
        out.push(7, "hm")?;
        out.finish(7)
    });
    let block = response(decoded).choice.remove(0);
    assert_eq!(
        block.native_item(),
        Some(&json!({
            "type": "thinking",
            "thinking": "hm",
            "signature": "abcdef",
            "novel": "xy",
            "list": [1, 2],
        }))
    );
    assert_eq!(
        block.canonical(),
        AssistantContent::Reasoning(Reasoning::new("hm"))
    );
}

#[test]
fn empty_text_survives_only_with_a_provider_item() {
    let decoded = write(|out| {
        out.open(0, Block::Text, Value::Null)?;
        out.finish(0)?;
        out.open(1, Block::Text, json!({"thoughtSignature": "c2ln"}))?;
        out.finish(1)?;
        out.open(2, Block::Reasoning { redacted: true }, Value::Null)?;
        out.finish(2)
    });
    let choice = response(decoded).choice;
    assert_eq!(choice.len(), 2);
    assert!(
        matches!(&choice[0], AssistantContent::Text(text) if text.text.is_empty() && text.native.is_some())
    );
    assert!(matches!(&choice[1], AssistantContent::Reasoning(reasoning) if reasoning.redacted));
}

#[test]
fn an_opaque_item_closes_as_its_assembled_item() {
    let decoded = write(|out| {
        out.open(
            0,
            Block::Opaque { replay: true },
            json!({"type": "compaction", "content": ""}),
        )?;
        out.merge(
            0,
            &object(json!({"type": "compaction_delta", "content": "summary"})),
        )?;
        out.finish(0)
    });
    assert_eq!(
        response(decoded).choice,
        vec![AssistantContent::Opaque(Opaque {
            item: json!({"type": "compaction", "content": "summary"}),
            replay: true,
        })]
    );
}

#[test]
fn a_buffered_call_opens_at_its_first_fragment_and_closes_whole() {
    let decoded = write(|out| {
        out.fragment(
            0,
            CallFragment {
                arguments: Some("{\"q\":"),
                ..CallFragment::default()
            },
        )?;
        out.run(Block::Text, "between")?;
        out.fragment(
            0,
            CallFragment {
                id: Some("call_1"),
                name: Some("lookup"),
                arguments: Some("1}"),
            },
        )?;
        out.end_run()?;
        out.finish(0)
    });
    let choice = response(decoded).choice;
    assert_eq!(
        choice,
        vec![
            AssistantContent::ToolCall(ToolCall::new(
                CallId::from_wire("call_1"),
                ToolFunction::new(name("lookup"), json!({"q": 1})),
            )),
            AssistantContent::Text(Text::new("between")),
        ]
    );
}

#[test]
fn a_call_that_never_got_an_id_is_issued_one() {
    let closed = write(|out| {
        out.fragment(
            3,
            CallFragment {
                name: Some("lookup"),
                arguments: Some("{}"),
                ..CallFragment::default()
            },
        )?;
        out.finish(3)
    });
    assert!(matches!(
        response(closed).choice.as_slice(),
        [AssistantContent::ToolCall(call)] if call.id.is_local()
    ));

    let ended = write(|out| {
        out.fragment(
            3,
            CallFragment {
                name: Some("lookup"),
                arguments: Some("{}"),
                ..CallFragment::default()
            },
        )
    });
    assert!(matches!(
        response(ended).choice.as_slice(),
        [AssistantContent::ToolCall(call)] if call.id.is_local()
    ));
}

#[test]
fn a_reused_call_id_is_renamed_and_loses_its_item() {
    let decoded = write(|out| {
        for index in 0..2 {
            out.open(
                index,
                Block::Call {
                    id: CallId::from_wire("call_1"),
                    name: name("lookup"),
                },
                json!({"type": "function_call", "call_id": "call_1"}),
            )?;
            out.push(index, "{}")?;
            out.finish(index)?;
        }
        Ok(())
    });
    let calls: Vec<ToolCall> = response(decoded).tool_calls().cloned().collect();
    let [first, second] = calls.as_slice() else {
        panic!("both calls are kept: {calls:?}");
    };
    assert_eq!(first.id, CallId::from_wire("call_1"));
    assert!(first.native.is_some());
    assert!(matches!(second.id, CallId::Local(_)), "{:?}", second.id);
    assert!(second.native.is_none());
}

#[test]
fn writing_to_an_item_that_is_not_open_is_an_error() {
    let decoded = write(|out| out.push(9, "orphan"));
    assert!(matches!(decoded.outcome, Err(ProviderError::Response(_))));
}

#[test]
fn a_boundaryless_run_continues_while_the_kind_stays_the_same() {
    let decoded = write(|out| {
        out.run(Block::Text, "a")?;
        out.run(Block::Text, "b")?;
        out.run(Block::Reasoning { redacted: false }, "r")?;
        out.run(Block::Text, "c")?;
        Ok(())
    });
    assert_eq!(
        response(decoded).choice,
        vec![
            AssistantContent::text("ab"),
            AssistantContent::reasoning("r"),
            AssistantContent::text("c"),
        ]
    );
}

#[test]
fn a_restated_response_streams_the_same_blocks() {
    let decoded = write(|out| {
        out.open(0, Block::Text, json!({"type": "text", "text": ""}))?;
        out.merge(0, &object(json!({"text": "hi"})))?;
        out.push(0, "hi")?;
        out.finish(0)?;
        out.open(
            1,
            Block::Opaque { replay: false },
            json!({"type": "computer_call"}),
        )?;
        out.finish(1)
    });
    let original = response(decoded);
    let events = events_of(&original).expect("restates");
    let transcript = Transcript::from_items(events.clone()).expect("a valid sequence");
    let ended: Vec<_> = transcript
        .events()
        .filter_map(|event| match event {
            StreamEvent::End { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ended, original.choice);
    assert!(events.iter().all(|item| matches!(item, Item::Event(_))));
}

#[test]
fn starts_follow_item_order_and_a_dropped_item_leaves_a_gap() {
    let decoded = write(|out| {
        out.open(0, Block::Text, Value::Null)?;
        out.open(1, Block::Text, Value::Null)?;
        out.push(1, "second")?;
        out.finish(0)?;
        out.finish(1)
    });
    let items: Vec<_> = decoded.items.iter().flatten().cloned().collect();
    let transcript = Transcript::from_items(items).expect("a gap is a valid sequence");
    let parts: Vec<_> = transcript
        .events()
        .map(|event| event.part().index())
        .collect();
    assert!(parts.iter().all(|part| *part == 1));
    assert_eq!(
        response(decoded).choice,
        vec![AssistantContent::text("second")]
    );
}

#[test]
fn a_call_closed_without_a_name_is_dropped() {
    let decoded = write(|out| {
        out.fragment(
            0,
            CallFragment {
                id: Some("call_1"),
                arguments: Some("{}"),
                ..CallFragment::default()
            },
        )?;
        out.finish(0)
    });
    assert!(response(decoded).choice.is_empty());
}
