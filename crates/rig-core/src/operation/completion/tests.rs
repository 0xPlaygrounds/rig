use serde_json::{Value, json};

use super::{Block, CallFragment, Completion, Finish, Turn, events_of};
use crate::completion::{CompletionResponse, Cost, Usage};
use crate::driver::{Decoded, decode_with};
use crate::error::ProviderError;
use crate::message::{
    AssistantContent, CallId, Reasoning, Source, SourceLocation, ToolCall, ToolName,
};
use crate::streaming::{Item, StreamEvent, Transcript};
use crate::wire::{Out, SpanUnit, WireCitation, WireSpan};

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
            out.edit(7, |item| super::merge(item, &object(delta)))?;
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
        out.edit(0, |item| super::merge(item, &object(json!({"text": "hi"}))))?;
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

/// One reply written by `write` on a wire fold, which states finish reasons
/// and whose call items hold their id at `slot`, then ended with `finish`.
fn write_on_wire(
    slot: Option<&'static str>,
    finish: Finish,
    write: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> Decoded<Completion> {
    let turn = Turn {
        wire: true,
        call_id_slot: slot,
        ..Turn::relayed("test")
    };
    decode_with(turn, "test", |reply| {
        let mut out = reply.out();
        write(&mut out)?;
        let _ = out.end(finish);
        Ok(())
    })
}

fn stop(reason: crate::completion::FinishReason) -> Finish {
    Finish {
        reason: Some(reason),
        ..Finish::default()
    }
}

#[test]
fn a_wire_reply_that_names_no_finish_reason_failed() {
    let decoded = write_on_wire(None, Finish::default(), |out| {
        out.whole(0, Block::Text, json!({"type": "text"}), "hi")
    });
    assert!(response(decoded).stop().is_failure());
}

#[test]
fn blocks_from_the_first_unfinished_item_on_replay_canonically() {
    use crate::completion::FinishReason;
    let decoded = write_on_wire(None, stop(FinishReason::Length), |out| {
        out.whole(
            0,
            Block::Reasoning { redacted: false },
            json!({"sig": "a"}),
            "plan",
        )?;
        out.open(1, Block::Reasoning { redacted: false }, json!({"sig": "b"}))?;
        out.push(1, "more")?;
        out.whole(2, Block::Text, json!({"id": "msg_1"}), "answer")
    });
    let choice = response(decoded).choice;
    let natives: Vec<bool> = choice
        .iter()
        .map(|block| block.native_item().is_some())
        .collect();
    assert_eq!(natives, [true, false, false], "{choice:?}");
}

#[test]
fn a_call_sent_without_an_id_keeps_its_item_only_where_the_item_has_an_id_slot() {
    use crate::completion::FinishReason;
    let idless = |out: &mut Out<'_, Completion>| {
        out.fragment(
            Some(0),
            CallFragment {
                id: None,
                name: Some("lookup"),
                arguments: Some("{}"),
            },
        )?;
        out.edit(0, |item| {
            *item = json!({"functionCall": {"name": "lookup"}})
        })?;
        out.finish(0)
    };
    let slotted = response(write_on_wire(
        Some("/functionCall/id"),
        stop(FinishReason::ToolCalls),
        idless,
    ));
    assert!(slotted.choice[0].native_item().is_some());
    let unslotted = response(write_on_wire(None, stop(FinishReason::ToolCalls), idless));
    assert!(unslotted.choice[0].native_item().is_none());
}

#[test]
fn index_less_fragments_with_new_ids_are_new_calls() {
    let fragment = |id: Option<&'static str>, arguments: &'static str| CallFragment {
        id,
        name: id.map(|_| "weather"),
        arguments: Some(arguments),
    };
    let decoded = write(|out| {
        out.fragment(None, fragment(Some("a1"), r#"{"city":"#))?;
        out.fragment(None, fragment(None, r#""Paris"}"#))?;
        out.fragment(None, fragment(Some("b2"), r#"{"city":"Rome"}"#))?;
        out.fragment(Some(0), fragment(Some("c3"), r#"{"city":"Oslo"}"#))?;
        out.fragment(Some(0), fragment(Some("d4"), r#"{"city":"Bern"}"#))?;
        Ok(())
    });
    let calls: Vec<(String, Value)> = response(decoded)
        .tool_calls()
        .map(|call| (call.id.wire().into_owned(), call.function.arguments_value()))
        .collect();
    assert_eq!(
        calls,
        [
            ("a1".to_owned(), json!({"city": "Paris"})),
            ("b2".to_owned(), json!({"city": "Rome"})),
            ("c3".to_owned(), json!({"city": "Oslo"})),
            ("d4".to_owned(), json!({"city": "Bern"})),
        ]
    );
}

#[test]
fn a_new_id_under_an_index_continues_a_call_whose_arguments_are_incomplete() {
    // GLM sends a fresh id, and no name, with every later chunk of one call.
    let decoded = write(|out| {
        for (id, name, arguments) in [("x1", "lookup", r#"{"q":"#), ("x2", "", r#""x"}"#)] {
            out.fragment(
                Some(0),
                CallFragment {
                    id: Some(id),
                    name: Some(name),
                    arguments: Some(arguments),
                },
            )?;
        }
        Ok(())
    });
    let calls: Vec<Value> = response(decoded)
        .tool_calls()
        .map(|call| call.function.arguments_value())
        .collect();
    assert_eq!(calls, [json!({"q": "x"})]);
}

#[test]
fn index_less_id_less_fragments_open_a_call_once_the_last_is_whole() {
    let decoded = write(|out| {
        for arguments in [r#"{"q":"a"}"#, r#"{"q":"#, r#""b"}"#] {
            out.fragment(
                None,
                CallFragment {
                    id: None,
                    name: Some("weather"),
                    arguments: Some(arguments),
                },
            )?;
        }
        Ok(())
    });
    let calls: Vec<Value> = response(decoded)
        .tool_calls()
        .map(|call| call.function.arguments_value())
        .collect();
    assert_eq!(calls, [json!({"q": "a"}), json!({"q": "b"})]);
}

#[test]
fn an_empty_or_null_id_names_no_call() {
    for stated in ["", "null"] {
        let decoded = write(|out| {
            out.fragment(
                Some(0),
                CallFragment {
                    id: Some(stated),
                    name: Some("lookup"),
                    arguments: Some("{}"),
                },
            )
        });
        let response = response(decoded);
        let call = response.tool_calls().next().expect("a call");
        assert!(call.id.provider().is_none(), "{:?}", call.id);
    }
}

/// The events a decoded reply carried, in order, checked as a stream that
/// may have stopped early.
fn events(decoded: &Decoded<Completion>) -> Vec<StreamEvent> {
    let items: Vec<Item<StreamEvent>> = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect();
    let transcript = Transcript::from_items(items).expect("a valid sequence");
    transcript.events().cloned().collect()
}

/// A compact view of `events`: `start:<name>`, `args:<json>`, `text:<text>`
/// and `end`, each with its part.
fn shape(events: &[StreamEvent]) -> Vec<String> {
    events
        .iter()
        .map(|event| match event {
            StreamEvent::Start { part, name, .. } => format!(
                "{}:start:{}",
                part.index(),
                name.as_ref().map_or("", ToolName::as_str)
            ),
            StreamEvent::Arguments { part, json } => format!("{}:args:{json}", part.index()),
            StreamEvent::Text { part, text } => format!("{}:text:{text}", part.index()),
            StreamEvent::Reasoning { part, text } => format!("{}:reasoning:{text}", part.index()),
            StreamEvent::End { part, .. } => format!("{}:end", part.index()),
        })
        .collect()
}

/// The joined argument fragments of the call at `part`, and the arguments
/// its end states.
fn streamed_and_final(events: &[StreamEvent], part: usize) -> (String, Value) {
    let streamed = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Arguments { part: at, json } if at.index() == part => Some(json.as_str()),
            _ => None,
        })
        .collect();
    let ended = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::End {
                part: at,
                content: AssistantContent::ToolCall(call),
            } if at.index() == part => Some(call.function.arguments_value()),
            _ => None,
        })
        .expect("the call ended");
    (streamed, ended)
}

#[test]
fn a_named_call_starts_when_it_opens_and_streams_each_fragment() {
    let decoded = write(|out| {
        out.open(
            0,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: name("lookup"),
            },
            Value::Null,
        )?;
        out.push(0, r#"{"city":"#)?;
        out.open(1, Block::Text, Value::Null)?;
        out.push(1, "checking")?;
        out.push(0, r#""Paris"}"#)?;
        out.finish(0)?;
        out.finish(1)
    });
    let events = events(&decoded);
    assert_eq!(
        shape(&events),
        [
            "0:start:lookup",
            r#"0:args:{"city":"#,
            "1:start:",
            "1:text:checking",
            r#"0:args:"Paris"}"#,
            "0:end",
            "1:end",
        ]
    );
    let (streamed, ended) = streamed_and_final(&events, 0);
    assert_eq!(serde_json::from_str::<Value>(&streamed).ok(), Some(ended));
}

#[test]
fn fragments_before_the_name_wait_for_it_then_stream() {
    let decoded = write(|out| {
        let fragment = |name, arguments| CallFragment {
            id: Some("call_1"),
            name,
            arguments,
        };
        out.fragment(Some(0), fragment(None, Some(r#"{"x":"#)))?;
        out.fragment(Some(0), fragment(Some("add"), None))?;
        out.fragment(Some(0), fragment(None, Some("1,")))?;
        out.fragment(Some(0), fragment(None, Some(r#""y":2}"#)))?;
        Ok(())
    });
    let events = events(&decoded);
    assert_eq!(
        shape(&events),
        [
            "0:start:add",
            r#"0:args:{"x":"#,
            "0:args:1,",
            r#"0:args:"y":2}"#,
            "0:end",
        ]
    );
    let (streamed, ended) = streamed_and_final(&events, 0);
    assert_eq!(streamed, r#"{"x":1,"y":2}"#);
    assert_eq!(ended, json!({"x": 1, "y": 2}));
}

#[test]
fn parallel_calls_interleave_under_their_own_parts() {
    let decoded = write(|out| {
        let fragment = |id, name, arguments| CallFragment {
            id: Some(id),
            name: Some(name),
            arguments: Some(arguments),
        };
        out.fragment(Some(0), fragment("a", "add", r#"{"x":"#))?;
        out.fragment(Some(1), fragment("b", "weather", r#"{"city":"#))?;
        out.fragment(Some(0), fragment("a", "add", "1}"))?;
        out.fragment(Some(1), fragment("b", "weather", r#""Rome"}"#))?;
        Ok(())
    });
    let events = events(&decoded);
    assert_eq!(
        shape(&events),
        [
            "0:start:add",
            r#"0:args:{"x":"#,
            "1:start:weather",
            r#"1:args:{"city":"#,
            "0:args:1}",
            r#"1:args:"Rome"}"#,
            "0:end",
            "1:end",
        ]
    );
    for (part, arguments) in [(0, json!({"x": 1})), (1, json!({"city": "Rome"}))] {
        let (streamed, ended) = streamed_and_final(&events, part);
        assert_eq!(ended, arguments);
        assert_eq!(
            serde_json::from_str::<Value>(&streamed).ok(),
            Some(arguments)
        );
    }
}

#[test]
fn a_call_sent_whole_starts_streams_once_and_ends() {
    let decoded = write(|out| {
        out.whole(
            0,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: name("add"),
            },
            Value::Null,
            r#"{"x":1}"#,
        )?;
        out.open(
            1,
            Block::Call {
                id: CallId::from_wire("call_2"),
                name: name("noop"),
            },
            Value::Null,
        )?;
        out.finish(1)
    });
    assert_eq!(
        shape(&events(&decoded)),
        [
            "0:start:add",
            r#"0:args:{"x":1}"#,
            "0:end",
            "1:start:noop",
            "1:args:{}",
            "1:end",
        ]
    );
}

#[test]
fn a_call_that_never_names_a_tool_streams_nothing() {
    let decoded = write(|out| {
        out.fragment(
            Some(0),
            CallFragment {
                id: Some("call_1"),
                name: None,
                arguments: Some(r#"{"x":1}"#),
            },
        )?;
        out.open(1, Block::Text, Value::Null)?;
        out.push(1, "done")?;
        out.finish(1)
    });
    assert_eq!(
        shape(&events(&decoded)),
        ["1:start:", "1:text:done", "1:end"]
    );
}

#[test]
fn a_placeholder_null_waits_for_the_real_arguments() {
    let decoded = write(|out| {
        let fragment = |arguments| CallFragment {
            id: Some("call_1"),
            name: Some("add"),
            arguments: Some(arguments),
        };
        out.fragment(Some(0), fragment("null"))?;
        out.fragment(Some(0), fragment(r#"{"x":"#))?;
        out.fragment(Some(0), fragment("1}"))?;
        Ok(())
    });
    let events = events(&decoded);
    assert_eq!(
        shape(&events),
        ["0:start:add", r#"0:args:{"x":"#, "0:args:1}", "0:end"]
    );
    let (streamed, ended) = streamed_and_final(&events, 0);
    assert_eq!(streamed, r#"{"x":1}"#);
    assert_eq!(ended, json!({"x": 1}));
}

#[test]
fn a_reply_that_fails_mid_call_never_ends_the_call() {
    let decoded = decode_with(Turn::relayed("test"), "test", |reply| {
        let mut out = reply.out();
        out.fragment(
            Some(0),
            CallFragment {
                id: Some("call_1"),
                name: Some("add"),
                arguments: Some(r#"{"x":"#),
            },
        )?;
        Err(ProviderError::Response("the connection dropped".to_owned()))
    });
    assert_eq!(shape(&events(&decoded)), ["0:start:add", r#"0:args:{"x":"#]);
    assert!(decoded.outcome.is_err());
}

#[test]
fn a_restated_call_starts_with_its_name() {
    let call = ToolCall::from_wire(
        "call_1",
        crate::message::ToolFunction::new(name("add"), json!({"x": 1})),
    );
    let response = CompletionResponse::new(
        vec![AssistantContent::ToolCall(call)],
        crate::completion::Usage::default(),
        crate::message::Origin::new("test", "test", "test"),
        Value::Null,
    );
    let items = events_of(&response).expect("restates");
    let transcript = Transcript::from_items(items).expect("a valid sequence");
    let events: Vec<StreamEvent> = transcript.events().cloned().collect();
    assert_eq!(
        shape(&events),
        ["0:start:add", r#"0:args:{"x":1}"#, "0:end"]
    );
}

/// A call keeps the first name it is given, so a later fragment naming
/// another tool cannot end a call its start named differently.
#[test]
fn a_started_call_keeps_the_name_its_start_stated() {
    let decoded = write(|out| {
        let fragment = |name, arguments| CallFragment {
            id: Some("call_1"),
            name,
            arguments,
        };
        out.fragment(Some(0), fragment(Some("add"), Some(r#"{"x":1,"#)))?;
        out.fragment(Some(0), fragment(Some("multiply"), Some(r#""y":2}"#)))?;
        Ok(())
    });
    let events = events(&decoded);
    assert_eq!(
        shape(&events),
        [
            "0:start:add",
            r#"0:args:{"x":1,"#,
            r#"0:args:"y":2}"#,
            "0:end"
        ]
    );
    let ended = events.iter().find_map(|event| match event {
        StreamEvent::End {
            content: AssistantContent::ToolCall(call),
            ..
        } => Some(call.function.name.clone()),
        _ => None,
    });
    assert_eq!(ended, Some(name("add")));
}

/// A placeholder `null` no real fragment replaced streams the arguments
/// the call ends with, so the joined fragments still state them.
#[test]
fn a_placeholder_null_never_replaced_streams_the_ended_arguments() {
    let decoded = write(|out| {
        out.fragment(
            Some(0),
            CallFragment {
                id: Some("call_1"),
                name: Some("noop"),
                arguments: Some("null"),
            },
        )?;
        Ok(())
    });
    let events = events(&decoded);
    assert_eq!(shape(&events), ["0:start:noop", "0:args:{}", "0:end"]);
    let (streamed, ended) = streamed_and_final(&events, 0);
    assert_eq!(serde_json::from_str::<Value>(&streamed).ok(), Some(ended));
}

/// Restating an item with no text of its own changes nothing.
#[test]
fn restating_an_opaque_item_keeps_it() {
    let decoded = write(|out| {
        out.open(
            0,
            Block::Opaque { replay: true },
            json!({"type": "compaction", "summary": "kept"}),
        )?;
        out.restate(0, "ignored")?;
        out.finish(0)
    });
    let choice = response(decoded).choice;
    assert!(
        matches!(&choice[..], [AssistantContent::Opaque(opaque)] if opaque.item["summary"] == "kept"),
        "{choice:?}"
    );
}

#[test]
fn every_item_the_writer_emits_reads_back_on_its_own() {
    let decoded = write(|out| {
        out.open(0, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(0, "think")?;
        out.finish(0)?;
        out.open(1, Block::Text, json!({"type": "text", "text": "hi"}))?;
        out.push(1, "hi")?;
        out.unknown(json!({"type": "web_search_call", "id": "ws_1"}).into());
        out.open(
            2,
            Block::Call {
                id: CallId::from_wire("call_1"),
                name: name("write_file"),
            },
            json!({"type": "function_call", "call_id": "call_1"}),
        )?;
        out.push(2, r#"{"path":"a.md","#)?;
        out.push(2, r#""content":"line\n"}"#)?;
        out.finish(2)?;
        out.finish(1)?;
        out.open(
            3,
            Block::Opaque { replay: true },
            json!({"type": "compaction"}),
        )?;
        out.finish(3)
    });
    let items: Vec<Item<StreamEvent>> = decoded
        .items
        .iter()
        .filter_map(|item| item.as_ref().ok().cloned())
        .collect();
    assert_eq!(
        shape(
            &Transcript::from_items(items.clone())
                .expect("a valid sequence")
                .events()
                .cloned()
                .collect::<Vec<_>>()
        ),
        [
            "0:start:",
            "0:reasoning:think",
            "0:end",
            "1:start:",
            "1:text:hi",
            "2:start:write_file",
            r#"2:args:{"path":"a.md","#,
            r#"2:args:"content":"line\n"}"#,
            "2:end",
            "1:end",
            "3:start:",
            "3:end",
        ]
    );
    assert!(items.iter().any(|item| matches!(item, Item::Unknown(_))));
    for item in items {
        let value = serde_json::to_value(&item).expect("an item serializes");
        let back: Item<StreamEvent> =
            serde_json::from_value(value.clone()).expect("an item reads back");
        assert_eq!(back, item);
        assert_eq!(serde_json::to_value(&back).expect("serializes"), value);
    }
}

/// One reply from `origin` written by `write`, then ended with `end`.
fn write_from(
    origin: crate::message::Origin,
    end: Finish,
    write: impl for<'id> FnOnce(&mut Out<'id, Completion>) -> Result<(), ProviderError>,
) -> CompletionResponse {
    response(decode_with(Turn::new(origin), "test", |reply| {
        let mut out = reply.out();
        write(&mut out)?;
        let _ = out.end(end);
        Ok(())
    }))
}

fn url_source(url: &str) -> Source {
    Source::new(SourceLocation::Url {
        url: url.to_owned(),
    })
}

fn cited(response: &CompletionResponse) -> Vec<(Option<String>, Vec<Source>)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            text.citations().iter().map(|citation| {
                (
                    text.cited(citation).map(str::to_owned),
                    citation.sources.clone(),
                )
            })
        })
        .collect()
}

/// A citation that arrives before the text it covers resolves when the
/// block closes, against the whole text.
#[test]
fn a_citation_resolves_when_its_text_closes() {
    let decoded = write(|out| {
        out.open(0, Block::Text, json!({"type": "text", "text": ""}))?;
        out.push(0, "Le quai ")?;
        out.cite(
            0,
            WireCitation::new(
                Some(WireSpan::new(8, 19, SpanUnit::Chars).quoted("numéro sept")),
                vec![url_source("https://a.example")],
            ),
        );
        out.cite(
            0,
            WireCitation::new(None, vec![url_source("https://b.example")]),
        );
        out.push(0, "numéro sept est ouvert.")?;
        out.finish(0)
    });
    let response = response(decoded);
    assert_eq!(
        cited(&response),
        [
            (
                Some("numéro sept".to_owned()),
                vec![url_source("https://a.example")]
            ),
            (
                Some("Le quai numéro sept est ouvert.".to_owned()),
                vec![url_source("https://b.example")]
            ),
        ]
    );
    // The provider item stays current: citations are not in the fingerprint.
    assert!(response.choice[0].native_item().is_some());
}

/// A citation whose span does not fit, or covers other text than quoted,
/// is dropped and the reply decodes as without it.
#[test]
fn a_citation_that_does_not_resolve_never_fails_the_reply() {
    let decoded = write(|out| {
        out.open(0, Block::Text, Value::Null)?;
        out.push(0, "Dock Seven")?;
        for span in [
            WireSpan::new(0, 99, SpanUnit::Bytes),
            WireSpan::new(0, 4, SpanUnit::Bytes).quoted("Pier"),
        ] {
            out.cite(
                0,
                WireCitation::new(Some(span), vec![url_source("https://a.example")]),
            );
        }
        out.open(1, Block::Reasoning { redacted: false }, Value::Null)?;
        out.push(1, "think")?;
        out.cite(
            1,
            WireCitation::new(None, vec![url_source("https://b.example")]),
        );
        out.cite(
            7,
            WireCitation::new(None, vec![url_source("https://c.example")]),
        );
        out.finish(1)?;
        out.finish(0)
    });
    let response = response(decoded);
    assert!(cited(&response).is_empty());
    assert_eq!(response.choice.len(), 2);
}

/// A provider that cites after the text closed, or restates its citations,
/// reaches the block whether its end event is still queued or the fold
/// already holds it.
#[test]
fn late_citations_land_on_the_closed_text() {
    let decoded = write(|out| {
        out.open(0, Block::Text, Value::Null)?;
        out.push(0, "Dock Seven")?;
        out.finish(0)?;
        out.cite(
            0,
            WireCitation::new(
                Some(WireSpan::new(5, 10, SpanUnit::Utf16).quoted("Seven")),
                vec![url_source("https://a.example")],
            ),
        );
        Ok(())
    });
    assert_eq!(
        cited(&response(decoded)),
        [(
            Some("Seven".to_owned()),
            vec![url_source("https://a.example")]
        )]
    );

    // The consumer took the block before the provider cited it.
    let mut turn = Turn::relayed("test");
    let mut items = super::Items::new();
    turn.open_item(&mut items, 0, Block::Text, Value::Null)
        .expect("opens");
    turn.push_item(&mut items, 0, "Dock Seven").expect("writes");
    turn.close_item(&mut items, 0, super::Closing::Complete)
        .expect("closes");
    for item in items.drain(..) {
        if let Ok(Item::Event(event)) = item {
            crate::wire::Fold::<Completion>::absorb(&mut turn, &event).expect("absorbs");
        }
    }
    let whole = |url: &str| WireCitation::new(None, vec![url_source(url)]);
    turn.cite_item(&mut items, 0, vec![whole("https://a.example")], false);
    turn.cite_item(&mut items, 0, vec![whole("https://b.example")], false);
    let response = turn.response(Finish::default(), reply());
    assert_eq!(cited(&response).len(), 2);
    turn.cite_item(&mut items, 0, vec![whole("https://c.example")], true);
    let response = turn.response(Finish::default(), reply());
    assert_eq!(
        cited(&response),
        [(
            Some("Dock Seven".to_owned()),
            vec![url_source("https://c.example")]
        )]
    );
}

fn reply() -> crate::wire::Reply {
    crate::wire::Reply {
        provider: "test".to_owned(),
        raw: Value::Null,
        provider_request_id: None,
    }
}

fn origin(provider: &str, model: &str) -> crate::message::Origin {
    crate::message::Origin::new("test", provider, model)
}

fn tokens(input: u64, output: u64) -> Usage {
    Usage::new()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(input + output)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

/// A cost the provider reported is kept, whatever the catalog says.
#[test]
fn a_reported_cost_wins_over_the_catalog() {
    let end = Finish {
        usage: tokens(1_000, 100).cost(Cost::from_total(0.5)),
        ..Finish::default()
    };
    let response = write_from(origin("anthropic", "claude-sonnet-4-6"), end, |_| Ok(()));
    assert_eq!(response.usage.cost, Some(Cost::from_total(0.5)));
}

/// Without a reported cost the counters are priced at the catalog's
/// prices for the requested model: cache reads and writes at their own
/// prices, the rest of the input at the input price.
#[test]
fn an_unreported_cost_comes_from_the_catalog() {
    // Claude Sonnet 4.6: 3, 15, 0.3 and 3.75 USD per million tokens.
    let end = Finish {
        usage: tokens(1_000_000, 100_000)
            .cached_input_tokens(200_000)
            .cache_creation_input_tokens(100_000),
        ..Finish::default()
    };
    let response = write_from(origin("anthropic", "claude-sonnet-4-6"), end, |_| Ok(()));
    let cost = response.usage.cost.expect("the catalog prices the model");
    assert!(cost.input.is_some_and(|part| close(part, 2.1)), "{cost:?}");
    assert!(cost.output.is_some_and(|part| close(part, 1.5)), "{cost:?}");
    assert!(
        cost.cache_read.is_some_and(|part| close(part, 0.06)),
        "{cost:?}"
    );
    assert!(
        cost.cache_write.is_some_and(|part| close(part, 0.375)),
        "{cost:?}"
    );
    assert!(close(cost.total, 4.035), "{cost:?}");
    assert_eq!(
        response.usage.input_tokens,
        Some(1_000_000),
        "tokens unchanged"
    );

    // A dated snapshot of a listed model has its price.
    let end = Finish {
        usage: tokens(1_000_000, 0),
        ..Finish::default()
    };
    let response = write_from(
        origin("anthropic", "claude-sonnet-4-6-20260101"),
        end,
        |_| Ok(()),
    );
    assert!(
        response
            .usage
            .cost
            .is_some_and(|cost| close(cost.total, 3.0))
    );

    // A cache read with no price of its own is charged as input.
    let end = Finish {
        usage: tokens(1_000_000, 0).cached_input_tokens(1_000_000),
        ..Finish::default()
    };
    let response = write_from(origin("deepseek", "deepseek-v4-flash"), end, |_| Ok(()));
    let cost = response.usage.cost.expect("the catalog prices the model");
    assert!(
        cost.cache_read.is_some_and(|part| close(part, 0.003)),
        "{cost:?}"
    );
    let end = Finish {
        usage: tokens(1_000_000, 0).cache_creation_input_tokens(1_000_000),
        ..Finish::default()
    };
    let response = write_from(origin("deepseek", "deepseek-v4-flash"), end, |_| Ok(()));
    let cost = response.usage.cost.expect("the catalog prices the model");
    assert!(
        cost.cache_write.is_some_and(|part| close(part, 0.15)),
        "{cost:?}"
    );
}

/// The fold prices a reply by the facts of the wire that answers: the spec
/// it was connected to, and that spec's catalog for a request that names
/// another model. The built-in prices do not apply.
#[test]
fn a_reply_is_priced_by_the_wire_facts() {
    use crate::catalog::{Catalog, Pricing};
    use crate::wire::{Call, Descriptor, Mode, Operation};
    let anthropic = crate::providers::registry::ProviderId::catalog("anthropic").expect("a vendor");
    let mut catalog = Catalog::builtin().clone();
    for id in ["claude-sonnet-4-6", "claude-haiku-4-5"] {
        let spec = catalog.get_exact(anthropic, id).expect("listed").clone();
        catalog.insert(spec.with_pricing(Pricing::new(1.0, 2.0)));
    }
    let facts = catalog.facts(anthropic, "claude-sonnet-4-6");
    let descriptor = Descriptor::new("anthropic")
        .model("claude-sonnet-4-6")
        .facts(&facts);
    for request in [
        crate::completion::CompletionRequest::new("hi"),
        crate::completion::CompletionRequest::new("hi").model("claude-haiku-4-5"),
    ] {
        let turn = Completion::fold(&request, &mut Call::new(&descriptor, Mode::Unary));
        let end = Finish {
            usage: tokens(1_000_000, 1_000_000),
            ..Finish::default()
        };
        let response = response(decode_with(turn, "test", |reply| {
            let _ = reply.out().end(end);
            Ok(())
        }));
        let cost = response.usage.cost.expect("the facts price the model");
        assert!(close(cost.total, 3.0), "{:?}: {cost:?}", request.model);
    }
}

/// No cost is made up: an unlisted model, a model with no price, or a
/// reply missing its input or output count has none.
#[test]
fn a_cost_is_none_without_a_price_or_counters() {
    for (origin, usage) in [
        (origin("anthropic", "not-in-the-catalog"), tokens(10, 10)),
        (origin("test", "claude-sonnet-4-6"), tokens(10, 10)),
        (origin("anthropic", "claude-sonnet-4-6"), Usage::new()),
        (
            origin("anthropic", "claude-sonnet-4-6"),
            Usage::new().input_tokens(10),
        ),
        (
            origin("anthropic", "claude-sonnet-4-6"),
            Usage::new().output_tokens(10),
        ),
    ] {
        let end = Finish {
            usage,
            ..Finish::default()
        };
        let response = write_from(origin.clone(), end, |_| Ok(()));
        assert_eq!(response.usage.cost, None, "{origin:?}");
    }
}

/// A subscription or a local server is not billed at the catalog's API
/// rates: Copilot, ChatGPT and Ollama replies have no catalog cost, even
/// for a model the catalog prices under that provider, and a cost they
/// report themselves is kept.
#[test]
fn subscription_and_local_providers_have_no_catalog_cost() {
    assert!(
        crate::catalog::Catalog::builtin()
            .get_vendor("copilot", "gpt-5.3-codex")
            .map(|resolved| resolved.spec)
            .and_then(|spec| spec.pricing.as_ref())
            .is_some(),
        "the catalog prices Copilot's API rate"
    );
    assert!(
        crate::catalog::Catalog::builtin()
            .get_vendor("ollama", "gpt-oss:20b")
            .map(|resolved| resolved.spec)
            .and_then(|spec| spec.pricing.as_ref())
            .is_some(),
        "the catalog prices Ollama Cloud"
    );
    for origin in [
        origin("copilot", "gpt-5.3-codex"),
        origin("ollama", "gpt-oss:20b"),
        origin("chatgpt", "gpt-5.3-codex"),
    ] {
        let end = Finish {
            usage: tokens(1_000_000, 100_000),
            ..Finish::default()
        };
        let response = write_from(origin.clone(), end, |_| Ok(()));
        assert_eq!(response.usage.cost, None, "{origin:?}");

        let end = Finish {
            usage: tokens(10, 10).cost(Cost::from_total(0.5)),
            ..Finish::default()
        };
        let response = write_from(origin.clone(), end, |_| Ok(()));
        assert_eq!(
            response.usage.cost,
            Some(Cost::from_total(0.5)),
            "{origin:?}"
        );
    }
}
