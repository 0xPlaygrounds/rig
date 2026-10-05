//! Decoder tests. Each states a reply shape Ollama's `/api/chat` documents
//! (`docs/api.md`) or one the inline `<think>` split must survive; the
//! recorded replies are pinned by the `ollama` cassettes.

use super::*;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::providers::ollama::OllamaConfig;
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::wire::{Call, Mode, Operation, Shared, Wire};
use serde_json::json;

const MODEL: &str = "llama3.2";

fn frames(records: &[Value]) -> Vec<WireFrame> {
    records
        .iter()
        .map(|record| WireFrame::Text(record.to_string()))
        .collect()
}

/// One streamed record carrying `message`.
fn record(model: &str, message: Value) -> Value {
    json!({"model": model, "created_at": "2026-10-05T00:00:00Z", "message": message, "done": false})
}

/// The record that ends a reply on `reason`.
fn done(model: &str, reason: &str) -> Value {
    json!({
        "model": model,
        "created_at": "2026-10-05T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": true,
        "done_reason": reason,
        "total_duration": 100,
        "prompt_eval_count": 12,
        "prompt_eval_cached_count": 4,
        "eval_count": 7,
    })
}

/// A whole reply: `message`, ended on `reason`.
fn whole(model: &str, message: Value, reason: &str) -> Value {
    let mut reply = done(model, reason);
    reply["message"] = message;
    reply
}

fn decode(model: &str, mode: Mode, records: &[Value]) -> Result<CompletionResponse, ProviderError> {
    let wire = OllamaConfig::new().completion(model);
    crate::test_utils::history::decode(&wire, mode, frames(records))
}

/// The text and reasoning blocks of `response`, in order.
fn blocks(response: &CompletionResponse) -> Vec<(&'static str, String)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(("text", text.text.clone())),
            AssistantContent::Reasoning(reasoning) => Some(("reasoning", reasoning.text.clone())),
            _ => None,
        })
        .collect()
}

/// The events the consumer has seen after each of `records` is decoded,
/// before the reply ends: what streaming shows as it goes.
fn events_after_each(model: &str, records: &[Value]) -> Vec<Vec<StreamEvent>> {
    let wire = OllamaConfig::new().completion(model);
    let describe = wire.describe();
    let fold = Completion::fold(
        &CompletionRequest::new("hi"),
        &mut Call::new(&describe, Mode::Streaming),
    );
    let shared = std::sync::Mutex::new(Shared::new(fold));
    let mut decoder = wire.decoder();
    let mut seen = Vec::new();
    let mut taken = Vec::new();
    for frame in frames(records) {
        crate::driver::step(&mut decoder, &shared, frame, None)
            .map(drop)
            .expect("the record decodes");
        let mut shared = shared.lock().expect("the reply is not poisoned");
        while let Some(item) = shared.take() {
            if let Ok(Item::Event(event)) = item {
                taken.push(event);
            }
        }
        seen.push(taken.clone());
    }
    seen
}

/// The text a list of events streamed, by kind.
fn streamed(events: &[StreamEvent]) -> (String, String) {
    let (mut text, mut reasoning) = (String::new(), String::new());
    for event in events {
        match event {
            StreamEvent::Text { text: fragment, .. } => text.push_str(fragment),
            StreamEvent::Reasoning { text: fragment, .. } => reasoning.push_str(fragment),
            _ => {}
        }
    }
    (text, reasoning)
}

/// The record shape has no discriminator: a line with `message`, `done` or
/// `error` is a record, other JSON is not one, and a line that is not JSON
/// is corrupt. A record's fields are read leniently, so a mistyped `done`
/// does not end the reply.
#[test]
fn a_line_is_a_record_unknown_or_corrupt() {
    let decoder = ChatDecoder::default();
    let line = |text: &str| decoder.classify(WireFrame::Text(text.to_owned()));
    assert!(matches!(
        line(&record(MODEL, json!({"role": "assistant", "content": "hi"})).to_string()),
        WireEvent::Known(_)
    ));
    assert!(matches!(line("{not json"), WireEvent::Corrupt(_)));
    assert!(matches!(line(r#"{"done": 42}"#), WireEvent::Known(_)));
    assert!(matches!(
        line(r#"{"status": "ok"}"#),
        WireEvent::Unknown { .. }
    ));
    assert!(matches!(line("[1]"), WireEvent::Unknown { .. }));
    let mistyped = decode(MODEL, Mode::Streaming, &[json!({"done": 42, "message": 7})]);
    assert!(
        matches!(mistyped, Err(ProviderError::Truncated)),
        "{mistyped:?}"
    );
}

/// A streamed reply's records join into one text block, and the `done`
/// record ends it with its reason, counters and model.
#[test]
fn ndjson_records_stream_into_one_answer() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "The sky "})),
        record(MODEL, json!({"role": "assistant", "content": "is blue."})),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(blocks(&response), [("text", "The sky is blue.".to_owned())]);
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.input_tokens, Some(12));
    assert_eq!(response.usage.output_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(19));
    assert_eq!(response.usage.cached_input_tokens, Some(4));
    assert_eq!(response.model(), Some(MODEL));
    // A stream's raw is the record that ended it.
    assert_eq!(response.raw.get("done_reason"), Some(&json!("stop")));
}

/// The reply's `thinking` is reasoning, held before the answer.
#[test]
fn thinking_is_reasoning() {
    let records = [
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "thinking": "Rayleigh "}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "thinking": "scattering."}),
        ),
        record(MODEL, json!({"role": "assistant", "content": "Blue."})),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "Rayleigh scattering.".to_owned()),
            ("text", "Blue.".to_owned())
        ]
    );
}

/// Ollama sends each call whole, its arguments an object. Each call starts
/// and streams its arguments through the shared writer, keeps the daemon's
/// id, and a `stop` reply that called a tool ends on `ToolCalls`.
#[test]
fn ndjson_tool_calls_stream_through_the_writer() {
    let call = |id: &str, city: &str| json!({"id": id, "function": {"index": 0, "name": "weather", "arguments": {"city": city}}});
    let records = [
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "tool_calls": [call("call_a", "Paris")]}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": "", "tool_calls": [call("call_b", "Rome")]}),
        ),
        done(MODEL, "stop"),
    ];
    let response = decode(MODEL, Mode::Streaming, &records).expect("the reply decodes");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.wire(), "call_a");
    assert_eq!(calls[1].id.wire(), "call_b");
    assert_eq!(calls[0].function.name.as_str(), "weather");
    assert_eq!(
        calls[1].function.arguments.get("city"),
        Some(&json!("Rome"))
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));

    let events = events_after_each(MODEL, &records);
    let first = &events[0];
    assert!(first.iter().any(|event| matches!(
        event,
        StreamEvent::Start { kind: PartKind::ToolCall, name: Some(name), .. } if name.as_str() == "weather"
    )));
    assert!(first.iter().any(|event| matches!(
        event,
        StreamEvent::Arguments { json, .. } if json == r#"{"city":"Paris"}"#
    )));
}

/// A call whose arguments come as text, or not at all, still decodes; one
/// with no name is dropped.
#[test]
fn call_arguments_as_text_or_absent() {
    let reply = whole(
        MODEL,
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "a", "arguments": "{\"x\":1}"}},
            {"function": {"name": "b"}},
            {"function": {"arguments": {}}},
        ]}),
        "stop",
    );
    let response = decode(MODEL, Mode::Unary, &[reply]).expect("the reply decodes");
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.arguments.get("x"), Some(&json!(1)));
    assert!(calls[1].function.arguments.is_empty());

    // A call that is not an object names no tool, and is dropped too.
    let reply = whole(
        MODEL,
        json!({"role": "assistant", "tool_calls": ["call"]}),
        "stop",
    );
    let response = decode(MODEL, Mode::Unary, &[reply]).expect("the reply decodes");
    assert_eq!(response.tool_calls().count(), 0);
}

/// `done_reason` maps to the finish reason: `stop`, `length`, and anything
/// else carried as Ollama spells it.
#[test]
fn done_reason_is_the_finish_reason() {
    for (reason, expected) in [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("unload", FinishReason::Other("unload".to_owned())),
    ] {
        let reply = whole(
            MODEL,
            json!({"role": "assistant", "content": "partial"}),
            reason,
        );
        let response = decode(MODEL, Mode::Unary, std::slice::from_ref(&reply));
        let finish = match response {
            Ok(response) => response.finish_reason(),
            // An unknown reason fails the turn.
            Err(_) => Some(FinishReason::Other(reason.to_owned())),
        };
        assert_eq!(finish, Some(expected), "{reason}");
    }
}

/// An in-band `error` record fails the reply, and a stream that ends
/// before its `done` record is truncated.
#[test]
fn an_error_record_fails_and_a_cut_stream_is_truncated() {
    let error = decode(
        MODEL,
        Mode::Streaming,
        &[
            record(MODEL, json!({"role": "assistant", "content": "par"})),
            json!({"error": "model runner has unexpectedly stopped"}),
        ],
    )
    .expect_err("the error fails the reply");
    assert!(
        error.to_string().contains("unexpectedly stopped"),
        "{error}"
    );

    let cut = decode(
        MODEL,
        Mode::Streaming,
        &[record(
            MODEL,
            json!({"role": "assistant", "content": "par"}),
        )],
    )
    .expect_err("no done record");
    assert!(matches!(cut, ProviderError::Truncated), "{cut:?}");
}

/// The old whole-reply rule, unchanged: only a leading, terminated block
/// splits, and `qwen3` also splits at its exact prefilled boundary.
#[test]
fn leading_reasoning_splits_only_a_leading_terminated_block() {
    assert_eq!(
        leading_reasoning("<think>private reasoning</think>\n\nvisible answer", false),
        Held::Split("private reasoning", "visible answer")
    );
    assert_eq!(
        leading_reasoning("private reasoning\n</think>\n\nvisible answer", true),
        Held::Split("private reasoning", "visible answer")
    );
    assert_eq!(leading_reasoning("<think>unterminated", true), Held::Open);
    assert_eq!(leading_reasoning("  <thi", false), Held::Open);
    assert_eq!(leading_reasoning("", false), Held::Open);
    assert_eq!(
        leading_reasoning("The literal marker is <think>.", false),
        Held::Text
    );
    assert_eq!(
        leading_reasoning("  visible indentation", false),
        Held::Text
    );
    // Without the exact boundary a `qwen3` reply stays open, so a whole one
    // ends as the text it is.
    assert_eq!(
        leading_reasoning("Example:\n</think>\nis a closing tag.", true),
        Held::Open
    );
    assert_eq!(
        leading_reasoning("private reasoning\n</think>\n\nvisible answer", false),
        Held::Text
    );
}

/// Each inline case, whole and streamed one character at a time, folds to
/// the same blocks.
#[test]
fn inline_think_splits_the_same_whole_and_streamed() {
    let qwen = "qwen3:4b";
    let cases: [(&str, &str, Option<&str>, Vec<(&'static str, &str)>); 8] = [
        (
            "terminated",
            MODEL,
            Some("<think>private</think>\n\nThe answer."),
            vec![("reasoning", "private"), ("text", "The answer.")],
        ),
        (
            "unterminated stays text",
            MODEL,
            Some("<think>never closed"),
            vec![("text", "<think>never closed")],
        ),
        (
            "absent",
            MODEL,
            Some("Just an answer."),
            vec![("text", "Just an answer.")],
        ),
        (
            "qwen3 missing opening marker",
            qwen,
            Some("private\n</think>\n\nThe answer."),
            vec![("reasoning", "private"), ("text", "The answer.")],
        ),
        (
            "qwen3 without the boundary stays text",
            qwen,
            Some("Example:\n</think>\nis a tag."),
            vec![("text", "Example:\n</think>\nis a tag.")],
        ),
        (
            "missing opening marker on another model stays text",
            MODEL,
            Some("private\n</think>\n\nThe answer."),
            vec![("text", "private\n</think>\n\nThe answer.")],
        ),
        (
            "not at the start stays text",
            MODEL,
            Some("Use <think>x</think> tags."),
            vec![("text", "Use <think>x</think> tags.")],
        ),
        (
            "an empty block leaves only the answer",
            MODEL,
            Some("<think>\n\n</think>\n\nThe answer."),
            vec![("text", "The answer.")],
        ),
    ];
    for (case, model, content, expected) in cases {
        let content = content.unwrap_or_default();
        let expected: Vec<(&str, String)> = expected
            .into_iter()
            .map(|(kind, text)| (kind, text.to_owned()))
            .collect();
        let reply = whole(
            model,
            json!({"role": "assistant", "content": content}),
            "stop",
        );
        let response = decode(model, Mode::Unary, &[reply]).expect("the reply decodes");
        assert_eq!(blocks(&response), expected, "{case}, whole");

        let mut records: Vec<Value> = content
            .chars()
            .map(|ch| {
                record(
                    model,
                    json!({"role": "assistant", "content": ch.to_string()}),
                )
            })
            .collect();
        records.push(done(model, "stop"));
        let response = decode(model, Mode::Streaming, &records).expect("the reply decodes");
        assert_eq!(blocks(&response), expected, "{case}, streamed");
    }
}

/// A reply with a native `thinking` field splits nothing out of its
/// content, whole or streamed.
#[test]
fn native_thinking_turns_the_split_off() {
    let content = "<think>shown</think> as written";
    let reply = whole(
        "qwen3:4b",
        json!({"role": "assistant", "content": content, "thinking": "native"}),
        "stop",
    );
    let response = decode("qwen3:4b", Mode::Unary, &[reply]).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "native".to_owned()),
            ("text", content.to_owned())
        ]
    );

    let records = [
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "", "thinking": "native"}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "<think>shown"}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "</think> as written"}),
        ),
        done("qwen3:4b", "stop"),
    ];
    let response = decode("qwen3:4b", Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(
        blocks(&response),
        [
            ("reasoning", "native".to_owned()),
            ("text", content.to_owned())
        ]
    );
    // The content streams as it arrives.
    let events = events_after_each("qwen3:4b", &records);
    assert_eq!(streamed(&events[1]).0, "<think>shown");
}

/// Streamed content is held while it may be a leading block: nothing shows
/// until `</think>` arrives, then the block is reasoning and the rest
/// streams as text as it comes.
#[test]
fn a_streamed_block_is_held_until_it_closes() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "<thi"})),
        record(
            MODEL,
            json!({"role": "assistant", "content": "nk>weighing"}),
        ),
        record(
            MODEL,
            json!({"role": "assistant", "content": " it</think>"}),
        ),
        record(MODEL, json!({"role": "assistant", "content": "\n\n"})),
        record(MODEL, json!({"role": "assistant", "content": "Yes"})),
        record(MODEL, json!({"role": "assistant", "content": ", indeed."})),
        done(MODEL, "stop"),
    ];
    let events = events_after_each(MODEL, &records);
    assert_eq!(streamed(&events[1]), (String::new(), String::new()));
    assert_eq!(
        streamed(&events[2]),
        (String::new(), "weighing it".to_owned())
    );
    assert_eq!(streamed(&events[3]).0, "");
    assert_eq!(streamed(&events[4]).0, "Yes");
    assert_eq!(streamed(&events[5]).0, "Yes, indeed.");
}

/// A stream that ends inside an unterminated block shows everything it
/// held as text, so the answer is never swallowed; so does a call that
/// arrives while content is held.
#[test]
fn held_content_is_text_when_the_block_never_closes() {
    let records = [
        record(MODEL, json!({"role": "assistant", "content": "<think>"})),
        record(MODEL, json!({"role": "assistant", "content": "the answer"})),
        done(MODEL, "length"),
    ];
    let events = events_after_each(MODEL, &records);
    assert_eq!(streamed(&events[1]).0, "");
    assert_eq!(streamed(&events[2]).0, "<think>the answer");

    let records = [
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "Calling."}),
        ),
        record(
            "qwen3:4b",
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "lookup", "arguments": {}}}
            ]}),
        ),
        done("qwen3:4b", "stop"),
    ];
    let response = decode("qwen3:4b", Mode::Streaming, &records).expect("the reply decodes");
    assert_eq!(blocks(&response), [("text", "Calling.".to_owned())]);
    assert_eq!(response.tool_calls().count(), 1);
}

/// Observation reads the counters, the reason and the model off the
/// `done` record, and an in-band error's message.
#[test]
fn the_projection_reads_usage_verdict_and_error() {
    use crate::observe::{AdapterContext, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    let mut attempt = context
        .attempt_for(&http::Request::new(()), "/api/chat")
        .expect("an attempt starts");
    for payload in [
        done(MODEL, "stop").to_string(),
        json!({"error": "out of memory"}).to_string(),
        "not json".to_owned(),
    ] {
        attempt.project(|sink| ChatDecoder::project(payload.as_bytes(), sink));
    }
    drop(attempt);
    let observed = log.trace();
    let text = format!("{observed:?}");
    assert!(text.contains("input_tokens: Some(12)"), "{text}");
    assert!(text.contains("total_tokens: Some(19)"), "{text}");
    assert!(text.contains("cached_input_tokens: Some(4)"), "{text}");
    assert!(text.contains("\"stop\""), "{text}");
    assert!(text.contains("out of memory"), "{text}");
}
