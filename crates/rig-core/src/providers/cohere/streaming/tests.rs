use super::*;
use crate::completion::{AssistantContent, CompletionResponse};
use crate::driver::{Decoded, feed_frames};
use crate::streaming::{Item, StreamEvent};
use crate::wire::AdapterEvent;
use crate::wire::document::Reassemble;
use serde_json::json;

/// `frames` decoded, classifier included, as one native reply.
fn fed(frames: &[Value]) -> Decoded<Completion> {
    feed_frames!(
        ChatDecoder::default(),
        document::ChatResponse::default(),
        "cohere",
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string()))
    )
}

/// The document the native reassembler rebuilds from `frames`.
fn rebuilt(frames: &[Value]) -> Value {
    let mut document = document::ChatResponse::default();
    for frame in frames {
        document.absorb(&WireFrame::Text(frame.to_string()));
    }
    document.finish()
}

/// The response `frames` fold into.
fn folded(frames: &[Value]) -> Result<CompletionResponse, ProviderError> {
    fed(frames).outcome
}

/// The response `frames` fold into as a streamed reply on the native wire,
/// whose fold holds the reply to its finish reason.
fn on_wire(frames: &[Value]) -> Result<CompletionResponse, ProviderError> {
    let wire = super::super::NativeChat::new(super::super::CohereConfig::new("key"), "command-a");
    crate::test_utils::decode_reply(
        &wire,
        &crate::completion::CompletionRequest::new("hi"),
        crate::wire::Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
}

/// The provider item a block's `native` holds.
fn item(native: &Option<crate::message::Native>) -> Option<Value> {
    native.as_ref().map(|native| native.item.clone())
}

fn citation(start: u64, end: u64, id: &str, extra: Value) -> Value {
    let mut citation = json!({
        "start": start, "end": end, "text": "cited",
        "sources": [{"type": "document", "id": id, "document": {"id": id, "text": "source"}}],
        "type": "TEXT_CONTENT",
    });
    if let (Some(citation), Value::Object(extra)) = (citation.as_object_mut(), extra) {
        citation.extend(extra);
    }
    citation
}

fn usage() -> Value {
    json!({"billed_units": {"input_tokens": 56, "output_tokens": 20},
        "tokens": {"input_tokens": 1706, "output_tokens": 43, "reasoning_tokens": 7},
        "cached_tokens": 112})
}

/// A whole reply: `message` ending on `finish`.
fn whole(message: Value, finish: &str) -> Vec<Value> {
    vec![json!({"id": "resp_1", "message": message, "finish_reason": finish, "usage": usage()})]
}

/// A stream of `events` between `message-start` and a `message-end` on
/// `finish`.
fn stream(events: Vec<Value>, finish: &str) -> Vec<Value> {
    let start = json!({"id": "resp_1", "type": "message-start", "delta": {"message":
        {"role": "assistant", "content": [], "tool_plan": "", "tool_calls": [], "citations": []}}});
    let end = json!({"type": "message-end", "delta": {"finish_reason": finish, "usage": usage()}});
    std::iter::once(start).chain(events).chain([end]).collect()
}

fn content_start(index: usize, kind: &str) -> Value {
    json!({"type": "content-start", "index": index,
        "delta": {"message": {"content": {"type": kind, kind: ""}}}})
}

fn content_delta(index: usize, key: &str, text: &str) -> Value {
    json!({"type": "content-delta", "index": index,
        "delta": {"message": {"content": {key: text}}}})
}

fn content_end(index: usize) -> Value {
    json!({"type": "content-end", "index": index})
}

fn call_start(index: usize, id: &str, name: &str) -> Value {
    json!({"type": "tool-call-start", "index": index, "delta": {"message": {"tool_calls":
        {"id": id, "type": "function", "function": {"name": name, "arguments": ""}}}}})
}

fn call_delta(index: usize, arguments: &str) -> Value {
    json!({"type": "tool-call-delta", "index": index,
        "delta": {"message": {"tool_calls": {"function": {"arguments": arguments}}}}})
}

fn call_end(index: usize) -> Value {
    json!({"type": "tool-call-end", "index": index})
}

fn plan_delta(text: &str) -> Value {
    json!({"type": "tool-plan-delta", "delta": {"message": {"tool_plan": text}}})
}

fn citation_start(index: usize, citation: Value) -> Value {
    json!({"type": "citation-start", "index": index,
        "delta": {"message": {"citations": citation}}})
}

/// A whole reply's text keeps the citations that point at it in its item,
/// sources and all, and its usage counts what the model read and wrote.
#[test]
fn a_whole_reply_keeps_its_citations_on_the_text() {
    let cited = citation(4, 9, "doc-1", json!({}));
    let response = folded(&whole(
        json!({"role": "assistant",
            "content": [{"type": "text", "text": "The sky is green."}],
            "citations": [cited.clone()]}),
        "COMPLETE",
    ))
    .expect("the reply decodes");
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    assert_eq!(text.text, "The sky is green.");
    assert_eq!(
        item(&text.native),
        Some(json!({"type": "text", "text": "The sky is green.", "citations": [cited]}))
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage.input_tokens, Some(1706));
    assert_eq!(response.usage.output_tokens, Some(43));
    assert_eq!(response.usage.cached_input_tokens, Some(112));
    assert_eq!(response.usage.reasoning_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(1749));
    assert_eq!(response.response_id(), Some("resp_1"));
}

/// The tool plan is a reasoning block whose item says it is the plan and
/// holds the plan's citations; each call keeps its item with its id.
#[test]
fn a_whole_reply_keeps_its_tool_plan_and_calls() {
    let plan_citation = json!({"start": 0, "end": 3, "text": "use", "type": "PLAN",
        "sources": [{"type": "tool", "id": "t1"}]});
    let response = folded(&whole(
        json!({"role": "assistant",
            "tool_plan": "I will use the tool.",
            "tool_calls": [
                {"id": "get_weather_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                {"id": "get_weather_2", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}},
            ],
            "citations": [plan_citation.clone()]}),
        "TOOL_CALL",
    ))
    .expect("the reply decodes");
    let [
        AssistantContent::Reasoning(plan),
        AssistantContent::ToolCall(paris),
        AssistantContent::ToolCall(tokyo),
    ] = response.choice.as_slice()
    else {
        panic!("a plan and two calls: {:?}", response.choice);
    };
    assert_eq!(plan.text, "I will use the tool.");
    assert_eq!(
        item(&plan.native),
        Some(
            json!({"type": "tool_plan", "tool_plan": "I will use the tool.",
            "citations": [plan_citation]})
        )
    );
    assert_eq!(paris.id.wire(), "get_weather_1");
    assert_eq!(paris.function.arguments_value(), json!({"city": "Paris"}));
    assert_eq!(
        item(&paris.native).and_then(|item| item.at("/function/arguments").cloned()),
        Some(json!("{\"city\":\"Paris\"}"))
    );
    assert_eq!(tokyo.function.arguments_value(), json!({"city": "Tokyo"}));
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Streamed calls go through the writer: each argument fragment reaches the
/// consumer as its own `Arguments` event, after the plan streamed as
/// reasoning.
#[test]
fn streamed_tool_calls_stream_each_argument_fragment() {
    let decoded = fed(&stream(
        vec![
            plan_delta("I will"),
            plan_delta(" look."),
            call_start(0, "get_weather_1", "get_weather"),
            call_delta(0, "{\"city\""),
            call_delta(0, ": \"Paris\"}"),
            call_end(0),
            call_start(1, "get_weather_2", "get_weather"),
            call_delta(1, "{\"city\": \"Tokyo\"}"),
            call_end(1),
        ],
        "TOOL_CALL",
    ));
    let events: Vec<StreamEvent> = decoded
        .items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(event)) => Some(event.clone()),
            _ => None,
        })
        .collect();
    let fragments: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        fragments,
        ["{\"city\"", ": \"Paris\"}", "{\"city\": \"Tokyo\"}"]
    );
    let reasoning: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "I will look.");
    let response = decoded.outcome.expect("the stream folds");
    let [
        AssistantContent::Reasoning(plan),
        AssistantContent::ToolCall(paris),
        AssistantContent::ToolCall(tokyo),
    ] = response.choice.as_slice()
    else {
        panic!("a plan and two calls: {:?}", response.choice);
    };
    assert_eq!(
        item(&plan.native).and_then(|item| item.str(PLAN).map(str::to_owned)),
        Some("I will look.".to_owned())
    );
    assert_eq!(paris.function.arguments_value(), json!({"city": "Paris"}));
    assert_eq!(
        item(&paris.native),
        Some(json!({"id": "get_weather_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}))
    );
    assert_eq!(tokyo.id.wire(), "get_weather_2");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Streamed citations land on the part their `content_index` names, here
/// the text after the thinking, and thinking keeps its item.
#[test]
fn streamed_citations_attach_to_the_part_they_cite() {
    let first = citation(0, 3, "doc-1", json!({"content_index": 1}));
    let second = citation(4, 8, "doc-2", json!({"content_index": 1}));
    let response = folded(&stream(
        vec![
            content_start(0, "thinking"),
            content_delta(0, "thinking", "I need"),
            content_delta(0, "thinking", " to look."),
            content_end(0),
            content_start(1, "text"),
            content_delta(1, "text", "Sky is "),
            citation_start(0, first.clone()),
            json!({"type": "citation-end", "index": 0}),
            content_delta(1, "text", "green."),
            citation_start(1, second.clone()),
            json!({"type": "citation-end", "index": 1}),
            content_end(1),
        ],
        "COMPLETE",
    ))
    .expect("the stream folds");
    let [
        AssistantContent::Reasoning(thinking),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!("thinking then text: {:?}", response.choice);
    };
    assert_eq!(thinking.text, "I need to look.");
    assert_eq!(
        item(&thinking.native),
        Some(json!({"type": "thinking", "thinking": "I need to look."}))
    );
    assert_eq!(text.text, "Sky is green.");
    assert_eq!(
        item(&text.native),
        Some(json!({"type": "text", "text": "Sky is green.", "citations": [first, second]}))
    );
    assert_eq!(response.response_id(), Some("resp_1"));
}

/// A citation of a block that never opened is dropped, not fatal.
#[test]
fn a_citation_of_no_block_is_dropped() {
    let response = folded(&stream(
        vec![citation_start(
            0,
            citation(0, 1, "doc-1", json!({"content_index": 3})),
        )],
        "COMPLETE",
    ))
    .expect("the stream folds");
    assert!(response.choice.is_empty(), "{:?}", response.choice);
}

/// Each documented finish reason, and the failures: `ERROR` reports the
/// provider's error, and a reason Cohere does not document fails the turn.
#[test]
fn finish_reasons_map_to_rig_endings() {
    for (reason, finish) in [
        ("COMPLETE", FinishReason::Stop),
        ("STOP_SEQUENCE", FinishReason::Stop),
        ("MAX_TOKENS", FinishReason::Length),
        ("TOOL_CALL", FinishReason::ToolCalls),
        ("TIMEOUT", FinishReason::Other("TIMEOUT".to_owned())),
    ] {
        let response = folded(&whole(
            json!({"content": [{"type": "text", "text": "hi"}]}),
            reason,
        ))
        .expect("the reply decodes");
        assert_eq!(response.finish_reason(), Some(finish), "{reason}");
    }
    let failed = folded(&whole(json!({"content": []}), "ERROR")).expect("the reply decodes");
    assert_eq!(
        failed.error.as_deref(),
        Some("Cohere ended the reply with an error")
    );
    let stated = folded(&[
        json!({"type": "message-end", "delta": {"finish_reason": "ERROR", "error": "overloaded"}}),
    ])
    .expect("the stream folds");
    assert_eq!(stated.error.as_deref(), Some("overloaded"));
    assert_eq!(stated.raw.at("/error"), Some(&json!("overloaded")));
}

/// Usage without `tokens` falls back to the billed units.
#[test]
fn usage_falls_back_to_billed_units() {
    let usage = usage_of(Some(
        &json!({"billed_units": {"input_tokens": 3, "output_tokens": 4}}),
    ));
    assert_eq!(usage.input_tokens, Some(3));
    assert_eq!(usage.output_tokens, Some(4));
    assert_eq!(usage.total_tokens, Some(7));
    assert_eq!(usage.cached_input_tokens, None);
}

/// Blank text keeps no item, and a call whose arguments never became a
/// JSON object keeps none either; the reply's end closes a call whose
/// arguments are cut short without stating it complete.
#[test]
fn incomplete_blocks_keep_no_item() {
    let response = on_wire(&stream(
        vec![
            content_start(0, "text"),
            content_delta(0, "text", "  "),
            content_end(0),
            call_start(0, "c1", "lookup"),
            call_delta(0, "[1]"),
            call_end(0),
            call_start(1, "c2", "lookup"),
            call_delta(1, "{\"q\": "),
        ],
        "TOOL_CALL",
    ))
    .expect("the stream folds");
    let calls: Vec<&crate::message::ToolCall> = response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2, "{:?}", response.choice);
    assert!(calls.iter().all(|call| item(&call.native).is_none()));
    assert!(
        response.error.is_some(),
        "a call the provider never finished fails the turn"
    );
}

/// Events addressed to blocks that are not open fail the reply.
#[test]
fn malformed_replies_fail() {
    for frames in [
        vec![content_delta(0, "text", "x")],
        vec![call_delta(0, "{}")],
        vec![content_end(0)],
        vec![json!({"type": "content-start"})],
    ] {
        assert!(folded(&frames).is_err(), "{frames:?}");
    }
}

/// A delta without the open part's text, a `debug` event and a plan delta
/// without a plan change nothing; an event of a type Cohere may add later
/// is unknown, not corrupt.
#[test]
fn unknown_and_empty_events_are_skipped() {
    let response = folded(&stream(
        vec![
            json!({"type": "debug", "event_type": "stream-start"}),
            content_start(0, "text"),
            content_delta(0, "thinking", "ignored"),
            content_delta(0, "text", "ok"),
            json!({"type": "tool-plan-delta", "delta": {}}),
            content_end(0),
        ],
        "COMPLETE",
    ))
    .expect("the stream folds");
    assert_eq!(
        response.choice,
        vec![AssistantContent::text("ok").with_native(json!({"type": "text", "text": "ok"}))]
    );
    let decoder = ChatDecoder::default();
    assert!(matches!(
        decoder.classify(WireFrame::Text(r#"{"type":"x-new-event"}"#.into())),
        WireEvent::Unknown { .. }
    ));
    assert!(matches!(
        decoder.classify(WireFrame::Text("not json".into())),
        WireEvent::Corrupt(_)
    ));
}

/// A whole reply projects its usage and finish reason for observation,
/// through the wire's encoded request and the driver.
#[tokio::test]
async fn a_reply_projects_its_metadata() {
    use crate::observe::{Action, AdapterContext, ObservationLog, Subject};
    use std::sync::Arc;

    let body = json!({"id": "resp_1", "message": {"content": []}, "finish_reason": "COMPLETE",
        "usage": usage()});
    let http = crate::test_utils::RecordingHttpClient::new(body.to_string());
    let wire = super::super::NativeChat::new(super::super::CohereConfig::new("key"), "command-a");
    let log = Arc::new(ObservationLog::default());
    crate::driver::Model::new(wire, http)
        .call_observed(
            crate::completion::CompletionRequest::new("hi"),
            AdapterContext::new(log.clone(), Subject::default(), "call"),
        )
        .await
        .expect("the reply folds");
    let events: Vec<AdapterEvent> = log
        .trace()
        .observations
        .iter()
        .filter_map(|o| match &o.action {
            Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect();
    assert!(
        events.iter().any(|event| matches!(event,
            AdapterEvent::Usage { usage } if usage.input_tokens == Some(1706))),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(event,
            AdapterEvent::Provider { verdict } if verdict.finish_reason.as_deref() == Some("COMPLETE"))),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(event,
            AdapterEvent::Started { route, .. } if route == "/v2/chat")),
        "{events:?}"
    );
}

/// A content part of a type rig does not know is kept whole, its deltas
/// merged in, and replays as it came.
#[test]
fn an_unknown_part_is_kept_whole() {
    let response = folded(&stream(
        vec![
            json!({"type": "content-start", "index": 0,
                "delta": {"message": {"content": {"type": "x_rig_invented", "payload": "a"}}}}),
            json!({"type": "content-delta", "index": 0,
                "delta": {"message": {"content": {"payload": "b"}}}}),
            content_end(0),
        ],
        "COMPLETE",
    ))
    .expect("the stream folds");
    let [AssistantContent::Opaque(opaque)] = response.choice.as_slice() else {
        panic!("one opaque block: {:?}", response.choice);
    };
    assert!(opaque.replay);
    assert_eq!(
        opaque.item,
        json!({"type": "x_rig_invented", "payload": "ab"})
    );
}

/// A whole reply's part or call that is not an object is skipped.
#[test]
fn a_whole_reply_skips_parts_that_are_not_objects() {
    let response = folded(&whole(
        json!({"content": ["text", {"type": "text", "text": "ok"}], "tool_calls": [1]}),
        "COMPLETE",
    ))
    .expect("the reply decodes");
    assert_eq!(response.choice.len(), 1, "{:?}", response.choice);
}

/// Cohere's error body, `{"id", "message": "..."}`, sent with a success
/// status fails the reply with its text instead of folding to an empty
/// answer.
#[test]
fn an_error_body_with_a_success_status_fails_with_its_text() {
    let error = folded(&[json!({"id": "x", "message": "internal error"})])
        .expect_err("an error body fails the reply");
    assert!(error.to_string().contains("internal error"), "{error}");
}

/// A native reply whose message holds no content delivers no answer, and
/// the shared rules judge it as on every wire: an answerless `COMPLETE` is
/// for the run to judge, an answerless `MAX_TOKENS` fails the turn, and a
/// reply that states no finish reason fails the turn.
#[test]
fn a_reply_without_content_answers_nothing() {
    use crate::completion::message::{turn_delivered_no_answer, turn_failure};
    let empty = |finish: Option<&str>| {
        let mut reply = json!({"id": "x", "message": {"role": "assistant"}});
        if let (Some(reply), Some(finish)) = (reply.as_object_mut(), finish) {
            reply.insert("finish_reason".to_owned(), json!(finish));
        }
        let response = on_wire(&[reply]).expect("the reply folds");
        assert!(
            turn_delivered_no_answer(&response.choice),
            "{:?}",
            response.choice
        );
        let failure = turn_failure(
            &response.choice,
            response.head().stop.as_ref(),
            response.finish_reason().as_ref(),
        );
        (response, failure)
    };
    let (complete, failure) = empty(Some("COMPLETE"));
    assert_eq!(complete.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(failure, None);
    let (_, failure) = empty(Some("MAX_TOKENS"));
    assert!(
        failure.is_some_and(|failure| failure.contains("no answer")),
        "an answerless cut turn fails"
    );
    let (unstated, failure) = empty(None);
    assert_eq!(
        unstated.error.as_deref(),
        Some("the provider ended the reply without a finish reason")
    );
    assert!(failure.is_some(), "a reply with no finish reason fails");
}

/// A content, call or cited index that reaches the indices the decoder keeps
/// for calls and the tool plan fails the reply.
#[test]
fn an_index_past_the_reserved_range_fails() {
    for frames in [
        vec![content_start(CALLS, "text")],
        vec![call_start(CALLS, "c1", "lookup")],
        vec![
            content_start(0, "text"),
            content_delta(PLAN_INDEX, "text", "x"),
        ],
        vec![
            content_start(0, "text"),
            citation_start(0, citation(0, 1, "doc-1", json!({"content_index": CALLS}))),
        ],
        whole(
            json!({"content": [{"type": "text", "text": "hi"}],
                "citations": [citation(0, 1, "doc-1", json!({"content_index": PLAN_INDEX}))]}),
            "COMPLETE",
        ),
    ] {
        let error = folded(&frames).expect_err("the index is refused");
        assert!(
            matches!(&error, ProviderError::Response(message) if message.contains("index")),
            "{frames:?}: {error}"
        );
    }
}

/// A native finish reason Cohere does not document fails a text turn
/// through the driver, and a request that accepts unknown reasons gets a
/// turn that stops normally.
#[test]
fn an_unknown_finish_reason_fails_unless_accepted() {
    let wire = super::super::NativeChat::new(super::super::CohereConfig::new("key"), "command-a");
    let reply = |accept: bool| {
        crate::test_utils::decode_reply(
            &wire,
            &crate::completion::CompletionRequest::new("hi").accept_unknown_finish_reasons(accept),
            crate::wire::Mode::Unary,
            whole(
                json!({"content": [{"type": "text", "text": "hi"}]}),
                "SOMETHING_NEW",
            )
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
            Value::Null,
        )
        .expect("the reply folds")
    };
    let failed = reply(false);
    let stop = failed.head().stop;
    assert!(
        crate::completion::message::turn_failure(
            &failed.choice,
            stop.as_ref(),
            failed.finish_reason().as_ref()
        )
        .is_some_and(|failure| failure.contains("SOMETHING_NEW")),
        "{stop:?}"
    );
    let accepted = reply(true);
    assert_eq!(accepted.head().stop, Some(crate::message::StopReason::Stop));
    assert_eq!(
        crate::completion::message::turn_failure(
            &accepted.choice,
            accepted.head().stop.as_ref(),
            accepted.finish_reason().as_ref()
        ),
        None
    );
}

/// A stream rebuilds the chat response a unary call returns: parts with
/// their appended text, the plan, calls with their appended arguments,
/// citations, log probabilities, the finish reason and usage.
#[test]
fn a_stream_rebuilds_the_unary_chat_response() {
    let first = citation(0, 3, "doc-1", json!({"content_index": 1}));
    let second = citation(4, 8, "doc-2", json!({"content_index": 1}));
    let logprob = |text: &str| json!({"token_ids": [7], "text": text, "logprobs": [-0.5]});
    let mut delta = content_delta(1, "text", "Sky is ");
    delta["logprobs"] = logprob("Sky is ");
    let mut more = content_delta(1, "text", "green.");
    more["logprobs"] = logprob("green.");
    let streamed = rebuilt(&stream(
        vec![
            content_start(0, "thinking"),
            content_delta(0, "thinking", "I need"),
            content_delta(0, "thinking", " to look."),
            content_end(0),
            content_start(1, "text"),
            delta,
            citation_start(0, first.clone()),
            json!({"type": "citation-end", "index": 0}),
            more,
            citation_start(1, second.clone()),
            content_end(1),
            plan_delta("I will"),
            plan_delta(" look."),
            call_start(0, "get_weather_1", "get_weather"),
            call_delta(0, "{\"city\""),
            call_delta(0, ": \"Paris\"}"),
            call_end(0),
            json!({"type": "debug", "event": "ignored"}),
        ],
        "TOOL_CALL",
    ));
    assert_eq!(
        streamed,
        json!({
            "id": "resp_1",
            "finish_reason": "TOOL_CALL",
            "usage": usage(),
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to look."},
                    {"type": "text", "text": "Sky is green."},
                ],
                "tool_plan": "I will look.",
                "tool_calls": [{"id": "get_weather_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}],
                "citations": [first, second],
            },
            "logprobs": [logprob("Sky is "), logprob("green.")],
        })
    );
}

/// A whole reply is its own document, and a stream cut short rebuilds what
/// arrived.
#[test]
fn a_whole_reply_is_its_document_and_a_cut_stream_keeps_what_arrived() {
    let reply = whole(
        json!({"content": [{"type": "text", "text": "hi"}]}),
        "COMPLETE",
    );
    assert_eq!(Some(&rebuilt(&reply)), reply.first());

    let mut cut = stream(
        vec![content_start(0, "text"), content_delta(0, "text", "par")],
        "COMPLETE",
    );
    cut.pop();
    assert_eq!(
        rebuilt(&cut),
        json!({"id": "resp_1", "message": {"role": "assistant", "content": [{"type": "text", "text": "par"}],
            "tool_calls": [], "citations": []}})
    );
}

/// `frames` as a reply in `mode` from the native wire for Command A, which
/// the catalog prices.
fn command_a(frames: &[Value], mode: crate::wire::Mode) -> CompletionResponse {
    let wire = super::super::NativeChat::new(
        super::super::CohereConfig::new("key"),
        super::super::COMMAND_A_03_2025,
    );
    crate::test_utils::decode_reply(
        &wire,
        &crate::completion::CompletionRequest::new("hi"),
        mode,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
    .expect("the reply decodes")
}

/// Each text block's text and its citations' spans as text, with their
/// sources.
fn typed_citations(
    response: &CompletionResponse,
) -> Vec<(String, Vec<(Option<String>, Vec<crate::message::Source>)>)> {
    response
        .choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some((
                text.text.clone(),
                text.citations()
                    .iter()
                    .map(|citation| {
                        (
                            citation.span.and(text.cited(citation)).map(str::to_owned),
                            citation.sources.clone(),
                        )
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

fn document(id: &str) -> crate::message::Source {
    crate::message::Source::new(crate::message::SourceLocation::Document {
        index: None,
        id: Some(id.to_owned()),
        within: None,
    })
}

/// A cited answer after thinking, in a non-ASCII text so character offsets
/// differ from bytes. Hand-built: the recordings are ASCII and none thinks.
fn cited_answer() -> (Value, Vec<Value>) {
    let first = json!({"start": 0, "end": 4, "text": "Café", "type": "TEXT_CONTENT",
        "content_index": 1, "sources": [{"type": "document", "id": "doc-1",
            "document": {"id": "doc-1", "title": "Menu", "text": "Café opens at nine."}}]});
    let second = json!({"start": 14, "end": 18, "text": "nine", "type": "TEXT_CONTENT",
        "content_index": 1, "sources": [
            {"type": "tool", "id": "clock_1:0", "tool_output": {"hour": 9}},
            {"type": "document", "id": "doc-2", "document": {"text": "9am"}}]});
    let thinking = json!({"start": 0, "end": 5, "text": "Check", "type": "THINKING_CONTENT",
        "content_index": 0, "sources": [{"type": "document", "id": "doc-1"}]});
    let message = json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Check the menu."},
            {"type": "text", "text": "Café opens at nine."}],
        "citations": [first, second, thinking]});
    (message, vec![first, second, thinking])
}

/// Text citations resolve to the characters they quote, each source
/// becoming a document or a tool output, the same in a whole reply and a
/// stream; a thinking citation stays in its item only.
#[test]
fn citations_resolve_the_same_unary_and_streamed() {
    let (message, citations) = cited_answer();
    let [first, second, thinking] = citations.as_slice() else {
        panic!("three citations");
    };
    let unary = command_a(&whole(message, "COMPLETE"), crate::wire::Mode::Unary);
    let streamed = command_a(
        &stream(
            vec![
                content_start(0, "thinking"),
                content_delta(0, "thinking", "Check the menu."),
                citation_start(0, thinking.clone()),
                content_end(0),
                content_start(1, "text"),
                content_delta(1, "text", "Café opens"),
                citation_start(1, first.clone()),
                content_delta(1, "text", " at nine."),
                citation_start(2, second.clone()),
                content_end(1),
            ],
            "COMPLETE",
        ),
        crate::wire::Mode::Streaming,
    );
    let expected = vec![(
        "Café opens at nine.".to_owned(),
        vec![
            (
                Some("Café".to_owned()),
                vec![document("doc-1").title("Menu")],
            ),
            (
                Some("nine".to_owned()),
                vec![
                    crate::message::Source::new(crate::message::SourceLocation::ToolOutput {
                        id: "clock_1:0".to_owned(),
                    }),
                    document("doc-2"),
                ],
            ),
        ],
    )];
    for response in [&unary, &streamed] {
        assert_eq!(typed_citations(response), expected);
        let Some(AssistantContent::Reasoning(reasoning)) = response.choice.first() else {
            panic!("thinking first: {:?}", response.choice);
        };
        assert_eq!(
            item(&reasoning.native).and_then(|item| item.get("citations").cloned()),
            Some(json!([thinking]))
        );
    }
}

/// A citation whose quoted text is not at its offsets, or that names no
/// offsets, is dropped from the block's citations and kept in its item.
#[test]
fn a_citation_that_does_not_match_its_text_stays_in_the_item() {
    let wrong = citation(0, 3, "doc-1", json!({}));
    let unplaced = json!({"text": "Sky", "sources": [], "type": "TEXT_CONTENT"});
    let response = command_a(
        &whole(
            json!({"role": "assistant",
                "content": [{"type": "text", "text": "Sky is green."}],
                "citations": [wrong, unplaced]}),
            "COMPLETE",
        ),
        crate::wire::Mode::Unary,
    );
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    assert!(text.citations().is_empty(), "{:?}", text.citations());
    assert_eq!(
        item(&text.native).map(|item| item.arr("citations").len()),
        Some(2)
    );
}

/// The cost is the billed units at the catalog's Command A prices, not the
/// larger `tokens` the usage reports, the same on both paths; without
/// billed units the fold prices the usage.
#[test]
fn the_cost_prices_the_billed_units() {
    // 56 input tokens at $2.50 and 20 output tokens at $10 per million.
    let priced = |input: f64, output: f64| {
        crate::completion::Cost::from_parts(input * 2.5 / 1e6, output * 10.0 / 1e6, 0.0, 0.0)
    };
    let billed = priced(56.0, 20.0);
    let message = json!({"role": "assistant", "content": [{"type": "text", "text": "Hi."}]});
    let unary = command_a(&whole(message, "COMPLETE"), crate::wire::Mode::Unary);
    let streamed = command_a(
        &stream(
            vec![
                content_start(0, "text"),
                content_delta(0, "text", "Hi."),
                content_end(0),
            ],
            "COMPLETE",
        ),
        crate::wire::Mode::Streaming,
    );
    for response in [&unary, &streamed] {
        assert_eq!(response.usage.input_tokens, Some(1706));
        assert_eq!(response.usage.cost, Some(billed));
    }

    let tokens_only = json!({"id": "resp_1", "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
        "usage": {"tokens": {"input_tokens": 100, "output_tokens": 10}}});
    let response = command_a(&[tokens_only], crate::wire::Mode::Unary);
    assert_eq!(response.usage.cost, Some(priced(100.0, 10.0)));
}

/// A plan-less reply rebuilds no `tool_plan`, as its unary body states
/// none, so the extras read `None` both ways. The skeleton is Cohere's
/// recorded `message-start`
/// (`cohere/native/documents_ground_a_streamed_answer_with_citations.yaml`).
#[test]
fn a_streamed_reply_without_a_plan_rebuilds_no_tool_plan() {
    use crate::completion::ReplyExtras;
    let start = json!({"id": "c1", "type": "message-start", "delta": {"message": {
        "citations": [], "content": [], "role": "assistant", "tool_calls": [], "tool_plan": ""}}});
    let frames = vec![
        start,
        content_start(0, "text"),
        content_delta(0, "text", "hello"),
        content_end(0),
        json!({"type": "message-end", "delta": {"finish_reason": "COMPLETE", "usage": usage()}}),
    ];
    let streamed = rebuilt(&frames);
    assert_eq!(streamed.pointer("/message/tool_plan"), None, "{streamed}");
    let unary = json!({"id": "c1", "finish_reason": "COMPLETE", "usage": usage(),
        "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}});
    let api = crate::message::Api::from_static(crate::providers::cohere::chat::API);
    let extras = |raw: &Value| {
        crate::providers::cohere::extension::CohereExtras::from_reply(&api, raw)
            .expect("the extras read")
            .tool_plan
    };
    assert_eq!(extras(&streamed), None);
    assert_eq!(extras(&streamed), extras(&unary));

    let mut seeded = frames.clone();
    if let Some(plan) = seeded
        .first_mut()
        .and_then(|start| start.pointer_mut("/delta/message/tool_plan"))
    {
        *plan = json!("I will ");
    }
    seeded.insert(1, plan_delta("look."));
    assert_eq!(
        rebuilt(&seeded).pointer("/message/tool_plan"),
        Some(&json!("I will look."))
    );
}
