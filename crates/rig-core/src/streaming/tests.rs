use std::time::Duration;

use super::*;
use crate::completion::FinishReason;
use crate::message::{Reasoning, ReasoningContent, Text, ToolCall};
use crate::providers::internal::adapter::AdapterOutput;
use async_stream::stream;
use futures::StreamExt;
use tokio::time::sleep;

/// Provider descriptor used by the mock streams in this module.
const TEST_PROVIDER: &str = "test-provider";

/// Fixture params: the JSON literal is always a non-empty object.
fn fixture_params(value: serde_json::Value) -> crate::message::AdditionalParams {
    crate::message::AdditionalParams::try_from_value(value)
        .expect("fixture params must be a JSON object")
        .expect("fixture params must carry data")
}

/// Terminal record with a known total-token count.
fn mock_final_with_total_tokens(total_tokens: u64) -> StreamFinal {
    let mut usage = Usage::new();
    usage.total_tokens = total_tokens;
    StreamFinal::new(TEST_PROVIDER, usage)
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn to_stream_result(
    stream: impl futures::Stream<Item = Result<StreamEvent, CompletionError>> + Send + 'static,
) -> StreamingResult {
    Box::pin(stream)
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn to_stream_result(
    stream: impl futures::Stream<Item = Result<StreamEvent, CompletionError>> + 'static,
) -> StreamingResult {
    Box::pin(stream)
}

/// Script a provider's output through the same helpers adapters use, so
/// the scripted events speak exactly the grammar a wire would.
fn script(build: impl FnOnce(&mut AdapterOutput)) -> Vec<Result<StreamEvent, CompletionError>> {
    let mut out = AdapterOutput::new();
    build(&mut out);
    out.into_items()
}

/// A stream over a scripted event sequence.
fn scripted(build: impl FnOnce(&mut AdapterOutput)) -> StreamingCompletionResponse {
    StreamingCompletionResponse::stream(
        TEST_PROVIDER,
        to_stream_result(futures::stream::iter(script(build))),
    )
}

/// A whole tool call under a wire key, the key doubling as the tool id.
fn whole_call(id: &str, name: &str, arguments: serde_json::Value) -> (BlockId, ToolCallEnd) {
    (
        BlockId::wire(id),
        ToolCallEnd::whole(name, arguments).with_tool_id(id),
    )
}

/// The text a delta carries, if the event is a text delta.
fn text_of(event: &StreamEvent) -> Option<&str> {
    match event {
        StreamEvent::BlockDelta {
            delta: Delta::Text { text },
            ..
        } => Some(text),
        _ => None,
    }
}

/// The completed reasoning block an end published, with its block id.
fn completed_reasoning(event: &StreamEvent) -> Option<(BlockId, Reasoning)> {
    match event {
        StreamEvent::BlockEnd {
            id,
            block: Some(AssistantContent::Reasoning(reasoning)),
            ..
        } => Some((id.clone(), reasoning.clone())),
        _ => None,
    }
}

/// The reasoning delta's block id, if the event is one.
fn reasoning_delta_id(event: &StreamEvent) -> Option<BlockId> {
    match event {
        StreamEvent::BlockDelta {
            id,
            delta: Delta::Reasoning { .. },
        } => Some(id.clone()),
        _ => None,
    }
}

fn create_mock_stream() -> StreamingCompletionResponse {
    let items = script(|out| {
        out.text("hello 1");
        out.text("hello 2");
        out.text("hello 3");
        out.final_record(mock_final_with_total_tokens(15));
    });
    let stream = stream! {
        for item in items {
            yield item;
            sleep(Duration::from_millis(50)).await;
        }
    };

    StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(stream))
}

/// #2258 review P3: non-yielding events (duplicate terminal records
/// here — the first one latched) drive the `poll_next` loop instead of
/// synchronous self-recursion, so a long run of them cannot grow the
/// stack. Pre-fix, each of these frames was one recursive `poll_next`
/// stack frame and a run this long overflowed in debug builds.
#[tokio::test]
async fn a_long_run_of_non_yielding_events_does_not_grow_the_stack() {
    let raw = stream! {
        for n in 0..50_000u32 {
            yield Ok(StreamEvent::BlockStart {
                id: BlockId::wire(format!("msg_{n}")),
                kind: BlockKind::Message,
            });
        }
        yield Ok(StreamEvent::text(BlockId::wire("t"), "done"));
        yield Ok(StreamEvent::Final(mock_final_with_total_tokens(1)));
        for _ in 0..50_000u32 {
            yield Ok(StreamEvent::Final(mock_final_with_total_tokens(99)));
        }
    };
    let mut stream = StreamingCompletionResponse::stream(TEST_PROVIDER, to_stream_result(raw));

    let mut texts = Vec::new();
    let mut terminals = 0;
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            event @ StreamEvent::BlockDelta { .. } => {
                texts.extend(text_of(&event).map(str::to_owned));
            }
            StreamEvent::Final(_) => terminals += 1,
            _ => {}
        }
    }
    assert_eq!(texts, vec!["done".to_string()]);
    assert_eq!(
        terminals, 1,
        "the first terminal latched; duplicates never yield"
    );
    assert_eq!(stream.usage().total_tokens, 1);
    // The last id recorded wins.
    assert_eq!(stream.message_id.as_deref(), Some("msg_49999"));
}

/// A stream that never saw a message-id block takes all three identity
/// axes from the terminal record.
#[tokio::test]
async fn stream_identity_falls_back_to_the_terminal_records_ids() {
    let mut stream = scripted(|out| {
        out.text("done");
        out.final_record(
            mock_final_with_total_tokens(1)
                .with_message_id("msg_terminal")
                .with_response_id("resp_1")
                .with_provider_request_id("req_1"),
        );
    });
    while stream.next().await.is_some() {}

    assert_eq!(
        stream.identity(),
        crate::completion::ResponseIdentity {
            message_id: Some("msg_terminal".to_string()),
            response_id: Some("resp_1".to_string()),
            provider_request_id: Some("req_1".to_string()),
        }
    );
}

/// An explicit message-id block outranks the terminal record's message
/// id; the response-scoped and transport ids still come from the terminal.
#[tokio::test]
async fn stream_identity_prefers_an_explicit_message_id_event() {
    let mut stream = scripted(|out| {
        out.message_id("msg_event");
        out.text("done");
        out.final_record(
            mock_final_with_total_tokens(1)
                .with_message_id("msg_terminal")
                .with_response_id("resp_1"),
        );
    });
    while stream.next().await.is_some() {}

    assert_eq!(
        stream.identity(),
        crate::completion::ResponseIdentity {
            message_id: Some("msg_event".to_string()),
            response_id: Some("resp_1".to_string()),
            provider_request_id: None,
        }
    );
}

fn create_reasoning_stream() -> StreamingCompletionResponse {
    scripted(|out| {
        out.reasoning_block(
            BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            ReasoningContent::Text {
                text: "step one".to_string(),
                signature: Some("sig_1".to_string()),
            },
        );
        out.text("final answer");
        out.final_record(mock_final_with_total_tokens(5));
    })
}

fn create_reasoning_only_stream() -> StreamingCompletionResponse {
    scripted(|out| {
        out.reasoning_block(
            BlockId::wire("rs_only"),
            non_empty_id("rs_only"),
            ReasoningContent::Summary("hidden summary".to_string()),
        );
        out.final_record(mock_final_with_total_tokens(2));
    })
}

fn create_interleaved_stream() -> StreamingCompletionResponse {
    scripted(|out| {
        out.reasoning_block(
            BlockId::wire("rs_interleaved"),
            non_empty_id("rs_interleaved"),
            ReasoningContent::Text {
                text: "chain-of-thought".to_string(),
                signature: None,
            },
        );
        out.text("final-text");
        let (id, end) = whole_call("tool_1", "mock_tool", serde_json::json!({"arg": 1}));
        out.tool_call(id, end);
        out.final_record(mock_final_with_total_tokens(3));
    })
}

fn create_text_tool_text_stream() -> StreamingCompletionResponse {
    scripted(|out| {
        out.text("first");
        let (id, end) = whole_call("tool_split", "mock_tool", serde_json::json!({"arg": "x"}));
        out.tool_call(id, end);
        out.text("second");
        out.final_record(mock_final_with_total_tokens(3));
    })
}

fn create_text_metadata_stream() -> StreamingCompletionResponse {
    scripted(|out| {
        out.text_start(BlockId::wire("block-0"), None);
        out.text("first");
        out.text_meta(fixture_params(serde_json::json!({
            "citations": [{
                "type": "char_location",
                "cited_text": "First citation.",
                "document_index": 0,
                "start_char_index": 0,
                "end_char_index": 15
            }]
        })));
        out.text_meta(fixture_params(serde_json::json!({
            "citations": [{
                "type": "char_location",
                "cited_text": "Second citation.",
                "document_index": 0,
                "start_char_index": 16,
                "end_char_index": 32
            }]
        })));
        out.text_start(
            BlockId::wire("block-1"),
            crate::message::AdditionalParams::try_from_value(serde_json::json!({
                "block": 2
            }))
            .expect("object params"),
        );
        out.text("second");
        out.final_record(mock_final_with_total_tokens(3));
    })
}

#[tokio::test]
async fn finish_derives_usage_from_final_response() {
    let mut stream = create_mock_stream();

    // Drain the stream so the final response (and its usage) is captured.
    while stream.next().await.is_some() {}

    // usage() surfaces the final response's token usage...
    assert_eq!(stream.usage().total_tokens, 15);

    // ...and finishing carries it instead of a zero sentinel.
    let response: CompletionResponse = stream.finish();
    assert_eq!(response.usage.total_tokens, 15);
    assert_eq!(response.provider, TEST_PROVIDER);
}

/// Regression (rig#2265): the transport request id captured on the
/// terminal record must survive stream→`CompletionResponse` conversion,
/// exactly like the response id, usage, finish reason, and model do.
#[tokio::test]
async fn finish_carries_the_terminal_request_id() {
    let mut stream = scripted(|out| {
        out.text("hi");
        out.final_record(
            StreamFinal::new(TEST_PROVIDER, Usage::new())
                .with_response_id("resp_1")
                .with_provider_request_id("req_transport_1"),
        );
    });
    while stream.next().await.is_some() {}

    let response: CompletionResponse = stream.finish();
    assert_eq!(response.response_id.as_deref(), Some("resp_1"));
    assert_eq!(
        response.provider_request_id.as_deref(),
        Some("req_transport_1")
    );
}

#[tokio::test]
async fn a_stream_without_a_terminal_record_still_names_its_provider() {
    // The provider is known when the stream is opened, so a stream that
    // errors or is truncated before its terminal record must not degrade
    // `provider` to an empty string — every other missing value has a
    // documented sentinel (`Usage::new`, `None`) and this one should too.
    let mut stream = scripted(|out| {
        out.text("truncated");
    });
    while stream.next().await.is_some() {}

    // No terminal record was ever yielded, so none may be synthesized.
    assert!(stream.response.is_none());

    let response: CompletionResponse = stream.finish();
    assert_eq!(response.provider, TEST_PROVIDER);
    assert_eq!(response.usage, Usage::new());
    assert_eq!(response.finish_reason(), None);
    assert_eq!(response.model, None);
}

#[tokio::test]
async fn a_stream_that_errors_mid_stream_keeps_content_and_omits_the_terminal() {
    // A transport error after some content must forward the error, keep
    // the content already aggregated, and never fabricate a terminal
    // record the provider did not send.
    let mut stream = scripted(|out| {
        out.text("partial");
        out.error(CompletionError::ProviderError(
            "connection reset".to_string(),
        ));
    });

    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if item.is_err() {
            saw_error = true;
        }
    }
    assert!(saw_error, "the mid-stream error must be forwarded");

    // No StreamFinal may be synthesized for the aborted stream...
    assert!(stream.response.is_none());

    // ...but the content delivered before the error is preserved.
    assert_eq!(
        stream.snapshot().first(),
        Some(&AssistantContent::text("partial".to_string())),
    );
}

#[tokio::test]
async fn a_stop_that_carried_a_tool_call_is_upgraded_to_tool_calls() {
    // Several gateways report a plain `stop` on a tool-calling turn. The
    // streaming path must reconcile it exactly as the unary path does.
    let mut stream = scripted(|out| {
        let (id, end) = whole_call("call_1", "lookup", serde_json::json!({}));
        out.tool_call(id, end);
        out.final_record(
            StreamFinal::new(TEST_PROVIDER, Usage::new()).with_finish_reason(FinishReason::Stop),
        );
    });
    while stream.next().await.is_some() {}

    assert_eq!(
        stream
            .response
            .as_ref()
            .and_then(|final_record| final_record.finish_reason.clone()),
        Some(FinishReason::ToolCalls),
    );
}

#[tokio::test]
async fn a_stop_without_tool_calls_is_left_alone() {
    let mut stream = scripted(|out| {
        out.text("done");
        out.final_record(
            StreamFinal::new(TEST_PROVIDER, Usage::new()).with_finish_reason(FinishReason::Stop),
        );
    });
    while stream.next().await.is_some() {}

    assert_eq!(
        stream
            .response
            .as_ref()
            .and_then(|final_record| final_record.finish_reason.clone()),
        Some(FinishReason::Stop),
    );
}

#[test]
fn stream_final_round_trips_and_is_distinguishable_from_unknown_content() {
    let final_record = StreamFinal::new(
        "example",
        Usage {
            input_tokens: 4,
            output_tokens: 6,
            total_tokens: 10,
            cached_input_tokens: 1,
            cache_creation_input_tokens: 2,
            tool_use_prompt_tokens: 3,
            reasoning_tokens: 4,
        },
    )
    .with_finish_reason(FinishReason::Other("future_reason".to_owned()))
    .with_message_id("msg_123")
    .with_model("provider-model-v2");

    let encoded = serde_json::to_value(StreamEvent::Final(final_record.clone()))
        .expect("serialize final item");
    assert_eq!(encoded["event"], serde_json::json!("final"));

    let decoded = serde_json::from_value::<StreamEvent>(encoded).expect("deserialize final item");
    assert_eq!(decoded, StreamEvent::Final(final_record));

    // An unmodeled provider item lands in `Unknown` and round-trips as
    // such — it is never mistaken for a terminal record, and a bare
    // provider payload without the event tag does not decode as one.
    let provider_item = serde_json::json!({
        "provider_native_event": "future_terminal",
        "usage": {"total_tokens": 10}
    });
    let unknown = StreamEvent::Unknown(provider_item.clone().into());
    let encoded = serde_json::to_value(&unknown).expect("serialize unknown item");
    assert_eq!(encoded["event"], serde_json::json!("unknown"));
    let decoded = serde_json::from_value::<StreamEvent>(encoded).expect("deserialize unknown item");
    assert_eq!(decoded, unknown);
    assert!(
        serde_json::from_value::<StreamEvent>(provider_item).is_err(),
        "an untagged provider payload is not a stream event"
    );
}

/// Deserialization funnels through `new` + the setters, so the invariants
/// hold on persisted values too: a `""` identifier comes back as `None`.
#[test]
fn deserializing_stream_final_filters_empty_identifiers() {
    let decoded = serde_json::from_value::<StreamFinal>(serde_json::json!({
        "usage": Usage::new(),
        "message_id": "",
        "response_id": "",
        "model": "",
        "provider": "example",
    }))
    .expect("deserialize terminal record");

    assert_eq!(decoded.message_id, None);
    assert_eq!(decoded.response_id, None);
    assert_eq!(decoded.model, None);
}

/// A provider-native terminal type standing in for the real ones: it
/// carries a field the normalized record does not model, so the test can
/// tell "the raw payload is the terminal record" from "some value was
/// attached".
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct ProviderTerminal {
    usage: Usage,
    provider_only: String,
}

/// What an adapter does with its native terminal: map the normalized
/// fields and serialize the record onto `raw`.
fn provider_terminal_stream() -> StreamingResult {
    let terminal = ProviderTerminal {
        usage: Usage {
            input_tokens: 3,
            output_tokens: 5,
            total_tokens: 8,
            ..Usage::new()
        },
        provider_only: "kept".to_string(),
    };
    to_stream_result(futures::stream::iter(script(|out| {
        out.text("done");
        let raw = serde_json::to_value(&terminal).expect("serialize terminal");
        out.final_record(StreamFinal::new(TEST_PROVIDER, terminal.usage).with_raw(raw));
    })))
}

async fn drain(events: StreamingResult) -> StreamFinal {
    let mut stream = StreamingCompletionResponse::stream(TEST_PROVIDER, events);
    while stream.next().await.is_some() {}
    stream
        .response
        .expect("stream should end with a terminal record")
}

/// The load-bearing streaming test: `raw` is the provider's terminal
/// record serialized — it deserializes back into the provider's own type
/// and re-serializes equal — and the normalized fields are what the
/// adapter produced.
#[tokio::test]
async fn the_terminal_record_captures_the_providers_raw_terminal() {
    let final_record = drain(provider_terminal_stream()).await;
    let raw = &final_record.raw;

    let typed = ProviderTerminal::deserialize(raw).expect("raw is the provider's terminal");
    assert_eq!(typed.provider_only, "kept");
    assert_eq!(&serde_json::to_value(&typed).expect("re-serialize"), raw);

    assert_eq!(final_record.usage.total_tokens, 8);
    assert_eq!(final_record.provider, TEST_PROVIDER);
    assert_eq!(final_record.finish_reason, None);
}

/// Finish-reason reconciliation is unchanged by capture: a `stop` that
/// carried a tool call is still upgraded, with `raw` attached.
#[tokio::test]
async fn finish_reason_is_reconciled_with_raw_attached() {
    let events = to_stream_result(futures::stream::iter(script(|out| {
        let (id, end) = whole_call("call_1", "lookup", serde_json::json!({}));
        out.tool_call(id, end);
        let usage = Usage::new();
        out.final_record(
            StreamFinal::new(TEST_PROVIDER, usage)
                .with_finish_reason(FinishReason::Stop)
                .with_raw(serde_json::to_value(usage).expect("serialize usage")),
        );
    })));
    let final_record = drain(events).await;
    assert_eq!(final_record.finish_reason, Some(FinishReason::ToolCalls));
    assert!(!final_record.raw.is_null());
}

/// The deserialization mirror carries `raw`: a terminal record with a
/// captured payload survives serialize → deserialize with the payload
/// intact, both bare and wrapped in `StreamEvent::Final` (the shape the
/// agent forwards). A record serialized before the field existed still
/// loads, with `raw` unset.
#[test]
fn stream_final_raw_round_trips_through_serde_mirror() {
    let payload = serde_json::json!({
        "usage": {"total_tokens": 8},
        "provider_only": "kept"
    });
    let final_record = StreamFinal::new("example", Usage::new())
        .with_message_id("msg_123")
        .with_raw(payload.clone());

    let encoded = serde_json::to_value(&final_record).expect("serialize");
    assert_eq!(encoded["raw"], payload);
    let decoded = serde_json::from_value::<StreamFinal>(encoded.clone()).expect("deserialize");
    assert_eq!(decoded.raw, payload);
    assert_eq!(decoded, final_record);
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );

    let wrapped = StreamEvent::Final(final_record);
    let encoded = serde_json::to_value(&wrapped).expect("serialize wrapped");
    let decoded = serde_json::from_value::<StreamEvent>(encoded).expect("deserialize wrapped");
    assert_eq!(decoded, wrapped);

    // Pre-field JSON: no `raw` key.
    let legacy = serde_json::json!({
        "usage": serde_json::to_value(Usage::new()).unwrap(),
        "provider": "example"
    });
    let decoded = serde_json::from_value::<StreamFinal>(legacy).expect("legacy loads");
    assert!(decoded.raw.is_null());

    // Unset `raw` is not written, so a record without capture serializes
    // exactly as it did before the field existed.
    let bare = serde_json::to_value(StreamFinal::new("example", Usage::new())).unwrap();
    assert!(bare.get("raw").is_none());
}

/// The deserialization mirror must not change the wire format: a fully
/// populated terminal record round-trips to byte-identical JSON.
#[test]
fn stream_final_serde_round_trip_is_identity() {
    let final_record = StreamFinal::new(
        "example",
        Usage {
            input_tokens: 4,
            output_tokens: 6,
            total_tokens: 10,
            cached_input_tokens: 1,
            cache_creation_input_tokens: 2,
            tool_use_prompt_tokens: 3,
            reasoning_tokens: 4,
        },
    )
    .with_finish_reason(FinishReason::Stop)
    .with_message_id("msg_123")
    .with_response_id("resp_456")
    .with_model("provider-model-v2");

    let encoded = serde_json::to_value(&final_record).expect("serialize terminal record");

    let decoded = serde_json::from_value::<StreamFinal>(encoded.clone()).expect("deserialize");
    assert_eq!(decoded, final_record);
    assert_eq!(
        serde_json::to_value(&decoded).expect("re-serialize"),
        encoded
    );
}

#[tokio::test]
async fn usage_is_zero_sentinel_before_final_response() {
    // A stream that never yields a FinalResponse reports the zero sentinel.
    let stream = scripted(|out| {
        out.text("no final response");
    });
    assert_eq!(stream.usage().total_tokens, 0);
}

#[tokio::test]
async fn test_stream_cancellation() {
    let mut stream = create_mock_stream();

    println!("Response: ");
    let mut chunk_count = 0;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamEvent::BlockStart { id, kind }) => {
                println!("\nBlock start: id={id:?}, kind={kind:?}");
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            }) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                chunk_count += 1;
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Reasoning { text },
                ..
            }) => {
                println!("Reasoning delta: {text}");
                chunk_count += 1;
            }
            Ok(StreamEvent::BlockDelta { id, delta }) => {
                println!("\nBlock delta: id={id:?}, delta={delta:?}");
                chunk_count += 1;
            }
            Ok(StreamEvent::BlockEnd {
                id,
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                println!("\nTool Call: {tool_call:?}, block_id={id:?}");
                chunk_count += 1;
            }
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::Reasoning(reasoning)),
                ..
            }) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(StreamEvent::BlockEnd { id, end, block }) => {
                println!("\nBlock end: id={id:?}, end={end:?}, block={block:?}");
            }
            Ok(StreamEvent::Final(res)) => {
                println!("\nFinal response: {res:?}");
            }
            Ok(StreamEvent::Unknown(value)) => {
                println!("\nUnknown item: {value:?}");
                chunk_count += 1;
            }
            Err(e) => {
                eprintln!("Error: {e:?}");
                break;
            }
        }

        if chunk_count >= 2 {
            println!("\nCancelling stream...");
            stream.cancel();
            println!("Stream cancelled.");
            break;
        }
    }

    let next_chunk = stream.next().await;
    assert!(
        next_chunk.is_none(),
        "Expected no further chunks after cancellation, got {next_chunk:?}"
    );
}

#[tokio::test]
async fn test_stream_pause_resume() {
    let stream = create_mock_stream();

    // Test pause
    stream.pause();
    assert!(stream.is_paused());

    // Test resume
    stream.resume();
    assert!(!stream.is_paused());
}

/// #2258 H7: a paused stream parks on the pause channel instead of
/// re-waking itself, which turned a pause into a busy poll loop. The
/// `is_woken` assertion is the pin: pre-fix the paused poll woke the task
/// immediately, so it failed.
///
/// Not inducible from a recorded provider turn — pause/resume is
/// consumer-side control flow with no wire representation.
#[tokio::test]
async fn a_paused_stream_parks_until_resume_instead_of_busy_waking() {
    let stream = StreamingCompletionResponse::stream(
        TEST_PROVIDER,
        to_stream_result(stream! {
            yield Ok(StreamEvent::text(BlockId::wire("t"), "hello"));
        }),
    );
    let resume = stream.pause_control.clone();
    stream.pause();

    let mut task = tokio_test::task::spawn(stream);
    assert!(
        task.poll_next().is_pending(),
        "a paused stream yields nothing"
    );
    assert!(
        !task.is_woken(),
        "a paused stream must idle, not re-wake itself"
    );

    resume.resume();
    assert!(task.is_woken(), "resuming must wake the parked stream");
    assert!(matches!(
        task.poll_next(),
        Poll::Ready(Some(Ok(event))) if text_of(&event) == Some("hello")
    ));
}

/// #2258 B7: cancelling a paused stream must not deadlock — the consumer
/// parked on the pause channel observes the termination because
/// `cancel()` also resumes.
#[tokio::test]
async fn cancelling_a_paused_stream_terminates_instead_of_deadlocking() {
    let mut stream = create_mock_stream();
    stream.pause();
    stream.cancel();
    assert!(
        !stream.is_paused(),
        "cancel must lift the pause so the termination is observable"
    );
    assert!(
        stream.next().await.is_none(),
        "a cancelled stream terminates"
    );
}

/// #2258 H6: a second poll of a drained stream must not disturb the
/// aggregated choice — pre-fix the re-poll re-ran the destructive
/// accumulator finish and replaced a fully aggregated choice with the
/// empty-text fallback.
///
/// Not inducible from a recorded provider turn: re-polling a terminated
/// stream is consumer behavior (`Stream` permits it, and combinators do
/// it), independent of any wire.
#[tokio::test]
async fn re_polling_a_drained_stream_preserves_the_aggregated_choice() {
    let mut stream = create_mock_stream();
    while stream.next().await.is_some() {}

    let drained = stream.snapshot();
    assert_eq!(
        drained,
        vec![AssistantContent::text("hello 1hello 2hello 3")]
    );

    for _ in 0..3 {
        assert!(
            stream.next().await.is_none(),
            "a drained stream stays drained"
        );
    }
    assert_eq!(
        stream.snapshot(),
        drained,
        "re-polling must not disturb the aggregated choice"
    );

    // Finishing into a unary response still carries the content.
    let response: CompletionResponse = stream.finish();
    assert_eq!(response.choice.into_iter().collect::<Vec<_>>(), drained);
}

/// #2258 H8: a `ProviderError` whose text happens to contain "aborted"
/// is an error like any other. It used to be swallowed as clean EOF,
/// discarding both the failure and the content streamed before it.
///
/// Not inducible from a recorded provider turn: no in-tree provider emits
/// this sentinel, and real cancellation arrives as `Ready(None)` through
/// `Abortable` rather than as an error item.
#[tokio::test]
async fn a_provider_error_mentioning_aborted_reaches_the_consumer() {
    let mut stream = scripted(|out| {
        out.text("partial");
        out.error(CompletionError::ProviderError(
            "upstream aborted the request".to_string(),
        ));
    });

    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            errors.push(err.to_string());
        }
    }
    assert_eq!(errors.len(), 1, "the error must not be swallowed");
    assert!(errors[0].contains("upstream aborted the request"));

    // The content streamed before the failure is still aggregated.
    assert_eq!(
        stream.snapshot().first(),
        Some(&AssistantContent::text("partial".to_string()))
    );
    assert!(stream.response.is_none());
}

/// #2258 F1, at the stream boundary: a wire that fragments a call's input
/// and then restates it as one complete block must publish the completed
/// call under the id its deltas already published — the correlation
/// contract on a `BlockEnd` carrying a completed call. Pre-fix the
/// completed call carried a fresh id no delta ever mentioned.
///
/// Not inducible from a recorded provider turn: no in-tree wire mixes the
/// two shapes for one call, though out-of-tree adapters can.
#[tokio::test]
async fn a_full_tool_call_correlates_with_the_deltas_of_the_same_id() {
    let mut stream = scripted(|out| {
        out.tool_name(&BlockId::wire("tc1"), "add");
        out.tool_arguments(&BlockId::wire("tc1"), "{\"x\":1}");
        let (id, end) = whole_call("tc1", "add", serde_json::json!({"x": 1}));
        out.tool_call(id, end);
        // The trailing end event for a call the full block already delivered.
        out.push(Ok(StreamEvent::BlockEnd {
            id: BlockId::wire("tc1"),
            end: BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Drop)),
            block: None,
        }));
        out.final_record(mock_final_with_total_tokens(1));
    });

    let mut delta_ids = Vec::new();
    let mut completed_ids = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamEvent::BlockDelta {
                id,
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
            } => delta_ids.push(id),
            StreamEvent::BlockEnd {
                id,
                block: Some(AssistantContent::ToolCall(_)),
                ..
            } => completed_ids.push(id),
            _ => {}
        }
    }

    assert_eq!(delta_ids.len(), 2);
    assert_eq!(delta_ids[0], delta_ids[1], "one call, one block id");
    assert_eq!(
        completed_ids,
        vec![delta_ids[0].clone()],
        "the completed call must carry the id its deltas published"
    );

    // The trailing end event for a call a full block already delivered
    // finalizes nothing: exactly one tool call reaches the choice.
    let choice = stream.snapshot();
    let tool_calls: Vec<&ToolCall> = choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect();
    assert_eq!(tool_calls.len(), 1, "got {choice:?}");
}

#[tokio::test]
async fn test_stream_aggregates_reasoning_content() {
    let mut stream = create_reasoning_stream();
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();

    assert!(choice_items.iter().any(|item| matches!(
        item,
        AssistantContent::Reasoning(Reasoning {
            id: Some(id),
            content
        }) if id == "rs_1"
            && matches!(
                content.first(),
                Some(ReasoningContent::Text {
                    text,
                    signature: Some(signature)
                }) if text == "step one" && signature == "sig_1"
            )
    )));
}

/// A full reasoning block replaces its own delta accumulation, so the
/// aggregated choice matches unary normalization of the same turn: one
/// reasoning item carrying the completed block, not delta-plus-duplicate.
#[tokio::test]
async fn full_reasoning_block_supersedes_its_accumulated_deltas() {
    let mut stream = scripted(|out| {
        out.reasoning_delta(&BlockId::wire("rs_1"), non_empty_id("rs_1"), "partial ");
        out.reasoning_block(
            BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            ReasoningContent::Text {
                text: "the complete chain".to_string(),
                signature: Some("sig_1".to_string()),
            },
        );
        out.final_record(mock_final_with_total_tokens(2));
    });
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    let reasoning_items: Vec<&Reasoning> = choice_items
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();

    assert_eq!(reasoning_items.len(), 1, "got {choice_items:?}");
    let reasoning = reasoning_items.first().expect("one reasoning item");
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert!(matches!(
        reasoning.content.first(),
        Some(ReasoningContent::Text { text, signature: Some(signature) })
            if text == "the complete chain" && signature == "sig_1"
    ));
}

/// A full block whose ID differs from the accumulating item's ID is a
/// distinct reasoning item and is appended, not a replacement.
#[tokio::test]
async fn full_reasoning_block_with_a_different_id_appends() {
    let mut stream = scripted(|out| {
        out.reasoning_delta(
            &BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            "first item deltas",
        );
        out.reasoning_block(
            BlockId::wire("rs_2"),
            non_empty_id("rs_2"),
            ReasoningContent::Text {
                text: "a different item".to_string(),
                signature: None,
            },
        );
        out.final_record(mock_final_with_total_tokens(2));
    });
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    let reasoning_ids: Vec<Option<&str>> = choice_items
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.id.as_deref()),
            _ => None,
        })
        .collect();

    assert_eq!(reasoning_ids, vec![Some("rs_1"), Some("rs_2")]);
}

/// A bare end the wire actually sent yields the completed block (the
/// wire announced the boundary and the consumer must see it — e.g.
/// anthropic's `content_block_stop` on an unsigned thinking block); a
/// bare end an adapter synthesized stays silent.
#[tokio::test]
async fn wire_sent_bare_end_yields_the_completed_block_synthesized_stays_silent() {
    let run = |wire_sent: bool| async move {
        let mut stream = scripted(|out| {
            let key = BlockId::minted(MintKind::Block, 0);
            out.reasoning_delta(&key, None, "unsigned thoughts");
            out.reasoning_end(key, None, None, wire_sent);
            out.final_record(mock_final_with_total_tokens(2));
        });
        let mut completed = Vec::new();
        while let Some(item) = stream.next().await {
            if let Some((_, reasoning)) = completed_reasoning(&item.expect("stream item")) {
                completed.push(reasoning);
            }
        }
        completed
    };

    let wire = run(true).await;
    assert_eq!(wire.len(), 1, "a wire-sent end announces the boundary");
    assert!(matches!(
        wire[0].content.first(),
        Some(ReasoningContent::Text { text, signature: None }) if text == "unsigned thoughts"
    ));

    let synthesized = run(false).await;
    assert!(
        synthesized.is_empty(),
        "a synthesized bare end fabricates nothing: {synthesized:?}"
    );
}

/// The completed reasoning event restates the block id its deltas
/// carried (the anthropic shape: id-less deltas, wire-sent bare stop),
/// keeping it distinct from the durable provider handle, which stays
/// absent.
#[tokio::test]
async fn completed_reasoning_restates_the_delta_block_id() {
    let key = || BlockId::minted(MintKind::Block, 0);
    let mut stream = scripted(|out| {
        out.reasoning_delta(&key(), None, "unsigned thoughts");
        out.reasoning_end(key(), None, None, true);
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut delta_ids = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        let event = item.expect("stream item");
        delta_ids.extend(reasoning_delta_id(&event));
        completed.extend(completed_reasoning(&event));
    }

    let (block, reasoning) = completed.first().expect("one completed block");
    assert_eq!(
        Some(block),
        delta_ids.first(),
        "the completed block restates its deltas' block id"
    );
    assert_eq!(
        reasoning.id, None,
        "no provider handle exists on this wire; the block id must not leak into it"
    );
}

/// On a signed end (the gemini shape) the completed event carries BOTH
/// identities as distinct values: the block id matching the deltas, and
/// the durable provider handle in `reasoning.id`.
#[tokio::test]
async fn completed_reasoning_keeps_block_id_and_provider_handle_distinct() {
    let mut stream = scripted(|out| {
        out.reasoning_delta(
            &BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            "signed thoughts",
        );
        out.reasoning_end(BlockId::wire("rs_1"), None, Some("sig_1".to_string()), true);
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut delta_ids = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        let event = item.expect("stream item");
        delta_ids.extend(reasoning_delta_id(&event));
        completed.extend(completed_reasoning(&event));
    }

    let (block, reasoning) = completed.first().expect("one completed block");
    assert_eq!(Some(block), delta_ids.first());
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        *block,
        BlockId::wire("rs_1"),
        "the block id is the wire id the deltas streamed under"
    );
}

/// A trailing signature after a synthesized silent end restates the
/// deltas' block id (the gemini shape: thought deltas, visible text
/// forcing a synthesized boundary, then a bare `thoughtSignature`
/// frame). The suppressed end must not discard the part's identity —
/// a fresh id here strands the signed completion where the
/// streamed-turn assembler cannot match it, duplicating the part.
#[tokio::test]
async fn late_signature_after_synthesized_end_restates_the_delta_block_id() {
    let key = || BlockId::minted(MintKind::Reasoning, 0);
    let mut stream = scripted(|out| {
        out.reasoning_delta(&key(), None, "hidden thoughts");
        // The adapter saw visible text begin and synthesized a
        // silent boundary the wire never sent.
        out.reasoning_end(key(), None, None, false);
        out.text("visible");
        // The trailing signature frame closes the same part.
        out.reasoning_end(key(), None, Some("sig_late".to_string()), true);
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut delta_ids = Vec::new();
    let mut completed = Vec::new();
    while let Some(item) = stream.next().await {
        let event = item.expect("stream item");
        delta_ids.extend(reasoning_delta_id(&event));
        completed.extend(completed_reasoning(&event));
    }

    assert_eq!(completed.len(), 1, "one signed completion, no duplicate");
    let (block, reasoning) = completed.first().expect("one completed block");
    assert_eq!(
        Some(block),
        delta_ids.first(),
        "the signed completion restates the block id its deltas carried"
    );
    assert!(
        reasoning.content.iter().any(|content| matches!(
            content,
            ReasoningContent::Text { signature: Some(sig), .. } if sig == "sig_late"
        )),
        "the trailing signature landed on the completed part"
    );
}

/// A delta-less reasoning start under a reused key opens a NEW part
/// (sequence O9: the finished map must never leak the previous part
/// onto a distinct part) — even when that part closes with a
/// signature-only end. Both parts publish under the key they streamed
/// under.
#[tokio::test]
async fn a_delta_less_start_under_a_reused_key_opens_a_new_part() {
    let key = || BlockId::minted(MintKind::Reasoning, 0);
    let mut stream = scripted(|out| {
        out.reasoning_delta(&key(), None, "part one");
        out.reasoning_end(key(), None, None, true);
        out.reasoning_start(&key(), None);
        out.reasoning_end(key(), None, Some("sig2".to_string()), true);
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut completed_ids = Vec::new();
    while let Some(item) = stream.next().await {
        if let Some((id, _)) = completed_reasoning(&item.expect("stream item")) {
            completed_ids.push(id);
        }
    }

    assert_eq!(completed_ids.len(), 2, "two distinct parts complete");
    assert_eq!(
        completed_ids.first(),
        completed_ids.get(1),
        "both parts carry the key they streamed under; a wire that wants distinct \
         ids mints one per block at the adapter (MintedReasoningLifecycle)"
    );
}

/// Ending a part and streaming new deltas under the same accumulation
/// key opens a NEW part, published under the same key.
#[tokio::test]
async fn reused_accumulation_key_opens_a_new_part_after_an_end() {
    let key = || BlockId::minted(MintKind::Reasoning, 0);
    let mut stream = scripted(|out| {
        out.reasoning_delta(&key(), None, "first part");
        out.reasoning_end(key(), None, None, true);
        out.reasoning_delta(&key(), None, "second part");
        out.reasoning_end(key(), None, None, true);
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut completed_ids = Vec::new();
    while let Some(item) = stream.next().await {
        if let Some((id, _)) = completed_reasoning(&item.expect("stream item")) {
            completed_ids.push(id);
        }
    }

    assert_eq!(completed_ids.len(), 2, "two parts under the reused key");
    assert_eq!(
        completed_ids.first(),
        completed_ids.get(1),
        "a reused key opens a new part under the same key"
    );
    assert_eq!(
        stream.snapshot().len(),
        2,
        "two distinct parts in the choice"
    );
}

/// A whole-block reasoning event with no prior deltas still carries its
/// wire id as the block id, and two such parts never share one.
#[tokio::test]
async fn whole_block_reasoning_carries_its_wire_id() {
    let mut stream = scripted(|out| {
        out.reasoning_block(
            BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            ReasoningContent::Text {
                text: "first".to_string(),
                signature: None,
            },
        );
        out.reasoning_block(
            BlockId::wire("rs_2"),
            non_empty_id("rs_2"),
            ReasoningContent::Text {
                text: "second".to_string(),
                signature: None,
            },
        );
        out.final_record(mock_final_with_total_tokens(2));
    });

    let mut block_ids = Vec::new();
    while let Some(item) = stream.next().await {
        if let Some((id, _)) = completed_reasoning(&item.expect("stream item")) {
            block_ids.push(id);
        }
    }

    assert_eq!(
        block_ids,
        vec![BlockId::wire("rs_1"), BlockId::wire("rs_2")],
        "distinct parts must not share a block id"
    );
}

#[tokio::test]
async fn full_reasoning_block_supersedes_deltas_across_interleaved_output() {
    // Providers may emit the completed reasoning item after other output
    // (reasoning -> tool call -> completed block). The tool call clears
    // the active reasoning index, so replacement must fall back to the
    // by-ID scan rather than appending a duplicate.
    let mut stream = scripted(|out| {
        out.reasoning_delta(&BlockId::wire("rs_1"), non_empty_id("rs_1"), "partial ");
        let (id, end) = whole_call("call_1", "probe", serde_json::json!({}));
        out.tool_call(id, end);
        out.reasoning_block(
            BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            ReasoningContent::Text {
                text: "the full block".to_string(),
                signature: None,
            },
        );
        out.final_record(mock_final_with_total_tokens(2));
    });
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    let reasoning_items: Vec<&Reasoning> = choice_items
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();

    assert_eq!(
        reasoning_items.len(),
        1,
        "the full block must replace the delta-built item, not join it"
    );
    let only = reasoning_items.first().expect("one reasoning item");
    assert_eq!(only.id.as_deref(), Some("rs_1"));
    assert!(
        only.content.iter().any(|content| matches!(
            content,
            ReasoningContent::Text { text, .. } if text == "the full block"
        )),
        "the surviving item must carry the full block's content"
    );
}

#[tokio::test]
async fn minted_id_full_reasoning_block_does_not_clobber_a_wire_id_item() {
    // Ids are mandatory on the grammar; a provider-minted id (the
    // "reasoning-0"-style boundary fallback) is a distinct identity from
    // a wire-supplied one, so the block appends rather than overwriting
    // an unrelated item's deltas.
    let mut stream = scripted(|out| {
        out.reasoning_delta(
            &BlockId::wire("rs_1"),
            non_empty_id("rs_1"),
            "identified deltas",
        );
        out.reasoning_block(
            BlockId::wire("reasoning-0"),
            non_empty_id("reasoning-0"),
            ReasoningContent::Text {
                text: "anonymous block".to_string(),
                signature: None,
            },
        );
        out.final_record(mock_final_with_total_tokens(2));
    });
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    let reasoning_ids: Vec<Option<&str>> = choice_items
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.id.as_deref()),
            _ => None,
        })
        .collect();

    assert_eq!(reasoning_ids, vec![Some("rs_1"), Some("reasoning-0")]);
}

#[tokio::test]
async fn test_stream_reasoning_only_does_not_inject_empty_text() {
    let mut stream = create_reasoning_only_stream();
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    assert_eq!(choice_items.len(), 1);
    assert!(matches!(
        choice_items.first(),
        Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_only"
    ));
}

#[tokio::test]
async fn test_stream_aggregates_assistant_items_in_arrival_order() {
    let mut stream = create_interleaved_stream();
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    assert_eq!(choice_items.len(), 3);
    assert!(matches!(
        choice_items.first(),
        Some(AssistantContent::Reasoning(Reasoning { id: Some(id), .. })) if id == "rs_interleaved"
    ));
    assert!(matches!(
        choice_items.get(1),
        Some(AssistantContent::Text(Text { text, .. })) if text == "final-text"
    ));
    assert!(matches!(
        choice_items.get(2),
        Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_1"
    ));
}

#[tokio::test]
async fn unknown_choice_reaches_consumer_but_not_aggregated_choice() {
    let unknown = serde_json::json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
    });
    let yielded = unknown.clone();
    let mut stream = scripted(move |out| {
        out.unknown(yielded.into());
        out.text("done");
        out.final_record(mock_final_with_total_tokens(1));
    });

    let mut consumer_unknown = None;
    let mut consumer_text = String::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamEvent::Unknown(value) => consumer_unknown = Some(value),
            event => consumer_text.extend(text_of(&event)),
        }
    }

    // The consumer receives the unmodeled item verbatim ...
    assert_eq!(consumer_unknown.as_ref(), Some(&unknown.into()));
    assert_eq!(consumer_text, "done");

    // ... but it is structurally absent from the aggregated assistant choice
    // (the sole source of persisted history): only the text item remains.
    let choice_items = stream.snapshot();
    assert_eq!(choice_items.len(), 1);
    assert!(matches!(
        choice_items.first(),
        Some(AssistantContent::Text(Text { text, .. })) if text == "done"
    ));
}

#[tokio::test]
async fn test_stream_keeps_non_contiguous_text_chunks_split_by_tool_call() {
    let mut stream = create_text_tool_text_stream();
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    assert_eq!(choice_items.len(), 3);
    assert!(matches!(
        choice_items.first(),
        Some(AssistantContent::Text(Text { text, .. })) if text == "first"
    ));
    assert!(matches!(
        choice_items.get(1),
        Some(AssistantContent::ToolCall(ToolCall { id, .. })) if id == "tool_split"
    ));
    assert!(matches!(
        choice_items.get(2),
        Some(AssistantContent::Text(Text { text, .. })) if text == "second"
    ));
}

#[tokio::test]
async fn test_stream_preserves_text_additional_params() {
    let mut stream = create_text_metadata_stream();
    while stream.next().await.is_some() {}

    let choice_items = stream.snapshot();
    assert_eq!(choice_items.len(), 2);

    let Some(AssistantContent::Text(Text {
        text,
        additional_params: Some(additional_params),
    })) = choice_items.first()
    else {
        panic!("expected first text item with metadata");
    };
    assert_eq!(text, "first");
    assert_eq!(
        additional_params["citations"]
            .as_array()
            .expect("citations should be an array")
            .len(),
        2
    );

    let Some(AssistantContent::Text(Text {
        text,
        additional_params: Some(additional_params),
    })) = choice_items.get(1)
    else {
        panic!("expected second text item with metadata");
    };
    assert_eq!(text, "second");
    assert_eq!(additional_params["block"], 2);
}

/// The public stream fills `block` on the end that finalized a call, and
/// leaves it `None` on an end that dropped one.
#[tokio::test]
async fn a_tool_end_carries_the_completed_call_or_nothing_when_dropped() {
    let mut stream = scripted(|out| {
        out.tool_name(&BlockId::wire("ok"), "add");
        out.tool_arguments(&BlockId::wire("ok"), "{\"x\":1}");
        out.tool_end(
            BlockId::wire("ok"),
            ToolCallEnd::new(UnparseableToolInput::Drop),
        );
        out.tool_name(&BlockId::wire("truncated"), "add");
        out.tool_arguments(&BlockId::wire("truncated"), "{\"x\":");
        out.tool_end(
            BlockId::wire("truncated"),
            ToolCallEnd::new(UnparseableToolInput::Drop),
        );
        out.final_record(mock_final_with_total_tokens(1));
    });

    let mut ends = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamEvent::BlockEnd { id, block, .. } = item.expect("stream item") {
            ends.push((id, block));
        }
    }
    assert_eq!(ends.len(), 2);
    assert!(matches!(
        &ends[0],
        (id, Some(AssistantContent::ToolCall(call)))
            if *id == BlockId::wire("ok") && call.function.arguments == serde_json::json!({"x": 1})
    ));
    assert_eq!(ends[1], (BlockId::wire("truncated"), None));
    assert_eq!(
        stream.snapshot().len(),
        1,
        "only the completed call is in the choice"
    );
}
