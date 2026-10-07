use super::*;
use crate::completion::{AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_SONNET_4_6, Converse};
use rig_core::completion::CompletionResponse;
use rig_core::message::{
    AssistantContent, DocumentRange, Opaque, Source, SourceLocation, StopReason,
};
use rig_core::test_utils::history::decode;
use rig_core::wire::Mode;

pub(crate) const CLAUDE: &str = ANTHROPIC_CLAUDE_SONNET_4_6;
pub(crate) const NOVA: &str = AMAZON_NOVA_LITE;

pub(crate) fn usage() -> Value {
    json!({ "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 })
}

/// A whole reply holding `content`.
pub(crate) fn document(content: Vec<Value>, stop_reason: &str) -> Value {
    json!({
        "output": { "message": { "role": "assistant", "content": content } },
        "stopReason": stop_reason,
        "usage": usage(),
        "metrics": { "latencyMs": 5 },
    })
}

pub(crate) fn delta(index: usize, delta: Value) -> Value {
    json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } })
}

pub(crate) fn start(index: usize, start: Value) -> Value {
    json!({ "contentBlockStart": { "contentBlockIndex": index, "start": start } })
}

pub(crate) fn stop(index: usize) -> Value {
    json!({ "contentBlockStop": { "contentBlockIndex": index } })
}

pub(crate) fn ended(stop_reason: &str) -> [Value; 2] {
    [
        json!({ "messageStop": { "stopReason": stop_reason } }),
        json!({ "metadata": { "usage": usage(), "metrics": { "latencyMs": 5 } } }),
    ]
}

pub(crate) fn reasoning(text: &str, signature: Option<&str>) -> Value {
    let mut reasoning = json!({ "text": text });
    if let Some(signature) = signature {
        reasoning["signature"] = json!(signature);
    }
    json!({ "reasoningContent": { "reasoningText": reasoning } })
}

pub(crate) fn tool_use(id: &str, name: &str, input: Value) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": name, "input": input } })
}

pub(crate) fn hosted() -> [Value; 2] {
    [
        json!({ "toolUse": {
            "toolUseId": "srv_1", "name": "nova_grounding", "input": { "q": "harbor" }, "type": "server_tool_use",
        } }),
        json!({ "toolResult": { "toolUseId": "srv_1", "content": [{ "text": "nine" }] } }),
    ]
}

/// The response a whole reply holding `content` decodes to on `model`.
pub(crate) fn whole(model: &str, content: Vec<Value>, stop_reason: &str) -> CompletionResponse {
    let frames = [ConverseFrame::Whole(document(content, stop_reason))];
    decode(&Converse::new(model), Mode::Unary, frames).expect("the whole reply decodes")
}

/// The response a stream of `events` decodes to on `model`.
pub(crate) fn streamed(
    model: &str,
    events: Vec<Value>,
) -> Result<CompletionResponse, ProviderError> {
    let frames = events.into_iter().map(ConverseFrame::Event);
    decode(&Converse::new(model), Mode::Streaming, frames)
}

/// A reply holding every block kind a turn keeps an item for.
pub(crate) fn rich() -> Vec<Value> {
    let [used, result] = hosted();
    vec![
        reasoning("let me think", Some("sig-abc")),
        json!({ "reasoningContent": { "redactedContent": "AGNpcGhlcnRleHT/" } }),
        json!({ "citationsContent": {
            "content": [{ "text": "The harbor opens at nine." }],
            "citations": [{ "title": "hours", "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 24 } } }],
        } }),
        used,
        result,
        json!({ "text": "Calling." }),
        tool_use("tooluse_1", "lookup", json!({ "q": "harbor" })),
    ]
}

/// Streamed signed thinking keeps its text and signature, and signature-only
/// thinking keeps its item; empty thinking is no content; unsigned
/// reasoning keeps no item, since it is rebuilt from its text.
#[test]
fn reasoning_keeps_an_item_only_when_signed_or_redacted() {
    let thought = |index, field: &str, text: &str| {
        delta(index, json!({ "reasoningContent": { field: text } }))
    };
    let mut events = vec![
        thought(0, "text", "let me "),
        thought(0, "text", "think"),
        thought(0, "signature", "sig-"),
        thought(0, "signature", "1"),
        stop(0),
        thought(1, "signature", "only"),
        stop(1),
        thought(2, "text", ""),
        stop(2),
        thought(3, "text", "unsigned"),
        stop(3),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(CLAUDE, events).expect("decodes");
    let items: Vec<_> = response
        .choice
        .iter()
        .map(|block| block.native_item().cloned())
        .collect();
    assert_eq!(
        items,
        [
            Some(reasoning("let me think", Some("sig-1"))),
            Some(reasoning("", Some("only"))),
            None,
        ]
    );
    assert_eq!(response.choice[2], AssistantContent::reasoning("unsigned"));
}

/// Redacted reasoning arrives in base64 chunks whose bytes join before they
/// are encoded once.
#[test]
fn redacted_reasoning_encodes_its_bytes_once() {
    let chunk = |bytes: &[u8]| {
        delta(
            0,
            json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } }),
        )
    };
    let mut events = vec![chunk(b"\x00c"), chunk(b"iph"), chunk(b"er\xff"), stop(0)];
    events.extend(ended("end_turn"));
    let response = streamed(CLAUDE, events).expect("decodes");
    assert_eq!(
        response.choice[0].native_item(),
        Some(
            &json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(b"\x00cipher\xff") } })
        )
    );
}

/// #2250, #989: streamed client calls are each kept, one with no input at
/// all, and each keeps its `toolUse` with the input its arguments were
/// read from. A call that never names its tool is dropped.
#[test]
fn streamed_calls_are_all_kept_with_their_items() {
    let mut events = vec![
        delta(0, json!({ "text": "calling" })),
        stop(0),
        start(
            1,
            json!({ "toolUse": { "toolUseId": "t1", "name": "add" } }),
        ),
        delta(1, json!({ "toolUse": { "input": "{\"x\":" } })),
        delta(1, json!({ "toolUse": { "input": "1}" } })),
        stop(1),
        start(
            2,
            json!({ "toolUse": { "toolUseId": "t2", "name": "now" } }),
        ),
        stop(2),
        delta(3, json!({ "toolUse": { "input": "{}" } })),
        stop(3),
    ];
    events.extend(ended("tool_use"));
    let response = streamed(CLAUDE, events).expect("decodes");
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| (call.id.wire().into_owned(), call.function.arguments_value()))
        .collect();
    assert_eq!(
        calls,
        [
            ("t1".to_owned(), json!({ "x": 1 })),
            ("t2".to_owned(), json!({}))
        ]
    );
    assert_eq!(
        response.choice[1].native_item(),
        Some(&tool_use("t1", "add", json!({ "x": 1 })))
    );
    assert_eq!(response.stop(), StopReason::ToolUse);
}

/// An image arrives with its start and its bytes and is written whole at
/// its stop; one Converse sent no bytes for does not replay.
#[test]
fn image_and_tool_result_deltas_assemble_at_their_stop() {
    let mut events = vec![
        start(0, json!({ "image": { "format": "png" } })),
        delta(0, json!({ "image": { "source": { "bytes": "cG5n" } } })),
        stop(0),
        start(
            1,
            json!({ "toolResult": { "toolUseId": "srv_1", "status": "success" } }),
        ),
        delta(1, json!({ "toolResult": [{ "text": "nine" }] })),
        delta(1, json!({ "toolResult": [{ "json": { "opens": 9 } }] })),
        stop(1),
        start(2, json!({ "image": { "format": "heic" } })),
        stop(2),
    ];
    events.extend(ended("end_turn"));
    let response = streamed(NOVA, events).expect("decodes");
    assert!(matches!(&response.choice[0], AssistantContent::Image(image)
        if image.data == DocumentSourceKind::Base64("cG5n".to_owned())));
    assert_eq!(
        response.choice[1],
        AssistantContent::Opaque(Opaque {
            item: json!({ "toolResult": {
                "toolUseId": "srv_1", "status": "success",
                "content": [{ "text": "nine" }, { "json": { "opens": 9 } }],
            } }),
            replay: true,
        })
    );
    assert!(matches!(
        &response.choice[2],
        AssistantContent::Opaque(Opaque { replay: false, .. })
    ));
}

/// A stream that ends at its metadata without `messageStop` names no
/// reason, so it fails, as pi's "Bedrock stream ended without a stop
/// reason" does.
#[test]
fn a_stream_without_a_stop_reason_fails() {
    let events = vec![
        delta(0, json!({ "text": "half an ans" })),
        ended("end_turn")[1].clone(),
    ];
    let response = streamed(NOVA, events).expect("decodes");
    assert!(response.stop().is_failure(), "{:?}", response.stop());
}

/// An exception Bedrock sends mid-stream fails the reply with its type as
/// the code.
#[test]
fn an_in_band_exception_fails_the_stream() {
    let events = vec![
        delta(0, json!({ "text": "par" })),
        json!({ "throttlingException": { "message": "slow down" } }),
    ];
    let error = streamed(NOVA, events).expect_err("the stream fails");
    assert_eq!(error.report().code.as_deref(), Some("ThrottlingException"));
    assert!(error.is_retryable());
}

/// The document the Converse reassembler rebuilds from `events`.
fn rebuilt(events: &[Value]) -> Value {
    let mut document = document::ConverseOutput::default();
    for event in events {
        rig_core::wire::document::Reassemble::absorb(
            &mut document,
            &ConverseFrame::Event(event.clone()),
        );
    }
    rig_core::wire::document::Reassemble::finish(document)
}

/// `event` with the padding ConverseStream adds to every event.
fn padded(mut event: Value) -> Value {
    if let Some(payload) = event
        .as_object_mut()
        .and_then(|event| event.values_mut().next())
        .and_then(Value::as_object_mut)
    {
        payload.insert("p".to_owned(), json!("abcdefgh"));
    }
    event
}

/// A stream rebuilds the `ConverseOutput` a unary call returns, block by
/// block: signed and redacted reasoning, cited text, a hosted tool's use
/// and result, text and a client call. The message-level fields land where
/// the unary reply has them, and the stream's padding is dropped.
#[test]
fn a_stream_rebuilds_the_unary_converse_output() {
    let thought =
        |field: &str, text: &str| delta(0, json!({ "reasoningContent": { field: text } }));
    let redacted = BASE64_STANDARD
        .decode("AGNpcGhlcnRleHT/")
        .expect("the fixture is base64");
    let (head, tail) = redacted.split_at(4);
    let chunk = |bytes: &[u8]| {
        delta(
            1,
            json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(bytes) } }),
        )
    };
    let events: Vec<Value> = [
        json!({ "messageStart": { "role": "assistant" } }),
        thought("text", "let me "),
        thought("text", "think"),
        thought("signature", "sig-abc"),
        stop(0),
        chunk(head),
        chunk(tail),
        stop(1),
        delta(2, json!({ "text": "The harbor opens at nine." })),
        delta(2, json!({ "citation": { "title": "hours", "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 24 } } } })),
        stop(2),
        start(3, json!({ "toolUse": { "toolUseId": "srv_1", "name": "nova_grounding", "type": "server_tool_use" } })),
        delta(3, json!({ "toolUse": { "input": "{\"q\": \"harbor\"}" } })),
        stop(3),
        start(4, json!({ "toolResult": { "toolUseId": "srv_1" } })),
        delta(4, json!({ "toolResult": [{ "text": "nine" }] })),
        stop(4),
        delta(5, json!({ "text": "Calling." })),
        stop(5),
        start(6, json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "lookup" } })),
        delta(6, json!({ "toolUse": { "input": "{\"q\":" } })),
        delta(6, json!({ "toolUse": { "input": " \"harbor\"}" } })),
        stop(6),
        json!({ "messageStop": { "stopReason": "tool_use" } }),
        json!({ "metadata": { "usage": usage(), "metrics": { "latencyMs": 5 } } }),
    ]
    .into_iter()
    .map(padded)
    .collect();
    let unary = document(rich(), "tool_use");
    assert_eq!(rebuilt(&events), unary);

    let response = streamed(CLAUDE, events).expect("decodes");
    assert_eq!(response.raw, unary);
    assert_eq!(response.raw, whole(CLAUDE, rich(), "tool_use").raw);
}

/// The fields only `messageStop` and `metadata` carry land at the top
/// level, as the unary reply states them, and a block that opens with a
/// delta needs no start.
#[test]
fn message_level_fields_land_where_the_unary_reply_has_them() {
    let events = vec![
        json!({ "messageStart": { "role": "assistant", "p": "ab" } }),
        delta(0, json!({ "text": "hi" })),
        stop(0),
        json!({ "messageStop": { "stopReason": "end_turn", "additionalModelResponseFields": { "x": 1 }, "p": "abc" } }),
        json!({ "metadata": {
            "usage": usage(),
            "metrics": { "latencyMs": 5 },
            "trace": { "promptRouter": { "invokedModelId": NOVA } },
            "performanceConfig": { "latency": "optimized" },
            "serviceTier": { "type": "priority" },
            "p": "abcd",
        } }),
    ];
    let response = streamed(NOVA, events).expect("decodes");
    assert_eq!(
        response.raw,
        json!({
            "output": { "message": { "role": "assistant", "content": [{ "text": "hi" }] } },
            "stopReason": "end_turn",
            "additionalModelResponseFields": { "x": 1 },
            "usage": usage(),
            "metrics": { "latencyMs": 5 },
            "trace": { "promptRouter": { "invokedModelId": NOVA } },
            "performanceConfig": { "latency": "optimized" },
            "serviceTier": { "type": "priority" },
        })
    );
}

/// An image's chunks join as bytes, a tool result's content collects, a
/// call with no input has an empty object, and a stream cut by an
/// exception rebuilds what arrived.
#[test]
fn images_results_and_cut_streams_rebuild_what_arrived() {
    let events = vec![
        start(0, json!({ "image": { "format": "png" } })),
        delta(
            0,
            json!({ "image": { "source": { "bytes": BASE64_STANDARD.encode(b"pn") } } }),
        ),
        delta(
            0,
            json!({ "image": { "source": { "bytes": BASE64_STANDARD.encode(b"g") } } }),
        ),
        start(
            1,
            json!({ "toolResult": { "toolUseId": "srv_1", "status": "success" } }),
        ),
        delta(1, json!({ "toolResult": [{ "text": "nine" }] })),
        delta(1, json!({ "toolResult": [{ "json": { "opens": 9 } }] })),
        start(
            2,
            json!({ "toolUse": { "toolUseId": "t2", "name": "now" } }),
        ),
        json!({ "throttlingException": { "message": "slow down" } }),
    ];
    assert_eq!(
        rebuilt(&events),
        json!({ "output": { "message": { "content": [
            { "image": { "format": "png", "source": { "bytes": BASE64_STANDARD.encode(b"png") } } },
            { "toolResult": { "toolUseId": "srv_1", "status": "success",
                "content": [{ "text": "nine" }, { "json": { "opens": 9 } }] } },
            { "toolUse": { "toolUseId": "t2", "name": "now", "input": {} } },
        ] } } })
    );
    assert_eq!(rebuilt(&[]), Value::Null);
}

/// Rig's input counts Bedrock's cache reads and writes, and its total is
/// input plus output.
#[test]
fn usage_counts_cache_reads_and_writes_in_input() {
    let counted = super::usage(&json!({
        "inputTokens": 10, "outputTokens": 5, "totalTokens": 15,
        "cacheReadInputTokens": 100, "cacheWriteInputTokens": 7,
    }));
    assert_eq!(counted.input_tokens, Some(117));
    assert_eq!(counted.output_tokens, Some(5));
    assert_eq!(counted.total_tokens, Some(122));
    assert_eq!(counted.cached_input_tokens, Some(100));
    assert_eq!(counted.cache_creation_input_tokens, Some(7));
}

/// Serves one recorded reply to every request the SDK sends.
#[derive(Clone, Debug)]
struct Recorded {
    body: bytes::Bytes,
    content_type: &'static str,
}

impl aws_smithy_runtime_api::client::http::HttpConnector for Recorded {
    fn call(
        &self,
        _request: aws_smithy_runtime_api::client::orchestrator::HttpRequest,
    ) -> aws_smithy_runtime_api::client::http::HttpConnectorFuture {
        let mut response = aws_smithy_runtime_api::client::orchestrator::HttpResponse::new(
            aws_smithy_runtime_api::http::StatusCode::try_from(200_u16).expect("200 is a status"),
            aws_smithy_types::body::SdkBody::from(self.body.clone()),
        );
        response
            .headers_mut()
            .insert("content-type", self.content_type);
        aws_smithy_runtime_api::client::http::HttpConnectorFuture::ready(Ok(response))
    }
}

impl aws_smithy_runtime_api::client::http::HttpClient for Recorded {
    fn http_connector(
        &self,
        _settings: &aws_smithy_runtime_api::client::http::HttpConnectorSettings,
        _components: &aws_smithy_runtime_api::client::runtime_components::RuntimeComponents,
    ) -> aws_smithy_runtime_api::client::http::SharedHttpConnector {
        aws_smithy_runtime_api::client::http::SharedHttpConnector::new(self.clone())
    }
}

/// A runtime whose every call is answered with the reply of the recorded
/// cassette `relative`, under the Bedrock corpus.
fn recorded(relative: &str, content_type: &'static str) -> crate::client::BedrockRuntime {
    use aws_sdk_bedrockruntime::config::{BehaviorVersion, Credentials, Region};
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes/bedrock")
        .join(relative);
    let cassette = std::fs::read_to_string(&file).expect("the cassette is readable");
    let body = rig_core::test_utils::raw_parity::recorded_reply(&cassette, 1)
        .expect("the cassette records a reply");
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("test", "test", None, None, "parity"))
        .http_client(Recorded { body, content_type })
        .build();
    aws_sdk_bedrockruntime::Client::from_conf(config).into()
}

/// Guarantee 5 on Bedrock's recorded pair, one forced tool call answered by
/// `/converse` and by `/converse-stream`: through the SDK transport, the
/// stream rebuilds the unary `ConverseOutput`. Each answer times itself.
#[tokio::test]
async fn the_recorded_pair_agrees() {
    use rig_core::test_utils::raw_parity::{comparable, raw_pair_over};
    let (unary, streamed) = raw_pair_over(
        Converse::new(NOVA),
        recorded(
            "tool_choice/specific_add_raw_nonstreaming.yaml",
            "application/json",
        ),
        recorded(
            "tool_choice/specific_add_raw_streaming.yaml",
            "application/vnd.amazon.eventstream",
        ),
    )
    .await
    .expect("both replies decode");
    assert_eq!(
        unary.pointer("/output/message/content/0/toolUse/input"),
        Some(&json!({"x": 20, "y": 22}))
    );
    let minted = ["/metrics"];
    assert_eq!(comparable(&unary, &minted), comparable(&streamed, &minted));
}

/// One cited claim per `citationsContent` block, one per location kind
/// Converse documents, plus a kind this crate does not know. Hand-built:
/// rig never turns Converse citations on, so no recording carries them.
fn cited_blocks() -> Vec<(&'static str, Value)> {
    vec![
        (
            "The grass is green.",
            json!({ "title": "Lawn", "sourceContent": [{ "text": "The grass is green." }],
                "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 20 } } }),
        ),
        (
            "Pages two and three.",
            json!({ "sourceContent": [{ "text": "Two. " }, { "text": "Three." }],
                "location": { "documentPage": { "documentIndex": 1, "start": 2, "end": 4 } } }),
        ),
        (
            "Chunk one.",
            json!({ "title": "Chunks",
                "location": { "documentChunk": { "documentIndex": 2, "start": 1, "end": 2 } } }),
        ),
        (
            "Café opens at nine.",
            json!({ "title": "Café", "source": "https://example.com/cafe",
                "sourceContent": [{ "text": "Opens 9am." }],
                "location": { "searchResultLocation": { "searchResultIndex": 3, "start": 0, "end": 1 } } }),
        ),
        (
            "Rust 2.0 shipped.",
            json!({ "title": "Rust",
                "location": { "web": { "url": "https://example.com/rust", "domain": "example.com" } } }),
        ),
        ("Unknown.", json!({ "location": { "frobnicate": {} } })),
    ]
}

/// The sources [`cited_blocks`] resolve to, block by block.
fn cited_sources() -> Vec<Vec<Source>> {
    let document = |index, within| {
        Source::new(SourceLocation::Document {
            index: Some(index),
            id: None,
            within: Some(within),
        })
    };
    vec![
        vec![
            document(0, DocumentRange::Chars(0..20))
                .title("Lawn")
                .cited_text("The grass is green."),
        ],
        vec![document(1, DocumentRange::Pages(2..4)).cited_text("Two. Three.")],
        vec![document(2, DocumentRange::Blocks(1..2)).title("Chunks")],
        vec![
            Source::new(SourceLocation::SearchResult {
                index: 3,
                source: "https://example.com/cafe".to_owned(),
                blocks: Some(0..1),
            })
            .title("Café")
            .cited_text("Opens 9am."),
        ],
        vec![
            Source::new(SourceLocation::Url {
                url: "https://example.com/rust".to_owned(),
            })
            .title("Rust"),
        ],
        vec![],
    ]
}

/// Each text block's text, and its citations' spans and sources.
fn citations_of(
    response: &CompletionResponse,
) -> Vec<(String, Vec<(Option<std::ops::Range<usize>>, Vec<Source>)>)> {
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
                            citation.span.map(|span| span.range()),
                            citation.sources.clone(),
                        )
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

/// Every documented location kind decodes to a whole-block citation with
/// its source, title and quoted passage, the same in a whole reply and a
/// stream, with the citation streamed before the text it cites. The
/// `citationsContent` item keeps Converse's JSON, so replay is unchanged.
#[test]
fn every_citation_kind_resolves_the_same_unary_and_streamed() {
    let blocks = cited_blocks();
    let expected: Vec<_> = blocks
        .iter()
        .zip(cited_sources())
        .map(|((text, _), sources)| {
            let cited = (!sources.is_empty()).then_some((None, sources));
            ((*text).to_owned(), cited.into_iter().collect::<Vec<_>>())
        })
        .collect();
    let content: Vec<Value> = blocks
        .iter()
        .map(|(text, citation)| {
            json!({ "citationsContent": {
                "content": [{ "text": text }],
                "citations": [citation],
            } })
        })
        .collect();
    let unary = whole(CLAUDE, content.clone(), "end_turn");
    let mut events = Vec::new();
    for (index, (text, citation)) in blocks.iter().enumerate() {
        events.push(delta(index, json!({ "citation": citation })));
        events.push(delta(index, json!({ "text": text })));
        events.push(stop(index));
    }
    events.extend(ended("end_turn"));
    let streamed = streamed(CLAUDE, events).expect("decodes");
    for response in [&unary, &streamed] {
        assert_eq!(citations_of(response), expected);
        let items: Vec<_> = response
            .choice
            .iter()
            .map(|block| block.native_item().cloned())
            .collect();
        assert_eq!(items, content.iter().cloned().map(Some).collect::<Vec<_>>());
    }
}
