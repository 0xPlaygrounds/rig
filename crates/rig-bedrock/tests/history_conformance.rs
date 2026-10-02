//! The history conformance suite for the Bedrock Converse wire, on a Claude
//! model and on Nova. Each reply is Converse JSON that the AWS SDK reads
//! over a canned HTTP exchange, so the frames are the SDK's own output and
//! stream events; each request body is the JSON the SDK sends, with the
//! model it addresses in `$path`.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::{Arc, Mutex};

use aws_sdk_bedrockruntime::config::{
    BehaviorVersion, Credentials, Region, StalledStreamProtectionConfig, retry::RetryConfig,
};
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_runtime_api::http::StatusCode;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use futures::StreamExt;
use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::completion::{
    AMAZON_NOVA_LITE, AMAZON_NOVA_MICRO, AMAZON_NOVA_PRO, ANTHROPIC_CLAUDE_HAIKU_4_5,
    ANTHROPIC_CLAUDE_SONNET_4_5, Converse, ConverseFrame, ConverseRequest,
};
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Model, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::test_utils::history_conformance::{Ablation, Ending, HistoryFixture, Shape};
use rig_core::wire::{Mode, Wire};
use serde_json::{Value, json};

/// One canned HTTP exchange: the reply it gives, and the request it got.
#[derive(Clone, Debug)]
struct Canned {
    reply: Arc<(&'static str, Vec<u8>)>,
    sent: Arc<Mutex<Option<(String, Vec<u8>)>>>,
}

impl Canned {
    fn new(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            reply: Arc::new((content_type, body)),
            sent: Arc::default(),
        }
    }

    fn client(&self) -> aws_sdk_bedrockruntime::Client {
        let config = aws_sdk_bedrockruntime::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new("id", "secret", None, None, "test"))
            .retry_config(RetryConfig::disabled())
            .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
            .http_client(self.clone())
            .build();
        aws_sdk_bedrockruntime::Client::from_conf(config)
    }

    fn runtime(&self) -> BedrockRuntime {
        BedrockRuntime::from(self.client())
    }
}

impl HttpConnector for Canned {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let body = request.body().bytes().unwrap_or_default().to_vec();
        *self.sent.lock().expect("sent") = Some((request.uri().to_owned(), body));
        let (content_type, body) = &*self.reply;
        let mut response = HttpResponse::new(
            StatusCode::try_from(200).expect("a status"),
            SdkBody::from(body.clone()),
        );
        response.headers_mut().insert("content-type", *content_type);
        HttpConnectorFuture::ready(Ok(response))
    }
}

impl HttpClient for Canned {
    fn http_connector(
        &self,
        _settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(self.clone())
    }
}

/// The runtime, keeping every frame it hands the decoder.
#[derive(Clone)]
struct Keep {
    runtime: BedrockRuntime,
    frames: Arc<Mutex<Vec<ConverseFrame>>>,
}

impl Transport<Converse> for Keep {
    fn send(&self, payload: ConverseRequest, exchange: Exchange) -> Opening<ConverseFrame> {
        let sent = Transport::<Converse>::send(&self.runtime, payload, exchange);
        let frames = Arc::clone(&self.frames);
        Opening::new(async move {
            let opened = sent.await?;
            Ok(opened.map_frames(move |stream| {
                stream.inspect(move |frame: &Result<ConverseFrame, ProviderError>| {
                    if let Ok(frame) = frame {
                        frames.lock().expect("frames").push(frame.clone());
                    }
                })
            }))
        })
    }
}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(future)
}

/// The frames the SDK reads a unary Converse `document` into.
fn unary_frames(document: Value) -> Vec<ConverseFrame> {
    let body = serde_json::to_vec(&document).expect("JSON");
    frames(Mode::Unary, Canned::new("application/json", body))
}

/// The frames the SDK reads a Converse stream of `events` into, each
/// `{"<event type>": <payload>}`.
fn streamed_frames(events: Vec<Value>) -> Vec<ConverseFrame> {
    let mut body = Vec::new();
    for event in events {
        let (kind, payload) = event
            .as_object()
            .and_then(|event| event.iter().next())
            .expect("an event");
        let message = Message::new(serde_json::to_vec(payload).expect("JSON"))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String(kind.clone().into()),
            ))
            .add_header(Header::new(
                ":content-type",
                HeaderValue::String("application/json".into()),
            ));
        aws_smithy_eventstream::frame::write_message_to(&message, &mut body)
            .expect("an event message");
    }
    frames(
        Mode::Streaming,
        Canned::new("application/vnd.amazon.eventstream", body),
    )
}

fn frames(mode: Mode, canned: Canned) -> Vec<ConverseFrame> {
    let keep = Keep {
        runtime: canned.runtime(),
        frames: Arc::default(),
    };
    let model = Model::new(Converse::new(CLAUDE), keep.clone());
    block_on(async {
        let request = CompletionRequest::new("restate");
        match mode {
            Mode::Unary => drop(model.call(request).await),
            Mode::Streaming => {
                let mut stream = model.stream(request).expect("the stream opens");
                while stream.next().await.is_some() {}
            }
        }
    });
    keep.frames.lock().expect("frames").clone()
}

/// The JSON body `payload` sends in `mode`, with its request path,
/// percent-decoded, as `$path`.
fn sent_body(payload: ConverseRequest, mode: Mode) -> Result<Value, EncodeError> {
    let canned = Canned::new("application/json", b"{}".to_vec());
    let client = canned.client();
    let ConverseRequest {
        model,
        request,
        guardrail,
    } = payload;
    let additional = request.additional_params();
    let inference = request.inference_config();
    let fail = |error: ProviderError| EncodeError::request(error.to_string());
    let tools = request.tools_config().map_err(fail)?;
    let output = request.output_config().map_err(fail)?;
    let system = request.system_prompt().map_err(fail)?;
    let messages = request.messages().map_err(fail)?;
    block_on(async {
        match mode {
            Mode::Unary => drop(
                client
                    .converse()
                    .model_id(model)
                    .set_additional_model_request_fields(additional)
                    .set_inference_config(Some(inference))
                    .set_tool_config(tools)
                    .set_system(system)
                    .set_messages(Some(messages))
                    .set_output_config(output)
                    .set_guardrail_config(guardrail)
                    .send()
                    .await,
            ),
            Mode::Streaming => drop(
                client
                    .converse_stream()
                    .model_id(model)
                    .set_additional_model_request_fields(additional)
                    .set_inference_config(Some(inference))
                    .set_tool_config(tools)
                    .set_system(system)
                    .set_messages(Some(messages))
                    .set_output_config(output)
                    .send()
                    .await,
            ),
        }
    });
    let (uri, body) = canned.sent.lock().expect("sent").take().expect("a request");
    let mut body: Value = serde_json::from_slice(&body)?;
    let path = uri
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/'))
        .map(|(_, path)| format!("/{path}"))
        .unwrap_or(uri);
    if let Value::Object(fields) = &mut body {
        fields.insert("$path".to_owned(), Value::String(percent_decoded(&path)));
    }
    Ok(body)
}

fn percent_decoded(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        let hex = text
            .get(at + 1..at + 3)
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match (bytes[at], hex) {
            (b'%', Some(byte)) => {
                decoded.push(byte);
                at += 3;
            }
            (byte, _) => {
                decoded.push(byte);
                at += 1;
            }
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

const CLAUDE: &str = ANTHROPIC_CLAUDE_SONNET_4_5;
/// `AGNpcGhlcnRleHQA`, the redacted reasoning's bytes as base64.
const REDACTED: &str = "AGNpcGhlcnRleHQA";

fn usage() -> Value {
    json!({ "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 })
}

/// A whole reply holding `content`.
fn document(content: Vec<Value>, stop_reason: &str) -> Value {
    json!({
        "output": { "message": { "role": "assistant", "content": content } },
        "stopReason": stop_reason,
        "usage": usage(),
        "metrics": { "latencyMs": 5 },
    })
}

fn delta(index: usize, delta: Value) -> Value {
    json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } })
}

fn start(index: usize, start: Value) -> Value {
    json!({ "contentBlockStart": { "contentBlockIndex": index, "start": start } })
}

fn stop(index: usize) -> Value {
    json!({ "contentBlockStop": { "contentBlockIndex": index } })
}

/// The events Converse streams for the whole-reply `content`.
fn events(content: &[Value], stop_reason: &str) -> Vec<Value> {
    let mut events = vec![json!({ "messageStart": { "role": "assistant" } })];
    for (index, block) in content.iter().enumerate() {
        let (kind, body) = block
            .as_object()
            .and_then(|block| block.iter().next())
            .expect("a block");
        match kind.as_str() {
            "text" => events.push(delta(index, json!({ "text": body }))),
            "reasoningContent" => {
                if let Some(redacted) = body.get("redactedContent") {
                    events.push(delta(
                        index,
                        json!({ "reasoningContent": { "redactedContent": redacted } }),
                    ));
                } else {
                    let reasoning = &body["reasoningText"];
                    events.push(delta(
                        index,
                        json!({ "reasoningContent": { "text": reasoning["text"] } }),
                    ));
                    if let Some(signature) = reasoning.get("signature") {
                        events.push(delta(
                            index,
                            json!({ "reasoningContent": { "signature": signature } }),
                        ));
                    }
                }
            }
            "citationsContent" => {
                for part in body["content"].as_array().into_iter().flatten() {
                    events.push(delta(index, json!({ "text": part["text"] })));
                }
                for citation in body["citations"].as_array().into_iter().flatten() {
                    events.push(delta(index, json!({ "citation": citation })));
                }
            }
            "toolUse" => {
                let mut opened = body.clone();
                if let Value::Object(fields) = &mut opened {
                    fields.shift_remove("input");
                }
                events.push(start(index, json!({ "toolUse": opened })));
                events.push(delta(
                    index,
                    json!({ "toolUse": { "input": body["input"].to_string() } }),
                ));
            }
            "toolResult" => {
                let mut opened = body.clone();
                if let Value::Object(fields) = &mut opened {
                    fields.shift_remove("content");
                }
                events.push(start(index, json!({ "toolResult": opened })));
                events.push(delta(index, json!({ "toolResult": body["content"] })));
            }
            other => panic!("no stream carries a `{other}` block"),
        }
        events.push(stop(index));
    }
    events.push(json!({ "messageStop": { "stopReason": stop_reason } }));
    events.push(json!({ "metadata": { "usage": usage(), "metrics": { "latencyMs": 5 } } }));
    events
}

fn reasoning(text: &str, signature: Option<&str>) -> Value {
    match signature {
        Some(signature) => {
            json!({ "reasoningContent": { "reasoningText": { "text": text, "signature": signature } } })
        }
        None => json!({ "reasoningContent": { "reasoningText": { "text": text } } }),
    }
}

fn call(id: &str, input: Value) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": "lookup", "input": input } })
}

/// The suite's fixture for one model family.
struct BedrockHistory {
    model: &'static str,
    other_model: &'static str,
    /// The signature the model's reasoning carries, when its family signs.
    signature: Option<&'static str>,
}

impl BedrockHistory {
    fn signature(&self, suffix: &str) -> Option<String> {
        self.signature
            .map(|signature| format!("{signature}-{suffix}"))
    }

    fn content(&self, shape: Shape) -> Option<(Vec<Value>, &'static str)> {
        Some(match shape {
            // Signed and redacted reasoning, cited text, a hosted tool's use
            // and result, text, and a call: every block kind Converse
            // replies with.
            Shape::Rich => (
                vec![
                    reasoning("plan the lookup", self.signature("rich").as_deref()),
                    json!({ "reasoningContent": { "redactedContent": REDACTED } }),
                    json!({ "citationsContent": {
                        "content": [{ "text": "The harbor opens at nine." }],
                        "citations": [{
                            "title": "hours",
                            "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 24 } },
                        }],
                    } }),
                    json!({ "toolUse": {
                        "toolUseId": "srv_1",
                        "name": "nova_grounding",
                        "input": { "query": "harbor hours" },
                        "type": "server_tool_use",
                    } }),
                    json!({ "toolResult": {
                        "toolUseId": "srv_1",
                        "content": [{ "text": "nine" }, { "json": { "opens": 9 } }],
                        "status": "success",
                    } }),
                    json!({ "text": "Looking it up." }),
                    call("tooluse_1", json!({ "q": "rig" })),
                ],
                "tool_use",
            ),
            Shape::Interleaved => (
                vec![
                    reasoning("first", self.signature("first").as_deref()),
                    json!({ "text": "between" }),
                    reasoning("second", self.signature("second").as_deref()),
                    call("tooluse_2", json!({ "q": "rig" })),
                ],
                "tool_use",
            ),
            // The SDK reads an item and a field it does not model as its
            // payload-less `Unknown`, or not at all, and cannot send one
            // back: the invented shape cannot reach the decoder or the
            // request.
            Shape::Unknown => return None,
        })
    }
}

impl HistoryFixture for BedrockHistory {
    type Wire = Converse;

    fn wire(&self, model: &str) -> Converse {
        Converse::new(model)
    }

    fn model(&self) -> &'static str {
        self.model
    }

    fn other_model(&self) -> &'static str {
        self.other_model
    }

    fn text_only_model(&self) -> Option<&'static str> {
        Some(AMAZON_NOVA_MICRO)
    }

    fn body(
        &self,
        wire: &Converse,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        sent_body(wire.encode(request, mode)?, mode)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<ConverseFrame>> {
        let (content, stop_reason) = self.content(shape)?;
        Some(match mode {
            Mode::Unary => unary_frames(document(content, stop_reason)),
            Mode::Streaming => streamed_frames(events(&content, stop_reason)),
        })
    }

    /// A whole reply's input is a JSON document, so text that is not JSON
    /// only arrives streamed.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<ConverseFrame>> {
        match mode {
            Mode::Unary => {
                let input: Value = serde_json::from_str(arguments).ok()?;
                Some(unary_frames(document(
                    vec![call("tooluse_m", input)],
                    "tool_use",
                )))
            }
            Mode::Streaming => Some(streamed_frames(vec![
                start(
                    0,
                    json!({ "toolUse": { "toolUseId": "tooluse_m", "name": "lookup" } }),
                ),
                delta(0, json!({ "toolUse": { "input": arguments } })),
                stop(0),
                json!({ "messageStop": { "stopReason": "tool_use" } }),
                json!({ "metadata": { "usage": usage() } }),
            ])),
        }
    }

    /// Every `stopReason` Converse documents, and one it does not.
    fn finishes(&self) -> Vec<(&'static str, Vec<ConverseFrame>, Ending)> {
        [
            ("end_turn", Ending::Success),
            ("tool_use", Ending::Success),
            ("max_tokens", Ending::Success),
            ("stop_sequence", Ending::Success),
            ("model_context_window_exceeded", Ending::Success),
            ("guardrail_intervened", Ending::Failure),
            ("content_filtered", Ending::Failure),
            ("malformed_model_output", Ending::Failure),
            ("malformed_tool_use", Ending::Failure),
            ("x_rig_invented", Ending::Failure),
        ]
        .into_iter()
        .map(|(reason, ending)| {
            let frames = unary_frames(document(vec![json!({ "text": "x" })], reason));
            (reason, frames, ending)
        })
        .collect()
    }

    /// Only a union's member is required: the SDK refuses a union with
    /// none, and fills any other missing field with its default. A call
    /// that names no tool is dropped.
    fn ablation(&self) -> Option<Ablation<ConverseFrame>> {
        let (content, stop_reason) = self.content(Shape::Rich)?;
        Some(Ablation {
            document: document(content, stop_reason),
            required: &[
                "/output/message",
                "/output/message/content/*/*",
                "/output/message/content/*/reasoningContent/*",
                "/output/message/content/*/citationsContent/content/*/*",
                "/output/message/content/*/citationsContent/citations/*/location/*",
                "/output/message/content/*/toolResult/content/*/*",
            ],
            frames: unary_frames,
        })
    }
}

mod claude {
    rig_core::history_conformance_suite! {
        wire: "bedrock_claude",
        fixture: super::BedrockHistory {
            model: super::CLAUDE,
            other_model: super::ANTHROPIC_CLAUDE_HAIKU_4_5,
            signature: Some("sig"),
        },
    }
}

mod nova {
    rig_core::history_conformance_suite! {
        wire: "bedrock_nova",
        fixture: super::BedrockHistory {
            model: super::AMAZON_NOVA_PRO,
            other_model: super::AMAZON_NOVA_LITE,
            signature: None,
        },
    }
}

/// The SDK's reading of a whole reply holding `block`, and the response it
/// decodes to.
fn decoded_block(
    block: Value,
) -> (
    aws_sdk_bedrockruntime::types::ContentBlock,
    rig_core::completion::CompletionResponse,
) {
    let frames = unary_frames(document(vec![block.clone()], "end_turn"));
    let sdk = frames
        .iter()
        .find_map(|frame| match frame {
            ConverseFrame::Whole(output) => output
                .output
                .as_ref()
                .and_then(|output| output.as_message().ok())
                .and_then(|message| message.content.first().cloned()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the SDK reads {block}"));
    let response =
        rig_core::test_utils::history::decode(&Converse::new(CLAUDE), Mode::Unary, frames)
            .unwrap_or_else(|error| panic!("{block} decodes: {error}"));
    (sdk, response)
}

/// Every Converse content block becomes a block, an invented one too: the
/// SDK reads an invented block as its payload-less `Unknown`, which is a
/// marker, and drops an invented field. A block Rig cannot send back is a
/// marker naming its kind; the blocks a turn keeps hold their JSON.
#[test]
#[allow(clippy::wildcard_enum_match_arm)]
fn every_converse_block_becomes_a_block() {
    use aws_sdk_bedrockruntime::types::ContentBlock as Content;
    use rig_core::message::{AssistantContent, Opaque};
    let cited = json!({ "citationsContent": {
        "content": [{ "text": "cited" }],
        "citations": [{ "title": "note" }],
    } });
    let hosted =
        json!({ "toolResult": { "toolUseId": "srv_1", "content": [{ "text": "found" }] } });
    let samples = [
        json!({ "audio": { "format": "mp3", "source": { "bytes": "bXAz" } } }),
        json!({ "cachePoint": { "type": "default" } }),
        cited.clone(),
        json!({ "document": { "format": "txt", "name": "doc", "source": { "text": "body" } } }),
        json!({ "guardContent": { "text": { "text": "guarded" } } }),
        json!({ "image": { "format": "png", "source": { "bytes": "cG5n" } } }),
        reasoning("thought", Some("sig")),
        json!({ "searchResult": { "source": "web", "title": "t", "content": [{ "text": "found" }] } }),
        json!({ "text": "hello" }),
        hosted.clone(),
        json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "lookup", "input": {}, "x_rig_field": 1 } }),
        json!({ "video": { "format": "mp4", "source": { "bytes": "bXA0" } } }),
        json!({ "x_rig_invented": { "a": 1 } }),
    ];
    let index = |block: &Content| match block {
        Content::Audio(_) => 0,
        Content::CachePoint(_) => 1,
        Content::CitationsContent(_) => 2,
        Content::Document(_) => 3,
        Content::GuardContent(_) => 4,
        Content::Image(_) => 5,
        Content::ReasoningContent(_) => 6,
        Content::SearchResult(_) => 7,
        Content::Text(_) => 8,
        Content::ToolResult(_) => 9,
        Content::ToolUse(_) => 10,
        Content::Video(_) => 11,
        _ => 12,
    };
    let decoded: Vec<_> = samples.into_iter().map(decoded_block).collect();
    let blocks: Vec<Content> = decoded.iter().map(|(block, _)| block.clone()).collect();
    rig_core::test_utils::history::assert_every_variant(&blocks, index, 13);
    let marker = |kind: &str| {
        AssistantContent::Opaque(Opaque {
            item: json!({ "type": kind }),
            replay: false,
        })
    };
    for (block, response) in &decoded {
        let choice = &response.choice;
        match index(block) {
            0 => assert_eq!(choice, &[marker("audio")]),
            1 => assert_eq!(choice, &[marker("cache_point")]),
            2 => assert_eq!(choice[0].native_item(), Some(&cited)),
            3 => assert_eq!(choice, &[marker("document")]),
            4 => assert_eq!(choice, &[marker("guard_content")]),
            5 => assert!(
                matches!(&choice[..], [AssistantContent::Image(_)]),
                "{choice:?}"
            ),
            6 => assert_eq!(
                choice[0].native_item(),
                Some(&reasoning("thought", Some("sig")))
            ),
            7 => assert_eq!(choice, &[marker("search_result")]),
            8 => assert_eq!(choice, &[AssistantContent::text("hello")]),
            9 => assert_eq!(
                choice,
                &[AssistantContent::Opaque(Opaque {
                    item: hosted.clone(),
                    replay: true
                })]
            ),
            10 => assert_eq!(response.tool_calls().count(), 1),
            11 => assert_eq!(choice, &[marker("video")]),
            _ => assert_eq!(choice, &[marker("unknown")]),
        }
    }
}

/// `raw` is the JSON Bedrock sent: a whole reply's body, guardrail trace,
/// performance configuration, service tier and model-specific fields
/// included, and a stream's message-level events.
#[test]
fn raw_is_the_json_bedrock_sent() {
    let mut whole = document(vec![json!({ "text": "blocked" })], "guardrail_intervened");
    if let Value::Object(fields) = &mut whole {
        fields.insert("trace".to_owned(), json!({ "guardrail": { "actionReason": "Guardrail blocked.", "inputAssessment": { "g1": { "contentPolicy": { "filters": [{ "type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED" }] } } } } }));
        fields.insert(
            "performanceConfig".to_owned(),
            json!({ "latency": "optimized" }),
        );
        fields.insert("serviceTier".to_owned(), json!({ "type": "priority" }));
        fields.insert(
            "additionalModelResponseFields".to_owned(),
            json!({ "stop_sequence": null }),
        );
    }
    let stream = vec![
        json!({ "messageStart": { "role": "assistant" } }),
        delta(0, json!({ "text": "hi" })),
        stop(0),
        json!({ "messageStop": { "stopReason": "end_turn", "additionalModelResponseFields": { "x": 1 } } }),
        json!({ "metadata": {
            "usage": usage(),
            "metrics": { "latencyMs": 7 },
            "trace": { "promptRouter": { "invokedModelId": "amazon.nova-lite-v1:0" } },
            "serviceTier": { "type": "flex" },
        } }),
    ];
    let mut body = Vec::new();
    for event in &stream {
        let (kind, payload) = event
            .as_object()
            .and_then(|event| event.iter().next())
            .expect("an event");
        let message = Message::new(serde_json::to_vec(payload).expect("JSON"))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String(kind.clone().into()),
            ));
        aws_smithy_eventstream::frame::write_message_to(&message, &mut body).expect("a message");
    }
    let unary = Canned::new(
        "application/json",
        serde_json::to_vec(&whole).expect("JSON"),
    );
    let streamed = Canned::new("application/vnd.amazon.eventstream", body);
    let (unary, streamed) = block_on(async {
        let unary = Model::new(Converse::new(CLAUDE), unary.runtime())
            .call(CompletionRequest::new("hi"))
            .await
            .expect("the reply decodes");
        let mut stream = Model::new(Converse::new(CLAUDE), streamed.runtime())
            .stream(CompletionRequest::new("hi"))
            .expect("the stream opens");
        while stream.next().await.is_some() {}
        (unary, stream.finish().await.expect("the stream ends"))
    });
    assert_eq!(unary.raw, whole);
    assert_eq!(
        streamed.raw,
        json!({
            "messageStart": stream[0]["messageStart"],
            "messageStop": stream[3]["messageStop"],
            "metadata": stream[4]["metadata"],
        })
    );
}
