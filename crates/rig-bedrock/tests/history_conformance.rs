//! The history conformance suite for the Bedrock Converse wire, on a Claude
//! model and on Nova. Each reply is the Converse JSON the transport hands
//! the decoder; each request body is the JSON the transport sends in place
//! of the SDK's serialization, with the model it addresses in `$path`. The
//! tests after the suite drive both through the AWS SDK over a canned HTTP
//! exchange.

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
use rig_core::driver::Model;
use rig_core::error::EncodeError;
use rig_core::message::AssistantContent;
use rig_core::wire::{Mode, Operation, Wire};
use rig_history_conformance::{Ablation, CallShape, Ending, HistoryFixture, Rng, Shape, replies};
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

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(future)
}

/// The frame carrying the whole unary reply `document`.
fn unary_frames(document: Value) -> Vec<ConverseFrame> {
    vec![ConverseFrame::Whole(document)]
}

/// The frames of a Converse stream of `events`, each `{"<event type>":
/// <payload>}`.
fn streamed_frames(events: Vec<Value>) -> Vec<ConverseFrame> {
    events.into_iter().map(ConverseFrame::Event).collect()
}

/// The event-stream body carrying `events`.
fn event_stream(events: &[Value]) -> Vec<u8> {
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
            ));
        aws_smithy_eventstream::frame::write_message_to(&message, &mut body)
            .expect("an event message");
    }
    body
}

/// The JSON body `payload` sends in `mode`, with its request path as
/// `$path`. The transport sends this body as it stands
/// (`the_transport_sends_the_encoded_body`).
fn sent_body(payload: ConverseRequest, mode: Mode) -> Value {
    let operation = match mode {
        Mode::Unary => "converse",
        Mode::Streaming => "converse-stream",
    };
    let mut body = payload.body;
    body["$path"] = json!(format!("/model/{}/{operation}", payload.model));
    body
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
            // A block Converse adds later starts whole.
            _ => events.push(start(index, block.clone())),
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
            Shape::Unknown => (
                vec![
                    json!({ "x_rig_invented": { "a": 1 } }),
                    json!({ "toolUse": {
                        "toolUseId": "tooluse_3",
                        "name": "lookup",
                        "input": { "q": "rig" },
                        "x_rig_field": 1,
                    } }),
                    json!({ "text": "done" }),
                ],
                "tool_use",
            ),
        })
    }
}

impl HistoryFixture for BedrockHistory {
    type Wire = Converse;

    fn wire(&self, model: &str) -> Converse {
        Converse::new(model)
    }

    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        Some(replies::converse_spec(rng, self.signature))
    }

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<ConverseFrame>> {
        let (whole, events) = replies::converse_build(spec);
        Some(replies::Frames {
            whole: unary_frames(whole),
            streamed: streamed_frames(events),
        })
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
        Ok(sent_body(wire.encode(request, mode)?, mode))
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

    /// Nothing is required: every field is read leniently.
    fn ablation(&self) -> Option<Ablation<ConverseFrame>> {
        let (content, stop_reason) = self.content(Shape::Rich)?;
        Some(Ablation {
            document: document(content, stop_reason),
            required: &[],
            frames: unary_frames,
        })
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/stopReason")
    }

    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let item = block.native_item()?.clone();
        let frames = unary_frames(document(vec![item], "end_turn"));
        let wire = Converse::new(self.model);
        let response = rig_core::test_utils::history::decode(&wire, Mode::Unary, frames).ok()?;
        response.choice.into_iter().next()
    }

    /// Converse indexes every block: two calls under one index follow each
    /// other, and a whole reply lists them.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<ConverseFrame>> {
        let calls = [("a1", "Paris"), ("b2", "Rome")];
        match (shape, mode) {
            (CallShape::ReusedIndex, Mode::Streaming) => {
                let mut events = Vec::new();
                for (id, city) in calls {
                    events.push(start(
                        0,
                        json!({ "toolUse": { "toolUseId": id, "name": "weather" } }),
                    ));
                    let input = json!({ "city": city }).to_string();
                    events.push(delta(0, json!({ "toolUse": { "input": input } })));
                    events.push(stop(0));
                }
                events.push(json!({ "messageStop": { "stopReason": "tool_use" } }));
                events.push(json!({ "metadata": { "usage": usage() } }));
                Some(streamed_frames(events))
            }
            (CallShape::WholeList, Mode::Unary) => {
                let content = calls
                    .iter()
                    .map(|(id, city)| json!({ "toolUse": { "toolUseId": id, "name": "weather", "input": { "city": city } } }))
                    .collect();
                Some(unary_frames(document(content, "tool_use")))
            }
            _ => None,
        }
    }

    fn empty_reply(&self, mode: Mode) -> Option<Vec<ConverseFrame>> {
        Some(match mode {
            Mode::Unary => unary_frames(document(Vec::new(), "end_turn")),
            Mode::Streaming => streamed_frames(events(&[], "end_turn")),
        })
    }

    fn error_frame(&self) -> Option<ConverseFrame> {
        Some(ConverseFrame::Event(
            json!({ "throttlingException": { "message": "slow down" } }),
        ))
    }

    fn strict_roles(&self) -> bool {
        true
    }
}

mod claude {
    rig_history_conformance::history_conformance_suite! {
        wire: "bedrock_claude",
        fixture: super::BedrockHistory {
            model: super::CLAUDE,
            other_model: super::ANTHROPIC_CLAUDE_HAIKU_4_5,
            signature: Some("sig"),
        },
    }
}

mod nova {
    rig_history_conformance::history_conformance_suite! {
        wire: "bedrock_nova",
        fixture: super::BedrockHistory {
            model: super::AMAZON_NOVA_PRO,
            other_model: super::AMAZON_NOVA_LITE,
            signature: None,
        },
    }
}

/// Every Converse content block becomes a block. A block Rig has no
/// canonical form for, an invented one too, replays to the same model as
/// Bedrock sent it; the blocks a turn keeps hold their JSON.
#[test]
fn every_converse_block_becomes_a_block() {
    use rig_core::message::{AssistantContent, Opaque};
    let cited = json!({ "citationsContent": {
        "content": [{ "text": "cited" }],
        "citations": [{ "title": "note" }],
    } });
    let hosted =
        json!({ "toolResult": { "toolUseId": "srv_1", "content": [{ "text": "found" }] } });
    let opaque = [
        json!({ "audio": { "format": "mp3", "source": { "bytes": "bXAz" } } }),
        json!({ "cachePoint": { "type": "default" } }),
        json!({ "document": { "format": "txt", "name": "doc", "source": { "text": "body" } } }),
        json!({ "guardContent": { "text": { "text": "guarded" } } }),
        json!({ "searchResult": { "source": "web", "title": "t", "content": [{ "text": "found" }] } }),
        json!({ "video": { "format": "mp4", "source": { "bytes": "bXA0" } } }),
        json!({ "x_rig_invented": { "a": 1 } }),
        hosted.clone(),
    ];
    let decoded = |block: Value| {
        rig_core::test_utils::history::decode(
            &Converse::new(CLAUDE),
            Mode::Unary,
            unary_frames(document(vec![block.clone()], "end_turn")),
        )
        .unwrap_or_else(|error| panic!("{block} decodes: {error}"))
    };
    for block in opaque {
        let response = decoded(block.clone());
        assert_eq!(
            response.choice,
            [AssistantContent::Opaque(Opaque {
                item: block,
                replay: true
            })]
        );
    }
    assert_eq!(decoded(cited.clone()).choice[0].native_item(), Some(&cited));
    let signed = reasoning("thought", Some("sig"));
    assert_eq!(
        decoded(signed.clone()).choice[0].native_item(),
        Some(&signed)
    );
    assert_eq!(
        decoded(json!({ "text": "hello" })).choice,
        [AssistantContent::text("hello")]
    );
    let image = decoded(json!({ "image": { "format": "png", "source": { "bytes": "cG5n" } } }));
    assert!(
        matches!(&image.choice[..], [AssistantContent::Image(_)]),
        "{:?}",
        image.choice
    );
    let call = json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "lookup", "input": {}, "x_rig_field": 1 } });
    let response = decoded(call.clone());
    assert_eq!(response.tool_calls().count(), 1);
    assert_eq!(response.choice[0].native_item(), Some(&call));
}

/// The transport sends the encoded body as it stands, in place of the
/// SDK's serialization, to the model's Converse path.
#[test]
fn the_transport_sends_the_encoded_body() {
    for mode in [Mode::Unary, Mode::Streaming] {
        let canned = match mode {
            Mode::Unary => Canned::new(
                "application/json",
                serde_json::to_vec(&document(vec![json!({ "text": "ok" })], "end_turn"))
                    .expect("JSON"),
            ),
            Mode::Streaming => Canned::new(
                "application/vnd.amazon.eventstream",
                event_stream(&events(&[json!({ "text": "ok" })], "end_turn")),
            ),
        };
        let mut request = CompletionRequest::new("hi");
        request.temperature = Some(0.5);
        let wire = Converse::new(CLAUDE).with_prompt_caching();
        let prepared = rig_core::operation::Completion::prepare(request.clone(), &wire.describe())
            .expect("prepares");
        let expected = sent_body(wire.encode(prepared, mode).expect("encodes"), mode);
        let model = Model::new(wire, canned.runtime());
        let text = block_on(async {
            match mode {
                Mode::Unary => model.call(request).await.expect("a reply").text(),
                Mode::Streaming => {
                    let mut stream = model.stream(request).expect("the stream opens");
                    while stream.next().await.is_some() {}
                    stream.finish().await.expect("the stream ends").text()
                }
            }
        });
        assert_eq!(text, "ok");
        let (uri, body) = canned.sent.lock().expect("sent").take().expect("a request");
        let mut body: Value = serde_json::from_slice(&body).expect("a JSON body");
        let path = uri
            .split_once("amazonaws.com")
            .map_or(uri.as_str(), |(_, path)| path);
        body["$path"] = json!(percent_decoded(path));
        assert_eq!(body, expected, "{mode:?}");
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
    let unary = Canned::new(
        "application/json",
        serde_json::to_vec(&whole).expect("JSON"),
    );
    let streamed = Canned::new("application/vnd.amazon.eventstream", event_stream(&stream));
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
    assert!(unary.stop().is_failure());
    assert_eq!(
        streamed.raw,
        json!({
            "messageStart": stream[0]["messageStart"],
            "messageStop": stream[3]["messageStop"],
            "metadata": stream[4]["metadata"],
        })
    );
}

/// A reply the SDK cannot read, here for a field of an unexpected type,
/// decodes from the JSON Bedrock sent like any other.
#[test]
fn a_reply_the_sdk_cannot_read_decodes_from_its_json() {
    let content = vec![
        reasoning("plan", Some("sig")),
        json!({ "text": "Looking it up." }),
        call("tooluse_1", json!({ "q": "rig" })),
    ];
    let read = document(content, "tool_use");
    let mut unread = read.clone();
    unread["metrics"]["latencyMs"] = json!("5");
    let decode = |document: &Value| {
        let canned = Canned::new(
            "application/json",
            serde_json::to_vec(document).expect("JSON"),
        );
        block_on(
            Model::new(Converse::new(CLAUDE), canned.runtime()).call(CompletionRequest::new("q")),
        )
        .expect("the reply decodes")
    };
    let (expected, response) = (decode(&read), decode(&unread));
    assert_eq!(response.choice, expected.choice);
    assert_eq!(response.usage, expected.usage);
    assert_eq!(response.raw, unread);
    assert!(response.tool_calls().count() == 1);
}

/// Round 5 generated-history finding 1: a window cut between a call and
/// its result leaves a first user message of only an orphan result. The
/// adapter drops it, and the request still opens with a user message, as
/// Converse requires ("A conversation must start with a user message").
#[test]
fn an_orphan_first_result_still_leaves_a_user_message_first() {
    use rig_core::message::{
        AssistantMessage, CallId, Message, ToolName, ToolResult, ToolResultContent, UserContent,
    };
    let fixture = BedrockHistory {
        model: CLAUDE,
        other_model: CLAUDE,
        signature: Some("sig"),
    };
    let history = vec![
        Message::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: CallId::from_wire("gone"),
                name: ToolName::new("lookup").expect("a tool name"),
                content: vec![ToolResultContent::text("stale")],
                is_error: false,
            })],
        },
        Message::Assistant(AssistantMessage::new(vec![AssistantContent::text("hello")])),
    ];
    let body = rig_history_conformance::sent(&fixture, CLAUDE, history, Mode::Unary)
        .expect("the history encodes");
    assert_eq!(body["messages"][0]["role"], "user", "{}", body["messages"]);
}
