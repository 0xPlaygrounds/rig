//! The model layer's canonical programs, written against the public API
//! only, as a crate outside `rig-core` would write them.
//!
//! A streamed chat completion with tools over HTTP; an OpenAI-compatible
//! provider declared out of tree; a pose-estimation operation this crate has
//! never heard of, served remotely over HTTP and locally in process with
//! one event per video frame; and a transport swapped in under wires that
//! do not change. The erased, observed and replayed model an agent holds is
//! `corpus_host`'s config-chosen cell in `rig-cassette`.

#![allow(clippy::expect_used)]

use bytes::Bytes;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::driver::{Exchange, Local, Model, Opened, Sending, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::http_client::HttpClientExt;
use rig_core::message::AssistantContent;
use rig_core::providers::anthropic::AnthropicConfig;
use rig_core::providers::openai::wire::{Dialect, OpenAIConfig};
use rig_core::streaming::{StreamEvent, Update};
use rig_core::test_utils::{MockHttpResponse, MockStreamingClient, SequencedHttpClient};
use rig_core::wire::{
    Body, Call, Decoder, Descriptor, Encoded, Fold, Framing, Mode, Operation, Out, Reply, Wire,
    WireEvent, WireFrame,
};
use serde::{Deserialize, Serialize};

// ── 1. a streamed chat completion with tools, over HTTP ────────────────

const CLAUDE_TURN: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"Adding."},{"type":"tool_use","id":"toolu_1","name":"add","input":{"a":1,"b":2}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":12,"output_tokens":7}}"#;

const CLAUDE_STREAM: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Adding.\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"add\",\"input\":{}}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1,\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"b\\\":2}\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":7}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

fn add_tool() -> ToolDefinition {
    ToolDefinition {
        name: "add".to_owned(),
        description: "Add two numbers".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        }),
    }
}

fn add_one_and_two() -> CompletionRequest {
    let mut request = CompletionRequest::new("Add 1 and 2.");
    request.tools.push(add_tool());
    request
}

fn tool_calls(choice: &[AssistantContent]) -> Vec<(String, serde_json::Value)> {
    choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::ToolCall(call) => {
                Some((call.function.name.clone(), call.function.arguments.clone()))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn program_1_a_chat_model_calls_unary_then_streams_with_tools() {
    let anthropic = AnthropicConfig::new("key");
    let unary = anthropic
        .clone()
        .connect(SequencedHttpClient::new([MockHttpResponse::success(
            CLAUDE_TURN,
        )]))
        .completion("claude-sonnet-4-6");
    let turn = unary
        .call(add_one_and_two())
        .await
        .expect("the turn decodes");
    assert_eq!(turn.text(), "Adding.");
    assert_eq!(
        tool_calls(&turn.choice),
        [("add".to_owned(), serde_json::json!({"a": 1, "b": 2}))]
    );

    let streaming = anthropic
        .connect(MockStreamingClient {
            sse_bytes: Bytes::from_static(CLAUDE_STREAM.as_bytes()),
        })
        .completion("claude-sonnet-4-6");
    let mut stream = streaming
        .stream(add_one_and_two())
        .expect("the stream opens");
    let mut text = String::new();
    let mut updates = stream.updates();
    while let Some(update) = updates.next().await {
        if let Update::Delta { text: delta, .. } = update.expect("an update") {
            text.push_str(&delta);
        }
    }
    drop(updates);
    let streamed = stream.finish().expect("the stream reached its terminal");
    assert!(text.contains("Adding."), "{text}");
    assert_eq!(tool_calls(&streamed.choice), tool_calls(&turn.choice));
    assert_eq!(streamed.usage.output_tokens, turn.usage.output_tokens);
}

// ── 2. an OpenAI-compatible provider, declared out of tree ─────────────

/// A provider this crate has never heard of: data, not code.
const ACME: Dialect = Dialect::gateway("acme", "https://api.acme.test/v1", "ACME_API_KEY");

const ACME_TURN: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"acme-large","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#;

#[tokio::test]
async fn program_2_an_out_of_tree_compatible_provider_reuses_the_shared_wire() {
    let http = SequencedHttpClient::new([MockHttpResponse::success(ACME_TURN)]);
    let model = OpenAIConfig::with_key(&ACME, "key")
        .connect(http.clone())
        .completion("acme-large");
    let turn = model.call("hi").await.expect("the turn decodes");
    assert_eq!(turn.text(), "hello");
    assert_eq!(turn.provider, "acme");
    assert_eq!(
        http.requests()
            .into_iter()
            .map(|request| request.uri)
            .collect::<Vec<_>>(),
        ["https://api.acme.test/v1/chat/completions"]
    );
}

// ── 3 and 4. pose estimation: an operation this crate never heard of ───

/// Video frames in; one pose per frame out.
struct PoseEstimation;

/// Encoded images, in order.
#[derive(Clone, Serialize)]
struct Frames(Vec<Vec<u8>>);

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Pose {
    frame: usize,
    keypoints: Vec<(f32, f32)>,
    last: bool,
}

/// Every pose of a video, in frame order.
#[derive(Debug, Default, PartialEq)]
struct Track(Vec<Pose>);

impl Operation for PoseEstimation {
    type Request = Frames;
    type Event = Pose;
    type Response = Track;
    type Fold = Track;

    fn is_terminal(pose: &Pose) -> bool {
        pose.last
    }

    fn fold(_: &Frames, _: &mut Call<'_>) -> Track {
        Track::default()
    }
}

impl Fold<PoseEstimation> for Track {
    fn absorb(&mut self, pose: &Pose) -> Result<(), ProviderError> {
        self.0.push(pose.clone());
        Ok(())
    }

    fn finish(self, _: Reply) -> Result<Track, ProviderError> {
        Ok(self)
    }
}

// 3. remote, over HTTP

/// A hosted pose model: plain data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PoseCloud {
    base_url: String,
}

impl Wire for PoseCloud {
    type Op = PoseEstimation;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder = PoseDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("pose-cloud")
    }

    fn encode(&self, frames: Frames, _: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::post(format!("{}/v1/pose", self.base_url))
            .body(Body::Bytes(serde_json::to_vec(&frames)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _: Mode) -> PoseDecoder {
        PoseDecoder
    }
}

struct PoseDecoder;

#[derive(Deserialize)]
struct PoseReply {
    poses: Vec<Pose>,
}

impl Decoder<PoseEstimation> for PoseDecoder {
    type Event = PoseReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<PoseReply> {
        rig_core::providers::internal::wire::classify_marker_keyed_frame(
            &frame.as_str(),
            &["poses"],
        )
    }

    fn interpret(&mut self, reply: PoseReply, out: &mut Out<'_, PoseEstimation>) {
        for pose in reply.poses {
            out.push(Ok(pose));
        }
    }
}

const POSES: &str = r#"{"poses":[{"frame":0,"keypoints":[[0.5,0.25]],"last":false},{"frame":1,"keypoints":[[0.5,0.5]],"last":true}]}"#;

#[tokio::test]
async fn program_3_a_remote_pose_model_serves_the_operation_over_http() {
    let http = SequencedHttpClient::new([MockHttpResponse::success(POSES)]);
    let model = Model::new(
        PoseCloud {
            base_url: "https://pose.test".to_owned(),
        },
        http,
    );
    let track = model
        .call(Frames(vec![b"jpeg-0".to_vec(), b"jpeg-1".to_vec()]))
        .await
        .expect("the poses decode");
    assert_eq!(
        track.0.iter().map(|pose| pose.frame).collect::<Vec<_>>(),
        [0, 1]
    );
}

// 4. local, in process, one event per video frame

/// An in-process runtime (candle, ONNX): the transport of a local wire.
#[derive(Clone)]
struct OnnxRuntime;

impl Transport<Local<PoseEstimation>> for OnnxRuntime {
    fn send(
        &self,
        frames: Frames,
        _: Exchange,
    ) -> Result<Sending<Result<Pose, ProviderError>>, ProviderError> {
        let count = frames.0.len();
        let poses = futures::stream::iter(frames.0.into_iter().enumerate()).then(
            move |(frame, image)| async move {
                // One inference per frame.
                let x = image.len() as f32 / 10.0;
                Ok(Ok(Pose {
                    frame,
                    keypoints: vec![(x, x)],
                    last: frame + 1 == count,
                }))
            },
        );
        Ok(Sending::opened(Opened::new(poses)))
    }
}

#[tokio::test]
async fn program_4_a_local_pose_model_streams_one_event_per_frame() {
    let model = Model::new(Local::<PoseEstimation>::new("onnx"), OnnxRuntime);
    let video = Frames(vec![vec![0; 3], vec![0; 5], vec![0; 7]]);

    let mut stream = model.stream(video.clone()).expect("the stream opens");
    let mut frames = Vec::new();
    while let Some(pose) = stream.next().await {
        frames.push(pose.expect("a pose").frame);
    }
    assert_eq!(frames, [0, 1, 2]);
    assert_eq!(stream.finish().expect("the track folds").0.len(), 3);

    // Providers of one operation are interchangeable once erased.
    let erased: Vec<DynModel<PoseEstimation>> = vec![
        model.erase(),
        Model::new(
            PoseCloud {
                base_url: "https://pose.test".to_owned(),
            },
            SequencedHttpClient::new([MockHttpResponse::success(POSES)]),
        )
        .erase(),
    ];
    for model in erased {
        let track = tokio::spawn(model.call(video.clone()))
            .await
            .expect("the call runs")
            .expect("the track folds");
        assert_eq!(track.0.last().map(|pose| pose.last), Some(true));
    }
}

// ── 6. a custom transport under any HTTP wire ──────────────────────────

/// Sends every request through a gateway, whatever the wire: the wires do
/// not change.
#[derive(Clone)]
struct Gateway<H> {
    inner: H,
}

impl<W, H> Transport<W> for Gateway<H>
where
    W: Wire<Payload = Encoded, Frame = WireFrame>,
    H: HttpClientExt + Clone + 'static,
{
    fn send(
        &self,
        mut payload: Encoded,
        exchange: Exchange,
    ) -> Result<Sending<WireFrame>, ProviderError> {
        for request in &mut payload.requests {
            let uri = format!("https://gateway.test{}", request.uri().path());
            *request.uri_mut() = uri
                .parse()
                .map_err(|_| ProviderError::Request("the gateway path is not a URI".into()))?;
        }
        Transport::<W>::send(&self.inner, payload, exchange)
    }
}

#[tokio::test]
async fn program_6_a_custom_transport_carries_any_wire_unchanged() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(CLAUDE_TURN),
        MockHttpResponse::success(POSES),
    ]);
    let gateway = Gateway {
        inner: http.clone(),
    };
    let chat = Model::new(
        AnthropicConfig::new("key")
            .connect(http.clone())
            .completion("claude-sonnet-4-6")
            .wire,
        gateway.clone(),
    );
    let turn = chat.call("hi").await.expect("the turn decodes");
    assert_eq!(turn.text(), "Adding.");
    let pose = Model::new(
        PoseCloud {
            base_url: "https://pose.test".to_owned(),
        },
        gateway,
    );
    pose.call(Frames(vec![Vec::new()]))
        .await
        .expect("the poses decode");
    assert_eq!(
        http.requests()
            .into_iter()
            .map(|request| request.uri)
            .collect::<Vec<_>>(),
        [
            "https://gateway.test/v1/messages",
            "https://gateway.test/v1/pose"
        ]
    );
}

/// A streamed chat ends at the provider's terminal record, which carries the
/// turn's usage.
#[tokio::test]
async fn a_streamed_turn_ends_at_its_terminal_record() {
    let model = AnthropicConfig::new("key")
        .connect(MockStreamingClient {
            sse_bytes: Bytes::from_static(CLAUDE_STREAM.as_bytes()),
        })
        .completion("claude-sonnet-4-6");
    let events: Vec<_> = model
        .stream("hi")
        .expect("the stream opens")
        .collect()
        .await;
    assert!(matches!(
        events.last(),
        Some(Ok(StreamEvent::Final(terminal))) if terminal.usage.output_tokens == Some(7)
    ));
}
