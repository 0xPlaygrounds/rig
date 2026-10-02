//! A completion decoder written outside `rig-core`, against the public
//! writer only: it writes each call under the vendor's own wire index, and
//! the writer opens it at its first fragment and closes it with the id and
//! name that arrived. Two calls interleave by wire index, and the first
//! call's id arrives after its arguments.

#![allow(clippy::expect_used, clippy::panic)]

use futures::{StreamExt, stream};
use rig_core::completion::{CompletionRequest, FinishReason, ReplayTarget, Usage};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message::{Api, AssistantContent};
use rig_core::operation::{Block, CallFragment, Completion, Finish, IfMalformed};
use rig_core::streaming::{Item, PartKind, StreamEvent};
use rig_core::wire::{Decoder, Descriptor, Flow, Mode, Out, Wire, WireEvent};

/// One frame of a made-up vendor's reply.
#[derive(Debug, Clone)]
enum Frame {
    Text(&'static str),
    /// A fragment of the call at `index`; its id and name may come late.
    Call {
        index: u32,
        id: Option<&'static str>,
        name: Option<&'static str>,
        arguments: &'static str,
    },
    Done,
}

#[derive(Debug, Clone, PartialEq)]
struct Vendor;

#[derive(Default)]
struct VendorDecoder<'id> {
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> Decoder<'id, Completion, Frame> for VendorDecoder<'id> {
    type Event = Frame;

    fn classify(&self, frame: Frame) -> WireEvent<Frame> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: Frame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match frame {
            Frame::Text(text) => {
                out.run(Block::Text, text)?;
            }
            Frame::Call {
                index,
                id,
                name,
                arguments,
            } => out.fragment(
                index as usize,
                CallFragment {
                    id,
                    name,
                    arguments: Some(arguments),
                },
            )?,
            Frame::Done => {
                out.end_run()?;
                for index in out.open_items() {
                    out.close(index, IfMalformed::Fail)?;
                }
                return Ok(out.end(Finish {
                    usage: Usage::default(),
                    reason: Some(FinishReason::ToolCalls),
                    ..Finish::default()
                }));
            }
        }
        Ok(Flow::More)
    }
}

impl ReplayTarget for Vendor {
    fn api(&self) -> Api {
        Api::from_static("vendor.chat")
    }

    fn provider(&self) -> &str {
        "vendor"
    }

    fn model(&self) -> &str {
        ""
    }
}

impl Wire for Vendor {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = Frame;
    type Decoder<'id> = VendorDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("vendor").replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<CompletionRequest, EncodeError> {
        Ok(request)
    }

    fn decoder<'id>(&self) -> VendorDecoder<'id> {
        VendorDecoder::default()
    }
}

/// Replays one recorded reply.
#[derive(Clone)]
struct Recorded(Vec<Frame>);

impl Transport<Vendor> for Recorded {
    fn send(&self, _payload: CompletionRequest, _exchange: Exchange) -> Opening<Frame> {
        Opening::ready(Opened::new(stream::iter(
            self.0.clone().into_iter().map(Ok),
        )))
    }
}

/// Text, then two calls whose fragments interleave; call 0's id arrives
/// only after two of its argument fragments.
fn reply() -> Vec<Frame> {
    vec![
        Frame::Text("Adding "),
        Frame::Call {
            index: 0,
            id: None,
            name: Some("add"),
            arguments: r#"{"a":"#,
        },
        Frame::Call {
            index: 1,
            id: Some("call_b"),
            name: Some("multiply"),
            arguments: r#"{"x":2,"#,
        },
        Frame::Text("and multiplying."),
        Frame::Call {
            index: 0,
            id: None,
            name: None,
            arguments: r#"1,"b":"#,
        },
        Frame::Call {
            index: 1,
            id: None,
            name: None,
            arguments: r#""y":3}"#,
        },
        Frame::Call {
            index: 0,
            id: Some("call_a"),
            name: None,
            arguments: "2}",
        },
        Frame::Done,
    ]
}

fn model() -> Model<Vendor, Recorded> {
    Model::new(Vendor, Recorded(reply()))
}

fn calls(choice: &[AssistantContent]) -> Vec<(String, String, serde_json::Value)> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some((
                call.id.to_string(),
                call.function.name.to_string(),
                call.function.arguments.clone(),
            )),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn interleaved_calls_with_late_ids_fold_with_their_ids() {
    let response = model()
        .call(CompletionRequest::new("add and multiply"))
        .await
        .expect("the reply folds");

    assert_eq!(response.text(), "Adding and multiplying.");
    assert_eq!(
        calls(&response.choice),
        vec![
            (
                "call_a".to_owned(),
                "add".to_owned(),
                serde_json::json!({"a": 1, "b": 2})
            ),
            (
                "call_b".to_owned(),
                "multiply".to_owned(),
                serde_json::json!({"x": 2, "y": 3})
            ),
        ],
        "each call carries the id that arrived late, in the order the calls closed"
    );
}

#[tokio::test]
async fn the_stream_shows_each_call_whole_and_finishes_as_call_does() {
    let mut stream = model()
        .stream(CompletionRequest::new("add and multiply"))
        .expect("the stream opens");
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("no item is an error") {
            Item::Event(event) => events.push(event),
            Item::Unknown(payload) => panic!("nothing unmodeled was sent: {payload:?}"),
        }
    }
    let streamed = stream.finish().await.expect("the stream finishes");
    let called = model()
        .call(CompletionRequest::new("add and multiply"))
        .await
        .expect("the reply folds");
    assert_eq!(streamed, called, "one decoder and one fold per reply");

    // Text is not held back by the calls buffered around it: all of it
    // streams before either call, which surfaces when it closes.
    let text: String = events
        .iter()
        .take_while(|event| {
            !matches!(
                event,
                StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    ..
                }
            )
        })
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Adding and multiplying.");

    // Each call surfaces with its arguments whole, in one event.
    let arguments: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Arguments { json, .. } => Some(json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(arguments, [r#"{"a":1,"b":2}"#, r#"{"x":2,"y":3}"#]);
}

#[test]
fn the_stream_of_an_out_of_tree_wire_is_send_and_static() {
    fn send_static<T: Send + 'static>(_: &T) {}
    let stream = model()
        .stream(CompletionRequest::new("add and multiply"))
        .expect("the stream opens");
    send_static(&stream);
}
