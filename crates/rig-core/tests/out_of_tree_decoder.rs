//! A completion decoder written outside `rig-core`, against the public
//! writer only: it keeps its own call bookkeeping, holds branded part
//! handles across frames, and opens a call once both its id and its name
//! have arrived. Two calls interleave by wire index, and the first call's
//! id arrives after its arguments.

#![allow(clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use futures::{StreamExt, stream};
use rig_core::completion::{CompletionRequest, FinishReason, Usage};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message::{AssistantContent, CallId, ToolName};
use rig_core::operation::{CallPart, Completion, Finish, TextPart};
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

/// A call the decoder has seen part of: buffered until its id and name
/// are both known, then open on the reply.
enum Call<'id> {
    Buffered {
        id: Option<&'static str>,
        name: Option<&'static str>,
        arguments: String,
    },
    Open(CallPart<'id>),
}

#[derive(Default)]
struct VendorDecoder<'id> {
    text: Option<TextPart<'id>>,
    calls: BTreeMap<u32, Call<'id>>,
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
                let part = self.text.get_or_insert_with(|| out.text());
                out.push_text(part, text);
            }
            Frame::Call {
                index,
                id,
                name,
                arguments,
            } => {
                let call = self.calls.entry(index).or_insert(Call::Buffered {
                    id: None,
                    name: None,
                    arguments: String::new(),
                });
                match call {
                    Call::Open(part) => out.push_arguments(part, arguments),
                    Call::Buffered {
                        id: seen_id,
                        name: seen_name,
                        arguments: buffered,
                    } => {
                        *seen_id = seen_id.or(id);
                        *seen_name = seen_name.or(name);
                        buffered.push_str(arguments);
                        if let (Some(id), Some(name)) = (*seen_id, *seen_name) {
                            let name = ToolName::new(name)
                                .map_err(|error| ProviderError::Provider(error.to_string()))?;
                            let part = out.call(CallId::from_wire(id), name)?;
                            out.push_arguments(&part, buffered);
                            *call = Call::Open(part);
                        }
                    }
                }
            }
            Frame::Done => {
                if let Some(part) = self.text.take() {
                    out.close_text(part);
                }
                for (_, call) in std::mem::take(&mut self.calls) {
                    if let Call::Open(part) = call {
                        out.close_call(part)?;
                    }
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

impl Wire for Vendor {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = Frame;
    type Decoder<'id> = VendorDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new("vendor")
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
