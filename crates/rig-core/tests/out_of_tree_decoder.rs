//! A completion decoder written outside `rig-core`, against the public
//! writer only: it writes each call under the vendor's own wire index, and
//! the writer opens it at its first fragment and closes it with the id and
//! name that arrived. Two calls interleave by wire index, and the first
//! call's id arrives after its arguments.

#![allow(clippy::expect_used, clippy::panic)]

use futures::stream;
use rig_core::completion::{CompletionRequest, FinishReason, ReplayTarget, Usage};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message::{Api, AssistantContent};
use rig_core::operation::{Block, CallFragment, Completion, Finish};
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
                Some(index as usize),
                CallFragment {
                    id,
                    name,
                    arguments: Some(arguments),
                },
            )?,
            Frame::Done => {
                out.end_run()?;
                out.finish_open()?;
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

    fn accepts(&self, _model: &str) -> rig_core::completion::Accepts {
        rig_core::completion::Accepts::ALL
    }

    /// The vendor takes no option, so each set one is refused.
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = fields;
        let refused = |set: bool| match set {
            true => Mapping::unsupported("the vendor takes no options"),
            false => Mapping::Nothing,
        };
        OptionMap {
            reasoning: refused(reasoning.is_some()),
            cache: refused(cache.is_some()),
            service_tier: refused(service_tier.is_some()),
            verbosity: refused(verbosity.is_some()),
            parallel_tool_calls: refused(parallel_tool_calls.is_some()),
            top_p: refused(top_p.is_some()),
            seed: refused(seed.is_some()),
            stop: refused(!stop.is_empty()),
        }
    }
}

impl Wire for Vendor {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = Frame;
    type Decoder<'id> = VendorDecoder<'id>;
    type Reassembler = rig_core::wire::document::Unreassembled;

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
                call.function.arguments_value(),
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
