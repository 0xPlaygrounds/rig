//! Completion helpers for deterministic agent-loop tests.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::driver::{Model, Opened, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::operation::{AdapterOutput, Completion};
use crate::streaming::{
    BlockClose, BlockId, BlockKind, Delta, MintKind, StreamEvent, StreamFinal, SyntheticIds,
    ToolCallEnd,
};
use crate::wire::{Decoder, Mode, Wire, WireEvent, WireFrame};
use crate::{
    completion::{AssistantContent, CompletionRequest, Usage},
    message::{ToolCall, ToolFunction},
};

use super::streaming::{MOCK_PROVIDER, MockStreamEvent};

/// Scripted error returned by [`MockCompletionModel`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MockError {
    /// Provider error.
    Provider(String),
    /// Request construction error.
    Request(String),
    /// A preserved provider error response (rig#2314), id included.
    ProviderResponse(crate::provider_response::ProviderResponseError),
}

impl MockError {
    /// Create a provider error.
    pub fn provider(message: impl Into<String>) -> Self {
        Self::Provider(message.into())
    }

    /// Create a request error.
    pub fn request(message: impl Into<String>) -> Self {
        Self::Request(message.into())
    }

    pub(crate) fn into_completion_error(self) -> ProviderError {
        match self {
            Self::Provider(message) => ProviderError::Provider(message),
            Self::Request(message) => ProviderError::Request(message.into()),
            Self::ProviderResponse(response) => ProviderError::ProviderResponse(response),
        }
    }
}

/// A scripted non-streaming mock completion turn.
///
/// A turn is data: a script serializes, so a scripted model can be written
/// to a fixture and read back (see [`MockCompletionModel::script`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MockTurn {
    response: Result<MockTurnResponse, MockError>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct MockTurnResponse {
    choice: Vec<AssistantContent>,
    usage: Usage,
    message_id: Option<String>,
    response_id: Option<String>,
    provider_request_id: Option<String>,
    finish_reason: Option<crate::completion::FinishReason>,
    /// A scripted provider document, when the test supplies one; otherwise
    /// the turn itself, serialized, is the mock's document. Absent from the
    /// serialized turn when unscripted, so that document never nests
    /// itself; a scripted one survives a serde round trip of the script.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_scripted_raw"
    )]
    raw: Option<serde_json::Value>,
}

fn deserialize_scripted_raw<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error> {
    // Only a missing field means unscripted; explicit JSON null is a document.
    serde::Deserialize::deserialize(deserializer).map(Some)
}

impl MockTurn {
    /// Create a text response turn.
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_content(AssistantContent::text(text.into()))
    }

    /// Create a tool-call response turn.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::from_content(AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::new(name.into(), arguments),
        )))
    }

    /// Create a provider-error response turn.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            response: Err(MockError::provider(message)),
        }
    }

    /// Create a provider-response error turn carrying a transport request id
    /// (rig#2314): the scripted failure a test uses to assert error-identity
    /// attribution.
    pub fn provider_response_error(
        status: http::StatusCode,
        body: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Self {
        Self {
            response: Err(MockError::ProviderResponse(
                crate::provider_response::ProviderResponseError::new(status, body)
                    .with_provider_request_id(Some(request_id.into())),
            )),
        }
    }

    /// Create a request-error response turn.
    pub fn request_error(message: impl Into<String>) -> Self {
        Self {
            response: Err(MockError::request(message)),
        }
    }

    /// Create a response turn from one assistant content item.
    pub fn from_content(content: AssistantContent) -> Self {
        Self {
            response: Ok(MockTurnResponse {
                choice: vec![content],
                usage: Usage::default(),
                message_id: None,
                response_id: None,
                provider_request_id: None,
                finish_reason: None,
                raw: None,
            }),
        }
    }

    /// Create a response turn from assistant content items.
    ///
    /// Infallible now that content is a `Vec`: an empty turn is a shape a
    /// provider can genuinely return, so it is a value to build, not an error.
    pub fn from_contents(content: impl IntoIterator<Item = AssistantContent>) -> Self {
        Self {
            response: Ok(MockTurnResponse {
                choice: content.into_iter().collect(),
                usage: Usage::default(),
                message_id: None,
                response_id: None,
                provider_request_id: None,
                finish_reason: None,
                raw: None,
            }),
        }
    }

    /// Attach a provider-specific call ID to a tool-call response turn.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        let call_id = call_id.into();
        if let Ok(response) = &mut self.response {
            for content in response.choice.iter_mut() {
                if let AssistantContent::ToolCall(tool_call) = content {
                    tool_call.provider = crate::message::ProviderCallId::new(call_id);
                    break;
                }
            }
        }
        self
    }

    /// Override usage for this turn.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        if let Ok(response) = &mut self.response {
            response.usage = usage;
        }
        self
    }

    /// Set a provider-assigned assistant message ID for this turn.
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        if let Ok(response) = &mut self.response {
            response.message_id = Some(message_id.into());
        }
        self
    }

    /// Set a provider-assigned response-scoped ID for this turn.
    pub fn with_response_id(mut self, response_id: impl Into<String>) -> Self {
        if let Ok(response) = &mut self.response {
            response.response_id = Some(response_id.into());
        }
        self
    }

    /// Set a provider transport request id for this turn.
    pub fn with_provider_request_id(mut self, request_id: impl Into<String>) -> Self {
        if let Ok(response) = &mut self.response {
            response.provider_request_id = Some(request_id.into());
        }
        self
    }

    /// Set the terminal finish reason for this turn.
    ///
    /// Without this, a mocked blocking turn always reports `None`, which
    /// leaves the whole blocking half of the truncation contract (rig#2322)
    /// unexercisable — the streamed mock could script a reason and the
    /// blocking one could not.
    pub fn with_finish_reason(mut self, finish_reason: crate::completion::FinishReason) -> Self {
        if let Ok(response) = &mut self.response {
            response.finish_reason = Some(finish_reason);
        }
        self
    }

    /// Script the provider's own response for this turn — what a real seam
    /// would serialize from its raw type. Attached to the response as-is, so
    /// agent tests can prove the payload reaches every observer of the turn
    /// without a live provider. A turn without a scripted payload carries
    /// the scripted turn itself, serialized — the mock's own document, the
    /// same capture every real adapter performs — so a scripted value in a
    /// test is distinguishable from the mock's default by content.
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        if let Ok(response) = &mut self.response {
            response.raw = Some(raw);
        }
        self
    }

    /// The provider document the mock attaches to this turn's response: the
    /// scripted payload when one was supplied, otherwise the turn itself,
    /// serialized. Public so a test can state the expected `raw` of a
    /// recorded call without repeating the mock's serialization. An error
    /// turn has no document.
    pub fn raw(&self) -> Result<serde_json::Value, ProviderError> {
        let response = self
            .response
            .as_ref()
            .map_err(|error| error.clone().into_completion_error())?;
        match &response.raw {
            Some(raw) => Ok(raw.clone()),
            None => Ok(serde_json::to_value(response)?),
        }
    }
}

type MockInvocation = (CompletionRequest, Option<crate::observe::AdapterContext>);

#[derive(Default)]
struct MockScriptState {
    turns: Mutex<VecDeque<MockTurn>>,
    stream_turns: Mutex<VecDeque<Vec<MockStreamEvent>>>,
    requests: Mutex<Vec<MockInvocation>>,
}

/// The scripted transport behind [`MockCompletionModel`]: it records each
/// request with the observation context its call carried, and answers with
/// the next scripted turn.
///
/// Each call consumes exactly one turn of its mode. With none left, the call
/// fails with [`ProviderError::Provider`] and a clear message instead of
/// repeating an earlier answer.
#[derive(Clone, Default)]
pub struct MockScript {
    state: Arc<MockScriptState>,
}

/// The mock completion endpoint. Its payload is the request itself, and its
/// reply frames are the scripted turn, serialized.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MockWire;

/// A cloneable scripted completion model for tests: the mock wire over its
/// script, driven like any provider. Clones share the script and the
/// recorded requests.
pub type MockCompletionModel = Model<MockWire, MockScript>;

impl MockCompletionModel {
    /// Create a mock model that returns one text completion.
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_turns([MockTurn::text(text)])
    }

    /// Create a mock model from scripted non-streaming turns.
    pub fn from_turns(turns: impl IntoIterator<Item = MockTurn>) -> Self {
        Self::scripted(turns.into_iter().collect(), VecDeque::new())
    }

    /// Create a mock model from scripted streaming turns.
    pub fn from_stream_turns(
        stream_turns: impl IntoIterator<Item = impl IntoIterator<Item = MockStreamEvent>>,
    ) -> Self {
        Self::scripted(
            VecDeque::new(),
            stream_turns
                .into_iter()
                .map(|turn| turn.into_iter().collect())
                .collect(),
        )
    }

    fn scripted(turns: VecDeque<MockTurn>, stream_turns: VecDeque<Vec<MockStreamEvent>>) -> Self {
        Model::new(
            MockWire,
            MockScript {
                state: Arc::new(MockScriptState {
                    turns: Mutex::new(turns),
                    stream_turns: Mutex::new(stream_turns),
                    requests: Mutex::new(Vec::new()),
                }),
            },
        )
    }

    /// Return cloned requests received by this model.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        lock(&self.transport.state.requests)
            .iter()
            .map(|(request, _)| request.clone())
            .collect()
    }

    /// Return invocation contexts in the same order as the captured requests.
    pub fn contexts(&self) -> Vec<Option<crate::observe::AdapterContext>> {
        lock(&self.transport.state.requests)
            .iter()
            .map(|(_, context)| context.clone())
            .collect()
    }

    /// Return the number of requests received by this model.
    pub fn request_count(&self) -> usize {
        lock(&self.transport.state.requests).len()
    }

    /// The non-streaming turns not yet consumed, in order: the read-back
    /// half of the script, so a script is serde in and serde out.
    pub fn script(&self) -> Vec<MockTurn> {
        lock(&self.transport.state.turns).iter().cloned().collect()
    }

    /// The streaming turns not yet consumed, in order.
    pub fn stream_script(&self) -> Vec<Vec<MockStreamEvent>> {
        lock(&self.transport.state.stream_turns)
            .iter()
            .cloned()
            .collect()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// One decoded frame of the mock endpoint's reply.
pub struct MockFrame(Frame);

/// A whole scripted turn, or one scripted stream event.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Frame {
    Turn(MockTurnResponse),
    Event(MockStreamEvent),
}

impl Wire for MockWire {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = WireFrame;
    type Decoder = MockDecoder;

    fn name(&self) -> &str {
        MOCK_PROVIDER
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<CompletionRequest, EncodeError> {
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> MockDecoder {
        MockDecoder {
            // An id-less scripted tool call mints per stream, like a wire
            // that carries no ids (`tool-0`, `tool-1`, …).
            tool_ids: SyntheticIds::tool(),
            written: AdapterOutput::new(),
            open_minted_reasoning: Vec::new(),
        }
    }
}

impl Transport for MockScript {
    type Payload = CompletionRequest;
    type Frame = WireFrame;

    async fn send(
        &self,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Opened<CompletionRequest, WireFrame>, ProviderError> {
        let context = request
            .extensions
            .get::<crate::observe::AdapterContext>()
            .cloned();
        lock(&self.state.requests).push((request, context));
        let frames = match mode {
            Mode::Unary => {
                let turn = lock(&self.state.turns).pop_front().ok_or_else(|| {
                    ProviderError::Provider(
                        "mock completion model has no scripted completion turn".to_string(),
                    )
                })?;
                let raw = turn.raw()?;
                let response = turn.response.map_err(MockError::into_completion_error)?;
                let frame = serde_json::to_string(&Frame::Turn(response))?;
                (vec![Ok(WireFrame::Text(frame))], Some(raw))
            }
            Mode::Streaming => {
                let events = lock(&self.state.stream_turns).pop_front().ok_or_else(|| {
                    ProviderError::Provider(
                        "mock completion model has no scripted streaming turn".to_string(),
                    )
                })?;
                let frames = events
                    .into_iter()
                    .map(|event| {
                        serde_json::to_string(&Frame::Event(event))
                            .map(WireFrame::Text)
                            .map_err(ProviderError::from)
                    })
                    .collect();
                (frames, None)
            }
        };
        let (frames, raw) = frames;
        let body = raw
            .map(|raw| serde_json::to_vec(&raw))
            .transpose()?
            .map(bytes::Bytes::from);
        Ok(Opened {
            body,
            ..Opened::new(futures::stream::iter(frames))
        })
    }
}

/// Decodes the mock endpoint's frames: a unary turn into the events a
/// stream sends for its parts, and a scripted stream event through the same
/// [`AdapterOutput`] helpers every real adapter uses.
///
/// A script need not spell every reasoning end: like a boundary-less wire's
/// adapter, the mock closes a minted reasoning block before text or tool
/// content, so a script obeys the wire grammar.
pub struct MockDecoder {
    tool_ids: SyntheticIds,
    written: AdapterOutput,
    open_minted_reasoning: Vec<BlockId>,
}

impl Decoder<Completion> for MockDecoder {
    type Event = MockFrame;

    fn classify(&self, frame: WireFrame) -> WireEvent<MockFrame> {
        match serde_json::from_str(&frame.as_str()) {
            Ok(frame) => WireEvent::Known(MockFrame(frame)),
            Err(error) => WireEvent::Corrupt(error),
        }
    }

    fn interpret(&mut self, MockFrame(frame): MockFrame, out: &mut AdapterOutput) {
        match frame {
            Frame::Turn(response) => turn_events(response, out),
            Frame::Event(event) => {
                if let Err(error) = event.emit(&mut self.written, &mut self.tool_ids) {
                    self.written.error(error);
                }
                let written: Vec<_> = self.written.drain().collect();
                for item in written {
                    if let Ok(event) = &item {
                        self.close_minted_reasoning_before(event, out);
                    }
                    out.push(item);
                }
            }
        }
    }
}

impl MockDecoder {
    /// Close the open minted reasoning blocks when `event` is text or tool
    /// content, and track which minted reasoning blocks it opens or closes.
    fn close_minted_reasoning_before(&mut self, event: &StreamEvent, out: &mut AdapterOutput) {
        let content = matches!(
            event,
            StreamEvent::BlockStart {
                kind: BlockKind::Text { .. } | BlockKind::ToolCall,
                ..
            } | StreamEvent::BlockDelta {
                delta: Delta::Text { .. }
                    | Delta::TextMeta { .. }
                    | Delta::ToolName { .. }
                    | Delta::ToolArguments { .. },
                ..
            }
        );
        if content {
            for id in self.open_minted_reasoning.drain(..) {
                out.push(Ok(StreamEvent::BlockEnd {
                    id,
                    end: BlockClose::Reasoning {
                        reasoning: None,
                        signature: None,
                        wire_sent: false,
                    },
                    block: None,
                }));
            }
        }
        match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { .. },
            }
            | StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { .. },
            } if id.is_minted() && !self.open_minted_reasoning.contains(id) => {
                self.open_minted_reasoning.push(id.clone());
            }
            StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning { .. },
                ..
            } => self.open_minted_reasoning.retain(|open| open != id),
            _ => {}
        }
    }
}

/// A whole turn as the events a stream sends for it: one complete block per
/// part, in order, closed by the terminal record.
fn turn_events(response: MockTurnResponse, out: &mut AdapterOutput) {
    for (index, content) in response.choice.into_iter().enumerate() {
        let index = index as u64;
        match content {
            AssistantContent::Text(text) => {
                let id = BlockId::minted(MintKind::Text, index);
                out.text_start(id.clone(), text.additional_params);
                out.text(text.text);
                out.text_end(id);
            }
            AssistantContent::Reasoning(reasoning) => {
                let id = reasoning
                    .id
                    .as_deref()
                    .map(BlockId::wire)
                    .unwrap_or_else(|| BlockId::minted(MintKind::Reasoning, index));
                out.reasoning_end(id, Some(reasoning), None, true);
            }
            AssistantContent::ToolCall(call) => {
                let mut end = ToolCallEnd::whole(call.function.name, call.function.arguments)
                    .with_durable_id(call.id)
                    .with_signature(call.signature)
                    .with_additional_params(call.additional_params);
                if let Some(provider) = call.provider {
                    end = match provider.item_id {
                        Some(item_id) => end.with_call_id(provider.call_id).with_tool_id(item_id),
                        None => end.with_tool_id(provider.call_id),
                    };
                }
                out.tool_call(BlockId::minted(MintKind::Tool, index), end);
            }
            // Images have no block of their own: they travel as unmodeled
            // payloads, as a relayed turn carries them.
            AssistantContent::Image(image) => match serde_json::to_value(image) {
                Ok(value) => out.unknown(crate::streaming::UnknownPayload::new(value)),
                Err(error) => out.error(ProviderError::Json(error)),
            },
        }
    }
    out.final_record(
        StreamFinal::new(MOCK_PROVIDER, response.usage, serde_json::Value::Null)
            .with_optional_message_id(response.message_id)
            .with_optional_response_id(response.response_id)
            .with_optional_provider_request_id(response.provider_request_id)
            .with_optional_finish_reason(response.finish_reason),
    );
}

#[cfg(test)]
mod tests;
