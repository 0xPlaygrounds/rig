//! Completion helpers for deterministic agent-loop tests.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::driver::{Exchange, Model, Opened, Opening, Transport};
use crate::error::{EncodeError, ProviderError};
use crate::operation::Completion;
use crate::wire::{Capabilities, Descriptor, Mode, Wire};
use crate::{
    completion::{AssistantContent, CompletionRequest, CompletionResponse, Usage},
    message::{ToolCall, ToolFunction},
};

use super::streaming::{MOCK_PROVIDER, MockDecoder, MockFrame, MockStreamEvent};

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
            Self::Request(message) => ProviderError::request(message),
            Self::ProviderResponse(response) => ProviderError::ProviderResponse(response),
        }
    }
}

/// A scripted non-streaming mock completion turn.
///
/// A turn is data: a script serializes, so a scripted model can be written
/// to a fixture and read back (see `MockCompletionModel::script`).
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

    /// Create a tool-call response turn. An empty `name` scripts a provider
    /// error instead: no call can be built without a name.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        match crate::message::ToolName::new(name) {
            Ok(name) => Self::from_content(AssistantContent::ToolCall(ToolCall::from_wire(
                id,
                ToolFunction::new(name, arguments),
            ))),
            Err(error) => Self::error(error.to_string()),
        }
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
                    tool_call.id = crate::message::CallId::from_wire(call_id);
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

    fn into_completion_response(self) -> Result<CompletionResponse, ProviderError> {
        let raw = self.raw()?;
        let response = self.response.map_err(MockError::into_completion_error)?;
        let mut completion =
            CompletionResponse::new(response.choice, response.usage, MOCK_PROVIDER, raw)
                .with_optional_finish_reason(response.finish_reason);
        completion.message_id = response.message_id;
        completion.response_id = response.response_id;
        completion.provider_request_id = response.provider_request_id;
        Ok(completion)
    }
}

type MockInvocation = (CompletionRequest, Option<crate::observe::AdapterContext>);

#[derive(Default)]
struct MockScriptState {
    turns: Mutex<VecDeque<MockTurn>>,
    stream_turns: Mutex<VecDeque<Vec<MockStreamEvent>>>,
    requests: Mutex<Vec<MockInvocation>>,
}

/// The scripted completion wire: its payload is the request, and each frame
/// is a scripted step its decoder writes through the completion writer's
/// part handles, as any wire's decoder does. A runtime that answers from a
/// script ([`MockRuntime`], or a test's own) is its transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockScript {
    name: String,
    id: Option<String>,
    capabilities: Capabilities,
}

impl MockScript {
    /// A scripted wire named `name`, as records and telemetry name it, with
    /// default capabilities.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: None,
            capabilities: Capabilities::default(),
        }
    }

    /// The same wire, addressing the model `id`.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The same wire, reporting `capabilities`.
    pub fn with_capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = capabilities;
        self
    }
}

impl Default for MockScript {
    fn default() -> Self {
        Self::new(MOCK_PROVIDER)
    }
}

impl Wire for MockScript {
    type Op = Completion;
    type Payload = CompletionRequest;
    type Frame = MockFrame;
    type Decoder<'id> = MockDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(&self.name)
            .model(self.id.as_deref())
            .capabilities(self.capabilities)
    }

    /// A completion replays only the reasoning this runtime issued, as every
    /// completion wire's encode scopes it.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<CompletionRequest, EncodeError> {
        let issuers = [crate::message::Issuer::from(self.name.clone())];
        let mut request = request.replayable_to(&issuers, None)?;
        for message in request.chat_history.iter_mut() {
            // `replayable_to` left only messages with a part to keep.
            if let crate::message::Message::Assistant { content, .. } = message {
                content.retain(|part| match part {
                    AssistantContent::Reasoning(reasoning) => {
                        reasoning.open_for(&issuers).is_some()
                    }
                    _ => true,
                });
            }
        }
        Ok(request)
    }

    fn decoder<'id>(&self) -> MockDecoder<'id> {
        MockDecoder::default()
    }
}

/// The scripted runtime behind [`MockCompletionModel`]: the transport of a
/// [`MockScript`] wire named [`MOCK_PROVIDER`].
///
/// Each call consumes exactly one scripted turn. If no turn is available,
/// the call fails with [`ProviderError::Provider`] and a clear message
/// instead of repeating previous responses.
#[derive(Clone, Default)]
pub struct MockRuntime {
    state: Arc<MockScriptState>,
}

/// A cloneable scripted completion model for tests: the scripted wire over
/// its runtime. Clones share the script and the recorded requests.
pub type MockCompletionModel = Model<MockScript, MockRuntime>;

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
            MockScript::default(),
            MockRuntime {
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
        self.transport
            .requests_guard()
            .iter()
            .map(|(request, _)| request.clone())
            .collect()
    }

    /// Return invocation contexts in the same order as the captured requests.
    pub fn contexts(&self) -> Vec<Option<crate::observe::AdapterContext>> {
        self.transport
            .requests_guard()
            .iter()
            .map(|(_, context)| context.clone())
            .collect()
    }

    /// Return the number of requests received by this model.
    pub fn request_count(&self) -> usize {
        self.transport.requests_guard().len()
    }

    /// The non-streaming turns not yet consumed, in order — the read-back
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

impl MockRuntime {
    fn requests_guard(&self) -> MutexGuard<'_, Vec<MockInvocation>> {
        lock(&self.state.requests)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Transport<MockScript> for MockRuntime {
    fn send(&self, request: CompletionRequest, exchange: Exchange) -> Opening<MockFrame> {
        let mode = exchange.mode;
        self.requests_guard().push((request, exchange.observation));
        match mode {
            // A whole turn is one response, its document the response's
            // `raw`; a scripted failure fails the reply.
            Mode::Unary => {
                let Some(turn) = lock(&self.state.turns).pop_front() else {
                    return Opening::failed(ProviderError::Provider(
                        "mock completion model has no scripted completion turn".to_string(),
                    ));
                };
                match turn.into_completion_response() {
                    Ok(response) => {
                        let document = response.raw.clone();
                        let request_id = response.provider_request_id.clone();
                        Opening::ready(
                            Opened::new(futures::stream::iter([Ok(MockFrame::Response(
                                Box::new(response),
                            ))]))
                            .with_document(document)
                            .with_request_id(request_id),
                        )
                    }
                    Err(error) => Opening::ready(Opened::failed(error)),
                }
            }
            Mode::Streaming => {
                let Some(turn) = lock(&self.state.stream_turns).pop_front() else {
                    return Opening::failed(ProviderError::Provider(
                        "mock completion model has no scripted streaming turn".to_string(),
                    ));
                };
                let request_id = turn.iter().find_map(|event| match event {
                    MockStreamEvent::RequestId(id) => Some(id.clone()),
                    _ => None,
                });
                Opening::ready(
                    Opened::new(futures::stream::iter(
                        turn.into_iter()
                            .filter(|event| !matches!(event, MockStreamEvent::RequestId(_)))
                            .map(|event| Ok(MockFrame::Event(event))),
                    ))
                    .with_request_id(request_id),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests;
