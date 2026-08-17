//! Completion helpers for deterministic agent-loop tests.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::{
    completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        Usage,
    },
    message::{ToolCall, ToolFunction},
    streaming::StreamingCompletionResponse,
};

use super::streaming::{MOCK_PROVIDER, MockStreamEvent};

/// Scripted error returned by [`MockCompletionModel`].
#[derive(Clone, Debug)]
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

    pub(crate) fn into_completion_error(self) -> CompletionError {
        match self {
            Self::Provider(message) => CompletionError::ProviderError(message),
            Self::Request(message) => CompletionError::RequestError(message.into()),
            Self::ProviderResponse(response) => CompletionError::ProviderResponse(response),
        }
    }
}

/// A scripted non-streaming mock completion turn.
#[derive(Clone, Debug)]
pub struct MockTurn {
    response: Result<MockTurnResponse, MockError>,
}

#[derive(Clone, Debug)]
struct MockTurnResponse {
    choice: Vec<AssistantContent>,
    usage: Usage,
    message_id: Option<String>,
    response_id: Option<String>,
    provider_request_id: Option<String>,
    finish_reason: Option<crate::completion::FinishReason>,
    raw: serde_json::Value,
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
                usage: Usage::new(),
                message_id: None,
                response_id: None,
                provider_request_id: None,
                finish_reason: None,
                raw: serde_json::Value::Null,
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
                usage: Usage::new(),
                message_id: None,
                response_id: None,
                provider_request_id: None,
                finish_reason: None,
                raw: serde_json::Value::Null,
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
    /// without a live provider. A turn without a scripted payload reports
    /// `raw: Value::Null`, so a non-null `raw` in a test means the scripted
    /// value arrived, never that the mock invented one.
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        if let Ok(response) = &mut self.response {
            response.raw = raw;
        }
        self
    }

    fn into_completion_response(self) -> Result<CompletionResponse, CompletionError> {
        let response = self.response.map_err(MockError::into_completion_error)?;
        Ok(
            CompletionResponse::new(response.choice, response.usage, MOCK_PROVIDER)
                .with_optional_message_id(response.message_id)
                .with_optional_response_id(response.response_id)
                .with_optional_provider_request_id(response.provider_request_id)
                .with_optional_finish_reason(response.finish_reason)
                .with_raw(response.raw),
        )
    }
}

#[derive(Default)]
struct MockCompletionModelState {
    turns: Mutex<VecDeque<MockTurn>>,
    stream_turns: Mutex<VecDeque<Vec<MockStreamEvent>>>,
    requests: Mutex<Vec<CompletionRequest>>,
}

/// A cloneable scripted [`CompletionModel`] for tests.
///
/// Each completion or stream call consumes exactly one scripted turn. If no turn
/// is available, the model returns [`CompletionError::ProviderError`] with a
/// clear message instead of repeating previous responses.
#[derive(Clone, Default)]
pub struct MockCompletionModel {
    state: Arc<MockCompletionModelState>,
}

impl MockCompletionModel {
    /// Create a mock model from scripted non-streaming turns.
    pub fn new(turns: impl IntoIterator<Item = MockTurn>) -> Self {
        Self::from_turns(turns)
    }

    /// Create a mock model that returns one text completion.
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_turns([MockTurn::text(text)])
    }

    /// Create a mock model from scripted non-streaming turns.
    pub fn from_turns(turns: impl IntoIterator<Item = MockTurn>) -> Self {
        Self {
            state: Arc::new(MockCompletionModelState {
                turns: Mutex::new(turns.into_iter().collect()),
                stream_turns: Mutex::new(VecDeque::new()),
                requests: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Create a mock model from scripted streaming turns.
    pub fn from_stream_turns(
        stream_turns: impl IntoIterator<Item = impl IntoIterator<Item = MockStreamEvent>>,
    ) -> Self {
        Self {
            state: Arc::new(MockCompletionModelState {
                turns: Mutex::new(VecDeque::new()),
                stream_turns: Mutex::new(
                    stream_turns
                        .into_iter()
                        .map(|turn| turn.into_iter().collect())
                        .collect(),
                ),
                requests: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Return cloned requests received by this model.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.requests_guard().clone()
    }

    /// Return the number of requests received by this model.
    pub fn request_count(&self) -> usize {
        self.requests_guard().len()
    }

    fn record_request(&self, request: CompletionRequest) {
        self.requests_guard().push(request);
    }

    fn next_turn(&self) -> Option<MockTurn> {
        self.turns_guard().pop_front()
    }

    fn next_stream_turn(&self) -> Option<Vec<MockStreamEvent>> {
        self.stream_turns_guard().pop_front()
    }

    fn turns_guard(&self) -> MutexGuard<'_, VecDeque<MockTurn>> {
        match self.state.turns.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn stream_turns_guard(&self) -> MutexGuard<'_, VecDeque<Vec<MockStreamEvent>>> {
        match self.state.stream_turns.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn requests_guard(&self) -> MutexGuard<'_, Vec<CompletionRequest>> {
        match self.state.requests.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl CompletionModel for MockCompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.record_request(request);
        let Some(turn) = self.next_turn() else {
            return Err(CompletionError::ProviderError(
                "mock completion model has no scripted completion turn".to_string(),
            ));
        };

        turn.into_completion_response()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        self.record_request(request);
        let Some(events) = self.next_stream_turn() else {
            return Err(CompletionError::ProviderError(
                "mock completion model has no scripted streaming turn".to_string(),
            ));
        };

        let stream = async_stream::stream! {
            for event in events {
                yield event.into_raw_choice();
            }
        };
        // Scripted terminals go through `normalize_stream` like every real
        // provider's, so the mock observes the same `Stop` -> `ToolCalls`
        // reconciliation callers see in production — and the same raw
        // capture: the mock's terminal type is `StreamFinal` itself, so `raw`
        // is the scripted terminal serialized.
        let stream = crate::streaming::normalize_stream(Box::pin(stream), Ok);
        Ok(StreamingCompletionResponse::stream(MOCK_PROVIDER, stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::Message,
        streaming::{StreamFinal, StreamedAssistantContent, ToolCallDeltaContent},
    };
    use futures::StreamExt;

    fn request(prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![Message::user(prompt)],
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[tokio::test]
    async fn completion_consumes_scripted_turns_and_records_requests() {
        let model = MockCompletionModel::new([
            MockTurn::text("first").with_message_id("msg_1"),
            MockTurn::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
                .with_call_id("call_1"),
        ]);

        let first = model
            .completion(request("hello"))
            .await
            .expect("first scripted turn should succeed");
        assert_eq!(first.message_id.as_deref(), Some("msg_1"));
        assert!(matches!(
            first.choice.first(),
            Some(AssistantContent::Text(text)) if text.text == "first"
        ));

        let second = model
            .completion(request("use a tool"))
            .await
            .expect("second scripted turn should succeed");
        assert!(matches!(
            second.choice.first(),
            Some(AssistantContent::ToolCall(tool_call))
                if tool_call.id == "tool_1"
                    && tool_call
                        .provider
                        .as_ref()
                        .is_some_and(|provider| provider.call_id == "call_1")
        ));

        assert_eq!(model.request_count(), 2);
        assert_eq!(model.requests().len(), 2);
    }

    /// The mock behaves like a real seam: a scripted raw payload rides on the
    /// normalized response unconditionally, and a turn that scripted none
    /// reports `raw: Value::Null` — the mock never invents a payload, so `Value::Null`
    /// here means "no provider record was scripted behind this turn".
    #[tokio::test]
    async fn completion_attaches_scripted_raw_and_reports_null_when_unscripted() {
        let payload = serde_json::json!({"provider_only": "kept", "id": "resp_1"});
        let model = MockCompletionModel::new([
            MockTurn::text("first").with_raw(payload.clone()),
            MockTurn::text("second"),
        ]);

        let scripted = model
            .completion(request("hello"))
            .await
            .expect("first scripted turn should succeed");
        assert_eq!(scripted.raw, payload);

        let unscripted = model
            .completion(request("hello"))
            .await
            .expect("second scripted turn should succeed");
        assert!(unscripted.raw.is_null());

        assert_eq!(model.requests().len(), 2);
    }

    /// The streaming half of the same contract: the scripted terminal goes
    /// through `normalize_stream`, so the terminal's `raw` is the scripted
    /// terminal record serialized (the mock's own terminal type is
    /// `StreamFinal`).
    #[tokio::test]
    async fn stream_terminal_raw_is_the_scripted_terminal_serialized() {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text("hello"),
            MockStreamEvent::final_response(Usage {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
                ..Usage::new()
            }),
        ]]);

        let mut stream = model
            .stream(request("hello"))
            .await
            .expect("stream should open");
        while stream.next().await.is_some() {}
        let terminal = stream.response.expect("terminal record");
        let raw = &terminal.raw;
        let typed: StreamFinal = serde_json::from_value(raw.clone()).expect("terminal type");
        assert_eq!(typed.usage.total_tokens, 3);
        assert!(
            typed.raw.is_null(),
            "the scripted terminal itself carried no raw (Value::Null)"
        );
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            *raw,
            "the capture must be exactly what the scripted terminal serializes to"
        );
        assert_eq!(terminal.usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn missing_completion_turn_returns_provider_error() {
        let model = MockCompletionModel::default();

        let err = model
            .completion(request("hello"))
            .await
            .expect_err("missing turn should error");

        assert!(matches!(
            err,
            CompletionError::ProviderError(message)
                if message.contains("no scripted completion turn")
        ));
    }

    #[tokio::test]
    async fn stream_yields_scripted_events_and_records_requests() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::message_id("msg_stream"),
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::tool_call_name_delta("tool_1", "calculator"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
            MockStreamEvent::tool_call("tool_1", "calculator", serde_json::json!({"x": 1}))
                .with_call_id("call_1"),
            MockStreamEvent::final_response_with_total_tokens(7),
        ]]);

        let mut stream = model
            .stream(request("stream"))
            .await
            .expect("stream should be created");

        let mut text = String::new();
        let mut saw_name_delta = false;
        let mut saw_arguments_delta = false;
        let mut saw_tool_call = false;
        let mut saw_final = false;

        while let Some(item) = stream.next().await {
            match item.expect("stream event should succeed") {
                StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                StreamedAssistantContent::ToolCallDelta { content, .. } => match content {
                    ToolCallDeltaContent::Name(name) => {
                        saw_name_delta = name == "calculator";
                    }
                    ToolCallDeltaContent::Delta(arguments) => {
                        saw_arguments_delta = arguments == "{\"x\":1}";
                    }
                },
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    saw_tool_call = tool_call
                        .provider
                        .as_ref()
                        .is_some_and(|provider| provider.call_id == "call_1");
                }
                StreamedAssistantContent::Final(response) => {
                    saw_final = matches!(
                        response.usage,
                        Usage {
                            total_tokens: 7,
                            ..
                        }
                    );
                }
                _ => {}
            }
        }

        assert_eq!(text, "hello");
        assert!(saw_name_delta);
        assert!(saw_arguments_delta);
        assert!(saw_tool_call);
        assert!(saw_final);
        assert_eq!(stream.message_id.as_deref(), Some("msg_stream"));
        assert_eq!(model.request_count(), 1);
    }

    #[tokio::test]
    async fn stream_error_event_is_returned() {
        let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::error("boom")]]);
        let mut stream = model
            .stream(request("stream"))
            .await
            .expect("stream should be created");

        let err = stream
            .next()
            .await
            .expect("stream should yield one event")
            .expect_err("scripted event should error");

        assert!(matches!(
            err,
            CompletionError::ProviderError(message) if message == "boom"
        ));
    }
}
