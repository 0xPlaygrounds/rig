#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{Stream, StreamExt, stream};
use rig_agent::{
    Agent, AgentBuilder, AgentStreamFinal, ModelHandle,
    agent::{
        AgentHook, CompletionCallAction, HookContext, InvalidToolCallAction, ModelTurnAction,
        ModelTurnFinished, NoToolConfig, PromptRequest, Standard, StreamingError, StreamingResult,
        ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    },
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage,
        Message, Prompt, TypedPrompt, Usage,
    },
    extractor::{Extractor, ExtractorBuilder},
    streaming::{
        RawStreamingChoice, RawStreamingToolCall, StreamedAssistantContent,
        StreamingCompletionResponse, StreamingPrompt,
    },
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_core::{
    OneOrMany,
    message::{AssistantContent, ReasoningContent, ToolCall, ToolFunction, UserContent},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

async fn wait_for_notification(notify: &Notify) {
    tokio::time::timeout(Duration::from_secs(5), notify.notified())
        .await
        .expect("model was polled before timeout");
}

fn usage(total_tokens: u64) -> Usage {
    Usage {
        total_tokens,
        ..Usage::new()
    }
}

#[derive(Clone)]
enum Turn {
    Text {
        text: String,
        usage: Usage,
        message_id: String,
    },
    Tool {
        id: String,
        name: String,
        arguments: serde_json::Value,
        usage: Usage,
        message_id: String,
    },
    Rich {
        text: String,
        usage: Usage,
        message_id: String,
    },
}

impl Turn {
    fn text(text: &str, total_tokens: u64, message_id: &str) -> Self {
        Self::Text {
            text: text.to_owned(),
            usage: usage(total_tokens),
            message_id: message_id.to_owned(),
        }
    }

    fn tool(name: &str, total_tokens: u64, message_id: &str) -> Self {
        Self::Tool {
            id: format!("{name}-call"),
            name: name.to_owned(),
            arguments: serde_json::json!({"query": "rust"}),
            usage: usage(total_tokens),
            message_id: message_id.to_owned(),
        }
    }

    fn rich(text: &str, total_tokens: u64, message_id: &str) -> Self {
        Self::Rich {
            text: text.to_owned(),
            usage: usage(total_tokens),
            message_id: message_id.to_owned(),
        }
    }

    fn usage(&self) -> Usage {
        match self {
            Self::Text { usage, .. } | Self::Tool { usage, .. } | Self::Rich { usage, .. } => {
                *usage
            }
        }
    }

    fn message_id(&self) -> String {
        match self {
            Self::Text { message_id, .. }
            | Self::Tool { message_id, .. }
            | Self::Rich { message_id, .. } => message_id.clone(),
        }
    }

    fn choice(&self) -> OneOrMany<AssistantContent> {
        match self {
            Self::Text { text, .. } => OneOrMany::one(AssistantContent::text(text)),
            Self::Tool {
                id,
                name,
                arguments,
                ..
            } => OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                id.clone(),
                ToolFunction::new(name.clone(), arguments.clone()),
            ))),
            Self::Rich { text, .. } => OneOrMany::many(vec![
                AssistantContent::reasoning("considering the evidence"),
                AssistantContent::text(text),
            ])
            .expect("rich turn contains two items"),
        }
    }
}

struct Script {
    provider: &'static str,
    secret: String,
    turns: Mutex<VecDeque<Turn>>,
    fallback: Turn,
    requests: Mutex<Vec<CompletionRequest>>,
    composes_native_output_with_tools: bool,
}

impl Script {
    fn new(
        provider: &'static str,
        turns: impl IntoIterator<Item = Turn>,
        fallback: Turn,
    ) -> Arc<Self> {
        Arc::new(Self {
            provider,
            secret: format!("{provider}-credential-must-not-leak"),
            turns: Mutex::new(turns.into_iter().collect()),
            fallback,
            requests: Mutex::new(Vec::new()),
            composes_native_output_with_tools: false,
        })
    }

    fn composing(
        provider: &'static str,
        turns: impl IntoIterator<Item = Turn>,
        fallback: Turn,
    ) -> Arc<Self> {
        let mut script = Self {
            provider,
            secret: format!("{provider}-credential-must-not-leak"),
            turns: Mutex::new(turns.into_iter().collect()),
            fallback,
            requests: Mutex::new(Vec::new()),
            composes_native_output_with_tools: true,
        };
        script.secret.shrink_to_fit();
        Arc::new(script)
    }

    fn next_turn(&self) -> Turn {
        self.turns
            .lock()
            .expect("script turn lock")
            .pop_front()
            .unwrap_or_else(|| self.fallback.clone())
    }

    fn record(&self, request: CompletionRequest) {
        self.requests
            .lock()
            .expect("script request lock")
            .push(request);
    }

    fn requests(&self) -> Vec<CompletionRequest> {
        self.requests.lock().expect("script request lock").clone()
    }
}

trait TestRaw:
    Clone + Unpin + Send + Sync + Serialize + for<'de> Deserialize<'de> + GetTokenUsage + 'static
{
    fn new(provider: &str, usage: Usage) -> Self;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AlphaRaw {
    provider: String,
    usage: Usage,
}

impl TestRaw for AlphaRaw {
    fn new(provider: &str, usage: Usage) -> Self {
        Self {
            provider: provider.to_owned(),
            usage,
        }
    }
}

impl GetTokenUsage for AlphaRaw {
    fn token_usage(&self) -> Usage {
        self.usage
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BetaRaw {
    provider: String,
    usage: Usage,
}

impl TestRaw for BetaRaw {
    fn new(provider: &str, usage: Usage) -> Self {
        Self {
            provider: provider.to_owned(),
            usage,
        }
    }
}

impl GetTokenUsage for BetaRaw {
    fn token_usage(&self) -> Usage {
        self.usage
    }
}

fn completion_from_script<R>(script: &Script, request: CompletionRequest) -> CompletionResponse<R>
where
    R: TestRaw,
{
    script.record(request);
    let turn = script.next_turn();
    CompletionResponse {
        choice: turn.choice(),
        usage: turn.usage(),
        raw_response: R::new(script.provider, turn.usage()),
        message_id: Some(turn.message_id()),
    }
}

fn stream_from_script<R>(
    script: &Script,
    request: CompletionRequest,
) -> StreamingCompletionResponse<R>
where
    R: TestRaw,
{
    script.record(request);
    let turn = script.next_turn();
    let mut events = vec![Ok(RawStreamingChoice::MessageId(turn.message_id()))];
    match &turn {
        Turn::Text { text, .. } => {
            events.push(Ok(RawStreamingChoice::Message(text.clone())));
        }
        Turn::Tool {
            id,
            name,
            arguments,
            ..
        } => {
            let raw = RawStreamingToolCall::new(id.clone(), name.clone(), arguments.clone())
                .with_internal_call_id(format!("{id}-internal"));
            events.push(Ok(RawStreamingChoice::ToolCallDelta {
                id: id.clone(),
                internal_call_id: raw.internal_call_id.clone(),
                content: rig_agent::streaming::ToolCallDeltaContent::Name(name.clone()),
            }));
            events.push(Ok(RawStreamingChoice::ToolCallDelta {
                id: id.clone(),
                internal_call_id: raw.internal_call_id.clone(),
                content: rig_agent::streaming::ToolCallDeltaContent::Delta(arguments.to_string()),
            }));
            events.push(Ok(RawStreamingChoice::ToolCall(raw)));
        }
        Turn::Rich { text, .. } => {
            events.push(Ok(RawStreamingChoice::Reasoning {
                id: Some("reasoning-1".to_owned()),
                content: ReasoningContent::Summary("summary".to_owned()),
            }));
            events.push(Ok(RawStreamingChoice::ReasoningDelta {
                id: Some("reasoning-2".to_owned()),
                reasoning: "reasoning delta".to_owned(),
            }));
            events.push(Ok(RawStreamingChoice::Unknown(serde_json::json!({
                "type": "provider_native_event",
                "provider": script.provider,
            }))));
            events.push(Ok(RawStreamingChoice::Message(text.clone())));
        }
    }
    events.push(Ok(RawStreamingChoice::FinalResponse(R::new(
        script.provider,
        turn.usage(),
    ))));

    StreamingCompletionResponse::stream(Box::pin(stream::iter(events)))
}

#[derive(Clone)]
struct AlphaModel(Arc<Script>);

#[derive(Clone)]
struct BetaModel(Arc<Script>);

macro_rules! impl_test_model {
    ($model:ty, $raw:ty) => {
        impl CompletionModel for $model {
            type Response = $raw;
            type StreamingResponse = $raw;
            type Client = ();

            fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
                Self(Script::new(
                    "constructed",
                    [],
                    Turn::text("constructed", 0, "constructed-message"),
                ))
            }

            async fn completion(
                &self,
                request: CompletionRequest,
            ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
                Ok(completion_from_script(&self.0, request))
            }

            async fn stream(
                &self,
                request: CompletionRequest,
            ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
                Ok(stream_from_script(&self.0, request))
            }

            fn composes_native_output_with_tools(&self) -> bool {
                self.0.composes_native_output_with_tools
            }
        }
    };
}

impl_test_model!(AlphaModel, AlphaRaw);
impl_test_model!(BetaModel, BetaRaw);

fn alpha_static(text: &str) -> AlphaModel {
    AlphaModel(Script::new(
        "alpha",
        [],
        Turn::text(text, 1, "alpha-message"),
    ))
}

fn beta_static(text: &str) -> BetaModel {
    BetaModel(Script::new("beta", [], Turn::text(text, 2, "beta-message")))
}

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(Message::user(prompt)),
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

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct ExtractedValue {
    value: String,
}

fn assert_agent(_: Agent) {}
fn assert_builder(_: AgentBuilder<NoToolConfig>) {}
fn assert_prompt_request(_: PromptRequest<Standard>) {}
fn assert_extractor(_: Extractor<ExtractedValue>) {}
fn assert_agent_stream(_: StreamingResult) {}

#[tokio::test]
async fn downstream_models_keep_typed_low_level_apis_and_share_a_concrete_agent_type() {
    let alpha = alpha_static("alpha");
    let beta = beta_static("beta");

    let alpha_agent = AgentBuilder::new(alpha.clone()).build();
    let beta_agent = AgentBuilder::new(beta.clone()).build();
    let agents: Vec<Agent> = vec![alpha_agent.clone(), beta_agent];
    assert_eq!(agents.len(), 2);

    assert_agent(alpha_agent.clone());
    assert_builder(AgentBuilder::new(alpha.clone()));
    assert_prompt_request(alpha_agent.prompt("typed request"));
    assert_extractor(ExtractorBuilder::<ExtractedValue>::new(alpha.clone()).build());
    assert_agent_stream(alpha_agent.stream_prompt("stream type").await);

    let unary = alpha
        .completion(request("low-level unary"))
        .await
        .expect("typed unary response");
    let raw: AlphaRaw = unary.raw_response;
    assert_eq!(raw.provider, "alpha");

    let mut low_level_stream = beta
        .stream(request("low-level stream"))
        .await
        .expect("typed provider stream");
    let mut typed_final: Option<BetaRaw> = None;
    while let Some(item) = low_level_stream.next().await {
        if let StreamedAssistantContent::Final(final_) = item.expect("stream item") {
            typed_final = Some(final_);
        }
    }
    assert_eq!(
        typed_final.expect("typed final").provider,
        "beta",
        "direct model streams retain their provider raw type"
    );

    let extraction_turn = Turn::Tool {
        id: "submit-call".to_owned(),
        name: "submit".to_owned(),
        arguments: serde_json::json!({"value": "external model extraction"}),
        usage: usage(3),
        message_id: "extract-message".to_owned(),
    };
    let extracted = ExtractorBuilder::<ExtractedValue>::new(AlphaModel(Script::new(
        "extractor",
        [extraction_turn.clone()],
        extraction_turn,
    )))
    .build()
    .extract("extract a value")
    .await
    .expect("custom model extraction");
    assert_eq!(extracted.value, "external model extraction");

    let handle = ModelHandle::named("diagnostic-alpha", alpha);
    let debug = format!("{handle:?}");
    assert!(debug.contains("diagnostic-alpha"));
    assert!(!debug.contains("credential-must-not-leak"));
}

#[tokio::test]
async fn replacement_and_override_scopes_have_value_semantics() {
    let alpha = alpha_static("alpha");
    let beta = beta_static("beta");
    let beta_handle = ModelHandle::named("beta", beta.clone());
    let alpha_handle = ModelHandle::named("alpha", alpha.clone());

    let mut agent = AgentBuilder::new(alpha.clone()).build();
    let runner_before_replacement = agent.runner("runner snapshot");
    agent.set_model(beta.clone());

    assert_eq!(
        runner_before_replacement
            .run()
            .await
            .expect("old runner")
            .output,
        "alpha"
    );
    assert_eq!(
        agent.prompt("new runner").await.expect("new default"),
        "beta"
    );

    let original = AgentBuilder::new(alpha.clone()).build();
    let changed_clone = original.clone().with_model_handle(beta_handle.clone());
    assert_eq!(
        original.prompt("original").await.expect("original"),
        "alpha"
    );
    assert_eq!(
        changed_clone.prompt("clone").await.expect("changed clone"),
        "beta"
    );

    assert_eq!(
        original
            .prompt("one run")
            .using_model(beta_handle.clone())
            .await
            .expect("fixed override"),
        "beta"
    );
    assert_eq!(
        original.prompt("default remains").await.expect("default"),
        "alpha"
    );

    let typed: ExtractedValue = original
        .prompt_typed("typed one-run override")
        .using_model(ModelHandle::new(beta_static(r#"{"value":"typed beta"}"#)))
        .await
        .expect("typed override");
    assert_eq!(typed.value, "typed beta");

    assert_eq!(
        original
            .prompt("selector after fixed")
            .using_model(alpha_handle.clone())
            .select_model(move |_| beta_handle.clone())
            .await
            .expect("selector takes precedence"),
        "beta"
    );

    let beta_for_cleared_selector = ModelHandle::new(beta);
    assert_eq!(
        original
            .prompt("fixed after selector")
            .select_model(move |_| beta_for_cleared_selector.clone())
            .using_model(alpha_handle)
            .await
            .expect("fixed override clears selector"),
        "alpha"
    );
}

#[tokio::test]
async fn extraction_override_is_run_local_and_pins_all_retries() {
    let default_turn = Turn::Tool {
        id: "default-submit".to_owned(),
        name: "submit".to_owned(),
        arguments: serde_json::json!({"value": "default"}),
        usage: usage(1),
        message_id: "default-extraction".to_owned(),
    };
    let specialist_turn = Turn::Tool {
        id: "specialist-submit".to_owned(),
        name: "submit".to_owned(),
        arguments: serde_json::json!({"value": "specialist"}),
        usage: usage(2),
        message_id: "specialist-extraction".to_owned(),
    };
    let typed_turn = Turn::Tool {
        id: "typed-submit".to_owned(),
        name: "submit".to_owned(),
        arguments: serde_json::json!({"value": "typed specialist"}),
        usage: usage(3),
        message_id: "typed-extraction".to_owned(),
    };

    let default_script = Script::new("default", [], default_turn);
    let specialist_script = Script::new(
        "specialist",
        [
            Turn::text("retry without submit", 1, "specialist-retry"),
            specialist_turn.clone(),
        ],
        specialist_turn,
    );
    let extractor = ExtractorBuilder::<ExtractedValue>::new(AlphaModel(default_script.clone()))
        .retries(1)
        .build();

    let specialist = extractor
        .using_model(ModelHandle::new(BetaModel(specialist_script.clone())))
        .extract("use the specialist")
        .await
        .expect("run-local extraction override");
    assert_eq!(specialist.value, "specialist");
    assert_eq!(specialist_script.requests().len(), 2);
    assert!(default_script.requests().is_empty());

    let typed = extractor
        .using_model_value(BetaModel(Script::new("typed", [], typed_turn)))
        .extract_with_usage("use a typed model value")
        .await
        .expect("typed extraction override");
    assert_eq!(typed.data.value, "typed specialist");
    assert_eq!(typed.usage, usage(3));

    let default = extractor
        .extract("use the default again")
        .await
        .expect("default extraction remains unchanged");
    assert_eq!(default.value, "default");
    assert_eq!(default_script.requests().len(), 1);
}

#[derive(Clone)]
struct LookupTool {
    calls: Arc<AtomicUsize>,
}

impl Tool for LookupTool {
    const NAME: &'static str = "lookup";
    type Error = ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Look up deterministic evidence".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok("durable evidence".to_owned())
    }
}

#[derive(Clone, Default)]
struct LifecycleLog(Arc<Mutex<Vec<String>>>);

impl LifecycleLog {
    fn entries(&self) -> Vec<String> {
        self.0.lock().expect("lifecycle lock").clone()
    }

    fn push(&self, entry: impl Into<String>) {
        self.0.lock().expect("lifecycle lock").push(entry.into());
    }
}

impl AgentHook for LifecycleLog {
    async fn on_completion_call(
        &self,
        context: &HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.push(format!("completion:{}", context.turn()));
        CompletionCallAction::continue_run()
    }

    async fn on_model_turn_finished(
        &self,
        context: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.push(format!("model:{}", context.turn()));
        ModelTurnAction::continue_run()
    }

    async fn on_tool_call(
        &self,
        _context: &HookContext,
        event: ToolCallEvent<'_>,
    ) -> ToolCallAction {
        self.push(format!("tool-call:{}", event.tool_name));
        ToolCallAction::run()
    }

    async fn on_tool_result(
        &self,
        _context: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        self.push(format!("tool-result:{}", event.tool_name));
        ToolResultAction::keep()
    }
}

fn routing_models() -> (AlphaModel, BetaModel) {
    let alpha_turn = Turn::tool("lookup", 3, "alpha-tool-message");
    let beta_turn = Turn::text("synthesized answer", 5, "beta-answer-message");
    (
        AlphaModel(Script::new("alpha", [alpha_turn.clone()], alpha_turn)),
        BetaModel(Script::new("beta", [beta_turn.clone()], beta_turn)),
    )
}

fn history_has_tool_result(request: &CompletionRequest) -> bool {
    request.chat_history.iter().any(|message| {
        matches!(
            message,
            Message::User { content }
                if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
        )
    })
}

#[tokio::test]
async fn blocking_and_streaming_switch_after_tools_with_equivalent_semantics() {
    async fn run_blocking() -> (
        rig_agent::agent::PromptResponse,
        Vec<String>,
        usize,
        Vec<CompletionRequest>,
        Vec<String>,
    ) {
        let (alpha, beta) = routing_models();
        let beta_script = beta.0.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let lifecycle = LifecycleLog::default();
        let selected = Arc::new(Mutex::new(Vec::new()));
        let selected_for_router = selected.clone();
        let alpha_handle = ModelHandle::named("alpha", alpha);
        let beta_handle = ModelHandle::named("beta", beta);
        let response = AgentBuilder::from_model_handle(alpha_handle.clone())
            .tool(LookupTool {
                calls: calls.clone(),
            })
            .add_hook(lifecycle.clone())
            .build()
            .prompt("research then synthesize")
            .max_turns(3)
            .select_model(move |context| {
                selected_for_router
                    .lock()
                    .expect("selection lock")
                    .push(context.turn);
                if context.turn == 1 {
                    alpha_handle.clone()
                } else {
                    beta_handle.clone()
                }
            })
            .extended_details()
            .await
            .expect("blocking routed run");
        let selections = selected.lock().expect("selection lock").clone();
        (
            response,
            lifecycle.entries(),
            calls.load(Ordering::SeqCst),
            beta_script.requests(),
            selections
                .into_iter()
                .map(|turn| turn.to_string())
                .collect(),
        )
    }

    async fn run_streaming() -> (
        rig_agent::agent::PromptResponse,
        Vec<String>,
        usize,
        Vec<CompletionRequest>,
        Vec<usize>,
        Vec<&'static str>,
        Vec<String>,
    ) {
        let (alpha, beta) = routing_models();
        let beta_script = beta.0.clone();
        let calls = Arc::new(AtomicUsize::new(0));
        let lifecycle = LifecycleLog::default();
        let selected = Arc::new(Mutex::new(Vec::new()));
        let selected_for_router = selected.clone();
        let alpha_handle = ModelHandle::named("alpha", alpha);
        let beta_handle = ModelHandle::named("beta", beta);
        let agent = AgentBuilder::from_model_handle(alpha_handle.clone())
            .tool(LookupTool {
                calls: calls.clone(),
            })
            .add_hook(lifecycle.clone())
            .build();
        let mut stream = agent
            .stream_prompt("research then synthesize")
            .max_turns(3)
            .select_model(move |context| {
                selected_for_router
                    .lock()
                    .expect("selection lock")
                    .push(context.turn);
                if context.turn == 1 {
                    alpha_handle.clone()
                } else {
                    beta_handle.clone()
                }
            })
            .await;
        let mut final_response = None;
        let mut events = Vec::new();
        let mut internal_call_ids = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        internal_call_id, ..
                    },
                ) => {
                    events.push("tool-delta");
                    internal_call_ids.push(internal_call_id);
                }
                rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall {
                        internal_call_id, ..
                    },
                ) => {
                    events.push("tool-call");
                    internal_call_ids.push(internal_call_id);
                }
                rig_agent::agent::MultiTurnStreamItem::ToolExecutionCommitted {
                    internal_call_id,
                    ..
                } => {
                    events.push("tool-commit");
                    internal_call_ids.push(internal_call_id);
                }
                rig_agent::agent::MultiTurnStreamItem::StreamUserItem(
                    rig_agent::streaming::StreamedUserContent::ToolResult {
                        internal_call_id, ..
                    },
                ) => {
                    events.push("tool-result");
                    internal_call_ids.push(internal_call_id);
                }
                rig_agent::agent::MultiTurnStreamItem::FinalResponse(response) => {
                    final_response = Some(response)
                }
                _ => {}
            }
        }
        (
            final_response.expect("stream final response"),
            lifecycle.entries(),
            calls.load(Ordering::SeqCst),
            beta_script.requests(),
            selected.lock().expect("selection lock").clone(),
            events,
            internal_call_ids,
        )
    }

    let (blocking, blocking_hooks, blocking_tool_calls, blocking_beta, blocking_selected) =
        run_blocking().await;
    let (
        streaming,
        streaming_hooks,
        streaming_tool_calls,
        streaming_beta,
        streaming_selected,
        stream_events,
        stream_internal_call_ids,
    ) = run_streaming().await;

    assert_eq!(blocking.output, "synthesized answer");
    assert_eq!(blocking.output, streaming.output);
    assert_eq!(blocking.usage, usage(8));
    assert_eq!(blocking.usage, streaming.usage);
    assert_eq!(blocking.messages, streaming.messages);
    assert_eq!(blocking_hooks, streaming_hooks);
    assert_eq!(blocking_tool_calls, 1);
    assert_eq!(streaming_tool_calls, 1);
    assert_eq!(blocking_selected, vec!["1", "2"]);
    assert_eq!(streaming_selected, vec![1, 2]);
    assert!(blocking_beta.first().is_some_and(history_has_tool_result));
    assert!(streaming_beta.first().is_some_and(history_has_tool_result));
    assert_eq!(
        stream_events,
        vec![
            "tool-delta",
            "tool-delta",
            "tool-call",
            "tool-commit",
            "tool-result"
        ]
    );
    assert_eq!(
        stream_internal_call_ids,
        vec!["lookup-call-internal"; 5],
        "deltas, the completed call, execution, and result retain one correlation id"
    );
}

#[derive(Clone)]
struct RetryFirst(Arc<AtomicUsize>);

impl AgentHook for RetryFirst {
    async fn on_model_turn_finished(
        &self,
        _context: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            ModelTurnAction::repeat()
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

#[tokio::test]
async fn retries_reenter_selection_without_leaking_rejected_turn_state() {
    let alpha = alpha_static("rejected draft");
    let beta = beta_static("accepted answer");
    let alpha_handle = ModelHandle::named("alpha", alpha);
    let beta_script = beta.0.clone();
    let beta_handle = ModelHandle::named("beta", beta);
    let selections = Arc::new(Mutex::new(Vec::new()));
    let selections_for_router = selections.clone();

    let response = AgentBuilder::from_model_handle(alpha_handle.clone())
        .add_hook(RetryFirst(Arc::new(AtomicUsize::new(0))))
        .build()
        .prompt("try twice")
        .max_turns(2)
        .select_model(move |context| {
            selections_for_router.lock().expect("selection lock").push((
                context.turn,
                context
                    .previous_model
                    .and_then(ModelHandle::label)
                    .map(str::to_owned),
            ));
            if context.turn == 1 {
                alpha_handle.clone()
            } else {
                beta_handle.clone()
            }
        })
        .extended_details()
        .await
        .expect("retry routed run");

    assert_eq!(response.output, "accepted answer");
    assert_eq!(
        selections.lock().expect("selection lock").as_slice(),
        &[(1, None), (2, Some("alpha".to_owned()))]
    );
    let beta_request = beta_script
        .requests()
        .into_iter()
        .next()
        .expect("beta request");
    assert!(!beta_request.chat_history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "rejected draft"
                ))
        )
    }));
}

#[derive(Clone)]
struct RetryInvalidTool;

impl AgentHook for RetryInvalidTool {
    async fn on_invalid_tool_call(
        &self,
        _context: &HookContext,
        _event: &rig_agent::agent::InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::retry(
            "answer directly instead of calling that tool",
        ))
    }
}

#[tokio::test]
async fn invalid_tool_retry_reenters_selection_exactly_once() {
    let invalid = Turn::tool("missing_tool", 2, "invalid-tool-message");
    let alpha = AlphaModel(Script::new("alpha", [invalid.clone()], invalid));
    let beta = beta_static("recovered after invalid tool");
    let alpha_handle = ModelHandle::named("alpha", alpha);
    let beta_handle = ModelHandle::named("beta", beta);
    let selections = Arc::new(Mutex::new(Vec::new()));
    let selections_for_router = selections.clone();

    let output = AgentBuilder::from_model_handle(alpha_handle.clone())
        .add_hook(RetryInvalidTool)
        .build()
        .prompt("recover from an invalid tool")
        .max_turns(2)
        .max_invalid_tool_call_retries(1)
        .select_model(move |context| {
            selections_for_router
                .lock()
                .expect("selection lock")
                .push(context.turn);
            if context.turn == 1 {
                alpha_handle.clone()
            } else {
                beta_handle.clone()
            }
        })
        .await
        .expect("invalid-tool retry run");

    assert_eq!(output, "recovered after invalid tool");
    assert_eq!(
        selections.lock().expect("selection lock").as_slice(),
        &[1, 2]
    );
}

#[tokio::test]
async fn normalized_stream_preserves_events_message_id_usage_and_raw_json() {
    let rich = Turn::rich("final text", 13, "rich-message-id");
    let alpha = AlphaModel(Script::new("alpha", [rich.clone()], rich));
    let agent = AgentBuilder::new(alpha).build();
    let mut stream = agent.stream_prompt("rich stream").await;
    let mut saw_reasoning = false;
    let mut saw_reasoning_delta = false;
    let mut saw_unknown = false;
    let mut provider_final: Option<AgentStreamFinal> = None;
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        match item.expect("normalized stream item") {
            rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Reasoning(_),
            ) => saw_reasoning = true,
            rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { .. },
            ) => saw_reasoning_delta = true,
            rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Unknown(value),
            ) => {
                saw_unknown = value["type"] == "provider_native_event";
            }
            rig_agent::agent::MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Final(final_),
            ) => provider_final = Some(final_),
            rig_agent::agent::MultiTurnStreamItem::FinalResponse(response) => {
                final_response = Some(response)
            }
            _ => {}
        }
    }

    assert!(saw_reasoning);
    assert!(saw_reasoning_delta);
    assert!(saw_unknown);
    let provider_final = provider_final.expect("normalized provider final");
    assert_eq!(provider_final.usage, usage(13));
    assert_eq!(provider_final.raw_response["provider"], "alpha");
    assert_eq!(provider_final.raw_response["usage"]["total_tokens"], 13);
    let final_response = final_response.expect("agent final response");
    assert_eq!(final_response.output, "final text");
    assert_eq!(final_response.usage, usage(13));
    assert!(final_response.messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            matches!(message, Message::Assistant { id: Some(id), .. } if id == "rich-message-id")
        })
    }));
}

#[derive(Debug, Clone, Deserialize)]
struct SerializationFails;

impl Serialize for SerializationFails {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Err(serde::ser::Error::custom(
            "intentional raw response failure",
        ))
    }
}

impl GetTokenUsage for SerializationFails {
    fn token_usage(&self) -> Usage {
        usage(1)
    }
}

#[derive(Clone)]
struct SerializationFailingModel;

impl CompletionModel for SerializationFailingModel {
    type Response = AlphaRaw;
    type StreamingResponse = SerializationFails;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text("unused")),
            usage: usage(1),
            raw_response: AlphaRaw::new("bad", usage(1)),
            message_id: None,
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Ok(StreamingCompletionResponse::stream(Box::pin(stream::iter(
            [Ok(RawStreamingChoice::FinalResponse(SerializationFails))],
        ))))
    }
}

#[tokio::test]
async fn provider_final_serialization_failure_is_a_structured_stream_error() {
    let agent = AgentBuilder::new(SerializationFailingModel).build();
    let mut stream = agent.stream_prompt("serialize").await;
    let error = stream
        .next()
        .await
        .expect("stream error item")
        .expect_err("serialization must fail");
    assert!(matches!(
        error,
        StreamingError::Completion(CompletionError::JsonError(_))
    ));
}

#[tokio::test]
async fn selected_model_capability_is_used_for_each_prepared_attempt() {
    let composing_turn = Turn::text("retry me", 1, "compose-message");
    let composing = BetaModel(Script::composing(
        "composing",
        [composing_turn.clone()],
        composing_turn,
    ));
    let composing_script = composing.0.clone();
    let noncomposing_turn = Turn::text("accepted", 1, "noncompose-message");
    let noncomposing = AlphaModel(Script::new(
        "noncomposing",
        [noncomposing_turn.clone()],
        noncomposing_turn,
    ));
    let noncomposing_script = noncomposing.0.clone();
    let composing_handle = ModelHandle::new(composing);
    let noncomposing_handle = ModelHandle::new(noncomposing);

    let _response = AgentBuilder::from_model_handle(composing_handle.clone())
        .output_schema::<ExtractedValue>()
        .output_mode(rig_agent::agent::OutputMode::Auto)
        .tool(LookupTool {
            calls: Arc::new(AtomicUsize::new(0)),
        })
        .add_hook(RetryFirst(Arc::new(AtomicUsize::new(0))))
        .build()
        .prompt("capability routing")
        .max_turns(2)
        .select_model(move |context| {
            if context.turn == 1 {
                composing_handle.clone()
            } else {
                noncomposing_handle.clone()
            }
        })
        .await
        .expect("capability run");

    let composing_request = composing_script
        .requests()
        .into_iter()
        .next()
        .expect("composing request");
    assert!(composing_request.output_schema.is_some());
    assert!(
        !composing_request
            .tools
            .iter()
            .any(|tool| tool.name.starts_with("final_result"))
    );

    let noncomposing_request = noncomposing_script
        .requests()
        .into_iter()
        .next()
        .expect("noncomposing request");
    assert!(noncomposing_request.output_schema.is_none());
    assert!(
        noncomposing_request
            .tools
            .iter()
            .any(|tool| tool.name.starts_with("final_result"))
    );
}

#[derive(Clone)]
struct GatedToolModel {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

impl CompletionModel for GatedToolModel {
    type Response = AlphaRaw;
    type StreamingResponse = AlphaRaw;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            started: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        }
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        self.started.notify_one();
        self.release.notified().await;
        let turn = Turn::tool("lookup", 3, "gated-tool-message");
        Ok(CompletionResponse {
            choice: turn.choice(),
            usage: turn.usage(),
            raw_response: AlphaRaw::new("gated", turn.usage()),
            message_id: Some(turn.message_id()),
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Ok(StreamingCompletionResponse::stream(Box::pin(stream::iter(
            [
                Ok(RawStreamingChoice::Message("unused".to_owned())),
                Ok(RawStreamingChoice::FinalResponse(AlphaRaw::new(
                    "gated",
                    usage(1),
                ))),
            ],
        ))))
    }
}

#[tokio::test]
async fn routing_changes_cannot_rebind_an_in_flight_attempt_but_affect_the_next_call() {
    let gated = GatedToolModel {
        started: Arc::new(Notify::new()),
        release: Arc::new(Notify::new()),
    };
    let started = gated.started.clone();
    let release = gated.release.clone();
    let gated_handle = ModelHandle::named("gated", gated);
    let beta_handle = ModelHandle::named("beta", beta_static("after tool"));
    let use_beta = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let use_beta_for_router = use_beta.clone();
    let tool_calls = Arc::new(AtomicUsize::new(0));

    let agent = AgentBuilder::from_model_handle(gated_handle.clone())
        .tool(LookupTool {
            calls: tool_calls.clone(),
        })
        .build();
    let task = tokio::spawn(async move {
        agent
            .prompt("bind the attempt")
            .max_turns(2)
            .select_model(move |_| {
                if use_beta_for_router.load(Ordering::SeqCst) {
                    beta_handle.clone()
                } else {
                    gated_handle.clone()
                }
            })
            .await
    });

    wait_for_notification(&started).await;
    use_beta.store(true, Ordering::SeqCst);
    release.notify_one();

    assert_eq!(
        task.await
            .expect("routing task join")
            .expect("routing task result"),
        "after tool"
    );
    assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
}

struct DropGuard(Arc<AtomicUsize>);

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct PendingUnaryModel {
    started: Arc<Notify>,
    dropped: Arc<AtomicUsize>,
}

impl CompletionModel for PendingUnaryModel {
    type Response = AlphaRaw;
    type StreamingResponse = AlphaRaw;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            started: Arc::new(Notify::new()),
            dropped: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let _guard = DropGuard(self.dropped.clone());
        self.started.notify_one();
        std::future::pending::<Result<CompletionResponse<AlphaRaw>, CompletionError>>().await
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Ok(StreamingCompletionResponse::stream(Box::pin(
            stream::empty(),
        )))
    }
}

struct PendingRawStream {
    started: Arc<Notify>,
    dropped: Arc<AtomicUsize>,
    notified: bool,
}

impl Stream for PendingRawStream {
    type Item = Result<RawStreamingChoice<AlphaRaw>, CompletionError>;

    fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.notified {
            self.notified = true;
            self.started.notify_one();
        }
        Poll::Pending
    }
}

impl Drop for PendingRawStream {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct PendingStreamingModel {
    started: Arc<Notify>,
    dropped: Arc<AtomicUsize>,
}

impl CompletionModel for PendingStreamingModel {
    type Response = AlphaRaw;
    type StreamingResponse = AlphaRaw;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            started: Arc::new(Notify::new()),
            dropped: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text("unused")),
            usage: Usage::new(),
            raw_response: AlphaRaw::new("pending", Usage::new()),
            message_id: None,
        })
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let raw: rig_agent::streaming::StreamingResult<AlphaRaw> = Box::pin(PendingRawStream {
            started: self.started.clone(),
            dropped: self.dropped.clone(),
            notified: false,
        });
        Ok(StreamingCompletionResponse::stream(raw))
    }
}

#[tokio::test]
async fn dropping_pending_unary_and_streaming_attempts_cancels_by_drop() {
    let unary_started = Arc::new(Notify::new());
    let unary_dropped = Arc::new(AtomicUsize::new(0));
    let unary_agent = AgentBuilder::new(PendingUnaryModel {
        started: unary_started.clone(),
        dropped: unary_dropped.clone(),
    })
    .build();
    let unary_task = tokio::spawn(async move { unary_agent.prompt("pending unary").await });
    wait_for_notification(&unary_started).await;
    unary_task.abort();
    let _join_error = unary_task.await.expect_err("unary task was aborted");
    assert_eq!(unary_dropped.load(Ordering::SeqCst), 1);

    let stream_started = Arc::new(Notify::new());
    let stream_dropped = Arc::new(AtomicUsize::new(0));
    let stream_agent = AgentBuilder::new(PendingStreamingModel {
        started: stream_started.clone(),
        dropped: stream_dropped.clone(),
    })
    .build();
    let pending_stream = stream_agent.stream_prompt("pending stream").await;
    let stream_task = tokio::spawn(async move {
        let mut pending_stream = pending_stream;
        pending_stream.next().await
    });
    wait_for_notification(&stream_started).await;
    stream_task.abort();
    let _join_error = stream_task.await.expect_err("stream task was aborted");
    assert_eq!(stream_dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn concurrent_runs_and_handle_calls_are_independent() {
    let alpha_handle = ModelHandle::named("alpha", alpha_static("alpha concurrent"));
    let beta_handle = ModelHandle::named("beta", beta_static("beta concurrent"));
    let agent = AgentBuilder::from_model_handle(alpha_handle.clone()).build();

    let alpha_run = agent
        .prompt("alpha run")
        .select_model(move |_| alpha_handle.clone());
    let beta_run = agent
        .prompt("beta run")
        .select_model(move |_| beta_handle.clone());
    let (alpha, beta) = tokio::join!(alpha_run, beta_run);
    assert_eq!(alpha.expect("alpha concurrent run"), "alpha concurrent");
    assert_eq!(beta.expect("beta concurrent run"), "beta concurrent");

    let shared = ModelHandle::new(alpha_static("shared handle"));
    let first = agent.prompt("first shared").using_model(shared.clone());
    let second = agent.prompt("second shared").using_model(shared);
    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.expect("first shared call"), "shared handle");
    assert_eq!(second.expect("second shared call"), "shared handle");
}
