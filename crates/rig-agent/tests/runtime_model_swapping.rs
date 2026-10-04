#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures::{StreamExt, stream};
use rig_agent::{
    Agent, AgentBuilder, ModelRef,
    agent::{
        AgentHook, AgentRunner, CompletionCallAction, HookContext, InvalidToolCallAction,
        ModelSelection, ModelSelectionAction, ModelTurnAction, ModelTurnFinished, NoToolConfig,
        RequestPatch, StreamingResult,
    },
    completion::{
        CompletionRequest, CompletionResponse, Message, PromptError, ProviderCapabilities, Usage,
    },
    extractor::{Extractor, ExtractorBuilder},
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::operation::Finish;
use rig_core::test_utils::{MockFrame, MockScript, MockStreamEvent};
use rig_core::wire::{Capabilities, Mode};
use rig_core::{
    error::ErrorKind,
    message::{AssistantContent, Origin, ToolCall, ToolFunction},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

struct SelectWith<F>(F);

impl<F> AgentHook for SelectWith<F>
where
    F: for<'a> Fn(&HookContext, ModelSelection<'a>) -> ModelSelectionAction + Send + Sync,
{
    fn on_model_select(
        &self,
        context: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        (self.0)(context, event)
    }
}

fn usage(total_tokens: u64) -> Usage {
    Usage {
        total_tokens: Some(total_tokens),
        ..Usage::default()
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
    Error(String),
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

    fn error(message: &str) -> Self {
        Self::Error(message.to_owned())
    }

    fn usage(&self) -> Usage {
        match self {
            Self::Text { usage, .. } | Self::Tool { usage, .. } => *usage,
            Self::Error(_) => Usage::default(),
        }
    }

    fn message_id(&self) -> String {
        match self {
            Self::Text { message_id, .. } | Self::Tool { message_id, .. } => message_id.clone(),
            Self::Error(_) => String::new(),
        }
    }

    fn choice(&self) -> Vec<AssistantContent> {
        match self {
            Self::Text { text, .. } => vec![AssistantContent::text(text)],
            Self::Tool {
                id,
                name,
                arguments,
                ..
            } => vec![AssistantContent::ToolCall(ToolCall::from_wire(
                id.clone(),
                ToolFunction::new(
                    rig_core::message::ToolName::new(name.clone()).expect("tool name"),
                    arguments.clone(),
                ),
            ))],
            Self::Error(_) => vec![AssistantContent::text("unreachable")],
        }
    }
}

struct Script {
    provider: &'static str,
    // Never read: the Debug assertion checks that it stays out of the output.
    #[allow(dead_code)]
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

fn completion_from_script(
    script: &Script,
    request: CompletionRequest,
) -> Result<CompletionResponse, ProviderError> {
    script.record(request);
    let turn = script.next_turn();
    if let Turn::Error(message) = &turn {
        return Err(ProviderError::Provider(message.clone()));
    }
    Ok({
        let mut response = CompletionResponse::new(
            turn.choice(),
            turn.usage(),
            Origin::new("test.api", script.provider, ""),
            serde_json::json!({}),
        );
        response.origin.response_id = Some(turn.message_id());
        response
    })
}

fn stream_from_script(
    script: &Script,
    request: CompletionRequest,
) -> Result<Vec<MockStreamEvent>, ProviderError> {
    script.record(request);
    let turn = script.next_turn();
    if let Turn::Error(message) = &turn {
        return Err(ProviderError::Provider(message.clone()));
    }
    let mut events = Vec::new();
    match &turn {
        Turn::Text { text, .. } => {
            events.push(MockStreamEvent::text(text.clone()));
        }
        Turn::Tool {
            id,
            name,
            arguments,
            ..
        } => {
            // A fragmenting wire's shape: the name and the arguments as
            // fragments, closed by the call's end.
            events.push(MockStreamEvent::tool_call_name_delta(
                id.clone(),
                name.clone(),
            ));
            events.push(MockStreamEvent::tool_call_arguments_delta(
                id.clone(),
                arguments.to_string(),
            ));
            events.push(MockStreamEvent::tool_call_end(id.clone()));
        }
        // Handled by the early return above.
        Turn::Error(_) => return Err(ProviderError::Provider("unreachable".to_owned())),
    }
    events.push(MockStreamEvent::FinalResponse(Finish {
        usage: turn.usage(),
        response_id: Some(turn.message_id()),
        ..Finish::default()
    }));

    Ok(events)
}

type FakeReply = Pin<Box<dyn Future<Output = Opened<MockFrame>> + Send>>;
type FakeSend = dyn Fn(CompletionRequest, Mode) -> Result<FakeReply, ProviderError> + Send + Sync;

/// A test completion runtime: `send` answers each attempt. It is the
/// transport of a scripted completion wire.
#[derive(Clone)]
struct Fake {
    send: Arc<FakeSend>,
    /// The script a scripted model answers from.
    script: Option<Arc<Script>>,
}

impl Fake {
    fn model(
        provider: &'static str,
        send: impl Fn(CompletionRequest, Mode) -> Result<FakeReply, ProviderError>
        + Send
        + Sync
        + 'static,
    ) -> FakeModel {
        Model::new(
            wire(provider, false),
            Self {
                send: Arc::new(send),
                script: None,
            },
        )
    }
}

type FakeModel = Model<MockScript, Fake>;

/// The scripted completion wire of `provider`.
fn wire(provider: &'static str, composes_native_output_with_tools: bool) -> MockScript {
    MockScript::new(provider).with_capabilities(Capabilities::completion(
        ProviderCapabilities::new()
            .with_native_output_tool_composition(composes_native_output_with_tools),
    ))
}

/// A reply of `events`, ready at once.
fn replied(events: Vec<MockStreamEvent>) -> FakeReply {
    Box::pin(std::future::ready(Opened::new(stream::iter(
        events.into_iter().map(|event| Ok(MockFrame::Event(event))),
    ))))
}

/// A whole `response`, its `raw` the reply's document.
fn answered(response: CompletionResponse) -> Opened<MockFrame> {
    let document = response.raw.clone();
    Opened::new(stream::iter([Ok(MockFrame::Response(Box::new(response)))])).with_document(document)
}

impl Transport<MockScript> for Fake {
    fn send(&self, request: CompletionRequest, exchange: Exchange) -> Opening<MockFrame> {
        match (self.send)(request, exchange.mode) {
            Ok(reply) => Opening::new(async move { Ok(reply.await) }),
            Err(error) => Opening::failed(error),
        }
    }
}

/// The script `model` answers from.
fn script_of(model: &FakeModel) -> Arc<Script> {
    Arc::clone(model.transport.script.as_ref().expect("a scripted model"))
}

/// A model answering from `script`.
fn scripted(script: Arc<Script>) -> FakeModel {
    let composes = script.composes_native_output_with_tools;
    let kept = Arc::clone(&script);
    let mut model = Fake::model(script.provider, move |request, mode| match mode {
        Mode::Unary => completion_from_script(&script, request)
            .map(|response| Box::pin(std::future::ready(answered(response))) as FakeReply),
        Mode::Streaming => stream_from_script(&script, request).map(replied),
    });
    model.wire = wire(kept.provider, composes);
    model.transport.script = Some(kept);
    model
}

fn alpha_static(text: &str) -> FakeModel {
    scripted(Script::new(
        "alpha",
        [],
        Turn::text(text, 1, "alpha-message"),
    ))
}

fn beta_static(text: &str) -> FakeModel {
    scripted(Script::new("beta", [], Turn::text(text, 2, "beta-message")))
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct ExtractedValue {
    value: String,
}

fn assert_agent(_: Agent) {}
fn assert_builder(_: AgentBuilder<NoToolConfig>) {}
fn assert_prompt_request(_: AgentRunner) {}
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
    assert_agent_stream(alpha_agent.prompt("stream type").stream());

    let unary = alpha
        .call(CompletionRequest::new("low-level unary"))
        .await
        .expect("direct unary response");
    assert_eq!(unary.provider(), "alpha");

    let mut low_level_stream = beta
        .stream(CompletionRequest::new("low-level stream"))
        .expect("direct provider stream");
    while let Some(item) = low_level_stream.next().await {
        item.expect("stream item");
    }
    assert_eq!(
        low_level_stream
            .finish()
            .await
            .expect("the stream ends")
            .provider()
            .to_owned(),
        "beta",
        "direct model streams report their provider on the response"
    );

    let extraction_turn = Turn::Tool {
        id: "submit-call".to_owned(),
        name: rig_core::message::ToolName::new("submit")
            .expect("tool name")
            .into(),
        arguments: serde_json::json!({"value": "external model extraction"}),
        usage: usage(3),
        message_id: "extract-message".to_owned(),
    };
    let extracted = ExtractorBuilder::<ExtractedValue>::new(scripted(Script::new(
        "extractor",
        [extraction_turn.clone()],
        extraction_turn,
    )))
    .build()
    .extract("extract a value")
    .await
    .expect("custom model extraction");
    assert_eq!(extracted.output.value, "external model extraction");

    let diagnostic = AgentBuilder::named_model("diagnostic-alpha", alpha).build();
    assert_eq!(
        diagnostic.model_label(),
        Some(ModelRef::from("diagnostic-alpha")),
        "the agent's default model is addressed by its registered label"
    );
    let debug = format!(
        "{:?}",
        diagnostic
            .model_descriptor()
            .expect("registered model descriptor")
    );
    assert!(debug.contains("diagnostic-alpha"));
    assert!(!debug.contains("credential-must-not-leak"));
}

#[tokio::test]
async fn replacement_and_override_scopes_have_value_semantics() {
    let alpha = alpha_static("alpha");
    let beta = beta_static("beta");
    let mut agent = AgentBuilder::new(alpha.clone()).build();
    let runner_before_replacement = agent.prompt("runner snapshot");
    agent.set_model(beta.clone());

    assert_eq!(
        runner_before_replacement
            .run()
            .await
            .expect("old runner")
            .output(),
        "alpha"
    );
    assert_eq!(
        agent
            .prompt("new runner")
            .await
            .expect("new default")
            .output(),
        "beta"
    );

    let original = AgentBuilder::named_model("alpha", alpha.clone())
        .model_route("beta", beta.clone())
        .build();
    let changed_clone = original.clone().with_model_label("beta");
    assert_eq!(
        original
            .prompt("original")
            .await
            .expect("original")
            .output(),
        "alpha"
    );
    assert_eq!(
        changed_clone
            .prompt("clone")
            .await
            .expect("changed clone")
            .output(),
        "beta"
    );

    assert_eq!(
        original
            .prompt("one run")
            .using_model("beta")
            .await
            .expect("fixed override")
            .output(),
        "beta"
    );
    assert_eq!(
        original
            .prompt("default remains")
            .await
            .expect("default")
            .output(),
        "alpha"
    );

    let typed: ExtractedValue = original
        .prompt_typed("typed one-run override")
        .using_model_value(beta_static(r#"{"value":"typed beta"}"#))
        .await
        .expect("typed override")
        .output;
    assert_eq!(typed.value, "typed beta");

    let observed_candidates = Arc::new(Mutex::new(Vec::new()));
    let observed_for_hook = observed_candidates.clone();
    assert_eq!(
        original
            .prompt("hook after run default")
            .using_model("alpha")
            .add_hook(SelectWith(
                move |_context: &HookContext, event: ModelSelection<'_>| {
                    observed_for_hook
                        .lock()
                        .expect("candidate observations")
                        .push((
                            Some(event.default_model.to_string()),
                            Some(event.selected_model.to_string()),
                        ));
                    ModelSelectionAction::select("beta")
                }
            ))
            .await
            .expect("hook overrides run default")
            .output(),
        "beta"
    );
    assert_eq!(
        observed_candidates
            .lock()
            .expect("candidate observations")
            .as_slice(),
        &[(Some("alpha".to_owned()), Some("alpha".to_owned()))]
    );
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
    let beta_script = script_of(&beta);
    let selections = Arc::new(Mutex::new(Vec::new()));
    let selections_for_router = selections.clone();

    let response = AgentBuilder::named_model("alpha", alpha)
        .model_route("beta", beta)
        .add_hook(RetryFirst(Arc::new(AtomicUsize::new(0))))
        .build()
        .prompt("try twice")
        .max_turns(2)
        .add_hook(SelectWith(
            move |context: &HookContext, event: ModelSelection<'_>| {
                selections_for_router.lock().expect("selection lock").push((
                    context.turn(),
                    event
                        .previous_model
                        .map(ModelRef::as_str)
                        .map(str::to_owned),
                ));
                ModelSelectionAction::select(if context.turn() == 1 { "alpha" } else { "beta" })
            },
        ))
        .await
        .expect("retry routed run");

    assert_eq!(response.output(), "accepted answer");
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
            Message::Assistant(rig_core::message::AssistantMessage { content, .. })
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

// ---------------------------------------------------------------------------
// Ordering parity tests: completion-call hooks -> merged RequestPatch ->
// ModelSelection -> preparation -> issue attempt; previous_model reflects
// issued attempts only. Each scenario runs on both surfaces.
// ---------------------------------------------------------------------------

/// Records what every selection event observed: (turn, previous_model label,
/// merged-patch temperature, merged-patch preamble).
type SelectionObservations = Arc<Mutex<Vec<(usize, Option<String>, Option<f64>, Option<String>)>>>;

fn observing_selector(
    observations: SelectionObservations,
) -> SelectWith<impl for<'a> Fn(&HookContext, ModelSelection<'a>) -> ModelSelectionAction> {
    SelectWith(move |context: &HookContext, event: ModelSelection<'_>| {
        observations.lock().expect("observation lock").push((
            context.turn(),
            event
                .previous_model
                .map(ModelRef::as_str)
                .map(str::to_owned),
            event.request_patch.and_then(|patch| patch.temperature),
            event.request_patch.and_then(|patch| patch.preamble.clone()),
        ));
        ModelSelectionAction::continue_run()
    })
}

/// Drive a streaming run to its terminal item, returning the first error if
/// the stream yields one.
async fn drain_stream(mut stream: StreamingResult) -> Result<(), PromptError> {
    while let Some(item) = stream.next().await {
        item?;
    }
    Ok(())
}

#[tokio::test]
async fn failed_preparation_follows_selection_and_does_not_issue_an_attempt() {
    for streaming in [false, true] {
        // Turn 1 issues a tool-call attempt on alpha; turn 2's completion-call
        // patch names a tool that does not exist, so preparation fails after
        // model selection resolves.
        let alpha_turn = Turn::tool("lookup", 3, "alpha-tool-message");
        let model = scripted(Script::new("alpha", [alpha_turn.clone()], alpha_turn));
        let script = script_of(&model);
        let observations: SelectionObservations = Arc::new(Mutex::new(Vec::new()));
        let bad_patch = BadSecondTurnPatch;
        let agent = AgentBuilder::named_model("alpha", model.clone())
            .tool(LookupTool {
                calls: Arc::new(AtomicUsize::new(0)),
            })
            .add_hook(bad_patch)
            .add_hook(observing_selector(observations.clone()))
            .build();

        let failed = if streaming {
            drain_stream(agent.prompt("prepare fails").max_turns(2).stream())
                .await
                .is_err()
        } else {
            agent.prompt("prepare fails").max_turns(2).await.is_err()
        };
        assert!(failed, "streaming={streaming}: preparation must fail");

        // Selection ran on both turns; only turn 1's attempt was issued, so
        // turn 2 observes previous_model == alpha, and the failed preparation
        // never reached the provider.
        let observed = observations.lock().expect("observation lock").clone();
        assert_eq!(observed.len(), 2, "streaming={streaming}");
        assert_eq!(observed[0].0, 1);
        assert_eq!(observed[0].1, None);
        assert_eq!(observed[1].0, 2);
        assert_eq!(observed[1].1, Some("alpha".to_owned()));
        assert_eq!(
            script.requests().len(),
            1,
            "streaming={streaming}: the failed turn must not reach the provider"
        );
    }
}

/// Patches turn 2 with an `active_tools` allow-list naming a missing tool, so
/// request preparation fails locally on that turn.
#[derive(Clone)]
struct BadSecondTurnPatch;

impl AgentHook for BadSecondTurnPatch {
    async fn on_completion_call(
        &self,
        context: &HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if context.turn() == 2 {
            CompletionCallAction::patch(
                RequestPatch::new().active_tools(["no_such_tool".to_owned()]),
            )
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

#[tokio::test]
async fn an_errored_provider_attempt_still_counts_as_the_previous_model() {
    for streaming in [false, true] {
        // Turn 1's provider attempt errors after being issued; the invalid
        // reply is not needed — instead, the recovery path that keeps the run
        // alive is a fresh extraction retry driven by the caller. Within a
        // single run the driver terminates on a provider error, so the
        // issued-attempt semantics are observed through the invalid-tool-call
        // retry: alpha's turn-1 attempt is issued and defective, and turn 2's
        // selection still sees previous_model == alpha. The direct
        // provider-error path is asserted below to issue exactly one request
        // and fail with the provider error (not a cancellation), proving the
        // attempt was issued after selection resolved.
        let flaky = scripted(Script::new(
            "flaky",
            [Turn::error("provider exploded")],
            Turn::text("unreachable", 1, "unreachable-message"),
        ));
        let script = script_of(&flaky);
        let observations: SelectionObservations = Arc::new(Mutex::new(Vec::new()));
        let agent = AgentBuilder::named_model("flaky", flaky)
            .add_hook(observing_selector(observations.clone()))
            .build();

        // The bus carries a provider failure across the hop as a
        // provider-kind `ErrorReport` whose message is the provider's own.
        let failed_with_provider_error = if streaming {
            matches!(
                drain_stream(agent.prompt("boom").stream()).await,
                Err(PromptError::Report(report))
                    if report.kind == ErrorKind::Provider
                        && report.message.ends_with("provider exploded")
            )
        } else {
            matches!(
                agent.prompt("boom").await,
                Err(PromptError::Report(report))
                    if report.kind == ErrorKind::Provider
                        && report.message.ends_with("provider exploded")
            )
        };
        assert!(
            failed_with_provider_error,
            "streaming={streaming}: the issued attempt's provider error must surface"
        );
        // The attempt WAS issued: selection resolved, preparation succeeded,
        // and the provider received exactly one request before erroring.
        assert_eq!(observations.lock().expect("observation lock").len(), 1);
        assert_eq!(script.requests().len(), 1);
    }

    // The advancement itself (an issued-but-failed attempt counts) is
    // observable when the run continues: alpha's turn-1 attempt returns an
    // invalid tool call (issued and defective), and turn 2's selection sees
    // previous_model == "alpha" even though nothing from that attempt was
    // committed.
    let invalid = Turn::tool("missing_tool", 2, "invalid-message");
    let alpha = scripted(Script::new("alpha", [invalid.clone()], invalid));
    let observations: SelectionObservations = Arc::new(Mutex::new(Vec::new()));
    let observations_for_router = observations.clone();
    let output = AgentBuilder::named_model("alpha", alpha)
        .model_route("beta", beta_static("recovered"))
        .add_hook(RetryInvalidTool)
        .build()
        .prompt("recover")
        .max_turns(2)
        .max_invalid_tool_call_retries(1)
        .add_hook(SelectWith(
            move |context: &HookContext, event: ModelSelection<'_>| {
                observations_for_router
                    .lock()
                    .expect("observation lock")
                    .push((
                        context.turn(),
                        event
                            .previous_model
                            .map(ModelRef::as_str)
                            .map(str::to_owned),
                        None,
                        None,
                    ));
                ModelSelectionAction::select(if context.turn() == 1 { "alpha" } else { "beta" })
            },
        ))
        .await
        .expect("recovered run");
    assert_eq!(output.output(), "recovered");
    let observed = observations.lock().expect("observation lock").clone();
    assert_eq!(observed[0], (1, None, None, None));
    assert_eq!(observed[1], (2, Some("alpha".to_owned()), None, None));
}
