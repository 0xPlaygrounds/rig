use crate::agent::{
    DispatchAction, DispatchEvent, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction,
    ModelTurnFinished, ObservationAction, OutcomeAction, OutcomeEvent, ReasoningDelta, TextDelta,
    ToolCallDelta,
};

use super::*;
use crate::agent::AgentBuilder;
use crate::agent::engine::drive_tool_calls;
use crate::agent::hook::{AgentHook, HookContext};
use crate::agent::run::{AgentRun, AgentRunStep};
use crate::completion::{CompletionRequest, FinishReason, PromptError, ToolDefinition, Usage};
use crate::run::transcript::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER;
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::{
    CapturedSpan, FailingMemory, MockAddTool, MockCompletionModel, MockContextProbeTool,
    MockStreamEvent, MockSubtractTool, MockToolError, SessionId, TraceCapture, mock_final,
};
use crate::tool::{Tool, ToolContext};
use futures::{StreamExt, TryStreamExt};
use rig_core::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig_core::operation::Finish;
use rig_core::providers::anthropic;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn stream_to_stdout_returns_the_final_response() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("go").stream();
    let response = stream_to_stdout(&mut stream)
        .await
        .expect("the run succeeds");

    assert_eq!(response.output(), "done");
    assert_eq!(response.messages().len(), 2);
}

#[tokio::test]
async fn text_only_stream_without_terminal_record_is_rejected_as_truncated() {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::text("partial answer")]]);
    let agent = Arc::new(AgentBuilder::new(model.clone()).build());

    let mut stream = agent.prompt("go").stream();
    let mut saw_error = false;
    let mut saw_completion_call = false;
    while let Some(item) = stream.next().await {
        match item {
            Err(error) => {
                assert!(
                    error
                        .to_string()
                        .contains("the reply ended before the provider ended it"),
                    "truncation should surface as a truncated reply, got: {error}"
                );
                saw_error = true;
                break;
            }
            Ok(MultiTurnStreamItem::CompletionCall(_)) => saw_completion_call = true,
            Ok(_) => {}
        }
    }
    assert!(
        saw_error,
        "a stream ending without a terminal record must be rejected, not \
             treated as a successful completion"
    );
    // The rejection happens before any usage fallback records the call, so
    // the runner never produces a `CompletionCall` whose `raw` is `Null`:
    // a `Null` payload can only come from a hand-driven `AgentRun`.
    assert!(
        !saw_completion_call,
        "no completion call may be recorded for a truncated stream"
    );
}

fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
    history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant(rig_core::message::AssistantMessage { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                ))
        )
    })
}

/// The invalid-call retry transcript pairs 1:1 by construction: every tool
/// call in the assistant turn carries a unique non-empty id (minted at the
/// provider boundary when the wire issued none), and the retry results
/// answer exactly those ids.
fn assert_retry_transcript_ids_pair(assistant: &Message, results: &Message) {
    let Message::Assistant(rig_core::message::AssistantMessage { content, .. }) = assistant else {
        panic!("expected the assistant tool-call turn, got {assistant:?}");
    };
    let call_ids: Vec<&rig_core::message::CallId> = content
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(&tool_call.id),
            _ => None,
        })
        .collect();
    let Message::User { content } = results else {
        panic!("expected the user retry-result turn, got {results:?}");
    };
    let result_ids: Vec<&rig_core::message::CallId> = content
        .iter()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(&result.call),
            _ => None,
        })
        .collect();
    let unique_calls: BTreeSet<&rig_core::message::CallId> = call_ids.iter().copied().collect();
    assert_eq!(
        unique_calls.len(),
        call_ids.len(),
        "tool-call ids must be unique: {call_ids:?}"
    );
    let unique_results: BTreeSet<&rig_core::message::CallId> = result_ids.iter().copied().collect();
    assert_eq!(
        unique_results.len(),
        result_ids.len(),
        "retry-result ids must be unique: {result_ids:?}"
    );
    assert_eq!(
        unique_calls, unique_results,
        "retry results must answer exactly the turn's tool calls"
    );
}

fn history_contains_text(history: &[Message], expected: &str) -> bool {
    history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant(rig_core::message::AssistantMessage { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == expected
                ))
        )
    })
}

fn assistant_reasoning_precedes_tool_call(
    history: &[Message],
    expected_reasoning: &str,
    tool_name: &str,
) -> bool {
    history.iter().any(|message| {
        let Message::Assistant(rig_core::message::AssistantMessage { content, .. }) = message
        else {
            return false;
        };

        // A replay to another model carries reasoning as text.
        let reasoning_index = content.iter().position(|item| match item {
            AssistantContent::Reasoning(reasoning) => reasoning.text == expected_reasoning,
            AssistantContent::Text(text) => text.text == expected_reasoning,
            _ => false,
        });
        let tool_index = content.iter().position(|item| {
            matches!(
                item,
                AssistantContent::ToolCall(tool_call)
                    if tool_call.function.name == tool_name
            )
        });

        matches!((reasoning_index, tool_index), (Some(reasoning), Some(tool)) if reasoning < tool)
    })
}

fn assistant_reasoning_precedes_text_and_tool_call(
    history: &[Message],
    expected_reasoning: &str,
    expected_text: &str,
    tool_name: &str,
) -> bool {
    history.iter().any(|message| {
        let Message::Assistant(rig_core::message::AssistantMessage { content, .. }) = message
        else {
            return false;
        };

        // A replay to another model carries reasoning as text.
        let reasoning_index = content.iter().position(|item| match item {
            AssistantContent::Reasoning(reasoning) => reasoning.text == expected_reasoning,
            AssistantContent::Text(text) => text.text == expected_reasoning,
            _ => false,
        });
        let text_index = content.iter().position(|item| {
            matches!(
                item,
                AssistantContent::Text(text) if text.text == expected_text
            )
        });
        let tool_index = content.iter().position(|item| {
            matches!(
                item,
                AssistantContent::ToolCall(tool_call)
                    if tool_call.function.name == tool_name
            )
        });

        // The parts keep the order they started in; both precede the call.
        matches!(
            (reasoning_index, text_index, tool_index),
            (Some(reasoning), Some(text), Some(tool))
                if reasoning < tool && text < tool
        )
    })
}

#[derive(Clone)]
struct PanicOnUnknownToolHook;

impl AgentHook for PanicOnUnknownToolHook {
    /// A call to a known tool streams its arguments; an unknown one's are
    /// held for its end and never reach the hook.
    async fn on_tool_call_delta(
        &self,
        _: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        if !["add", "subtract"].contains(&event.tool_name) {
            panic!("unknown tool call delta should fail before delta hooks run")
        }
        ObservationAction::continue_run()
    }
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_some() {
            panic!("unknown tool call should fail before tool hooks run")
        }
        DispatchAction::proceed()
    }
    async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.completion().is_some() {
            panic!("unknown tool call should fail before completion outcome hooks run")
        }
        OutcomeAction::proceed()
    }
}

#[derive(Clone)]
struct CountingAddTool {
    calls: Arc<AtomicU32>,
}

#[derive(Clone)]
struct CountingSubtractTool {
    calls: Arc<AtomicU32>,
}

#[derive(Deserialize)]
struct CountingOperationArgs {
    x: i32,
    y: i32,
}

fn arithmetic_tool_definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new(name).expect("tool name"),
        description: description.to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first operand"
                },
                "y": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["x", "y"],
        }),
    }
}

impl Tool for CountingAddTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = CountingOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        arithmetic_tool_definition(Self::NAME, "Add x and y together").parameters
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

impl Tool for CountingSubtractTool {
    const NAME: &'static str = "subtract";
    type Error = MockToolError;
    type Args = CountingOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        arithmetic_tool_definition(Self::NAME, "Subtract y from x").parameters
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}

fn streaming_tool_then_text_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tool_call_1", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ])
}

/// The record a streamed mock turn scripted with
/// `MockStreamEvent::final_response(usage)` leaves on `completion_calls`:
/// index, usage, and the mock's terminal record serialized onto `raw` —
/// the terminal is always captured, so an expected call without it never
/// matches.
fn streamed_call(call_index: usize, usage: Usage) -> CompletionCall {
    let terminal = mock_final(usage);
    CompletionCall::new(
        call_index,
        usage,
        serde_json::to_value(&terminal).expect("mock terminal serializes"),
    )
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        total_tokens: Some(input_tokens + output_tokens),
        ..Usage::default()
    }
}

#[tokio::test]
async fn execution_commit_items_are_not_emitted_when_run_commit_fails() {
    let runner = AgentBuilder::new(MockCompletionModel::from_turns([]))
        .build()
        .prompt("go");
    let tool_snapshot = Arc::new(
        runner
            .tool_server_handle
            .snapshot_tool_defs(None)
            .await
            .expect("empty tool snapshot should build"),
    );

    let mut run = AgentRun::new("go").max_turns(2);
    assert!(matches!(
        run.next_step().expect("initial model step"),
        AgentRunStep::CallModel { .. }
    ));

    let tool_name = "missing".to_string();
    let advertised = BTreeSet::from([tool_name.clone()]);
    let turn = crate::agent::run::ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![AssistantContent::ToolCall(
            rig_core::message::ToolCall::new(
                rig_core::message::CallId::from_wire("expected_call"),
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new(tool_name).expect("tool name"),
                    serde_json::json!({}),
                ),
            ),
        )],
        Usage::default(),
        advertised.clone(),
        advertised,
        serde_json::json!({"origin": "hand-built test turn"}),
    );
    assert!(matches!(
        run.model_response(turn)
            .expect("tool turn should be accepted"),
        crate::agent::run::ModelTurnOutcome::Continue { .. }
    ));

    let mut calls = match run.next_step().expect("tool step") {
        AgentRunStep::CallTools { calls } => calls,
        other => panic!("expected tool step, got {other:?}"),
    };
    // Corrupt only the driver's copy so execution settles successfully but
    // `AgentRun` rejects the result before any commit-labelled item escapes.
    calls[0].tool_call.id = rig_core::message::CallId::from_wire("mismatched_call");

    let hook_context = HookContext::new(true, None, None);
    hook_context.set_turn(1);
    let mut stream = drive_tool_calls(
        &runner,
        &hook_context,
        &mut run,
        calls,
        tool_snapshot,
        |span| span,
        true,
    );

    let mut saw_commit = false;
    let mut saw_result = false;
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => saw_commit = true,
            Ok(MultiTurnStreamItem::ToolResult { .. }) => saw_result = true,
            Err(_) => saw_error = true,
            _ => {}
        }
    }

    assert!(
        saw_error,
        "the mismatched result must fail run-state commit"
    );
    assert!(!saw_commit, "a failed run-state commit cannot be announced");
    assert!(!saw_result, "an uncommitted result cannot be surfaced");
}

async fn assert_stream_usage_recorded_on_chat_spans(
    agent: crate::agent::Agent,
    prompt: &str,
    max_turns: usize,
    expected_usages: &[Usage],
) {
    // Scoped-subscriber tests must not run concurrently; the warm-up below
    // explains the callsite-interest hazard this guards against. The
    // guard's own docs carry that recipe plus the rule it cannot enforce:
    // an absence assertion needs a positive anchor, or it passes vacuously.
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let spans = TraceCapture::default();
    let _default = tracing::subscriber::set_default(spans.subscriber());

    // Span callsites in the driver are shared with every other test in
    // this binary. The FIRST thread to hit a callsite caches its interest
    // from that thread's dispatcher (`Dispatchers::Rebuilder::JustOne`
    // consults `dispatcher::get_default`), so a parallel test without a
    // subscriber can permanently cache `Interest::never` for the very
    // spans this harness asserts on. Defend in two steps, both under the
    // isolation guard: (1) warm the whole driver path from THIS thread so
    // unregistered callsites first-register against this subscriber, then
    // (2) rebuild the interest cache to heal callsites a foreign thread
    // already poisoned.
    let warmup_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("warmup"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let warmup_agent = crate::agent::AgentBuilder::new(warmup_model).build();
    let mut warmup_stream = warmup_agent.prompt("warmup").max_turns(1).stream();
    while let Some(item) = warmup_stream
        .try_next()
        .await
        .expect("warmup stream should not error")
    {
        if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
            break;
        }
    }
    tracing::callsite::rebuild_interest_cache();
    spans.clear();

    let empty_history: &[Message] = &[];
    // Declare the fields the guard protects so a regression (recording onto
    // a caller span) is actually observable, not silently a no-op.
    let outer_span = tracing::info_span!("outer", gen_ai.completion = tracing::field::Empty);

    async {
        let mut stream = agent
            .prompt(prompt)
            .history(empty_history)
            .max_turns(max_turns)
            .stream();

        while let Some(item) = stream.try_next().await.expect("stream should not error") {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }
    }
    .instrument(outer_span)
    .await;

    let span_snapshot = spans.spans();
    let outer_span_id = span_snapshot
        .iter()
        .find(|span| span.name == "outer")
        .map(|span| span.id)
        .expect("outer span should be captured");
    let chat_spans = span_snapshot
        .iter()
        .filter(|span| span.target == "rig::agent_chat")
        .collect::<Vec<_>>();

    assert_eq!(chat_spans.len(), expected_usages.len());
    assert!(
        span_snapshot.iter().all(|span| span.name != "invoke_agent"),
        "outer span path should not create invoke_agent"
    );

    for (chat_span, expected_usage) in chat_spans.into_iter().zip(expected_usages) {
        assert_eq!(chat_span.parent, Some(outer_span_id));
        // The provider's streaming span adopts the agent's chat span and
        // records its own operation onto it.
        assert_eq!(
            chat_span.text("gen_ai.operation.name").as_deref(),
            Some("chat")
        );
        assert_eq!(chat_span.name, "chat");
        assert_eq!(
            chat_span.value("gen_ai.request.stream"),
            Some(&serde_json::json!(true))
        );
        // A counter the provider did not report leaves its span field unset.
        let field = |name: &str| chat_span.u64(name);
        assert_eq!(
            field("gen_ai.usage.input_tokens"),
            expected_usage.input_tokens
        );
        assert_eq!(
            field("gen_ai.usage.output_tokens"),
            expected_usage.output_tokens
        );
        assert_eq!(
            field("gen_ai.usage.cache_read.input_tokens"),
            expected_usage.cached_input_tokens
        );
        assert_eq!(
            field("gen_ai.usage.cache_creation.input_tokens"),
            expected_usage.cache_creation_input_tokens
        );
        assert_eq!(
            field("gen_ai.usage.tool_use_prompt_tokens"),
            expected_usage.tool_use_prompt_tokens
        );
        assert_eq!(
            field("gen_ai.usage.reasoning_tokens"),
            expected_usage.reasoning_tokens
        );
    }

    let outer_span = span_snapshot
        .iter()
        .find(|span| span.id == outer_span_id)
        .expect("outer span should be present");
    assert!(
        outer_span
            .recorded
            .iter()
            .all(|(field, _)| !field.starts_with("gen_ai.usage.")),
        "usage should not be recorded onto the caller's outer span"
    );
    assert!(
        outer_span.record_count("gen_ai.completion") == 0,
        "gen_ai.completion should not be recorded onto the caller's outer span \
             (parity with the blocking driver)"
    );
}

async fn capture_stream_message_telemetry(
    record_telemetry_content: bool,
) -> (CapturedSpan, Vec<CompletionRequest>) {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let spans = TraceCapture::default();
    let _default = tracing::subscriber::set_default(spans.subscriber());

    let warmup_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("warmup"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let warmup_agent = crate::agent::AgentBuilder::new(warmup_model).build();
    let mut warmup_stream = warmup_agent.prompt("warmup").max_turns(1).stream();
    while let Some(item) = warmup_stream
        .try_next()
        .await
        .expect("warmup stream should not error")
    {
        if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
            break;
        }
    }
    tracing::callsite::rebuild_interest_cache();
    spans.clear();

    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("stream response secret"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let recorded_model = model.clone();
    let builder = AgentBuilder::new(model);
    let agent = if record_telemetry_content {
        builder
            .record_content_telemetry(true)
            .context("static stream context secret")
            .build()
    } else {
        builder.context("static stream context secret").build()
    };

    let mut stream = agent.prompt("stream prompt secret").max_turns(1).stream();
    while let Some(item) = stream.try_next().await.expect("stream should not error") {
        if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
            break;
        }
    }

    let span = spans
        .spans()
        .into_iter()
        .find(|span| span.target == "rig::agent_chat")
        .expect("the agent chat span should be captured");
    (span, recorded_model.requests())
}

async fn capture_unary_message_telemetry(
    record_telemetry_content: bool,
) -> (CapturedSpan, CapturedSpan, Vec<CompletionRequest>) {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let spans = TraceCapture::default();
    let _default = tracing::subscriber::set_default(spans.subscriber());

    let warmup_agent = crate::agent::AgentBuilder::new(MockCompletionModel::text("warmup")).build();
    warmup_agent
        .prompt("warmup")
        .await
        .expect("warmup prompt should not error");
    tracing::callsite::rebuild_interest_cache();
    spans.clear();

    let model = MockCompletionModel::text("blocking response secret");
    let recorded_model = model.clone();
    let builder = AgentBuilder::new(model).preamble("blocking system secret");
    let agent = if record_telemetry_content {
        builder.record_content_telemetry(true).build()
    } else {
        builder.build()
    };

    agent
        .prompt("blocking prompt secret")
        .await
        .expect("prompt should not error");

    let snapshot = spans.spans();
    let chat_span = snapshot
        .iter()
        .find(|span| span.name == "chat")
        .cloned()
        .expect("chat span should be captured");
    let agent_span = snapshot
        .into_iter()
        .find(|span| span.name == "invoke_agent")
        .expect("invoke_agent span should be captured");
    (chat_span, agent_span, recorded_model.requests())
}

#[tokio::test]
async fn stream_prompt_message_telemetry_is_opt_in() {
    let (default_span, default_requests) = capture_stream_message_telemetry(false).await;
    assert!(
        default_span.record_count("gen_ai.input.messages") == 0,
        "default streaming prompt should not record input message contents"
    );
    assert!(
        default_span.record_count("gen_ai.output.messages") == 0,
        "default streaming prompt should not record output message contents"
    );

    assert_eq!(default_requests.len(), 1);
    assert!(
        !default_requests[0].record_telemetry_content,
        "default agent stream should keep provider request message telemetry disabled"
    );

    let (opt_in_span, opt_in_requests) = capture_stream_message_telemetry(true).await;
    let input = opt_in_span
        .text("gen_ai.input.messages")
        .expect("opt-in should record input messages");
    assert!(input.contains("stream prompt secret"));
    assert!(input.contains("static stream context secret"));
    let output = opt_in_span
        .text("gen_ai.output.messages")
        .expect("opt-in should record output messages");
    assert!(output.contains("stream response secret"));
    assert_eq!(
        opt_in_span.record_count("gen_ai.input.messages"),
        1,
        "agent-owned input message telemetry should be recorded once"
    );
    assert_eq!(
        opt_in_span.record_count("gen_ai.output.messages"),
        1,
        "agent-owned output message telemetry should be recorded once"
    );
    assert_eq!(opt_in_requests.len(), 1);
    assert!(
        !opt_in_requests[0].record_telemetry_content,
        "agent-owned stream telemetry should clear the provider request flag"
    );
}

#[tokio::test]
async fn unary_prompt_message_telemetry_records_accepted_output_when_opted_in() {
    let (default_span, default_agent_span, default_requests) =
        capture_unary_message_telemetry(false).await;
    assert!(
        default_span.record_count("gen_ai.input.messages") == 0,
        "default blocking prompt should not record input message contents"
    );
    assert!(
        default_span.record_count("gen_ai.output.messages") == 0,
        "default blocking prompt should not record output message contents"
    );
    assert!(
        default_span.value("gen_ai.system_instructions").is_none(),
        "default blocking prompt should not record system instructions"
    );
    assert!(default_agent_span.value("gen_ai.prompt").is_none());
    assert!(default_agent_span.value("gen_ai.completion").is_none());
    assert_eq!(default_requests.len(), 1);
    assert!(
        !default_requests[0].record_telemetry_content,
        "default blocking prompt should keep provider request message telemetry disabled"
    );

    let (opt_in_span, opt_in_agent_span, opt_in_requests) =
        capture_unary_message_telemetry(true).await;
    let input = opt_in_span
        .text("gen_ai.input.messages")
        .expect("opt-in should record blocking input messages");
    assert!(input.contains("blocking prompt secret"));
    let output = opt_in_span
        .text("gen_ai.output.messages")
        .expect("opt-in should record blocking output messages");
    assert!(output.contains("blocking response secret"));
    assert_eq!(
        opt_in_span.text("gen_ai.system_instructions").as_deref(),
        Some(r#"[{"type":"text","content":"blocking system secret"}]"#)
    );
    assert_eq!(
        opt_in_agent_span.text("gen_ai.prompt").as_deref(),
        Some("blocking prompt secret")
    );
    assert_eq!(
        opt_in_agent_span.text("gen_ai.completion").as_deref(),
        Some("blocking response secret")
    );
    assert_eq!(opt_in_requests.len(), 1);
    assert!(
        !opt_in_requests[0].record_telemetry_content,
        "agent-owned blocking telemetry should clear the provider request flag"
    );
}

#[tokio::test]
async fn streaming_rejected_message_telemetry_does_not_record_output() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let spans = TraceCapture::default();
    let _default = tracing::subscriber::set_default(spans.subscriber());

    let warmup_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("warmup"),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let warmup_agent = crate::agent::AgentBuilder::new(warmup_model).build();
    let mut warmup_stream = warmup_agent.prompt("warmup").max_turns(1).stream();
    while let Some(item) = warmup_stream
        .try_next()
        .await
        .expect("warmup stream should not error")
    {
        if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
            break;
        }
    }
    tracing::callsite::rebuild_interest_cache();
    spans.clear();

    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("rejected stream output secret"),
        MockStreamEvent::tool_call(
            "tool_call_1",
            "default_api",
            serde_json::json!({"x": 2, "y": 3}),
        ),
        MockStreamEvent::final_response(Usage::default()),
    ]]);
    let agent = AgentBuilder::new(model)
        .record_content_telemetry(true)
        .build();

    let mut stream = agent
        .prompt("stream rejection prompt")
        .max_turns(1)
        .stream();
    let err = loop {
        match stream.try_next().await {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("rejected stream should error"),
            Err(err) => break err,
        }
    };
    assert!(
        err.to_string().contains("default_api"),
        "expected invalid tool error, got {err}"
    );

    let chat_span = spans
        .spans()
        .into_iter()
        .find(|span| span.target == "rig::agent_chat")
        .expect("the agent chat span should be captured");
    assert!(
        chat_span.record_count("gen_ai.input.messages") > 0,
        "opt-in rejected stream should still record input messages"
    );
    assert!(
        chat_span.record_count("gen_ai.output.messages") == 0,
        "rejected streaming turn must not record output message contents"
    );
}

#[test]
fn completion_calls_stream_item_serializes_and_deserializes_expected_shape() {
    let item: MultiTurnStreamItem = MultiTurnStreamItem::CompletionCall(CompletionCall::new(
        2,
        usage(3, 4),
        serde_json::json!({"id": "resp_2"}),
    ));

    let value = serde_json::to_value(&item).expect("serialize completion call event");

    assert_eq!(
        value,
        serde_json::json!({
            "type": "completionCall",
            "call_index": 2,
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
            "raw": {"id": "resp_2"}
        })
    );

    let item: MultiTurnStreamItem =
        serde_json::from_value(value).expect("deserialize completion call event");
    match item {
        MultiTurnStreamItem::CompletionCall(call) => assert_eq!(
            call,
            CompletionCall::new(2, usage(3, 4), serde_json::json!({"id": "resp_2"}))
        ),
        other => panic!("expected completion call event, got {other:?}"),
    }

    let item: MultiTurnStreamItem = MultiTurnStreamItem::CompletionCall(CompletionCall::new(
        3,
        Usage::default(),
        serde_json::json!({"id": "resp_3"}),
    ));
    let value = serde_json::to_value(&item).expect("serialize missing usage event");

    // Unreported usage serializes as an empty object: every counter is
    // `None`, and `None` counters are omitted rather than encoded as zero.
    assert_eq!(
        value,
        serde_json::json!({
            "type": "completionCall",
            "call_index": 3,
            "usage": {},
            "raw": {"id": "resp_3"}
        })
    );
}

#[test]
fn final_response_serializes_completion_calls_with_missing_usage() {
    let item: MultiTurnStreamItem = MultiTurnStreamItem::final_response_with_completion_calls(
        vec![AssistantContent::text("done")],
        usage(3, 4),
        vec![
            CompletionCall::new(0, Usage::default(), serde_json::json!({"id": "resp_0"})),
            CompletionCall::new(1, usage(3, 4), serde_json::json!({"id": "resp_1"})),
        ],
        Vec::new(),
    );

    if let MultiTurnStreamItem::FinalResponse(response) = &item {
        assert_eq!(response.requests(), 2);
    }

    let value = serde_json::to_value(&item).expect("serialize final response");

    assert_eq!(
        value.get("completion_calls"),
        Some(&serde_json::json!([
            {
                "call_index": 0,
                "usage": {},
                "raw": {"id": "resp_0"}
            },
            {
                "call_index": 1,
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                },
                "raw": {"id": "resp_1"}
            }
        ]))
    );
}

/// A run relayed as newline-delimited JSON reads back line by line: every
/// kind of item, each to the value it was written as, and a streamed call's
/// fragments, joined, read as the arguments the call ended with.
#[tokio::test]
async fn a_streamed_run_reads_back_item_by_item() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::reasoning_delta("rejected"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
        vec![
            MockStreamEvent::reasoning_delta("accepted"),
            MockStreamEvent::tool_call_name_delta("tool_call_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", r#"{"x": 2"#),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", r#", "y": 3}"#),
            MockStreamEvent::tool_call_end("tool_call_1"),
            MockStreamEvent::final_response_with_total_tokens(2),
        ],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();
    let items: Vec<MultiTurnStreamItem> = agent
        .prompt("add 2 and 3")
        .add_hook(RetryFirstReasoningTurnHook::default())
        .max_turns(3)
        .stream()
        .try_collect()
        .await
        .expect("the run succeeds");

    let mut ndjson = String::new();
    for item in &items {
        ndjson.push_str(&serde_json::to_string(item).expect("an item serializes"));
        ndjson.push('\n');
    }

    let mut kinds = BTreeSet::new();
    let mut fragments = String::new();
    let mut ended = None;
    for (line, item) in ndjson.lines().zip(&items) {
        let back: MultiTurnStreamItem = serde_json::from_str(line).expect("an item reads back");
        assert_eq!(
            serde_json::to_value(&back).expect("serializes"),
            serde_json::to_value(item).expect("serializes"),
        );
        let value: serde_json::Value = serde_json::from_str(line).expect("a JSON line");
        kinds.insert(value["type"].as_str().expect("a tag").to_owned());
        match back {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                json,
                ..
            })) => fragments.push_str(&json),
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            })) => ended = Some(call.function.arguments_value()),
            _ => {}
        }
    }

    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        [
            "completionCall",
            "finalResponse",
            "modelTurnRetried",
            "streamAssistantItem",
            "toolCall",
            "toolExecutionCommitted",
            "toolResult",
        ]
    );
    assert_eq!(fragments, r#"{"x": 2, "y": 3}"#);
    assert_eq!(
        Some(serde_json::Value::Object(
            rig_core::streaming::parse_partial_arguments(&fragments)
        )),
        ended
    );
}

fn streaming_text_then_final_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("hello"),
        MockStreamEvent::text(" world"),
        MockStreamEvent::final_response_with_total_tokens(3),
    ]])
}

fn streaming_final_only_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([[MockStreamEvent::final_response_with_total_tokens(1)]])
}

/// A recorded tool-call delta: its part's position, the tool it names, and the fragment.
type RecordedToolCallDelta = (usize, String, String);
type RecordedReasoningDelta = (usize, String, String);

#[derive(Clone)]
struct RepairDefaultApiHook;

impl AgentHook for RepairDefaultApiHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(match event {
            context => {
                assert_eq!(context.tool_name, "default_api");
                InvalidToolCallAction::repair("add")
            }
            _ => InvalidToolCallAction::fail(),
        })
    }
}

#[derive(Clone)]
struct RetryDefaultApiHook;

impl AgentHook for RetryDefaultApiHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(match event {
            context => {
                assert_eq!(context.tool_name, "default_api");
                if let Some(args) = context.args.as_deref() {
                    assert!(!args.is_empty());
                }
                InvalidToolCallAction::retry("Use the add tool instead")
            }
            _ => InvalidToolCallAction::fail(),
        })
    }
}

#[derive(Clone)]
struct SkipDefaultApiHook;

impl AgentHook for SkipDefaultApiHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(match event {
            context => {
                assert_eq!(context.tool_name, "default_api");
                InvalidToolCallAction::skip("default_api was skipped")
            }
            _ => InvalidToolCallAction::fail(),
        })
    }
}

#[derive(Clone, Default)]
struct RecordingToolCallDeltaHook {
    deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
}

impl RecordingToolCallDeltaHook {
    fn observed(&self) -> Vec<RecordedToolCallDelta> {
        self.deltas
            .lock()
            .expect("tool call delta hook records mutex was poisoned")
            .clone()
    }
}

impl AgentHook for RecordingToolCallDeltaHook {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        let record = (
            event.part.index(),
            event.tool_name.to_string(),
            event.delta.to_string(),
        );
        self.deltas
            .lock()
            .expect("tool call delta hook records mutex was poisoned")
            .push(record);
        ObservationAction::continue_run()
    }
}

#[derive(Clone, Default)]
struct RecordingTextDeltaHook {
    deltas: Arc<Mutex<Vec<(String, String)>>>,
}

impl RecordingTextDeltaHook {
    fn observed(&self) -> Vec<(String, String)> {
        self.deltas
            .lock()
            .expect("text delta hook records mutex was poisoned")
            .clone()
    }
}

impl AgentHook for RecordingTextDeltaHook {
    async fn on_text_delta(&self, _ctx: &HookContext, event: TextDelta<'_>) -> ObservationAction {
        match event {
            TextDelta { delta, aggregated } => {
                let record = (delta.to_string(), aggregated.to_string());
                self.deltas
                    .lock()
                    .expect("text delta hook records mutex was poisoned")
                    .push(record);
                ObservationAction::continue_run()
            }
            _ => ObservationAction::continue_run(),
        }
    }
}

#[derive(Clone, Default)]
struct RecordingReasoningDeltaHook {
    deltas: Arc<Mutex<Vec<RecordedReasoningDelta>>>,
}

impl RecordingReasoningDeltaHook {
    fn observed(&self) -> Vec<RecordedReasoningDelta> {
        self.deltas
            .lock()
            .expect("reasoning delta hook records mutex was poisoned")
            .clone()
    }
}

impl AgentHook for RecordingReasoningDeltaHook {
    async fn on_reasoning_delta(
        &self,
        _ctx: &HookContext,
        event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        let record = (
            event.part.index(),
            event.delta.to_string(),
            event.aggregated.to_string(),
        );
        self.deltas
            .lock()
            .expect("reasoning delta hook records mutex was poisoned")
            .push(record);
        ObservationAction::continue_run()
    }
}

#[derive(Clone, Default)]
struct TerminatingReasoningDeltaHook {
    recorder: RecordingReasoningDeltaHook,
}

impl TerminatingReasoningDeltaHook {
    fn observed(&self) -> Vec<RecordedReasoningDelta> {
        self.recorder.observed()
    }
}

impl AgentHook for TerminatingReasoningDeltaHook {
    async fn on_reasoning_delta(
        &self,
        ctx: &HookContext,
        event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        self.recorder.on_reasoning_delta(ctx, event).await;
        ObservationAction::stop("stop on reasoning delta")
    }
}

#[derive(Clone, Default)]
struct RetryFirstReasoningTurnHook {
    recorder: RecordingReasoningDeltaHook,
    retried: Arc<AtomicBool>,
}

impl AgentHook for RetryFirstReasoningTurnHook {
    async fn on_reasoning_delta(
        &self,
        ctx: &HookContext,
        event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        self.recorder.on_reasoning_delta(ctx, event).await
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if self.retried.swap(true, Ordering::SeqCst) {
            ModelTurnAction::continue_run()
        } else {
            ModelTurnAction::repeat()
        }
    }
}

#[derive(Clone)]
struct RecordingTextAndRetryInvalidToolHook {
    text: RecordingTextDeltaHook,
}

impl AgentHook for RecordingTextAndRetryInvalidToolHook {
    async fn on_text_delta(&self, ctx: &HookContext, event: TextDelta<'_>) -> ObservationAction {
        self.text.on_text_delta(ctx, event).await
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        RetryDefaultApiHook.on_invalid_tool_call(ctx, event).await
    }
}

#[derive(Clone)]
struct RecordingDeltaAndRetryInvalidToolHook {
    delta: RecordingToolCallDeltaHook,
}

impl AgentHook for RecordingDeltaAndRetryInvalidToolHook {
    async fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        self.delta.on_tool_call_delta(ctx, event).await
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        RetryDefaultApiHook.on_invalid_tool_call(ctx, event).await
    }
}

#[derive(Clone)]
struct RecordingDeltaAndSkipInvalidToolHook {
    delta: RecordingToolCallDeltaHook,
}

impl AgentHook for RecordingDeltaAndSkipInvalidToolHook {
    async fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        self.delta.on_tool_call_delta(ctx, event).await
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        SkipDefaultApiHook.on_invalid_tool_call(ctx, event).await
    }
}

#[derive(Clone, Default)]
struct TerminatingToolCallDeltaHook {
    deltas: Arc<Mutex<Vec<RecordedToolCallDelta>>>,
}

impl TerminatingToolCallDeltaHook {
    fn observed(&self) -> Vec<RecordedToolCallDelta> {
        self.deltas
            .lock()
            .expect("tool call delta hook records mutex was poisoned")
            .clone()
    }
}

impl AgentHook for TerminatingToolCallDeltaHook {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        let record = (
            event.part.index(),
            event.tool_name.to_string(),
            event.delta.to_string(),
        );
        self.deltas
            .lock()
            .expect("tool call delta hook records mutex was poisoned")
            .push(record);
        ObservationAction::stop("stop on tool call delta")
    }
}

/// The streaming driver threads the per-call `ToolContext` to executed
/// tools, exactly like the blocking path.
#[tokio::test]
async fn tool_context_reaches_tool_through_streaming_loop() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tool_call_1", "context_probe", serde_json::json!({}))
                .with_call_id("call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let probe = MockContextProbeTool::default();
    let agent = AgentBuilder::new(model).tool(probe.clone()).build();
    let empty_history: &[Message] = &[];

    let mut tool_context = ToolContext::new();
    tool_context
        .insert(SessionId("xyz-789".to_string()))
        .unwrap();

    let mut stream = agent
        .prompt("do tool work")
        .tool_context(tool_context)
        .history(empty_history)
        .max_turns(3)
        .stream();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Err(err) => panic!("unexpected streaming error: {err:?}"),
            Ok(_) => {}
        }
    }

    assert_eq!(probe.observed().as_deref(), Some("session:xyz-789"));
}

#[tokio::test]
async fn invalid_tool_call_hook_can_repair_streaming_tool_name() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call(
                "tool_call_1",
                "default_api",
                serde_json::json!({"x": 2, "y": 3}),
            ),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(RepairDefaultApiHook)
        .max_turns(3)
        .history(Vec::<Message>::new())
        .stream();
    let mut saw_repaired_tool_call = false;
    let mut saw_tool_result = false;
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolCall { tool_call, .. }) => {
                assert_eq!(tool_call.function.name, "add");
                saw_repaired_tool_call = true;
            }
            Ok(MultiTurnStreamItem::ToolResult { tool_result, .. }) => {
                assert!(tool_result.content.iter().any(|content| {
                    matches!(
                        content,
                        ToolResultContent::Json { value }
                            if value == &serde_json::json!(5)
                    )
                }));
                saw_tool_result = true;
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert!(saw_repaired_tool_call);
    assert!(saw_tool_result);
    assert_eq!(final_response_text.as_deref(), Some("done"));
    assert_eq!(recorded.request_count(), 2);
}

#[tokio::test]
async fn invalid_tool_call_hook_skip_emits_streaming_tool_result() {
    let add_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call(
                "tool_call_1",
                "default_api",
                serde_json::json!({"x": 2, "y": 3}),
            )
            .with_call_id("call_1"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("continued"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(SkipDefaultApiHook)
        .max_turns(3)
        .history(Vec::<Message>::new())
        .stream();
    let mut skipped_tool_result = None;
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolResult { tool_result }) => {
                skipped_tool_result = Some(tool_result);
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let skipped_tool_result =
        skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
    // The correlator ("call_1") is the durable id; the wire's item id
    // ("tool_call_1") travels on `provider`.
    assert_eq!(
        skipped_tool_result
            .call
            .provider()
            .map(|provider| provider.as_str()),
        Some("call_1")
    );
    assert!(
        skipped_tool_result
            .call
            .provider()
            .as_ref()
            .is_some_and(|provider| { provider.as_str() == "call_1" })
    );
    assert!(skipped_tool_result.content.iter().any(|content| matches!(
        content,
        ToolResultContent::Text(text) if text.text == "default_api was skipped"
    )));
    assert_eq!(final_response_text.as_deref(), Some("continued"));
    assert_eq!(add_calls.load(Ordering::SeqCst), 0);

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let follow_up_history = requests[1].chat_history.clone();
    assert!(matches!(
        follow_up_history.get(2),
        Some(Message::User { content })
            if content.iter().any(|item| matches!(
                item,
                UserContent::ToolResult(result)
                    if result.call.provider().map(|provider| provider.as_str()) == Some("call_1")
                        && result.content.iter().any(|content| matches!(
                            content,
                            ToolResultContent::Text(text)
                                if text.text == "default_api was skipped"
                        ))
            ))
    ));
}

#[tokio::test]
async fn invalid_tool_call_hook_retries_mixed_streaming_turn_without_executing_valid_call() {
    let add_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("checking "),
            MockStreamEvent::tool_call("tool_call_1", "add", serde_json::json!({"x": 2, "y": 3}))
                .with_call_id("call_1"),
            MockStreamEvent::tool_call(
                "tool_call_2",
                "default_api",
                serde_json::json!({"x": 4, "y": 5}),
            )
            .with_call_id("call_2"),
            MockStreamEvent::final_response(usage(3, 1)),
        ],
        vec![
            MockStreamEvent::text("retried"),
            MockStreamEvent::final_response(usage(4, 2)),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(RetryDefaultApiHook)
        .max_turns(3)
        .history(Vec::<Message>::new())
        .max_invalid_tool_call_retries(1)
        .stream();
    let mut completion_call_events = Vec::new();
    let mut final_response_text = None;
    let mut final_response_usage = Usage::default();
    let mut final_completion_calls = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                completion_call_events.push(completion_call);
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                final_response_usage = response.usage();
                final_completion_calls = response.completion_calls().to_vec();
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(final_response_text.as_deref(), Some("retried"));
    assert_eq!(add_calls.load(Ordering::SeqCst), 0);
    let first_usage = usage(3, 1);
    let second_usage = usage(4, 2);
    let expected_completion_calls = vec![
        streamed_call(0, first_usage),
        streamed_call(1, second_usage),
    ];
    assert_eq!(completion_call_events, expected_completion_calls);
    assert_eq!(final_completion_calls, expected_completion_calls);
    assert_eq!(final_response_usage.total_tokens, Some(10));

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let retry_history = requests[1].chat_history.clone();
    assert_eq!(retry_history.len(), 3);
    assert!(matches!(
        retry_history.get(1),
        Some(Message::Assistant(rig_core::message::AssistantMessage { content, .. }))
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if text.text == "checking "
            ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
                            && tool_call.function.name == "add"
                ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_2")
                            && tool_call.function.name == "default_api"
                ))
    ));
    assert!(matches!(
        retry_history.get(2),
        Some(Message::User { content })
            if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_1")
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                            ))
                ))
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_2")
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "Use the add tool instead"
                            ))
                ))
    ));
    assert_retry_transcript_ids_pair(
        retry_history.get(1).expect("assistant tool-call turn"),
        retry_history.get(2).expect("retry-result turn"),
    );
}

#[tokio::test]
async fn invalid_tool_call_hook_skips_mixed_streaming_turn_without_executing_valid_call() {
    let add_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("checking "),
            MockStreamEvent::tool_call("tool_call_1", "add", serde_json::json!({"x": 2, "y": 3}))
                .with_call_id("call_1"),
            MockStreamEvent::tool_call(
                "tool_call_2",
                "default_api",
                serde_json::json!({"x": 4, "y": 5}),
            )
            .with_call_id("call_2"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("continued"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(SkipDefaultApiHook)
        .max_turns(3)
        .history(Vec::<Message>::new())
        .stream();
    let mut skipped_tool_result = None;
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolResult { tool_result, .. }) => {
                skipped_tool_result = Some(tool_result);
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let skipped_tool_result =
        skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
    // The correlator ("call_2") is the durable id; the wire's item id
    // ("tool_call_2") travels on `provider`.
    assert_eq!(
        skipped_tool_result
            .call
            .provider()
            .map(|provider| provider.as_str()),
        Some("call_2")
    );
    assert!(
        skipped_tool_result
            .call
            .provider()
            .as_ref()
            .is_some_and(|provider| { provider.as_str() == "call_2" })
    );
    assert_eq!(final_response_text.as_deref(), Some("continued"));
    assert_eq!(add_calls.load(Ordering::SeqCst), 0);

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let follow_up_history = requests[1].chat_history.clone();
    assert_eq!(follow_up_history.len(), 3);
    assert!(matches!(
        follow_up_history.get(1),
        Some(Message::Assistant(rig_core::message::AssistantMessage { content, .. }))
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if text.text == "checking "
            ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_1")
                            && tool_call.function.name == "add"
                ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_2")
                            && tool_call.function.name == "default_api"
                ))
    ));
    assert!(matches!(
        follow_up_history.get(2),
        Some(Message::User { content })
            if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_1")
                            && result.call.provider().as_ref().is_some_and(
                                |provider| provider.as_str() == "call_1"
                            )
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                            ))
                ))
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_2")
                            && result.call.provider().as_ref().is_some_and(
                                |provider| provider.as_str() == "call_2"
                            )
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "default_api was skipped"
                            ))
        ))
    ));
    assert_retry_transcript_ids_pair(
        follow_up_history.get(1).expect("assistant tool-call turn"),
        follow_up_history.get(2).expect("skip-result turn"),
    );
}

#[tokio::test]
async fn invalid_completed_tool_call_skip_preserves_streaming_reasoning_history() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("checking "),
            MockStreamEvent::reasoning("reasoned step").with_reasoning_id("rs_1"),
            MockStreamEvent::tool_call(
                "tool_call_1",
                "default_api",
                serde_json::json!({"x": 2, "y": 3}),
            ),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("continued"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(SkipDefaultApiHook)
        .max_turns(3)
        .history(Vec::<Message>::new())
        .stream();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let follow_up_history = requests[1].chat_history.clone();
    assert!(history_contains_text(&follow_up_history, "checking "));
    assert!(assistant_reasoning_precedes_tool_call(
        &follow_up_history,
        "reasoned step",
        "default_api"
    ));
    assert!(
        assistant_reasoning_precedes_text_and_tool_call(
            &follow_up_history,
            "reasoned step",
            "checking ",
            "default_api"
        ),
        "{follow_up_history:?}"
    );
}

#[tokio::test]
async fn invalid_tool_call_delta_retry_uses_structured_tool_feedback() {
    let delta_hook = RecordingToolCallDeltaHook::default();
    let add_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("checking "),
            MockStreamEvent::reasoning_delta_with_id("rs_1", "diagnostic reason"),
            MockStreamEvent::tool_call("tool_call_0", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_0"),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", r#"{"x":2,"y":3}"#),
            MockStreamEvent::tool_call_name_delta("tool_call_1", "default_api"),
            MockStreamEvent::final_response(usage(3, 1)),
        ],
        vec![
            MockStreamEvent::text("retried"),
            MockStreamEvent::final_response(usage(4, 2)),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(RecordingDeltaAndRetryInvalidToolHook {
            delta: delta_hook.clone(),
        })
        .max_turns(3)
        .history(Vec::<Message>::new())
        .max_invalid_tool_call_retries(1)
        .stream();
    let mut completion_call_events = Vec::new();
    let mut final_response_text = None;
    let mut final_response_usage = Usage::default();
    let mut final_completion_calls = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                completion_call_events.push(completion_call);
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: rig_core::message::AssistantContent::ToolCall(call),
                ..
            }))) if call.function.name == "default_api" => {
                panic!("an invalid tool call should not be emitted")
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                final_response_usage = response.usage();
                final_completion_calls = response.completion_calls().to_vec();
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(final_response_text.as_deref(), Some("retried"));
    // Only the valid call's arguments stream; the invalid call's never do.
    assert!(
        delta_hook
            .observed()
            .iter()
            .all(|(_, name, _)| name == "add"),
        "{:?}",
        delta_hook.observed()
    );
    assert_eq!(add_calls.load(Ordering::SeqCst), 0);
    let first_usage = usage(3, 1);
    let second_usage = usage(4, 2);
    let expected_completion_calls = vec![
        streamed_call(0, first_usage),
        streamed_call(1, second_usage),
    ];
    assert_eq!(completion_call_events, expected_completion_calls);
    assert_eq!(final_completion_calls, expected_completion_calls);
    assert_eq!(final_response_usage.total_tokens, Some(10));

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let retry_history = requests[1].chat_history.clone();
    assert!(matches!(
        retry_history.get(1),
        Some(Message::Assistant(rig_core::message::AssistantMessage { content, .. }))
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if text.text == "checking "
            ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_0")
                            && tool_call.function.name == "add"
                ))
                && content.iter().any(|item| matches!(
                item,
                // The invalid call keeps the id the provider named it.
                AssistantContent::ToolCall(tool_call)
                    if tool_call.id == rig_core::message::CallId::from_wire("tool_call_1")
                        && tool_call.function.name == "default_api"
                        && tool_call.function.arguments_value() == serde_json::json!({"x": 2, "y": 3})
            ))
    ));
    assert!(matches!(
        retry_history.get(2),
        Some(Message::User { content })
            if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_0")
                            && result.call.provider().as_ref().is_some_and(
                                |provider| provider.as_str() == "call_0"
                            )
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                            ))
                ))
                && content.iter().any(|item| matches!(
                item,
                UserContent::ToolResult(result)
                    if result.call == rig_core::message::CallId::from_wire("tool_call_1")
                        && result.name == "default_api"
                        && result.content.iter().any(|content| matches!(
                            content,
                            ToolResultContent::Text(text)
                                if text.text == "Use the add tool instead"
                        ))
            ))
    ));
    assert_retry_transcript_ids_pair(
        retry_history.get(1).expect("assistant tool-call turn"),
        retry_history.get(2).expect("retry-result turn"),
    );
}

#[tokio::test]
async fn invalid_tool_call_delta_retry_resets_streaming_text_delta_state() {
    let text_hook = RecordingTextDeltaHook::default();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("stale "),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", r#"{"x":2,"y":3}"#),
            MockStreamEvent::tool_call_name_delta("tool_call_1", "default_api"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("fresh"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(RecordingTextAndRetryInvalidToolHook {
            text: text_hook.clone(),
        })
        .max_turns(3)
        .history(Vec::<Message>::new())
        .max_invalid_tool_call_retries(1)
        .stream();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(
        text_hook.observed(),
        vec![
            ("stale ".to_string(), "stale ".to_string()),
            ("fresh".to_string(), "fresh".to_string()),
        ]
    );
}

#[tokio::test]
async fn invalid_tool_call_delta_skip_uses_structured_tool_feedback() {
    let delta_hook = RecordingToolCallDeltaHook::default();
    let add_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("checking "),
            MockStreamEvent::tool_call("tool_call_0", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_0"),
            MockStreamEvent::tool_call_arguments_delta("tool_call_1", r#"{"x":2,"y":3}"#),
            MockStreamEvent::tool_call_name_delta("tool_call_1", "default_api"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("continued"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .build();

    let mut stream = agent
        .prompt("use the tool")
        .add_hook(RecordingDeltaAndSkipInvalidToolHook {
            delta: delta_hook.clone(),
        })
        .max_turns(3)
        .history(Vec::<Message>::new())
        .stream();
    let mut skipped_tool_result = None;
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: rig_core::message::AssistantContent::ToolCall(call),
                ..
            }))) if call.function.name == "default_api" => {
                panic!("an invalid tool call should not be emitted")
            }
            Ok(MultiTurnStreamItem::ToolResult { tool_result }) => {
                skipped_tool_result = Some(tool_result);
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_string());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let skipped_tool_result =
        skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
    // The synthetic result answers the call under the id the provider
    // named it.
    assert_eq!(
        skipped_tool_result.call,
        rig_core::message::CallId::from_wire("tool_call_1")
    );
    assert_eq!(skipped_tool_result.name, "default_api");
    assert!(skipped_tool_result.content.iter().any(|content| matches!(
        content,
        ToolResultContent::Text(text) if text.text == "default_api was skipped"
    )));
    assert_eq!(final_response_text.as_deref(), Some("continued"));
    // Only the valid call's arguments stream; the invalid call's never do.
    assert!(
        delta_hook
            .observed()
            .iter()
            .all(|(_, name, _)| name == "add"),
        "{:?}",
        delta_hook.observed()
    );
    assert_eq!(add_calls.load(Ordering::SeqCst), 0);

    let requests = recorded.requests();
    assert_eq!(requests.len(), 2);
    let follow_up_history = requests[1].chat_history.clone();
    assert!(matches!(
        follow_up_history.get(1),
        Some(Message::Assistant(rig_core::message::AssistantMessage { content, .. }))
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if text.text == "checking "
            ))
                && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id.provider().map(|provider| provider.as_str()) == Some("call_0")
                            && tool_call.function.name == "add"
                ))
                && content.iter().any(|item| matches!(
                item,
                // The invalid call keeps the id the provider named it.
                AssistantContent::ToolCall(tool_call)
                    if tool_call.id == rig_core::message::CallId::from_wire("tool_call_1")
                        && tool_call.function.name == "default_api"
                        && tool_call.function.arguments_value() == serde_json::json!({"x": 2, "y": 3})
            ))
    ));
    assert!(matches!(
        follow_up_history.get(2),
        Some(Message::User { content })
            if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.call.provider().map(|provider| provider.as_str()) == Some("call_0")
                            && result.call.provider().as_ref().is_some_and(
                                |provider| provider.as_str() == "call_0"
                            )
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                            ))
                ))
                && content.iter().any(|item| matches!(
                item,
                UserContent::ToolResult(result)
                    if result.call == rig_core::message::CallId::from_wire("tool_call_1")
                        && result.name == "default_api"
                        && result.content.iter().any(|content| matches!(
                            content,
                            ToolResultContent::Text(text)
                                if text.text == "default_api was skipped"
                        ))
            ))
    ));
    assert_retry_transcript_ids_pair(
        follow_up_history.get(1).expect("assistant tool-call turn"),
        follow_up_history.get(2).expect("skip-result turn"),
    );
}

#[tokio::test]
async fn multiple_valid_streaming_tool_calls_execute_after_batch_validation() {
    let add_calls = Arc::new(AtomicU32::new(0));
    let subtract_calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tool_call_1", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_1"),
            MockStreamEvent::tool_call(
                "tool_call_2",
                "subtract",
                serde_json::json!({"x": 8, "y": 3}),
            )
            .with_call_id("call_2"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: add_calls.clone(),
        })
        .tool(CountingSubtractTool {
            calls: subtract_calls.clone(),
        })
        .build();

    let mut stream = agent.prompt("use tools").max_turns(3).stream();
    let mut tool_call_names = Vec::new();
    let mut tool_result_ids = Vec::new();
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolCall { tool_call, .. }) => {
                tool_call_names.push(tool_call.function.name);
            }
            Ok(MultiTurnStreamItem::ToolResult { tool_result, .. }) => {
                tool_result_ids.push(
                    tool_result
                        .call
                        .provider()
                        .map(|provider| provider.as_str())
                        .expect("explicit provider ID")
                        .to_owned(),
                );
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.output().to_owned());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(
        tool_call_names,
        vec!["add".to_string(), "subtract".to_string()]
    );
    // The correlators drive the durable ids the results answer with.
    assert_eq!(
        tool_result_ids,
        vec!["call_1".to_string(), "call_2".to_string()]
    );
    assert_eq!(add_calls.load(Ordering::SeqCst), 1);
    assert_eq!(subtract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(final_response_text.as_deref(), Some("done"));
    assert_eq!(recorded.request_count(), 2);
}

#[tokio::test]
async fn disallowed_specific_tool_call_fails_before_streaming_second_request() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call(
                "tool_call_1",
                "subtract",
                serde_json::json!({"x": 3, "y": 1}),
            ),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("should not be requested"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new("add").expect("tool name")],
        })
        .build();

    let mut stream = agent
        .prompt("use the allowed tool")
        .add_hook(PanicOnUnknownToolHook)
        .max_turns(3)
        .stream();
    let mut saw_tool_call = false;
    let mut error = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolCall { .. }) => {
                saw_tool_call = true;
            }
            Ok(_) => {}
            Err(err) => {
                error = Some(err);
                break;
            }
        }
    }

    assert!(!saw_tool_call);
    let error = error.expect("disallowed model-emitted tool should fail");
    match error {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            allowed_tools,
            chat_history,
        } => {
            assert_eq!(tool_name, "subtract");
            assert_eq!(
                available_tools,
                vec!["add".to_string(), "subtract".to_string()]
            );
            assert_eq!(allowed_tools, vec!["add".to_string()]);
            assert!(history_contains_tool_call(&chat_history, "subtract"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

#[tokio::test]
async fn tool_call_fragments_before_the_name_stream_once_the_call_is_named() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let hook = RecordingToolCallDeltaHook::default();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("stream a tool call")
        .add_hook(hook.clone())
        .max_turns(2)
        .stream();
    let mut arguments = Vec::new();
    let mut ended = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                json,
                ..
            }))) => arguments.push(json),
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: rig_core::message::AssistantContent::ToolCall(call),
                ..
            }))) => ended.push(call),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    // The fragment sent before the name waits for it, then each fragment
    // streams as it arrives, ahead of the call's end.
    let [call] = ended.as_slice() else {
        panic!("one call: {ended:?}");
    };
    assert_eq!(call.function.name, "add");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"x": 1, "y": 2})
    );
    assert_eq!(arguments, vec!["{\"x\":1,", "\"y\":2}"]);
    assert_eq!(
        hook.observed(),
        vec![
            (0, "add".to_string(), "{\"x\":1,".to_string()),
            (0, "add".to_string(), "\"y\":2}".to_string()),
        ]
    );
}

#[tokio::test]
async fn stream_prompt_observes_interleaved_reasoning_deltas_before_unchanged_emit() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning_delta("first "),
        MockStreamEvent::reasoning_delta_with_id("rs_b", "beta"),
        MockStreamEvent::reasoning_delta("second"),
        MockStreamEvent::reasoning("first second"),
        MockStreamEvent::final_response_with_total_tokens(3),
    ]]);
    let hook = RecordingReasoningDeltaHook::default();
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent
        .prompt("reason about this")
        .add_hook(hook.clone())
        .stream();
    let mut stream_deltas = Vec::new();
    let mut completed = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                part,
                text,
            }))) => stream_deltas.push((part.index(), text)),
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                part,
                content: rig_core::message::AssistantContent::Reasoning(reasoning),
            }))) => completed.push((part.index(), reasoning)),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    // Interleaved parts stay apart; each delta names its part.
    assert_eq!(
        stream_deltas,
        vec![
            (0, "first ".to_string()),
            (1, "beta".to_string()),
            (0, "second".to_string()),
        ]
    );
    // `rs_b`, which the script never ends, is closed at the provider's end
    // like the restated first part, and carries its provider id.
    assert_eq!(completed.len(), 2, "{completed:?}");
    let rs_b = completed
        .iter()
        .find(|(part, _)| *part == 1)
        .map(|(_, reasoning)| reasoning)
        .expect("the second part ends");
    assert_eq!(
        rs_b.native
            .as_ref()
            .and_then(|native| native.item.get("id")),
        Some(&serde_json::json!("rs_b"))
    );
    assert_eq!(
        hook.observed(),
        vec![
            (0, "first ".to_string(), "first ".to_string()),
            (1, "beta".to_string(), "beta".to_string()),
            (0, "second".to_string(), "first second".to_string()),
        ]
    );
}

#[tokio::test]
async fn stream_prompt_reasoning_delta_stop_prevents_emit_and_later_hook_dispatch() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning_delta_with_id("rs_1", "blocked"),
        MockStreamEvent::reasoning_delta_with_id("rs_1", "later"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let stopping = TerminatingReasoningDeltaHook::default();
    let later = RecordingReasoningDeltaHook::default();
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent
        .prompt("reason about this")
        .add_hook(stopping.clone())
        .add_hook(later.clone())
        .stream();
    let mut saw_delta = false;
    let mut saw_final_response = false;
    let mut error_message = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                ..
            }))) => saw_delta = true,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
            Ok(_) => {}
            Err(err) => {
                error_message = Some(err.to_string());
                break;
            }
        }
    }

    let observed = stopping.observed();
    assert_eq!(
        observed,
        vec![(0, "blocked".to_string(), "blocked".to_string())]
    );
    assert!(later.observed().is_empty());
    assert!(!saw_delta);
    assert!(!saw_final_response);
    assert!(
        error_message.as_deref().is_some_and(
            |message| message.contains("the run was cancelled: stop on reasoning delta")
        ),
        "expected hook termination error, got {error_message:?}"
    );
}

#[tokio::test]
async fn stream_prompt_reasoning_delta_hook_observes_retried_turns_as_provisional() {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::reasoning_delta("rejected"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
        [
            MockStreamEvent::reasoning_delta("accepted"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let hook = RetryFirstReasoningTurnHook::default();
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent
        .prompt("reason about this")
        .add_hook(hook.clone())
        .max_turns(2)
        .stream();
    let mut order = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                text: reasoning,
                ..
            }))) => order.push(reasoning),
            Ok(MultiTurnStreamItem::ModelTurnRetried { turn }) => {
                order.push(format!("retry:{turn}"));
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(order, vec!["rejected", "retry:1", "accepted"]);
    let observed = hook.recorder.observed();
    assert_eq!(observed.len(), 2);
    assert_eq!(observed[0].1, "rejected");
    assert_eq!(observed[0].2, "rejected");
    assert_eq!(observed[1].1, "accepted");
    assert_eq!(observed[1].2, "accepted");
}

#[tokio::test]
async fn stream_prompt_emits_tool_call_deltas_after_hook_continue() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let hook = RecordingToolCallDeltaHook::default();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("stream a tool call")
        .add_hook(hook.clone())
        .max_turns(2)
        .stream();
    let mut arguments = Vec::new();
    let mut call_ids = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                json,
                ..
            }))) => arguments.push(json),
            Ok(MultiTurnStreamItem::ToolCall { tool_call, .. }) => call_ids.push(tool_call.id),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert_eq!(call_ids.len(), 1, "one call: {call_ids:?}");
    assert_eq!(
        hook.observed(),
        vec![
            (0, "add".to_string(), "{\"x\":1,".to_string()),
            (0, "add".to_string(), "\"y\":2}".to_string()),
        ]
    );
    assert_eq!(arguments, vec!["{\"x\":1,", "\"y\":2}"]);
}

#[tokio::test]
async fn stream_prompt_tool_call_deltas_hook_termination_prevents_delta_emit() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call_name_delta("tool_1", "add"),
        MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
        MockStreamEvent::final_response_with_total_tokens(3),
    ]]);
    let hook = TerminatingToolCallDeltaHook::default();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("stream a tool call")
        .add_hook(hook.clone())
        .stream();
    let mut saw_delta = false;
    let mut saw_final_response = false;
    let mut error_message = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                ..
            }))) => {
                saw_delta = true;
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                saw_final_response = true;
            }
            Ok(_) => {}
            Err(err) => {
                error_message = Some(err.to_string());
                break;
            }
        }
    }

    let observed = hook.observed();
    assert_eq!(observed.len(), 1);
    let first = observed.first().expect("one observed delta");

    assert_eq!(first.1, "add");
    assert_eq!(first.2, "{\"x\":1}");
    assert!(!saw_delta);
    assert!(!saw_final_response);
    assert!(
        error_message.as_deref().is_some_and(
            |message| message.contains("the run was cancelled: stop on tool call delta")
        ),
        "expected hook termination error, got {error_message:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stream_prompt_records_single_call_usage_on_chat_span_under_outer_span() {
    let call_usage = usage(10, 2);
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response(call_usage),
    ]]);
    let agent = AgentBuilder::new(model).build();

    assert_stream_usage_recorded_on_chat_spans(agent, "say done", 1, &[call_usage]).await;
}

#[tokio::test(flavor = "current_thread")]
async fn stream_prompt_records_multi_turn_usage_on_chat_spans_under_outer_span() {
    let first_call_usage = usage(10, 2);
    let second_call_usage = usage(25, 5);
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tool_call_1", "add", serde_json::json!({"x": 1, "y": 2}))
                .with_call_id("call_1"),
            MockStreamEvent::final_response(first_call_usage),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(second_call_usage),
        ],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    assert_stream_usage_recorded_on_chat_spans(
        agent,
        "do tool work",
        3,
        &[first_call_usage, second_call_usage],
    )
    .await;
}

#[tokio::test]
async fn final_response_can_remain_empty_for_truly_textless_turns() {
    let agent = AgentBuilder::new(streaming_final_only_model()).build();

    let mut stream = agent.prompt("say nothing").stream();
    let mut streamed_text = String::new();
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            }))) => streamed_text.push_str(&text),
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_response_text = Some(res.output().to_owned());
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    assert!(streamed_text.is_empty());
    assert_eq!(final_response_text.as_deref(), Some(""));
}

/// rig#2322 — a turn that produced **nothing** and was cut short at the
/// output-token limit must not finalize as a successful empty answer.
///
/// Not a cassette test: a provider cannot be made to emit an exactly-empty
/// `MAX_TOKENS` turn on demand, so the wire shape is scripted. The Gemini
/// cassette suite pins the *request* side of rig#2322; this pins what the
/// agent does with the response.
///
/// This is the failure users actually saw. The 4096 cap truncated the turn,
/// the assembler dropped `FinishReason::Length`, and the run finished as a
/// successful `""` — a blank answer with no error and nothing to inspect.
#[tokio::test]
async fn empty_turn_truncated_at_max_tokens_is_an_error_not_an_empty_answer() {
    let model =
        MockCompletionModel::from_stream_turns([[MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..mock_final(Usage::default())
        })]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("write a long essay").stream();
    let mut error = None;
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_response_text = Some(res.output().to_owned());
                break;
            }
            Ok(_) => {}
            Err(err) => {
                error = Some(err);
                break;
            }
        }
    }

    assert!(
        final_response_text.is_none(),
        "a truncated, content-less turn must not finalize as a successful \
             answer — it did, yielding {final_response_text:?}"
    );
    let error = error.expect("the truncated turn should surface an error");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("Length"),
        "the error must name the terminal reason so the cause is diagnosable, got: {rendered}"
    );
    assert!(
        rendered.contains("max_tokens"),
        "a budget truncation must point at the setting that fixes it: {rendered}"
    );
}

/// rig#2322 — the guard against over-correcting the test above: a turn that
/// streamed **real output** before hitting the limit stays valid.
///
/// Truncation after partial output is a normal, useful result — the caller
/// gets the prefix the model produced. Only a turn that delivered nothing
/// is an error. Scripted for the same reason as above.
#[tokio::test]
async fn partial_output_truncated_at_max_tokens_stays_a_valid_answer() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::Text("a partial ans".to_string()),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..mock_final(Usage::default())
        }),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("write a long essay").stream();
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_response = Some(res);
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("a truncated turn that produced text must not error: {err:?}"),
        }
    }

    let final_response = final_response.expect("a turn with partial output should still finalize");
    assert_eq!(final_response.output(), "a partial ans");

    // ...and the reason is preserved, so a caller can tell this answer was
    // cut short rather than complete.
    let truncated = final_response
        .completion_calls
        .iter()
        .any(|call| call.finish_reason == Some(FinishReason::Length));
    assert!(
        truncated,
        "the terminal reason must reach the caller on completion_calls; \
             without it a truncated answer is indistinguishable from a complete \
             one — calls: {:?}",
        final_response.completion_calls
    );
}

/// rig#2322 — a content-filtered turn that delivered nothing gets the same
/// treatment as a truncated one: it is not a successful empty answer.
///
/// Scripted rather than recorded because a safety filter cannot be
/// provoked reliably or ethically on demand.
#[tokio::test]
async fn empty_content_filtered_turn_is_an_error_not_an_empty_answer() {
    let model =
        MockCompletionModel::from_stream_turns([[MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::ContentFilter),
            ..mock_final(Usage::default())
        })]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("something the filter rejects").stream();
    let mut errored = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => panic!(
                "a content-filtered, content-less turn must not finalize as a \
                     successful answer, got {:?}",
                res.output()
            ),
            Ok(_) => {}
            Err(err) => {
                errored = Some(err);
                break;
            }
        }
    }

    let rendered = format!("{:?}", errored.expect("the filtered turn should error"));
    assert!(
        rendered.contains("ContentFilter"),
        "the error must name the terminal reason, got: {rendered}"
    );
    assert!(
        !rendered.contains("max_tokens"),
        "a safety block must not advise raising max_tokens — that setting \
             cannot fix a filtered response: {rendered}"
    );
}

/// A finish reason the wire does not map is a failure (stops fail closed;
/// each wire maps its benign reasons to `Stop` itself), so an answerless
/// turn ending in one fails the run instead of finishing with `""`.
#[tokio::test]
async fn empty_turn_with_an_unknown_finish_reason_fails_the_run() {
    let model =
        MockCompletionModel::from_stream_turns([[MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Other("PROVIDER_SPECIFIC".to_string())),
            ..mock_final(Usage::default())
        })]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("say nothing").stream();
    let mut failure = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                panic!(
                    "a failed, answerless turn must not finish: {:?}",
                    res.output()
                );
            }
            Ok(_) => {}
            Err(err) => {
                failure = Some(err.to_string());
                break;
            }
        }
    }
    assert!(
        failure
            .as_deref()
            .is_some_and(|error| error.contains("PROVIDER_SPECIFIC")),
        "{failure:?}"
    );
}

/// rig#2322 — a turn that spent its whole budget **thinking** and was cut
/// off before answering must error, not report success with `""`.
///
/// This is the common shape of the bug, not a corner of it: Gemini counts
/// thinking tokens against `maxOutputTokens` (the committed cassettes show
/// `thoughtsTokenCount` of 176–307 on ordinary prompts), so a truncated
/// thinking turn *typically* carries reasoning and no text.
///
/// The first version of this guard keyed on `is_empty_assistant_turn`,
/// which is false for a reasoning-only turn — so the headline scenario
/// still finalized as a successful empty answer. The predicate is now
/// `turn_delivered_no_answer`.
///
/// Synthetic: a provider cannot be made to truncate mid-thought on demand.
#[tokio::test]
async fn reasoning_only_turn_truncated_at_max_tokens_is_an_error() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning("thinking hard and never reaching an answer"),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..mock_final(Usage::default())
        }),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("solve this carefully").stream();
    let mut error = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => panic!(
                "a turn that only produced reasoning before being truncated must \
                     not finalize as a successful answer, got {:?}",
                res.output()
            ),
            Ok(_) => {}
            Err(err) => {
                error = Some(err);
                break;
            }
        }
    }

    let rendered = format!("{:?}", error.expect("the truncated turn should error"));
    assert!(
        rendered.contains("Length"),
        "the error must name the terminal reason, got: {rendered}"
    );
}

/// rig#2322 — the same shape under a content filter takes the same path.
///
/// Synthetic for the same reason, plus: a safety filter cannot be provoked
/// reliably or ethically on demand.
#[tokio::test]
async fn reasoning_only_turn_content_filtered_is_an_error() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning("considering something the filter rejects"),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::ContentFilter),
            ..mock_final(Usage::default())
        }),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("something borderline").stream();
    let mut errored = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => panic!(
                "a reasoning-only filtered turn must not finalize successfully, \
                     got {:?}",
                res.output()
            ),
            Ok(_) => {}
            Err(err) => {
                errored = Some(err);
                break;
            }
        }
    }

    let rendered = format!(
        "{:?}",
        errored.expect("the filtered reasoning-only turn should error")
    );
    assert!(
        rendered.contains("ContentFilter") && !rendered.contains("max_tokens"),
        "a filtered turn must name its reason and must not advise raising \
             max_tokens: {rendered}"
    );
}

/// rig#2322 — the guard against over-correcting into "reasoning present
/// means failure": a turn that thought **and then answered** before being
/// truncated is a valid answer.
///
/// Synthetic: same reason as above.
#[tokio::test]
async fn reasoning_then_text_truncated_stays_a_valid_answer() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning("weighing the options"),
        MockStreamEvent::Text("the answer so f".to_string()),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..mock_final(Usage::default())
        }),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("solve this").stream();
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_response = Some(res);
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("a truncated turn that produced text must not error: {err:?}"),
        }
    }

    let final_response = final_response.expect("a turn with text should finalize");
    assert_eq!(final_response.output(), "the answer so f");
    assert!(
        final_response
            .completion_calls
            .iter()
            .any(|call| call.finish_reason == Some(FinishReason::Length)),
        "the terminal reason must still reach the caller on a valid truncated turn"
    );
}

/// rig#2322 — what happens to the partial reasoning when the turn errors.
///
/// A caller debugging a truncated thinking turn wants to see how far the
/// model got. The reasoning reaches them because it was *streamed*: every
/// part's end is delivered before the run reads the finish reason. History
/// keeps nothing — the truncation guard runs before the push, so a
/// reasoning-only turn nobody can answer around is not committed on either
/// runtime (CONTRACT §4; `run::tests::a_truncated_reasoning_only_turn_commits_nothing`).
/// This test once pinned the opposite ordering (push, then guard); the
/// stream is the surface that keeps the partial reasoning debuggable.
#[tokio::test]
async fn partial_reasoning_reaches_the_consumer_when_the_truncated_turn_errors() {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::reasoning("partial thinking worth keeping"),
        MockStreamEvent::FinalResponse(Finish {
            reason: Some(FinishReason::Length),
            ..mock_final(Usage::default())
        }),
    ]]);
    let agent = AgentBuilder::new(model).build();

    let mut stream = agent.prompt("solve this").stream();
    let mut streamed_reasoning = String::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: rig_core::message::AssistantContent::Reasoning(reasoning),
                ..
            }))) => {
                streamed_reasoning.push_str(&reasoning.text);
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                panic!("the truncated reasoning-only turn should error")
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }

    assert!(
        streamed_reasoning.contains("partial thinking worth keeping"),
        "the partial reasoning must reach the consumer before the error, so a \
             truncated thinking turn is debuggable — got {streamed_reasoning:?}"
    );
}

/// Background task that logs periodically to detect span leakage.
/// If span leakage occurs, these logs will be prefixed with `invoke_agent{...}`.
async fn background_logger(stop: Arc<AtomicBool>, leak_count: Arc<AtomicU32>) {
    let mut interval = tokio::time::interval(Duration::from_millis(50));
    let mut count = 0u32;

    while !stop.load(Ordering::Relaxed) {
        interval.tick().await;
        count += 1;

        tracing::event!(
            target: "background_logger",
            tracing::Level::INFO,
            count = count,
            "Background tick"
        );

        // Check if we're inside an unexpected span
        let current = tracing::Span::current();
        if !current.is_disabled() && !current.is_none() {
            leak_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    tracing::info!(target: "background_logger", total_ticks = count, "Background logger stopped");
}

/// Test that span context doesn't leak to concurrent tasks during streaming.
///
/// This test verifies that using `.instrument()` instead of `span.enter()` in
/// async_stream prevents thread-local span context from leaking to other tasks.
///
/// Uses single-threaded runtime to force all tasks onto the same thread,
/// making the span leak deterministic (it only occurs when tasks share a thread).
#[tokio::test(flavor = "current_thread")]
#[ignore = "This requires an API key"]
async fn test_span_context_isolation() -> anyhow::Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let leak_count = Arc::new(AtomicU32::new(0));

    // Start background logger
    let bg_stop = stop.clone();
    let bg_leak = leak_count.clone();
    let bg_handle = tokio::spawn(async move {
        background_logger(bg_stop, bg_leak).await;
    });

    // Small delay to let background logger start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Make streaming request WITHOUT an outer span so rig creates its own invoke_agent span
    // (rig reuses current span if one exists, so we need to ensure there's no current span)
    let agent = AgentBuilder::new(
        anthropic::wire::AnthropicConfig::from_env()?
            .connect(rig_reqwest::shared())
            .completion(anthropic::completion::CLAUDE_HAIKU_4_5),
    )
    .preamble("You are a helpful assistant.")
    .temperature(0.1)
    .max_tokens(100)
    .build();

    let mut stream = agent.prompt("Say 'hello world' and nothing else.").stream();

    let mut full_content = String::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            }))) => {
                full_content.push_str(&text);
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                break;
            }
            Err(e) => {
                tracing::warn!("Error: {:?}", e);
                break;
            }
            _ => {}
        }
    }

    tracing::info!("Got response: {:?}", full_content);

    // Stop background logger
    stop.store(true, Ordering::Relaxed);
    bg_handle.await?;

    let leaks = leak_count.load(Ordering::Relaxed);
    anyhow::ensure!(
        leaks == 0,
        "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times. \
             This indicates that span.enter() is being used inside async_stream instead of .instrument()"
    );

    Ok(())
}

/// Test that FinalResponse contains the updated chat history when a starting
/// history is provided via `.history(..)`.
///
/// This verifies that:
/// 1. PromptResponse.messages() returns Some when a starting history was provided
/// 2. The history contains both the user prompt and assistant response
#[tokio::test]
#[ignore = "This requires an API key"]
async fn test_chat_history_in_final_response() -> anyhow::Result<()> {
    use rig_core::message::Message;

    let agent = AgentBuilder::new(
        anthropic::wire::AnthropicConfig::from_env()?
            .connect(rig_reqwest::shared())
            .completion(anthropic::completion::CLAUDE_HAIKU_4_5),
    )
    .preamble("You are a helpful assistant. Keep responses brief.")
    .temperature(0.1)
    .max_tokens(50)
    .build();

    // Send streaming request with history
    let empty_history: &[Message] = &[];
    let mut stream = agent
        .prompt("Say 'hello' and nothing else.")
        .history(empty_history)
        .stream();

    // Consume the stream and collect FinalResponse
    let mut response_text = String::new();
    let mut final_history = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            }))) => {
                response_text.push_str(&text);
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_history = Some(res.messages().to_vec());
                break;
            }
            Err(e) => {
                return Err(e.into());
            }
            _ => {}
        }
    }

    let history =
        final_history.ok_or_else(|| anyhow::anyhow!("final response should include history"))?;

    // Should contain at least the user message
    anyhow::ensure!(
        history.iter().any(|m| matches!(m, Message::User { .. })),
        "History should contain the user message"
    );

    // Should contain the assistant response
    anyhow::ensure!(
        history.iter().any(|m| matches!(m, Message::Assistant(_))),
        "History should contain the assistant response"
    );

    tracing::info!(
        "History after streaming: {} messages, response: {:?}",
        history.len(),
        response_text
    );

    Ok(())
}

/// The streaming twin of the blocking refused-append test: the final item
/// is still delivered, and it reports the append the backend refused.
#[tokio::test]
async fn streaming_reports_a_refused_append_on_the_final_response() {
    let agent = AgentBuilder::new(streaming_text_then_final_model())
        .memory(rig_core::test_utils::AppendFailingMemory::default())
        .build();

    let mut stream = agent
        .prompt("hi there")
        .conversation("stream-thread")
        .stream();

    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_response = Some(res);
                break;
            }
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }
    let response = final_response.expect("the answer stands when the append is refused");
    assert_eq!(response.messages().len(), 2);
    let report = response
        .memory_append()
        .and_then(crate::run::MemoryAppend::failure)
        .expect("the refused append is reported on the final item");
    assert_eq!(report.kind, rig_core::error::ErrorKind::MemoryBackend);
    assert!(report.message.contains("append boom"), "{report:?}");
}

#[tokio::test]
async fn streaming_load_error_yields_memory_error() {
    let agent = AgentBuilder::new(streaming_text_then_final_model())
        .memory(FailingMemory::default())
        .build();

    let mut stream = agent.prompt("hi").conversation("t1").stream();

    let first = stream.next().await.expect("at least one item");
    match first {
        Err(err) => match err {
            PromptError::Memory(err) => {
                assert!(err.to_string().contains("load boom"));
            }
            other => panic!("expected PromptError::Memory, got {other:?}"),
        },
        other => panic!("expected PromptError::Memory, got {other:?}"),
    }
}

/// Dropping the event feed does not abort the run: the future still
/// drives the loop to completion and resolves.
#[tokio::test]
async fn run_channel_survives_dropped_events() {
    let model = streaming_tool_then_text_model();
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let (run, events) = agent.prompt("do tool work").max_turns(3).run_channel();
    drop(events);

    let response = run.await.expect("run succeeds without a consumer");
    assert_eq!(response.output(), "done");
    assert_eq!(recorded.requests().len(), 2);
}

/// The feed can be drained without awaiting — the shape a frame/tick loop
/// uses — and reports completion once the run is over and drained.
#[tokio::test]
async fn run_channel_try_next_drains_from_a_tick_loop() {
    let model = streaming_tool_then_text_model();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let (run, mut events) = agent.prompt("do tool work").max_turns(3).run_channel();
    let run = tokio::spawn(run);

    let mut seen = Vec::new();
    while !events.is_done() {
        match events.try_next() {
            Some(item) => seen.push(item),
            None => tokio::task::yield_now().await,
        }
    }

    let response = run.await.expect("join").expect("run succeeds");
    assert_eq!(response.output(), "done");
    assert!(matches!(
        seen.last(),
        Some(MultiTurnStreamItem::FinalResponse(_))
    ));
}

/// Load failures surface on the future as a `PromptError`, and the feed
/// closes without a final response.
#[tokio::test]
async fn run_channel_reports_stream_errors_on_the_future() {
    let model = MockCompletionModel::text("unused");
    let agent = AgentBuilder::new(model).build();

    let (run, events) = agent.prompt("budget of zero").max_turns(0).run_channel();
    let _: (_, RunEvents) = agent.prompt("plain entry point type-checks").run_channel();
    let (response, items) = futures::join!(run, events.collect::<Vec<_>>());

    assert!(response.is_err(), "zero-turn budget must fail the run");
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, MultiTurnStreamItem::FinalResponse(_)))
    );
}

/// The blocking terminal follows the same rule: `run()` called inside
/// `outer` and awaited from a spawned task adopts `outer` — no
/// `invoke_agent`, the chat span is `outer`'s child. (These tests rely on
/// `#[tokio::test]`'s current-thread runtime: the spawned task polls on
/// the thread holding the thread-local `set_default` subscriber.)
#[tokio::test]
async fn a_blocking_run_belongs_to_the_span_it_was_started_in() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let spans = TraceCapture::default();
    let _default = tracing::subscriber::set_default(spans.subscriber());

    let warmup_agent = AgentBuilder::new(MockCompletionModel::text("warmup")).build();
    warmup_agent.prompt("warmup").await.expect("warmup");
    tracing::callsite::rebuild_interest_cache();
    spans.clear();

    let agent = AgentBuilder::new(MockCompletionModel::text("done")).build();
    let outer_span = tracing::info_span!("outer");
    let run = outer_span.in_scope(|| agent.prompt("go").run());
    let response = tokio::spawn(run)
        .await
        .expect("join")
        .expect("run succeeds");
    assert_eq!(response.output(), "done");

    let snapshot = spans.spans();
    let outer_id = snapshot
        .iter()
        .find(|span| span.name == "outer")
        .map(|span| span.id)
        .expect("outer span captured");
    assert!(snapshot.iter().all(|span| span.name != "invoke_agent"));
    let chat_spans: Vec<_> = snapshot
        .iter()
        .filter(|span| span.name == "chat" && span.target == "rig::agent_chat")
        .collect();
    assert_eq!(chat_spans.len(), 1, "one model turn: {snapshot:?}");
    assert_eq!(chat_spans[0].parent, Some(outer_id));
}

/// The argument fragments a streamed run forwarded, and the error that
/// ended it, if any.
async fn forwarded_arguments(mut stream: StreamingResult) -> (Vec<String>, Option<PromptError>) {
    let mut arguments = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                json,
                ..
            }))) => arguments.push(json),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => return (arguments, Some(err)),
        }
    }
    (arguments, None)
}

#[tokio::test]
async fn stream_prompt_emits_tool_call_deltas_without_hook() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let (arguments, error) =
        forwarded_arguments(agent.prompt("stream a tool call").max_turns(2).stream()).await;

    assert!(error.is_none(), "{error:?}");
    assert_eq!(arguments, ["{\"x\":1,", "\"y\":2}"]);
}

/// Two calls streamed in one turn interleave their fragments, each under
/// its own part, and both execute once the turn ends.
#[tokio::test]
async fn interleaved_tool_calls_stream_under_their_own_parts() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::tool_call_name_delta("tool_2", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_arguments_delta("tool_2", "{\"x\":3,"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::tool_call_arguments_delta("tool_2", "\"y\":4}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let hook = RecordingToolCallDeltaHook::default();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("stream two tool calls")
        .add_hook(hook.clone())
        .max_turns(2)
        .stream();
    let mut fragments: Vec<(usize, String)> = Vec::new();
    let mut results = 0;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                part,
                json,
            }))) => fragments.push((part.index(), json)),
            Ok(MultiTurnStreamItem::ToolResult { .. }) => {
                // Every fragment of the turn arrived before any tool ran.
                assert_eq!(fragments.len(), 4);
                results += 1;
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let expected = [
        (0, "{\"x\":1,".to_string()),
        (1, "{\"x\":3,".to_string()),
        (0, "\"y\":2}".to_string()),
        (1, "\"y\":4}".to_string()),
    ];
    assert_eq!(fragments, expected);
    assert_eq!(
        hook.observed(),
        expected
            .iter()
            .map(|(part, delta)| (*part, "add".to_string(), delta.clone()))
            .collect::<Vec<_>>()
    );
    assert_eq!(results, 2);
}

#[tokio::test]
async fn unknown_tool_call_name_delta_fails_before_streaming_delta_hook_or_emit() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "default_api"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("should not be requested"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let (arguments, error) = forwarded_arguments(
        agent
            .prompt("stream a bad tool call")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .stream(),
    )
    .await;

    assert!(arguments.is_empty(), "{arguments:?}");
    match error.expect("an unknown tool-call name fails the run") {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            allowed_tools,
            chat_history,
        } => {
            assert_eq!(tool_name, "default_api");
            assert_eq!(available_tools, vec!["add".to_string()]);
            assert_eq!(allowed_tools, vec!["add".to_string()]);
            assert!(history_contains_tool_call(&chat_history, "default_api"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

#[tokio::test]
async fn tool_call_args_delta_before_unknown_name_fails_before_hook_or_emit() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
            MockStreamEvent::tool_call_name_delta("tool_1", "default_api"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("should not be requested"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let (arguments, error) = forwarded_arguments(
        agent
            .prompt("stream a bad tool call")
            .add_hook(PanicOnUnknownToolHook)
            .max_turns(3)
            .stream(),
    )
    .await;

    assert!(arguments.is_empty(), "{arguments:?}");
    match error.expect("an unknown tool-call name rejects the buffered arguments") {
        PromptError::UnknownToolCall {
            tool_name,
            chat_history,
            ..
        } => {
            assert_eq!(tool_name, "default_api");
            assert!(history_contains_tool_call(&chat_history, "default_api"));
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

#[tokio::test]
async fn tool_choice_none_holds_args_then_rejects_name_without_emit() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1}"),
            MockStreamEvent::tool_call_name_delta("tool_1", "add"),
            MockStreamEvent::final_response_with_total_tokens(4),
        ],
        vec![
            MockStreamEvent::text("should not be requested"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ],
    ]);
    let recorded = model.clone();
    let agent = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool_choice(ToolChoice::None)
        .build();

    let hook = RecordingToolCallDeltaHook::default();
    let (arguments, error) = forwarded_arguments(
        agent
            .prompt("do not use tools")
            .add_hook(hook.clone())
            .max_turns(3)
            .stream(),
    )
    .await;

    assert!(arguments.is_empty(), "{arguments:?}");
    assert!(hook.observed().is_empty(), "{:?}", hook.observed());
    match error.expect("ToolChoice::None rejects a streamed call") {
        PromptError::UnknownToolCall {
            tool_name,
            allowed_tools,
            ..
        } => {
            assert_eq!(tool_name, "add");
            assert!(allowed_tools.is_empty());
        }
        other => panic!("expected UnknownToolCall, got {other:?}"),
    }
    assert_eq!(recorded.request_count(), 1);
}

/// A call to an unknown tool that a hook repairs streams its held start
/// and fragments under the repaired name, then executes.
#[tokio::test]
async fn a_repaired_call_streams_its_held_fragments_under_the_repaired_name() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tool_1", "default_api"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "{\"x\":1,"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "\"y\":2}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ],
        vec![
            MockStreamEvent::text("3"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let delta = RecordingToolCallDeltaHook::default();
    let agent = AgentBuilder::new(model).tool(MockAddTool).build();

    let mut stream = agent
        .prompt("stream a misnamed tool call")
        .add_hook(RepairDefaultApiHook)
        .add_hook(delta.clone())
        .max_turns(2)
        .stream();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(event))) => {
                events.push(event);
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
            Ok(_) => {}
            Err(err) => panic!("unexpected streaming error: {err:?}"),
        }
    }

    let call_events: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Start {
                name: Some(name), ..
            } => Some(format!("start:{}", name.as_str())),
            StreamEvent::Arguments { json, .. } => Some(format!("args:{json}")),
            StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            } => Some(format!("end:{}", call.function.name)),
            _ => None,
        })
        .collect();
    assert_eq!(
        call_events,
        ["start:add", "args:{\"x\":1,", "args:\"y\":2}", "end:add"]
    );
    assert_eq!(
        delta.observed(),
        [
            (0, "add".to_string(), "{\"x\":1,".to_string()),
            (0, "add".to_string(), "\"y\":2}".to_string()),
        ]
    );
}

/// A streamed answer that ends in an unknown finish reason fails the run by
/// default, after its text streamed; accepted, it is the run's answer.
#[tokio::test]
async fn a_streamed_answer_with_an_unknown_finish_reason_needs_acceptance() {
    for accept in [false, true] {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("answer"),
            MockStreamEvent::FinalResponse(Finish {
                reason: Some(FinishReason::Other("weird".to_string())),
                ..mock_final(Usage::default())
            }),
        ]]);
        let agent = AgentBuilder::new(model)
            .accept_unknown_finish_reasons(accept)
            .build();
        let mut stream = agent.prompt("hello").stream();
        let mut outcome = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    outcome = Some(Ok(res.output().to_owned()));
                }
                Ok(_) => {}
                Err(err) => {
                    outcome = Some(Err(err.to_string()));
                    break;
                }
            }
        }
        match outcome.expect("the run ends") {
            Ok(answer) => {
                assert!(accept, "only an accepted reason answers");
                assert_eq!(answer, "answer");
            }
            Err(error) => {
                assert!(!accept, "an accepted reason does not fail: {error}");
                assert!(error.contains("weird"), "{error}");
            }
        }
    }
}
