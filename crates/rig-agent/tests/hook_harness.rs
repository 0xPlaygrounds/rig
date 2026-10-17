#![cfg(all(feature = "test-utils", not(target_family = "wasm")))]

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use rig_agent::{
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent,
        HookContext, HookStack, InvalidToolCallAction, InvalidToolCallContext, ModelSelection,
        ModelSelectionAction, ModelTurnAction, ModelTurnFinished, ObservationAction, OutcomeAction,
        OutcomeEvent, ReasoningDelta, RunEntry, RunId, RunSettled, RunStart, RunStartAction,
        SettledOutcome, StepEventKind, TextDelta, ToolCallDelta,
    },
    test_utils::HookTestHarness,
    tool::{ToolContext, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult},
};
use rig_core::{
    completion::{FinishReason, ModelRef, ResponseIdentity, Usage},
    effect::{EffectId, EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message},
    streaming::Part,
    tool::ContextValue,
};
use serde_json::json;
use tokio::sync::Barrier;

fn dispatch(id: u64, kind: &EffectKind) -> DispatchEvent<'_> {
    DispatchEvent {
        id: EffectId::from_raw(id),
        kind,
        turn: 1,
        call_id: None,
        context: None,
    }
}

fn outcome<'a>(
    id: u64,
    kind: &'a EffectKind,
    result: &'a std::result::Result<Outcome, ErrorReport>,
) -> OutcomeEvent<'a> {
    OutcomeEvent {
        id: EffectId::from_raw(id),
        kind,
        outcome: result,
        turn: 1,
        call_id: None,
        context: None,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Visit {
    run: RunId,
    turn: usize,
    streaming: bool,
    agent: Option<String>,
}

#[derive(Clone, Default)]
struct Visits(Vec<Visit>);

struct RecordContext;

impl RecordContext {
    fn record(ctx: &HookContext) -> std::result::Result<(), serde_json::Error> {
        ctx.scratchpad().update::<Visits, _>(|visits| {
            visits.0.push(Visit {
                run: ctx.run_id(),
                turn: ctx.turn(),
                streaming: ctx.is_streaming(),
                agent: ctx.agent_name().map(str::to_owned),
            });
        });
        ctx.append_entry("visits", &ctx.turn())
    }
}

impl AgentHook for RecordContext {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        match Self::record(ctx) {
            Ok(()) => RunStartAction::Continue,
            Err(error) => RunStartAction::stop(error.to_string()),
        }
    }

    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        match Self::record(ctx) {
            Ok(()) => CompletionCallAction::Continue,
            Err(error) => CompletionCallAction::stop(error.to_string()),
        }
    }
}

#[tokio::test]
async fn metadata_and_entries_persist_across_events_without_leaking_between_runs() -> Result<()> {
    let run = RunId::new();
    let seed = RunEntry {
        kind: "visits".into(),
        turn: 2,
        value: json!(2),
    };
    let mut harness = HookTestHarness::new()
        .with_hook(RecordContext)
        .with_run_id(run)
        .with_agent_name("policy-test")
        .with_streaming(true)
        .with_turn(3)
        .with_entries(std::slice::from_ref(&seed));
    let prompt = Message::user("test policy");
    ensure!(matches!(
        harness
            .on_run_start(RunStart {
                prompt: &prompt,
                history: &[]
            })
            .await,
        RunStartAction::Continue
    ));
    harness.set_turn(4);
    ensure!(matches!(
        harness
            .on_completion_call(CompletionCallEvent {
                prompt: &prompt,
                history: &[],
                turn: 4,
            })
            .await,
        CompletionCallAction::Continue
    ));
    let visits = harness
        .context()
        .scratchpad()
        .get::<Visits>()
        .context("recorded visits")?;
    ensure!(
        visits.0
            == [3, 4].map(|turn| Visit {
                run,
                turn,
                streaming: true,
                agent: Some("policy-test".into()),
            })
    );
    let entries = harness.context().entries("visits");
    let [resumed, first, second] = entries.as_slice() else {
        anyhow::bail!("expected one resumed entry and two hook appends");
    };
    ensure!(*resumed == seed);
    ensure!(first.turn == 3 && first.value == json!(3));
    ensure!(second.turn == 4 && second.value == json!(4));
    ensure!(harness.context().last_entry("visits") == entries.last().cloned());

    let other = HookTestHarness::default();
    ensure!(other.context().run_id() != run);
    ensure!(other.context().turn() == 0);
    ensure!(!other.context().is_streaming());
    ensure!(other.context().agent_name().is_none());
    ensure!(!other.context().scratchpad().contains::<Visits>());
    ensure!(other.context().entries("visits").is_empty());
    Ok(())
}

#[derive(Clone, Default)]
struct DispatchTrace(Vec<(&'static str, String)>);

struct DispatchPolicy(&'static str);

impl AgentHook for DispatchPolicy {
    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        ctx.scratchpad().update::<DispatchTrace, _>(|trace| {
            trace
                .0
                .push((self.0, event.tool_args().unwrap_or_default().to_owned()));
        });
        match self.0 {
            "rewrite" => DispatchAction::rewrite_tool_args(event.kind, json!({"approved": false})),
            "deny" => DispatchAction::deny(
                ErrorReport::new(ErrorKind::Other, "approval required").with_code("approval"),
            ),
            _ => DispatchAction::Proceed,
        }
    }
}

struct Uninterested(Arc<AtomicUsize>);

impl AgentHook for Uninterested {
    fn observes(&self, _kind: StepEventKind) -> bool {
        false
    }

    async fn on_dispatch(&self, _ctx: &HookContext, _event: DispatchEvent<'_>) -> DispatchAction {
        self.0.fetch_add(1, Ordering::Relaxed);
        DispatchAction::stop("must not run")
    }

    async fn on_outcome(&self, _ctx: &HookContext, _event: OutcomeEvent<'_>) -> OutcomeAction {
        self.0.fetch_add(1, Ordering::Relaxed);
        OutcomeAction::stop("must not run")
    }
}

#[tokio::test]
async fn ordered_hooks_share_rewrites_and_scratchpad_until_denied() -> Result<()> {
    let ignored_calls = Arc::new(AtomicUsize::new(0));
    let mut hooks = HookStack::new();
    hooks.push(Uninterested(ignored_calls.clone()));
    hooks.push(DispatchPolicy("rewrite"));
    hooks.push(DispatchPolicy("deny"));
    let harness = HookTestHarness::new()
        .with_hooks(hooks)
        .with_hook(DispatchPolicy("late"));
    ensure!(harness.hooks().names().len() == 4);
    let kind = EffectKind::ToolCall {
        name: "deploy".into(),
        args: "{}".into(),
    };
    let action = harness.on_dispatch(dispatch(1, &kind)).await;
    ensure!(matches!(action, DispatchAction::Deny(report)
        if report.message == "approval required" && report.code.as_deref() == Some("approval")));
    let trace = harness
        .context()
        .scratchpad()
        .get::<DispatchTrace>()
        .context("dispatch trace")?;
    ensure!(
        trace.0
            == [
                ("rewrite", "{}".to_owned()),
                ("deny", json!({"approved": false}).to_string()),
            ]
    );
    ensure!(ignored_calls.load(Ordering::Relaxed) == 0);
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize, PartialEq)]
struct FailureMetadata {
    resource: String,
}

impl ContextValue for FailureMetadata {
    const KEY: &'static str = "test.failure";
}

#[derive(Clone, Debug, PartialEq)]
struct SeenResource(Option<String>);

struct RedactFailure;

impl AgentHook for RedactFailure {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if let Some(result) = event.tool_result()
            && result.is_error_kind(ToolErrorKind::RateLimited)
        {
            let metadata = event
                .tool_context()
                .and_then(|context| context.result::<FailureMetadata>().ok().flatten());
            ctx.scratchpad()
                .insert(SeenResource(metadata.map(|value| value.resource)));
            return OutcomeAction::rewrite_tool_result(&event, "Try again later");
        }
        if let Err(report) = event.outcome
            && !report.retryable
        {
            return OutcomeAction::stop(format!("fatal: {}", report.message));
        }
        OutcomeAction::Proceed
    }
}

struct ObserveReplacement;

impl AgentHook for ObserveReplacement {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        ctx.scratchpad().insert(
            event
                .tool_result()
                .and_then(|result| result.output().as_text())
                .map(str::to_owned),
        );
        OutcomeAction::Proceed
    }
}

#[tokio::test]
async fn outcome_policies_inspect_structured_failures_and_chain_replacements() -> Result<()> {
    let ignored_calls = Arc::new(AtomicUsize::new(0));
    let harness = HookTestHarness::new()
        .with_hook(Uninterested(ignored_calls.clone()))
        .with_hook(RedactFailure)
        .with_hook(ObserveReplacement);
    let kind = EffectKind::ToolCall {
        name: "fetch".into(),
        args: "{}".into(),
    };
    let failed = Ok(Outcome::ToolResult {
        result: ToolResult::failed(
            ToolExecutionError::rate_limited("private upstream detail")
                .with_code("quota")
                .with_http_status(429),
        ),
    });
    let mut context = ToolContext::new();
    context.insert_result(FailureMetadata {
        resource: "catalog".into(),
    })?;
    let action = harness
        .on_outcome(OutcomeEvent {
            context: Some(&context),
            ..outcome(2, &kind, &failed)
        })
        .await;
    let OutcomeAction::Replace(Ok(Outcome::ToolResult { result })) = action else {
        anyhow::bail!("expected the redacted tool result");
    };
    ensure!(result.output().as_text() == Some("Try again later"));
    let error = result.error().context("failure classification retained")?;
    ensure!(error.kind() == ToolErrorKind::RateLimited);
    ensure!(error.code() == Some("quota") && error.http_status() == Some(429));
    ensure!(error.message() == "private upstream detail");
    // The later hook sees the replacement, and the published metadata remains available.
    ensure!(
        harness.context().scratchpad().get::<Option<String>>()
            == Some(Some("Try again later".into()))
    );
    ensure!(
        harness.context().scratchpad().get::<SeenResource>()
            == Some(SeenResource(Some("catalog".into())))
    );
    ensure!(context.require_result::<FailureMetadata>()?.resource == "catalog");
    ensure!(ignored_calls.load(Ordering::Relaxed) == 0);

    let fatal = Err(ErrorReport::new(ErrorKind::Other, "transport failed").with_code("offline"));
    harness.context().scratchpad().remove::<Option<String>>();
    ensure!(
        matches!(harness.on_outcome(outcome(3, &kind, &fatal)).await,
        OutcomeAction::Replace(Err(report)) if report.kind == ErrorKind::Cancelled && report.message == "fatal: transport failed")
    );
    ensure!(
        !harness.context().scratchpad().contains::<Option<String>>(),
        "cancellation must stop later hooks"
    );
    Ok(())
}

#[derive(Clone, Default)]
struct Calls(Arc<Mutex<HashMap<EffectId, String>>>);

#[derive(Clone, Default)]
struct Completed(Arc<Mutex<Vec<(EffectId, String, String)>>>);

struct CorrelateCalls {
    barrier: Barrier,
}

impl AgentHook for CorrelateCalls {
    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if let Some(calls) = ctx.scratchpad().get::<Calls>() {
            calls
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(event.id, event.tool_args().unwrap_or_default().to_owned());
        }
        self.barrier.wait().await;
        tokio::task::yield_now().await;
        DispatchAction::Proceed
    }

    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.barrier.wait().await;
        tokio::task::yield_now().await;
        let args = ctx.scratchpad().get::<Calls>().and_then(|calls| {
            calls
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&event.id)
        });
        if let (Some(args), Some(result), Some(completed)) = (
            args,
            event.tool_result(),
            ctx.scratchpad().get::<Completed>(),
        ) {
            completed
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((
                    event.id,
                    args,
                    result.output().as_text().unwrap_or_default().to_owned(),
                ));
        }
        OutcomeAction::Proceed
    }
}

#[tokio::test]
async fn concurrent_calls_correlate_shared_state_by_effect_id() -> Result<()> {
    let harness = HookTestHarness::new().with_hook(CorrelateCalls {
        barrier: Barrier::new(2),
    });
    let calls = Calls::default();
    let completed = Completed::default();
    harness.context().scratchpad().insert(calls.clone());
    harness.context().scratchpad().insert(completed.clone());
    let first = EffectKind::ToolCall {
        name: "lookup".into(),
        args: json!({"key": "first"}).to_string(),
    };
    let second = EffectKind::ToolCall {
        name: "lookup".into(),
        args: json!({"key": "second"}).to_string(),
    };
    let (first_action, second_action) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            harness.on_dispatch(dispatch(10, &first)),
            harness.on_dispatch(dispatch(11, &second))
        )
    })
    .await?;
    ensure!(matches!(first_action, DispatchAction::Proceed));
    ensure!(matches!(second_action, DispatchAction::Proceed));
    ensure!(
        calls
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            == 2
    );
    let first_result = Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text("first answer")),
    });
    let second_result = Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text("second answer")),
    });
    // Deliver the second call first; correlation must not depend on arrival order.
    let (second_action, first_action) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(
            harness.on_outcome(outcome(11, &second, &second_result)),
            harness.on_outcome(outcome(10, &first, &first_result))
        )
    })
    .await?;
    ensure!(matches!(first_action, OutcomeAction::Proceed));
    ensure!(matches!(second_action, OutcomeAction::Proceed));
    let mut seen = completed
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    seen.sort_by_key(|(id, _, _)| id.as_u64());
    ensure!(
        seen == [
            (
                EffectId::from_raw(10),
                json!({"key": "first"}).to_string(),
                "first answer".into()
            ),
            (
                EffectId::from_raw(11),
                json!({"key": "second"}).to_string(),
                "second answer".into()
            ),
        ]
    );
    ensure!(
        calls
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    );
    Ok(())
}

struct LifecyclePolicy;

impl AgentHook for LifecyclePolicy {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if event.selected_model == event.default_model {
            ModelSelectionAction::select("fallback")
        } else {
            ModelSelectionAction::Continue
        }
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event.finish_reason == Some(&FinishReason::Length) {
            ModelTurnAction::retry_with_feedback("answer was truncated")
        } else {
            ModelTurnAction::Continue
        }
    }

    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::skip(format!(
            "unknown tool: {}",
            event.tool_name
        )))
    }

    async fn on_text_delta(&self, _ctx: &HookContext, event: TextDelta<'_>) -> ObservationAction {
        if event.aggregated.contains("secret") {
            ObservationAction::stop("redacted")
        } else {
            ObservationAction::Continue
        }
    }

    async fn on_reasoning_delta(
        &self,
        ctx: &HookContext,
        event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        ctx.scratchpad()
            .insert((event.part.index(), event.aggregated.to_owned()));
        ObservationAction::Continue
    }

    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        if event.tool_name == "forbidden" {
            ObservationAction::stop("tool blocked")
        } else {
            ObservationAction::Continue
        }
    }

    async fn on_run_settled(&self, ctx: &HookContext, event: RunSettled<'_>) {
        if let SettledOutcome::Error(reason) = event.outcome {
            ctx.scratchpad().insert(reason.to_owned());
        }
    }
}

#[tokio::test]
async fn typed_lifecycle_events_return_policy_actions_without_a_runner() -> Result<()> {
    let harness = HookTestHarness::new().with_hook(LifecyclePolicy);
    let prompt = Message::user("hello");
    let model = ModelRef::from("default");
    ensure!(
        matches!(harness.on_model_select(ModelSelection::new(&prompt, &[], None, None, &model, &model)),
        ModelSelectionAction::Select(selected) if selected == "fallback")
    );
    let content = vec![AssistantContent::text("partial")];
    let identity = ResponseIdentity::default();
    let raw = json!({});
    ensure!(
        harness
            .on_model_turn_finished(ModelTurnFinished {
                turn: 1,
                content: &content,
                usage: Usage::default(),
                identity: &identity,
                finish_reason: Some(&FinishReason::Length),
                max_tokens: Some(10),
                raw: &raw,
            })
            .await
            == ModelTurnAction::retry_with_feedback("answer was truncated")
    );
    let invalid: InvalidToolCallContext = serde_json::from_value(json!({
        "tool_name": "missing", "tool_call_id": null, "args": "{}", "available_tools": [],
        "allowed_tools": [], "tool_choice": null, "chat_history": [], "is_streaming": true,
        "reason": {"reason": "unknown_tool"}
    }))?;
    ensure!(
        harness.on_invalid_tool_call(&invalid).await
            == Some(InvalidToolCallAction::skip("unknown tool: missing"))
    );
    ensure!(
        harness
            .on_text_delta(TextDelta {
                delta: "cret",
                aggregated: "secret"
            })
            .await
            == ObservationAction::stop("redacted")
    );
    let part: Part = serde_json::from_value(json!(2))?;
    ensure!(
        harness
            .on_reasoning_delta(ReasoningDelta {
                part,
                delta: "thought",
                aggregated: "a thought"
            })
            .await
            == ObservationAction::Continue
    );
    ensure!(
        harness.context().scratchpad().get::<(usize, String)>() == Some((2, "a thought".into()))
    );
    ensure!(
        harness
            .on_tool_call_delta(ToolCallDelta {
                part,
                tool_name: "forbidden",
                delta: "{",
                aggregated: "{"
            })
            .await
            == ObservationAction::stop("tool blocked")
    );
    harness
        .on_run_settled(RunSettled {
            outcome: SettledOutcome::Error("failed"),
            messages: Some(&[]),
        })
        .await;
    ensure!(harness.context().scratchpad().get::<String>() == Some("failed".into()));
    Ok(())
}
