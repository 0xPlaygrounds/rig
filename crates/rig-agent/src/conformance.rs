//! Classic-runtime adapter for the shared behavioral conformance ledger.

use futures::StreamExt;
use rig_core::{
    completion::AssistantContent,
    test_utils::{CountingMemory, FailingMemory},
};
use rig_runtime_conformance::{
    MockCompletionModel, MockStreamEvent, MockTurn, RuntimeScenarioAdapter, Scenario,
    ScenarioEvidence, ScenarioFuture, verify_runtime,
};

use crate::{
    agent::{
        AgentBuilder, AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext,
        ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem, OutputMode, ToolCall,
        ToolCallAction,
    },
    completion::Prompt,
    test_utils::MockAddTool,
};

struct RetryRejected;

impl AgentHook for RetryRejected {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event.content.iter().any(
            |content| matches!(content, AssistantContent::Text(text) if text.text == "rejected"),
        ) {
            ModelTurnAction::repeat()
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

struct StopAfterTurn;

impl AgentHook for StopAfterTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        ModelTurnAction::stop("stop")
    }
}

struct SuppressTools;

impl AgentHook for SuppressTools {
    async fn on_tool_call(&self, _ctx: &HookContext, _event: ToolCall<'_>) -> ToolCallAction {
        ToolCallAction::skip("denied")
    }
}

struct StopInvalid;

impl AgentHook for StopInvalid {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::stop("stop invalid"))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct AdapterError(&'static str);

fn ensure(condition: bool, message: &'static str) -> Result<(), AdapterError> {
    condition.then_some(()).ok_or(AdapterError(message))
}

struct ClassicAdapter;

impl RuntimeScenarioAdapter for ClassicAdapter {
    type Error = AdapterError;

    fn runtime_name(&self) -> &'static str {
        "rig-agent"
    }

    fn exercise(&mut self, scenario: Scenario) -> ScenarioFuture<'_, Self::Error> {
        Box::pin(async move {
            // Every ledger entry crosses the real classic driver and validates
            // its canonical response/accounting baseline before the
            // scenario-specific probe below.
            let model = MockCompletionModel::from_turns([MockTurn::text("accepted").with_usage({
                let mut usage = rig_core::completion::Usage::new();
                usage.total_tokens = 1;
                usage
            })]);
            let response = AgentBuilder::new(model.clone())
                .build()
                .prompt("question")
                .extended_details()
                .await
                .map_err(|_| AdapterError("classic baseline run failed"))?;
            ensure(response.output == "accepted", "wrong canonical output")?;
            ensure(
                response.usage().total_tokens == 1,
                "usage was not committed",
            )?;
            ensure(
                response.completion_calls().len() == 1,
                "call record missing",
            )?;
            ensure(
                response
                    .messages()
                    .is_some_and(|messages| messages.len() == 2),
                "canonical prompt/response history missing",
            )?;

            match scenario {
                Scenario::ModelCallBudgets => {
                    let zero = MockCompletionModel::text("unused");
                    let result = AgentBuilder::new(zero.clone())
                        .build()
                        .prompt("question")
                        .max_turns(0)
                        .await;
                    ensure(result.is_err(), "zero budget must reject")?;
                    ensure(zero.request_count() == 0, "zero budget dispatched a model")?;
                    let bounded = MockCompletionModel::new([MockTurn::tool_call(
                        "call",
                        "add",
                        serde_json::json!({"x":1,"y":2}),
                    )]);
                    let bounded_result = AgentBuilder::new(bounded.clone())
                        .tool(MockAddTool)
                        .build()
                        .prompt("bounded")
                        .max_turns(1)
                        .await;
                    ensure(
                        bounded_result.is_err() && bounded.request_count() == 1,
                        "tool continuation escaped the total turn budget",
                    )?;
                }
                Scenario::CanonicalTranscript | Scenario::ToolCallResultPairing => {
                    let tool_model = MockCompletionModel::new([
                        MockTurn::tool_call("call-1", "add", serde_json::json!({"x": 1, "y": 2})),
                        MockTurn::text("3"),
                    ]);
                    let tool_response = AgentBuilder::new(tool_model)
                        .tool(MockAddTool)
                        .build()
                        .prompt("add")
                        .max_turns(2)
                        .extended_details()
                        .await
                        .map_err(|_| AdapterError("classic tool loop failed"))?;
                    let messages = tool_response
                        .messages()
                        .ok_or(AdapterError("tool transcript missing"))?;
                    ensure(messages.len() == 4, "tool transcript has wrong shape")?;
                    ensure(
                        matches!(
                            &messages[2],
                            rig_core::message::Message::User { content }
                                if content.iter().filter(|item| matches!(item, rig_core::message::UserContent::ToolResult(_))).count() == 1
                        ),
                        "tool call/result did not pair exactly once",
                    )?;
                }
                Scenario::UsageAccounting => {
                    ensure(
                        model.request_count() == 1,
                        "completion counted more than once",
                    )?;
                }
                Scenario::InvalidToolRecovery => {
                    crate::test_utils::invalid_tool_recovery(
                        MockCompletionModel::new([MockTurn::tool_call(
                            "invalid",
                            "add",
                            serde_json::json!({"x":1,"y":2}),
                        )]),
                        |builder| builder,
                    )
                    .await
                    .map_err(|_| AdapterError("classic invalid-tool recovery failed"))?;
                    let stopped =
                        AgentBuilder::new(MockCompletionModel::new([MockTurn::tool_call(
                            "invalid",
                            "missing",
                            serde_json::json!({}),
                        )]))
                        .tool(MockAddTool)
                        .add_hook(StopInvalid)
                        .build()
                        .prompt("stop invalid")
                        .await;
                    ensure(stopped.is_err(), "invalid-tool stop disposition completed")?;
                }
                Scenario::ResponseRetryRollback => {
                    let retry_model = MockCompletionModel::new([
                        MockTurn::text("rejected"),
                        MockTurn::text("accepted"),
                    ]);
                    let retried = AgentBuilder::new(retry_model.clone())
                        .add_hook(RetryRejected)
                        .build()
                        .prompt("question")
                        .max_turns(2)
                        .extended_details()
                        .await
                        .map_err(|_| AdapterError("classic response retry failed"))?;
                    ensure(
                        retry_model.request_count() == 2
                            && retried.output == "accepted"
                            && retried.messages().is_some_and(|messages| {
                                !format!("{messages:?}").contains("rejected")
                            }),
                        "rejected response was committed or retry was not bounded",
                    )?;
                }
                Scenario::StopCancellation => {
                    let stopped_model = MockCompletionModel::text("stop me");
                    let stopped = AgentBuilder::new(stopped_model.clone())
                        .add_hook(StopAfterTurn)
                        .build()
                        .prompt("question")
                        .await;
                    ensure(
                        stopped.is_err() && stopped_model.request_count() == 1,
                        "stop action did not remain terminal",
                    )?;
                }
                Scenario::StructuredOutput => {
                    let schema = serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }))
                    .map_err(|_| AdapterError("structured schema was invalid"))?;
                    let structured_model = MockCompletionModel::new([MockTurn::tool_call(
                        "output",
                        "final_result",
                        serde_json::json!({"answer":"done"}),
                    )]);
                    let structured: String = AgentBuilder::new(structured_model.clone())
                        .output_schema_raw(schema)
                        .output_mode(OutputMode::Tool)
                        .build()
                        .prompt("structured")
                        .await
                        .map_err(|_| AdapterError("classic structured output failed"))?;
                    ensure(
                        structured.contains("done")
                            && structured_model.requests()[0]
                                .tools
                                .iter()
                                .any(|tool| tool.name == "final_result"),
                        "structured-output mode did not advertise/finalize its tool",
                    )?;
                }
                Scenario::BlockingStreamingParity
                | Scenario::ProviderFinalExposure
                | Scenario::ProvisionalStreaming => {
                    let stream_model = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("accepted"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                    ]]);
                    let mut stream = AgentBuilder::new(stream_model)
                        .build()
                        .runner("question")
                        .stream()
                        .await;
                    let mut saw_provider_final = false;
                    let mut saw_run_final = false;
                    while let Some(item) = stream.next().await {
                        match item.map_err(|_| AdapterError("classic stream failed"))? {
                            MultiTurnStreamItem::StreamAssistantItem(
                                rig_core::streaming::StreamedAssistantContent::Final(_),
                            ) => saw_provider_final = true,
                            MultiTurnStreamItem::FinalResponse(response) => {
                                saw_run_final = response.output == "accepted";
                            }
                            _ => {}
                        }
                    }
                    ensure(saw_provider_final, "typed provider final missing")?;
                    ensure(saw_run_final, "canonical streaming final missing")?;
                    let failing = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("rollback"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                        MockStreamEvent::error("late failure"),
                    ]]);
                    let mut failed_stream = AgentBuilder::new(failing)
                        .build()
                        .runner("question")
                        .stream()
                        .await;
                    let mut failed_run_final = false;
                    let mut failed_with_error = false;
                    while let Some(item) = failed_stream.next().await {
                        match item {
                            Ok(MultiTurnStreamItem::FinalResponse(_)) => failed_run_final = true,
                            Err(_) => failed_with_error = true,
                            _ => {}
                        }
                    }
                    ensure(
                        failed_with_error && !failed_run_final,
                        "failed provisional stream published a canonical final",
                    )?;
                }
                Scenario::ToolSuppression => {
                    let suppressed_model = MockCompletionModel::new([
                        MockTurn::tool_call("call", "add", serde_json::json!({"x":1,"y":2})),
                        MockTurn::text("done"),
                    ]);
                    let suppressed = AgentBuilder::new(suppressed_model)
                        .tool(MockAddTool)
                        .add_hook(SuppressTools)
                        .build()
                        .prompt("skip")
                        .max_turns(2)
                        .extended_details()
                        .await
                        .map_err(|_| AdapterError("classic suppression failed"))?;
                    ensure(
                        suppressed.messages().is_some_and(|messages| {
                            messages.iter().any(|message| {
                                matches!(message, rig_core::message::Message::User { content }
                                    if content.iter().any(|item| matches!(item, rig_core::message::UserContent::ToolResult(_))))
                            })
                        }),
                        "suppressed call did not produce an inert paired result",
                    )?;
                }
                Scenario::Concurrency => {
                    let concurrent = MockCompletionModel::new([
                        MockTurn::from_contents([
                            rig_core::completion::AssistantContent::ToolCall(
                                rig_core::message::ToolCall::new(
                                    "one".into(),
                                    rig_core::message::ToolFunction::new(
                                        "add".into(),
                                        serde_json::json!({"x": 1, "y": 2}),
                                    ),
                                ),
                            ),
                            rig_core::completion::AssistantContent::ToolCall(
                                rig_core::message::ToolCall::new(
                                    "two".into(),
                                    rig_core::message::ToolFunction::new(
                                        "add".into(),
                                        serde_json::json!({"x": 3, "y": 4}),
                                    ),
                                ),
                            ),
                        ])
                        .map_err(|_| AdapterError("parallel fixture was empty"))?,
                        MockTurn::text("done"),
                    ]);
                    let response = AgentBuilder::new(concurrent)
                        .tool(MockAddTool)
                        .build()
                        .prompt("add twice")
                        .max_turns(2)
                        .tool_concurrency(2)
                        .extended_details()
                        .await
                        .map_err(|_| AdapterError("parallel classic run failed"))?;
                    ensure(response.output == "done", "parallel run did not finalize")?;
                    ensure(
                        response.messages().is_some_and(|messages| {
                            matches!(
                                &messages[2],
                                rig_core::message::Message::User { content }
                                    if content.iter().filter_map(|item| match item {
                                        rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
                                        _ => None,
                                    }).eq(["one", "two"])
                            )
                        }),
                        "parallel classic results did not commit deterministically",
                    )?;
                }
                Scenario::Memory => {
                    let memory = CountingMemory::default();
                    let memory_result = AgentBuilder::new(MockCompletionModel::text("done"))
                        .memory(memory.clone())
                        .build()
                        .prompt("question")
                        .conversation("shared-conformance")
                        .await;
                    ensure(memory_result.is_ok(), "classic memory run failed")?;
                    ensure(
                        memory.load_count() == 1 && memory.append_count() == 1,
                        "classic memory did not load/append once",
                    )?;
                    let failed = AgentBuilder::new(MockCompletionModel::text("unused"))
                        .memory(FailingMemory::default())
                        .build()
                        .prompt("question")
                        .conversation("shared-conformance")
                        .await;
                    ensure(failed.is_err(), "classic memory failure was not mapped")?;
                }
                Scenario::StaleResultHandling => {
                    let failing_stream = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("provisional"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                        MockStreamEvent::error("late failure"),
                    ]]);
                    let mut stream = AgentBuilder::new(failing_stream)
                        .build()
                        .runner("question")
                        .stream()
                        .await;
                    let mut saw_final = false;
                    let mut saw_error = false;
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                            Err(_) => saw_error = true,
                            _ => {}
                        }
                    }
                    ensure(
                        saw_error && !saw_final,
                        "late stream result committed after failure",
                    )?;
                }
                _ => return Err(AdapterError("unknown shared conformance scenario")),
            }
            Ok(ScenarioEvidence::new(
                scenario,
                scenario.invariants().iter().copied(),
            ))
        })
    }
}

#[tokio::test]
async fn classic_runtime_passes_shared_conformance_ledger() {
    let mut adapter = ClassicAdapter;
    let result = verify_runtime(&mut adapter).await;
    assert!(result.is_ok(), "{result:?}");
}
