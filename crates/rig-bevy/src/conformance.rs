//! ECS-runtime adapter for the shared behavioral conformance ledger.

use bevy_ecs::world::World;
use rig_core::{
    completion::Usage,
    test_utils::{CountingMemory, FailingMemory, MockCompletionModel, MockStreamEvent, MockTurn},
    tool::Tool,
};
use rig_runtime_conformance::{
    RuntimeScenarioAdapter, Scenario, ScenarioEvidence, ScenarioFuture, verify_runtime,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    components::{
        CallBudget, CommittedTranscript, ModelOperation, ProgressState, RunNode, RunPhase,
        TerminalReason, ToolCallNode, UsageLedger,
    },
    effects::{SubscriptionEvent, classify_model_ingress, commit_model_accounting},
    policy::{InvalidToolPolicy, OutputMode, ResponseRetryPolicy},
    runtime::{AgentSpec, BevyRunError, BevyRuntime},
    schedule::initialize_world,
    topology::{AgentId, EffectIdentity, Generation, OperationId, RunId, TenantId, WorldId},
};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct AdapterError(&'static str);

fn ensure(condition: bool, message: &'static str) -> Result<(), AdapterError> {
    condition.then_some(()).ok_or(AdapterError(message))
}

#[derive(Deserialize)]
struct AddArgs {
    value: i32,
}

#[derive(JsonSchema)]
#[expect(dead_code, reason = "the field shape is consumed by JsonSchema")]
struct StructuredAnswer {
    answer: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("add failed")]
struct AddError;

struct AddOne;

impl Tool for AddOne {
    const NAME: &'static str = "add_one";
    type Args = AddArgs;
    type Output = i32;
    type Error = AddError;

    fn description(&self) -> String {
        "Add one".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","required":["value"]})
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.value + 1)
    }
}

struct BevyAdapter;

impl RuntimeScenarioAdapter for BevyAdapter {
    type Error = AdapterError;

    fn runtime_name(&self) -> &'static str {
        "rig-bevy"
    }

    fn exercise(&mut self, scenario: Scenario) -> ScenarioFuture<'_, Self::Error> {
        Box::pin(async move {
            let model = MockCompletionModel::new([MockTurn::text("accepted").with_usage({
                let mut usage = Usage::new();
                usage.total_tokens = 1;
                usage
            })]);
            let runtime = BevyRuntime::default();
            let outcome = runtime
                .spawn_agent(AgentSpec::new(model.clone()).max_calls(1))
                .prompt("question")
                .await
                .map_err(|_| AdapterError("ECS baseline run failed"))?;
            ensure(
                outcome.transcript.len() == 2,
                "canonical transcript missing",
            )?;
            ensure(outcome.usage.total_tokens == 1, "usage was not committed")?;
            ensure(
                outcome.handle.terminal().ok().flatten() == Some(TerminalReason::Completed),
                "terminal completion was not retained",
            )?;

            match scenario {
                Scenario::ModelCallBudgets => {
                    let zero_model = MockCompletionModel::text("unused");
                    let result = runtime
                        .spawn_agent(AgentSpec::new(zero_model.clone()).max_calls(0))
                        .prompt("question")
                        .await;
                    ensure(
                        matches!(result, Err(BevyRunError::BudgetExhausted { .. })),
                        "zero budget must reject",
                    )?;
                    ensure(
                        zero_model.request_count() == 0,
                        "zero budget dispatched a model",
                    )?;
                    let bounded_model = MockCompletionModel::new([MockTurn::tool_call(
                        "call",
                        "add_one",
                        serde_json::json!({"value":1}),
                    )]);
                    let bounded_runtime = BevyRuntime::default();
                    let revision = bounded_runtime.register_tool(TenantId::default(), AddOne);
                    let bounded = bounded_runtime
                        .spawn_agent(
                            AgentSpec::new(bounded_model.clone())
                                .max_calls(1)
                                .grant_tool("add_one", revision),
                        )
                        .prompt("bounded")
                        .await;
                    ensure(
                        matches!(bounded, Err(BevyRunError::BudgetExhausted { .. }))
                            && bounded_model.request_count() == 1,
                        "continuation escaped the total call budget",
                    )?;
                }
                Scenario::CanonicalTranscript | Scenario::ToolCallResultPairing => {
                    let tool_model = MockCompletionModel::new([
                        MockTurn::tool_call("call-1", "add_one", serde_json::json!({"value": 2})),
                        MockTurn::text("3"),
                    ]);
                    let tool_runtime = BevyRuntime::default();
                    let revision = tool_runtime.register_tool(TenantId::default(), AddOne);
                    let tool_outcome = tool_runtime
                        .spawn_agent(
                            AgentSpec::new(tool_model)
                                .max_calls(2)
                                .grant_tool("add_one", revision),
                        )
                        .prompt("add")
                        .await
                        .map_err(|_| AdapterError("ECS tool loop failed"))?;
                    ensure(
                        tool_outcome.transcript.len() == 4,
                        "tool history has wrong shape",
                    )?;
                    ensure(
                        matches!(
                            &tool_outcome.transcript[2],
                            rig_core::message::Message::User { content }
                                if content.iter().filter(|item| matches!(item, rig_core::message::UserContent::ToolResult(_))).count() == 1
                        ),
                        "tool result did not pair exactly once",
                    )?;
                }
                Scenario::UsageAccounting => {
                    let identity = runtime.inspect(|world| {
                        world
                            .query::<&ModelOperation>()
                            .iter(world)
                            .find(|operation| operation.effect.run == outcome.handle.identity().run)
                            .map(|operation| operation.effect)
                    });
                    let identity = identity.ok_or(AdapterError("model operation missing"))?;
                    let duplicate = runtime
                        .inspect(|world| commit_model_accounting(world, identity, Usage::new()));
                    ensure(!duplicate, "duplicate accounting mutated usage")?;
                }
                Scenario::InvalidToolRecovery => {
                    let invalid =
                        || MockTurn::tool_call("bad", "not_advertised", serde_json::json!({}));
                    let fail = BevyRuntime::default()
                        .spawn_agent(AgentSpec::new(MockCompletionModel::new([invalid()])))
                        .prompt("fail")
                        .await;
                    ensure(
                        matches!(fail, Err(BevyRunError::UnknownTool(_))),
                        "fail disposition did not reject",
                    )?;
                    for policy in [
                        InvalidToolPolicy::Retry {
                            feedback: "retry".into(),
                        },
                        InvalidToolPolicy::Skip {
                            reason: "skip".into(),
                        },
                    ] {
                        let recovered = BevyRuntime::default()
                            .spawn_agent(
                                AgentSpec::new(MockCompletionModel::new([
                                    invalid(),
                                    MockTurn::text("recovered"),
                                ]))
                                .max_calls(2)
                                .invalid_tool_policy(policy),
                            )
                            .prompt("recover")
                            .await;
                        ensure(recovered.is_ok(), "retry or skip did not recover")?;
                    }
                    let repair_runtime = BevyRuntime::default();
                    let revision = repair_runtime.register_tool(TenantId::default(), AddOne);
                    let repaired = repair_runtime
                        .spawn_agent(
                            AgentSpec::new(MockCompletionModel::new([
                                invalid(),
                                MockTurn::text("repaired"),
                            ]))
                            .max_calls(2)
                            .grant_tool("add_one", revision)
                            .invalid_tool_policy(
                                InvalidToolPolicy::Repair {
                                    name: "add_one".into(),
                                    arguments: r#"{"value":1}"#.into(),
                                },
                            ),
                        )
                        .prompt("repair")
                        .await;
                    ensure(repaired.is_ok(), "repair did not execute the allowed tool")?;
                    let stopped = BevyRuntime::default()
                        .spawn_agent(
                            AgentSpec::new(MockCompletionModel::new([invalid()]))
                                .invalid_tool_policy(InvalidToolPolicy::Stop {
                                    reason: "stop".into(),
                                }),
                        )
                        .prompt("stop")
                        .await;
                    ensure(
                        matches!(stopped, Err(BevyRunError::Stopped(reason)) if reason == "stop"),
                        "stop disposition did not terminate",
                    )?;
                }
                Scenario::ResponseRetryRollback | Scenario::StructuredOutput => {
                    let structured_model = MockCompletionModel::new([
                        MockTurn::text(r#"{"answer":"wrong"}"#),
                        MockTurn::text(r#"{"answer":42}"#),
                    ]);
                    let structured = BevyRuntime::default()
                        .spawn_agent(
                            AgentSpec::new(structured_model.clone())
                                .max_calls(2)
                                .output_schema::<StructuredAnswer>()
                                .output_mode(OutputMode::Native)
                                .response_retry_policy(ResponseRetryPolicy {
                                    max_retries: 1,
                                    retries: 0,
                                    feedback: "valid JSON".into(),
                                    best_effort: false,
                                }),
                        )
                        .prompt("structured")
                        .await
                        .map_err(|_| AdapterError("structured retry failed"))?;
                    ensure(
                        structured_model.request_count() == 2,
                        "structured retry did not consume one fresh call",
                    )?;
                    ensure(
                        structured.transcript.len() == 2
                            && !format!("{:?}", structured.transcript).contains("wrong"),
                        "rejected structured content committed",
                    )?;
                }
                Scenario::Memory => {
                    let memory = CountingMemory::default();
                    let memory_runtime = BevyRuntime::default();
                    memory_runtime.bind_memory("memory", memory.clone());
                    memory_runtime
                        .spawn_agent(
                            AgentSpec::new(MockCompletionModel::text("done"))
                                .memory("memory", "conversation")
                                .max_calls(1),
                        )
                        .prompt("question")
                        .await
                        .map_err(|_| AdapterError("memory run failed"))?;
                    ensure(memory.load_count() == 1, "memory did not load once")?;
                    ensure(memory.append_count() == 1, "memory did not append once")?;
                    let failing = BevyRuntime::default();
                    failing.bind_memory("failing", FailingMemory::default());
                    let failed = failing
                        .spawn_agent(
                            AgentSpec::new(MockCompletionModel::text("unused"))
                                .memory("failing", "conversation"),
                        )
                        .prompt("question")
                        .await;
                    ensure(
                        matches!(failed, Err(BevyRunError::Memory(_))),
                        "memory failure was not mapped",
                    )?;
                }
                Scenario::BlockingStreamingParity => {
                    let blocking = BevyRuntime::default()
                        .spawn_agent(AgentSpec::new(MockCompletionModel::new([MockTurn::text(
                            "accepted",
                        )
                        .with_usage({
                            let mut usage = Usage::new();
                            usage.total_tokens = 1;
                            usage
                        })])))
                        .prompt("question")
                        .await
                        .map_err(|_| AdapterError("blocking parity run failed"))?;
                    let stream_model = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("accepted"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                    ]]);
                    let stream_runtime = BevyRuntime::default();
                    let mut events = Vec::new();
                    let streamed = stream_runtime
                        .spawn_agent(AgentSpec::new(stream_model).max_calls(1))
                        .stream_prompt("question", |event| events.push(event))
                        .await
                        .map_err(|_| AdapterError("ECS stream failed"))?;
                    ensure(
                        streamed.transcript == blocking.transcript
                            && streamed.usage == blocking.usage
                            && streamed.terminal == blocking.terminal,
                        "blocking and streaming canonical state diverged",
                    )?;
                }
                Scenario::ProviderFinalExposure | Scenario::ProvisionalStreaming => {
                    let stream_model = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("accepted"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                    ]]);
                    let stream_runtime = BevyRuntime::default();
                    let mut events = Vec::new();
                    stream_runtime
                        .spawn_agent(AgentSpec::new(stream_model).max_calls(1))
                        .stream_prompt("question", |event| events.push(event))
                        .await
                        .map_err(|_| AdapterError("ECS stream failed"))?;
                    ensure(
                        events
                            .iter()
                            .any(|event| matches!(event, SubscriptionEvent::ProviderFinal(_)))
                            && events.iter().any(|event| {
                                matches!(event, SubscriptionEvent::ProvisionalText(_))
                            }),
                        "successful stream events were incomplete",
                    )?;
                    let failing_model = MockCompletionModel::from_stream_turns([[
                        MockStreamEvent::text("rollback"),
                        MockStreamEvent::final_response_with_total_tokens(1),
                        MockStreamEvent::error("late failure"),
                    ]]);
                    let mut failed_events = Vec::new();
                    let failed = BevyRuntime::default()
                        .spawn_agent(AgentSpec::new(failing_model))
                        .stream_prompt("question", |event| failed_events.push(event))
                        .await;
                    ensure(failed.is_err(), "late stream failure completed")?;
                    ensure(
                        failed_events
                            .iter()
                            .any(|event| matches!(event, SubscriptionEvent::RolledBack(_)))
                            && !failed_events.iter().any(|event| {
                                matches!(
                                    event,
                                    SubscriptionEvent::ProviderFinal(_)
                                        | SubscriptionEvent::Accepted(_)
                                )
                            }),
                        "failed provisional stream published false success",
                    )?;
                }
                Scenario::ToolSuppression => {
                    let suppressed_runtime = BevyRuntime::default();
                    let suppressed = suppressed_runtime
                        .spawn_agent(
                            AgentSpec::new(MockCompletionModel::new([
                                MockTurn::tool_call("bad", "not_advertised", serde_json::json!({})),
                                MockTurn::text("done"),
                            ]))
                            .max_calls(2)
                            .invalid_tool_policy(
                                InvalidToolPolicy::Skip {
                                    reason: "skip".into(),
                                },
                            ),
                        )
                        .prompt("skip")
                        .await
                        .map_err(|_| AdapterError("suppressed run failed"))?;
                    ensure(
                        suppressed.transcript.len() == 4
                            && suppressed_runtime.inspect(|world| {
                                world
                                    .query::<&ToolCallNode>()
                                    .iter(world)
                                    .all(|call| call.suppressed && call.committed)
                            }),
                        "suppressed tool was dispatched or left unpaired",
                    )?;
                }
                Scenario::Concurrency => {
                    let parallel_model = MockCompletionModel::new([
                        MockTurn::from_contents([
                            rig_core::completion::AssistantContent::ToolCall(
                                rig_core::message::ToolCall::new(
                                    "first".into(),
                                    rig_core::message::ToolFunction::new(
                                        "add_one".into(),
                                        serde_json::json!({"value":1}),
                                    ),
                                ),
                            ),
                            rig_core::completion::AssistantContent::ToolCall(
                                rig_core::message::ToolCall::new(
                                    "second".into(),
                                    rig_core::message::ToolFunction::new(
                                        "add_one".into(),
                                        serde_json::json!({"value":2}),
                                    ),
                                ),
                            ),
                        ])
                        .map_err(|_| AdapterError("parallel fixture was empty"))?,
                        MockTurn::text("done"),
                    ]);
                    let parallel_runtime = BevyRuntime::default();
                    let revision = parallel_runtime.register_tool(TenantId::default(), AddOne);
                    let parallel = parallel_runtime
                        .spawn_agent(
                            AgentSpec::new(parallel_model)
                                .max_calls(2)
                                .max_tool_concurrency(1)
                                .grant_tool("add_one", revision),
                        )
                        .prompt("two")
                        .await
                        .map_err(|_| AdapterError("bounded tool run failed"))?;
                    ensure(
                        matches!(
                            &parallel.transcript[2],
                            rig_core::message::Message::User { content }
                                if content.iter().filter_map(|item| match item {
                                    rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
                                    _ => None,
                                }).eq(["first", "second"])
                        ),
                        "parallel results did not commit in call order",
                    )?;
                }
                Scenario::StaleResultHandling => stale_ingress_probe()?,
                Scenario::StopCancellation => {
                    let cancelled_model = MockCompletionModel::text("unused");
                    let cancelled = BevyRuntime::default()
                        .spawn_agent(AgentSpec::new(cancelled_model.clone()))
                        .begin_prompt("cancel")
                        .map_err(|_| AdapterError("cancel run did not start"))?;
                    cancelled
                        .handle()
                        .cancel("caller")
                        .map_err(|_| AdapterError("cancel fact was rejected"))?;
                    let result = cancelled.run().await;
                    ensure(
                        matches!(result, Err(BevyRunError::Cancelled { .. }))
                            && cancelled_model.request_count() == 0,
                        "cancellation allowed later dispatch",
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

fn stale_ingress_probe() -> Result<(), AdapterError> {
    let mut world = World::new();
    initialize_world(&mut world);
    let world_id = WorldId::allocate();
    let run = RunId::allocate();
    let operation = OperationId::allocate();
    let identity = EffectIdentity {
        world: world_id,
        tenant: TenantId(9),
        run,
        operation,
        generation: Generation(1),
        correlation: 3,
    };
    world.spawn((
        RunNode {
            id: run,
            agent: AgentId::allocate(),
            tenant: TenantId(9),
            generation: Generation(1),
            phase: RunPhase::Waiting,
        },
        CallBudget {
            limit: 1,
            dispatched: 1,
        },
        CommittedTranscript::default(),
        UsageLedger(Usage::new()),
        ProgressState {
            changes: 0,
            idle_passes: 0,
            max_idle_passes: 4,
        },
    ));
    world.spawn((ModelOperation {
        id: operation,
        effect: identity,
        committed: false,
        retired: false,
    },));

    let mut foreign = identity;
    foreign.world = WorldId(world_id.0.saturating_add(1));
    ensure(
        matches!(
            classify_model_ingress(&mut world, world_id, foreign),
            crate::effects::IngressDecision::ForeignWorld
        ),
        "foreign-world completion was accepted",
    )?;
    let mut wrong_tenant = identity;
    wrong_tenant.tenant = TenantId(10);
    ensure(
        classify_model_ingress(&mut world, world_id, wrong_tenant)
            == crate::effects::IngressDecision::WrongTenant,
        "wrong-tenant completion was accepted",
    )?;
    let mut wrong_generation = identity;
    wrong_generation.generation = Generation(2);
    ensure(
        classify_model_ingress(&mut world, world_id, wrong_generation)
            == crate::effects::IngressDecision::StaleGeneration,
        "wrong-generation completion was accepted",
    )?;
    let mut wrong_correlation = identity;
    wrong_correlation.correlation = 4;
    ensure(
        classify_model_ingress(&mut world, world_id, wrong_correlation)
            == crate::effects::IngressDecision::WrongCorrelation,
        "wrong-correlation completion was accepted",
    )?;
    ensure(
        commit_model_accounting(&mut world, identity, Usage::new()),
        "first commit rejected",
    )?;
    ensure(
        !commit_model_accounting(&mut world, identity, Usage::new()),
        "duplicate completion committed twice",
    )?;
    ensure(
        classify_model_ingress(&mut world, world_id, identity)
            == crate::effects::IngressDecision::Duplicate,
        "duplicate completion was not classified",
    )?;

    let retired_operation = OperationId::allocate();
    let retired = EffectIdentity {
        operation: retired_operation,
        correlation: 5,
        ..identity
    };
    world.spawn(ModelOperation {
        id: retired_operation,
        effect: retired,
        committed: false,
        retired: true,
    });
    ensure(
        classify_model_ingress(&mut world, world_id, retired)
            == crate::effects::IngressDecision::Retired,
        "superseded completion was not inert",
    )?;

    let late_operation = OperationId::allocate();
    let late = EffectIdentity {
        operation: late_operation,
        correlation: 6,
        ..identity
    };
    world.spawn(ModelOperation {
        id: late_operation,
        effect: late,
        committed: false,
        retired: false,
    });
    if let Some(mut run) = world
        .query::<&mut RunNode>()
        .iter_mut(&mut world)
        .find(|run| run.id == identity.run)
    {
        run.phase = RunPhase::Terminal;
    }
    ensure(
        classify_model_ingress(&mut world, world_id, late) == crate::effects::IngressDecision::Late,
        "cancelled or terminal completion was not inert",
    )
}

#[tokio::test]
async fn bevy_runtime_passes_shared_conformance_ledger() {
    let mut adapter = BevyAdapter;
    let result = verify_runtime(&mut adapter).await;
    assert!(result.is_ok(), "{result:?}");
}
