//! Durable execution, as one property: a run interrupted after tool call
//! *N* — its `AgentRun` state serialized together with its effect log so
//! far — resumes in a fresh process image (a fresh bus, a replayer registered
//! from the log, the state deserialized) to the same output as a run that was
//! never interrupted, and its continuation of the log matches the reference
//! log's tail. This is what Burckhardt et al.'s durable functions and
//! RecPlay call re-execution from the record: every input the continuation
//! needs must be in the record.
//!
//! The interruption is a hand driver of `AgentRun` over the agent's own bus
//! keys (the sans-IO machine is the only place a run can be suspended
//! mid-flight today); the resumption goes through `Agent::runner(..).resume`
//! — the bus-driven engine — so the property crosses the two interpreters.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use rig_agent::{
    AgentBuilder,
    run::{AgentRun, AgentRunStep, ModelTurn, RunSpec, prepare_request},
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_bus::{Bus, ModelHandle, ToolHandle};
use rig_core::{
    completion::CompletionRequestBuilder,
    effect::EffectFamily,
    test_utils::{MockCompletionModel, MockTurn},
    transcript,
};
use rig_effect_log::{Checkpoint, EffectLog, EffectLogRecorder, EffectLogReplayer, RequestCheck};
use serde::Deserialize;
use serde_json::json;

/// A record as data: the request and the answer (the tool context is not
/// on the wire since format 5, so nothing is stripped).
fn as_data(record: &rig_core::effect::EffectRecord) -> (serde_json::Value, serde_json::Value) {
    let kind = serde_json::to_value(&record.kind).expect("a kind serializes");
    (
        kind,
        serde_json::to_value(&record.outcome).expect("an outcome serializes"),
    )
}

async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a run over the bus never hangs")
}

#[derive(Deserialize)]
struct TagArgs {
    tag: String,
}

/// Answers with its tag and records every call.
#[derive(Clone, Default)]
struct Tag {
    calls: Arc<Mutex<Vec<String>>>,
}

impl Tool for Tag {
    const NAME: &'static str = "tag";
    type Args = TagArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "answers with its tag".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"tag": {"type": "string"}}})
    }

    async fn call(&self, _context: &mut ToolContext, args: TagArgs) -> Result<String, Self::Error> {
        self.calls.lock().expect("lock").push(args.tag.clone());
        Ok(format!("tagged:{}", args.tag))
    }
}

fn tool_call_turn(id: &str, tag: &str) -> MockTurn {
    MockTurn::from_contents([rig_core::message::AssistantContent::ToolCall(
        rig_core::message::ToolCall::from_wire(
            id,
            rig_core::message::ToolFunction::new("tag".to_owned(), json!({"tag": tag})),
        ),
    )])
}

/// Three turns: tool, tool, then the answer.
fn script() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        tool_call_turn("tc-1", "first"),
        tool_call_turn("tc-2", "second"),
        MockTurn::text("done"),
    ])
}

/// Two turns: two tool calls at once, then the answer.
fn two_calls_script() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::from_contents([
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-1",
                rig_core::message::ToolFunction::new("tag".to_owned(), json!({"tag": "left"})),
            )),
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-2",
                rig_core::message::ToolFunction::new("tag".to_owned(), json!({"tag": "right"})),
            )),
        ]),
        MockTurn::text("both"),
    ])
}

const OWNER: &str = "durable";

/// How a scenario's agent is built and run.
#[derive(Clone, Copy)]
struct Scenario {
    two_calls: bool,
    tool_concurrency: usize,
    serial_per_handler: bool,
}

impl Scenario {
    fn model(self) -> MockCompletionModel {
        if self.two_calls {
            two_calls_script()
        } else {
            script()
        }
    }

    fn builder(self) -> rig_agent::agent::AgentBuilder<rig_agent::agent::WithBuilderTools> {
        AgentBuilder::with_bus_config(
            rig_core::serve::ServingPolicy {
                serial_per_handler: self.serial_per_handler,
                ..rig_core::serve::ServingPolicy::default()
            },
            "default",
            self.model(),
        )
        .owner(OWNER)
        .tool(Tag::default())
        .record_effects()
    }
}

/// The reference: the agent runs uninterrupted, recording every dispatch.
async fn reference_run(scenario: Scenario) -> (String, EffectLog) {
    let agent = scenario.builder().build();
    let response = within(
        agent
            .prompt("go")
            .max_turns(3)
            .tool_concurrency(scenario.tool_concurrency)
            .run(),
    )
    .await
    .expect("the reference run");
    let log = agent.take_effect_log().expect("recording was enabled");
    (response.output, log)
}

/// The interruption: the same program driven by hand over the same keys,
/// stopped after `tools_before_stop` tool calls, with its state and its log.
async fn drive_until_tool(scenario: Scenario, tools_before_stop: usize) -> (AgentRun, EffectLog) {
    let agent = scenario.builder().build();
    let model_key = agent.model_key().clone();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar: _,
        driver,
        agent,
    } = parts;
    let driver_task = tokio::spawn(driver);
    let model: ModelHandle = dispatcher.bind(&model_key).expect("the model");
    let tool_key = dispatcher
        .keys()
        .into_iter()
        .find(|key| key.as_str().contains("/tool:tag#"))
        .expect("the tool is published");
    let tool: ToolHandle = dispatcher.handle(&tool_key).expect("the tool");
    let catalog = agent.tool_server_handle().snapshot();

    let spec = RunSpec {
        max_turns: Some(3),
        ..RunSpec::new()
    };
    let mut run = AgentRun::from_spec(&spec, "go", None);
    let mut tools_done = 0;
    loop {
        match run.next_step().expect("a step") {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                let prepared = prepare_request(
                    &spec,
                    &model.capabilities(),
                    &history,
                    catalog.definitions().to_vec(),
                    run.output_tool_name(),
                    None,
                )
                .expect("prepared");
                run.advertise_tools(turn, prepared.tools.clone());
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared
                    .apply(CompletionRequestBuilder::unbound(prompt))
                    .build();
                let response = within(model.complete(request)).await.expect("the model");
                run.model_response(ModelTurn::new(
                    None,
                    response.choice,
                    response.usage,
                    executable,
                    allowed,
                ))
                .expect("a model turn");
            }
            AgentRunStep::CallTools { calls } => {
                if tools_done == tools_before_stop {
                    // Suspended with the calls pending: the state the driver
                    // persists between steps.
                    break;
                }
                let mut results = Vec::with_capacity(calls.len());
                for call in calls {
                    let name = call.tool_call.function.name.clone();
                    let answer = within(tool.call(
                        name.clone(),
                        call.tool_call.function.arguments.to_string(),
                        ToolContext::new(),
                    ))
                    .await
                    .expect("the tool");
                    tools_done += 1;
                    results.push(transcript::tool_result_output(
                        call.tool_call.id.clone(),
                        call.tool_call.provider.clone(),
                        name,
                        answer.result.output().clone(),
                    ));
                }
                run.tool_results(results).expect("tool results");
            }
            AgentRunStep::Done(_) => panic!("the run finished before the interruption"),
        }
    }
    let log = agent.take_effect_log().expect("recording was enabled");
    drop((model, tool, dispatcher, agent));
    within(driver_task).await.expect("driver task");
    (run, log)
}

/// The property, for one scenario and one interruption point.
async fn resumes_identically(scenario: Scenario, tools_before_stop: usize) {
    let (reference_output, reference_log) = reference_run(scenario).await;

    // Interrupt; persist the state and the log.
    let (suspended, partial_log) = drive_until_tool(scenario, tools_before_stop).await;
    let state = serde_json::to_string(&suspended).expect("the run state serializes");
    let partial = serde_json::to_string(&partial_log).expect("the log serializes");

    // A fresh process image: a fresh bus, the reference log's records
    // replayed from the point the interruption reached, the state restored.
    let restored: AgentRun = serde_json::from_str(&state).expect("the run state restores");
    let partial_log: EffectLog = serde_json::from_str(&partial).expect("the log restores");
    // Keys included: the agent is named (its owner), and its own tool
    // registry takes that owner, so two builds mint the same keys.
    assert_eq!(
        partial_log
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        reference_log[..partial_log.len()]
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        "the hand driver's record is the reference log's head, request and outcome"
    );
    let continuation: EffectLog = reference_log.tail(partial_log.len());
    let (dispatcher, registrar, mut driver) = Bus::channel_with(rig_core::serve::ServingPolicy {
        serial_per_handler: scenario.serial_per_handler,
        ..rig_core::serve::ServingPolicy::default()
    });
    EffectLogReplayer::register_all(&continuation, &mut driver).expect("fresh keys");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let replay_task = tokio::spawn(driver);
    let model_key = reference_log[0].key.clone();
    let tool_key = reference_log
        .iter()
        .find(|record| record.kind.family() == EffectFamily::Tool)
        .expect("a tool record")
        .key
        .clone();
    let tool_replayer =
        EffectLogReplayer::for_key(&continuation, &tool_key).expect("the tool's records");
    let resumed_agent =
        AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), OWNER, model_key)
            .tool_server_handle({
                let server = rig_agent::tool::server::ToolServer::new().run();
                server.add_registered_tool(
                    rig_agent::tool::RegisteredTool::from_handler(tool_replayer)
                        .expect("a tool-family replayer"),
                );
                server
            })
            .build();
    let response = within(
        resumed_agent
            .runner("ignored")
            .tool_concurrency(scenario.tool_concurrency)
            .resume(restored)
            .run(),
    )
    .await
    .expect("the resumed run");
    assert_eq!(
        response.output, reference_output,
        "the same answer from the record"
    );

    // The continuation's records are the reference log's tail.
    let resumed_log = recorder.take();
    assert_eq!(
        resumed_log
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        continuation
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        "the resumed run performed exactly the reference run's remaining effects, request and outcome"
    );
    drop((resumed_agent, dispatcher, registrar));
    within(replay_task).await.expect("replay driver");
}

#[tokio::test]
async fn a_run_resumes_from_its_state_and_its_log_to_the_same_answer() {
    let scenario = Scenario {
        two_calls: false,
        tool_concurrency: 1,
        serial_per_handler: false,
    };
    let (output, log) = reference_run(scenario).await;
    assert_eq!(output, "done");
    assert_eq!(
        log.iter()
            .map(|record| record.kind.family())
            .collect::<Vec<_>>(),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
        ]
    );
    resumes_identically(scenario, 1).await;
    // Interrupted before any tool ran: the first turn's calls are pending.
    resumes_identically(scenario, 0).await;
}

#[tokio::test]
async fn a_run_with_concurrent_tool_calls_resumes_identically() {
    resumes_identically(
        Scenario {
            two_calls: true,
            tool_concurrency: 2,
            serial_per_handler: false,
        },
        0,
    )
    .await;
}

#[tokio::test]
async fn a_run_under_serial_serving_resumes_identically() {
    resumes_identically(
        Scenario {
            two_calls: true,
            tool_concurrency: 2,
            serial_per_handler: true,
        },
        0,
    )
    .await;
}

/// A hook's decision is part of the program, not of the record: the record
/// holds the *patched* effect, so a resume under the same hook replays it,
/// and a resume without the hook is refused as a divergence rather than
/// silently answered with the wrong record.
struct PatchesTag;

impl rig_agent::agent::AgentHook for PatchesTag {
    async fn on_dispatch(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::DispatchEvent<'_>,
    ) -> rig_agent::agent::DispatchAction {
        match event.kind {
            rig_core::effect::EffectKind::ToolCall { name, .. } if name == "tag" => {
                rig_agent::agent::DispatchAction::patch(rig_core::effect::EffectKind::ToolCall {
                    name: name.clone(),
                    args: json!({"tag": "patched"}).to_string(),
                })
            }
            _ => rig_agent::agent::DispatchAction::proceed(),
        }
    }
}

#[tokio::test]
async fn a_hooks_decision_is_program_not_record() {
    let scenario = Scenario {
        two_calls: false,
        tool_concurrency: 1,
        serial_per_handler: false,
    };
    // Reference under the patching hook: the record holds the patched call.
    let agent = scenario.builder().add_hook(PatchesTag).build();
    let reference = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the reference run");
    let reference_log = agent.take_effect_log().expect("recording");
    let recorded_args = reference_log
        .iter()
        .find_map(|record| match &record.kind {
            rig_core::effect::EffectKind::ToolCall { args, .. } => Some(args.clone()),
            _ => None,
        })
        .expect("a tool record");
    assert!(
        recorded_args.contains("patched"),
        "the record holds the patched effect"
    );

    // Interrupt before any tool; resume with the hook: same answer.
    let (suspended, partial_log) = drive_until_tool(scenario, 0).await;
    let continuation: EffectLog = reference_log.tail(partial_log.len());
    let model_key = reference_log[0].key.clone();
    let tool_key = reference_log
        .iter()
        .find(|record| record.kind.family() == EffectFamily::Tool)
        .expect("a tool record")
        .key
        .clone();
    let resume = |with_hook: bool, continuation: EffectLog, suspended: AgentRun| {
        let model_key = model_key.clone();
        let tool_key = tool_key.clone();
        async move {
            let (dispatcher, registrar, mut driver) = Bus::channel();
            EffectLogReplayer::register_all(&continuation, &mut driver).expect("fresh keys");
            let replay_task = tokio::spawn(driver);
            let tool_replayer =
                EffectLogReplayer::for_key(&continuation, &tool_key).expect("records");
            let mut builder =
                AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), OWNER, model_key)
                    .tool_server_handle({
                        let server = rig_agent::tool::server::ToolServer::new().run();
                        server.add_registered_tool(
                            rig_agent::tool::RegisteredTool::from_handler(tool_replayer)
                                .expect("tool"),
                        );
                        server
                    });
            if with_hook {
                builder = builder.add_hook(PatchesTag);
            }
            let agent = builder.build();
            let result = within(agent.runner("ignored").resume(suspended).run()).await;
            drop((agent, dispatcher, registrar));
            within(replay_task).await.expect("replay driver");
            result
        }
    };
    let with_hook = resume(true, continuation.clone(), suspended.clone())
        .await
        .expect("resumed under the same hook");
    assert_eq!(with_hook.output, reference.output);

    // Without the hook the program is different: the dispatch is unpatched,
    // the record is patched, and the replayer says so.
    let without_hook = resume(false, continuation, suspended).await;
    let error = without_hook.expect_err("a divergence, not a silent answer");
    assert!(
        error.to_string().contains("replay divergence"),
        "the record refuses the changed program: {error}"
    );
}

/// A resumed run brought its history with it: it loads nothing from
/// memory and saves nothing to it — no `Memory` dispatch, no memory record
/// — so a resumed log is exactly the reference log's tail even when the
/// program has a memory backend, and a backend that is down cannot fail a
/// resume. (Before this held, the resume performed the load and threw the
/// result away: a `Memory{Load}` record led every resumed log.)
#[tokio::test]
async fn a_resumed_run_loads_nothing_from_memory() {
    use rig_core::memory::InMemoryConversationMemory;
    fn builder() -> rig_agent::agent::AgentBuilder<rig_agent::agent::WithBuilderTools> {
        AgentBuilder::new(script())
            .owner(OWNER)
            .tool(Tag::default())
            .memory(InMemoryConversationMemory::new())
            .conversation("durable-memory")
            .record_effects()
    }
    // The reference program with memory: a load first, a save last.
    let agent = builder().build();
    let response = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the reference run");
    assert_eq!(response.output, "done");
    let log = agent.take_effect_log().expect("recording");
    let families: Vec<EffectFamily> = log.iter().map(|record| record.kind.family()).collect();
    assert_eq!(
        families.first(),
        Some(&EffectFamily::Memory),
        "loaded first"
    );
    assert_eq!(families.last(), Some(&EffectFamily::Memory), "saved last");

    // Resumed from a fresh state through an agent that has the same
    // memory configured: no memory dispatch at either end.
    let agent = builder().build();
    let spec = RunSpec {
        max_turns: Some(3),
        ..RunSpec::new()
    };
    let state = AgentRun::from_spec(&spec, "go", None);
    let response = within(agent.runner("ignored").resume(state).run())
        .await
        .expect("the resumed run");
    assert_eq!(response.output, "done");
    let resumed = agent.take_effect_log().expect("recording");
    let families: Vec<EffectFamily> = resumed.iter().map(|record| record.kind.family()).collect();
    assert!(
        !families.contains(&EffectFamily::Memory),
        "a resumed run performed a memory dispatch: {families:?}"
    );
    assert_eq!(
        families,
        log.iter()
            .map(|record| record.kind.family())
            .filter(|family| *family != EffectFamily::Memory)
            .collect::<Vec<_>>(),
        "the resumed log is the reference log without its memory ends"
    );
}

/// The property through a checkpoint (L9): the interruption's state and
/// position become a `Checkpoint` beside the reference log's tail; a fresh
/// image loads both, `EffectLog::from_checkpoint` names the continuation,
/// the replayers serve it under `check`, and the resumed run performs
/// exactly the tail. The full log in the tail's place is refused by its
/// first id before any dispatch.
async fn resumes_from_a_checkpoint(
    scenario: Scenario,
    tools_before_stop: usize,
    check: RequestCheck,
) {
    let (reference_output, reference_log) = reference_run(scenario).await;
    let (suspended, partial_log) = drive_until_tool(scenario, tools_before_stop).await;
    let (checkpoint, tail) = reference_log.checkpoint(
        partial_log.len(),
        serde_json::to_value(&suspended).expect("the run state serializes"),
    );
    assert_eq!(checkpoint.at, partial_log.len());
    let persisted = (
        serde_json::to_string(&checkpoint).expect("a checkpoint serializes"),
        serde_json::to_string(&tail).expect("the tail serializes"),
    );

    // A fresh image: the checkpoint and the tail restored, the continuation
    // named, the state deserialized from the checkpoint.
    let checkpoint: Checkpoint<serde_json::Value> =
        serde_json::from_str(&persisted.0).expect("a checkpoint restores");
    let tail: EffectLog = serde_json::from_str(&persisted.1).expect("the tail restores");
    let refused = EffectLog::from_checkpoint(&checkpoint, reference_log.clone())
        .expect_err("the full log is not the tail");
    assert!(
        refused
            .message
            .starts_with("resume refused: the checkpoint at"),
        "{}",
        refused.message
    );
    let continuation = EffectLog::from_checkpoint(&checkpoint, tail).expect("the tail follows");
    let restored: AgentRun =
        serde_json::from_value(checkpoint.state.clone()).expect("the run state restores");
    let (dispatcher, registrar, mut driver) = Bus::channel_with(rig_core::serve::ServingPolicy {
        serial_per_handler: scenario.serial_per_handler,
        ..rig_core::serve::ServingPolicy::default()
    });
    EffectLogReplayer::register_all_checking(&continuation, &mut driver, check)
        .expect("fresh keys");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let replay_task = tokio::spawn(driver);
    let model_key = reference_log[0].key.clone();
    let tool_key = reference_log
        .iter()
        .find(|record| record.kind.family() == EffectFamily::Tool)
        .expect("a tool record")
        .key
        .clone();
    let tool_replayer = EffectLogReplayer::for_key(&continuation, &tool_key)
        .expect("the tool's records")
        .checking(check);
    let resumed_agent =
        AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), OWNER, model_key)
            .tool_server_handle({
                let server = rig_agent::tool::server::ToolServer::new().run();
                server.add_registered_tool(
                    rig_agent::tool::RegisteredTool::from_handler(tool_replayer)
                        .expect("a tool-family replayer"),
                );
                server
            })
            .build();
    let response = within(
        resumed_agent
            .runner("ignored")
            .tool_concurrency(scenario.tool_concurrency)
            .resume(restored)
            .run(),
    )
    .await
    .expect("the resumed run");
    assert_eq!(response.output, reference_output);
    let resumed_log = recorder.take();
    assert_eq!(
        resumed_log
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        continuation
            .iter()
            .map(|record| (record.key.clone(), as_data(record)))
            .collect::<Vec<_>>(),
        "the resumed run performed exactly the checkpoint's continuation"
    );
    // Head and tail together are the reference log.
    let mut whole: Vec<_> = partial_log.iter().map(as_data).collect();
    whole.extend(resumed_log.iter().map(as_data));
    assert_eq!(whole, reference_log.iter().map(as_data).collect::<Vec<_>>());
    drop((resumed_agent, dispatcher, registrar));
    within(replay_task).await.expect("replay driver");
}

#[tokio::test]
async fn a_run_resumes_from_a_checkpoint_and_its_tail() {
    let scenario = Scenario {
        two_calls: false,
        tool_concurrency: 1,
        serial_per_handler: false,
    };
    resumes_from_a_checkpoint(scenario, 1, RequestCheck::Payload).await;
}

#[tokio::test]
async fn a_run_resumes_from_a_checkpoint_under_hash_checked_replay() {
    let scenario = Scenario {
        two_calls: true,
        tool_concurrency: 2,
        serial_per_handler: false,
    };
    // The two calls are one batch: the interruption is before it.
    resumes_from_a_checkpoint(scenario, 0, RequestCheck::Hash).await;
}
