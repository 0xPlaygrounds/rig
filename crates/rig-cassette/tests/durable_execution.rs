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
//! mid-flight today); the resumption goes through `Agent::resume(run)`
//! — the bus-driven engine — so the property crosses the two interpreters.

use rig_cassette::agent::AgentReplayExt;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use rig_agent::handlers::{Handlers, ModelHandle, ToolHandle};
use rig_agent::{
    AgentBuilder,
    run::{AgentRun, AgentRunStep, ModelTurn, PendingToolCall, RunSpec, prepare_request},
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use rig_core::{
    completion::CompletionRequest,
    effect::EffectFamily,
    test_utils::{MockCompletionModel, MockTurn},
};
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
            rig_core::message::ToolFunction::new(
                rig_core::message::ToolName::new("tag".to_owned()).expect("tool name"),
                json!({"tag": tag}),
            ),
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
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("tag".to_owned()).expect("tool name"),
                    json!({"tag": "left"}),
                ),
            )),
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                "tc-2",
                rig_core::message::ToolFunction::new(
                    rig_core::message::ToolName::new("tag".to_owned()).expect("tool name"),
                    json!({"tag": "right"}),
                ),
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
}

impl Scenario {
    fn model(self) -> MockCompletionModel {
        if self.two_calls {
            two_calls_script()
        } else {
            script()
        }
    }

    fn builder(self) -> rig_agent::agent::AgentBuilder {
        AgentBuilder::named_model("default", self.model())
            .owner(OWNER)
            .tool(Tag::default())
    }
}

/// The reference: the agent runs uninterrupted, recording every dispatch.
async fn reference_run(scenario: Scenario) -> (String, EffectLog) {
    let recorder = EffectLogRecorder::new();
    let agent = scenario.builder().record_to(recorder.clone()).build();
    let response = within(
        agent
            .prompt("go")
            .max_turns(3)
            .tool_concurrency(scenario.tool_concurrency)
            .run(),
    )
    .await
    .expect("the reference run");
    let log = agent.stamp(recorder.take());
    (response.output(), log)
}

/// The interruption: the same program driven by hand over the same keys,
/// stopped after `tools_before_stop` tool calls, with its state and its log.
async fn drive_until_tool(scenario: Scenario, tools_before_stop: usize) -> (AgentRun, EffectLog) {
    let recorder = EffectLogRecorder::new();
    let agent = scenario.builder().record_to(recorder.clone()).build();
    let model_key = agent.model_key().clone();
    let handlers = agent.handlers().clone();
    let model: ModelHandle = handlers.bind(&model_key).expect("the model");
    let tool_key = handlers
        .keys()
        .into_iter()
        .find(|key| key.as_str().contains("/tool:tag#"))
        .expect("the tool is published");
    let tool: ToolHandle = handlers.handle(&tool_key).expect("the tool");
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
                let policy = prepared.policy.clone();
                let request = prepared.apply(CompletionRequest::new(prompt));
                let response = within(model.call(request)).await.expect("the model");
                run.model_response(ModelTurn::new(
                    response.head(),
                    response.choice,
                    response.usage,
                    policy,
                    response.raw,
                ))
                .expect("a model turn");
            }
            AgentRunStep::CallTools { calls } => {
                if tools_done == tools_before_stop {
                    // Suspended with the calls pending: the state the driver
                    // persists between steps.
                    break;
                }
                for call in calls {
                    let PendingToolCall::Execute(call) = call else {
                        panic!("an executable call: {call:?}");
                    };
                    let name = call.name().clone();
                    let answer = within(tool.call(
                        name.clone(),
                        call.arguments().to_string(),
                        ToolContext::new(),
                    ))
                    .await
                    .expect("the tool");
                    tools_done += 1;
                    run.answer(call.answer(answer.result))
                        .expect("tool results");
                }
            }
            AgentRunStep::Done(_) => panic!("the run finished before the interruption"),
        }
    }
    let log = agent.stamp(recorder.take());
    drop((model, tool, agent));
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
    let handlers = Handlers::default();
    rig_cassette::agent::replay::register_all(&continuation, &handlers).expect("fresh keys");
    let recorder = EffectLogRecorder::new();
    handlers.record_to(recorder.clone());
    let model_key = reference_log[0].key.clone();
    let tool_key = reference_log
        .iter()
        .find(|record| record.kind.family() == EffectFamily::Tool)
        .expect("a tool record")
        .key
        .clone();
    let tool_replayer =
        EffectLogReplayer::for_key(&continuation, &tool_key).expect("the tool's records");
    let resumed_agent = AgentBuilder::over_handlers(handlers.clone(), OWNER, model_key)
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
            .resume(restored)
            .tool_concurrency(scenario.tool_concurrency)
            .run(),
    )
    .await
    .expect("the resumed run");
    assert_eq!(
        response.output(),
        reference_output,
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
    drop(resumed_agent);
}

#[tokio::test]
async fn a_run_resumes_from_its_state_and_its_log_to_the_same_answer() {
    let scenario = Scenario {
        two_calls: false,
        tool_concurrency: 1,
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
    };
    // Reference under the patching hook: the record holds the patched call.
    let recorder = EffectLogRecorder::new();
    let agent = scenario
        .builder()
        .add_hook(PatchesTag)
        .record_to(recorder.clone())
        .build();
    let reference = within(agent.prompt("go").max_turns(3).run())
        .await
        .expect("the reference run");
    let reference_log = agent.stamp(recorder.take());
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
            let handlers = Handlers::default();
            rig_cassette::agent::replay::register_all(&continuation, &handlers)
                .expect("fresh keys");
            let tool_replayer =
                EffectLogReplayer::for_key(&continuation, &tool_key).expect("records");
            let mut builder = AgentBuilder::over_handlers(handlers.clone(), OWNER, model_key)
                .tool_server_handle({
                    let server = rig_agent::tool::server::ToolServer::new().run();
                    server.add_registered_tool(
                        rig_agent::tool::RegisteredTool::from_handler(tool_replayer).expect("tool"),
                    );
                    server
                });
            if with_hook {
                builder = builder.add_hook(PatchesTag);
            }
            let agent = builder.build();
            let result = within(agent.resume(suspended).run()).await;
            drop(agent);
            result
        }
    };
    let with_hook = resume(true, continuation.clone(), suspended.clone())
        .await
        .expect("resumed under the same hook");
    assert_eq!(with_hook.output(), reference.output());

    // Without the hook the program is different: the dispatch is unpatched,
    // the record is patched, and the replayer says so.
    let without_hook = resume(false, continuation, suspended).await;
    let error = without_hook.expect_err("a divergence, not a silent answer");
    assert!(
        error.to_string().contains("replay divergence"),
        "the record refuses the changed program: {error}"
    );
}

/// Answers a custom effect with its payload.
struct Echo;

impl rig_core::serve::Serve for Echo {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: "echo".into(),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: "test:echo".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        let payload = match kind {
            rig_core::effect::EffectKind::Custom { payload, .. } => payload,
            _ => serde_json::Value::Null,
        };
        rig_core::serve::Reply::Outcome(Ok(rig_core::effect::Answered::Custom { payload }))
    }
}

/// A host restarted from a checkpoint numbers its dispatches after the
/// checkpoint's last id, so a by-id replay of the tail answers each one.
#[tokio::test]
async fn a_table_restored_from_a_checkpoint_replays_the_tail_by_id() {
    use rig_cassette::effect_log::Checkpoint;
    use rig_core::effect::{Answered, EffectKind, HandlerKey};
    let key = HandlerKey::from("echo");
    let echo = |n: u64| EffectKind::Custom {
        kind: "test:echo".into(),
        payload: json!(n),
    };
    let recorder = EffectLogRecorder::new();
    let live = Handlers::default();
    live.register(key.clone(), Echo).expect("a fresh key");
    live.record_to(recorder.clone());
    for n in 1..=8 {
        live.dispatch(&key, echo(n)).await.expect("answered");
    }
    let (checkpoint, tail) = recorder.take().checkpoint(5, ());

    // A restart: the checkpoint and the tail come back from storage.
    let checkpoint: Checkpoint<()> =
        serde_json::from_str(&serde_json::to_string(&checkpoint).expect("serializes"))
            .expect("restores");
    let tail = EffectLog::from_checkpoint(&checkpoint, tail).expect("the tail follows");
    let resumed = Handlers::default();
    for replayer in EffectLogReplayer::for_log_by_id(&tail).expect("a valid log") {
        resumed
            .register(replayer.key().clone(), replayer)
            .expect("a fresh key");
    }
    resumed.resume_ids_after(checkpoint.last.expect("five records came before"));
    for n in 6..=8 {
        let answer = resumed.dispatch(&key, echo(n)).await.expect("replayed");
        assert!(
            matches!(&answer, Answered::Custom { payload } if *payload == json!(n)),
            "{answer:?}"
        );
    }
}
