//! Two interpreters of the agent program agree (Swierstra: any two
//! interpreters of one syntax tree must). The bus-driven engine
//! (`Agent::runner`) and a hand driver of `AgentRun::next_step` (the shape
//! of `tests/fixtures/agent_run_stepper`) run the same scripted model and
//! the same tools **over the same bus keys**, and produce the same sequence
//! of effects — every request and every outcome as data, in order, not
//! only families and names — and the same answer. Stated as a proptest
//! property over generated scripts, tool concurrency and serial serving;
//! shrinking yields the smallest disagreeing script, and there is none.
//!
//! Strategy size: one to four turns, each either text or one to three tool
//! calls from a fixed set of two tools, always ending in text; 256 cases by
//! default, under a second.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::time::Duration;

use proptest::prelude::*;
use rig_agent::{
    AgentBuilder,
    run::{AgentRun, AgentRunStep, ModelTurn, RunSpec, prepare_request},
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_bus::{ModelHandle, ToolHandle};
use rig_core::serve::ServingPolicy;
use rig_core::{
    completion::CompletionRequestBuilder,
    effect::{EffectKind, EffectRecord},
    test_utils::{MockCompletionModel, MockTurn},
    transcript,
};
use rig_effect_log::EffectLogRecorder;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Args {
    n: u64,
}

macro_rules! tool {
    ($name:ident, $wire:literal) => {
        #[derive(Clone, Default)]
        struct $name;

        impl Tool for $name {
            const NAME: &'static str = $wire;
            type Args = Args;
            type Output = String;
            type Error = ToolExecutionError;

            fn description(&self) -> String {
                concat!($wire, " answers deterministically").into()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {"n": {"type": "integer"}}})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                args: Args,
            ) -> Result<String, Self::Error> {
                Ok(format!("{}:{}", $wire, args.n))
            }
        }
    };
}

tool!(Alpha, "alpha");
tool!(Beta, "beta");

/// One model turn of a generated script.
#[derive(Debug, Clone)]
enum Turn {
    Text(String),
    /// Tool calls by (tool index, argument).
    Calls(Vec<(u8, u64)>),
}

#[derive(Debug, Clone)]
struct Case {
    turns: Vec<Turn>,
    tool_concurrency: usize,
    serial_per_handler: bool,
}

fn case() -> impl Strategy<Value = Case> {
    let call = (0u8..2, 0u64..100);
    let turn = prop_oneof![
        "[a-z]{1,8}".prop_map(Turn::Text),
        prop::collection::vec(call, 1..=3).prop_map(Turn::Calls),
    ];
    (
        prop::collection::vec(turn, 0..=3),
        "[a-z]{1,8}",
        1usize..=2,
        any::<bool>(),
    )
        .prop_map(|(mut turns, last, tool_concurrency, serial_per_handler)| {
            // A text turn ends the run, so only the final turn is text.
            turns.retain(|turn| matches!(turn, Turn::Calls(_)));
            turns.push(Turn::Text(last));
            Case {
                turns,
                tool_concurrency,
                serial_per_handler,
            }
        })
}

fn tool_name(index: u8) -> &'static str {
    if index == 0 { "alpha" } else { "beta" }
}

fn model_for(case: &Case) -> MockCompletionModel {
    let mut id = 0;
    MockCompletionModel::from_turns(case.turns.iter().map(|turn| match turn {
        Turn::Text(text) => MockTurn::text(text.clone()),
        Turn::Calls(calls) => MockTurn::from_contents(calls.iter().map(|(tool, n)| {
            id += 1;
            rig_core::message::AssistantContent::ToolCall(rig_core::message::ToolCall::from_wire(
                format!("tc-{id}"),
                rig_core::message::ToolFunction::new(tool_name(*tool).to_owned(), json!({"n": n})),
            ))
        })),
    }))
}

/// What one interpreter did: every effect it performed — the request as
/// data, with the tool context stripped (it is the one field the two
/// interpreters legitimately fill differently) — and what it got back, in
/// the order it performed them.
type Trace = Vec<(serde_json::Value, serde_json::Value)>;

fn trace_of<'a>(records: impl IntoIterator<Item = &'a EffectRecord>) -> Trace {
    records
        .into_iter()
        .map(|record| {
            let mut kind = serde_json::to_value(&record.kind).expect("a kind serializes");
            if let Some(object) = kind.as_object_mut() {
                object.remove("context");
            }
            let outcome = serde_json::to_value(&record.outcome).expect("an outcome serializes");
            (kind, outcome)
        })
        .collect()
}

async fn bus_interpreter(case: &Case) -> (String, Trace) {
    let agent = AgentBuilder::with_bus_config(
        ServingPolicy {
            serial_per_handler: case.serial_per_handler,
            ..ServingPolicy::default()
        },
        "default",
        model_for(case),
    )
    .tool(Alpha)
    .tool(Beta)
    .record_effects()
    .build();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        agent
            .prompt("go")
            .max_turns(case.turns.len())
            .tool_concurrency(case.tool_concurrency)
            .run(),
    )
    .await
    .expect("never hangs")
    .expect("the bus-driven run");
    let log = agent.take_effect_log().expect("recording");
    assert!(
        log.iter().all(|record| matches!(
            record.kind,
            EffectKind::Completion { .. } | EffectKind::ToolCall { .. }
        )),
        "only completions and tool calls in this program"
    );
    (response.output, trace_of(log.iter()))
}

async fn hand_interpreter(case: &Case) -> (String, Trace) {
    // The same program's bus — the same keys, the same handlers — driven
    // by hand: the model and the tools are reached through typed views,
    // never called directly, so the record is the bus's, as the engine's is.
    let agent = AgentBuilder::with_bus_config(
        ServingPolicy {
            serial_per_handler: case.serial_per_handler,
            ..ServingPolicy::default()
        },
        "default",
        model_for(case),
    )
    .tool(Alpha)
    .tool(Beta)
    .build();
    let model_key = agent.model_key().clone();
    let parts = agent
        .into_parts()
        .unwrap_or_else(|_| panic!("the only clone"));
    let rig_agent::agent::AgentParts {
        dispatcher,
        registrar: _,
        mut driver,
        agent,
    } = parts;
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver_task = tokio::spawn(driver);
    let model: ModelHandle = dispatcher.bind(&model_key).expect("the model");
    let catalog = agent.tool_server_handle().snapshot();
    let tool_handle = |name: &str| -> ToolHandle {
        let key = dispatcher
            .keys()
            .into_iter()
            .find(|key| key.as_str().contains(&format!("/tool:{name}#")))
            .expect("the tool is published");
        dispatcher.handle(&key).expect("a tool")
    };
    let spec = RunSpec {
        max_turns: Some(case.turns.len()),
        ..RunSpec::new()
    };
    let mut run = AgentRun::from_spec(&spec, "go", None);
    let output = loop {
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
                let response =
                    tokio::time::timeout(Duration::from_secs(5), model.complete(request))
                        .await
                        .expect("never hangs")
                        .expect("the model");
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
                let mut results = Vec::with_capacity(calls.len());
                for call in calls {
                    let name = call.tool_call.function.name.clone();
                    let answer = tokio::time::timeout(
                        Duration::from_secs(5),
                        tool_handle(&name).call(
                            name.clone(),
                            call.tool_call.function.arguments.to_string(),
                            ToolContext::new(),
                        ),
                    )
                    .await
                    .expect("never hangs")
                    .expect("the tool");
                    results.push(transcript::tool_result_output(
                        call.tool_call.id.clone(),
                        call.tool_call.provider.clone(),
                        name,
                        answer.result.output().clone(),
                    ));
                }
                run.tool_results(results).expect("tool results");
            }
            AgentRunStep::Done(response) => break response.output,
        }
    };
    drop((model, dispatcher, agent));
    tokio::time::timeout(Duration::from_secs(5), driver_task)
        .await
        .expect("never hangs")
        .expect("driver task");
    let log = recorder.take();
    (output, trace_of(log.iter()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn the_bus_engine_and_the_hand_driver_agree(case in case()) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        let (bus_output, bus_trace) = runtime.block_on(bus_interpreter(&case));
        let (hand_output, hand_trace) = runtime.block_on(hand_interpreter(&case));
        prop_assert_eq!(&bus_output, &hand_output, "the same answer");
        prop_assert_eq!(
            bus_trace.len(),
            hand_trace.len(),
            "the same number of effects"
        );
        for (index, (bus, hand)) in bus_trace.iter().zip(&hand_trace).enumerate() {
            prop_assert_eq!(&bus.0, &hand.0, "effect {} differs in its request", index);
            prop_assert_eq!(&bus.1, &hand.1, "effect {} differs in its outcome", index);
        }
    }
}
