//! Two interpreters of the agent program agree (Swierstra: any two
//! interpreters of one syntax tree must). The bus-driven engine
//! (`Agent::runner`) and a hand driver of `AgentRun::next_step` (the shape
//! of `tests/fixtures/agent_run_stepper`) run the same scripted model and
//! the same tools, and produce the same sequence of effects — families and
//! tool names, in order — and the same answer. Stated as a proptest property
//! over generated scripts, tool concurrency and serial serving; shrinking
//! yields the smallest disagreeing script, and there is none.
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
    tool::{Tool, ToolContext, ToolExecutionError, ToolSet},
};
use rig_bus::BusConfig;
use rig_core::{
    completion::{CompletionModel, CompletionRequestBuilder},
    effect::EffectKind,
    test_utils::{MockCompletionModel, MockTurn},
    transcript,
};
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

/// What one interpreter did: `completion` per model call, `tool:<name>` per
/// tool call, in the order the interpreter performed them.
type Trace = Vec<String>;

async fn bus_interpreter(case: &Case) -> (String, Trace) {
    let agent = AgentBuilder::with_bus_config(
        BusConfig {
            serial_per_handler: case.serial_per_handler,
            ..BusConfig::default()
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
    let trace = agent
        .take_effect_log()
        .expect("recording")
        .iter()
        .map(|record| match &record.kind {
            EffectKind::Completion { .. } => "completion".to_owned(),
            EffectKind::ToolCall { name, .. } => format!("tool:{name}"),
            other => format!("other:{}", other.name()),
        })
        .collect();
    (response.output, trace)
}

async fn hand_interpreter(case: &Case) -> (String, Trace) {
    let model = model_for(case);
    let mut tools = ToolSet::default();
    tools.add_tool(Alpha);
    tools.add_tool(Beta);
    let catalog = tools.catalog();
    let spec = RunSpec {
        max_turns: Some(case.turns.len()),
        ..RunSpec::new()
    };
    let mut run = AgentRun::from_spec(&spec, "go", None);
    let mut trace = Vec::new();
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
                trace.push("completion".to_owned());
                let response = model.completion(request).await.expect("the model");
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
                    trace.push(format!("tool:{name}"));
                    let result = catalog
                        .execute(
                            &name,
                            &call.tool_call.function.arguments.to_string(),
                            &mut ToolContext::new(),
                        )
                        .await;
                    results.push(transcript::tool_result_output(
                        call.tool_call.id.clone(),
                        call.tool_call.provider.clone(),
                        name,
                        result.output().clone(),
                    ));
                }
                run.tool_results(results).expect("tool results");
            }
            AgentRunStep::Done(response) => return (response.output, trace),
        }
    }
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
        prop_assert_eq!(&bus_trace, &hand_trace, "the same effects, in order");
    }
}
