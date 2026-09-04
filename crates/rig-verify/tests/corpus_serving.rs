//! Matrix C: serving policy, routing and bus ownership.
//!
//! The bus-level axes: how the bus serves a program's dispatches
//! (`ServingPolicy`), which model answers each turn (`model_route` and
//! `on_model_select`), and whose bus it is (the agent's own, or a host's
//! via `over_bus`). The claim of the serving cells is that the trace is
//! independent of the policy: the same program under serial or concurrent
//! serving, one or two runner slots, default or one-deep buffers, records
//! the same dispatches in the same order (dispatch order), so one cassette
//! serves every cell. The routing cells show the required row naming
//! every route and a never-selected route advertised on replay from the
//! row alone. The host-bus cells show a header with no policy and a log
//! the agent stamped for its host.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | `serial_per_handler` | false · true |
//! | `tool_concurrency` | 1 · 2 |
//! | buffers | default · `command_capacity: 1, stream_capacity: 1` |
//! | keys served | model + 2 tools · model + memory + tool |
//! | tool wire | anthropic (two calls in one turn) · gemini (two turns, id-less) · openai (two turns, dual ids) |
//! | events | dropped · kept |
//! | routing | one model · a route selected after the first turn · a route registered, never selected |
//! | bus ownership | own · a host's (unary · streamed with events) |
//!
//! Full cross-product: 2 × 2 × 2 × 2 × 3 × 2 × 3 × 3 = 864. Recorded:
//! the 12 cells below. Pruned: the serving axes are crossed only on the
//! program that can show them (anthropic's two calls in one turn, the one
//! provider that emits them); a two-turn program has nothing for
//! concurrency to overlap, so it gets one serial cell (gemini) and one
//! concurrency-two cell (openai) to show the trace does not change;
//! buffers at one are crossed with concurrency two only, the case that
//! parks; routing and ownership are crossed with the tool-call-turn
//! program only, since neither changes what is asked.
//!
//! # Cells
//!
//! | golden | producer | policy | cassette |
//! |---|---|---|---|
//! | `anthropic_serving_serial_concurrency_one` | `corpus_serving.rs` `serial_concurrency_one_…` | serial, 1 slot | the two-tool stream (`streaming_tools/…`) |
//! | `anthropic_serving_concurrent_concurrency_one` | `concurrent_concurrency_one_…` | concurrent, 1 slot | the same |
//! | `anthropic_serving_concurrent_concurrency_two` | `concurrent_concurrency_two_…` | concurrent, 2 slots | the same |
//! | `anthropic_serving_concurrent_concurrency_two_events` | `concurrent_concurrency_two_events_…` | concurrent, 2 slots, events kept | the same |
//! | `anthropic_serving_capacity_one` | `capacity_one_…` | concurrent, 2 slots, every buffer at 1 | the same |
//! | `anthropic_serving_serial_memory_tools` | `serial_memory_tools_…` | serial over memory + tool | `corpus_hooks/observe_everything` |
//! | `gemini_serving_two_turns_serial` | gemini `corpus_serving.rs` `two_turns_serial_…` | serial, two turns, id-less calls | `hook_stress/…` |
//! | `openai_serving_two_turns_concurrency_two` | openai `corpus_serving.rs` `two_turns_concurrency_two_…` | 2 slots, two turns, dual ids | `effect_corpus/tool_call_turns` |
//! | `anthropic_serving_model_route` | `model_route_…` | route `fast` (Haiku 4.5) selected after the first turn | recorded: `corpus_serving/model_route` |
//! | `anthropic_serving_model_route_unselected` | `model_route_unselected_…` | route registered, never selected | `effect_corpus/tool_call_turn` |
//! | `anthropic_serving_host_bus` | `host_bus_…` | a host's bus, `bus: None` | recorded: `corpus_serving/host_bus` |
//! | `anthropic_serving_host_bus_streamed` | `host_bus_streamed_…` | the same, events kept | recorded: `corpus_serving/host_bus_streamed` |
//!
//! # What the matrix found
//!
//! - The required row omitted model routes: a program with a route the
//!   hook never selected had the same row as one without, and a golden
//!   whose route was selected named it only in the signature. Fixed in
//!   `rig-agent` (`route_keys`); the replay registers a route from the
//!   log through `model_route_handler`, added for it.
//! - A required key of the completion, memory or retrieval family that the
//!   record never dispatched to could not be replayed at all
//!   (`describe_required` knew tools only), so the unselected-route golden
//!   was unreplayable. Fixed in `rig-effect-log`.
//! - An agent over a host's bus had no way to stamp the log its host
//!   recorded: `Agent::stamp` was private. It is public now.
//! - The bus-policy check is one-sided by design: a host-bus golden names
//!   no policy and is accepted by any program, an own-bus golden is
//!   accepted by a host-bus program. `the_policy_check_is_one_sided` pins
//!   that contract.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Hook, Program, ROUTE};
use rig_agent::AgentBuilder;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError};

#[derive(serde::Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

/// An `add` of the program's own, so a relabelled owner mints its key.
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds two integers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
    }

    async fn call(&self, _context: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        Ok(args.x + args.y)
    }
}

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
const CHAIN_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
     subtract tool. Report the final number.";

const TWO_TOOLS: Program = Program {
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    streamed: true,
    ..Program::DEFAULT
};
const TOOL_TURN: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

const SERIAL_CONCURRENCY_ONE: Program = Program {
    fixture: "anthropic_serving_serial_concurrency_one",
    tool_concurrency: Some(1),
    ..TWO_TOOLS
};
const CONCURRENT_CONCURRENCY_ONE: Program = Program {
    fixture: "anthropic_serving_concurrent_concurrency_one",
    tool_concurrency: Some(1),
    ..TWO_TOOLS
};
const CONCURRENT_CONCURRENCY_TWO: Program = Program {
    fixture: "anthropic_serving_concurrent_concurrency_two",
    tool_concurrency: Some(2),
    ..TWO_TOOLS
};
const CONCURRENT_CONCURRENCY_TWO_EVENTS: Program = Program {
    fixture: "anthropic_serving_concurrent_concurrency_two_events",
    tool_concurrency: Some(2),
    ..TWO_TOOLS
};
const CAPACITY_ONE: Program = Program {
    fixture: "anthropic_serving_capacity_one",
    tool_concurrency: Some(2),
    ..TWO_TOOLS
};
const SERIAL_MEMORY_TOOLS: Program = Program {
    fixture: "anthropic_serving_serial_memory_tools",
    conversation: Some("golden-conversation"),
    ..TOOL_TURN
};
const GEMINI_TWO_TURNS_SERIAL: Program = Program {
    fixture: "gemini_serving_two_turns_serial",
    owner: "stress-agent",
    preamble: Some(CHAIN_PREAMBLE),
    prompt: CHAIN_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(6),
    streamed: true,
    ..Program::DEFAULT
};
const OPENAI_TWO_TURNS_CONCURRENCY_TWO: Program = Program {
    fixture: "openai_serving_two_turns_concurrency_two",
    preamble: Some(CHAIN_PREAMBLE),
    prompt: CHAIN_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(6),
    tool_concurrency: Some(2),
    ..Program::DEFAULT
};
const MODEL_ROUTE: Program = Program {
    fixture: "anthropic_serving_model_route",
    route: Some(ROUTE),
    hooks: &[Hook::RouteAfterFirstTurn],
    ..TOOL_TURN
};
const MODEL_ROUTE_UNSELECTED: Program = Program {
    fixture: "anthropic_serving_model_route_unselected",
    route: Some(ROUTE),
    ..TOOL_TURN
};
const HOST_BUS: Program = Program {
    fixture: "anthropic_serving_host_bus",
    ..TOOL_TURN
};
const HOST_BUS_STREAMED: Program = Program {
    fixture: "anthropic_serving_host_bus_streamed",
    streamed: true,
    ..TOOL_TURN
};

both_interpreters! {
    serial_concurrency_one: SERIAL_CONCURRENCY_ONE,
    concurrent_concurrency_one: CONCURRENT_CONCURRENCY_ONE,
    concurrent_concurrency_two: CONCURRENT_CONCURRENCY_TWO,
    concurrent_concurrency_two_events: CONCURRENT_CONCURRENCY_TWO_EVENTS,
    capacity_one: CAPACITY_ONE,
    serial_memory_tools: SERIAL_MEMORY_TOOLS,
    gemini_two_turns_serial: GEMINI_TWO_TURNS_SERIAL,
    openai_two_turns_concurrency_two: OPENAI_TWO_TURNS_CONCURRENCY_TWO,
    model_route: MODEL_ROUTE,
    model_route_unselected: MODEL_ROUTE_UNSELECTED,
    host_bus: HOST_BUS,
    host_bus_streamed: HOST_BUS_STREAMED,
}

/// The serving cells' claim, read off the goldens: every policy recorded
/// the same dispatches in the same order.
#[test]
fn the_trace_is_independent_of_the_serving_policy() {
    let cells = [
        &SERIAL_CONCURRENCY_ONE,
        &CONCURRENT_CONCURRENCY_ONE,
        &CONCURRENT_CONCURRENCY_TWO,
        &CAPACITY_ONE,
    ];
    let reference = corpus::golden("anthropic_concurrent_tools_serial");
    let shape = |log: &rig_effect_log::EffectLog| {
        log.iter()
            .map(|record| {
                (
                    record.key.clone(),
                    serde_json::to_value(&record.kind).expect("data"),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut policies = std::collections::BTreeSet::new();
    for cell in cells {
        let log = corpus::golden(cell.fixture);
        assert_eq!(shape(&log), shape(&reference), "{}", cell.fixture);
        policies.insert(format!("{:?}", log.header.bus.expect("an own bus")));
    }
    assert_eq!(policies.len(), 3, "three distinct policies: {policies:?}");
}

/// A program named otherwise mints other keys for its own tools, so its
/// required row is another program's: the golden refuses it, naming the
/// key the program needs and the log never served. (A host key is used
/// as given and a handler registered from the log keeps the log's key,
/// so those two do not relabel — the owner names what the program mints.)
#[tokio::test]
async fn a_relabelled_owner_is_refused_with_both_rows() {
    let program = Program {
        owner: "other",
        ..TOOL_TURN
    };
    let replay = corpus::Replay::open(&HOST_BUS);
    // The golden's tool is served too, so the refusal is the row's, not
    // the signature's.
    let served = replay.tool_server();
    served.attach(&replay.registrar);
    let agent = AgentBuilder::over_bus(
        replay.dispatcher.clone(),
        replay.registrar.clone(),
        program.owner,
        replay.model_key.clone(),
    )
    .name(program.owner)
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool(Add)
    .build();
    let refusal = agent
        .check_replayable(&replay.log)
        .expect_err("another owner is another program")
        .to_string();
    assert!(
        refusal.contains("other/tool:add#0") && refusal.contains("never served"),
        "{refusal}"
    );
    drop(agent);
    replay.close().await;
}

/// The policy check compares two policies only when both sides name one:
/// a host-bus golden (no policy) is accepted by an own-bus program, and
/// an own-bus golden by a host-bus program. The policy is the host's.
#[tokio::test]
async fn the_policy_check_is_one_sided() {
    // An own-bus program over the host-bus golden.
    let host = corpus::golden(HOST_BUS.fixture);
    assert_eq!(host.header.bus, None);
    let own = corpus::golden(MODEL_ROUTE_UNSELECTED.fixture);
    assert!(own.header.bus.is_some());

    let replay = corpus::Replay::open(&HOST_BUS);
    let server = replay.tool_server();
    // Never dispatched: the check reads headers and descriptors only.
    let owned = AgentBuilder::named_model(
        "default",
        rig_core::test_utils::MockCompletionModel::text("unused"),
    )
    .name("golden")
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool_server_handle(server)
    .build();
    assert!(owned.bus_config().is_some(), "an own bus names its policy");
    owned
        .check_replayable(&replay.log)
        .expect("a host-bus golden names no policy to compare");
    drop(owned);
    replay.close().await;
}
