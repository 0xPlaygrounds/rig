//! Matrix R: checkpoints and hash-checked replay.
//!
//! A `Checkpoint` cuts a log where a run was suspended: the position, the
//! id the tail begins with, and what the driver persisted (the agent's
//! serialized run). The continuation is `EffectLog::from_checkpoint` over
//! the tail — a full log in the tail's place is refused by its first id —
//! and the resumed engine replays it under a `RequestCheck`: by payload
//! (the divergence names the first differing JSON pointer) or by
//! `stable_hash` (the divergence names the hash pair). Resuming from a
//! checkpoint is test-side (rig-agent is frozen; `resume` takes the run
//! the checkpoint's `state` holds).
//!
//! Every cell interprets an **existing** golden: the hand driver takes the
//! program to a checkpoint after `n` tool turns, round-trips the checkpoint
//! as JSON, names the continuation, and the bus engine resumes it to the
//! golden's ending; the records of head and tail are the golden's, so the
//! oracle is `assert_same_records` as everywhere.
//!
//! # Dimensions
//!
//! | axis | values |
//! |---|---|
//! | checkpoint position | after turn 1's tools · after turn 2's tools · a streamed head |
//! | program | plain tools · hooks · memory · a committed output tool · a route · retrieval · a Matrix Q parent chain · a Matrix P layer |
//! | check mode | `Payload` · `Hash` |
//! | replayed against | the tail · the full log (refused by id) |
//!
//! Full cross-product: 3 × 8 × 2 × 2 = 96. Recorded: 14 programs × 2 modes
//! × 2 targets = 56 rows, plus the rows below. Pruned: the position is the
//! program's (a two-turn program checkpoints after turn 2, a streamed one
//! has a streamed head); a mode crossed with the full-log target beyond
//! payload (the refusal is by id, before any request is compared).
//!
//! # Cells
//!
//! | program (golden) | position |
//! |---|---|
//! | `anthropic_hooks_patch_tool_args` | after turn 1's tools |
//! | `anthropic_hooks_patch_tool_args_streamed` | a streamed head |
//! | `anthropic_hooks_replace_tool_result` | after turn 1's tools |
//! | `anthropic_hooks_two_hooks` | after turn 1's tools |
//! | `anthropic_hooks_lookup_before_run` | after turn 1's tools |
//! | `anthropic_host_custom_at_outcome` | after turn 1's tools |
//! | `anthropic_serving_serial_memory_tools` | after turn 1's tools (memory) |
//! | `anthropic_memory_serial_two_tools` | a streamed head (memory, two tools in one turn) |
//! | `anthropic_output_tool_with_real_tool` | after turn 1's tools (a committed output tool) |
//! | `anthropic_serving_model_route` | after turn 1's tools (a route) |
//! | `gemini_retrieval_context_and_tools` | after turn 1's tools (retrieval) |
//! | `openai_tool_call_turns` | after turn 2's tools |
//! | `mock_causal_depth_two` | after turn 1's tools (a parent chain: the ids survive) |
//! | `anthropic_layers_replace_tool_result` | after turn 1's tools (a layer) |
//!
//! Plus: `hash_mode_accepts_every_golden` (every golden replayed record by
//! record under `Hash`, counted); `a_one_byte_change_is_refused_by_hash_or_by_pointer`;
//! `a_checkpoint_of_another_format_is_refused_by_name`;
//! `a_tail_that_does_not_begin_at_the_checkpoint_is_refused`;
//! `a_resumed_memory_program_appends_once_and_loads_nothing`.
//!
//! # What the matrix found
//!
//! - Nothing moved in the goldens: a checkpoint is a cut of the log, and
//!   the resumed engine's records were the tail's under both modes.
//! - The parent chain survives a checkpoint by construction: the tail's
//!   records keep their `parent` ids, which name records of the head.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{
    Hook, LayerAt, LayerKind, LayerSpec, NESTING, NestedChild, Nesting, Output, Program, ROUTE,
};
use rig_core::effect::{EffectFamily, EffectKind, HandlerKey};
use rig_core::error::ErrorKind;
use rig_effect_log::{Checkpoint, EffectLog, EffectLogReplayer, RequestCheck};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;
const RETRIEVED_TOOLS_PREAMBLE: &str =
    "You are a calculator. You must use the provided tools for every arithmetic operation.";
const SUBTRACT_PROMPT: &str =
    "Subtract 8 from 50 with the subtract tool, then reply with just the number.";
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
const CHAIN_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
     subtract tool. Report the final number.";
const MOCK_PROMPT: &str = "Look up the capital of France and reply with just the lookup result.";
const LOOKUP_PREAMBLE: &str = "You are a research assistant. Use the lookup tool to answer.";

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

const PATCH_TOOL_ARGS: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args",
    hooks: &[Hook::PatchAddArgs],
    ..TOOLS
};
const PATCH_TOOL_ARGS_STREAMED: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args_streamed",
    hooks: &[Hook::PatchAddArgs],
    streamed: true,
    ..TOOLS
};
const REPLACE_TOOL_RESULT: Program = Program {
    fixture: "anthropic_hooks_replace_tool_result",
    hooks: &[Hook::ReplaceAddResult],
    ..TOOLS
};
const TWO_HOOKS: Program = Program {
    fixture: "anthropic_hooks_two_hooks",
    hooks: &[Hook::PatchAddArgs, Hook::ReplaceAddResult],
    ..TOOLS
};
const LOOKUP_BEFORE_RUN: Program = Program {
    fixture: "anthropic_hooks_lookup_before_run",
    hooks: &[Hook::LookupBeforeRun],
    ..TOOLS
};
const CUSTOM_AT_OUTCOME: Program = Program {
    fixture: "anthropic_host_custom_at_outcome",
    hooks: &[Hook::NoteAtOutcome],
    ..TOOLS
};
const SERIAL_MEMORY_TOOLS: Program = Program {
    fixture: "anthropic_serving_serial_memory_tools",
    conversation: Some(corpus::CONVERSATION),
    ..TOOLS
};
const MEMORY_SERIAL_TWO_TOOLS: Program = Program {
    fixture: "anthropic_memory_serial_two_tools",
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    streamed: true,
    conversation: Some(corpus::CONVERSATION),
    ..TOOLS
};
const OUTPUT_TOOL_WITH_REAL_TOOL: Program = Program {
    fixture: "anthropic_output_tool_with_real_tool",
    prompt: SUM_EVENT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Tool),
    ..TOOLS
};
const MODEL_ROUTE: Program = Program {
    fixture: "anthropic_serving_model_route",
    route: Some(ROUTE),
    hooks: &[Hook::RouteAfterFirstTurn],
    ..TOOLS
};
const GEMINI_CONTEXT_AND_TOOLS: Program = Program {
    fixture: "gemini_retrieval_context_and_tools",
    preamble: Some(RETRIEVED_TOOLS_PREAMBLE),
    prompt: SUBTRACT_PROMPT,
    retrieved_tools: Some(1),
    retrievable: &["add", "subtract"],
    dynamic_context: Some(1),
    ..TOOLS
};
const OPENAI_TOOL_CALL_TURNS: Program = Program {
    fixture: "openai_tool_call_turns",
    preamble: Some(CHAIN_PREAMBLE),
    prompt: CHAIN_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(6),
    ..Program::DEFAULT
};
const DEPTH_TWO: Program = Program {
    fixture: "mock_causal_depth_two",
    preamble: Some(LOOKUP_PREAMBLE),
    prompt: MOCK_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    nesting: Some(Nesting {
        child: NestedChild::Relay,
        ..NESTING
    }),
    ..Program::DEFAULT
};
const LAYERED_REPLACE: Program = Program {
    fixture: "anthropic_layers_replace_tool_result",
    layers: &[LayerSpec {
        at: LayerAt::Tool,
        layer: LayerKind::ReplaceAddResult,
    }],
    ..TOOLS
};

/// The cells: each program checkpointed after `turns` tool turns, resumed
/// against its tail under both modes, and against the full log (refused).
macro_rules! checkpointed {
    ($($name:ident: $program:ident after $turns:literal,)*) => {
        mod payload {
            $(
                #[tokio::test]
                async fn $name() {
                    $crate::corpus::checkpoint_reproduces(
                        &super::$program,
                        $turns,
                        rig_effect_log::RequestCheck::Payload,
                        $crate::corpus::Against::Tail,
                    )
                    .await;
                }
            )*
        }
        mod hash {
            $(
                #[tokio::test]
                async fn $name() {
                    $crate::corpus::checkpoint_reproduces(
                        &super::$program,
                        $turns,
                        rig_effect_log::RequestCheck::Hash,
                        $crate::corpus::Against::Tail,
                    )
                    .await;
                }
            )*
        }
        mod full_log_refused {
            $(
                #[tokio::test]
                async fn $name() {
                    $crate::corpus::checkpoint_reproduces(
                        &super::$program,
                        $turns,
                        rig_effect_log::RequestCheck::Payload,
                        $crate::corpus::Against::FullLog,
                    )
                    .await;
                }
            )*
        }
    };
}

checkpointed! {
    patch_tool_args: PATCH_TOOL_ARGS after 1,
    patch_tool_args_streamed: PATCH_TOOL_ARGS_STREAMED after 1,
    replace_tool_result: REPLACE_TOOL_RESULT after 1,
    two_hooks: TWO_HOOKS after 1,
    lookup_before_run: LOOKUP_BEFORE_RUN after 1,
    custom_at_outcome: CUSTOM_AT_OUTCOME after 1,
    serial_memory_tools: SERIAL_MEMORY_TOOLS after 1,
    memory_serial_two_tools: MEMORY_SERIAL_TWO_TOOLS after 1,
    output_tool_with_real_tool: OUTPUT_TOOL_WITH_REAL_TOOL after 1,
    model_route: MODEL_ROUTE after 1,
    gemini_context_and_tools: GEMINI_CONTEXT_AND_TOOLS after 1,
    openai_tool_call_turns_after_two: OPENAI_TOOL_CALL_TURNS after 2,
    depth_two: DEPTH_TWO after 1,
    layered_replace: LAYERED_REPLACE after 1,
}

/// Every golden, record by record, under `Hash`: what payload mode
/// accepts, hash mode accepts. The count is the corpus's.
#[tokio::test]
async fn hash_mode_accepts_every_golden() {
    let mut fixtures: Vec<_> = std::fs::read_dir(corpus::fixtures_dir())
        .expect("the fixtures directory")
        .filter_map(|entry| {
            let path = entry.expect("an entry").path();
            let name = path.file_name()?.to_str()?.to_owned();
            name.strip_suffix(".effects.json").map(str::to_owned)
        })
        .collect();
    fixtures.sort();
    assert!(fixtures.len() >= 207, "the corpus: {}", fixtures.len());
    let mut replayed = 0usize;
    for fixture in &fixtures {
        let log = corpus::golden(fixture);
        let (dispatcher, _registrar, mut driver) = rig_bus::Bus::channel();
        EffectLogReplayer::register_all_checking(&log, &mut driver, RequestCheck::Hash)
            .unwrap_or_else(|report| panic!("{fixture}: {report}"));
        let driver = tokio::spawn(driver);
        for record in log.iter() {
            let outcome = if record.events.is_some() {
                // A stream recorded verbatim replays verbatim: consume it and
                // fold it back to the record's outcome.
                use futures::StreamExt;
                let mut fold = rig_core::serve::StreamTap::new();
                let mut stream = dispatcher.dispatch_stream(&record.key, record.kind.clone());
                let mut folded = None;
                while let Some(item) = corpus::within(stream.next()).await {
                    if let Some(outcome) = fold.observe(&item) {
                        folded = Some(outcome);
                    }
                }
                folded.unwrap_or_else(|| Err(rig_core::serve::stream_truncated()))
            } else {
                corpus::within(dispatcher.dispatch(&record.key, record.kind.clone())).await
            };
            assert_eq!(
                serde_json::to_value(&outcome).expect("data"),
                serde_json::to_value(&record.outcome).expect("data"),
                "{fixture}: record {} replayed differently under hash mode",
                record.id
            );
            replayed += 1;
        }
        drop(dispatcher);
        corpus::within(driver).await.expect("the driver ends");
    }
    println!(
        "hash mode replayed {replayed} records of {} goldens",
        fixtures.len()
    );
}

/// One byte of a preamble changed: hash mode names the pair, payload mode
/// the pointer.
#[tokio::test]
async fn a_one_byte_change_is_refused_by_hash_or_by_pointer() {
    let log = corpus::golden(PATCH_TOOL_ARGS.fixture);
    let model = HandlerKey::from("golden/model:default");
    let EffectKind::Completion { request, stream } = &log[0].kind else {
        panic!("a completion first");
    };
    let mut changed = request.clone();
    // The preamble is the system message the request begins with.
    match changed.chat_history.first_mut() {
        Some(rig_core::message::Message::System { content }) => content.push('!'),
        other => panic!("the request begins with its preamble, not {other:?}"),
    }
    let changed = EffectKind::Completion {
        request: changed,
        stream: *stream,
    };
    for check in [RequestCheck::Payload, RequestCheck::Hash] {
        let (dispatcher, _registrar, mut driver) = rig_bus::Bus::channel();
        EffectLogReplayer::register_all_checking(&log, &mut driver, check).expect("fresh keys");
        let driver = tokio::spawn(driver);
        let report = corpus::within(dispatcher.dispatch(&model, changed.clone()))
            .await
            .expect_err("one byte differs");
        assert_eq!(report.kind, ErrorKind::Divergence);
        match check {
            // The pointer to the byte: the preamble is the first message.
            RequestCheck::Payload => assert!(
                report
                    .message
                    .contains("payload.request.chat_history[0].content differs"),
                "{}",
                report.message
            ),
            RequestCheck::Hash => {
                let recorded = rig_effect_log::stable_hash(&log[0].kind).expect("hashes");
                let arrived = rig_effect_log::stable_hash(&changed).expect("hashes");
                assert_ne!(recorded, arrived);
                assert!(
                    report.message.ends_with(&format!(
                        "hash {recorded:#018x} was recorded, {arrived:#018x} arrived"
                    )),
                    "{}",
                    report.message
                );
            }
        }
        drop(dispatcher);
        corpus::within(driver).await.expect("the driver ends");
    }
}

#[test]
fn a_checkpoint_of_another_format_is_refused_by_name() {
    let log = corpus::golden(PATCH_TOOL_ARGS.fixture);
    let (mut checkpoint, tail) = log.checkpoint(2, serde_json::json!({"run": "state"}));
    checkpoint.format = 5;
    let refused = EffectLog::from_checkpoint(&checkpoint, tail).expect_err("format 5");
    assert_eq!(
        refused.message,
        "resume refused: the checkpoint is format 5, this rig reads format 4"
    );
    let json = serde_json::to_string(&checkpoint).expect("serializes");
    let restored: Checkpoint = serde_json::from_str(&json).expect("restores: the number is data");
    assert_eq!(restored.format, 5);
}

#[test]
fn a_tail_that_does_not_begin_at_the_checkpoint_is_refused() {
    let log = corpus::golden(OPENAI_TOOL_CALL_TURNS.fixture);
    let (checkpoint, _) = log.checkpoint(2, serde_json::Value::Null);
    let elsewhere = log.tail(3);
    let refused = EffectLog::from_checkpoint(&checkpoint, elsewhere).expect_err("off by one");
    assert_eq!(
        refused.message,
        format!(
            "resume refused: the checkpoint at 2 expects record {} next, the tail begins at {}",
            log[2].id, log[3].id
        )
    );
}

/// A resumed memory program appends once and loads nothing: the head
/// loaded, the tail appends.
#[test]
fn a_resumed_memory_program_appends_once_and_loads_nothing() {
    let log = corpus::golden(SERIAL_MEMORY_TOOLS.fixture);
    let memory = HandlerKey::from("golden/memory");
    let ops: Vec<&str> = log
        .iter()
        .filter(|record| record.key == memory)
        .map(|record| match &record.kind {
            EffectKind::Memory { op } => match op {
                rig_core::effect::MemoryOp::Load { .. } => "load",
                rig_core::effect::MemoryOp::Append { .. } => "append",
                rig_core::effect::MemoryOp::Clear { .. } => "clear",
            },
            other => panic!("a memory op, not {other:?}"),
        })
        .collect();
    assert_eq!(ops, ["load", "append"]);
    // The cut after the first tool turn leaves the append to the tail, and
    // only the append: the resumed engine loads nothing.
    let first_tool = log
        .iter()
        .position(|record| record.kind.family() == EffectFamily::Tool)
        .expect("a tool record");
    let (_, tail) = log.checkpoint(first_tool + 1, serde_json::Value::Null);
    let tail_ops: Vec<bool> = tail
        .iter()
        .filter(|record| record.key == memory)
        .map(|record| {
            matches!(
                &record.kind,
                EffectKind::Memory {
                    op: rig_core::effect::MemoryOp::Append { .. }
                }
            )
        })
        .collect();
    assert_eq!(tail_ops, [true], "one append, no load, in the tail");
}
