//! Prompt caching over long live runs of features that can move a cached
//! prefix, on Claude Opus 5.5, recorded and replayed: a mid-conversation
//! system message every ten turns, and active tools that change every ten
//! turns.
//!
//! Record with `RIG_PROVIDER_TEST_MODE=record` and `ANTHROPIC_API_KEY`; see
//! `tests/README.md`.

use rig::AgentBuilder;
use rig::agent::Agent;
use rig::providers::anthropic::completion::CLAUDE_OPUS_5_5;
use rig_test_support::cache_longrun::workloads::{self, OrderHistory, ShippingLog, ToolSchedule};
use rig_test_support::cache_longrun::{
    self, Figures, Limits, LookupOrder, Recording, SUPPORT_PREAMBLE,
};
use serde_json::Value;

use super::super::support::with_anthropic_long_run_cassette;
use super::long_run_workloads::{OPUS_5_5_OUTPUT, OPUS_5_5_RATES, cached, check, support_agent};

/// The system-role entries of a request's `messages` and the texts of its
/// top-level `system` blocks.
fn system_placement(body: &Value) -> (usize, Vec<String>) {
    let in_messages = body["messages"].as_array().map_or(0, |messages| {
        messages
            .iter()
            .filter(|message| message["role"] == "system")
            .count()
    });
    let top_level = body["system"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| block["text"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    (in_messages, top_level)
}

/// A system message every ten turns, alternately where Anthropic takes one
/// and where rig has to move it: every one stays in `messages`, and no call
/// after the first read reads nothing.
#[tokio::test]
async fn mid_system_every_10_60() {
    let (log, sent) = with_anthropic_long_run_cassette(
        "long_run_caching/mid_system_every_10_60",
        |models, clock| async move {
            let agent = support_agent(cached(&models, CLAUDE_OPUS_5_5));
            workloads::mid_system_chat(&agent, &clock, 60, 10, "MID SYSTEM", "M").await
        },
    )
    .await;
    assert_eq!(sent, 5);
    let (recording, _) = check(
        "long_run_caching/mid_system_every_10_60",
        CLAUDE_OPUS_5_5,
        OPUS_5_5_RATES,
        OPUS_5_5_OUTPUT,
        Some(Limits {
            min_saving: Some(0.60),
            min_call_share: Some(0.80),
            max_writes_share: Some(0.25),
        }),
        &log,
    );
    let last = recording.calls.last().expect("the run made calls");
    let (in_messages, top_level) = system_placement(&last.body);
    assert_eq!(in_messages, sent, "every instruction stays in `messages`");
    assert!(
        top_level.iter().all(|text| !text.contains("Support desk")),
        "no instruction is hoisted into `system`: {top_level:?}"
    );
}

/// The tools the request advertised, by name.
fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn schedule() -> ToolSchedule {
    ToolSchedule::new(
        10,
        vec![
            vec!["lookup_order"],
            vec!["lookup_order", "order_history"],
            vec!["lookup_order", "shipping_log"],
        ],
    )
}

fn dynamic_agent(
    model: rig::DynModel<rig::operation::Completion>,
    schedule: ToolSchedule,
) -> Agent {
    AgentBuilder::new(model)
        .preamble(SUPPORT_PREAMBLE)
        .tool(LookupOrder)
        .tool(OrderHistory)
        .tool(ShippingLog)
        .max_tokens(800)
        .default_max_turns(4)
        .add_hook(schedule)
        .build()
}

/// The calls whose advertised tools differ from the call before, and what
/// each and the call after it read.
fn tool_changes(recording: &Recording) -> Vec<(usize, u64, Option<u64>)> {
    let reads = |index: usize| {
        recording.calls[index]
            .usage
            .cached_input_tokens
            .unwrap_or(0)
    };
    (1..recording.calls.len())
        .filter(|&index| {
            tool_names(&recording.calls[index].body) != tool_names(&recording.calls[index - 1].body)
        })
        .map(|index| {
            (
                recording.calls[index].index,
                reads(index),
                (index + 1 < recording.calls.len()).then(|| reads(index + 1)),
            )
        })
        .collect()
}

/// Active tools that change every ten turns on Claude Opus 5.5, whose
/// thinking blocks are bound to the tools they were produced with: the first
/// request after the change (turn 11) replays earlier thinking blocks under a
/// different `tools` list and is refused with a 400, the documented
/// thinking-block binding. The ten turns before it cache as usual. Dropping
/// the blocks instead needs the binding-controls beta and a request field
/// rig has no typed setting for.
#[tokio::test]
async fn dynamic_tools_30() {
    let run = with_anthropic_long_run_cassette(
        "long_run_caching/dynamic_tools_30",
        |models, clock| async move {
            let schedule = schedule();
            let agent = dynamic_agent(cached(&models, CLAUDE_OPUS_5_5), schedule.clone());
            workloads::dynamic_tools_chat(&agent, &schedule, &clock, 30, "DYNAMIC TOOLS", "T").await
        },
    )
    .await;
    let (recording, figures) = check(
        "long_run_caching/dynamic_tools_30",
        CLAUDE_OPUS_5_5,
        OPUS_5_5_RATES,
        OPUS_5_5_OUTPUT,
        None,
        &run.log,
    );
    report_dynamic(&recording, &figures, &run);
    assert_eq!(run.completed, 10, "the first ten turns complete");
    let (turn, error) = run
        .refused
        .as_ref()
        .expect("the changed tool list is refused");
    assert_eq!(*turn, 11);
    assert!(
        error.contains("400")
            && error.contains("The `tools` list differs from the one this block was created with"),
        "the refusal is the documented thinking-block binding: {error}"
    );
    assert!(
        tool_changes(&recording).is_empty(),
        "every successful call advertised the first tool set"
    );
}

fn report_dynamic(recording: &Recording, figures: &Figures, run: &workloads::DynamicToolsRun) {
    let changes = tool_changes(recording);
    let refused = run
        .refused
        .as_ref()
        .map(|(turn, error)| format!("turn {turn}: {error}"));
    cache_longrun::print_limit(
        figures,
        &format!(
            "dynamic tools: {} turns completed; tool changes (interaction, reads, next reads) \
             {changes:?}; refused {refused:?}",
            run.completed
        ),
    );
}
